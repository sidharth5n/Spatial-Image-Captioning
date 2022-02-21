import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.utils import pack_wrapper, clip_att, subsequent_mask
from .CaptionModel import CaptionModel

class Ensemble(CaptionModel):
    """
    Implements greedy decoding, sampling and beam search of an ensemble of
    models.
    """
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.vocab_size = models[0].vocab_size
        self.seq_length = models[0].seq_length

    def get_logprobs_state(self, it, enc_out, mask, state):
        """
        Finds the log probability distribution of the next word given the encoder
        output of all the models and all previous words.

        Parameters
        ----------
        it        : torch.tensor of shape (B,)
                    Current word
        enc_out   : list of length M containing torch.tensor of shape (B, P, E)
                    Encoder output of each model
        mask      : list of length M containing torch.tensor of shape (B, 1, P)
                    Mask for encoder output
        state     : list of length 1 or None
                    torch.tensor of shape (1, B, t) containing indices of words
                    generated upto time step t-1 including <START>.

        Returns
        -------
        log_probs : torch.tensor of shape (B, V)
                    Log softmax distribution over vocabulary for t+1
        state     : list of length 1 containing torch.tensor of shape (1, B, t+1)
                    Words generated upto time step t including <START>.
        """
        if state is None:
            ys = it.unsqueeze(1) # (B,1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1) # (B,t+1)
        # Get causal sequence mask, (1,t+1,t+1)
        seq_mask = subsequent_mask(ys.size(1)).to(enc_out[0])
        # Compute target sequence embedding and run the decoder, [(B,t+1,H)]*M
        dec_out = [m.model.decode(enc_out[i], mask[i], ys, seq_mask,
                                  local_mask(ys.size(1), m.local_win_size).to(enc_out[i]) if m.use_local_attn else None)
                   for i, m in enumerate(self.models)]
        # Compute output distribution for the last time step, [(B, V)]*M -> (B,V)
        logprobs = torch.stack([F.softmax(m.logit(dec_out[i][:, -1]), -1).unsqueeze(-1) for i, m in enumerate(self.models)], 2).mean(2).log()
        # New state, [(1,B,t+1)]
        state = [ys.unsqueeze(0)]
        return logprobs, state

    def _prepare_feature(self, img_feats, boxes = None, img_masks = None):
        """
        Computes the embeddings of visual features.

        Parameters
        ----------
        img_feats : torch.tensor of shape (B, L, D)
                    Output of last conv layer of CNN or bottom-up features
        boxes     : torch.tensor of shape (B, L, 4), optional
                    Coordinates of bounding boxes. Default is None.
        img_masks : torch.tensor of shape (B, L), optional
                    Image feature mask. Default is None.

        Returns
        -------
        img_feats : list of length M containing torch.tensor of shape (B, P, E)
                    img_feats after passing through attention embedding layer
        img_masks : list of length M containing torch.tensor of shape (B, 1, P)
                    Image feature mask.
        """
        # Clip to maximum feature length. Required when using multi-GPU. (B,L,D)->(B,P,D)
        img_feats, boxes, img_masks = clip_att(img_feats, boxes, img_masks)
        # Applies m.att_embed layer on img_feats, (B,P,D)->[(B,P,E)]*M
        img_feats = [pack_wrapper(m.att_embed, img_feats, img_masks) for m in self.models]
        if img_masks is None:
            img_masks = img_feats[0].new_ones(img_feats[0].shape[:2], dtype = torch.long)
        # Create M copies of image feature mask
        img_masks = [img_masks.unsqueeze(-2)] * len(self.models)

        return img_feats, boxes, img_masks

    def _sample_beam(self, img_feats, img_masks = None, boxes = None, opt = {}):
        """
        Generates captions by beam search.

        Parameters
        ----------
        img_feats   : torch.tensor of shape (B, L, D)
                      Output of last conv layer of CNN or bottom-up features
        img_masks   : torch.tensor of shape (B, L), optional
                      Image feature mask. Default is None.
        boxes       : torch.tensor of shape (B, L, 4), optional
                      Coordinates of bounding boxes. Default is None.
        opt         : dict, optional
                      Parameters for sampling.
                      beam_size        : Size of beam
                      group_size       : Group size for diverse beam search
                      diversity_lambda : Lambda for diverse beam search
                      max_ppl          : Beam search by max perplexity or probability

        Returns
        -------
        seq         : torch.tensor of shape (B, T)
                      Indexed sequence
        seqLogprobs : torch.tensor of shape (B, T)
                      Diversity augmented log probabilities of the words in the
                      sequence.
        """
        # Get beam size, default is 10
        beam_size = opt.get('beam_size', 10)
        batch_size = img_feats.size(0)
        # Compute the visual embedding, [(B,P,E)]*M, [(B,P)]*M
        img_feats, boxes, img_masks = self._prepare_feature(img_feats, boxes, img_masks)
        # Encode the visual features, [(B,P,E)]*M
        enc_out = [m.model.encode(img_feats[i], img_masks[i], boxes) for i,m in enumerate(self.models)]

        assert beam_size <= self.vocab_size + 1, 'Lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)

        # Lets process every image independently for now, for simplicity
        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = None
            # Get kth visual feature and replicate beam_size times, [(beam_size,P,E)]*M
            tmp_enc_out = [m[k:k+1].expand(*((beam_size,)+m.size()[1:])).contiguous() for m in enc_out]
            # Get kth attention mask and replicate beam_size times, [(beam_size,P)]*M
            tmp_img_masks = [img_mask[k:k+1].expand(*((beam_size,)+img_mask.size()[1:])).contiguous() for img_mask in img_masks]
            # Initial input = <START>
            it = img_feats[0].new_zeros([beam_size], dtype = torch.long)
            # Get the log probability of the next word, (beam_size,V), list-(1,beam_size,1)
            logprobs, state = self.get_logprobs_state(it, tmp_enc_out, tmp_img_masks, state)
            # Perform beam search
            self.done_beams[k] = self.beam_search(state, logprobs, tmp_enc_out, tmp_img_masks, opt = opt)
            # The first beam has highest cumulative score
            seq[:, k] = self.done_beams[k][0]['seq']
            # Get the sequence log probabilities of the first beam
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # Return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, img_feats, img_masks = None, opt = {}):
        """
        Generates sequences for the given input by greedy decoding (default),
        sampling or beam search.

        Parameters
        ----------
        img_feats   : torch.tensor of shape (B, L, D)
                      Output of last conv layer of CNN or bottom-up features
        img_masks   : torch.tensor of shape (B, L), optional
                      Image feature mask. Default is None.
        opt         : dict, optional
                      Parameters for sampling.
                      sample_max          : 1 - Greedy decoding, Otherwise - Sampling
                      beam_size           : Beam width for beam search
                      temperature         : Temperature parameter for sampling
                      decoding_constraint : 1 - Not to allow same word in a row

        Returns
        -------
        seq         : torch.tensor of shape (B, T)
                      Sampled sequence
        seqLogprobs : torch.tensor of shape (B, T)
                      Log probability of the samples in the generated sequence.
        """
        # Get greedy decoding status, default is greedy decoding
        sample_max = opt.get('sample_max', 1)
        # Get beam size for beam search, default is no beam search
        beam_size = opt.get('beam_size', 1)
        # Get temperature for sampling, default is no temperature sampling
        temperature = opt.get('temperature', 1.0)
        # Whether not to allow same words in a row, default is allow
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(img_feats, img_masks, boxes, opt)

        batch_size = img_feats.shape[0]
        # Compute the visual embedding, [(B,P,E)]*M, [(B,P)]*M
        img_feats, boxes, img_masks = self._prepare_feature(img_feats, boxes, img_masks)

        state = None
        # Encode the visual features, [(B,P,E)]*M
        enc_out = [m.model.encode(img_feats[i], img_masks[i], boxes) for i,m in enumerate(self.models)]
        # Tensor for storing sequence and log probabilities
        seq = img_feats[0].new_zeros((batch_size, self.seq_length), dtype = torch.long)
        seqLogprobs = img_feats[0].new_zeros((batch_size, self.seq_length))

        for t in range(self.seq_length + 1):
            if t == 0: # input <START>
                it = img_feats[0].new_zeros(batch_size, dtype = torch.long)
            # Get the log probability of the next word, (B, V), list-(1, B, t+1)
            logprobs, state = self.get_logprobs_state(it, enc_out, img_masks, state)
            # Whether not to allow same word in a row
            if decoding_constraint and t > 0:
                tmp = torch.zeros_like(logprobs)
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp
            # Skip if we achieve maximum length
            if t == self.seq_length:
                break
            # Perform greedy decoding
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            # Perform temperature scaling if required
            else:
                if temperature == 1.0: # Fetch prev distribution, (B,V)
                    prob_prev = torch.exp(logprobs.data)
                else: # Scale logprobs by temperature SHOULDN'T THE PROBABILITY BE RE-NORMALIZED?
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1) #(B,V)->(B,1)
                # Gather the logprobs at sampled positions (B,V)->(B,1)
                sampleLogprobs = logprobs.gather(1, it)
                it = it.view(-1).long() # Flatten indices for downstream processing (B,)

            # Find sequences which have not generated <END> so far
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,t] = it
            seqLogprobs[:,t] = sampleLogprobs.view(-1)
            # Quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs
