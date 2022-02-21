from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import copy

from misc.utils import pack_wrapper, clip_att, subsequent_mask
from .CaptionModel import CaptionModel
from .layers import *

class TransformerModel(CaptionModel):
    """
    Vanilla transformer captioning model with spatial positional encoding
    """
    def __init__(self, opt):
        super(TransformerModel, self).__init__()
        self.opt = opt
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.seq_length = opt.seq_length
        self.img_feat_size = opt.img_feat_size
        self.use_bn = getattr(opt, 'use_bn', 0)
        self.ss_prob = 0.0 # Schedule sampling probability


        self.att_embed = nn.Sequential(nn.Linear(self.img_feat_size, self.input_encoding_size),
                                       nn.ReLU(),
                                       nn.Dropout(opt.dropout))

        tgt_vocab = self.vocab_size + 1
        self.model = self.make_model(tgt_vocab, N = opt.num_layers,
                                     grids = opt.num_grids,
                                     d_model = opt.input_encoding_size,
                                     d_ff = opt.ff_size, heads = opt.heads,
                                     use_grid = opt.use_grid,
                                     enc_learnable_pos = getattr(opt, 'enc_pos_embedding', False),
                                     enc_learnable_pos_type = getattr(opt, 'enc_pos_type', 'gxg'),
                                     cross_attn = opt.cross_attn,
                                     norm = opt.norm,
                                     ff_activation = opt.ff_activation,
                                     dropout = opt.dropout)

    def make_model(self, tgt_vocab, N = 6, grids = 576, d_model = 512, d_ff = 2048,
                   heads = 8, use_grid = False, enc_learnable_pos = False,
                   enc_learnable_pos_type = 'gxg', cross_attn = 'dot-product',
                   norm = 'layer', ff_activation = 'RELU', dropout = 0.1):
        """
        Constructs the model from hyperparameters with Xavier initialization.

        Parameters
        ----------
        src_vocab : int
                    Size of source vocabulary
        tgt_vocab : int
                    Size of target vocabulary
        N         : int, optional
                    Number of layers. Default is 6.
        d_model   : int, optional
                    Input feature size. Default is 512.
        d_ff      : int, optional
                    Intermediate feature size of Position wise Feed Forward N/W.
                    Default is 2048.
        heads     : int, optional
                    Number of attention heads. Default is 8.
        dropout   : float, optional
                    Dropout probability. Default is 0.1.

        Returns
        -------
        model     : EncoderDecoder
                    Model constructed with the given hyperparameters and Xavier
                    initialized.
        """
        c = copy.deepcopy
        CrossAttention = {'xlinear' : XLinearMultiHeadedAttention, 'dot-product' : MultiHeadedAttention}
        self_attn = MultiHeadedAttention(heads, d_model, dropout)
        cross_attn = CrossAttention[cross_attn](heads if cross_attn == 'dot-product' else 1, d_model, dropout = dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout, ff_activation)
        enc_position = SpatialPositionalEncoding(grids, d_model, dropout, enc_learnable_pos, enc_learnable_pos_type) if use_grid else lambda x,y : x
        dec_position = PositionalEncoding(d_model, dropout)
        model = EncoderDecoder(Encoder(EncoderLayer(d_model, c(self_attn), c(ff), norm, dropout), norm, N),
                               Decoder(DecoderLayer(d_model, c(self_attn), c(cross_attn), c(ff), norm, dropout), norm, N),
                               enc_position,
                               nn.Sequential(Embeddings(d_model, tgt_vocab), c(dec_position)),
                               Generator(d_model, tgt_vocab))
        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def logit(self, x): # unsafe way
        return self.model.generator.proj(x)

    def _forward(self, att_feats, seq, att_masks = None, boxes = None):
        """
        Parameters
        ----------
        att_feats : torch.tensor of shape (B, L, D)
                    Output of last conv layer of CNN or bottom-up features
        seq       : torch.tensor of shape (B, T+2)
                    1-indexed captions including <START> and <END>
        att_masks : torch.tensor of shape (B, L) or None
                    Attention mask when no. of bottom-up proposals are
                    unequal across batch.

        Returns
        -------
        outputs   : torch.tensor of shape (B, T+1, V)
                    Distribution over vocabulary of the output sequence.
        """
        # Compute the visual embedding, remove end token and get causal sequence mask
        att_feats, seq, boxes, att_masks, seq_mask = self._prepare_feature(att_feats, boxes, att_masks, seq)
        # Pass all the features through transformer encoder-decoder, (B,T+1,H)
        out = self.model(att_feats, seq, boxes, att_masks, seq_mask)
        # Project output to vocabulary size and compute log_softmax, (B,T+1,H)->(B,T+1,V)
        outputs = self.model.generator(out)
        return outputs

    def _prepare_feature(self, att_feats, boxes, att_masks = None, seq = None):
        """
        Computes the embeddings of visual features, removes end token and
        prepares causal sequence mask.

        Parameters
        ----------
        att_feats : torch.tensor of shape (B, L, D)
                    Output of last conv layer of CNN or bottom-up features
        att_masks : torch.tensor of shape (B, L), optional
                    Attention mask when no. of bottom-up proposals are unequal
                    across batch. Default is None.
        seq       : torch.tensor of shape (B, T+2), optional
                    1-indexed captions including <START> and <END>. Default is
                    None.

        Returns
        -------
        att_feats : torch.tensor of shape (B, P, E)
                    att_feats after passing through attention embedding layers.
        seq       : torch.tensor of shape (B, T+1)
                    1-indexed captions including <START> but excluding <END>.
        att_masks : torch.tensor of shape (B, 1, P)
                    Attention mask.
        seq_mask  : torch.tensor of shape (B, T+1, T+1)
                    Causal sequence mask.
        """
        # Clip to maximum feature length. Required when using multi-GPU. (B,L,D)->(B,P,D)
        att_feats, boxes, att_masks = clip_att(att_feats, boxes, att_masks)
        # Applies self.att_embed layer on att_feats, (B,P,D)->(B,P,E)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        # If no attention mask, create a mask with all ones, (B,P)
        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        # (B,P)->(B,1,P)
        att_masks = att_masks.unsqueeze(-2)
        # If sequence is available as input
        if seq is not None:
            # Crop the last token. Not to be decoded. (B,T+2)->(B,T+1)
            seq = seq[:,:-1]
            # Create sequence mask
            seq_mask = (seq > 0)
            # Unmask <START> token
            seq_mask[:,0] += True
            # (B,T+1)->(B,1,T+1)
            seq_mask = seq_mask.unsqueeze(-2)
            # Create causal sequence mask, (B,1,T+1)->(B,T+1,T+1)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, boxes, att_masks, seq_mask

    def get_logprobs_state(self, it, enc_out, mask, state):
        """
        Finds the log probability distribution of the next word given the encoder
        output and all previous words.

        Parameters
        ----------
        it        : torch.tensor of shape (B,)
                    Current word
        enc_out   : torch.tensor of shape (B, P, E)
                    Encoder output
        mask      : torch.tensor of shape (B, P)
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
            ys = it.unsqueeze(1) #(B,1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim = 1) # (B,t+1)
        # Compute target sequence embedding and run the decoder, (B,t+1,H), (B,2,H)
        dec_out = self.model.decode(enc_out, mask, ys, subsequent_mask(ys.size(1)).to(enc_out.device))
        # Compute output distribution for the last time step, (B, V)
        logprobs = self.model.generator(dec_out[:, -1])

        return logprobs, [ys.unsqueeze(0)] # (B,V), (1,B,t+1)

    def _sample_beam(self, att_feats, boxes, att_masks = None, opt = {}):
        """
        Generates captions by beam search.

        Parameters
        ----------
        att_feats   : torch.tensor of shape (B, L, D)
                      Output of last conv layer of CNN or bottom-up features
        att_masks   : torch.tensor of shape (B, L) or None
                      Attention mask when no. of bottom-up proposals are unequal
                      across batch.
        opt         : dict
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
        batch_size = att_feats.size(0)
        # Compute the visual embedding, seq = None, seq_mask = None
        att_feats, seq, boxes, att_masks, seq_mask = self._prepare_feature(att_feats, boxes, att_masks)
        # Encode the visual features, (B,P,E)
        memory = self.model.encode(att_feats, boxes, att_masks)

        assert beam_size <= self.vocab_size + 1, 'Lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)

        # Lets process every image independently for now, for simplicity
        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = None
            # Get kth visual feature and replicate beam_size times
            tmp_memory = memory[k:k+1].expand(*((beam_size,)+memory.size()[1:])).contiguous()
            # Get kth attention mask and replicate beam_size times
            tmp_att_masks = att_masks[k:k+1].expand(*((beam_size,)+att_masks.size()[1:])).contiguous() if att_masks is not None else None
            # Initial input = <START>
            it = memory.new_zeros([beam_size], dtype=torch.long)
            # Get the log probability of the next word, (beam_size, V), list-(1, beam_size, 1)
            logprobs, state = self.get_logprobs_state(it, tmp_memory, tmp_att_masks, state)
            # Perform beam serach
            self.done_beams[k] = self.beam_search(state, logprobs, tmp_memory, tmp_att_masks, opt=opt)
            # The first beam has highest cumulative score
            seq[:, k] = self.done_beams[k][0]['seq']
            # Get the sequence log probabilities of the first beam
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # Return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, att_feats, att_masks = None, boxes = None, opt = {}):
        """
        Generates sequences for the given input by greedy decoding (default),
        sampling or beam search.

        Parameters
        ----------
        att_feats   : torch.tensor of shape (B, L, D)
                      Output of last conv layer of CNN or bottom-up features
        att_masks   : torch.tensor of shape (B, L), optional
                      Attention mask when no. of bottom-up proposals are unequal
                      across batch. Default is None.
        opt         : dict, optional
                      Parameters for sampling.
                      sample_max          : 1 - Greedy decoding, Otherwise - Sampling
                      beam_size           : Beam width for beam search
                      temperature         : Temperature parameter for sampling
                      decoding_constraint : 1 - Not to allow same words in a row
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
            return self._sample_beam(att_feats, boxes, att_masks, opt)

        batch_size = att_feats.shape[0]
        # Compute the visual embedding, (B,P,E), seq = None, (B,P), seq_mask = None
        att_feats, seq, boxes, att_masks, seq_mask = self._prepare_feature(att_feats, boxes, att_masks)

        state = None
        # Encode the visual features, (B,P,E)
        memory = self.model.encode(att_feats, boxes, att_masks)
        # Tensor for storing sequence and log probabilities
        seq = att_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = att_feats.new_zeros(batch_size, self.seq_length)

        for t in range(self.seq_length + 1):
            if t == 0: # input <START>
                it = memory.new_zeros(batch_size, dtype=torch.long)
            # Get the log probability of the next word, (B,V), [(1,B,t+1)]
            logprobs, state = self.get_logprobs_state(it, memory, att_masks, state)
            # Whether not to allow same word in a row
            if decoding_constraint and t > 0:
                tmp = seqLogprobs.new_zeros(seqLogprobs.shape[0], self.vocab_size + 1)
                tmp.scatter_(1, seq[:,t-1].unsqueeze(1), float('-inf'))
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
