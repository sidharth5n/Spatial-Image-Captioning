import torch
import torch.nn as nn
import torch.nn.functional as F

from .CaptionModel2 import CaptionModel

class ShowAndTell(CaptionModel):
    """
    Show and Tell model
    """
    def __init__(self, opt):
        super(ShowAndTell, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.hidden_size = opt.hidden_size
        self.num_layers = opt.num_layers
        self.dropout = opt.dropout
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.img_feat_size
        self.ss_prob = 0.0 # Schedule sampling probability

        self.img_embed = nn.Linear(self.fc_feat_size, self.input_encoding_size)
        self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size, self.hidden_size, self.num_layers, bias=False, dropout=self.dropout if self.num_layers > 1 else 0)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.hidden_size, self.vocab_size + 1)
        self.dropout = nn.Dropout(self.dropout)

        self.init_weights()

    def init_weights(self):
        alpha = 0.1
        self.embed.weight.data.uniform_(-alpha, alpha)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-alpha, alpha)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            return (weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
                    weight.new_zeros(self.num_layers, batch_size, self.hidden_size))
        else:
            return weight.new_zeros(self.num_layers, batch_size, self.hidden_size)

    def _forward(self, fc_feats, seq, att_masks = None, boxes = None):
        """
        Parameters
        ----------
        fc_feats  : torch.tensor of shape (B, D)
                    Output of last conv layer of CNN
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
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []

        for t in range(seq.size(1)):
            if t == 0:
                x = self.img_embed(fc_feats.float())
            else:
                if self.training and t >= 2 and self.ss_prob > 0.0: # otherwiste no need to sample
                    sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, t-1].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, t-1].data.clone()
                        prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                        it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                else:
                    it = seq[:, t-1].clone()
                # break if all the sequences end
                if t >= 2 and seq[:, t-1].data.sum() == 0:
                    break
                x = self.embed(it)

            output, state = self.rnn(x.unsqueeze(0), state)
            output = F.log_softmax(self.logit(self.dropout(output.squeeze(0))), dim = 1)
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs[1:]], 1).contiguous()

    def get_logprobs_state(self, it, state):
        """
        it : torch.tensor of shape (B, )
             Current word index
        state : list of length 1 containing torch.tensor of shape (1, N, B, H) or (2, N, B, H)
                (h, c) or h of RNN at time step t-1

        Returns
        -------
        logprobs : torch.tensor of shape (B, V)
                   Log softmax distribution over vocabulary for t+1
        state : list of length 1 containing torch.tensor of shape (1, N, B, H) or (2, N, B, H)
                (h, c) or h of RNN at time step t
        """
        # 'it' contains a word index
        xt = self.embed(it)
        state = state[0]
        if state.size(0) > 1:
            state = torch.chunk(state, 2)
            state = tuple([x.squeeze(0) for x in state])
        output, state = self.rnn(xt.unsqueeze(0), state)
        logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(0))), dim = 1)

        return logprobs, [torch.stack(tuple(state))]

    def _sample_beam(self, fc_feats, boxes = None, att_masks = None, opt = {}):
        # Get beam size, default is 10
        beam_size = opt.get('beam_size', 10)
        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'

        batch_size = fc_feats.size(0)

        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity
        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            for t in range(2):
                if t == 0:
                    xt = self.img_embed(fc_feats[k:k+1].float()).expand(beam_size, self.input_encoding_size)
                elif t == 1: # input <bos>
                    it = fc_feats.new_zeros(beam_size, dtype = torch.long)
                    xt = self.embed(it)

                output, state = self.rnn(xt.unsqueeze(0), state)
                logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(0))), dim = 1)

            self.done_beams[k] = self.beam_search(state, logprobs, opt = opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_masks=None, boxes = None, opt = {}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_masks, boxes, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        seq = fc_feats.new_zeros(batch_size, self.seq_length, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 2):
            if t == 0:
                xt = self.img_embed(fc_feats.float())
            else:
                if t == 1: # input <bos>
                    it = fc_feats.new_zeros(batch_size, dtype=torch.long)
                xt = self.embed(it)

            output, state = self.rnn(xt.unsqueeze(0), state)
            logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(0))), dim=1)

            # sample the next word
            if t == self.seq_length + 1: # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu() # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                it = it * unfinished.type_as(it)
                seq[:,t-1] = it #seq[t] the input of t+2 time step
                seqLogprobs[:,t-1] = sampleLogprobs.view(-1)
                if unfinished.sum() == 0:
                    break

        return seq, seqLogprobs
