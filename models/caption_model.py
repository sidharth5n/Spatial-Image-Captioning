from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from functools import reduce

class CaptionModel(nn.Module):
    """
    Base class for all other models. Implements beam search and diverse beam
    search.
    """
    def __init__(self):
        super(CaptionModel, self).__init__()

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_'+mode)(*args, **kwargs)

    def beam_search(self, init_state, init_logprobs, *args, **kwargs):
        """
        Implements beam search and augments log probabilities with diversity
        terms when number of groups > 1.

        Parameters
        ----------
        init_state       : list of length 1 containing torch.tensor of shape (1, B, 1)
                           Initial state i.e. <START>.
        init_logprobs    : torch.tensor of shape (B, V)
                           Log softmax distribution over vocabulary for t = 0

        Returns
        -------
        done_beams_table : list
                           All completed beams sorted by their joint probability.
                           Each element is a dict
                           seq     : torch.tensor of shape (T,)
                                     Indexed sequence
                           logps   : torch.tensor of shape (T,)
                                     Diversity augmented log probabilities of the
                                     words in the sequence
                           unaug_p : float
                                     Log probability the sequence
                           p       : float
                                     Diversity augmented log probability of the
                                     sequence.
        """

        def add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash):
            """
            Adds diversity term to the sequence log probabilities.

            Parameters
            ----------
            beam_seq_table   : list of length G
                               Each element is a torch.tensor of shape (T, B/G)
                               containing the beam sequences of each group.
            logprobsf        : torch.tensor of shape (B/G, V)
                               Log probabilities of the current group
            t                : int
                               Time step of decoding
            divm             : int
                               Beam group
            diversity_lambda : float
                               Weight given to diversity term
            bdash            : int
                               No. of beams per group
            """
            local_time = t - divm
            unaug_logprobsf = logprobsf.clone()
            for prev_choice in range(divm):
                prev_decisions = beam_seq_table[prev_choice][local_time]
                for sub_beam in range(bdash):
                    for prev_labels in range(bdash):
                        logprobsf[sub_beam][prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[prev_labels]] - diversity_lambda
            return unaug_logprobsf

        def beam_step(logprobsf, unaug_logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            """
            Performs one step of classical beam search.

            Parameters
            ----------
            logprobsf         : torch.tensor of shape (B/G, V)
                                Diversity augmented log probabilities
            unaug_logprobsf   : torch.tensor of shape (B/G, V)
                                Log probabilities
            beam_size         : int
                                Size of the beam (B')
            t                 : int
                                Time step of decoding
            beam_seq          : torch.tensor of shape (T, B')
                                Word indices of the beam sequences
            beam_seq_logprobs : torch.tensor of shape (T, B')
                                Log probabilities of the beam sequences
            beam_logprobs_sum : torch.tensor of shape (B',)
                                Joint log probability of each beam
            state             : list of length 1 containing torch.tensor of shape
                                (1, B/G, t+1)
                                Words generated upto time step t including <START>.

            Returns
            -------
            beam_seq          : torch.tensor of shape (T, B')
                                Word indices of the beam sequences
            beam_seq_logprobs : torch.tensor of shape (T, B')
                                Log probabilities of the beam sequences
            beam_logprobs_sum : torch.tensor of shape (B',)
                                Joint log probability of each beam
            state             : list of length 1 containing torch.tensor of shape
                                (1, B/G, t+1)
                                Words generated upto time step t including <START>.
            candidates        : list
                                All possible beam expansions. Each element is a
                                dict.
                                c : int
                                    New word
                                q : int
                                    Beam no.
                                p : float
                                    Diversity augmented joint log probability
                                r : float
                                    Joint log probability
            """
            # Sort the augmented probabilities in decreasing order
            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            ###################################################
            # WHY IS cols DEPENDENT ON beam_size????
            # NEED TO CHECK ALL WORDS IN vocab, RIGHT??
            # NOT REQUIRED, PROBABILITIES ALREADY SORTED -> MAX TO BE CHECKED beam_size * beam_size ONLY.
            ###################################################
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols): # for each column (word, essentially)
                for q in range(rows): # for each beam expansion
                    # Compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q, c].item()
                    # Compute joint logprob
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    # Compute unaugmented logprob
                    local_unaug_logprob = unaug_logprobsf[q, ix[q, c]]
                    candidates.append({'c': ix[q, c], 'q': q, 'p': candidate_logprob, 'r': local_unaug_logprob})
            # Sort the beams in descending order of joint logprob
            candidates = sorted(candidates,  key = lambda x: -x['p'])
            # [(1,B/G,t+1)]
            new_state = [_.clone() for _ in state]
            # beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
            # We'll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                # Fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                # Rearrange recurrent states
                for state_ix in range(len(new_state)):
                    # Copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']] # dimension one is time step
                # Append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c'] # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r'] # the raw logprob here
                beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates

        # Start diverse_beam_search
        opt = kwargs['opt']
        # Get beam size, default is 10
        beam_size = opt.get('beam_size', 10)
        # Get group size for diverse beam search, default is normal beam search
        group_size = opt.get('group_size', 1)
        # Get lambda for diverse beam search, default is 0.5
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        # Whether not to allow same words in a row, default is allow
        decoding_constraint = opt.get('decoding_constraint', 0)
        # Find beam search by max perplexity or max probability
        max_ppl = opt.get('max_ppl', 0)
        # Find no. of beams per group (for diverse beam search)
        bdash = beam_size // group_size # beam per group

        # INITIALIZATIONS
        # [(T,B')]*G
        beam_seq_table = [torch.LongTensor(self.seq_length, bdash).zero_() for _ in range(group_size)]
        # [(T,B')]*G
        beam_seq_logprobs_table = [torch.FloatTensor(self.seq_length, bdash).zero_() for _ in range(group_size)]
        # [(B',)]*G
        beam_logprobs_sum_table = [torch.zeros(bdash) for _ in range(group_size)]

        # logprobs # logprobs predicted in last time step, shape (beam_size, vocab_size+1)
        done_beams_table = [[] for _ in range(group_size)]
        # [(1,B,1)]->(1,1,B,1)->tuple((1,1,B/G,1))*G->[[(1,B/G,1)]*G]
        state_table = [list(torch.unbind(_)) for _ in torch.stack(init_state).chunk(group_size, 2)]
        # (B,V)->tuple((B/G,V))*G->[(B/G,V)]*G
        logprobs_table = list(init_logprobs.chunk(group_size, 0))
        # END INIT

        if len(args) > 0:
            # Chunk elements in the args
            if isinstance(args[0], list): # Ensemble of models, (enc_out, img_mask), [(B,L,E)*m], [(B,1,L)*m]
                # arg_name, model_name, group_name
                args = [[_.chunk(group_size) if _ is not None else [None]*group_size for _ in args_] for args_ in args]
                # group_name, arg_name, model_name
                args = [[[args[j][i][k] for i in range(len(self.models))] for j in range(len(args))] for k in range(group_size)]
            else: # Single model, (enc_out, img_mask), (B,L,E), (B,1,L)
                # [tuple(enc_out1, ... enc_outB/G)*G, tuple(img_mask1, ... img_maskB/G)*G]
                args = [_.chunk(group_size) if _ is not None else [None]*group_size for _ in args]
                # [[enc_out1, img_mask1], ... [enc_outB/G, img_maskB/G]]
                args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]

        for t in range(self.seq_length + group_size - 1):
            for divm in range(group_size):
                if t >= divm and t <= self.seq_length + divm - 1:
                    # Add diversity, (B/G,V)
                    logprobsf = logprobs_table[divm].data.float()
                    # Suppress previous word
                    if decoding_constraint and t-divm > 0:
                        logprobsf.scatter_(1, beam_seq_table[divm][t-divm-1].unsqueeze(1).to(init_logprobs), float('-inf'))
                    # Suppress <UNK> tokens in the decoding
                    logprobsf[:, logprobsf.size(1)-1] = logprobsf[:, logprobsf.size(1)-1] - 1000
                    # Diversity is added here
                    # the function directly modifies the logprobsf values and hence, we need to return
                    # the unaugmented ones for sorting the candidates in the end. # for historical
                    # reasons :-), (B/G,V)
                    unaug_logprobsf = add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash)

                    # infer new beams
                    beam_seq_table[divm],\
                    beam_seq_logprobs_table[divm],\
                    beam_logprobs_sum_table[divm],\
                    state_table[divm],\
                    candidates_divm = beam_step(logprobsf,
                                                unaug_logprobsf,
                                                bdash,
                                                t-divm,
                                                beam_seq_table[divm],
                                                beam_seq_logprobs_table[divm],
                                                beam_logprobs_sum_table[divm],
                                                state_table[divm])

                    # if time's up... or if end token is reached then copy beams
                    for vix in range(bdash):
                        if beam_seq_table[divm][t-divm,vix] == 0 or t == self.seq_length + divm - 1:
                            final_beam = {
                                'seq': beam_seq_table[divm][:, vix].clone(),
                                'logps': beam_seq_logprobs_table[divm][:, vix].clone(),
                                'unaug_p': beam_seq_logprobs_table[divm][:, vix].sum().item(),
                                'p': beam_logprobs_sum_table[divm][vix].item()
                            }
                            if max_ppl:
                                final_beam['p'] = final_beam['p'] / (t-divm+1)
                            done_beams_table[divm].append(final_beam)
                            # Don't continue beams from finished sequences
                            beam_logprobs_sum_table[divm][vix] = -1000

                    # Move the current group one step forward in time
                    it = beam_seq_table[divm][t-divm].to(init_logprobs.device)
                    # Get the log probability of the next word, (B, V), list-(1, B, t+1)
                    logprobs_table[divm], state_table[divm] = self.get_logprobs_state(it, *(args[divm] + [state_table[divm]]))

        # All beams are sorted by their log-probabilities
        done_beams_table = [sorted(done_beams_table[i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
        done_beams = reduce(lambda a,b:a+b, done_beams_table)
        return done_beams
