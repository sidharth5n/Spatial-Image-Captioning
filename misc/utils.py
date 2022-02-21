from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import numpy as np
import copy
import sys

try:
    import cPickle as pickle
except:
    import pickle

def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def apply_along_batch(func, M):
    #apply torch function for each image in a batch, and concatenate results back into a single tensor
    tensorList = [func(m) for m in torch.unbind(M, dim=0) ]
    result = torch.stack(tensorList, dim=0)
    return result

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        out.append(txt)
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    """
    Loss based on self critical reward.
    """
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        """
        Computes log(y_t) * reward * mask_t  (where mask_t zeroes out non-words
        in the sequence)

        Parameters
        ----------
        input  : torch.tensor of shape (B, T)
                 Log probability of the samples in the generated sequence.
        seq    : torch.tensor of shape (B, T)
                 Generated seqeuence
        reward : torch.tensor of shape (B, T)
                 Self critical reward

        Returns
        -------
        output : torch.tensor of shape ([])
                 Mean loss
        """

        input = to_contiguous(input).view(-1) # (B,T)->(B*T,)
        reward = to_contiguous(reward).view(-1) # (B,T)->(B*T,)
        mask = (seq > 0).float() # (B,T)
        # Add additional 1 in the beginning to include <END> in the mask
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1) # (B*T,1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class LanguageModelCriterion(nn.Module):
    """
    Cross entropy loss
    """
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        """
        Parameters
        ----------
        input  : torch.tensor of shape (B, T, V)
                 Log probability distribution over vocabulary of the input
                 sequence.
        target : torch.tensor of shape (B, T)
                 Padded ground truth sequence.
        mask   : torch.tensor of shape (B, T)
                 Mask for finding the loss of actual sequence only.

        Returns
        -------
        output : torch.tensor of shape ([])
                 Mean cross entropy loss
        """
        # Truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        # Compute cross entropy loss
        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        # Compute mean loss
        output = torch.sum(output) / torch.sum(mask)
        return output

class ContextRegularization(nn.Module):
    """
    Encourages the context lengths to decrease by 1 across the
    decoder stack.

    No regularization on the first N query positions as they
    cannot enforce a difference of 1.
    """
    def __init__(self):
        super(ContextRegularization, self).__init__()
        print("Context Regularization")

    def forward(self, contexts):
        # [(B,H,Q,1)]*N
        N = len(contexts)
        contexts = torch.cat(contexts, dim = -1)
        contexts = contexts[:, :, N:]
        contexts = contexts.reshape(-1, N)
        diff = contexts[:, 1:] - contexts[:, :-1]
        return torch.mean((diff + 1)**2)

class LabelSmoothing(nn.Module):
    """
    Implements label smoothing and computes KL divergence between the two
    distributions.
    """
    def __init__(self, vocab_size, smoothing=0.0):
        """
        Parameters
        ----------
        vocab_size : int
                     Size of vocabulary
        smoothing  : float
                     Smoothing value between 0 and 1.
        """
        assert 0.0 <= smoothing < 1.0, "Smoothing should be between 0 and 1"
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction = 'none')
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing / (vocab_size - 1)

    def forward(self, input, target, mask):
        """
        Computes the mean KL divergence loss with label smoothing.

        Parameters
        ----------
        input  : torch.tensor of shape (B, T, V)
                 Log probability distribution over vocabulary of the input
                 sequence.
        target : torch.tensor of shape (B, T)
                 Padded ground truth sequence.
        mask   : torch.tensor of shape (B, T)
                 Mask for finding the loss of actual sequence only.

        Returns
        -------
        loss   : torch.tensor of shape ([])
                 Mean KL divergence loss
        """
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        # Remove time axis
        input = to_contiguous(input).view(-1, input.size(-1)) # (B,T,V)->(B*T,V)
        target = to_contiguous(target).view(-1) # (B,T)->(B*T,)
        mask = to_contiguous(mask).view(-1) # (B,T)->(B*T,)
        # Compute label smooth distribution
        true_dist = input.data.clone()
        true_dist.fill_(self.smoothing)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # Compute mean KL divergence loss (B*T,V)->(B*T)->([])
        loss = (self.criterion(input, true_dist).sum(1) * mask).sum() / mask.sum()
        return loss

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group['lr']

def clip_gradient(optimizer, grad_clip):
    i = 0
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is None:
                print(param.shape, i)
            param.grad.data.clamp_(-grad_clip, grad_clip)
            i += 1

def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))


class NoamOpt(object):
    """
    Optim wrapper that implements rate variation as proposed in original
    Transformer paper.
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        """
        model_size : int
                     Size of the target embedding
        factor     : float
                     Coefficient for scaling the learning rate
        warmup     : int
                     No. of steps of increasing learning rate linearly.
        optimizer  : torch.optim
                     Optimizer
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """
        Updates the training step, learning rate in the optimizer and updates the
        parameters within the optimizer.
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        """
        Computes the learning rate as given below :
        lr = factor * d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))

        This increases the learning rate linearly for the first warmup training
        steps, and decreases it thereafter proportionally to the inverse square
        root of the step number.

        Parameters
        ----------
        step : int, optional
               Training step. Default is None.

        Returns
        -------
        lr   : float
               Learning rate for the given step or the current learning rate
               being used.
        """
        if step is None:
            step = self._step
        lr = self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
        return lr

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

class ReduceLROnPlateau(object):
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, mode = 'min', factor = 0.1, patience = 10, verbose = False, threshold = 0.0001, threshold_mode = 'rel', cooldown = 0, min_lr = 0, eps = 1e-08):
        """
        Parameters
        ----------
        optimizer : torch.optim
                    Optimizer to be used

        Refer PyTorch documentation on optim.lr_scheduler.ReduceLROnPlateau for
        details about remaining arguments.
        """
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode, factor, patience, verbose, threshold, threshold_mode, cooldown, min_lr, eps)
        self.optimizer = optimizer
        self.current_lr = get_lr(optimizer)

    def step(self):
        "Update parameters and rate"
        self.optimizer.step()

    def scheduler_step(self, val):
        self.scheduler.step(val)
        self.current_lr = get_lr(self.optimizer)

    def state_dict(self):
        return {'current_lr':self.current_lr,
                'scheduler_state_dict': {key: value for key, value in self.scheduler.__dict__.items() if key not in {'optimizer', 'is_better'}},
                'optimizer_state_dict': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        """
        Loads the given optimizer state dictionary.

        Parameters
        ----------
        state_dict : collections.OrderedDict
                     State dictionary of the optimizer
        """
        if 'current_lr' not in state_dict:
            # it's normal optimizer
            self.optimizer.load_state_dict(state_dict)
            set_lr(self.optimizer, self.current_lr) # use the lr from the option
        else:
            # it's a schduler
            self.current_lr = state_dict['current_lr']
            self.scheduler.__dict__.update(state_dict['scheduler_state_dict'])
            self.scheduler._init_is_better(mode=self.scheduler.mode, threshold=self.scheduler.threshold, threshold_mode=self.scheduler.threshold_mode)
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            # current_lr is actually useless in this case

    def rate(self, step = None):
        """
        Updates the learning rate as given below :
        lr = factor * d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))

        This increases the learning rate linearly for the first warmup training
        steps, and decreases it thereafter proportionally to the inverse square
        root of the step number.

        Parameters
        ----------
        step : int, optional
               Training step. Default is None.

        Returns
        -------
        lr   : float
               Learning rate for the given step or the current learning rate
               being used.
        """
        if step is None:
            step = self._step
        lr = self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
        return lr

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

def clones(module, N):
    """
    Produce N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def clip_att(img_feats, boxes = None, img_masks = None):
    """
    Clip the length of img_masks, boxes and img_feats to the maximum length.
    This is required when multiple GPUs are being used.

    Parameters
    ----------
    img_feats : torch.tensor of shape (B, L, D)
                Output of last conv layer of CNN or bottom-up features
    boxes     : torch.tensor of shape (B, L, 4), optional
                Coordinates of bounding boxes. Default is None.
    img_masks : torch.tensor of shape (B, L, L, 3) or (B, L)
                Attention mask when no. of bottom-up proposals are
                unequal across batch. Default is None.

    Returns
    -------
    img_feats : torch.tensor of shape (B, P, D)
                Output of last conv layer of CNN or bottom-up features
    boxes     : torch.tensor of shape (B, P, 4) or None
                Coordinates of bounding boxes.
    img_masks : torch.tensor of shape (B, P) or None
                Attention mask when no. of bottom-up proposals are
                unequal across batch.

    P is the maximum length of features in the batch. P <= L
    """
    if img_masks is not None:
        if img_masks.ndim == 4:
            max_len = img_masks.long().sum([3,2]).max().max()
        else:
            max_len = img_masks.long().sum(1).max()
        img_feats = img_feats[:, :max_len].contiguous()
        img_masks = img_masks[:, :max_len].contiguous()
        if boxes is not None:
            boxes = boxes[:, :max_len].contiguous()

    return img_feats, boxes, img_masks

def subsequent_mask(size):
    """
    Returns a lower triangular mask for causal attention.

    Parameters
    ----------
    size            : int
                      Size of the mask
    Returns
    -------
    subsequent_mask : torch.tensor of shape (1, size, size)
                      Causal mask
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.tril(np.ones(attn_shape), k=0).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 1

def sort_pack_padded_sequence(input, lengths):
    """
    Packs the given input sequence and returns the indices to unsort the data.

    Parameters
    ----------
    input  : torch.tensor of shape (B, *)
             Padded sequence
    length : torch.tensor of shape (B,)
             Length of each sequence

    Returns
    -------
    tmp    : PackedSequence
             Packed input
    inv_ix : torch.tensor of shape (B,)
             Indices to unsort the data
    """
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    """
    Pads a packed sequence and returns the data in the order specified.

    Parameters
    ----------
    input  : PackedSequence
             Packed input
    inv_ix : torch.tensor of shape (B,)
             Indices to unsort the data

    Returns
    -------
    tmp    : torch.tensor of shape (B, *)
             Padded data in the order specified in inv_ix
    """
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, img_feats, img_masks):
    """
    Applies the given module on img_feats.

    Parameters
    ----------
    module    : Module to be applied on img_feats
    img_feats : torch.tensor of shape (B, L, D)
    img_masks  : torch.tensor of shape (B, L) or (B, L, L, 3)

    Returns
    -------
    img_feats : torch.tensor of shape (B, L, E)
    """
    if img_masks is not None:
        if img_masks.ndim == 4:
            packed, inv_ix = sort_pack_padded_sequence(img_feats, img_masks.long().sum([3,2])[:,0])
        else:
            packed, inv_ix = sort_pack_padded_sequence(img_feats, img_masks.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(img_feats)

def get_std_opt(model, factor=1, warmup=2000):
    # return NoamOpt(model.tgt_embed[0].d_model, 2, 4000,
    #         torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    return NoamOpt(model.model.tgt_embed[0].d_model, factor, warmup,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def want_to_continue(found_issue):
    """
    Get user input whether to proceed further or not based on the issue found out.

    Parameters
    ----------
    found_issue : str
                  The issue found out
    Returns
    -------
    bool
    Whether to continue or not
    """
    print('--' * 10)
    print(found_issue + '. Would you like to continue? [y/N]')

    yes = {'yes','y', 'ye', 'Y'}
    no = {'no','n','','N'}

    choice = input().lower()
    if choice in yes:
        return True
    elif choice in no:
        return False
    else:
        sys.stdout.write("Please respond with 'y' or 'N'")

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
