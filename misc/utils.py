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
