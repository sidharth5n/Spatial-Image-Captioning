import torch.nn as nn

from misc.utils import to_contiguous

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