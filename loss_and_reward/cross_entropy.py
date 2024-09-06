import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    """
    Cross entropy loss
    """

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