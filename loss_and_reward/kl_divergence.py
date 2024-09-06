class LabelSmoothingLoss(nn.Module):
    """
    Implements label smoothing and computes KL divergence between the two
    distributions.
    """
    def __init__(self,
                 vocab_size: int,
                 smoothing: float = 0.0):
        """
        Parameters
        ----------
        vocab_size : int
                     Size of vocabulary
        smoothing  : float
                     Smoothing value between 0 and 1.
        """
        super().__init__()
        assert 0.0 <= smoothing < 1.0, "Smoothing should be between 0 and 1"
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