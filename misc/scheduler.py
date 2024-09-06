from torch.optim.lr_scheduler import LambdaLR

class ScheduledSamplingProbScheduler:

    def __init__(self,
                 start_epoch: int,
                 frequency: int,
                 factor: float,
                 max_prob: float
                 ):
        self.start_epoch = start_epoch
        self.frequency = frequency
        self.factor = factor
        self.max_prob = max_prob
    
    def __call__(self, step):
        if self.start_epoch > 0 and step > self.start_epoch:
            frac = (step - self.start_epoch) // self.frequency
            return min(self.factor * frac, self.max_prob)
        else:
            return 0.0

class DefaultScheduler(LambdaLR):

    def __init__(self,
                 optimizer,
                 start: int,
                 frequency: int,
                 factor: float):
        lr_lambda = lambda step: factor ** ((step - start) // frequency) if (start >= 0 and step > start) else 1
        super().__init__(optimizer, lr_lambda)

class NoamScheduler(LambdaLR):

    def __init__(optimizer, model_size, factor, warmup):
        lr0 = factor * d_model ** (-0.5)
        lr_lambda = lambda step: lr0 * min(step ** (-0.5), step * warmup ** (-1.5))
        super().__init__(optimizer, lr_lambda)