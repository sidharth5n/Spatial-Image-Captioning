from typing import Optional, Callable
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
import torch.nn as nn

from models.base import BaseModel

class CEModel(BaseModel):

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: OptimizerCallable,
                 scheduler: Optional[LRSchedulerCallable],
                 scheduled_sampling_scheduler: Optional[Callable],
                 beam_width: int
                 ):
        
        super().__init__(model, criterion, optimizer, scheduler)
    
    def training_step(self, batch, batch_idx):
        img_feats, labels, label_masks, img_masks, boxes = batch
        seqLogprobs = self.model(img_feats, labels, img_masks, boxes)
        loss = self.criterion(seqLogprobs, labels[:,1:], label_masks[:,1:])
        self.log('train/loss', loss, on_step = True, on_epoch = False, 
                 logger = True, prog_bar = True)
    
    
