from typing import Optional, Callable, List, Dict, Union
from abc import ABC, abstractmethod
import os
import json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable

from misc.scheduler import DefaultScheduler
from loss_and_reward.eval import language_eval
import misc.utils as utils

class BaseModel(pl.LightningModule, ABC):

    def __init__(self,
                 model: Callable[[int], nn.Module],
                 criterion: nn.Module,
                 optimizer: OptimizerCallable,
                 scheduler: Optional[LRSchedulerCallable],
                 scheduled_sampling_scheduler: Optional[Callable],
                #  vocab: Dict[int, str],
                 captions_per_image: int,
                 beam_width: int,
                 temperature: float,
                 decoding_constraint: bool,
                 image_root: str
                 ):
        super().__init__()
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.captions_per_image = captions_per_image
        self.beam_width = beam_width
        self.temperature = temperature
        self.decoding_constraint = decoding_constraint
        self.scheduled_sampling_scheduler = scheduled_sampling_scheduler
        # self.vocab = vocab
        self.image_root = image_root
        self.predictions: List[Dict] = []
        self.beam_predictions: List[Dict[str, Union[int, List[str]]]] = []
    
    @property
    def vocab(self):
        return self.trainer.datamodule.vocabulary
    
    def configure_model(self):
        print("model configuring")
        if not isinstance(self, nn.Module):
            self.model = self.model(self.trainer.datamodule.vocabular_size)
            
    def on_train_epoch_start(self):
        if self.scheduled_sampling_scheduler:
            self.model.ss_prob = self.scheduled_sampling_scheduler(self.trainer.current_epoch)
        
    @abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def validation_step(self, batch, batch_idx):
        img_feats, labels, label_masks, img_masks, boxes = batch
        
        seqLogprobs = self.model(img_feats, labels, img_masks, boxes)
        loss = self.criterion(seqLogprobs, labels[:,1:], label_masks[:,1:])

        indices = torch.arange(0, img_feats.shape[0], self.captions_per_image, device = img_feats.device)
        seq = self.model(img_feats[indices], img_masks[indices], boxes[indices] if boxes else None,
                         beam_width = self.beam_width,
                         temperature = self.temperature,
                         decoding_constraint = self.decoding_constraint,
                         mode = 'sample')[0]
        
        if self.beam_width > 1:
            for i in range(len(indices)):
                self.beam_predictions.append({'image_id' : batch['infos'][i // self.captions.per_image]['id'],
                                              'captions': [utils.decode_sequence(self.vocab, _['seq'].unsqueeze(0))[0] for _ in self.model.done_beams[i]]})
        
        sents = utils.decode_sequence(self.vocab, seq)

        for k, sent in enumerate(sents):
            image_id = batch['infos'][k]['id']
            entry = {'image_id'  : image_id,
                     'caption'   : sent,
                     'file_path' : batch['infos'][k]['file_path']}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = batch['infos'][k]['file_path']
            self.predictions.append(entry)
        
        self.log('val/loss', loss, on_step = False, on_epoch = True,
                 logger = True, prog_bar = True)
        
        return loss
    
    def on_validation_epoch_end(self):

        if self.beam_width > 1:
            with open(os.path.join(self.trainer.default_root_dir, f'captions_beam={self.beam_width}.json'), 'w', encoding = 'utf-8') as f:
                json.dump(self.beam_predictions, f)
            self.beam_predictions.clear()
        
        # Peform language evaluation on the generated captions
        lang_stats = language_eval(self.predictions,
                                   self.image_root,
                                   self.trainer.default_root_dir,
                                   'val')

        for key, val in [('CIDEr', 'C'), ('METEOR', 'M'), ('ROUGE_L', 'R'), ('Bleu_4', 'B4'), ('SPICE.All.f', 'S')]:
            if key in lang_stats:
                self.log(f'val/{val}', lang_stats[key], on_step = False, on_epoch = True,
                         logger = True, prog_bar = True)
        
        self.predictions.clear()
    
    # @abstractmethod
    def test_step(self, batch, batch_idx):
        # raise NotImplementedError
        pass
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters())

        if self.scheduler:
            scheduler = self.scheduler(optimizer)
            return {'optimizer': optimizer,
                    'lr_scheduler': {'scheduler': scheduler, 
                                     'interval': 'epoch' if isinstance(scheduler, DefaultScheduler) else 'step',
                                     'monitor': 'val/CIDEr' if isinstance(scheduler, ReduceLROnPlateau) else None
                                     }
                   }
        
        return optimizer