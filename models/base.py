from typing import Optional, Callable
from abc import ABC, abstractmethod
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable

from misc.scheduler import DefaultScheduler

class BaseModel(pl.LightningModule, ABC):

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: OptimizerCallable,
                 scheduler: Optional[LRSchedulerCallable],
                 scheduled_sampling_scheduler: Optional[Callable],
                 beam_width: int,
                 ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.beam_width = beam_width
        self.scheduled_sampling_scheduler = scheduled_sampling_scheduler
        self.predictions = []
        self.beam_predictions = []

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

        indices = torch.arange(0, batch_size, loader.get_seq_per_img())
        seq = self.model(img_feats[indices], img_masks[indices], boxes[indices], opt = eval_kwargs, mode = 'sample')[0]
        
        if self.beam_width > 1:
            for i in range(len(indices)):
                self.beam_predictions.append([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]])
        
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            image_id = data['infos'][k]['id']
            entry = {'image_id'  : image_id,
                     'caption'   : sent,
                     'file_path' : data['infos'][k]['file_path']}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            self.predictions.append(entry)
        
        self.log('val/loss', loss, on_step = False, on_epoch = True,
                 prog_bar = True, logger = True)
        
        return loss
    
    def on_validation_epoch_end(self):

        if self.beam_width > 1:
            with open(os.path.join(self.trainer.default_rootdir, f'captions_{self.trainer.current_epoch}.txt'), 'w') as f:
                for sents in self.beam_predictions:
                    f.write("\n".join(sents))
                    f.write('--' * 10 + '\n')
            self.beam_predictions.clear()
        
        # Peform language evaluation on the generated captions
        lang_stats = language_eval(dataset, self.predictions, eval_kwargs.get('id'),
                                eval_kwargs.get('image_root'), split)


        self.log('val/cider', lang_stats['CIDEr'], on_step = False, on_epoch = True,
                    logger = True, prog_bar = True)
        self.log('val/spice', lang_stats['cider'], on_step = False, on_epoch = True,
                    logger = True, prog_bar = True)
        self.log('val/rouge', lang_stats['cider'], on_step = False, on_epoch = True,
                    logger = True, prog_bar = True)
        self.log('val/cider', lang_stats['cider'], on_step = False, on_epoch = True,
                    logger = True, prog_bar = True)
        
        self.predictions.clear()
    
    @abstractmethod
    def test_step(self, batch, batch_idx):
        raise NotImplementedError
    
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