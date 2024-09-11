from typing import Literal, Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from functools import partial

from data.dataset.coco import COCO
from data.preprocess import (preprocess_labels,
                             preprocess_document_frequency,
                             make_bu_data,
                             get_bbox_relative_coords
                             )
from misc.utils import validate_path
from data.modules.utils import pad_collate


class COCODataModule(pl.LightningDataModule):

    def __init__(self,
                 image_root: str,
                 input_json: str,
                 output_json: str,
                 bottomup_root: str,
                 bu_data: str,
                 image_feature: Literal['fc', 'att'],
                 normalize_image_feature: bool,
                 box_feature: Optional[Literal['box', 'box_relative']],
                 normalize_box: bool,
                 output_pkl: str,
                 captions_per_image: int,
                 max_caption_length: int,
                 min_word_frequency: int,
                 num_workers: int,
                 batch_size: int
                 ):
        
        super().__init__()
        
        validate_path(image_root, 'image_root')
        validate_path(input_json, 'input_json', '.json')
        validate_path(bottomup_root, 'bottomup_root')
        validate_path(output_json, 'output_json', '.json', False)
        validate_path(bu_data, 'bu_data', exists = False)
        
        self.image_root = image_root
        self.input_json = input_json
        self.bottom_up = bottomup_root
        
        self.output_json = output_json
        self.bu_data = bu_data
        self.image_feature = image_feature
        self.normalize_image_feature = normalize_image_feature
        self.box_feature = box_feature
        self.normalize_box = normalize_box
        self.output_pkl = output_pkl
        
        self.captions_per_image = captions_per_image
        self.max_caption_length = max_caption_length
        self.min_word_frequency = min_word_frequency
        
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    @property
    def vocabulary(self):
        print("datamodule vocabulary called")
        if hasattr(self, 'train_data'):
            return self.train_data.vocabulary
        elif hasattr(self, 'val_data'):
            return self.val_data.vocabulary
        elif hasattr(self, 'test_data'):
            return self.test_data.vocabulary
        else:
            raise ValueError('vocabulary not available')
    
    @property
    def vocabulary_size(self):
        print("datamodule vocabulary size")
        if hasattr(self, 'train_data'):
            print("in tran")
            return self.train_data.vocabulary_size
        elif hasattr(self, 'val_data'):
            print("in val")
            return self.val_data.vocabulary_size
        elif hasattr(self, 'test_data'):
            print("in test")
            return self.test_data.vocabulary_size
        else:
            raise ValueError('vocabulary not available')
        
    def prepare_data(self):
        print("running data prepare")
        preprocess_labels(self.input_json, self.output_json, self.image_root, self.max_caption_length, self.min_word_frequency)
        preprocess_document_frequency(self.output_json, self.input_json, self.output_pkl, 'all')
        make_bu_data(self.input_json, self.bottom_up, self.bu_data)
        get_bbox_relative_coords(self.bu_data + '_box', self.input_json, self.image_root, self.bu_data + '_box_relative')

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_data = COCO(self.output_json,
                                   self.captions_per_image,
                                   self.bu_data + f'_{self.image_feature}',
                                   self.normalize_image_feature,
                                   self.bu_data + f'_{self.box_feature}' if self.box_feature else None,
                                   self.normalize_box,
                                   'train')
            self.val_data = COCO(self.output_json,
                                 self.captions_per_image,
                                 self.bu_data + f'_{self.image_feature}',
                                 self.normalize_image_feature,
                                 self.bu_data + f'_{self.box_feature}' if self.box_feature else None,
                                 self.normalize_box,
                                 'val')
        
        elif stage == 'validate':
            self.val_data = COCO(self.output_json,
                                 self.captions_per_image,
                                 self.bu_data + f'_{self.image_feature}',
                                 self.normalize_image_feature,
                                 self.bu_data + f'_{self.box_feature}' if self.box_feature else None,
                                 self.normalize_box,
                                 'val')
        
        elif stage == 'test':
            self.test_data = COCO(self.output_json,
                                  self.captions_per_image,
                                  self.bu_data + f'_{self.image_feature}',
                                  self.normalize_image_feature,
                                  self.bu_data + f'_{self.box_feature}' if self.box_feature else None,
                                  self.normalize_box,
                                  'test')
    
    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size = self.batch_size,
                          num_workers = self.num_workers,
                          pin_memory = True,
                          shuffle = True,
                          collate_fn = partial(pad_collate,
                                               captions_per_image = self.captions_per_image,
                                               max_caption_length = self.max_caption_length))
    
    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size = self.batch_size,
                          num_workers = self.num_workers,
                          pin_memory = True,
                          shuffle = False,
                          collate_fn = partial(pad_collate,
                                               captions_per_image = self.captions_per_image,
                                               max_caption_length = self.max_caption_length))
    
    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size = self.batch_size,
                          num_workers = self.num_workers,
                          pin_memory = True,
                          shuffle = False,
                          collate_fn = partial(pad_collate,
                                               captions_per_image = self.captions_per_image,
                                               max_caption_length = self.max_caption_length))