import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.dataset.coco import COCO

class COCODataModule(pl.LightningDataModule):

    def __init__(self,
                 root: str,
                 input_json: str,
                 input_fc_feat_dir: str,
                 input_img_feat_dir: str,
                 input_box_dir: str,
                 input_rel_box_dir: str,
                 input_grid_vec_dir: str,
                 input_label_h5: str,
                 norm_img_feat: bool,
                 norm_box_feat: bool,
                 seq_per_img: int,
                 batch_size: int,
                 trainonly: bool):
        
        self.input_json = os.path.join(root, input_json)
        self.input_fc_feat_dir = os.path.join(root, input_fc_feat_dir)
        self.input_img_feat_dir = os.path.join(root, input_img_feat_dir)
        self.input_box_dir = os.path.join(root, input_box_dir)
        self.input_rel_box_dir = os.path.join(root, input_rel_box_dir)
        self.input_grid_vec_dir = os.path.join(root, input_grid_vec_dir)
        self.input_label_h5 = os.path.join(root, input_label_h5)
        self.norm_img_feat = norm_img_feat
        self.norm_box_feat = norm_box_feat
        self.batch_size = batch_size
        self.trainonly = trainonly
    
    def prepare(self):

        preprocess_labels(self.input_json, self.output_json, self.output_h5, self.image_root, self.max_length, self.word_count_threshold)
        preprocess_ngrams(self.output_json, self.input_json, self.output_pkl, 'all')
        

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_data = COCO()
            self.val_data = COCO()
        
        elif stage == 'validate':
            self.val_data = COCO()
        
        elif stage == 'test':
            self.test_data = COCO()
    
    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size = self.batch_size,
                          num_workers = self.num_workers,
                          pin_memory = True,
                          shuffle = True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size = self.batch_size,
                          num_workers = self.num_workers,
                          pin_memory = True,
                          shuffle = False)
    
    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size = self.batch_size,
                          num_workers = self.num_workers,
                          pin_memory = True,
                          shuffle = False)