import os
import json
import numpy as np
from torch.utils.data import Dataset
import h5py

class COCO(Dataset):

    def __init__(self,
                 seq_per_img: int,
                 use_box: bool,
                 norm_att_feat: bool,
                 norm_box_feat: bool,
                 input_json: str,
                 input_img_feat_dir: str,
                 input_rel_box_dir: str,
                 input_label_h5: str,
                 ):
        
        self._seq_per_img = seq_per_img
        self.use_box = use_box
        self.norm_att_feat = norm_att_feat
        self.norm_box_feat = norm_box_feat
        self.input_json = input_json
        self.input_img_feat_dir = input_img_feat_dir
        self.rel_bboxes_dir = input_rel_box_dir
        self.input_label_h5 = input_label_h5

        with open(input_json, 'r', 'utf-8') as f:
            self.info = json.load(f)
        
        self.ix_to_word = self.info['ix_to_word']
        self._vocab_size = len(self.ix_to_word)

        self.h5_label_file = h5py.File(input_label_h5, 'r', driver = 'core')
        
        seq_size = self.h5_label_file['labels'].shape
        self._seq_length = seq_size[1]

        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        self.indices = []
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == split:
                self.indices.append(ix)
        
        self.iterator = 0
    
    def __len__(self):
        return len(self.indices)
    
    @property
    def vocab(self):
        return self.ix_to_word
    
    @property
    def vocabulary_size(self):
        return self._vocab_size
    
    @property
    def sequence_length(self):
        return self._seq_length
    
    @property
    def captions_per_image(self):
        return self._seq_per_img
    
    def get_captions(self, ix, seq_per_img):
        ix1 = self.label_start_ix[ix] - 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1
        if ncap < eq_per_img:
            seq = np.zeros([seq_per_img, self.sequence_length], dtype = int)
            for q in range(seq_per_img):
                ix1 = andom.randint(ix1, ix2)
                seq[q, :] = self.h5_label_file['labels'][ix1, :self.sequence_length]
        else:
            ix1 = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.h5_label_file['labels'][ix1:ix1 + seq_per_img, :self.sequence_length]
    
        return seq

    def __getitem__(self, index):
        ix = self.indices[index]
        
        img_feat = np.load(os.path.join(self.input_img_feat_dir, str(self.info['images'][ix]['id']) + '.npz'))['feat']
        if len(img_feat.shape) > 2:
            img_feat = img_feat.reshape(-1, img_feat.shape[-1])
        if self.norm_img_feat:
            img_feat = img_feat / np.linalg.norm(img_feat, 2, 1, keepdims = True)
        
        if self.use_box:
            box_file = os.path.join(self.rel_bboxes_dir, str(self.info['images'][ix]['id']) + '.npy')
            boxes = np.load(box_file)
        else:
            boxes = None
        
        seq = self.get_captions(ix, self.captions_per_image)

        gts = self.h5_label_file['labels'][self.label_start_ix[ix] - 1:self.label_end_ix[ix]]

        info_dict = {'ix' : ix,
                     'id' : self.info['images'][ix]['id'],
                     'file_path' : self.info['images'][ix]['file_path']}
        
        return img_feat, boxes, seq, gts, info_dict