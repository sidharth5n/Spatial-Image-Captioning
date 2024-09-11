from typing import Literal, Optional, Tuple, List, Dict
import os
import json
import numpy as np
from torch.utils.data import Dataset
# import h5py

class COCO(Dataset):

    def __init__(self,
                 input_json: str,
                 captions_per_image: int,
                 image_feature_dir: str,
                 normalize_image_feature: bool = False,
                 box_dir: Optional[str] = None,
                 normalize_box: bool = False,
                 split: Literal['train', 'val', 'test'] = 'train'
                 ):
        """
        Args:
            input_json (str): Path to processed json file
            captions_per_image (int): Number of captions per image
            image_feature_dir (str): Directory containing image features
            normalize_image_feature (bool, optional): Whether to normalize image feature. Defaults to False.
            box_dir (str): Directory containing bounding boxes
            normalize_box (bool, optional): Whether to normalize bounding box. Defaults to False.
            split (Literal[&#39;train&#39;, &#39;val&#39;, &#39;test&#39;]): Dataset split
        """
        self._captions_per_image = captions_per_image
        self.input_json = input_json
        self.image_feature_dir = image_feature_dir
        self.normalize_image_feature = normalize_image_feature
        self.box_dir = box_dir
        self.normalize_box = normalize_box
        # self.input_label_h5 = input_label_h5

        with open(input_json, 'r', encoding = 'utf-8') as f:
            info = json.load(f)
        info['images'] = list(filter(lambda img: img['split'] == split, info['images']))
        self.info = info
        
        self._captions = self._load_captions()
        
        self.get_image_feature = self._load_npz if self.image_feature_dir.endswith('att') else self._load_npy
    
    def __len__(self) -> int:
        return len(self.info['images'])#len(self.indices)
    
    @property
    def vocabulary(self) -> Dict[int, str]:
        """
        Dataset vocabulary

        Returns:
            Dict[int, str]: Vocabulary
        """
        return self.info['ix_to_word']
    
    @property
    def vocabulary_size(self) -> int:
        """
        Size of vocabulary

        Returns:
            int: Size of vocabulary
        """
        return len(self.vocabulary)
    
    @property
    def sequence_length(self) -> int:
        """
        Max length of caption

        Returns:
            int: Max length of caption
        """
        return self.info['seq_length']
    
    @property
    def captions_per_image(self)-> int:
        """
        Number of captions per image

        Returns:
            int: Captions per image
        """
        return self._captions_per_image
    
    def _load_captions(self) -> List[np.ndarray]:
        """
        Loads all captions from json file.

        Returns:
            List[np.ndarray]: Captions padded to sequence_length
        """
        all_captions = []
        for image in self.info['images']:
            seq = np.zeros((len(image['tokens']), self.sequence_length))
            for idx, token in enumerate(image.pop('tokens')):
                seq[idx, :len(token)] = token
            all_captions.append(seq)
        return all_captions

    @staticmethod
    def _load_npy(path):
        return np.load(os.path.splitext(path)[0] + '.npy')
    
    @staticmethod
    def _load_npz(path):
        return np.load(os.path.splitext(path)[0] + '.npz')['feat']
    
    def get_captions(self, ix: int) -> Tuple[np.ndarray, np.ndarray]:
        """

        Args:
            ix (int): Index

        Returns:
            Tuple[np.ndarray, np.ndarray]: Sampled captions for input, Ground truth captions
        """
        tokenized_captions = self._captions[ix]
        idxs = np.random.choice(len(tokenized_captions), self.captions_per_image, 
                                replace = len(tokenized_captions) < self.captions_per_image)
        return tokenized_captions[idxs], tokenized_captions

    def __getitem__(self, ix: int) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, Dict]:#index):
        # ix = self.indices[index]
        
        img_feat = self.get_image_feature(os.path.join(self.image_feature_dir, str(self.info['images'][ix]['id'])))
        if len(img_feat.shape) > 2:
            img_feat = img_feat.reshape(-1, img_feat.shape[-1])
        if self.normalize_image_feature:
            img_feat = img_feat / np.linalg.norm(img_feat, 2, 1, keepdims = True)
        
        if self.box_dir:
            box_file = os.path.join(self.box_dir, str(self.info['images'][ix]['id']) + '.npy')
            boxes = np.load(box_file)
        else:
            boxes = None
        
        seq, gts = self.get_captions(ix)

        info_dict = {'ix' : ix,
                     'id' : self.info['images'][ix]['id'],
                     'file_path' : self.info['images'][ix]['file_path']}
        
        return img_feat, boxes, seq, gts, info_dict