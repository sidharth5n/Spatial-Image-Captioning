from typing import List, Tuple, Optional, Dict, Union
import numpy as np
import torch

def pad_collate(batch: List[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, Dict]],
                captions_per_image: int,
                max_caption_length: int
                ) -> Dict[str, Union[Optional[torch.Tensor], np.ndarray, dict]]:
    """
    Pads the items received from dataset to same size and collates them.

    Args:
        batch (List[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, Dict]]): _description_
        captions_per_image (int): No. of captions per image
        max_caption_length (int): Maximum length of caption

    Returns:
        Dict[str, Union[Optional[torch.Tensor], np.ndarray, dict]]: _description_
    """
    img_feats, boxes, seq, gts, infos = zip(*batch)
    batch_size = len(batch)
    max_img_feat_len = max([feat.shape[0] for feat in img_feats])
    use_box = boxes[0] is not None
    
    n = batch_size * captions_per_image
    
    data = {'img_feats' : np.zeros((n, max_img_feat_len, img_feats[0].shape[1]), dtype = np.float32),
            'img_masks' : np.zeros((n, max_img_feat_len), dtype = np.float32),
            'boxes' : np.zeros((n, max_img_feat_len, boxes[0].shape[1]), dtype = np.float32) if use_box else None,
            'labels' : np.zeros((n, max_caption_length + 2), dtype = np.int16),
            'label_masks' : np.zeros((n, max_caption_length + 2), dtype = np.float32)}
    
    for i in range(batch_size):
        a = i * captions_per_image
        b = (i + 1) * captions_per_image
        data['img_feats'][a:b, :img_feats[i].shape[0]] = img_feats[i]
        data['img_masks'][a:b, :img_feats[i].shape[0]] = 1
        if use_box:
            data['boxes'][a:b, :boxes[i].shape[0]] = boxes[i]
        data['labels'][a:b, 1:max_caption_length + 1] = seq[i]
        lengths = (seq[i] != 0).sum(-1) + 2
        for j, l in enumerate(lengths):
            data['label_masks'][a + j, :l] = 1
    
    data = {key : (torch.from_numpy(value) if value else None) for key, value in data.items()}
    
    data['gts'] = gts
    data['infos'] = infos
    
    return data