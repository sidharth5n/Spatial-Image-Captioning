"""
Preprocess the absolute bounding box coordinates to relative coordinates.
Note : Requires image to get width and height.
"""

import os
from glob import glob
import json
import argparse
import numpy as np
from tqdm import tqdm
import imagesize


def get_bbox_relative_coords(input_box_dir: str,
                             input_json: str,
                             image_root: str,
                             output_dir: str):
    """
    Compute and save relative coordinates of bounding box.

    Args:
        input_box_dir (str): Directory containing bounding box information.
        input_json (str): Path to Karpathy json file.
        image_root (str): Directory containing images.
        output_dir (str): Directory where relative bounding boxes are to be saved.
    """
    with open(input_json, "r", encoding = 'utf-8') as f:
        coco_dict = json.load(f)
    
    coco_ids_to_paths = {str(img['cocoid']): os.path.join(image_root, img['filepath'], img['filename'])
                         for img in coco_dict['images']}
    
    remaining_files = list(filter(not os.path.exists, map(lambda img_id: os.path.join(output_dir, str(img_id) + '.npy'),
                                                          coco_ids_to_paths.keys())))

    os.makedirs(output_dir, exist_ok = True)

    for box_path in tqdm(remaining_files):
        filenumber = os.path.splitext(os.path.basename(box_path))[0]
        img_path = coco_ids_to_paths[filenumber]
        width, height = imagesize.get(img_path)
        box = np.load(os.path.join(input_box_dir, os.path.basename(box_path)))
        relative_box = box / np.array([width, height, width,height])
        relative_box = np.clip(relative_box,0.0,1.0)
        np.save(box_path, relative_box)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type = str, default = 'data/dataset_coco.json',
                        help = 'Path to Karpathy json file')
    parser.add_argument('--image_root', type=str, default = 'data/images',
                        help = 'Directory containing images')
    parser.add_argument('--input_box_dir', type=str, default = 'data/cocobu_box',
                        help = 'Directory containing bounding box information')
    parser.add_argument('--output_dir', type=str, default = 'data/cocobu_box_relative',
                        help = 'Directory where relative bounding boxes are to be saved')

    args = parser.parse_args()
    
    get_bbox_relative_coords(**vars(args))
