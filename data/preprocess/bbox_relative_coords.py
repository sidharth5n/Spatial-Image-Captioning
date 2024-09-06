"""
Preprocess the absolute bounding box coordinates from --input_box_dir,
To convert these into relative coordinates, this script loads the corresponding images from coco/val2014/ and coco/train2014/, to get (img_width, img_height)

Input:
input_box_dir="/mydisk/Data/captioning_data/cocobu_adaptive_box"
info_filepath="/mydisk/Data/captioning_data/dataset_coco.json"
img_dir      ="/mydisk/Data/captioning_data/coco"

Output:
A directory containing all the boxes relative coordinates, as npy files.
"""

import os
from glob import glob
import re
import json
import numpy as np
import argparse
from tqdm import tqdm

def get_bbox_relative_coords(input_box_dir: str,
                             input_json: str,
                             image_root: str,
                             output_dir: str):

    with open(input_json, "rb") as infile:
        coco_dict = json.load(infile)
    
    coco_ids_to_paths = {str(img['cocoid']): os.path.join(image_root, img['filepath'], img['filename']
                         for img in coco_dict['images'] }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    box_paths = sorted(glob(os.path.join(input_box_dir,'*')))
    for ind, box_path in tqdm(enumerate(box_paths)):
        filenumer = os.path.splitext(os.path.basename(box_path))[0]
        img_path = coco_ids_to_paths[filenumber]
        width, height = imagesize.get(img_path)
        box = np.load(box_file)
        relative_box = box / np.array([width, height, width,height])
        relative_box = np.clip(relative_box,0.0,1.0)
        new_filename = os.path.join(output_dir, filenumber + '.npy')
        np.save(new_filename, relative_box)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str, default='/mydisk/Data/captioning_data/dataset_coco.json', help='input json file to process into hdf5')
    parser.add_argument('--image_root', type=str, default='/mydisk/Data/captioning_data/coco',
                    help='In case the image paths have to be preprended with a root path to an image folder')
    parser.add_argument('--input_box_dir', type=str, default='/mydisk/Data/captioning_data/cocobu_adaptive_box',
                    help='path to the directory containing the boxes of att feats')
    parser.add_argument('--output_dir', type=str, default='/mydisk/Data/captioning_data/zcocobu_adaptive_box_relative',
                    help='directory containing the files with relative coordinates of the bboxes in --input_box_dir')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    get_bbox_relative_coords(params)
