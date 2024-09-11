"""
Download bottom up features from https://imagecaption.blob.core.windows.net/imagecaption/trainval.zip
Reads all the tsv files and saves all the object features, mean of the all the
object features in an image, box coordinates in separate npz/npy files.
"""

import os
import glob
import base64
import numpy as np
import csv
import json
from collections import defaultdict
import sys
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10 as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = maxInt//10
import argparse



FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
# infiles = ['trainval/karpathy_test_resnet101_faster_rcnn_genome.tsv',
#            'trainval/karpathy_val_resnet101_faster_rcnn_genome.tsv',
#            'trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.0',
#            'trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.1']


def make_bu_data(input_json: str,
                 source: str,
                 dest: str):

    os.makedirs(dest + '_att', exist_ok = True)
    os.makedirs(dest + '_fc', exist_ok = True)
    os.makedirs(dest + '_box', exist_ok = True)
    
    with open(input_json, "r", encoding = 'utf-8') as f:
        images = json.load(f)['images']
    
    remaining_files = defaultdict(list)
    for img in images:
        image_id = str(img['cocoid'])
        if all(os.path.exists(os.path.join(dest + suffix, image_id + ext)) for suffix, ext in [('_att', '.npz'),
                                                                                               ('_fc', '.npy'),
                                                                                               ('_box', '.npy')]):
            continue
        remaining_files[img['split']].append(image_id)

    for key in remaining_files:
        for infile in glob.glob(os.path.join(source, f'*{key}*.tsv*')):
            print('Reading ' + infile)
            with open(infile, "r", encoding = 'utf-8') as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
                for item in reader:
                    image_id = str(item['image_id'])
                    if image_id not in remaining_files[key]:
                        continue
                    item['num_boxes'] = int(item['num_boxes'])
                    for field in ['boxes', 'features']:
                        item[field] = np.frombuffer(base64.decodebytes(item[field].encode('utf-8')),
                                                    dtype = np.float32).reshape((item['num_boxes'],-1))
                    np.savez_compressed(os.path.join(dest + '_att', image_id + '.npz'), feat = item['features'])
                    np.save(os.path.join(dest + '_fc', image_id + '.npy'), item['features'].mean(0))
                    np.save(os.path.join(dest + '_box', image_id + '.npy'), item['boxes'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # output_dir
    parser.add_argument('--source', default = 'data/bu_data', help = 'downloaded feature directory')
    parser.add_argument('--dest', default = 'data/cocobu', help = 'output feature files')

    args = parser.parse_args()

    make_bu_data(args.source, args.dest)