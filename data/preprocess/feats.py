from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
from six.moves import cPickle
import numpy as np
import torch
import torchvision.models as models
import skimage.io

from torchvision import transforms
from misc.resnet_utils import myResnet
import misc.resnet as resnet

preprocess = transforms.Compose([#trn.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])



def compute_features(params):
    # Load the resnet architecture corresponding to params['model']
    net = getattr(resnet, params['model'])()
    # Load corresponding state dictionary
    net.load_state_dict(torch.load(os.path.join(params['model_root'],params['model']+'.pth')))
    # Load resnet class with forward function returning fc and att features
    my_resnet = myResnet(net)
    my_resnet.cuda()
    my_resnet.eval()
    # Load dataset json file
    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']
    N = len(imgs)

    seed(123) # make reproducible

    dir_fc = params['output_dir']+'_fc'
    dir_att = params['output_dir']+'_att'
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    if not os.path.isdir(dir_att):
        os.mkdir(dir_att)

    for i,img in enumerate(imgs):
        # load the image
        I = skimage.io.imread(os.path.join(params['images_root'], img['filepath'], img['filename']))
        # handle grayscale input images
        if len(I.shape) == 2:
            I = I[:,:,np.newaxis]
            I = np.concatenate((I,I,I), axis=2)
        # Scale pixels to 0 - 1
        I = I.astype('float32')/255.0
        # Change image from (H,W,C) to (C,H,W)
        I = torch.from_numpy(I.transpose([2,0,1])).cuda()
        # Apply transformation defined at the top
        I = preprocess(I)
        # Forward pass with the given att_size
        with torch.no_grad():
            tmp_fc, tmp_att = my_resnet(I, params['att_size'])
        # write to pkl
        np.save(os.path.join(dir_fc, str(img['cocoid'])), tmp_fc.data.cpu().float().numpy())
        np.savez_compressed(os.path.join(dir_att, str(img['cocoid'])), feat=tmp_att.data.cpu().float().numpy())
        if i % 1000 == 0:
            print('Processing {:d}/{:d} ({:.2f}% done)'.format(i, N, i*100.0/N))
    print('Wrote {}'.format(params['output_dir']))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--output_dir', default='data', help='output h5 file')

    # options
    parser.add_argument('--images_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--att_size', default=14, type=int, help='14x14 or 7x7')
    parser.add_argument('--model', default='resnet101', type=str, help='resnet101, resnet152')
    parser.add_argument('--model_root', default='./data/imagenet_weights', type=str, help='model root')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    main(params)
