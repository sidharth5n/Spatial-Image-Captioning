from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch

from .TransformerModel import TransformerModel
from .ShowAndTellModel import ShowAndTell
from .ShowAttendAndTellModel import ShowAttendAndTell

def setup(opt, device):
    if opt.caption_model == 'transformer':
        model = TransformerModel(opt)
    elif opt.caption_model == 'show_tell':
        model = ShowAndTell(opt)
    elif opt.caption_model == 'show_attend_tell':
        model = ShowAttendAndTell(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))
    
    model = model.to(device)

    # Check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # Check if all necessary files exist
        assert os.path.isdir(opt.start_from), f'{opt.start_from} must be a a path'
        assert os.path.isfile(os.path.join(opt.start_from, "infos.pkl")), f'infos.pkl file does not exist in path {opt.start_from}'
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pt'), map_location = device))

    return model
