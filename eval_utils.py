from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import torch
import json

from misc import utils
from misc.report import ReportData

REPORT_DATA_PKL_FILE_TEMPLATE = '%s_%s_report_data.pkl'

def language_eval(dataset, preds, model_id, image_root, split):
    """
    Computes the language evaluation metrics for the given predictions.

    Parameters
    ----------
    dataset    : str
                 Dataset being used
    preds      : list
                 Each element is a dict containing information about the prediction.
    model_id   : str
                 ID for identifying the run
    image_root :
    split      : str
                 Dataset split

    Returns
    -------
    out        : dict
                 Results of the evaluation
    """
    import sys
    sys.path.append("coco-caption")
    annFile = 'coco-caption/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from misc.correct_coco_eval_cap import CorrectCOCOEvalCap

    results_dir = 'eval_results'
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    cache_path = os.path.join(results_dir, model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('Using {:d}/{:d} predictions'.format(len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = CorrectCOCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    if image_root:
        # Save cocoEval and any other relevant information into a pickle to be used
        # later for generating a report and visualizing results.
        report_data = ReportData(cocoEval, preds, image_root, model_id, split)
        pickle_file_name = REPORT_DATA_PKL_FILE_TEMPLATE % (model_id, split)
        pickle_path = os.path.join(results_dir, pickle_file_name)
        report_data.save_to_pickle(pickle_path)

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def eval_split(model, crit, loader, eval_kwargs = {}, device = torch.device("cpu")):
    """
    Evaluates the model based on the given parameters.

    Parameters
    ----------
    model       : Model
    crit        : loss function
    loader      : DataLoader
    eval_kwargs : dict
                  Parameters

    Returns
    -------
    mean_loss   : float
                  Mean loss
    predictions : list
                  Generated captions
    lang_stats  : dict or None
                  Evaluation results
    """
    verbose = eval_kwargs.get('verbose', False)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)

    # Make sure in the evaluation mode
    model.eval()

    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []

    for data in loader:

        batch_size = data['img_feats'].shape[0]

        if data.get('labels', None) is not None and verbose_loss:
            # forward the model to get loss
            tmp = [data['img_feats'], data['labels'], data['label_masks'], data['img_masks'], data['boxes']]
            tmp = [x.to(device) if x is not None else x for x in tmp]
            img_feats, labels, label_masks, img_masks, boxes = tmp

            with torch.no_grad():
                out  = model(img_feats, labels, img_masks, boxes)
                loss = crit(out, labels[:,1:], label_masks[:,1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['img_feats'], data['img_masks'], data['boxes']]
        indices = torch.arange(0, batch_size, loader.get_seq_per_img())
        tmp = [x[indices].to(device) if x is not None else x for x in tmp]
        img_feats, img_masks, boxes = tmp

        # Forward the model to also get generated samples for each image
        with torch.no_grad():
            seq = model(img_feats, img_masks, boxes, opt = eval_kwargs, mode = 'sample')[0]

        # Print beam search results for each image
        if beam_size > 1 and verbose_beam:
            for i in range(len(indices)):
                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)

        # Decode the generated indexed sequences
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            image_id = data['infos'][k]['id']
            entry = {'image_id'  : image_id,
                     'caption'   : sent,
                     'file_path' : data['infos'][k]['file_path']}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/' + str(image_id) + '.jpg' # still gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('Image {:d}: {:s}'.format(entry['image_id'], entry['caption']))

        if verbose:
            print('Evaluating validation preformance... {:d} samples ({:f})'.format(num_images, loss))

    mean_loss = loss_sum/loss_evals

    lang_stats = None
    if lang_eval == 1:
        # Peform language evaluation on the generated captions
        lang_stats = language_eval(dataset, predictions, eval_kwargs.get('id'),
                                   eval_kwargs.get('image_root'), split)

    # Switch back to training mode
    model.train()

    return mean_loss, predictions, lang_stats
