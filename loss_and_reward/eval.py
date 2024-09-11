from typing import List, Dict
import sys
sys.path.append('coco-caption')
import os
import json

from pycocotools.coco import COCO
from misc.correct_coco_eval_cap import CorrectCOCOEvalCap
from misc.report import ReportData

REPORT_DATA_PKL_FILE_TEMPLATE = '%s_report_data.pkl'

def language_eval(preds: List[Dict],
                  image_root,
                  results_dir: str,
                  split):
    
    annFile = 'coco-caption/annotations/captions_val2014.json'
    
    cache_path = os.path.join(results_dir, split + '.json')
    
    coco = COCO(annFile)
    valids = coco.getImgIds()
    
    preds_filt = [p for p in preds if p['image_id'] in valids]
    with open(cache_path, 'w', encoding = 'utf-8') as f:
        json.dump(preds_filt, f)
    
    cocoRes = coco.loadRes(cache_path)
    cocoEval = CorrectCOCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    
    if image_root:
        report_data = ReportData(cocoEval, preds, image_root, split)
        pickle_file_name = REPORT_DATA_PKL_FILE_TEMPLATE % (split)
        pickle_path = os.path.join(results_dir, pickle_file_name)
        report_data.save_to_pickle(pickle_path)
    
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score
    
    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        imgToEval[p['image_id']]['caption'] = p['caption']
    
    with open(cache_path, 'w', encoding = 'utf-8') as f:
        json.dump({'overall': out, 'imgToEval': imgToEval}, f)
    
    return out
        