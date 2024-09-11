"""
Functions to compute term frequency and document frequency of all
groud truth captions.
"""

from typing import List, Dict, Tuple
import os
import json
import argparse
from collections import defaultdict, Counter

import misc.utils as utils


def precook(sentence: str, n = 4) -> Dict[Tuple[str,...], int]:
    """
    Computes n-gram counts of the given sentence for all 1 <= n-grams <= n.

    Args:
        sentence (str): Sentence to be converted into ngrams
        n (int, optional): Largest ngrams length for which representation is
        calculated. Defaults to 4.

    Returns:
        Dict[Tuple[str,...], int]: Term frequency vector for occuring ngrams
    """
    words = sentence.split()
    counts: Dict[Tuple[str, ...], int] = Counter()
    for k in range(1, n+1): # for each n-gram <= n
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts


def cook_refs(refs: List[str], n: int = 4) -> List[Dict[Tuple[str, ...], int]]:
    """
    Computes n-gram counts of all the reference sentences of an image.

    Args:
        refs (List[str]): All captions of an image
        n (int, optional): Largest ngrams length for which representation is calculated. Defaults to 4.

    Returns:
        List[Dict[Tuple[str, ...], int]]: Term frequency vector for occuring ngrams of each caption
    """
    return [precook(ref, n) for ref in refs]


def create_crefs(refs: List[List[str]]) -> List[List[Dict[Tuple[str, ...], int]]]:
    """
    Computes Term Frequency (TF) or n-gram counts of all the reference sentences
    in the dataset.

    Args:
        refs (List[List[str]]): Captions corresponding to every image in the dataset.

    Returns:
        List[List[Dict[Tuple[str, ...], int]]]: Term frequency vector for all ngrams of each caption.
    """
    crefs = [cook_refs(ref) for ref in refs]
    return crefs


def compute_doc_freq(crefs: List[List[Dict[Tuple[str, ...], int]]]) -> Dict[Tuple[str, ...], float]:
    """
    Computes document frequency of each n-gram.

    Args:
        crefs (List[List[Dict[Tuple[str, ...], int]]]): Frequency of each n-gram of each caption belong
        to all images in the dataset.

    Returns:
        Dict[Tuple[str, ...], float]: Document frequency of all the n-grams.
    """
    document_frequency: Dict[Tuple[str, ...], float] = defaultdict(float)
    for refs in crefs:
        # Find unique n-grams from all ref sentences corresponding to a image
        unique_ngrams = set([ngram for ref in refs for ngram in ref])
        # Update document frequency
        for ngram in unique_ngrams:
            document_frequency[ngram] += 1
    return document_frequency


def build_dict(imgs: List[Dict], wtoi: Dict[str, int], split: str):
    """
    Computes document frequency for all the reference captions in the given
    split.

    Parameters
    ----------
    imgs   : list
             Each element is a dict
    wtoi   : dict
             Word to index mapping (1-index)
    params : dict
             Parameters

    Returns
    -------
    ngram_words : dict
                  Document frequency of all the n-gram words.
    ngram_idxs  : dict
                  Document frequency of all the n-gram indices.
    count_imgs  : int
                  No. of images in the given split.
    """
    wtoi['<EOS>'] = 0
    count_imgs = 0

    refs_words = []
    refs_idxs = []
    for img in imgs:
        if (split == img['split']) or (split == 'train' and img['split'] == 'restval') or (split == 'all'):
            ref_words = []
            ref_idxs = []
            for sent in img['sentences']:
                tmp_tokens = sent['tokens'] + ['<EOS>']
                tmp_tokens = [token if token in wtoi else '<UNK>' for token in tmp_tokens]
                ref_words.append(' '.join(tmp_tokens))
                ref_idxs.append(' '.join([str(wtoi[token]) for token in tmp_tokens]))
            refs_words.append(ref_words)
            refs_idxs.append(ref_idxs)
            count_imgs += 1
    print('Total images : ', count_imgs)

    ngram_words = compute_doc_freq(create_crefs(refs_words))
    ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))
    
    return ngram_words, ngram_idxs, count_imgs


def preprocess_document_frequency(dict_json: str, input_json: str, output_pkl: str, split: str):
    """
    Compute and save document frequency of all n-gram in the reference
    captions in the given split.

    Args:
        dict_json (str): Path to processed json file
        input_json (str): Path to Karpathy json file
        output_pkl (str): Path where document frequency data is to be saved
        split (str): Dataset split
    """
    if all(os.path.exists(output_pkl + name) for name in ['-words.p', '-idxs.p']):
        return
    
    with open(dict_json, 'r', encoding = 'utf-8') as f:
        itow = json.load(f)['ix_to_word']
    
    wtoi = {w:i for i,w in itow.items()}

    with open(input_json, 'r', encoding = 'utf-8') as f:
        imgs = json.load(f)['images']

    ngram_words, ngram_idxs, ref_len = build_dict(imgs, wtoi, split)

    utils.save({'document_frequency': ngram_words, 'ref_len': ref_len},
               output_pkl + '-words.p')
    utils.save({'document_frequency': ngram_idxs, 'ref_len': ref_len},
               output_pkl + '-idxs.p')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type = str, default = 'data/dataset_coco.json',
                        help = 'Path to Karpathy json file')
    parser.add_argument('--dict_json', type = str, default = 'data/cocotalk.json',
                        help = 'Path to processed json file')
    parser.add_argument('--output_pkl', type = str, default = 'data/coco-all',
                        help = 'Path where document frequency data is to be saved')
    parser.add_argument('--split', type = str, default = 'all', choices = ['all', 'train', 'test', 'val'],
                        help = 'Dataset split to be used')
    
    args = parser.parse_args()

    preprocess_document_frequency(**vars(args))
