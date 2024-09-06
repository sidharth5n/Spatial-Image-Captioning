import json
import argparse
from collections import defaultdict

def precook(s, n=4):
    """
    Computes n-gram counts of the given sentence for all 1 <= n-grams <= n.

    Parameters
    ----------
    s      : string
             Sentence to be converted into ngrams
    n      : int
             Largest ngrams length for which representation is calculated

    Returns
    -------
    counts : defaultdict
             Term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n+1): # for each n-gram <= n
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    """
    Computes n-gram counts of all the reference sentences of an image.

    Parameters
    ----------
    refs   : list
             Each element is a string having reference sentences for some image.
    n      : int
             Number of ngrams for which (ngram) representation is calculated.

    Returns
    -------
    result : list
             Each element is dict containing the n-gram counts of a reference
             sentence.
    """
    return [precook(ref, n) for ref in refs]

def create_crefs(refs):
    """
    Computes Term Frequency (TF) or n-gram counts of all the reference sentences
    in the dataset.

    Parameters
    ----------
    refs  : list
            Each item is a list of string having reference sentences or indexes
            with <UNK> token.

    Returns
    -------
    crefs : list
            Each element is a list containing dict of n-gram counts of the
            reference sentences i.e. Term Frequency (TF).
    """
    crefs = []
    for ref in refs:
        # ref is a list of all captions of an image
        crefs.append(cook_refs(ref))
    return crefs

def compute_doc_freq(crefs):
    """
    Compute document frequency for reference data. This will be used to compute
    Inverse Document Frequency (IDF).

    Parameters
    ----------
    crefs              : list
                         Each element is a list containing dict of n-gram counts
                         of the reference sentences i.e. Term Frequency (TF).
    Returns
    -------
    document_frequency : dict
                         Document frequency of all the n-grams.
    """
    document_frequency = defaultdict(float)
    for refs in crefs:
        # Find unique n-grams from all ref sentences corresponding to a image
        unique_ngrams = set([ngram for ref in refs for (ngram,count) in ref.iteritems()])
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
    # WHY IS <EOS> BEING GIVEN THE SAME INDEX AS <PAD>?
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
                tmp_tokens = [_ if _ in wtoi else '<UNK>' for _ in tmp_tokens]
                ref_words.append(' '.join(tmp_tokens))
                ref_idxs.append(' '.join([str(wtoi[_]) for _ in tmp_tokens]))
            refs_words.append(ref_words)
            refs_idxs.append(ref_idxs)
            count_imgs += 1
    print('Total images : ', count_imgs)

    ngram_words = compute_doc_freq(create_crefs(refs_words))
    ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))
    return ngram_words, ngram_idxs, count_imgs

def main(dict_json: str, input_json: str, output_pkl: str, split: str):
    """
    Computes and saves document frequency of all the n-grams in the reference
    captions in the given split.

    Parameters
    ----------
    params : dict
             Parameters
    """
    with open(dict_json, 'r', 'utf-8') as f:
        itow = json.load(f)['ix_to_word']
    
    wtoi = {w:i for i,w in itow.items()}

    with open(input_json, 'r', 'utf-8') as f:
        imgs = json.load(f)['images']

    ngram_words, ngram_idxs, ref_len = build_dict(imgs, wtoi, split)

    save({'document_frequency': ngram_words, 'ref_len': ref_len}, output_pkl +'-words.p')
    save({'document_frequency': ngram_idxs, 'ref_len': ref_len}, output_pkl +'-idxs.p')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default = 'data/dataset_coco.json', help='input json file to process into hdf5')
    parser.add_argument('--dict_json', default = 'data/cocotalk.json', help='output json file')
    parser.add_argument('--output_pkl', default = 'data/coco-all', help='output pickle file')
    parser.add_argument('--split', default='all', help='test, val, train, all')
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict

    main(params)
