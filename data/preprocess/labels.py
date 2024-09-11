"""
Preprocess raw json dataset into a json file with vocabulary and processed captions.
"""

from typing import List, Dict
import os
import json
import argparse
from random import seed
from collections import Counter
import imagesize
from tabulate import tabulate
import matplotlib.pyplot as plt

def build_vocab(imgs: List[Dict],
                min_word_frequency: int,
                save_path: str) -> List[str]:
    """
    Builds vocabulary of words having frequency greater than the
    threshold provided. Also, adds 'final_captions' key in imgs replacing
    rare words (frequency <= min_word_frequency) with '<UNK>' token.

    Args:
        imgs (List[Dict]): cocoid    :
              imgid     :
              filepath  :
              sentids   :
              split     :
              filename  :
              sentences : list
                          Each item is a dict with the following keys
                          tokens :
                          raw    :
                          imgid  :
        min_word_frequency (int): Threshold to consider words as rare

    Returns:
        List[str]: Vocabulary of words
    """
    # find the count of each word
    counts: Dict[str, int] = Counter()
    for img in imgs:
        for sent in img['sentences']:
            for w in sent['tokens']:
                counts[w] += 1

    w, c = zip(*counts.most_common(20))
    plt.figure()
    plt.bar(range(len(c)), c)
    plt.xticks(range(len(c)), w, rotation = 90)
    plt.title('Distribution of 20 most common words')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'word_distribution.jpg'))

    total_words = counts.total()
    num_bad_words = 0
    bad_count = 0
    vocab = []
    for word, count in counts.items():
        if count > min_word_frequency:
            vocab.append(word)
        else:
            num_bad_words += 1
            bad_count += count

    info = [['Words', total_words, '-'],
            ['Unique words', len(vocab), '-'],
            ['Unique Bad words', num_bad_words, round(num_bad_words*100/len(counts),1)],
            ['<UNK>', bad_count, round(bad_count*100.0/total_words,1)]]
    print(tabulate(info, headers = ['Description', 'Count', '%'], tablefmt = 'orgtbl'))
    
    # lets look at the distribution of lengths as well
    sent_lengths: Dict[int, int] = Counter()
    for img in imgs:
        for sent in img['sentences']:
            txt = sent['tokens']
            nw = len(txt)
            sent_lengths[nw] += 1
            
    plt.figure()
    plt.bar(list(sent_lengths.keys()), list(sent_lengths.values()))
    plt.title('Distribution of sentence length')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'sentence_length_distribution.jpg'))

    # Add '<UNK>' token if there any bad words
    if bad_count > 0:
        vocab.append('<UNK>')

    # Create key 'final_captions' having tokens with words below minimum count
    # threshold replaced with '<UNK>'
    for img in imgs:
        img['final_captions'] = []
        for sent in img['sentences']:
            txt = sent['tokens']
            caption = [w if counts[w] > min_word_frequency else '<UNK>' for w in txt]
            img['final_captions'].append(caption)

    return vocab


def encode_captions(imgs: List[Dict],
                    max_caption_length: int,
                    wtoi: Dict[str, int]):
    """
    Adds a key 'tokenized_captions' to imgs which has captions encoded
    based on given word to index.

    Args:
        imgs (List[Dict]): _description_
        max_caption_length (int): Maximum permissible length of caption.
        wtoi (Dict[str, int]): Mapping from word to index.
    """
    for img in imgs:
        img['tokenized_captions'] = []
        for caption in img['final_captions']:
            img['tokenized_captions'].append([wtoi[word] for word in caption[:max_caption_length]])

# def encode_captions(imgs: List[Dict],
#                     max_caption_length: int,
#                     wtoi: Dict[str, int]):
#     """
#     Encodes all captions into a 1-indexed large array. Also produces
#     label_start_ix and label_end_ix which store 1-indexed and inclusive
#     (Lua-style) pointers to the first and last caption for each image in the
#     dataset.

#     Parameters
#     ----------
#     wtoi           : dict
#                      Word to index (1 indexed, 0 - padding)
#     imgs           : list
#                      Each item is a dict with multiple keys. 'final_captions' is
#                      used here. 'final_captions' is a list and each element is a
#                      list having tokens below minimum count replaced with <UNK>.
#     params         : dict
#                      Parameters

#     Returns
#     -------
#     L              : numpy.array
#                      1-indexed encoded captions.
#     label_start_ix : numpy.array
#                      Stores index of the first caption of each image.
#     label_end_ix   : numpy.array
#                      Stores index of the last caption of each image.
#     label_length   : numpy.array
#                      Length of each caption capped at params['max_length'].
#     """
#     # Find total no. of images and total no. of captions
#     N = len(imgs)
#     M = sum(len(img['final_captions']) for img in imgs) # total number of captions

#     label_arrays = []
#     label_start_ix = np.zeros(N, dtype = 'uint32') # note: these will be one-indexed
#     label_end_ix = np.zeros(N, dtype = 'uint32')
#     label_length = np.zeros(M, dtype = 'uint32')
#     caption_counter = 0
#     counter = 1
#     for i,img in enumerate(imgs):
#         n = len(img['final_captions'])
#         assert n > 0, 'error: some image has no captions'
#         Li = np.zeros((n, max_caption_length), dtype='uint32')
#         for j,s in enumerate(img['final_captions']):
#             # Record the length of this sequence capped at max_length
#             label_length[caption_counter] = min(max_caption_length, len(s))
#             caption_counter += 1
#             for k,w in enumerate(s[:max_caption_length]): # encode words only upto max_caption_length
#                 Li[j,k] = wtoi[w]

#         # note: word indices are 1-indexed, and captions are padded with zeros
#         label_arrays.append(Li)
#         label_start_ix[i] = counter
#         label_end_ix[i] = counter + n - 1

#         counter += n

#     L = np.concatenate(label_arrays, axis=0) # put all the labels together
#     assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
#     assert np.all(label_length > 0), 'error: some caption had no words?'

#     print('Encoded captions to array of size ', L.shape)
#     return L, label_start_ix, label_end_ix, label_length

def create_output_json(image_info, image_root, itow, max_caption_length, min_word_frequency):
    # create output json file
    out = {'ix_to_word': itow,
           'max_caption_length': max_caption_length,
           'min_word_frequency': min_word_frequency}
    # out['ix_to_word'] = itow # encode the (1-indexed) vocab
    out['images'] = []

    for img in image_info:
        jimg = {}
        jimg['split'] = img['split']
        jimg['tokens'] = img['tokenized_captions']
        if 'filename' in img:
            # Keep full path to the image
            if image_root != '':
                jimg['file_path'] = os.path.join(image_root, img['filepath'], img['filename'])
            else:
                jimg['file_path'] = os.path.join(img['filepath'], img['filename'])
        if 'cocoid' in img:
            # Keep cocoid, useful
            jimg['id'] = img['cocoid']

        if image_root != '':
            jimg['width'], jimg['height'] = imagesize.get(os.path.join(image_root, img['filepath'], img['filename']))

        out['images'].append(jimg)
    
    return out


def preprocess_labels(input_json: str,
                      output_json: str,
                    #   output_h5: str,
                      image_root: str,
                      max_caption_length: int,
                      min_word_frequency: int):
    """
    Builds vocabulary and tokenizes the captions based on the built vocabulary.

    Args:
        input_json (str): Path to Karpathy json file.
        output_json (str): Path to processed json file.
        image_root (str): Directory containing images.
        max_caption_length (int): Maximum permissible length of caption.
        min_word_frequency (int): Threshold to consider words as rare
    """
    if os.path.exists(output_json):
        with open(output_json, 'r', encoding = 'utf-8') as f:
            out = json.load(f)
        if out['max_caption_length'] == max_caption_length and out['min_word_frequency'] == min_word_frequency:
            return
        
    with open(input_json, 'r', encoding = 'utf-8') as f:
        imgs = json.load(f)['images']
    
    # make reproducible
    seed(123)
    # create the vocab
    vocab = build_vocab(imgs, min_word_frequency, os.path.split(output_json)[0])
    itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
    wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

    # encode captions in large arrays, ready to ship to hdf5 file
    encode_captions(imgs, max_caption_length, wtoi)
    # L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, max_caption_length, wtoi)

    # # create output h5 file
    # N = len(imgs)
    # f_lb = h5py.File(output_h5 +'_label.h5', "w")
    # f_lb.create_dataset("labels", dtype='uint32', data=L)
    # f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    # f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    # f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
    # f_lb.close()

    out = create_output_json(imgs, image_root, itow, max_caption_length, min_word_frequency)

    with open(output_json, 'w', encoding = 'utf-8') as f:
        json.dump(out, f)

    print(f'Wrote {output_json}')
    
    return itow


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type = str, default = 'data/dataset_coco.json',
                        help = 'Path to Karpathy json file')
    parser.add_argument('--output_json', type = str, default = 'data/cocotalk.json',
                        help = 'Path where processed json file is to be saved')
    # parser.add_argument('--output_h5', default='data', help='output h5 file')
    parser.add_argument('--image_root', type = str, default = 'data/images',
                        help = 'Directory containing images')
    parser.add_argument('--max_caption_length', type = int, default = 16,
                        help = 'Maximum permissible length of captions (longer captions will be clipped)')
    parser.add_argument('--min_word_frequency', type = int, default = 5,
                        help = 'Max frequency of a word to be considered rare')

    args = parser.parse_args()
    
    preprocess_labels(**vars(args))
