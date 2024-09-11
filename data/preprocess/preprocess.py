import os
import json

from preprocess.labels import build_vocab, encode_captions, create_output_json
from preprocess.ngrams import build_dict
import misc.utils as utils

class PreProcess:
    
    def __init__(self,
                 input_json: str,
                 image_root: str,
                 output_json: str
                 ):
        
        self.input_json = input_json
        self.image_root = image_root
        self.output_json = output_json
        
        with open(input_json, 'r', encoding = 'utf-8') as f:
            self.images = json.load(f)['images']
        
    def process_labels(self, max_caption_length, min_word_frequency):
        
        if os.path.exists(self.output_json):
            with open(self.output_json, 'r', encoding = 'utf-8') as f:
                out = json.load(f)
            if out['max_caption_length'] == max_caption_length and out['min_word_frequency'] == min_word_frequency:
                return
        
        vocab = build_vocab(self.images, min_word_frequency)
        itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
        wtoi = {w:i+1 for i,w in enumerate(vocab)}
    
        encode_captions(self.images, max_caption_length, wtoi)
        
        self.out = create_output_json(self.images, self.image_root, itow, max_caption_length, min_word_frequency)
    
    def process_document_freqency(self, split):
        
        if all(os.path.exists(self.output_pkl + name) for name in ['-words.p', '-idxs.p']):
            return
        
        itow = self.out['ix_to_word']
        wtoi = {w:i for i,w in itow.items()}
        
        ngram_words, ngram_idxs, ref_len = build_dict(self.images, wtoi, split)
        
        utils.save({'document_frequency': ngram_words, 'ref_len': ref_len},
                   self.output_pkl + '-words.p')
        utils.save({'document_frequency': ngram_idxs, 'ref_len': ref_len},
                   self.output_pkl + '-idxs.p')
    
    