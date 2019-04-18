# -*- coding: utf-8 -*-

import argparse
import os, sys
import codecs
from collections import Counter
import json
import numpy as np
import torch
import copy

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(install_path)
sys.path.append(install_path)

import xslu.Constants as Constants
from xslu.utils import process_sent, process_word

root_dir = '../../../dstc3-st/'
glove_path = '../../../../NLPData/glove.6B/glove.6B.100d.txt'


def process_class(class_string):
    classes = class_string.strip().split(';')
    results = []
    for cls in classes:
        results.append(' '.join(cls.strip().split('-', 2)))
    string = ' ; '.join(results)
    #print(class_string, string)
    #print(string.split())
    #input()
    return string

class Memory4BaseHD(object):

    """Class used to process DSTC2 texts and save the processed contents.

    Args:
        - Need settings:
            * train_file: containing liness <utterance, act-slot-pair triples>.
        - Manually setted in this file
            * root_dir: root dir for the data files and to-save files.
            * ontology_path: DSTC2 ontology path
            * glove_path: pretrained word embedding file
    Returns:
        - word2idx: mapping of the unfilterred words in train dataset;
        - word2idx_w_glove: glove enhanced word2idx;
        - act2idx: mapping of acts to ids;
        - slot2idx: mapping of slot types to ids;
        - act_slot2idx: mapping of act-slot pairs to ids; (slot can be empty)
        - act_slot_value2idx: mapping of act-slot-value triples to ids;
        - act2emb: mapping of act to embeedding;
        - slot2emb: mapping of slot to embedding;
        - act_slot2emb: mapping of act-slot pairs to embedding; (slot can be empty)
        - act_slot_value2emb: mapping of act-slot-value triples to embedding.
    """

    def __init__(self):
        super(Memory4BaseHD, self).__init__()

    def load_memory(self, memory_file):
        memory_path = os.path.join(root_dir, memory_file)
        memory = torch.load(memory_path)
        return memory

    def build_save_memory(self, train_file, class_file, memory_file):
        train_path = os.path.join(root_dir, train_file)
        class_path = os.path.join(root_dir, class_file)
        memory_path = os.path.join(root_dir, memory_file)

        memory = {}
        word2idx = self.build_word_vocab(train_path, class_path)
        memory['word2idx'] = word2idx
        memory['word2idx_w_glove'] = self.expand_word_vocab_with_glove(word2idx)

        # -----------------------------------------------------

        memory['word2idx_emb'] = self.build_word_embed(memory['word2idx'])
        memory['word2idx_w_glove_emb'] = self.build_word_embed(memory['word2idx_w_glove'])

        torch.save(memory, memory_path)
        print('Memory saved in {}'.format(memory_path))

    def glove_vocab(self, filename=glove_path):
        with open(filename, 'r') as f:
            lines = f.readlines()
            words = [line.strip().split(' ')[0] for line in lines]
        return words

    def get_class_vocab(self, filename):
        with codecs.open(filename, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        acts = []
        slots = []
        values = []
        for line in lines:
            lis = line.split('-', 2)
            if len(lis) >= 1:
                acts.append(lis[0])
                if len(lis) >= 2:
                    slots.append(lis[1])
                    if len(lis) == 3:
                        words = lis[2].strip().split()
                        values.extend(words)

        acts = sorted(list(set(acts)))
        slots = sorted(list(set(slots)))
        values = sorted(list(set(values)))

        vocab = sorted(list(set(acts+slots+values)))

        return vocab

    def build_word_vocab(self, filename, class_file, frequency=1):
        words = []
        sents = []
        with codecs.open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            sents.append(line.split('\t<=>\t')[0])
            sents.append(process_class(line.split('\t<=>\t')[1]))

        for sent in sents:
            ws = process_sent(sent)
            words.extend(ws)

        counter = Counter(words)
        lis = counter.most_common()
        print('Total words num: {}'.format(len(lis)))
        num = 0
        for (word, count) in lis:
            if count < frequency:
                break
            num += 1
        print('Words num with frequency >= {}: {}'.format(frequency, num))

        word2idx = {
            Constants.PAD_WORD: Constants.PAD,
            Constants.UNK_WORD: Constants.UNK,
            Constants.BOS_WORD: Constants.BOS,
            Constants.EOS_WORD: Constants.EOS
        }

        # ===========================================
        word2idx['dontcare'] = len(word2idx)
        # ===========================================

        class_vocab = self.get_class_vocab(class_file)

        for word in class_vocab:
            if word not in word2idx:
                word2idx[word] = len(word2idx)

        for (word, count) in lis:
            if count >= frequency:
                if word not in word2idx:
                    word2idx[word] = len(word2idx)
        print('Final voacb size: {}'.format(len(word2idx)))
        print('==========================================')
        return word2idx

    def expand_word_vocab_with_glove(self, word2idx):
        word2idx_w_glove = copy.deepcopy(word2idx)
        words = self.glove_vocab()
        for word in words:
            if word not in word2idx_w_glove:
                word2idx_w_glove[word] = len(word2idx_w_glove)
        print('Final vocab size with glove: {}'.format(len(word2idx_w_glove)))
        print('==========================================')
        return word2idx_w_glove

    def build_word_embed(self, word2idx, emb_file=glove_path):

        knowledge = ['pad', 'unk', 'price', 'range', 'address', 'has', 'tv', 'internet',
                'require', 'more', 'alternatives', 'children', 'allowed']
        words = list(word2idx.keys()) + knowledge
        words = list(set(words))

        word2emb = {word: np.zeros(100) for word in words}
        with open(emb_file, 'r') as f:
            emb_dim = 100
            for line in f:
                items = line.strip().split(' ')
                word = items[0]
                vector = np.array([float(value) for value in items[1:]])
                if word in word2emb:
                    word2emb[word] = vector
        word2emb['pricerange'] = word2emb['price'] + word2emb['range']
        word2emb['addr'] = word2emb['address']
        word2emb['reqmore'] = word2emb['require'] + word2emb['more']
        word2emb['reqalts'] = word2emb['require'] + word2emb['alternatives']
        word2emb['childrenallowed'] = word2emb['children'] + word2emb['allowed']
        word2emb['hastv'] = word2emb['has'] + word2emb['tv']
        word2emb['hasinternet'] = word2emb['has'] + word2emb['internet']

        unknowns = []
        for word in word2emb:
            if word2emb[word].sum() == 0:
                word2emb[word] = word2emb['unk']
                unknowns.append(word)
        if len(unknowns) == 0:
            print('All word of classes exist in glove.')
        else:
            print('Exist {} word of classes not in glove.'.format(len(unknowns)))
            print('******************************************')
            print(unknowns)
            print('******************************************')

        emb = torch.zeros(len(word2idx), 100)
        emb.uniform_(-0.1, 0.1)
        for word in word2idx:
            if word in word2emb:
                emb[word2idx[word]] = torch.from_numpy(word2emb[word])

        return emb

def gen_class_file(class_all_file, class_train_file, class_save_file):
    with open(class_all_file, 'r') as f:
        class_all = f.readlines()
    with open(class_train_file, 'r') as f:
        class_train = f.readlines()
    class_save = list(set(class_all)-set(class_train))
    with open(class_save_file, 'w') as f:
        for cls in class_save:
            f.write('{}\n'.format(cls.strip()))


if __name__ == '__main__':
    dstc2_memory =  Memory4BaseHD()
    dir_name = 'manual-da/'
    dstc2_memory.build_save_memory(
        dir_name+'tmp/dstc2.3.all.train',
        dir_name+'dstc2.3.all.class',
        dir_name+'tmp/memory.pt'
    )
    #dir_name = root_dir + '1best-live-da/'
    #gen_class_file(dir_name+'class.all', dir_name + 'class.train', dir_name + 'class.gen')
