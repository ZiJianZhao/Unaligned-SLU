# -*- coding: utf-8 -*-

import argparse
import os, sys
import codecs
from collections import Counter
import json
import numpy as np
import torch
import copy

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(install_path)
sys.path.append(install_path)

import xslu.Constants as Constants
from xslu.utils import process_sent, process_word

root_dir = '../../dstc2-slu/'
glove_path = '../../../NLPData/glove.6B/glove.6B.100d.txt'

def build_class_vocab(filename):
    with codecs.open(filename, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    class2idx = {}
    for line in lines: 
        class2idx[line] = len(class2idx)
    print('Class vocab size: {}'.format(len(class2idx)))
    return class2idx

class Memory4BaseSTC(object):

    """Class used to process DSTC2 texts and save the processed contents.
    
    Attention:
        - only support manual & 1best types.
        - used in semantic tuple classifier.
    
    Args:
        - Need settings:
            * train_file: containing liness <utterance, act-slot-value triples>.
            * class_file: file containing act-slot-value triples.
        - Manually setted in this file:
            * root_dir: root dir for the data files and to-save files.
            * ontology_path: DSTC2 ontology path.
            * glove_path: pretrained word embedding file .

    Returns:
        - word2idx: mapping of the unfilterred words in train dataset.
        - word2idx_w_glove: glove enhanced word2idx.
        - class2idx: mapping of the classes in class file.
    """

    def __init__(self):
        super(Memory4BaseSTC, self).__init__()

    def load_memory(self, memory_file):
        memory_path = os.path.join(root_dir, memory_file)
        memory = torch.load(memory_path)
        return memory

    def build_save_memory(self, train_file, class_file, memory_file):
        train_path = os.path.join(root_dir, train_file)
        class_path = os.path.join(root_dir, class_file)
        memory_path = os.path.join(root_dir, memory_file)

        memory = {}
        word2idx = self.build_word_vocab(train_path)
        memory['word2idx'] = word2idx
        memory['word2idx_w_glove'] = self.expand_word_vocab_with_glove(word2idx)
        memory['class2idx'] = self.build_class_vocab(class_path)

        torch.save(memory, memory_path)
        print('Memory saved in {}'.format(memory_path))

    def glove_vocab(self, filename=glove_path):
        with open(filename, 'r') as f:
            lines = f.readlines()
            words = [line.strip().split(' ')[0] for line in lines]
        return words

    def build_word_vocab(self, filename, frequency=2):
        words = []
        with codecs.open(filename, 'r') as f:
            lines = f.readlines()
            sents = [line.split('\t<=>\t')[0].strip() for line in lines]
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

        for (word, count) in lis:
            if count >= frequency:
                if word not in word2idx:
                    word2idx[word] = len(word2idx)
        print('Train voacb size: {}'.format(len(word2idx)))
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

    def build_class_vocab(self, filename):
        with codecs.open(filename, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        class2idx = {}
        for line in lines: 
            class2idx[line] = len(class2idx)
        print('Class vocab size: {}'.format(len(class2idx)))
        return class2idx

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
        word2idx = self.build_word_vocab(train_path)
        memory['word2idx'] = word2idx
        memory['word2idx_w_glove'] = self.expand_word_vocab_with_glove(word2idx)

        single_acts, double_acts, triple_acts = self.class_info()
        memory['single_acts'] = single_acts
        memory['double_acts'] = double_acts
        memory['triple_acts'] = triple_acts

        act2idx, slot2idx, value2idx = self.build_class_vocab(class_path)
        memory['act2idx'] = act2idx
        memory['idx2act'] = {v:k for k,v in act2idx.items()}
        memory['slot2idx'] = slot2idx
        memory['idx2slot'] = {v:k for k,v in slot2idx.items()}

        # -----------------------------------------------------
        # use word2idx performs better than value2idx
        #memory['value2idx'] = value2idx
        #memory['idx2value'] = {v:k for k,v in value2idx.items()}

        memory['value2idx'] = word2idx
        memory['idx2value'] = {v:k for k,v in word2idx.items()}
        # -----------------------------------------------------

        act_emb, slot_emb = self.build_class_embed(act2idx, slot2idx)
        memory['act_emb'] = act_emb
        memory['slot_emb'] = slot_emb
        
        torch.save(memory, memory_path)
        print('Memory saved in {}'.format(memory_path))
        
    def class_info(self):
        single_acts = ['ack', 'affirm', 'bye', 'hello', 'negate', 'repeat', 
            'reqalts', 'reqmore', 'restart', 'thankyou']
        double_acts = ['request']
        triple_acts = ['inform', 'confirm', 'deny']
        return single_acts, double_acts, triple_acts

    def glove_vocab(self, filename=glove_path):
        with open(filename, 'r') as f:
            lines = f.readlines()
            words = [line.strip().split(' ')[0] for line in lines]
        return words

    def build_word_vocab(self, filename, frequency=2):
        words = []
        with codecs.open(filename, 'r') as f:
            lines = f.readlines()
            sents = [line.split('\t<=>\t')[0].strip() for line in lines]
        
        """
        for line in lines:
            classes = line.split('\t<=>\t')[1].strip().split(';')
            for cls in classes:
                lis = cls.strip().split('-')
                if len(lis) == 3:
                    sents.append(lis[2].strip())
        """

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

    def build_class_vocab(self, filename):
        with codecs.open(filename, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        acts = []
        slots = []
        values = []
        for line in lines:
            lis = line.split('-')
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

        def lis2dic(lis, tag):
            class2idx = {Constants.PAD_WORD: Constants.PAD}
            for line in lis:
                if line not in class2idx:
                    class2idx[line] = len(class2idx)
            print('{} class vocab size: {}'.format(tag, len(class2idx)))
            return class2idx

        print('==========================================')
        act2idx = lis2dic(acts, 'act')
        slot2idx = lis2dic(slots, 'slot')
        value2idx = lis2dic(values, 'value')

        return act2idx, slot2idx, value2idx

    def build_class_embed(self, act2idx, slot2idx, emb_file=glove_path):
        
        knowledge = ['pad', 'unk', 'price', 'range', 'address', 'require', 'more', 'alternatives']
        words = list(act2idx.keys()) + list(slot2idx.keys()) + knowledge
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
        
        def class2emb(dic):
            emb = torch.zeros(len(dic), 100)
            emb.uniform_(-0.1, 0.1)
            for word in dic:
                if word in word2emb:
                    emb[dic[word]] = torch.from_numpy(word2emb[word])
            return emb

        act_emb = class2emb(act2idx)
        slot_emb = class2emb(slot2idx)
        
        return act_emb, slot_emb

if __name__ == '__main__':
    dstc2_memory =  Memory4BaseHD()
    dir_name = 'manual/'
    dstc2_memory.build_save_memory(dir_name+'train', dir_name+'class.train', dir_name+'memory.pt')
