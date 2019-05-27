# -*- coding: utf-8 -*-

import argparse
import os, sys
import codecs
from collections import Counter, defaultdict
import json
import numpy as np
import torch
import copy

from dstc2 import process_sent, process_word
import Constants

root_dir = '/slfs1/users/zjz17/SLU/dstc3-st/'
ontology_path = root_dir + 'scripts/config/ontology_dstc3.json'
glove_path = '/slfs1/users/zjz17/NLPData/glove.6B/glove.6B.100d.txt'


class DSTC2Memory(object):

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
        super(DSTC2Memory, self).__init__()

    def load_memory(self, memory_file):
        memory_path = os.path.join(memory_file)
        memory = torch.load(memory_path)
        return memory

    def train_classes(self, filename):
        with codecs.open(filename, 'r') as f:
            lines = f.readlines()
        labels = [line.split('\t<=>\t')[1].strip() for line in lines]
        classes = []
        for label in labels:
            if label == '':
                continue
            classes.extend(label.split(';'))
        classes = list(set(classes))
        return classes

    def build_save_memory(self, train_file, class_file, memory_file):
        #train_path = os.path.join(root_dir, train_file)
        #class_path = os.path.join(root_dir, class_file)
        #memory_path = os.path.join(root_dir, memory_file)
        train_path = train_file
        class_path = class_file
        memory_path = memory_file

        memory = {}
        word2idx = self.build_word_vocab(train_path)
        memory['word2idx'] = word2idx
        memory['word2idx'] = self.expand_word_vocab_with_glove(word2idx)

        single_acts, double_acts, triple_acts = self.class_info()
        memory['single_acts'] = single_acts
        memory['double_acts'] = double_acts
        memory['triple_acts'] = triple_acts

        memory['train_classes'] = self.train_classes(train_path)

        act2idx, slot2idx, act_slot2idx, act_slot_value2idx = self.build_class_vocab(class_path)
        memory['act2idx'] = act2idx
        memory['idx2act'] = {v:k for k,v in act2idx.items()}
        memory['slot2idx'] = slot2idx
        memory['idx2slot'] = {v:k for k,v in slot2idx.items()}
        memory['act_slot2idx'] = act_slot2idx
        memory['class2idx'] = act_slot_value2idx
        memory['triple2idx'] = act_slot_value2idx
        memory['idx2class'] = {v:k for k,v in act_slot_value2idx.items()}
        memory['triples'] = [key for key in act_slot_value2idx]

        memory['act_triple_dic'] = defaultdict(list)
        for key  in memory['triples']:
            act = key.strip().split('-')[0]
            memory['act_triple_dic'][act].append(key)

        act2emb, slot2emb, act_slot2emb, act_slot_value2emb = self.build_class_embed(class_path,
            act2idx, slot2idx, act_slot2idx, act_slot_value2idx)

        memory['act2emb'] = act2emb
        memory['slot2emb'] = slot2emb
        memory['act_slot2emb'] = act_slot2emb
        memory['triple2emb'] = act_slot_value2emb

        memory['triple2idx'] = {}
        emb = torch.zeros(len(memory['triple2emb']), 300)
        idx = 0
        for key in memory['triple2emb']:
            memory['triple2idx'][key] = idx
            emb[idx] = torch.from_numpy(memory['triple2emb'][key]).float()
            idx += 1
        memory['idx2triple'] = {v:k for k,v in memory['triple2idx'].items()}
        memory['labelemb'] = emb

        torch.save(memory, memory_path)
        print('Memory saved in {}'.format(memory_path))

    def class_info(self):
        single_acts = ['ack', 'affirm', 'bye', 'hello', 'negate', 'repeat',
            'reqalts', 'reqmore', 'restart', 'thankyou']
        double_acts = ['request']
        triple_acts = ['inform', 'confirm', 'deny']


        #double_acts = double_acts + triple_acts


        return single_acts, double_acts, triple_acts

    def glove_vocab(self, filename=glove_path):
        with open(filename, 'r+', encoding='utf-8') as f:
            lines = f.readlines()
            words = [line.strip().split(' ')[0] for line in lines]
        return words

    def ontology_vocab(self):
        dic = json.loads(open(ontology_path).read())
        lis = []
        for key in dic:
            if key == 'requestable':
                for value in dic[key]:
                    lis.extend(value.split())
            elif key == 'informable':
                for slot in dic[key]:
                    lis.extend(slot.split())
                    for value in dic[key][slot]:
                        lis.extend(value.split())
            else:
                continue
        dic = {}
        for word in lis:
            dic[word.strip()] = 0
        return dic

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

        on_vocab = self.ontology_vocab()
        for (word, count) in lis:
            #if count >= frequency or word in on_vocab:
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
        act_slots = []
        for line in lines:
            lis = line.split('-')
            if len(lis) == 1:
                acts.append(lis[0])
                act_slots.append(lis[0])
            if len(lis) >= 2:
                acts.append(lis[0])
                slots.append(lis[1])
                act_slots.append('-'.join([lis[0], lis[1]]))
        acts = sorted(list(set(acts)))
        slots = sorted(list(set(slots)))
        act_slots = sorted(list(set(act_slots)))
        act_slot_values = sorted(lines)

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
        act_slot2idx = lis2dic(act_slots, 'act-slot')
        act_slot_value2idx = lis2dic(act_slot_values, 'act-slot-value')
        return act2idx, slot2idx, act_slot2idx, act_slot_value2idx

    def build_class_embed(self, filename, act2idx, slot2idx, act_slot2idx,
            act_slot_value2idx, emb_file=glove_path):
        with codecs.open(filename, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        words = []
        for line in lines:
            lis = line.split('-', 2)
            for phrase in lis:
                words.extend(phrase.strip().split())
        words = words + ['pad', 'unk', 'price', 'range', 'address', 'require',
        'more', 'alternatives', 'children', 'allowed', 'has', 'tv',
        'internet']
        words = list(set(words))

        word2emb = {word: np.zeros(100) for word in words}

        with open(emb_file, 'r+', encoding='utf-8') as f:
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

        act2emb = {}
        for act in act2idx:
            act2emb[act] = word2emb[act]

        def class2emb(dic, num):
            label2emb = {}
            for key in dic:
                lis = key.split('-', 2)
                assert len(lis) <= num
                vecs = []
                for word in lis:
                    word_lis = word.strip().split(' ')
                    vv = np.zeros(100)
                    for w in word_lis:
                        vv += word2emb[w]
                    vv = vv / len(word_lis)
                    vecs.append(vv)
                for i in range(num-len(lis)):
                    vecs.append(np.zeros(100))
                vec = np.concatenate(vecs)
                label2emb[key] = vec
            emb = torch.zeros(len(dic), 100 * num)
            emb.uniform_(-0.1, 0.1)
            for word in dic:
                if word in label2emb:
                    emb[dic[word]] = torch.from_numpy(label2emb[word])
            return label2emb

        print('==========================================')
        act2emb = class2emb(act2idx, 1)
        slot2emb = class2emb(slot2idx, 1)
        act_slot2emb = class2emb(act_slot2idx, 2)
        act_slot_value2emb = class2emb(act_slot_value2idx, 3)

        return act2emb, slot2emb, act_slot2emb, act_slot_value2emb


if __name__ == '__main__':
    # i do not want to change dstc2 to dstc3 in the code
    # but the code is really for dstc3
    dstc2_memory =  DSTC2Memory()
    dstc2_memory.build_save_memory('dstc2train.3seed', 'dstc2.3.train.class', 'memory.pt')
    memory = dstc2_memory.load_memory('memory.pt')
    print(len(memory['word2idx']))
    print(len(memory['train_classes']))
