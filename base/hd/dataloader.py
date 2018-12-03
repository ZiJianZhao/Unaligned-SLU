# -*- coding: utf-8 -*-

import random, math
import numpy as np
import codecs
import json
from collections import defaultdict

import torch

from text.dstc2 import process_sent, process_word
import xslu.Constants as Constants

def seq2extend_ids(lis, word2idx):
    ids = []
    oovs = []
    for w in lis:
        if w in word2idx:
            ids.append(word2idx[w])
        else:
            if w not in oovs:
                oovs.append(w)
            oov_num = oovs.index(w)
            ids.append(len(word2idx) + oov_num)
    return ids, oovs

def value2ids(lis, word2idx):
    ids = []
    for w in lis:
        if w in word2idx:
            ids.append(word2idx[w])
        else:
            ids.append(Constants.UNK)
    return ids

def value2extend_ids(lis, word2idx, oovs):
    ids = []
    for w in lis:
        if w in word2idx:
            ids.append(word2idx[w])
        else:
            if w in oovs:
                ids.append(len(word2idx) + oovs.index(w))
            else:
                ids.append(Constants.UNK)
    return ids

class SLUDataset(object):
    """Specially for DSTC2 slu prediction"""

    def __init__(self, filename, memory, cuda, epoch_shuffle):
        self.filename = filename
        self.memory = memory
        self.cuda = cuda
        self.epoch_shuffle = epoch_shuffle
        self.datas = self.read_file(filename)
        self.data_len = len(self.datas)

        self.idx = 0
        self.indices = list(range(self.data_len))
        self.reset()

    @staticmethod
    def read_file(filename):
        with codecs.open(filename, 'r') as f:
            lines = f.readlines()
        lines = [line.split('\t<=>\t') for line in lines]
        datas = []
        for line in lines:
            sent_lis = process_sent(line[0])
            if len(sent_lis) > 0:
                datas.append(line)
        return datas

    @staticmethod
    def class_info(class_string):
        if class_string.strip() == '':
            classes = []
        else:
            classes = class_string.strip().split(';')
            classes = [cls.strip() for cls in classes]
        return classes

    @staticmethod
    def data_info(utterance, memory, cuda):

        lis = process_sent(utterance)
        if len(lis) == 0:
            raise Error("Input utterance can not be empty string")

        word2idx = memory['enc2idx']
        ids = [word2idx[w] if w in word2idx else Constants.UNK for w in lis]
        data = torch.tensor(ids).view(1, -1)

        word2idx = memory['dec2idx']
        ids, oov_list = seq2extend_ids(lis, word2idx)
        enc_batch_extend_vocab_idx = torch.tensor(ids).view(1, -1)

        if len(oov_list) == 0:
            extra_zeros = None
        else:
            extra_zeros = torch.zeros((1, len(oov_list)))

        if cuda:
            data = data.cuda()
            enc_batch_extend_vocab_idx = enc_batch_extend_vocab_idx.cuda()
            if extra_zeros is not None:
                extra_zeros = extra_zeros.cuda()

        return data, None, extra_zeros, enc_batch_extend_vocab_idx, oov_list

    @staticmethod
    def label_info(class_string, memory, enc_oov_list, cuda):
        """for all labels, we pad the empty position with pad token
            * if label is empty: --->>> pad-pad-pad
            * if label is act: --->>> act-slot-pad
            * if label is act-slot: --->>> act-slot-pad 
        """

        act2idx = memory['act2idx']
        slot2idx = memory['slot2idx']
        word2idx = memory['dec2idx']

        def sample_acts(act_ids, sample_prob=0.4, sample_num=1):
            return []
            p = random.random()
            if p < sample_prob:
                return []
            acts = [memory['act2idx'][word] for word in ['confirm', 'inform', 'deny', 'request'] ]
            lis = set(acts) - set(act_ids)
            new_ids = random.sample(lis, sample_num)
            return new_ids

        if class_string.strip() == '':
            lis = []
        else:
            lis = class_string.strip().split(';')

        # act predictor labels
        if len(lis) == 0:
            act_label = torch.zeros(1, len(act2idx))
        else:
            acts = [label.strip().split('-')[0] for label in lis]
            act_ids = [act2idx[act] for act in acts]
            act_label = torch.zeros(1, len(act2idx))
            for i in act_ids:
                act_label[0, i] = 1
        
        if cuda: act_label = act_label.cuda()

        lis = [string for string in lis if len(string.strip().split('-')) > 1]
        if len(lis) == 0:
            return act_label, None, None, None, None, None

        # slot predictor inputs and labels
        dic = defaultdict(list)
        for label in lis:
            tmp = label.strip().split('-')
            dic[tmp[0]].append(tmp[1])
        act_ids = []
        slot_ids = [[] for _ in range(len(dic))]
        for (i, (key, value)) in enumerate(dic.items()):
            act_ids.append(act2idx[key])
            for v in value:
                slot_ids[i].append(slot2idx[v])

        # negative sampling
        negative_ids = sample_acts(act_ids)
        if len(negative_ids) != 0:
            act_ids.extend(negative_ids)
            slot_ids.extend([[] for _ in range(len(negative_ids))])

        act_inputs = torch.tensor(act_ids).view(-1, 1)
        slot_label = torch.zeros(len(slot_ids), len(slot2idx))
        for i in range(len(slot_ids)):
            for j in slot_ids[i]:
                slot_label[i][j] = 1

        if cuda:
            act_inputs = act_inputs.cuda()
            slot_label = slot_label.cuda()

        lis = [string for string in lis if len(string.strip().split('-')) > 2]
        if len(lis) == 0:
            return act_label, act_inputs, slot_label, None, None, None

        # value decoder inputs and labels
        ## Note: one act-slot pair can only correspoding to one value
        dic = {}
        for label in lis:
            tmp = label.strip().split('-')
            act_slot = '-'.join(tmp[0:2])
            dic[act_slot] = tmp[2]
        act_slot_ids = [ [] for _ in range(len(dic))]
        value_inp_ids = [[] for _ in range(len(dic))]
        value_out_ids = [[] for _ in range(len(dic))]
        for (i, (key, value)) in enumerate(dic.items()):
            act_slot_lis = key.strip().split('-')
            act_slot_ids[i].append(act2idx[act_slot_lis[0]])
            act_slot_ids[i].append(slot2idx[act_slot_lis[1]])
            inp_ids = value2ids(value.strip().split(), word2idx)
            out_ids = value2extend_ids(value.strip().split(), word2idx, enc_oov_list)
            value_inp_ids[i] = [Constants.BOS] + inp_ids
            value_out_ids[i] = out_ids + [Constants.EOS]
        act_slot_pairs = [torch.tensor(ids).view(1, -1) for ids in act_slot_ids]
        values_inp = [torch.tensor(ids).view(1, -1) for ids in value_inp_ids]
        values_out = [torch.tensor(ids) for ids in value_out_ids]

        if cuda:
            act_slot_pairs = [asp.cuda() for asp in act_slot_pairs]
            values_inp = [v.cuda() for v in values_inp]
            values_out = [v.cuda() for v in values_out]
    
        return act_label, act_inputs, slot_label, act_slot_pairs, values_inp, values_out

    def reset(self):
        self.idx = 0
        if self.epoch_shuffle:
            random.shuffle(self.indices)
    
    def __len__(self):
        return self.data_len

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.data_len:
            self.reset()
            raise StopIteration
        
        utterance, class_string = self.datas[self.indices[self.idx]]
        self.idx += 1

        data, lengths, extra_zeros, enc_batch_extend_vocab_idx, oov_list = \
                self.data_info(utterance, self.memory, self.cuda)
        act_label, act_inputs, slot_label, act_slot_pairs, values_inp, values_out = \
                self.label_info(class_string, self.memory, oov_list, self.cuda)
        
        return data, lengths, extra_zeros, enc_batch_extend_vocab_idx, oov_list, \
                act_label, act_inputs, slot_label, act_slot_pairs, values_inp, values_out

class ActDataset(object):
    """Specially for DSTC2 act prediction"""

    def __init__(self, filename, memory, cuda, epoch_shuffle):
        self.filename = filename
        self.memory = memory
        self.cuda = cuda
        self.epoch_shuffle = epoch_shuffle
        self.datas = self.read_file(filename)
        self.data_len = len(self.datas)

        self.idx = 0
        self.indices = list(range(self.data_len))
        self.reset()

    @staticmethod
    def read_file(filename):
        with codecs.open(filename, 'r') as f:
            lines = f.readlines()
        lines = [line.split('\t<=>\t') for line in lines]
        datas = []
        for line in lines:
            sent_lis = process_sent(line[0])
            if len(sent_lis) > 0:
                datas.append(line)
        return datas

    @staticmethod
    def class_info(class_string):
    
        if class_string.strip() == '':
            acts = []
        else:
            lis = class_string.strip().split(';')
            acts = [label.strip().split('-')[0] for label in lis]
            acts = [a.strip() for a in acts]

        return list(set(acts))

    @staticmethod
    def data_info(utterance, memory, cuda):

        lis = process_sent(utterance)
        if len(lis) == 0:
            raise Error("Input utterance can not be empty string")

        word2idx = memory['enc2idx']
        ids = [word2idx[w] if w in word2idx else Constants.UNK for w in lis]
        data = torch.tensor(ids).view(1, -1)

        if cuda:
            data = data.cuda()

        return data, None

    @staticmethod
    def label_info(class_string, memory, cuda):
    
        act2idx = memory['act2idx']
        acts = ActDataset.class_info(class_string)

        act_label = torch.zeros(1, len(act2idx))
        act_ids = [act2idx[act] for act in acts]
        for i in act_ids:
            act_label[0, i] = 1
        
        if cuda: act_label = act_label.cuda()

        return act_label

    def reset(self):
        self.idx = 0
        if self.epoch_shuffle:
            random.shuffle(self.indices)
    
    def __len__(self):
        return self.data_len

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.data_len:
            self.reset()
            raise StopIteration
        
        utterance, class_string = self.datas[self.idx]
        self.idx += 1

        data, lengths = self.data_info(utterance, self.memory, self.cuda)
        label = self.label_info(class_string, self.memory, self.cuda)
        
        return data, lengths, label

class SlotDataset(object):
    """Specially for DSTC2 slot prediction given acts"""

    def __init__(self, filename, memory, cuda, epoch_shuffle):
        self.filename = filename
        self.memory = memory
        self.cuda = cuda
        self.epoch_shuffle = epoch_shuffle
        self.datas = self.read_file(filename)
        self.data_len = len(self.datas)

        self.idx = 0
        self.indices = list(range(self.data_len))
        self.reset()

    @staticmethod
    def read_file(filename):
        with codecs.open(filename, 'r') as f:
            lines = f.readlines()
        lines = [line.split('\t<=>\t') for line in lines]
        datas = []
        for line in lines:
            sent_lis = process_sent(line[0])
            if len(sent_lis) == 0:
                continue
            class_string = line[1]
            if class_string.strip() == '':
                continue
            classes = class_string.strip().split(';')
            classes = [string for string in classes if len(string.strip().split('-')) >= 2]
            classes = sorted(classes)
            if len(classes) == 0:
                continue

            pre_act = classes[0].strip().split('-')[0]
            tmp = []
            for string in classes:
                act = string.strip().split('-')[0]
                if act == pre_act:
                    tmp.append(string)
                else:
                    class_string = ';'.join(tmp)
                    datas.append((line[0], class_string))
                    tmp = [string]
            if len(tmp) != 0:
                class_string = ';'.join(tmp)
                datas.append((line[0], class_string))
        return datas

    @staticmethod
    def class_info(class_string):
        """influence result a little but can accept"""
        slots = []
        if class_string.strip() == '':
            classes = []
        else:
            classes = class_string.strip().split(';')
            for string in classes:
                lis = string.strip().split('-')
                if len(lis) >= 2:
                    slots.append(lis[1])
        return list(set(slots))

    @staticmethod
    def data_info(utterance, memory, cuda):

        lis = process_sent(utterance)
        if len(lis) == 0:
            raise Error("Input utterance can not be empty string")

        word2idx = memory['enc2idx']
        ids = [word2idx[w] if w in word2idx else Constants.UNK for w in lis]
        data = torch.tensor(ids).view(1, -1)

        if cuda:
            data = data.cuda()

        return data, None

    @staticmethod
    def label_info(class_string, memory, cuda):
    
        if class_string.strip() == '':
            lis = []
        else:
            lis = class_string.strip().split(';')

        act2idx = memory['act2idx']
        slot2idx = memory['slot2idx']

        act_ids = []
        slot_ids = []
        for string in lis:
            str_lis = string.strip().split('-')
            act = str_lis[0]
            slot = str_lis[1]
            act_ids.append(act2idx[act])
            slot_ids.append(slot2idx[slot])
        act_ids = list(set(act_ids))
        slot_ids = list(set(slot_ids))
        act_inputs = torch.tensor(act_ids).view(1, 1)
        slot_label = torch.zeros(1, len(slot2idx))
        for i in slot_ids:
            slot_label[0][i] = 1

        if cuda:
            act_inputs = act_inputs.cuda()
            slot_label = slot_label.cuda()

        return act_inputs, slot_label

    def reset(self):
        self.idx = 0
        if self.epoch_shuffle:
            random.shuffle(self.indices)
    
    def __len__(self):
        return self.data_len

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.data_len:
            self.reset()
            raise StopIteration
        
        utterance, class_string = self.datas[self.idx]
        self.idx += 1

        data, lengths = self.data_info(utterance, self.memory, self.cuda)
        act_inputs, slot_label = self.label_info(class_string, self.memory, self.cuda)
        
        return data, lengths, act_inputs, slot_label

class ValueDataset(object):
    """Specially for DSTC2 value decoder"""

    def __init__(self, filename, memory, cuda, epoch_shuffle):
        self.filename = filename
        self.memory = memory
        self.cuda = cuda
        self.epoch_shuffle = epoch_shuffle
        self.datas = self.read_file(filename)
        self.data_len = len(self.datas)

        self.idx = 0
        self.indices = list(range(self.data_len))
        self.reset()

    @staticmethod
    def read_file(filename):
        with codecs.open(filename, 'r') as f:
            lines = f.readlines()
        lines = [line.split('\t<=>\t') for line in lines]
        datas = []
        for line in lines:
            sent_lis = process_sent(line[0])
            if len(sent_lis) == 0:
                continue
            class_string = line[1]
            if class_string.strip() == '':
                continue
            classes = class_string.strip().split(';')

            for string in classes:
                num = len(string.strip().split('-'))
                if num >= 3:
                    datas.append((line[0], string))
                    # This is addition for debugging 
                    # break
                    # ==============================
        return datas

    @staticmethod
    def class_info(class_string):
        values = [ class_string.strip().split('-')[2].strip() ]
        return values

    @staticmethod
    def data_info(utterance, memory, cuda):

        lis = process_sent(utterance)
        if len(lis) == 0:
            raise Error("Input utterance can not be empty string")

        word2idx = memory['enc2idx']
        ids = [word2idx[w] if w in word2idx else Constants.UNK for w in lis]
        data = torch.tensor(ids).view(1, -1)

        word2idx = memory['dec2idx']
        ids, oov_list = seq2extend_ids(lis, word2idx)
        enc_batch_extend_vocab_idx = torch.tensor(ids).view(1, -1)

        if len(oov_list) == 0:
            extra_zeros = None
        else:
            extra_zeros = torch.zeros((1, len(oov_list)))

        if cuda:
            data = data.cuda()
            enc_batch_extend_vocab_idx = enc_batch_extend_vocab_idx.cuda()
            if extra_zeros is not None:
                extra_zeros = extra_zeros.cuda()

        return data, None, extra_zeros, enc_batch_extend_vocab_idx, oov_list

    @staticmethod
    def label_info(class_string, memory, enc_oov_list, cuda):

        act, slot, value = tuple(class_string.strip().split('-'))

        act2idx = memory['act2idx']
        slot2idx = memory['slot2idx']
        word2idx = memory['dec2idx']

        act_inputs = torch.tensor([act2idx[act]]).view(1, 1)

        # value decoder inputs and labels
        ## Note: one act-slot pair can only correspoding to one value

        act_slot_ids = torch.tensor([act2idx[act], slot2idx[slot]]).view(1, 2)
        inp_ids = value2ids(value.strip().split(), word2idx)
        out_ids = value2extend_ids(value.strip().split(), word2idx, enc_oov_list)
        value_inp_ids = [Constants.BOS] + inp_ids
        value_out_ids = out_ids + [Constants.EOS]
        value_inp_ids = torch.tensor(value_inp_ids).view(1, -1)
        value_out_ids = torch.tensor(value_out_ids)

        if cuda:
            act_inputs = act_inputs.cuda()
            act_slot_ids = act_slot_ids.cuda()
            value_inp_ids = value_inp_ids.cuda()
            value_out_ids = value_out_ids.cuda()

        act_slot_pairs = [act_slot_ids]
        values_inp = [value_inp_ids]
        values_out = [value_out_ids]
    
        return act_inputs, act_slot_pairs, values_inp, values_out

    def reset(self):
        self.idx = 0
        if self.epoch_shuffle:
            random.shuffle(self.indices)
    
    def __len__(self):
        return self.data_len

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.data_len:
            self.reset()
            raise StopIteration
        
        utterance, class_string = self.datas[self.idx]
        self.idx += 1

        data, lengths, extra_zeros, enc_batch_extend_vocab_idx, oov_list = \
                self.data_info(utterance, self.memory, self.cuda)
        act_inputs, act_slot_pairs, values_inp, values_out = \
                self.label_info(class_string, self.memory, oov_list, self.cuda)
        
        return data, lengths, extra_zeros, enc_batch_extend_vocab_idx, oov_list, \
                act_inputs, act_slot_pairs, values_inp, values_out

