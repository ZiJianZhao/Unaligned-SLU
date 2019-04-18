# -*- coding: utf-8 -*-

import random, math
import numpy as np
import codecs
import json
from collections import defaultdict

import torch

from xslu.utils import process_sent, process_word

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

class DADataset(object):

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
            utterance = line[0]
            triple = line[1]
            class_string = line[2]
            enc_lis = process_sent(class_string)
            dec_lis = process_sent(utterance)
            if len(enc_lis) > 0 and len(dec_lis) > 0:
                datas.append((utterance, class_string, triple))
        return datas

    @staticmethod
    def judge_utt_label(utterance, triple, class_string, memory, cuda):

        classes = triple.strip().split(';')
        new_string = class_string
        for cls in classes:
            lis = cls.strip().split('-', 2)
            if len(lis) == 3 and lis[2] in utterance:
                if random.random() > 0.5:
                    for word in lis[2].strip().split():
                        new_string = new_string.replace(word, 'unk')
        if new_string == class_string:
            return None
        else:
            lis = process_sent(new_string)
            word2idx = memory['enc2idx']
            ids = [word2idx[w] if w in word2idx else Constants.UNK for w in lis]
            data = torch.tensor(ids).view(1, -1)
            if cuda:
                data = data.cuda()
            return data


    @staticmethod
    def data_info(string, memory, cuda):

        lis = process_sent(string)
        if len(lis) == 0:
            raise Exception("Input string can not be empty string")

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
    def label_info(string, memory, enc_oov_list, cuda):

        lis = process_sent(string)

        word2idx = memory['dec2idx']

        inp_ids = value2ids(lis, word2idx)
        out_ids = value2extend_ids(lis, word2idx, enc_oov_list)
        inp_ids = [Constants.BOS] + inp_ids
        out_ids = out_ids + [Constants.EOS]
        inp_ids = torch.tensor(inp_ids).view(1, -1)
        out_ids = torch.tensor(out_ids)

        if cuda:
            inp_ids = inp_ids.cuda()
            out_ids = out_ids.cuda()

        return inp_ids, out_ids

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

        utterance, class_string, triple = self.datas[self.indices[self.idx]]
        self.idx += 1

        enc_data, enc_length, extra_zeros, enc_batch_extend_vocab_idx, oov_list = \
                self.data_info(class_string, self.memory, self.cuda)
        dec_inp, dec_out = self.label_info(utterance, self.memory, oov_list, self.cuda)

        if self.epoch_shuffle:
            pad_data = self.judge_utt_label(utterance, triple, class_string, self.memory, self.cuda)
            if pad_data is not None:
                enc_data = pad_data

        return enc_data, enc_length, extra_zeros, enc_batch_extend_vocab_idx, oov_list, \
                dec_inp, dec_out
