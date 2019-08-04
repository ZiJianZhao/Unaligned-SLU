# -*- coding: utf-8 -*-

import random, math
import numpy as np
import codecs
import json
from collections import defaultdict

import torch

from text.dstc2 import process_sent, process_word
import xslu.Constants as Constants

def enc2extend_ids(lis, word2idx):
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

def dec2ids(lis, word2idx):
    ids = []
    for w in lis:
        if w in word2idx:
            ids.append(word2idx[w])
        else:
            ids.append(Constants.UNK)
    return ids

def dec2extend_ids(lis, word2idx, oovs):
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

class DataLoader(object):

    def __init__(self, filename, word2idx, batch_size, cuda, epoch_shuffle):
        self.filename = filename
        self.word2idx = word2idx
        self.cuda = cuda
        self.epoch_shuffle = epoch_shuffle
        self.datas = self.read_file(filename)
        self.data_len = len(self.datas)
        self.batch_size = batch_size

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
            sent_lis = line[0].strip().split()
            if len(sent_lis) > 0:
                datas.append(line)
        return datas

    @staticmethod
    def enc_info(src, word2idx, cuda):

        lis = src.strip().split()

        ids = [word2idx[w] if w in word2idx else Constants.UNK for w in lis]
        data = torch.tensor(ids).view(1, -1)

        #word2idx = memory['dec2idx']
        ids, oov_list = enc2extend_ids(lis, word2idx)
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
    def dec_info(tgt, word2idx, enc_oov_list, cuda):

        if tgt.strip() == '':
            lis = []
        else:
            lis = tgt.strip().split()

        dec_inp_ids = [Constants.BOS] + dec2ids(lis, word2idx)
        dec_out_ids = dec2extend_ids(lis, word2idx, enc_oov_list) + [Constants.EOS]
        dec_inp_ids = torch.tensor(dec_inp_ids).view(1, -1)
        dec_out_ids = torch.tensor(dec_out_ids)

        if cuda:
            dec_inp_ids = dec_inp_ids.cuda()
            dec_out_ids = dec_out_ids.cuda()

        return dec_inp_ids, dec_out_ids

    @staticmethod
    def batch_info(lis, word2idx, cuda):

        lis.sort(key = lambda x: len(x[0].strip().split()), reverse=True)

        batch_size = len(lis)
        encs = []
        lengths = []
        enc_extends = []
        dec_inps = []
        dec_outs = []
        max_oov_num = 0

        for (src, tgt) in lis:
            src_lis = src.strip().split()
            src_ids = [word2idx[w] if w in word2idx else Constants.UNK for w in src_lis]
            encs.append(src_ids)
            lengths.append(len(src_ids))

            #word2idx = memory['dec2idx']
            src_ids, oov_list = enc2extend_ids(src_lis, word2idx)
            enc_extends.append(src_ids)
            enc_batch_extend_vocab_idx = torch.tensor(src_ids).view(1, -1)

            max_oov_num = max(max_oov_num, len(oov_list))

            if tgt.strip() == '':
                tgt_lis = []
            else:
                tgt_lis = tgt.strip().split()

            dec_inp_ids = [Constants.BOS] + dec2ids(tgt_lis, word2idx)
            dec_inps.append(dec_inp_ids)
            dec_out_ids = dec2extend_ids(tgt_lis, word2idx, oov_list) + [Constants.EOS]
            dec_outs.append(dec_out_ids)

        def pad_list(lis, cuda):
            max_len = max([len(l) for l in lis])
            lis = [l + [Constants.PAD for i in range(max_len - len(l))] for l in lis]
            vec = np.asarray(lis, dtype='int64')
            vec = torch.from_numpy(vec)
            if cuda: vec = vec.cuda()
            return vec

        enc_length = torch.LongTensor(lengths)
        if cuda: enc_length = enc_length.cuda()

        enc_ids = pad_list(encs, cuda)
        dec_inp_ids = pad_list(dec_inps, cuda)
        dec_out_ids = pad_list(dec_outs, cuda)
        enc_extend_ids = pad_list(enc_extends, cuda)

        if max_oov_num == 0:
            extra_zeros = None
        else:
            extra_zeros = torch.zeros((batch_size, max_oov_num))
            if cuda: extra_zeros = extra_zeros.cuda()

        return enc_ids, enc_length, extra_zeros, enc_extend_ids, dec_inp_ids, dec_out_ids


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

        endidx = min(self.idx + self.batch_size, self.data_len)
        indexs = self.indices[self.idx:endidx]
        lines = [self.datas[i] for i in indexs]

        self.idx = endidx

        return self.batch_info(lines, self.word2idx, self.cuda)
