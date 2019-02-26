# -*- coding: utf-8 -*-

import random, math
import numpy as np
import codecs
import json

import torch

import xslu.Constants as Constants
from xslu.utils import process_sent, process_word


class BertIter4STC(object):
    """
    Define an ordered iterator for multi-label-classification framework
    Note: In context of SLU, also named as semantic tuple classifier (STC)

    Args:
        filename (str): preprocessed file name.
        memory (dict): dict containing word2idx, class2dix and so on.
        batch_size (int): batch size.
        cuda (bool): whether to use gpu.
        epoch_shuffle (bool, optional): whether shuffle the whole dataset at the begining of each epoch.

        batch_sort (bool, optional): whether sort the data insides a batch, if you want to use the variable-length rnn, set it True.
        Note: Currently, the batch_sort is default (only) to True to eliminate the effects of padding.

    Returns: (generator)
        enc_data (torch.LongTensor): batch_size * max_enc_len
        enc_length (torch.LongTensor): [batch_size], containing lengths of each line in a batch
        label_data (torch.LongTensor): batch_size * max_dec_len
        raw_classes (list of strings): text labels.
        indices (list of ints): record the sorted indices inside of a batch to recover the original order.
            Especially useful when we decode the test data.
    """

    def __init__(self, filename, tokenizer, class2idx, batch_size, cuda, epoch_shuffle, batch_sort=True):

        super(BertIter4STC, self).__init__()

        self.filename = filename
        self.tokenizer = tokenizer
        self.class2idx = class2idx
        self.idx2class = {v:k for k,v in self.class2idx.items()}
        self.batch_size = batch_size
        self.cuda = cuda
        self.epoch_shuffle = epoch_shuffle
        self.batch_sort = batch_sort

        self.data = self.convert_file_to_ids(filename)
        self.num_classes = len(class2idx)
        self.data_len = len(self.data)
        self.indices = list(range(self.data_len))
        self.batches = (self.data_len + self.batch_size - 1) // self.batch_size
        self.reset()

    def convert_file_to_ids(self, filename):
        with codecs.open(filename, 'r') as f:
            lines = f.readlines()
        lines = [line.split('\t<=>\t') for line in lines]
        lis = []
        for (sent, label) in lines:

            tokens = self.tokenizer.tokenize(sent)
            if len(tokens) == 0:
                continue
            tokens = ["[CLS]"] + tokens
            sent_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            if label.strip() == '':
                label_lis = []
            else:
                label_lis = label.strip().split(';')
            label_ids = []
            for l in label_lis:
                # this is a error-prone line
                # if the class2idx does not have the class, then it will miss some one
                # typical example: class2idx from train not contain all of the classes in valid.
                # therefore, for valid, the class ids are not correct
                # we keep this just for code coherence.
                if l in self.class2idx:
                    label_ids.append(self.class2idx.get(l))
            lis.append((sent_ids, label_ids, label_lis))
        print('Total data num: {}'.format(len(lis)))
        return lis

    def reset(self):
        self.idx = 0
        if self.epoch_shuffle:
            random.shuffle(self.indices)

    def __len__(self):
        return self.batches

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.data_len:
            self.reset()
            raise StopIteration
        index = self.indices[self.idx:min(self.idx+self.batch_size, self.data_len)]
        data = [self.data[i] for i in index]
        indices = list(range(len(index)))
        tmp = list(zip(data, indices))
        tmp.sort(key=lambda x: len(x[0][0]), reverse=True)

        sents = [tmp[i][0][0] for i in range(len(tmp))]
        labels = [tmp[i][0][1] for i in range(len(tmp))]
        raw_classes = [tmp[i][0][2] for i in range(len(tmp))]
        indices = [tmp[i][1] for i in range(len(tmp))]
        enc_length = torch.LongTensor([len(sents[i]) for i in range(len(sents))])

        max_len = max([len(l) for l in sents])
        sents = [l + [Constants.PAD for i in range(max_len-len(l))] for l in sents]
        enc = np.asarray(sents, dtype='int64')
        enc_data = torch.from_numpy(enc)

        attention_mask = torch.ones(enc_data.size()).type_as(enc_data)
        for i,l in enumerate(enc_length):
            l = l.item()
            attention_mask[i, l:] = 0

        label_data = torch.zeros(enc_data.size(0), self.num_classes)
        for i in range(len(labels)):
            for v in labels[i]:
                label_data[i][v] = 1

        if self.cuda:
            enc_data, enc_length, label_data = enc_data.cuda(), enc_length.cuda(), label_data.cuda()
            attention_mask = attention_mask.cuda()

        self.idx += len(tmp)

        return enc_data, attention_mask, label_data, raw_classes, indices

