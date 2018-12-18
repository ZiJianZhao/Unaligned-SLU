# -*- coding: utf-8 -*-
import os, datetime
import logging
import time, math
import copy

import torch
import torch.nn as nn

import numpy as np

# ********************************* Model Utils *************************************

def read_emb(word2idx, emb_dim=100, filename=None): 
    if filename is None:
        filename='/slfs1/users/zjz17/NLPData/glove.6B/glove.6B.{}d.txt'.format(emb_dim)

    with open(filename, 'r') as f:
        emb = torch.zeros(len(word2idx), emb_dim)
        emb.uniform_(-0.1, 0.1)
        for line in f:
            items = line.strip().split(' ')
            if len(items) == 2:
                continue
            word = items[0]
            if word in word2idx:
                vector = torch.tensor([float(value) for value in items[1:]])
                emb[word2idx[word]] = vector
    print('Load pre-trained word vectors from {}'.format(filename))
    return emb

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
            "Not all arguments have the same value: " + str(args)

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    Borrow from OpenNMT-py: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Utils.py

    Args:
        lengths (LongTensor): containing sequence lengths
        max_len (int): maximum padded length
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

def tally_parameters(model, logger=None):
    if logger is not None:
        func = logger.info
    else:
        func = print
    n_params = sum([p.nelement() for p in model.parameters()])
    func('* number of parameters: {}'.format(n_params))

# ********************************* Train Utils *************************************

class EarlyStopping(object):

    def __init__(self, mode='min', min_delta=0.1, patience=5):

        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

        self.best_metric = None
        self.best_epoch = None
        self.best_model_state = None
        self.num_bad_epochs = 0

        self.is_better = None
        self._init_is_better(mode, min_delta)

    def __call__(self, epoch, metric, model_state):

        model_state = copy.deepcopy(model_state)

        if self.best_metric is None:
            self.best_metric = metric
            self.best_epoch = epoch
            self.best_model_state = model_state
            return False

        if np.isnan(metric):
            return True

        if self.is_better(metric, self.best_metric):
            self.num_bad_epochs = 0.
            self.best_metric = metric
            self.best_epoch = epoch
            self.best_model_state = model_state
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in ['min', 'max']:
            raise ValueError('mode ' + mode + ' is unknown')
        # the following change is based on the assumption:
        # more training, even near performance on valid, the latter epoch is better trained
        if mode == 'min':
            #self.is_better = lambda a, best: a < best - min_delta
            self.is_better = lambda a, best: a < best + min_delta
        if mode == 'max':
            #self.is_better = lambda a, best: a > best + min_delta
            self.is_better = lambda a, best: a > best - min_delta


def drop_chkpt(chkpt_prefix, epoch, model, optimizer, fscore=None, loss=None):
    """
    Drop a checkpoint, if have f-score or loss, display them in the name
    """
    i = datetime.datetime.now()
    #chkpt_dir = "Y_{}_M_{}_D_{}_chkpts".format(i.year, i.month, i.day)
    
    real_model = model.module if isinstance(model, nn.DataParallel) else model

    chkpt = {
        'epoch': epoch,
        'model': real_model.state_dict(),
        'optimizer': optimizer
    }
    if fscore is None or loss is None:
        name = chkpt_prefix + '_{:02}.pt'.format(epoch)
    else:
        name = chkpt_prefix + '_{:02}_fscore_{:.2f}_loss_{:.2f}.pt'.format(epoch, fscore, loss)
    torch.save(chkpt, name)
    print("Drop a checkpoint at {}".format(name))

def load_chkpt(chkpt, model, optimizer=None, use_gpu=True):

    print('Load the checkpoint from {}'.format(chkpt))
    print("Use gpu is: {}".format(use_gpu))

    chkpt = torch.load(chkpt,
            map_location = lambda storage, loc: storage)
    epoch = chkpt['epoch']
    model.load_state_dict(chkpt['model'], strict=False)

    if optimizer is not None:
        
        optimizer = chkpt['optimizer']
        saved_optimizer_state_dict = optimizer.optimizer.state_dict()
        optimizer.set_parameters(model.named_parameters())

        optimizer.optimizer.load_state_dict(saved_optimizer_state_dict)

        if use_gpu:
            for state in optimizer.optimizer.state.values():
                for k,v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
    
    if optimizer is not None:
        return epoch, model, optimizer
    else:
        return epoch, model

# ********************************* SLU Utils *************************************

def process_sent(string):
    lis = string.strip().split()
    lis = [''.join(word.strip().split("'")) for word in lis]
    return lis

def process_word(word):
    word = ''.join(word.strip().split("'"))
    return word

class Fscore(object):

    def __init__(self, tag):
        self.tag = tag
        self.TP = 0.
        self.FP = 0.
        self.FN = 0.

    def update_tp_fp_fn(self, pred_classes, gold_classes):
        for p in pred_classes:
            if p in gold_classes:
                self.TP += 1
            else:
                self.FP += 1
        for g in gold_classes:
            if g not in pred_classes:
                self.FN += 1

    def output_fscore(self, logger, epoch):
        if self.TP == 0:
            precision, recall, fscore = 0., 0., 0.
        else:
            precision = 100 * self.TP / (self.TP + self.FP)
            recall = 100 * self.TP / (self.TP + self.FN)
            fscore = 100 * 2 * self.TP / (2 * self.TP + self.FN + self.FP)

        logger.info("Epoch {:02} {}, precision: {:6.2f}; recall: {:6.2f}; fscore: {:6.2f}".format(
            epoch, self.tag, precision, recall, fscore
            ))
        return fscore

# ********************************* Program Utils *************************************

def make_logger(filename):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(filename)
    handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

