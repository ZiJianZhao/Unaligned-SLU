# -*- codind: utf-8 -*-

import os, sys, random, argparse, time
import math
import json
import codecs

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from xslu.utils import EarlyStopping, Statistics
import xslu.Constants as Constants

class Trainer(object):

    def __init__(self,  model, criterion, optimizer, logger, cuda=True, early_stop_mode='max', tag='S2SPointer'):
        self.tag = tag
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger
        self.early_stop = EarlyStopping(mode=early_stop_mode, min_delta=0.1, patience=10)
        self.cuda = cuda

    def get_stats(self, loss, probs, target):
        pred = probs.max(1)[1]
        non_padding = target.ne(Constants.PAD)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        return Statistics(loss.item(), non_padding.sum().item(), num_correct)

    def train_on_epoch(self, epoch, data_iter):

        self.model.train()
        self.logger.info("Epoch {:02} {} begins training, {:05} examples ...................".format(
            epoch, self.tag, len(data_iter))
        )

        stats = Statistics()

        for (itr, batch) in enumerate(data_iter):
            enc, lengths, extra_zeros, enc_batch_extend_vocab_idx, dec_inp_ids, dec_out_ids = batch
            probs = self.model(enc, lengths, dec_inp_ids, extra_zeros, enc_batch_extend_vocab_idx)
            target = dec_out_ids.transpose(0, 1).contiguous().view(-1)
            loss = self.criterion(probs, target)

            # statistics
            loss_data = loss.data.clone()
            batch_stat = self.get_stats(loss_data, probs.data, target.data)
            stats.update(batch_stat)

            # grad update
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

        # loss logging
        self.logger.info("Epoch {:02} {} accu: {:6.2f}; ppl: {:6.2f}; elapsed_time: {:6.0f}s".format(
            epoch, self.tag, stats.accuracy(), stats.ppl(), stats.elapsed_time()
        ))

        return stats.accuracy()

    def valid_on_epoch(self, epoch, data_iter):

        self.model.eval()
        self.logger.info("Epoch {:02} {} begins validation, {:05} examples ...................".format(
            epoch, self.tag, len(data_iter))
        )

        stats = Statistics()

        for (itr, batch) in enumerate(data_iter):
            enc, lengths, extra_zeros, enc_batch_extend_vocab_idx, dec_inp_ids, dec_out_ids = batch
            probs = self.model(enc, lengths, dec_inp_ids, extra_zeros, enc_batch_extend_vocab_idx)
            target = dec_out_ids.transpose(0, 1).contiguous().view(-1)
            loss = self.criterion(probs, target)

            # statistics
            loss_data = loss.data.clone()
            batch_stat = self.get_stats(loss_data, probs.data, target.data)
            stats.update(batch_stat)

        # loss logging
        self.logger.info("Epoch {:02} {} accu: {:6.2f}; ppl: {:6.2f}; elapsed_time: {:6.0f}s".format(
            epoch, self.tag, stats.accuracy(), stats.ppl(), stats.elapsed_time()
        ))

        return stats.accuracy()

    def train(self, epochs, train_data, valid_data, chkpt_path):

        for epoch in range(1, epochs+1):

            _ = self.train_on_epoch(epoch, train_data)
            f = self.valid_on_epoch(epoch, valid_data)

            flag = self.early_stop(epoch, f, self.model.state_dict())
            best_epoch = self.early_stop.best_epoch
            best_metric = self.early_stop.best_metric
            best_model_state = self.early_stop.best_model_state

            if flag:
                self.logger.info('Early Stopping at epoch {:02}'.format(best_epoch))
                self.logger.info('Best metric is {:6.2f}'.format(best_metric))
                torch.save(best_model_state, chkpt_path)
                self.logger.info('Save the model state at {}'.format(chkpt_path))
                break

        if not flag:
            self.logger.info('Finally Stopping at epoch {:02}'.format(best_epoch))
            self.logger.info('Best metric is {:6.2f}'.format(best_metric))
            torch.save(best_model_state, chkpt_path)
            self.logger.info('Save the model state at {}'.format(chkpt_path))


