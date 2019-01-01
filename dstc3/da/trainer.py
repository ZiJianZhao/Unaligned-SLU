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

from xslu.utils import EarlyStopping, Fscore, Statistics
import xslu.Constants as Constants

from dataloader import DADataset 

class DATrainer(object):

    def __init__(self, model, criterion, optimizer, logger, early_stop_mode='max', cuda=True, tag='utterance-decoder'):
        self.tag = tag
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger
        self.cuda = cuda
        self.early_stop = EarlyStopping(mode=early_stop_mode)

    def get_stats(self, loss, probs, target):
        """
        Args:
            loss (FloatTensor): the loss computed by the loss criterion.
            probs (FloatTensor): the generated probs of the model.
            target (LongTensor): true targets

        Returns:
            stats (Statistics): statistics for this batch
        """
        pred = probs.max(1)[1] # predicted targets
        non_padding = target.ne(Constants.PAD)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        return Statistics(loss.item(), non_padding.sum().item(), num_correct)

    def train_on_epoch(self, epoch, data_iter, batch_size):
        
        self.logger.info("Epoch {:02} {} begins training, {:05} examples ...................".format(
                epoch, self.tag, len(data_iter))
            )

        self.model.train()
        
        stats = Statistics()
        batch_loss = 0.

        for (itr, batch) in enumerate(data_iter):
            
            enc_data, enc_length, extra_zeros, enc_batch_extend_vocab_idx, oov_list, \
                    dec_inp, dec_out = batch

            dist = self.model(enc_data, enc_length, 
                    dec_inp, extra_zeros, enc_batch_extend_vocab_idx)

            loss = self.criterion(dist, dec_out)
            batch_loss += loss

            # statistics
            loss_data = loss.data.clone()
            batch_stat = self.get_stats(loss_data, dist.data, dec_out.data)
            stats.update(batch_stat)

            # grad update
            if (itr > 0) and (itr % batch_size == 0):
                batch_loss = batch_loss / batch_size
                self.model.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                
                batch_loss = 0.
 
        # loss logging
        self.logger.info("Epoch {:02} {} ends training, accu: {:6.2f}; ppl: {:6.2f}; elapsed_time: {:6.0f}s".format(
                epoch, self.tag, stats.accuracy(), stats.ppl(), stats.elapsed_time()
            ))

        return None

    def valid_on_epoch(self, epoch, data_iter):

        self.logger.info("Epoch {:02} {} begins validing ...................".format(epoch, self.tag))

        self.model.eval()
        stats = Statistics()

        for (itr, batch) in enumerate(data_iter):
            
            enc_data, enc_length, extra_zeros, enc_batch_extend_vocab_idx, oov_list, \
                    dec_inp, dec_out = batch

            dist = self.model(enc_data, enc_length, 
                    dec_inp, extra_zeros, enc_batch_extend_vocab_idx)

            loss = self.criterion(dist, dec_out)

            # statistics
            loss_data = loss.data.clone()
            batch_stat = self.get_stats(loss_data, dist.data, dec_out.data)
            stats.update(batch_stat)

        # loss logging
        self.logger.info("Epoch {:02} {} ends validing, accu: {:6.2f}; ppl: {:6.2f}; elapsed_time: {:6.0f}s".format(
                epoch, self.tag, stats.accuracy(), stats.ppl(), stats.elapsed_time()
            ))

        self.logger.info('*****************************************************')
        return stats.accuracy()

    def train(self, epochs, batch_size, train_data, valid_data, chkpt_path):

        for epoch in range(1, epochs+1):

            _ = self.train_on_epoch(epoch, train_data, batch_size)
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
