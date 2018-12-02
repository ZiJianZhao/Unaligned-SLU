# -*- coding: utf-8 -*-
import os, sys
import logging
import time

import torch
import torch.nn as nn

from xslu.utils import Fscore, EarlyStopping

class BertTrainer4STC(object):
    """
    Class that controls the training process

    Args:
        
    """

    def __init__(self, model, criterion, optimizer, logger, early_stop_mode='max', cuda=True):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger
        self.early_stop = EarlyStopping(mode=early_stop_mode)
        self.cuda = cuda

    def update_fscore(self, score, probs, raw_classes, idx2class):
        n_sents = probs.size(0)
        scores = probs.data.cpu().numpy()
       
        for idx in range(n_sents):
            pred_classes = [i for i,p in enumerate(scores[idx]) if p > 0.5]
            pred_classes = [idx2class[i] for i in pred_classes]
            gold_classes = raw_classes[idx]
            score.update_tp_fp_fn(pred_classes, gold_classes)

        return score

    def train_on_epoch(self, data_iter, epoch):
        
        self.logger.info("Epoch {:02} begins training .......................".format(epoch))
        self.model.train()

        train_score = Fscore('Train')
        total_loss = 0.
        total_sent = 0.
        start_time = time.time()

        for (i, data_batch) in enumerate(data_iter):
            
            # data batch definition
            input_ids, attention_mask, labels, raw_classes, _ = data_batch
            probs = self.model(input_ids, None, attention_mask)

            # loss calculation
            loss = self.criterion(probs, labels)
            
            # statistics
            loss_data = loss.data.clone()
            total_loss += loss_data.item()
            total_sent += input_ids.size(0)
            train_score = self.update_fscore(train_score, probs.data, raw_classes, data_iter.idx2class)

            # update parameters
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        elapsed_time = time.time() - start_time
        avg_loss = total_loss / total_sent
        self.logger.info("Epoch {:02} Train, loss: {:6.2f}; elapsed time: {:6.0f}s".format(
            epoch, avg_loss, elapsed_time
            ))
        fscore = train_score.output_fscore(self.logger, epoch)

        return fscore

    def valid_on_epoch(self, data_iter, epoch):
        
        self.logger.info("Epoch {:02} begins validation .......................".format(epoch))
        self.model.eval()
        
        valid_score = Fscore('Valid')
        start_time = time.time()

        for (i, data_batch) in enumerate(data_iter):
             
            input_ids, attention_mask, labels, raw_classes, _ = data_batch
            probs = self.model(input_ids, None, attention_mask)

            valid_score = self.update_fscore(valid_score, probs.data, raw_classes, data_iter.idx2class)

        elapsed_time = time.time() - start_time
        self.logger.info("Epoch {:02} Valid, elapsed time: {:6.0f}s".format(
            epoch, elapsed_time
            ))
        fscore = valid_score.output_fscore(self.logger, epoch)

        return fscore

    def train(self, epochs, train_data, valid_data, chkpt_path):

        for epoch in range(1, epochs+1):

            _ = self.train_on_epoch(train_data, epoch)
            f = self.valid_on_epoch(valid_data, epoch)
            
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


