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

from xslu.utils import EarlyStopping, Fscore

from dataloader import SLUDataset, ActDataset, SlotDataset, ValueDataset 
from decode import decode_slu, decode_act, decode_slot, decode_value

class ActTrainer(object):

    def __init__(self, model, criterion, optimizer, logger, early_stop_mode='max', cuda=True, tag='act-predictor'):
        self.tag = tag
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger
        self.cuda = cuda
        self.early_stop = EarlyStopping(mode=early_stop_mode)

    def train_on_epoch(self, epoch, data_iter, batch_size):
        
        self.logger.info("Epoch {:02} {} begins training, {:05} examples ...................".format(
                epoch, self.tag, len(data_iter))
            )

        self.model.train()

        start_time = time.time()
        batch_loss = 0.
        total_loss = 0.
        for (itr, batch) in enumerate(data_iter):
            
            data, lengths, label = batch
            
            scores, _, _ = self.model(data, lengths, None, None, None, None, None)
            
            # act predictor s
            loss = self.criterion(scores, label)
            batch_loss += loss
            total_loss += loss

            # grad update
            if (itr > 0) and (itr % batch_size == 0):
                batch_loss = batch_loss / batch_size
                self.model.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                
                batch_loss = 0.

        total_loss = total_loss.item() / len(data_iter)
        elapsed_time = time.time() - start_time
        self.logger.info("Epoch {:02} {} ends training, loss: {:6.2f}, elapsed_time: {:6.0f}s".format(
                epoch, self.tag, total_loss, elapsed_time)
            )

        return total_loss

    def valid_on_epoch(self, epoch, filename, memory):

        self.logger.info("Epoch {:02} {} begins validing ...................".format(epoch, self.tag))

        self.model.eval()

        start_time = time.time()
        score = Fscore(self.tag)

        lines = ActDataset.read_file(filename)

        for (utterance, class_string) in lines:

            gold_classes = ActDataset.class_info(class_string)
            pred_classes = decode_act(self.model, utterance, memory, self.cuda)

            score.update_tp_fp_fn(pred_classes, gold_classes)

        fscore = score.output_fscore(self.logger, epoch)

        elapsed_time = time.time() - start_time
        self.logger.info("Epoch {:02} {} end validing elapsed_time: {:6.0f}s".format(
            epoch, self.tag, elapsed_time)
        )
        self.logger.info('*****************************************************')

        return fscore

    def train(self, epochs, batch_size, memory, train_data, valid_file, chkpt_path):

        for epoch in range(1, epochs+1):

            _ = self.train_on_epoch(epoch, train_data, batch_size)
            f = self.valid_on_epoch(epoch, valid_file, memory)
            
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


class SlotTrainer(object):

    def __init__(self, model, criterion, optimizer, logger, early_stop_mode='max', cuda=True, tag='slot-predictor'):
        self.tag = tag
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger
        self.cuda = cuda
        self.early_stop = EarlyStopping(mode=early_stop_mode)

    def train_on_epoch(self, epoch, data_iter, batch_size):
        
        self.logger.info("Epoch {:02} {} begins training, {:05} examples ...................".format(
                epoch, self.tag, len(data_iter))
            )

        self.model.train()

        start_time = time.time()
        batch_loss = 0.
        total_loss = 0.

        for (itr, batch) in enumerate(data_iter):
            
            data, lengths, act_inputs, slot_label = batch
            
            _, slot_scores, _ = self.model(data, lengths, act_inputs, 
                    None, None, None, None)
            
            # slot predictor loss
            slot_loss = self.criterion(slot_scores, slot_label)
            batch_loss += slot_loss
            total_loss += slot_loss

            # grad update
            if (itr > 0) and (itr % batch_size == 0):
                batch_loss = batch_loss / batch_size
                self.model.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                
                batch_loss = 0.
 
        total_loss = total_loss.item() / len(data_iter)
        elapsed_time = time.time() - start_time
        self.logger.info("Epoch {:02} {} ends training, loss: {:6.2f}, elapsed_time: {:6.0f}s".format(
                epoch, self.tag, total_loss, elapsed_time)
            )

        return None

    def valid_on_epoch(self, epoch, filename, memory):

        self.logger.info("Epoch {:02} {} begins validing ...................".format(epoch, self.tag))

        self.model.eval()

        start_time = time.time()
        score = Fscore(self.tag)

        lines = SlotDataset.read_file(filename)

        for (utterance, class_string) in lines:

            gold_classes = SlotDataset.class_info(class_string)
            pred_classes = decode_slot(self.model, utterance, class_string, memory, self.cuda)

            score.update_tp_fp_fn(pred_classes, gold_classes)

        fscore = score.output_fscore(self.logger, epoch)

        elapsed_time = time.time() - start_time
        self.logger.info("Epoch {:02} {} end validing elapsed_time: {:6.0f}s".format(
            epoch, self.tag, elapsed_time)
        )
        self.logger.info('*****************************************************')

        return fscore

    def train(self, epochs, batch_size, memory, train_data, valid_file, chkpt_path):

        for epoch in range(1, epochs+1):

            _ = self.train_on_epoch(epoch, train_data, batch_size)
            f = self.valid_on_epoch(epoch, valid_file, memory)
            
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


class ValueTrainer(object):

    def __init__(self, model, criterion, optimizer, logger, early_stop_mode='max', cuda=True, tag='value-decoder'):
        self.tag = tag
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger
        self.cuda = cuda
        self.early_stop = EarlyStopping(mode=early_stop_mode)

    def train_on_epoch(self, epoch, data_iter, batch_size):
        
        self.logger.info("Epoch {:02} {} begins training, {:05} examples ...................".format(
                epoch, self.tag, len(data_iter))
            )

        self.model.train()

        start_time = time.time()
        batch_loss = 0.
        total_loss = 0.

        for (itr, batch) in enumerate(data_iter):
            
            data, lengths, extra_zeros, enc_batch_extend_vocab_idx, oov_list, \
                    act_inputs, act_slot_pairs, values_inp, values_out = batch

            _, _, value_dist_lis = self.model(data, lengths, act_inputs, act_slot_pairs,
                    values_inp, extra_zeros, enc_batch_extend_vocab_idx)

            # value predictor loss
            value_loss = 0.
            for i in range(len(value_dist_lis)):
                gold = values_out[i]
                prob = value_dist_lis[i]
                value_loss += self.criterion(prob, gold)
            batch_loss += value_loss
            total_loss += value_loss

            # grad update
            if (itr > 0) and (itr % batch_size == 0):
                batch_loss = batch_loss / batch_size
                self.model.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                
                batch_loss = 0.
 
        # loss logging
        total_loss = total_loss.item() / len(data_iter)
        elapsed_time = time.time() - start_time
        self.logger.info("Epoch {:02} {} ends training, loss: {:6.2f}, elapsed_time: {:6.0f}s".format(
                epoch, self.tag, total_loss, elapsed_time)
            )

        return None

    def valid_on_epoch(self, epoch, filename, memory):

        self.logger.info("Epoch {:02} {} begins validing ...................".format(epoch, self.tag))

        self.model.eval()

        start_time = time.time()
        score = Fscore(self.tag)

        lines = ValueDataset.read_file(filename)

        for (utterance, class_string) in lines:

            gold_classes = ValueDataset.class_info(class_string)
            pred_classes = decode_value(self.model, utterance, class_string, memory, self.cuda)

            score.update_tp_fp_fn(pred_classes, gold_classes)

        fscore = score.output_fscore(self.logger, epoch)

        elapsed_time = time.time() - start_time
        self.logger.info("Epoch {:02} {} end validing elapsed_time: {:6.0f}s".format(
            epoch, self.tag, elapsed_time)
        )
        self.logger.info('*****************************************************')

        return fscore

    def train(self, epochs, batch_size, memory, train_data, valid_file, chkpt_path):

        for epoch in range(1, epochs+1):

            _ = self.train_on_epoch(epoch, train_data, batch_size)
            f = self.valid_on_epoch(epoch, valid_file, memory)
            
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

class SLUTrainer(object):

    def __init__(self,  model, criterions, optimizer, logger, early_stop_mode='max', cuda=True, tag='slu-predictor'):
        self.tag = tag
        self.model = model
        self.stc_criterion, self.nll_criterion = criterions
        self.optimizer = optimizer
        self.logger = logger
        self.early_stop = EarlyStopping(mode=early_stop_mode)
        self.cuda = cuda

    def train_on_epoch(self, epoch, data_iter, batch_size):
        
        self.logger.info("Epoch {:02} {} begins training, {:05} examples ...................".format(
                epoch, self.tag, len(data_iter))
            )
        self.model.train()

        start_time = time.time()
        batch_loss_num_pairs = [ [0., 0.], [0., 0.], [0., 0.] ]
        total_loss_num_pairs = [ [0., 0.], [0., 0.], [0., 0.] ]

        for (itr, batch) in enumerate(data_iter):
            
            data, lengths, extra_zeros, enc_batch_extend_vocab_idx, oov_list, \
                    act_label, act_inputs, slot_label, act_slot_pairs, values_inp, values_out = batch
           
            act_scores, slot_scores, value_dist_lis = self.model(data, lengths, act_inputs, act_slot_pairs,
                    values_inp, extra_zeros, enc_batch_extend_vocab_idx)
            
            # act predictor loss
            act_loss = self.stc_criterion(act_scores, act_label)
            batch_loss_num_pairs[0][0] += act_loss
            batch_loss_num_pairs[0][1] += 1

            # slot predictor loss
            if slot_scores is not None:
                slot_loss = self.stc_criterion(slot_scores, slot_label)
                batch_loss_num_pairs[1][0] += slot_loss
                batch_loss_num_pairs[1][1] += 1

            # value predictor loss
            if value_dist_lis is not None:
                value_loss = 0.
                for i in range(len(value_dist_lis)):
                    gold = values_out[i]
                    prob = value_dist_lis[i]
                    value_loss += self.nll_criterion(prob, gold)
                batch_loss_num_pairs[2][0] += value_loss
                batch_loss_num_pairs[2][1] += 1

            # grad update
            if (itr > 0) and (itr % batch_size == 0):
                total_loss = 0.
                for loss_num in batch_loss_num_pairs:
                    if loss_num[1] > 0:
                        total_loss += loss_num[0] / loss_num[1]

                self.model.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # loss history update and reset
                for i in range(len(batch_loss_num_pairs)):
                    if isinstance(batch_loss_num_pairs[i][0], float):
                        total_loss_num_pairs[i][0] += batch_loss_num_pairs[i][0]
                    else:
                        total_loss_num_pairs[i][0] += batch_loss_num_pairs[i][0].item()
                    total_loss_num_pairs[i][1] += batch_loss_num_pairs[i][1]
                batch_loss_num_pairs = [ [0., 0.], [0., 0.], [0., 0.] ]

        # loss logging
        act_avg_loss = total_loss_num_pairs[0][0] / total_loss_num_pairs[0][1]
        slot_avg_loss = total_loss_num_pairs[1][0] / total_loss_num_pairs[1][1]
        value_avg_loss = total_loss_num_pairs[2][0] / total_loss_num_pairs[2][1]
        total_avg_loss = act_avg_loss + slot_avg_loss + value_avg_loss

        elapsed_time = time.time() - start_time
        self.logger.info("Epoch {:02} {} ends training act_loss: {:6.2f}; slot_loss: {:6.2f}; value_loss: {:6.2f}, total_loss: {:6.2f}, elapsed_time: {:6.0f}s".format(
                epoch, self.tag, act_avg_loss, slot_avg_loss, value_avg_loss, total_avg_loss, elapsed_time)
            )
        
        return total_avg_loss

    def valid_on_epoch(self, epoch, filename, memory):

        self.logger.info("Epoch {:02} {} begins validing ...................".format(epoch, self.tag))

        self.model.eval()

        start_time = time.time()
        score = Fscore(self.tag)

        lines = SLUDataset.read_file(filename)

        for (utterance, class_string) in lines:

            gold_classes = SLUDataset.class_info(class_string)
            pred_classes = decode_slu(self.model, utterance, memory, self.cuda)


            score.update_tp_fp_fn(pred_classes, gold_classes)

        fscore = score.output_fscore(self.logger, epoch)

        elapsed_time = time.time() - start_time
        self.logger.info("Epoch {:02} {} ends validing elapsed_time: {:6.0f}s".format(
            epoch, self.tag, elapsed_time)
        )
        self.logger.info('*****************************************************')

        return fscore

    def train(self, epochs, batch_size, memory, train_data, valid_file, chkpt_path):

        for epoch in range(1, epochs+1):

            _ = self.train_on_epoch(epoch, train_data, batch_size)
            f = self.valid_on_epoch(epoch, valid_file, memory)
            
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


