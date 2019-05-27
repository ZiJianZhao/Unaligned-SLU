# -*- codind: utf-8 -*-

import os, sys, random, argparse, time
import math
import json
import codecs
from collections import defaultdict

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

#install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
install_path = '/slfs1/users/zjz17/SLU/nn_slu_all'
sys.path.append(install_path)

root_dir = os.path.join(install_path, 'exp-le/')
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

from models.unaligned.modules import Embedding, EncoderRNN, Attention
from utils.unaligned import read_emb, tally_parameters, make_logger, process_sent, slot2dic
import utils.unaligned.Constants as Constants

# ============================== Data Processing =====================================

def text2ids(string, word2idx, cuda):
    lis = string.strip().split()
    lis = [''.join(word.strip().split("'")) for word in lis]
    ids = [word2idx.get(w) if w in word2idx else Constants.UNK for w in lis]
    data = torch.LongTensor([ids])
    if cuda: data = data.cuda()
    return data

def class2ids(string, class2idx, cuda):
    data = torch.zeros(1, len(class2idx))
    if string.strip() == '':
        label_lis = []
    else:
        label_lis = string.strip().split(';')
    label_ids = [class2idx[cls] for cls in label_lis]
    for i in label_ids:
        data[0, i] = 1
    if cuda: data = data.cuda()
    return data

class DataIter(object):
    """Specially for DSTC2 slu prediction"""

    def __init__(self, filename, memory, cuda, epoch_shuffle):
        self.filename = filename
        self.memory = memory
        self.cuda = cuda
        self.epoch_shuffle = epoch_shuffle
        self.datas = self.read_file(filename)
        self.data_len = len(self.datas)

        self.mask = self.init_mask()

        self.idx = 0
        self.indices = list(range(self.data_len))
        self.reset()

    def init_mask(self):
        class2idx = self.memory['class2idx']
        data = torch.zeros(1, len(class2idx)).float()
        classes = []
        for (utterance, label) in self.datas:
            if label.strip() == '':
                continue
            else:
                labels = label.strip().split(';')
                classes.extend(labels)
        classes = list(set(classes))
        for cls in classes:
            data[0, class2idx[cls]] = 1
        if self.cuda: data = data.cuda()
        return data

    def read_file(self, filename):
        with codecs.open(filename, 'r') as f:
            lines = f.readlines()
        lines = [line.split('\t<=>\t') for line in lines]
        datas = []
        for line in lines:
            if line[0].strip() == '':
                continue
            datas.append(line)
        return datas

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

        data = text2ids(utterance, self.memory['word2idx'], self.cuda)
        label = class2ids(class_string, self.memory['class2idx'], self.cuda)
        length = None
        mask = self.mask

        self.idx += 1

        return data, length, label, mask

# ============================== Model Definition =====================================
class LE(nn.Module):

    def __init__(self, word_vocab_size, class_size, emb_dim=100, hid_dim=128, dropout=0.5):
        super(LE, self).__init__()

        self.hid_dim = hid_dim

        self.word_emb = Embedding(word_vocab_size, emb_dim, Constants.PAD, dropout)
        self.encoder = EncoderRNN('LSTM', True, 1, hid_dim, self.word_emb, 0.)

        self.lin1 = nn.Linear(2 * hid_dim, emb_dim * 3)
        self.lin2 = nn.Linear(3 * emb_dim, class_size)
        #self.lin1 = nn.Linear(2 * hid_dim, 2 * hid_dim)
        #self.lin  = nn.Linear(2 * hid_dim, 3 * emb_dim)
        #self.lin2 = nn.Linear(3 * emb_dim, class_size)

        self.init_params()

    def init_params(self, initrange=0.1):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -initrange, initrange)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, data, length):
        """data is of batch size 1"""
        outputs, hiddens = self.encoder(data, length)  # 1 * seq_len * (hid_dim * 2)
        h_T = hiddens[0].transpose(0, 1).contiguous().view(-1, self.hid_dim * 2)
        vec = F.tanh(self.lin1(h_T))
        score = F.sigmoid(self.lin2(vec))
        #vec = F.relu(self.lin1(h_T))
        #vec = F.tanh(self.lin(vec))
        #score = F.sigmoid(self.lin2(vec))
        return score

# ============================== Training Details =====================================

seed = 3344
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

epochs = 100
batch_size = 20
torch.cuda.set_device(0)
cuda = True

root_dir = '/slfs1/users/zjz17/SLU/Unaligned-SLU/zs/'
manual_dir = ''
save_name = 'dstc2train.3seed.pt'
#save_name = 'dstc2train.pt'

experiment_dir = './'
task = 'manual'
data_dir = manual_dir

load_model='dstc2train.pt'
#load_model=None

test_json = os.path.join(data_dir, 'dstc3.test.json')
save_decode = os.path.join(data_dir, 'pred.json')

memory = torch.load(data_dir+'memory.pt')
exp_path = experiment_dir
save_path = os.path.join(exp_path, save_name)
log_path =  os.path.join(exp_path, 'log.train')
logger = make_logger(log_path)

class MaskedBCELoss(nn.Module):

    def __init__(self, cuda):
        super(MaskedBCELoss, self).__init__()
        self.criterion = nn.BCELoss(size_average=False, reduce=False)
        if cuda:
            self.criterion = self.criterion.cuda()

    def forward(self, preds, golds, mask=None):
        #print(preds.size(), golds.size(), mask.size())
        preds = preds.view(-1, 1)
        golds = golds.view(-1, 1)
        loss = self.criterion(preds, golds)
        #print(loss)
        if mask is not None:
            mask = mask.view(-1, 1)
            loss = loss * mask.float()
        #print(loss)
        loss = torch.sum(loss)
        return loss

def train_on_epoch(epoch, model, dataiter, criterion, optimizer):

    model.train()

    start = time.time()
    total_loss = 0.
    loss = 0.
    for (idx, (data, length, label, mask)) in enumerate(dataiter):
        score = model(data, length)
        loss += criterion(score, label, mask)
        #loss += criterion(score, label)

        if (idx > 0) and (idx % batch_size == 0):
            total_loss += loss.item()

            loss = loss / batch_size
            model.zero_grad()
            loss.backward()
            optimizer.step()

            loss = 0.

    logger.info('Epoch {}, loss: {}'.format(epoch, total_loss / len(dataiter)))
    print('Epoch {} ends'.format(epoch))
    print('Elapsed time: {}'.format(time.time() - start))

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

    def output_f_score(self, logger, epoch):
        if self.TP == 0:
            precision, recall, fscore = 0., 0., 0.
        else:
            precision = 100 * self.TP / (self.TP + self.FP)
            recall = 100 * self.TP / (self.TP + self.FN)
            fscore = 100 * 2 * self.TP / (2 * self.TP + self.FN + self.FP)

        logger.info("Epoch {:02} {} precision: {:6.2f}; recall: {:6.2f}; fscore: {:6.2f}".format(
            epoch, self.tag, precision, recall, fscore
            ))
        return fscore

def decode_utterance(model, utterance):
    data = text2ids(utterance, memory['word2idx'], cuda)
    triples = []
    score = model(data, None)
    score = score.data.cpu().view(-1).numpy()
    preds = [i for i, p in enumerate(score) if p > 0.5]
    triples = [memory['idx2class'][i] for i in preds]
    #"""
    classes = []
    for triple in triples:
        if triple in memory['train_classes']:
            classes.append(triple)
    #"""
    classes = triples
    return classes

def class_info(class_string):
    if class_string.strip() == '':
        classes = []
    else:
        classes = class_string.strip().split(';')
        classes = [cls.strip() for cls in classes]
    return classes

def valid_on_epoch(epoch, model, filename):
    model.eval()

    start = time.time()
    score = Fscore(tag='valid')

    with codecs.open(filename, 'r') as f:
        lines = f.readlines()
    lines = [line.split('\t<=>\t') for line in lines]

    for (utterance, class_string) in lines:
        gold_classes = class_info(class_string)
        if utterance.strip() == '':
            pred_classes = []
        else:
            lis = utterance.strip().split()
            lis = [''.join(word.strip().split("'")) for word in lis]
            if len(lis) == 0:
                pred_classes = []
            else:
                pred_classes = decode_utterance(model, utterance)
        score.update_tp_fp_fn(pred_classes, gold_classes)

    fscore = score.output_f_score(logger, epoch)
    print('Elapsed time: {}'.format(time.time() - start))

    return fscore

class EarlyStopping(object):

    def __init__(self, mode='min', min_delta=0., patience=5):

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
            self.is_better = lambda a, best: a < best - min_delta
            #self.is_better = lambda a, best: a < best + min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta
            #self.is_better = lambda a, best: a > best - min_delta

def train():

    tt_train_file = data_dir + 'dstc3.seed'
    tt_valid_file = data_dir + 'dstc3.valid'

    dataiter = DataIter(data_dir+tt_train_file, memory, cuda, True)

    model = LE(len(memory['word2idx']), len(memory['class2idx']))

    emb = read_emb(memory['word2idx'])
    model.word_emb.init_weight_from_pre_emb(emb, True)

    labelemb = memory['labelemb']
    model.lin2.weight.data = labelemb
    model.lin2.weight.requires_grad = False

    if load_model is not None:
        chkpt = torch.load(load_model, map_location=lambda storage, log: storage)
        model.load_state_dict(chkpt)
        print('Load model from {}'.format(load_model))
    if cuda:
        model = model.cuda()
    print(model)
    tally_parameters(model)

    criterion = MaskedBCELoss(cuda)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    early_stop = EarlyStopping(mode='max')

    for epoch in range(epochs):
        train_on_epoch(epoch, model, dataiter, criterion, optimizer)
        metric = valid_on_epoch(epoch, model, data_dir+tt_valid_file)

        flag = early_stop(epoch, metric, model.state_dict())
        best_epoch = early_stop.best_epoch
        best_metric = early_stop.best_metric
        best_model_state = early_stop.best_model_state

        if flag:
            print('Early stopping at Epoch {}'.format(best_epoch))
            print('Best metric is {:6.2f}'.format(best_metric))
            torch.save(best_model_state, save_path)
            print('Drop a checkpoint at {}'.format(save_path))
            break
    if not flag:
        print('Finally stopping at Epoch {}'.format(best_epoch))
        print('Best metric is {:6.2f}'.format(best_metric))
        torch.save(best_model_state, save_path)
        print('Drop a checkpoint at {}'.format(save_path))

def test():

    sessions = json.loads(open(test_json).read())['sessions']

    # Model loading
    model = LE(len(memory['word2idx']), len(memory['class2idx']))
    chkpt = torch.load(save_path, map_location = lambda storage, log: storage)
    model.load_state_dict(chkpt)
    if cuda:
        model = model.cuda()
    print(model)
    # ====== *********************** ================
    model.eval()
    # ===============================================
    # decode
    print('Decoding ...')
    decode_sessions = {'sessions': []}
    for session in sessions:
        n_session = {}
        n_session['session-id'] = session['session-id']
        n_session['turns'] = []
        for turn in session['turns']:

            asr_hyps = turn['asr-hyps']
            asr_hyp = asr_hyps[0]['asr-hyp']
            if asr_hyp.strip() == '':
                pred_classes = []
            else:
                lis = asr_hyp.strip().split()
                lis = [''.join(word.strip().split("'")) for word in lis]
                if len(lis) == 0:
                    pred_classes = []
                else:
                    pred_classes = decode_utterance(model, asr_hyp)

            classes = pred_classes

            slu_hyp = [slot2dic(string) for string in classes]

            n_session['turns'].append(
                {
                    'asr-hyps': asr_hyps,
                    'slu-hyps': [{'slu-hyp': slu_hyp, 'score': 1.0}]
                    }
                )

        decode_sessions['sessions'].append(n_session)
    string = json.dumps(decode_sessions, sort_keys=True, indent=4, separators=(',', ':'))
    with open(save_decode, 'w') as f:
        f.write(string)

    print('Decode results saved in {}'.format(save_decode))

if __name__ == '__main__':
    test()

