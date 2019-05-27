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

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(install_path)

root_dir = os.path.join(install_path, 'exps/')
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

from xslu.utils import read_emb, make_logger, process_sent, process_word
from xslu.optim import Optim

from model import DAModel
from dataloader import DADataset
from decode import decode_utterance
from trainer import DATrainer

def model_opts(parser):

    parser.add_argument('-emb_dim', default=100, type=int,
                       help="word embedding dimension")
    parser.add_argument('-hid_dim', default=128, type=int,
                       help="hidden vector dimension")
    parser.add_argument('-dropout', default=0.5, type=float,
                       help="the dropout value")

def train_opts(parser):
    # Data options

    parser.add_argument('-experiment', required=True,
                       help="Root path for saving results, models and logs")
    parser.add_argument('-data_root', required=True,
                       help="Path prefix to the train and valid and class")
    parser.add_argument('-memory_path', required=True,
                       help="Path to the memory")
    parser.add_argument('-train_file', required=True,
                       help="train file name")
    parser.add_argument('-valid_file', required=True,
                       help="valid file name")

    parser.add_argument('-save_model', default='best.pt',
                       help="Saved model filename")

    parser.add_argument('-load_model', default=None,
                       help="Pre-trained model filename")

    parser.add_argument('-enlarge_word_vocab', action='store_true',
                       help="whether to use enlarged word vocab")

    parser.add_argument('-load_word_emb', action='store_true',
                       help="whether to load pretrained word embeddings")
    parser.add_argument('-fix_word_emb', action='store_true',
                       help="whether to fix pretrained word embeddings")

    parser.add_argument('-deviceid', default=0, type=int,
                       help="device id to run, -1 for cpus")

    parser.add_argument('-batch_size', default=10, type=int,
                       help="batch size")
    parser.add_argument('-epochs', default=100, type=int,
                       help="epochs")

    parser.add_argument('-optim', default='adam', type=str,
                       help="optimizer")
    parser.add_argument('-lr', default=0.001, type=float,
                       help="learning rate")
    parser.add_argument('-max_norm', default=5, type=float,
                       help="threshold of gradient clipping (2 norm), < 0 for no clipping")

    parser.add_argument('-seed', default=3435,
                       help='random seed')


def test_opts(parser):
    # Data options

    parser.add_argument('-class_file', default='class.all', type=str,
                       help="class file")
    parser.add_argument('-test_file', default='test.json', type=str,
                       help="if mode is test, preprocessed test json file; else, normal test file as train file")
    parser.add_argument('-save_file', default='decode.json', type=str,
                       help="Path to the file of saving decoded results in test mode, error results in error mode")
    parser.add_argument('-nbest', default=1, type=int,
                       help="nbest in beam search of decode")

def parse_args():
    parser = argparse.ArgumentParser(
            description='Program Options',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-mode', required=True, type=str,
            help="run mode: train, test, gen")

    model_opts(parser)
    train_opts(parser)
    test_opts(parser)

    opt = parser.parse_args()
    print(opt)

    opt.memory = torch.load(opt.data_root + opt.memory_path)
    if opt.enlarge_word_vocab:
        opt.memory['enc2idx'] = opt.memory['word2idx_w_glove']
    else:
        opt.memory['enc2idx'] = opt.memory['word2idx']
    opt.memory['dec2idx'] = opt.memory['word2idx']
    opt.memory['idx2dec'] = {v:k for k,v in opt.memory['dec2idx'].items()}

    opt.enc_word_vocab_size = len(opt.memory['enc2idx'])
    opt.dec_word_vocab_size = len(opt.memory['dec2idx'])

    if opt.fix_word_emb:
        assert opt.load_word_emb is True

    if opt.deviceid >= 0:
        torch.cuda.set_device(opt.deviceid)
        opt.cuda = True
    else:
        opt.cuda = False

    # fix random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    return opt

def make_model(opt):
    model = DAModel(opt.enc_word_vocab_size, opt.dec_word_vocab_size,
            opt.emb_dim, opt.hid_dim, opt.dropout
            )
    return model

def train(opt):

    opt.experiment = os.path.join(root_dir, opt.experiment)
    if not os.path.exists(opt.experiment):
        os.makedirs(opt.experiment)

    opt.save_model = os.path.join(opt.experiment, opt.save_model)
    if opt.load_model == 'None':
        opt.load_model = None
    if opt.load_model is not None:
        opt.load_model = os.path.join(opt.experiment, opt.load_model)
    opt.log_path = os.path.join(opt.experiment, 'log.train')
    opt.logger = make_logger(opt.log_path)

    # memory info
    print("encoder word2idx number: {}".format(opt.enc_word_vocab_size))
    print("decoder word2idx number: {}".format(opt.dec_word_vocab_size))

    # Model definition
    model = make_model(opt)

    if opt.load_word_emb:
        if opt.enlarge_word_vocab:
            enc_emb = opt.memory['word2idx_w_glove_emb']
        else:
            enc_emb = opt.memory['word2idx_emb']
        dec_emb = opt.memory['word2idx_emb']
        model.enc_word_emb.init_weight_from_pre_emb(enc_emb, opt.fix_word_emb)
        model.dec_word_emb.init_weight_from_pre_emb(dec_emb, opt.fix_word_emb)

    if opt.enc_word_vocab_size == opt.dec_word_vocab_size:
        model.dec_word_emb.embedding.weight.data = model.enc_word_emb.embedding.weight.data
        model.dec_word_emb.embedding.weight.requires_grad = model.enc_word_emb.embedding.weight.requires_grad

    if opt.load_model is not None:
        chkpt = torch.load(opt.load_model, map_location = lambda storage, log: storage)
        model.load_state_dict(chkpt)
    if opt.cuda:
        model = model.cuda()
    print(model)

    # optimizer details
    optimizer = Optim(opt.optim, opt.lr, max_grad_norm=opt.max_norm)
    optimizer.set_parameters(model.named_parameters())
    print("training parameters number: {}".format(len(optimizer.params)))

    nll_criterion = nn.NLLLoss(reduction='sum')
    if opt.cuda:
        nll_criterion = nll_criterion.cuda()

    # training procedure
    train_iter = DADataset(opt.data_root + opt.train_file, opt.memory, opt.cuda, True)
    valid_iter = DADataset(opt.data_root + opt.valid_file, opt.memory, opt.cuda, False)

    trainer = DATrainer(model, nll_criterion, optimizer, opt.logger, cuda=opt.cuda)
    trainer.train(opt.epochs, opt.batch_size, train_iter, valid_iter, opt.save_model)

def test(opt):

    opt.experiment = os.path.join(root_dir, opt.experiment)
    opt.load_chkpt = os.path.join(opt.experiment, opt.save_model)
    opt.save_file = os.path.join(opt.experiment, opt.save_file)
    opt.test_file = os.path.join(opt.data_root, opt.test_file)

    # Model loading
    model = make_model(opt)
    chkpt = torch.load(opt.load_chkpt, map_location = lambda storage, log: storage)
    model.load_state_dict(chkpt)
    if opt.deviceid >= 0:
        model = model.cuda()
    print(model)
    # ====== *********************** ================
    model.eval()
    # ===============================================
    # decode
    print('Decoding ...')
    with codecs.open(opt.test_file, 'r') as f:
        lines = f.readlines()
    res = []
    for line in lines:
        lis = line.split('\t<=>\t')
        utterance = lis[0]
        triple = lis[1]
        class_string = lis[2].strip()
        pred_utterances = decode_utterance(model, class_string, opt.memory, opt.cuda, opt.nbest)
        for i in range(len(pred_utterances)):
            res.append((pred_utterances[i], triple, class_string))

    with open(opt.save_file, 'w') as f:
        for (utterance, triple, class_string) in res:
            f.write('{}\t<=>\t{}\t<=>\t{}\n'.format(utterance, triple, class_string))

    print('Decode results saved in {}'.format(opt.save_file))

def gen(opt):

    opt.experiment = os.path.join(root_dir, opt.experiment)
    opt.load_chkpt = os.path.join(opt.experiment, opt.save_model)
    opt.save_file = os.path.join(opt.data_root, opt.save_file)
    opt.class_file = os.path.join(opt.data_root, opt.class_file)

    dir_name = os.path.abspath(os.path.dirname(opt.save_file))
    print(dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Model loading
    model = make_model(opt)
    chkpt = torch.load(opt.load_chkpt, map_location = lambda storage, log: storage)
    model.load_state_dict(chkpt)
    if opt.deviceid >= 0:
        model = model.cuda()
    print(model)
    # ====== *********************** ================
    model.eval()
    # ===============================================
    # decode
    print('Decoding ...')
    classes = json.loads(open(opt.class_file, 'r').read())
    res = []
    for cls in classes:
        triple = cls[0].strip()
        class_string = cls[1].strip()
        value_dict = cls[2]
        pred_utterances = decode_utterance(model, class_string, opt.memory, opt.cuda, opt.nbest)
        for i in range(len(pred_utterances)):
            utt = pred_utterances[i]
            trp = triple
            for key in value_dict:
                utt = utt.replace(key, value_dict[key])
                trp = trp.replace(key, value_dict[key])
            res.append((utt, trp))

    #res = list(set(res))
    with open(opt.save_file, 'w') as f:
        for (utterance, class_string) in res:
            f.write('{}\t<=>\t{}\n'.format(utterance, class_string))

    print('Decode results saved in {}'.format(opt.save_file))

if __name__ == '__main__':
    opt = parse_args()
    if opt.mode == 'train':
        train(opt)
    elif opt.mode == 'test':
        test(opt)
    elif opt.mode == 'gen':
        gen(opt)
    else:
        raise ValueError("unsupported type of mode {}".format(opt.mode))
