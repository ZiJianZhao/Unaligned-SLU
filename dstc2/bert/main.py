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

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam

# **************************** file paths ******************************
bert_root = '/slfs1/users/zjz17/NLPData/bert/bert-base-uncased/'
bert_vocab_path = bert_root + 'bert-base-uncased-vocab.txt'

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(install_path)

root_dir = os.path.join(install_path, 'exps/')
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

from xslu.utils import make_logger
from xslu.optim import Optim
from text.text import build_class_vocab
from text.dstc2 import slot2dic
from model import BertSTC
from dataloader import BertIter4STC
from trainer import BertTrainer4STC

def model_opts(parser):

    parser.add_argument('-bert_mode', default='first', type=str,
            help="how to deal with the hiddens from the bert: first, avg, max")
    parser.add_argument('-fix_bert', action='store_true',
            help="whether to fix the parameters of the bert")
    parser.add_argument('-hid_dim', default=768, type=int,
                       help="hidden dimension")

def train_opts(parser):
    # Data options

    parser.add_argument('-experiment', required=True,
                       help="Root path for saving results, models and logs")
    parser.add_argument('-data_root', required=True,
                       help="Path prefix to the train and valid and class")
    parser.add_argument('-save_model', default='best.pt',
                       help="Saved model filename")

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

    parser.add_argument('-test_json', default='test.json', type=str,
                       help="preprocessed test json file")
    parser.add_argument('-save_decode', default='decode.json', type=str,
                       help="Path to the file of saving decoded results")
    parser.add_argument('-load_chkpt', default=None, type=str,
                       help="Path to the checkpoint file to be loaded")

def parse_args():
    parser = argparse.ArgumentParser(
            description='Program Options',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('-mode', default='train', type=str,
            help="run mode: train, test, error")

    model_opts(parser)
    train_opts(parser)
    test_opts(parser)

    opt = parser.parse_args()
    print(opt)

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

def make_bert():
    tokenizer = BertTokenizer.from_pretrained(bert_vocab_path)
    model = BertModel.from_pretrained(bert_root)
    return tokenizer, model

def make_model(opt, bert_model):
    model = BertSTC(opt.bert_mode, bert_model, opt.hid_dim, opt.class_size)
    if opt.cuda:
        model = model.cuda()
    return model

def train(opt):

    # basics definition
    opt.experiment = os.path.join(root_dir, opt.experiment)
    if not os.path.exists(opt.experiment):
        os.makedirs(opt.experiment)
    opt.save_model = os.path.join(opt.experiment, opt.save_model)
    opt.log_path = os.path.join(opt.experiment, 'log.train')
    opt.logger = make_logger(opt.log_path)

    bert_tokenizer, bert_model = make_bert()

    # dataIter definition
    class2idx = build_class_vocab(opt.data_root+'class.all')
    opt.class_size = len(class2idx)
    train_iter = BertIter4STC(opt.data_root+'train', bert_tokenizer, class2idx, 
            opt.batch_size, opt.cuda, True)
    valid_iter = BertIter4STC(opt.data_root+'valid', bert_tokenizer, class2idx, 
            opt.batch_size, opt.cuda, False)

    # model definition
    model = make_model(opt, bert_model)

    # criterion definition
    criterion = nn.BCELoss(reduction='sum')
    if opt.cuda:
        criterion = criterion.cuda()

    # optimizer definition
    if opt.fix_bert:
        for (name, parameter) in model.bert.named_parameters():
            parameter.requires_grad = False
    
    if opt.optim == 'bert':
        params = list(filter(lambda x: x[1].requires_grad == True, model.named_parameters()))
        print('Trainable parameter number: {}'.format(len(params)))
        print('Trainer: bert')
        no_decay = ['bias', 'gamma', 'beta']
        grouped_params = [
            {'params': [p for n,p in params if n not in no_decay], 'weight_decay_rate': 0.01},
            {'params': [p for n,p in params if n in no_decay], 'weight_decay_rate': 0.0}
        ]
        optimizer = BertAdam(grouped_params, opt.lr, 
                warmup=0.1, t_total=len(train_iter) * opt.epochs)
    else:
        optimizer = Optim(opt.optim, opt.lr, max_grad_norm=opt.max_norm)
        optimizer.set_parameters(model.named_parameters())
        print('Trainable parameter number: {}'.format(len(optimizer.params)))

    # training procedure
    trainer = BertTrainer4STC(model, criterion, optimizer, opt.logger)
    trainer.train(opt.epochs, train_iter, valid_iter, opt.save_model)


def test(opt):

    opt.experiment = os.path.join(root_dir, opt.experiment)
    opt.load_chkpt = os.path.join(opt.experiment, opt.load_chkpt)
    opt.save_decode = os.path.join(opt.experiment, opt.save_decode)
    opt.test_json = os.path.join(opt.data_root, opt.test_json)

    bert_tokenizer, bert_model = make_bert()
    class2idx = build_class_vocab(opt.data_root+'class.all')
    idx2class = {v:k for k,v in class2idx.items()}
    opt.class_size = len(class2idx)
    model = make_model(opt, bert_model)
    
    chkpt = torch.load(opt.load_chkpt, map_location=lambda storage, log: storage)
    model.load_state_dict(chkpt)

    # =======================================
    model.eval()
    # =======================================

    sessions = json.loads(open(opt.test_json).read())['sessions']
    print('Decoding ...')
    decode_sessions = {'sessions': []}
    for session in sessions:
        n_session = {}
        n_session['session-id'] = session['session-id']
        n_session['turns'] = []
        for turn in session['turns']:

            asr_hyps = turn['asr-hyps']
            sent = asr_hyps[0]['asr-hyp']
            tokens = bert_tokenizer.tokenize(sent)

            if len(tokens) == 0:
                slu_hyp = []
            else:
                tokens = ["[CLS]"] + tokens
                sent_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
                input_ids = torch.from_numpy(np.asarray(sent_ids, dtype='int64')).view(1, -1)
                if opt.cuda:
                    input_ids = input_ids.cuda()
                probs = model(input_ids, None, None)
                scores = probs.data.cpu().view(-1,).numpy()
                pred_classes = [i for i,p in enumerate(scores) if p > 0.5]
                classes = [idx2class[i] for i in pred_classes]
                slu_hyp = [slot2dic(string) for string in classes]

            n_session['turns'].append(
                {
                    'asr-hyps': asr_hyps,
                    'slu-hyps': [{'slu-hyp': slu_hyp, 'score': 1.0}]
                    }
                )

        decode_sessions['sessions'].append(n_session)

    string = json.dumps(decode_sessions, sort_keys=True, indent=4, separators=(',', ':'))
    with open(opt.save_decode, 'w') as f:
        f.write(string)
    print('Decode results saved in {}'.format(opt.save_decode))

if __name__ == '__main__':
    opt = parse_args()
    if opt.mode == 'train':
        train(opt)
    elif opt.mode == 'test':
        test(opt)
    else:
        raise ValueError("unsupported type of mode {}".format(opt.mode))

