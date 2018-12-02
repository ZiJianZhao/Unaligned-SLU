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

from xslu.utils import make_logger, read_emb
from xslu.optim import Optim
import xslu.Constants as Constants
from text.text import build_class_vocab
from text.dstc2 import slot2dic, process_sent
from model import RNN2One
from dataloader import OneBestIter4STC
from trainer import OneBestTrainer4STC

def model_opts(parser):

    parser.add_argument('-model_type', default='RNN2One', type=str,
            help="which model to use: RNN2One")

def train_opts(parser):
    # Data options

    parser.add_argument('-experiment', required=True,
                       help="Root path for saving results, models and logs")
    parser.add_argument('-data_root', required=True,
                       help="Path prefix to the train and valid and class")
    parser.add_argument('-save_model', default='best.pt',
                       help="Saved model filename")

    parser.add_argument('-load_emb', action='store_true',
                       help='whether to load pre-trained word embeddings')
    parser.add_argument('-fix_emb', action='store_true',
                       help='whether to fix pre-trained word embeddings')

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

    if opt.fix_emb:
        assert opt.load_emb is True

    opt.memory = torch.load(opt.data_root + 'memory.pt')
    opt.class2idx = opt.memory['class2idx']
    if opt.load_emb:
        opt.word2idx = opt.memory['word2idx_w_glove']
    else:
        opt.word2idx = opt.memory['word2idx']

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
    if opt.model_type == 'RNN2One':
        func = RNN2One
    else:
        raise Exception('Undefined model type!')
    model = func(len(opt.word2idx), len(opt.class2idx))
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

    # dataIter definition
    train_iter = OneBestIter4STC(opt.data_root+'train', opt.word2idx, opt.class2idx, 
            opt.batch_size, opt.cuda, True)
    valid_iter = OneBestIter4STC(opt.data_root+'valid', opt.word2idx, opt.class2idx, 
            opt.batch_size, opt.cuda, False)

    # model definition
    model = make_model(opt)
    if opt.load_emb:
        emb = read_emb(opt.word2idx)
        model.emb.init_weight_from_pre_emb(emb, opt.fix_emb)
    print(model)

    # criterion definition
    criterion = nn.BCELoss(reduction='sum')
    if opt.cuda:
        criterion = criterion.cuda()

    # optimizer definition
    optimizer = Optim(opt.optim, opt.lr, max_grad_norm=opt.max_norm)
    optimizer.set_parameters(model.named_parameters())
    print('Trainable parameter number: {}'.format(len(optimizer.params)))

    # training procedure
    trainer = OneBestTrainer4STC(model, criterion, optimizer, opt.logger)
    trainer.train(opt.epochs, train_iter, valid_iter, opt.save_model)


def test(opt):

    opt.experiment = os.path.join(root_dir, opt.experiment)
    opt.load_chkpt = os.path.join(opt.experiment, opt.load_chkpt)
    opt.save_decode = os.path.join(opt.experiment, opt.save_decode)
    opt.test_json = os.path.join(opt.data_root, opt.test_json)

    idx2class = {v:k for k,v in opt.class2idx.items()}
    model = make_model(opt)
    
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
            tokens = process_sent(sent)

            if len(tokens) == 0:
                slu_hyp = []
            else:
                sent_ids = [opt.word2idx.get(w) if w in opt.word2idx else Constants.UNK for w in tokens]
                datas = torch.from_numpy(np.asarray(sent_ids, dtype='int64')).view(1, -1)
                if opt.cuda:
                    datas = datas.cuda()
                probs = model(datas, None)
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

