# -*- codind: utf-8 -*-

import os, sys, random, argparse
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

from xslu.utils import make_logger, tally_parameters
from xslu.optim import Optim
import xslu.Constants as Constants

from model import make_model
from dataloader import DataLoader
from trainer import Trainer


def train_opts(parser):
    # Data options

    parser.add_argument('-data_dir', required=True,
                       help="data dir")
    parser.add_argument('-train_file', required=True,
                       help="train file")
    parser.add_argument('-valid_file', required=True,
                       help="valid file")
    parser.add_argument('-vocab_file', required=True,
                       help="vocab file")

    parser.add_argument('-experiment', default='dual/pointer/',
                       help="Saved model filename")
    parser.add_argument('-save_model', default='best.pt',
                       help="Saved model filename")

    parser.add_argument('-gpuid', default=0, type=int,
                       help="gpu id to run")
    parser.add_argument('-batch_size', default=10, type=int,
                       help="batch size")
    parser.add_argument('-epochs', default=20, type=int,
                       help="epochs")
    parser.add_argument('-seed', default=3435,
                       help='random seed')

def parse_args():
    parser = argparse.ArgumentParser(
            description='Train Options',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    train_opts(parser)
    opt = parser.parse_args()

    # fix random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    if opt.gpuid >= 0:
        torch.cuda.set_device(opt.gpuid)
        opt.cuda = True
    else:
        opt.cuda = False

    opt.train_file = os.path.join(opt.data_dir, opt.train_file)
    opt.valid_file = os.path.join(opt.data_dir, opt.valid_file)
    opt.vocab_file = os.path.join(opt.data_dir, opt.vocab_file)

    opt.word2idx = torch.load(opt.vocab_file)

    return opt

def train():

    # Initialization
    opt = parse_args()
    opt.experiment = os.path.join(root_dir, opt.experiment)
    if not os.path.exists(opt.experiment):
        os.makedirs(opt.experiment)
    opt.save_model = os.path.join(opt.experiment, opt.save_model)
    opt.log_path = os.path.join(opt.experiment, 'log.train')
    opt.logger = make_logger(opt.log_path)

    # Data iterator definition
    train_iter = DataLoader(opt.train_file, opt.word2idx, opt.batch_size, opt.cuda, epoch_shuffle=True)
    valid_iter = DataLoader(opt.valid_file, opt.word2idx, opt.batch_size, opt.cuda, epoch_shuffle=False)

    # Model definition
    model = make_model(len(opt.word2idx))
    if opt.gpuid >= 0:
        model = model.cuda()
    #print(model)
    tally_parameters(model)

    optimizer = Optim('Adam', 0.001, max_grad_norm=5)
    optimizer.set_parameters(model.named_parameters())

    criterion = nn.NLLLoss(reduction='sum')
    if opt.gpuid >= 0:
        criterion = criterion.cuda()

    trainer = Trainer(model, criterion, optimizer, opt.logger, opt.cuda)
    trainer.train(opt.epochs, train_iter, valid_iter, opt.save_model)

if __name__ == '__main__':
    train()
