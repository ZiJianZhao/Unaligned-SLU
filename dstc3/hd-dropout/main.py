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

from text.dstc2 import slot2dic, process_sent
from xslu.utils import read_emb, make_logger
from xslu.optim import Optim

from model import SLUSystem
from dataloader import SLUDataset, ActDataset, SlotDataset, ValueDataset
from decode import decode_slu, decode_act, decode_slot, decode_value
from trainer import ActTrainer, SLUTrainer, SlotTrainer, ValueTrainer

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

    parser.add_argument('-load_word_emb', action='store_true',
                       help="whether to load pretrained word embeddings")
    parser.add_argument('-fix_word_emb', action='store_true',
                       help="whether to fix pretrained word embeddings")

    parser.add_argument('-load_class_emb', action='store_true',
                       help="whether to load pretrained class embeddings")
    parser.add_argument('-fix_class_emb', action='store_true',
                       help="whether to fix pretrained class embeddings")

    parser.add_argument('-share_param', action='store_true',
                       help="whether to share some parameters")

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

    parser.add_argument('-test_file', default='test.json', type=str,
                       help="if mode is test, preprocessed test json file; else, normal test file as train file")
    parser.add_argument('-save_file', default='decode.json', type=str,
                       help="Path to the file of saving decoded results in test mode, error results in error mode")

def parse_args():
    parser = argparse.ArgumentParser(
            description='Program Options',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-task', required=True, type=str,
            help="run task: slu, act, slot, value.")
    parser.add_argument('-mode', required=True, type=str,
            help="run mode: train, test, error, stat")

    model_opts(parser)
    train_opts(parser)
    test_opts(parser)

    opt = parser.parse_args()
    print(opt)

    if opt.task not in ['slu', 'act', 'slot', 'value']:
        raise Exception('Unknown task.')

    opt.memory = torch.load(opt.data_root + opt.memory_path)
    if opt.load_word_emb:
        opt.memory['enc2idx'] = opt.memory['word2idx_w_glove']
    else:
        opt.memory['enc2idx'] = opt.memory['word2idx']
    opt.memory['dec2idx'] = opt.memory['value2idx']
    opt.memory['idx2dec'] = {v:k for k,v in opt.memory['dec2idx'].items()}

    opt.enc_word_vocab_size = len(opt.memory['enc2idx'])
    opt.dec_word_vocab_size = len(opt.memory['dec2idx'])
    opt.act_vocab_size = len(opt.memory['act2idx'])
    opt.slot_vocab_size = len(opt.memory['slot2idx'])

    if opt.fix_word_emb:
        assert opt.load_word_emb is True

    if opt.fix_class_emb:
        assert opt.load_class_emb is True

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
    model = SLUSystem(opt.enc_word_vocab_size, opt.dec_word_vocab_size,
            opt.act_vocab_size, opt.slot_vocab_size,
            opt.emb_dim, opt.hid_dim, opt.dropout)
    return model

class MaskedBCELoss(nn.Module):

    def __init__(self, cuda):
        super(MaskedBCELoss, self).__init__()
        self.criterion = nn.BCELoss(size_average=False, reduce=False)
        if cuda:
            self.criterion = self.criterion.cuda()

    def forward(self, preds, golds, mask=None):
        preds = preds.view(-1, 1)
        golds = golds.view(-1, 1)
        loss = self.criterion(preds, golds)
        if mask is not None:
            mask = mask.view(-1, 1)
            loss = loss * mask.float()
        loss = torch.sum(loss)
        return loss

def train(opt):

    opt.experiment = os.path.join(root_dir, opt.experiment)
    if not os.path.exists(opt.experiment):
        os.makedirs(opt.experiment)

    opt.save_model = os.path.join(opt.experiment, opt.save_model)
    if opt.load_model is None or opt.load_model == 'none':
        opt.load_model = None
    else:
        opt.load_model = os.path.join(opt.experiment, opt.load_model)
    opt.log_path = os.path.join(opt.experiment, 'log.train')
    opt.logger = make_logger(opt.log_path)

    # memory info
    print("encoder word2idx number: {}".format(opt.enc_word_vocab_size))
    print("decoder word2idx number: {}".format(opt.dec_word_vocab_size))
    print("act2idx number: {}".format(opt.act_vocab_size))
    print("slot2idx number: {}".format(opt.slot_vocab_size))

    # Model definition
    model = make_model(opt)
    if opt.load_word_emb:
        emb = read_emb(opt.memory['enc2idx'])
        model.enc_word_emb.init_weight_from_pre_emb(emb, opt.fix_word_emb)
    if opt.load_class_emb:
        emb = opt.memory['act_emb']
        model.act_emb.init_weight_from_pre_emb(emb, opt.fix_class_emb)
        emb = opt.memory['slot_emb']
        model.slot_emb.init_weight_from_pre_emb(emb, opt.fix_class_emb)
    if opt.share_param:
        #model.value_decoder.outlin.weight.data = model.word_emb.embedding.weight.data
        #model.value_decoder.outlin.weight.requires_grad = model.word_emb.embedding.weight.requires_grad
        model.act_stc.lin.weight.data = model.act_emb.embedding.weight.data
        model.act_stc.lin.weight.requires_grad = model.act_emb.embedding.weight.requires_grad
        model.slot_stc.lin.weight.data = model.slot_emb.embedding.weight.data
        model.slot_stc.lin.weight.requires_grad = model.slot_emb.embedding.weight.requires_grad

    if opt.load_model is not None:
        chkpt = torch.load(opt.load_model, map_location = lambda storage, log: storage)
        model.load_state_dict(chkpt)
        print('Load model from {}'.format(opt.load_model))
    if opt.cuda:
        model = model.cuda()
    print(model)

    # optimizer details
    optimizer = Optim(opt.optim, opt.lr, max_grad_norm=opt.max_norm)
    optimizer.set_parameters(model.named_parameters())
    print("training parameters number: {}".format(len(optimizer.params)))

    # loss definition
    #stc_criterion = MaskedBCELoss(opt.cuda)
    stc_criterion = nn.BCELoss(reduction='sum')
    nll_criterion = nn.NLLLoss(reduction='sum')
    if opt.cuda:
        stc_criterion = stc_criterion.cuda()
        nll_criterion = nll_criterion.cuda()

    # training procedure
    if opt.task == 'slu':
        data_iter = SLUDataset(opt.data_root + opt.train_file, opt.memory, opt.cuda, True)
        trainer = SLUTrainer(model, (stc_criterion, nll_criterion), optimizer, opt.logger, cuda=opt.cuda)
    elif opt.task == 'act':
        data_iter = ActDataset(opt.data_root + opt.train_file, opt.memory, opt.cuda, True)
        trainer = ActTrainer(model, stc_criterion, optimizer, opt.logger, cuda=opt.cuda)
    elif opt.task == 'slot':
        data_iter = SlotDataset(opt.data_root + opt.train_file, opt.memory, opt.cuda, True)
        trainer = SlotTrainer(model, stc_criterion, optimizer, opt.logger, cuda=opt.cuda)
    elif opt.task == 'value':
        data_iter = ValueDataset(opt.data_root + opt.train_file, opt.memory, opt.cuda, True)
        trainer = ValueTrainer(model, nll_criterion, optimizer, opt.logger, cuda=opt.cuda)

    trainer.train(opt.epochs, opt.batch_size, opt.memory, data_iter, opt.data_root+opt.valid_file, opt.save_model)

def test(opt):

    opt.experiment = os.path.join(root_dir, opt.experiment)
    opt.load_chkpt = os.path.join(opt.experiment, opt.save_model)
    opt.save_file = os.path.join(opt.experiment, opt.save_file)
    opt.test_file = os.path.join(opt.data_root, opt.test_file)

    sessions = json.loads(open(opt.test_file).read())['sessions']

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
    decode_sessions = {'sessions': []}
    for session in sessions:
        n_session = {}
        n_session['session-id'] = session['session-id']
        n_session['turns'] = []
        for turn in session['turns']:

            asr_hyps = turn['asr-hyps']
            asr_hyp = asr_hyps[0]['asr-hyp']

            classes = decode_slu(model, asr_hyp, opt.memory, opt.cuda)

            slu_hyp = [slot2dic(string) for string in classes]

            n_session['turns'].append(
                {
                    'asr-hyps': asr_hyps,
                    'slu-hyps': [{'slu-hyp': slu_hyp, 'score': 1.0}]
                    }
                )

        decode_sessions['sessions'].append(n_session)
    string = json.dumps(decode_sessions, sort_keys=True, indent=4, separators=(',', ':'))
    with open(opt.save_file, 'w') as f:
        f.write(string)

    print('Decode results saved in {}'.format(opt.save_file))

def error(opt):

    opt.experiment = os.path.join(root_dir, opt.experiment)
    opt.load_chkpt = os.path.join(opt.experiment, opt.save_model)
    opt.test_file = os.path.join(opt.data_root, opt.test_file)
    opt.save_file = os.path.join(opt.experiment, 'error.info')

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
    g = open(opt.save_file, 'w')
    if opt.task == 'act':
        lines = ActDataset.read_file(opt.test_file)
    elif opt.task == 'slot':
        lines = SlotDataset.read_file(opt.test_file)
    elif opt.task == 'value':
        lines = ValueDataset.read_file(opt.test_file)
    elif opt.task == 'slu':
        lines = SLUDataset.read_file(opt.test_file)

    for (utterance, class_string) in lines:
        if opt.task == 'act':
            gold_classes = ActDataset.class_info(class_string)
            pred_classes = decode_act(model, utterance, opt.memory, opt.cuda)
        elif opt.task == 'slot':
            gold_classes = SlotDataset.class_info(class_string)
            pred_classes = decode_slot(model, utterance, class_string, opt.memory, opt.cuda)
        elif opt.task == 'value':
            gold_classes = ValueDataset.class_info(class_string)
            pred_classes = decode_value(model, utterance, class_string, opt.memory, opt.cuda)
        elif opt.task == 'slu':
            gold_classes = SLUDataset.class_info(class_string)
            pred_classes = decode_slu(model, utterance, opt.memory, opt.cuda)
        gold_class = ';'.join(sorted(gold_classes))
        pred_class = ';'.join(sorted(pred_classes))
        if gold_class != pred_class:
            g.write('{}\t<=>\t{}\t<=>\t{}\n'.format(utterance, gold_class, pred_class))
    g.close()
    print('Decode results saved in {}'.format(opt.save_file))

def stat(opt):

    opt.experiment = os.path.join(root_dir, opt.experiment)
    opt.load_chkpt = os.path.join(opt.experiment, opt.save_model)
    opt.test_file = os.path.join(opt.data_root, opt.test_file)
    opt.log_path = os.path.join(opt.experiment, 'stat.info')
    opt.logger = make_logger(opt.log_path)

    # Model loading
    model = make_model(opt)
    chkpt = torch.load(opt.load_chkpt, map_location = lambda storage, log: storage)
    model.load_state_dict(chkpt)
    if opt.cuda:
        model = model.cuda()

    # ====== *********************** ================
    model.eval()
    # ===============================================

    # training procedure
    trainers = []
    trainers.append(ActTrainer(model, None, None, opt.logger, cuda=opt.cuda))
    trainers.append(SlotTrainer(model, None, None, opt.logger, cuda=opt.cuda))
    trainers.append(ValueTrainer(model, None, None, opt.logger, cuda=opt.cuda))
    trainers.append(SLUTrainer(model, (None, None), None, opt.logger, cuda=opt.cuda))

    for trainer in trainers:
        metric = trainer.valid_on_epoch(0, opt.test_file, opt.memory)

    print('Stat results saved in {}'.format(opt.log_path))

if __name__ == '__main__':
    opt = parse_args()
    if opt.mode == 'train':
        train(opt)
    elif opt.mode == 'test':
        test(opt)
    elif opt.mode == 'error':
        error(opt)
    elif opt.mode == 'stat':
        stat(opt)
    else:
        raise ValueError("unsupported type of mode {}".format(opt.mode))
