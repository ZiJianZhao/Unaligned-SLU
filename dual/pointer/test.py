# -*- codind: utf-8 -*-

import os, sys, random, argparse
import math
import json

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


from dual.text.process import trp_reverse_process
from text.dstc2 import slot2dic

from model import make_model
from translator import translate

def test_opts(parser):
    # Data options

    parser.add_argument('-experiment', default='dual/pointer/',
                       help="Saved model filename")
    parser.add_argument('-test_file', required=True,
                       help="filename of the source text to be translated")
    parser.add_argument('-vocab_file', required=True,
                       help="filename of the vocab")
    parser.add_argument('-chkpt', required=True,
                       help="Load model filename")
    parser.add_argument('-out_file', default='out.txt',
                       help="filename of translated text")
    parser.add_argument('-out_json', default='decode.json',
                       help="filename of translated json file")
    parser.add_argument('-gpuid', default=0, type=int,
                       help="gpu id to run")
    parser.add_argument('-n_best', default=1, type=int,
                       help="n_best")
    parser.add_argument('-max_length', default=80, type=int,
                       help="max length")
    parser.add_argument('-seed', default=3435,
                       help='random seed')

def parse_args():
    parser = argparse.ArgumentParser(
            description='Translate Options',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    test_opts(parser)
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

    opt.word2idx = torch.load(opt.vocab_file)
    opt.idx2word = {v:k for k,v in opt.word2idx.items()}

    return opt

def test():

    opt = parse_args()

    opt.experiment = os.path.join(root_dir, opt.experiment)
    opt.chkpt = os.path.join(opt.experiment, opt.chkpt)
    opt.out_file = os.path.join(opt.experiment, opt.out_file)
    opt.out_json = os.path.join(opt.experiment, opt.out_json)

    sessions = json.loads(open(opt.test_file).read())['sessions']

    # Model loading
    model = make_model(len(opt.word2idx))
    chkpt = torch.load(opt.chkpt, map_location = lambda storage, log: storage)
    model.load_state_dict(chkpt)
    if opt.gpuid >= 0:
        model = model.cuda()

    # ====== *********************** ================
    model.eval()
    # ===============================================

    # decode
    results = []
    print('Decoding ...')
    decode_sessions = {'sessions': []}
    for session in sessions:
        n_session = {}
        n_session['session-id'] = session['session-id']
        n_session['turns'] = []
        for turn in session['turns']:

            asr_hyps = turn['asr-hyps']
            asr_hyp = asr_hyps[0]['asr-hyp']

            string = translate(model, asr_hyp, opt.word2idx, opt.idx2word, opt.cuda)
            if string == '':
                classes = []
            else:
                classes = trp_reverse_process(string, 1)
                results.append((asr_hyp, string))

            slu_hyp = [slot2dic(string) for string in classes]

            n_session['turns'].append(
                {
                    'asr-hyps': asr_hyps,
                    'slu-hyps': [{'slu-hyp': slu_hyp, 'score': 1.0}]
                    }
                )

        decode_sessions['sessions'].append(n_session)
    string = json.dumps(decode_sessions, sort_keys=True, indent=4, separators=(',', ':'))
    with open(opt.out_json, 'w') as f:
        f.write(string)
    print('Decode results saved in {}'.format(opt.save_file))
    with open(opt.out_file, 'w') as f:
        for (enc, dec) in results:
            f.write('{}\t<=>\t{}\n'.format(enc.strip(), dec.strip()))

if __name__ == '__main__':
    test()
