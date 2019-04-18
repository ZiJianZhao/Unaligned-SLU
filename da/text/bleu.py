#-*- coding:utf-8 -*-

import re, os, sys, argparse, logging, collections
import codecs
from collections import namedtuple, defaultdict
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
import json


def read_ref_file(ref_file):
    with open(ref_file, 'r') as f:
        lines = f.readlines()
    dic = defaultdict(list)
    for line in lines:
        lis = line.split('\t<=>\t')
        utterance = lis[0].strip()
        triple = lis[1].strip()
        if triple == '':
            continue
        else:
            dic[triple].append(utterance)
    for key in dic:
        dic[key] = list(set(dic[key]))

    return dic

def cal_bleu(hyp_file, ref_file):
    with open(hyp_file, 'r') as f:
        lines = f.readlines()
    dic = read_ref_file(ref_file)

    hypothesis = []
    references = []
    for line in lines:
        line_list = line.split('\t<=>\t')
        #if len(line_list) != 2:
        #    continue
        utterance = line_list[0].strip()
        triple = line_list[1].strip()
        if triple == '':
            continue
        else:
            hypoth = utterance.split()
            hypothesis.append(hypoth)
            refs = [utt.strip().split() for utt in dic[triple]]
            references.append(refs)
    print(len(references))
    bleus = []
    for i in range(1, 5):
        weight = [0, 0, 0, 0]
        weight[i-1] = 1
        weight = tuple(weight)
        bleu = corpus_bleu(references, hypothesis,
            weights=weight,
            smoothing_function=SmoothingFunction().method2
        )
        bleus.append(bleu)
        print('test-blue-{}: {}'.format(i, bleu))
    bleu = corpus_bleu(references, hypothesis,
            weights=(0.33, 0.33, 0.33),
            smoothing_function=SmoothingFunction().method2
    )
    print('test-blue-3 average: {}'.format(bleu))
    bleus.append(bleu)
    bleu = corpus_bleu(references, hypothesis,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method2
    )
    bleus.append(bleu)
    print('test-blue-4 average: {}'.format(bleu))
    return bleus

def key_word_rate(filename, slot):
    with open(filename, 'r') as f:
        lines = f.readlines()
    total_num = 0.
    key_num = 0.
    for line in lines:
        line_list = line.split('\t<=>\t')
        if line_list[1].strip() != '':
            triples = line_list[1].strip().split(';')
            for triple in triples:
                lis = triple.strip().split('-', 2)
                if (len(lis) == 3) and (lis[1] in slot):
                    total_num += 1
                    if lis[2] in line_list[0]:
                        key_num += 1
    print('Key word rate: {}/{}, {}'.format(key_num, total_num, key_num/total_num))


if __name__ == '__main__':
    root_dir = '/slfs1/users/zjz17/SLU/'
    ref_file = root_dir + 'dstc2-slu/manual-da/new.value.test'
    hyp_file = root_dir + 'Unaligned-SLU/exps/dstc2-da/decode.gen'
    cal_bleu(hyp_file, ref_file)

    key_word_rate(hyp_file, ['food', 'name'])

