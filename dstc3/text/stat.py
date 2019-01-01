# -*- coding: utf-8 -*-

import argparse
import os, sys
import codecs
from collections import Counter
import json
import numpy as np
import torch
import copy

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(install_path)
sys.path.append(install_path)

import xslu.Constants as Constants
from xslu.utils import process_sent, process_word

root_dir = '../../../dstc3/'

def get_lack_acts(train_file, test_file):

    with codecs.open(train_file, 'r') as f:
        lines = f.readlines()
        labels = [line.split('\t<=>\t')[1].strip() for line in lines]
        train_labels = []
        for label in labels:
            lis = label.strip().split(';')
            if len(lis) > 0:
                train_labels.extend(lis)
        train_labels = ['-'.join(label.strip().split('-')[0:2]) for label in train_labels]

    with codecs.open(test_file, 'r') as f:
        lines = f.readlines()
        labels = [line.split('\t<=>\t')[1].strip() for line in lines]
        test_labels = []
        for label in labels:
            lis = label.strip().split(';')
            if len(lis) > 0:
                test_labels.extend(lis)
        test_labels = ['-'.join(label.strip().split('-')[0:2]) for label in test_labels]


    labels = set(test_labels) - set(train_labels)
    labels = list(labels)
    print(labels)

    labels = []
    for label in train_labels:
        labels.extend(label.split('-'))
    train_labels = labels
       
    labels = []
    for label in test_labels:
        labels.extend(label.split('-'))
    test_labels = labels

    labels = set(test_labels) - set(train_labels)
    labels = list(labels)
    print(labels)

if __name__ == '__main__':
    #get_lack_acts(root_dir+'manual/dstc2_seed_1.train', root_dir+'manual/test')
    get_lack_acts(root_dir+'manual/test', root_dir+'manual/dstc2_seed_0.train')
