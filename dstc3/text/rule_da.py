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
glove_path = '../../../../NLPData/glove.6B/glove.6B.100d.txt'

def rule_method_1(class_file, save_file):
    with open(class_file, 'r') as f:
        classes = f.readlines()
    lines = []
    for cls in classes:
        lis = cls.strip().split('-', 2)
        if len(lis) == 3:
            if lis[0] == 'inform':
                if lis[1] in ['food', 'near', 'type', 'name', 'pricerange', 'area']:
                    lines.append((lis[2], cls))
                elif lis[1] == 'hastv':
                    if lis[2].strip() == 'true':
                        lines.append(('has television', cls))
                    else:
                        lines.append(('no television', cls))
                elif lis[1] == 'hasinternet':
                    if lis[2].strip() == 'true':
                        lines.append(('with internet', cls))
                    else:
                        lines.append(('no internet', cls))
                elif lis[1] == 'childrenallowed':
                    if lis[2].strip() == 'True':
                        lines.append(('allows children', cls))
                    else:
                        lines.append(('no children', cls))
                elif lis[1] == 'this':
                    pass
                else:
                    raise Exception('unknown slots')
            elif lis[0] == 'deny':
                if lis[1] in ['food', 'near', 'type', 'name', 'pricerange', 'area']:
                    lines.append(('not {}'.format(lis[2]), cls))
            elif lis[0] == 'confirm':
                if lis[1] in ['food', 'near', 'type', 'name', 'pricerange', 'area']:
                    lines.append(('is it {}'.format(lis[2]), cls))
            else:
                raise Exception('unknown acts')
    with open(save_file, 'w') as f:
        for (utterance, class_string) in lines:
            f.write('{}\t<=>\t{}\n'.format(utterance, class_string.strip()))

if __name__ == '__main__':
    dir_name = root_dir + 'manual/'
    rule_method_1(dir_name + 'class.all', dir_name + 'rule_1.train')
