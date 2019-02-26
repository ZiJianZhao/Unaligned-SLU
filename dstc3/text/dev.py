# -*- coding: utf-8 -*-
""" To get a sound dev set """

import argparse
import os, sys
import codecs
import random
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

def method_1(file_name, save_file, num=100):

    with codecs.open(file_name, 'r') as f:
        lines = f.readlines()
    lines = list(set(lines))

    res = random.sample(lines, num)

    with open(save_file, 'w') as g:
        for line in res:
            g.write('{}'.format(line))


if __name__ == '__main__':
    method_1(root_dir+'manual/dstc2.valid', root_dir+'manual/dstc2.m1.valid')
