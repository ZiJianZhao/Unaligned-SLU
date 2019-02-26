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
stop_words = ['the', 'all', ]


def rule_method(class_file, save_file):
    with open(class_file, 'r') as f:
        classes = f.readlines()
    lines = []
    for cls in classes:
        lis = cls.strip().split('-', 2)
        if len(lis) == 3:
            if lis[0] == 'inform':
                if lis[1] in ['near', 'type', 'area']:
                    lines.append((lis[2], cls))
                elif lis[1] == 'hastv':
                    if lis[2].strip() == 'true':
                        lines.append(('has television', cls))
                        lines.append(('has tv', cls))
                    else:
                        lines.append(('no television', cls))
                        lines.append(('no tv', cls))
                elif lis[1] == 'hasinternet':
                    if lis[2].strip() == 'true':
                        lines.append(('with internet', cls))
                    else:
                        lines.append(('no internet', cls))
                elif lis[1] == 'childrenallowed':
                    if lis[2].strip() == 'True':
                        lines.append(('allows children', cls))
                    else:
                        lines.append(('no children allowed', cls))
                elif lis[1] == 'this':
                    pass
                else:
                    pass
                    #raise Exception('unknown slots')
            elif lis[0] == 'deny':
                if lis[1] in ['near', 'type', 'area']:
                    lines.append(('not {}'.format(lis[2]), cls))
            elif lis[0] == 'confirm':
                if lis[1] in ['near', 'type', 'area']:
                    lines.append(('is it {}'.format(lis[2]), cls))
            else:
                raise Exception('unknown acts')
        """
        elif len(lis) == 2:
            if lis[1] in ['price', 'area', 'phone', 'near', 'type']:
                lines.append(('what is {}'.format(lis[1]), cls))
            elif lis[1] in ['hastv']:
                lines.append(('dose it has tv', cls))
            elif lis[1] in ['hasinternet']:
                lines.append(('dose it has internet', cls))
            elif lis[1] in ['childrenallowed']:
                lines.append(('dose it allow children', cls))
        """
    print('Generated lines {}'.format(len(lines)))
    with open(save_file, 'w') as f:
        for (utterance, class_string) in lines:
            f.write('{}\t<=>\t{}\n'.format(utterance, class_string.strip()))

"""
if __name__ == '__main__':
    dir_name = root_dir + '1best-live/'
    rule_method(dir_name + 'class.all', dir_name + 'rule_1.train')
    #dir_name = root_dir + 'manual/'
    #dstc2_train = dir_name + 'dstc2.train'
    #dstc3_seed = dir_name + 'seed.train'
"""

def inform_type_restaurant(dstc2_train):
    with open(dstc2_train, 'r') as f:
        lines = f.readlines()
    res = []
    for line in lines:
        utterance, class_string = line.split('\t<=>\t')
        if ('restaurant' in utterance) or ('restaurants' in utterance):
            classes = class_string.strip().split(';')
            flag = True
            for cls in classes:
                cls_lis = cls.split('-')
                if len(cls_lis) == 3 and cls_lis[0] == 'inform' and cls_lis[1] == 'food':
                    pass
                else:
                    flag = False
                    break
            if not flag:
                continue
            if class_string.strip() == '':
                class_string = 'inform-type-restaurant'
            else:
                class_string = class_string.strip() + ';inform-type-restaurant'
            line = '{}\t<=>\t{}\n'.format(utterance, class_string)
            res.append(line)
    res = list(set(res))
    return res

def request_price(dstc2_train):
    with open(dstc2_train, 'r') as f:
        lines = f.readlines()
    res = []
    for line in lines:
        utterance, class_string = line.split('\t<=>\t')
        if ('price' in utterance) and ('range' in utterance) and ('request-pricerange' in class_string):
            utterance = utterance.replace('range', '')
            class_string = class_string.replace('pricerange', 'price')
            line = '{}\t<=>\t{}\n'.format(utterance, class_string.strip())
            res.append(line)
    res = list(set(res))
    return res

def inform_pricerange_free_1(dstc2_train):
    with open(dstc2_train, 'r') as f:
        lines = f.readlines()
    res = []
    for line in lines:
        utterance, class_string = line.split('\t<=>\t')
        if len(class_string.strip().split(';')) == 1 and class_string.startswith('inform-pricerange'):
            res.append(line)
    res = list(set(res))
    return res


def inform_pricerange_free_2(dstc2_train):
    res = ['{}\t<=>\t{}\n'.format('free', 'inform-pricerange-free')]
    return res

def inform_food_XXX(dstc2_train):
    res = []
    pattern = '{}\t<=>\t{}\n'
    for i in range(5):
        res.append(pattern.format('american', 'inform-food-american'))
        res.append(pattern.format('fast food', 'inform-food-fast food'))
        res.append(pattern.format('mexican', 'inform-food-mexican tex mex'))
        res.append(pattern.format('drinks and snacks', 'inform-food-drinks and snacks only'))
        res.append(pattern.format('chinese take away', 'inform-food-chinese takeaway'))
    return res

def type_slot(dstc2_train):
    res = []
    pattern = '{}\t<=>\t{}\n'
    res.append(pattern.format('restaurants', 'inform-type-restaurant'))
    res.append(pattern.format('a restaurant', 'inform-type-restaurant'))
    res.append(pattern.format('pubs', 'inform-type-pub'))
    res.append(pattern.format('a pub', 'inform-type-pub'))
    res.append(pattern.format('cafe', 'inform-type-coffeeshop'))
    res.append(pattern.format('cafes', 'inform-type-coffeeshop'))
    res.append(pattern.format('coffee', 'inform-type-coffeeshop'))
    res.append(pattern.format('coffee shop', 'inform-type-coffeeshop'))
    return res



def generate_new_file(dstc3_seed, new_lines, save_file):
    with open(dstc3_seed, 'r') as f:
        lines = f.readlines()
    lines = lines + new_lines

    with open(save_file, 'w') as f:
        for line in lines:
            f.write('{}'.format(line))

"""
if __name__ == '__main__':
    #rule_method_1(dir_name + 'class.all', dir_name + 'rule_1.train')
    dir_name = root_dir + 'manual/'
    dstc2_train = dir_name + 'dstc2.train'
    dstc3_seed = dir_name + 'seed.train'

    tag = 'inform-food-XXX'
    if tag == 'inform-type-restaurant':
        save_file = dir_name + 'seed.inform_type_restaurant.train'
        new_lines = inform_type_restaurant(dstc2_train)
    elif tag == 'request-price':
        save_file = dir_name + 'seed.request_price.train'
        new_lines = request_price(dstc2_train)
    elif tag == 'inform-pricerange-free-1':
        save_file = dir_name + 'seed.inform_pricerange_free_1.train'
        new_lines = inform_pricerange_free_1(dstc2_train)
    elif tag == 'inform-pricerange-free-2':
        save_file = dir_name + 'seed.inform_pricerange_free_2.train'
        new_lines = inform_pricerange_free_2(dstc2_train)
    elif tag == 'inform-food-XXX':
        save_file = dir_name + 'seed.inform_food_XXX.train'
        new_lines = inform_food_XXX(dstc2_train)
    elif tag == 'type-slot':
        save_file = dir_name + 'seed.type_slot.train'
        new_lines = type_slot(dstc2_train)
    else:
        raise Exception('Error')

    print('{}:\t{}'.format(tag, len(new_lines)))
    generate_new_file(dstc3_seed, new_lines, save_file)
"""


# for end-to-end generation


def filter_triples(class_all_file, class_save_file):
    with open(class_all_file, 'r') as f:
        classes = f.readlines()
    res = []
    for cls in classes:
        lis = cls.strip().split('-', 2)
        if len(lis) == 3:
            if lis[1] in ['near', 'type', 'area', 'hastv',
                'hasinternet', 'childrenallowed', 'this']:
                res.append(cls)
        elif len(lis) == 2:
            res.append(cls)
        else:
            pass
    print('Triples Number: {}'.format(len(res)))
    with open(class_save_file, 'w') as f:
        for cls in res:
            f.write('{}\n'.format(cls.strip()))

if __name__ == '__main__':
    dir_name = root_dir + 'manual-da/'
    filter_triples(dir_name + 'class.all', dir_name + 'class.gen')
    #dir_name = root_dir + 'manual/'
    #dstc2_train = dir_name + 'dstc2.train'
    #dstc3_seed = dir_name + 'seed.train'
