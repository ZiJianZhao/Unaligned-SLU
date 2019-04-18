# -*- coding: utf-8 -*-

import argparse
import os, sys
import codecs
from collections import Counter, defaultdict
import json
import numpy as np
import torch
import copy

root_dir = '../../../dstc3-st/manual-da/'
dstc2_non_enumerable_slots = ['food', 'name', 'area']
dstc2_enumerable_slots = ['pricerange']
dstc3_non_enumerable_slots = ['food', 'name', 'area', 'near']
dstc3_enumerable_slots = ['hastv', 'childrenallowed', 'hasinternet', 'pricerange', 'type']

def normalization(line):

    rules = {
        #'asian oriental': ['asian', 'oriental'],
        'barbeque': ['barbecue'],

        'chinese takeaway': ['chinese take away'],
        'fen ditton': ['fenditton'],
        'drinks and snacks only': ['drinks and snacks', 'snacks and drinks'],
        #'mexican tex mex': ['mexican', 'texmex'],
        'centre': ['center', 'central'],
        'coffeeshop': ['coffee shop'],
        #'castle hill': ['castlehill', 'castle'],
        'riverside': ['river side'],
        'seafood': ['sea food'],
        'addenbrookes': ['addonbrookes', 'addenbrooke'],
        'thai': ['thia'],
        'kings hedges': ['kings hedge', 'kings hedgess', 'king hedges'],
        #'northern': ['north'],
        #'eastern': ['east'],
        #'western': ['west'],
        #'southern': ['south'],
        }

    utt, trp = line.split('\t<=>\t')
    # the rules
    for t in trp.split(';'):
        tmp = t.strip().split('-', 2)
        if (len(tmp) == 3) and (tmp[2].startswith('the')):
            hah = tmp[2][3:].strip()
            utt = utt.replace(hah, tmp[2])

    for key in rules:
        for val in rules[key]:
            if val in utt:
                utt = utt.replace(val, key)

    # triple alias for hastv, childrenallowed, hasinternet

    line = '{}\t<=>\t{}'.format(utt, trp)
    return line

def judge_example(line, slots):
    utt, trp = line.split('\t<=>\t')
    utt_lis = utt.strip().split()
    if trp.strip() == '':
        return False
    else:
        trp_lis = trp.strip().split(';')
        for t in trp_lis:
            t_lis = t.strip().split('-', 2)
            if (len(t_lis) == 3) and (t_lis[1] in slots):
                if t_lis[2] == 'dontcare':
                    continue
                else:
                    if t_lis[2] not in utt:
                        return False
                    else:
                        for word in t_lis[2].strip().split():
                            if word not in utt_lis:
                                return False
    return True

def delexicalisation(file_name, save_file, slots):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    results = []
    count = 0
    silent = 0
    total = len(lines)
    for line in lines:
        if line.split('\t<=>\t')[1].strip() == '':
            silent += 1
            continue
        #line = normalization(line)
        if not judge_example(line, slots):
            #print(line)
            #input()
            count += 1
            continue
        utt, trp = line.split('\t<=>\t')
        trp_lis = trp.strip().split(';')
        new_lis = []
        for t in trp_lis:
            t_lis = t.strip().split('-', 2)
            if (len(t_lis) == 3) and (t_lis[1] in slots) and (t_lis[2] in utt):
                utt = utt.replace(t_lis[2], '[' + t_lis[1] + ']')
                t = t.replace(t_lis[2], '[' + t_lis[1] + ']')
            new_lis.append(t)
        #if len(new_lis) >= 6:
        #    print(line)
        #    input()
        trp = ';'.join(new_lis)
        results.append((utt, trp))
    #results = list(set(results))
    #results = sorted(results)
    print('Saved    Sentences: {}'.format(len(results)))
    print('Silent   Sentences: {}'.format(silent))
    print('Informal Sentences: {}'.format(count))
    print('Total    Sentences: {}'.format(total))
    print('--------------------------------------')

    if save_file is None:
        return results

    with open(save_file, 'w') as g:
        for (utt, trp) in results:
            g.write('{}\t<=>\t{}\n'.format(utt, trp))

"""
def get_del_triples(file_name, save_file, slots):
    lines = delexicalisation(
        file_name,
        None,
        slots
    )
    triples = [line[1] for line in lines]
    triples = list(set(triples))
    print('Final Triple Number: {}'.format(len(triples)))

    with open(save_file, 'w') as g:
        for trp in triples:
            g.write('{}\n'.format(trp))
"""

def format_ori_triples(class_file, save_file):
    with open(class_file, 'r') as f:
        triples = f.readlines()
    results = []
    for trp in triples:
        results.append((trp.strip(), {}))
    string = json.dumps(results, sort_keys=False, indent=4, separators=(',', ':'))
    with open(save_file, 'w') as g:
        g.write(string)


def get_del_triples(class_file, save_file, slots):
    with open(class_file, 'r') as f:
        triples = f.readlines()
    results = []
    for trp in triples:
        trp_lis = trp.strip().split(';')
        new_lis = []
        d = {}
        for t in trp_lis:
            t_lis = t.strip().split('-', 2)
            if (len(t_lis) == 3) and (t_lis[1] in slots):
                tag = '[' + t_lis[1] + ']'
                t = t.replace(t_lis[2], tag)
                d[tag] = t_lis[2]
            new_lis.append(t)
        trp = ';'.join(new_lis)
        results.append((trp, d))
    string = json.dumps(results, sort_keys=False, indent=4, separators=(',', ':'))
    with open(save_file, 'w') as g:
        g.write(string)


def get_slot_values(file_name, slots):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    dic = defaultdict(list)
    for line in lines:
        utt, trp = line.split('\t<=>\t')
        for t in trp.strip().split(';'):
            lis = t.strip().split('-', 2)
            if (len(lis) == 3) and (lis[1] in slots):
                dic['['+lis[1]+']'].append(lis[2])
    for key in dic:
        dic[key] = list(set(dic[key]))
    return dic

def empty_triple_utterances():
    return ['sil', 'system sil']

def realisation(lis, slot, values):
    res = []
    for (utt, trp) in lis:
        for val in values:
            new_utt = utt.replace(slot, val)
            new_trp = trp.replace(slot, val)
            res.append((new_utt, new_trp))
    return res

def surface_realisation(slots, ori_file, gen_file, save_file):
    slot_value_dict = get_slot_values(ori_file, slots)
    for slot in slot_value_dict:
        print('{} number: {}'.format(slot, len(slot_value_dict[slot])))

    with open(gen_file, 'r') as f:
        lines = f.readlines()
    results = []
    for line in lines:
        utt, trp = line.strip().split('\t<=>\t')
        lis = [(utt, trp)]
        for slot in slot_value_dict:
            if slot in trp:
                lis = realisation(lis, slot, slot_value_dict[slot])
        results.extend(lis)
    print('Final line number: {}'.format(len(results)))
    with open(save_file, 'w') as g:
        for (utt, trp) in results:
            g.write('{}\t<=>\t{}\n'.format(utt, trp))
        for utt in empty_triple_utterances():
            g.write('{}\t<=>\t \n'.format(utt))


def pre_alias(del_file, als_file):
    with open(del_file, 'r') as f:
        lines = f.readlines()
    results = []
    for line in lines:
        line = line.replace('hastv', 'tv')
        line = line.replace('hasinternet', 'internet')
        line = line.replace('childrenallowed', 'children')
        results.append(line)
    with open(als_file, 'w') as g:
        for line in results:
            g.write('{}'.format(line))

def post_alias(del_file, als_file):
    with open(del_file, 'r') as f:
        lines = f.readlines()
    results = []
    for line in lines:
        utt, trp = line.split('\t<=>\t')
        trp = trp.replace('tv', 'hastv')
        trp = trp.replace('internet', 'hasinternet')
        trp = trp.replace('children', 'childrenallowed')
        results.append((utt, trp))
    with open(als_file, 'w') as g:
        for (utt, trp) in results:
            g.write('{}\t<=>\t{}'.format(utt, trp))


if __name__ == '__main__':
    """
    delexicalisation(
        root_dir + 'ori/dstc2.train',
        root_dir + 'del/dstc2.train',
        dstc2_non_enumerable_slots
    )
    #"""
    #"""
    format_ori_triples(
        root_dir + 'ori/dstc3.test.triples',
        root_dir + 'ori/dstc3.test.triples.json'
    )
    """
    get_del_triples(
        root_dir + 'ori/dstc3.test.triples',
        root_dir + 'del/dstc3.test.triples.json',
        dstc3_non_enumerable_slots
    )
    #"""
    """
    surface_realisation(
        dstc3_non_enumerable_slots,
        root_dir + 'dstc3_ori_test',
        root_dir + 'decodes/dstc2-3train-on-test-tmp-1',
        root_dir + 'genates/dstc2-3train-on-test-tmp-1'
    )
    #"""
    """
    post_alias(
        root_dir + 'decodes/dstc2-3train-on-test-als-6',
        root_dir + 'decodes/dstc2-3train-on-test-als-post-6'
    )
    """
    """
    pre_alias(
        root_dir + 'dstc3_del_valid',
        root_dir + 'dstc3_als_valid'
    )
    #"""
