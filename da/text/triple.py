# -*- coding: utf-8 -*-

import argparse
import os, sys, random
import codecs, copy
from collections import Counter, defaultdict
import json
import numpy as np
import torch
import copy
import difflib
import Levenshtein
from itertools import combinations, product

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(install_path)
sys.path.append(install_path)

import xslu.Constants as Constants
from xslu.utils import process_sent, process_word

random.seed(1234)

dstc2_non_enumerable_slots = ['food', 'name', 'area']
dstc2_enumerable_slots = ['pricerange']
dstc3_non_enumerable_slots = ['food', 'name', 'area', 'near']
dstc3_enumerable_slots = ['hastv', 'childrenallowed', 'hasinternet', 'pricerange', 'type']

glove_path = '../../../../NLPData/glove.6B/glove.6B.100d.txt'
stop_words = ['the', 'all', ]

def extract_triples(file_name):
    with codecs.open(file_name, 'r') as f:
        lines = f.readlines()
    triples = [line.split('\t<=>\t')[1] for line in lines]
    triples = [trp.strip() for trp in triples if trp.strip() != '']
    return triples

def split_triples(triple):
    results = []
    triples = triple.strip().split(';')
    for i in range(1, len(triples)+1):
        lis = list(combinations(triples, i))
        for l in lis:
            results.append(';'.join(l))
    return results

def extract_values(file_name):
    triples = extract_triples(file_name)
    values = defaultdict(list)
    for trp in triples:
        lis = trp.strip().split(';')
        for l in lis:
            tmp = l.strip().split('-', 2)
            if len(tmp) == 3:
                values[tmp[1]].append(tmp[2])
    for key in values:
        values[key] = list(set(values[key]))
    return values

def extend_triples_es(triples, values):
    # nes: enumerable-slot
    def judge(triple, values):
        results = [triple]
        lis = triple.strip().split('-', 2)
        if len(lis) == 3:
            if lis[1] in dstc3_enumerable_slots:
                for val in values[lis[1]]:
                    string = '-'.join([lis[0], lis[1], val])
                    results.append(string)
        results = list(set(results))
        return results

    results = []
    for trp in triples:
        lis = trp.strip().split(';')
        tmp = []
        for l in lis:
            tmp.append(judge(l, values))
        tmp = list(product(*tmp))
        tmp = [';'.join(t) for t in tmp]
        results.extend(tmp)

    totals = []
    for r in results:
        if r not in triples:
            totals.append(r)
    print('Origin Triple Number: {}'.format(len(triples)))
    print('Added  Triple Number: {}'.format(len(totals)))
    results = triples + totals
    return results

def random_value_fill(triples, values, save_file):
    results = []
    for trp in triples:
        dic = {}
        lis = trp.strip().split(';')
        for t in lis:
            tmp = t.strip().split('-', 2)
            if (len(tmp) == 3) and (tmp[1] in dstc3_non_enumerable_slots):
                value = random.choice(values[tmp[1]])
                dic['['+tmp[1]+']'] = value
        results.append([trp, dic])
    string = json.dumps(results, sort_keys=False, indent=4, separators=(',', ':'))
    with open(save_file, 'w') as g:
        g.write(string)

def method_one(dstc3_seed_file, dstc3_test_file, save_file):
    values = extract_values(dstc3_test_file)
    triples = extract_triples(dstc3_seed_file)
    results = []
    for trp in triples:
        results.extend(split_triples(trp))
    results = extend_triples_es(results, values)
    print('Final Data Number: {}'.format(len(results)))
    random_value_fill(results, values, save_file)

def method_thr(dstc2_train_file, dstc3_seed_file, dstc3_test_file, save_file):
    values = extract_values(dstc3_test_file)
    triples = extract_triples(dstc3_seed_file)
    results = []
    for trp in triples:
        results.extend(split_triples(trp))
    results = extend_triples_es(results, values)
    triples = extract_triples(dstc2_train_file)
    triples = list(set(triples))
    results.extend(triples)
    print('Final Data Number: {}'.format(len(results)))
    random_value_fill(results, values, save_file)

def R1(ontology_path=None, maxm=None):

    triples = ['ack', 'hello', 'affirm', 'bye', 'negate',
        'repeat', 'reqalts', 'reqmore', 'restart', 'thankyou']
    results = list(combinations(triples, 1))

    return list(set(results))


def R2(ontology_path, maxm=3):

    ontology = json.loads(open(ontology_path).read())
    acts = ['request']
    slots = ontology['requestable']

    triples = []
    for act in acts:
        for slot in slots:
            triples.append(act+'-'+slot)

    results = []
    for i in range(1, maxm+1):
        results.extend(list(combinations(triples, i)))

    return list(set(results))

def R3(ontology_path, maxm=3):

    ontology = json.loads(open(ontology_path).read())
    slots = list(ontology['informable'].keys())
    values = ontology['informable']

    triples = []
    for slot in slots:
        if slot in ['hastv', 'childrenallowed', 'hasinternet']:
            triples.append('inform-' + slot + '-true')
            triples.append('inform-' + slot + '-false')
            #triples.append('inform-' + slot + '-dontcare')
        elif slot in ['pricerange', 'type']:
            for value in values[slot]:
                triples.append('inform-' + slot + '-' + value)
            #triples.append('inform-' + slot + '-dontcare')
        else:
            triples.append('inform-' + slot + '-' + '[' + slot + ']')
            #triples.append('inform-' + slot + '-dontcare')

    results = []
    for i in range(1, maxm+1):
        results.extend(list(combinations(triples, i)))

    return list(set(results))

def R4(tri2ples, tri3ples):

    results = []
    for tr2p in tri2ples:
        for tr3p in tri3ples:
            s2s = set([t.split('-')[1] for t in tr2p])
            s3s = set([t.split('-')[1] for t in tr3p])
            sss = s2s.intersection(s3s)
            if len(sss) == 0:
                results.append(tr2p + tr3p)

    return list(set(results))

def R5(tri1ples, tri2ples, tri3ples, tri4ples):

    results = []
    triples = tri2ples + tri3ples + tri4ples
    triples = list(set(triples))
    for tr1p in tri1ples:
        for trp in triples:
            results.append(tr1p + trp)

    return list(set(results))

def R6(ontology_path, maxm=None):

    ontology = json.loads(open(ontology_path).read())
    acts = ['deny', 'confirm']
    slots = list(ontology['informable'].keys())
    values = ontology['informable']

    triples = []
    for act in acts:
        for slot in slots:
            if slot in ['hastv', 'childrenallowed', 'hasinternet']:
                continue
            elif slot in ['pricerange', 'type']:
                for value in values[slot]:
                    triples.append(act + '-' + slot + '-' + value)
            else:
                triples.append(act + '-' + slot + '-' + '[' + slot + ']')

    for slot in slots:
        triples.append('inform-' + slot + '-dontcare')
    triples.append('inform-this-dontcare')

    results = []
    results.extend(list(combinations(triples, 1)))

    return list(set(results))

def R2T(triples):

    results = []
    for tup in triples:
        results.append(';'.join(tup))

    return results

def EE(ontology_path, maxm, save_file, times=3):

    r1trps = R1()
    r2trps = R2(ontology_path, 3)
    r3trps = R3(ontology_path, 3)
    r4trps = R4(r2trps, r3trps)
    r5trps = R5(r1trps, r2trps, r3trps, r4trps)
    r6trps = R6(ontology_path)
    triples = r1trps + r2trps + r3trps + r4trps + r5trps + r6trps

    results = []
    for trp in triples:
        if len(trp) <= maxm:
            results.append(trp)
    results = R2T(results)

    values = read_ontology(ontology_path)
    full_value_fill(results, values, save_file, times)

def read_ontology(file_name):
    dic = json.loads(open(file_name).read())
    values = dic['informable']
    return values

def method_one_ont(dstc3_seed_file, ontology_file, save_file, times):
    values = read_ontology(ontology_file)
    triples = extract_triples(dstc3_seed_file)
    results = []
    for trp in triples:
        results.extend(split_triples(trp))
    results = extend_triples_es(results, values)
    full_value_fill(results, values, save_file, times)

def method_thr_ont(dstc2_train_file, dstc3_seed_file, ontology_file, save_file, times):
    values = read_ontology(ontology_file)
    triples = extract_triples(dstc3_seed_file)
    results = []
    for trp in triples:
        results.extend(split_triples(trp))
    results = extend_triples_es(results, values)
    triples = extract_triples(dstc2_train_file)
    triples = list(set(triples))
    results.extend(triples)
    full_value_fill(results, values, save_file, times)


def full_value_fill(triples, values, save_file, times):

    print('-----------------------')
    print('Triple Number: {}'.format(len(triples)))

    results = []
    news = defaultdict(list)
    for key in values:
        if key in dstc3_non_enumerable_slots:
            for _ in range(times):
                news[key].extend(values[key])

    # first, ensure that each triple is filled with some value
    for trp in triples:
        dic = {}
        lis = trp.strip().split(';')
        for t in lis:
            tmp = t.strip().split('-', 2)
            if (len(tmp) == 3) and (tmp[1] in dstc3_non_enumerable_slots) and (tmp[2] != 'dontcare'):
                if len(news[tmp[1]]) > 0:
                    value = random.choice(news[tmp[1]])
                    dic['['+tmp[1]+']'] = value
                    news[tmp[1]].remove(value)
                else:
                    value = random.choice(values[tmp[1]])
                    dic['['+tmp[1]+']'] = value
        results.append([trp, dic])

    # then, ensure that values are filled with our expected times
    total = -1
    index = 0
    while True:
        index += 1
        for trp in triples:
            flag = False
            dic = {}
            lis = trp.strip().split(';')
            for t in lis:
                tmp = t.strip().split('-', 2)
                if (len(tmp) == 3) and (tmp[1] in dstc3_non_enumerable_slots) and (tmp[2] != 'dontcare'):
                    if len(news[tmp[1]]) > 0:
                        value = random.choice(news[tmp[1]])
                        dic['['+tmp[1]+']'] = value
                        news[tmp[1]].remove(value)
                    else:
                        flag = True
                        break
            if not flag:
                if len(dic) > 0:
                    results.append([trp, dic])
                else:
                    if index <= times - 1:
                        results.append([trp, dic])
        tmp = sum([len(news[key]) for key in news])
        if (total == 0) or (total == tmp):
            break
        else:
            total = tmp

    print('Final Data Number: {}'.format(len(results)))
    string = json.dumps(results, sort_keys=False, indent=4, separators=(',', ':'))
    with open(save_file, 'w') as g:
        g.write(string)

def read(file_name):
    triples = json.loads(open(file_name).read())
    oris = []
    dels = []
    for trp in triples:
        class_string = trp[0].strip()
        class_string = ';'.join(sorted(class_string.split(';')))
        dels.append(class_string)
        value_dict = trp[1]
        for key in value_dict:
            class_string = class_string.replace(key, value_dict[key])
        class_string = ';'.join(sorted(class_string.split(';')))
        oris.append(class_string)
    print(file_name)
    print(len(oris))
    oris_v2 = list(set(oris))
    dels_v2 = list(set(dels))
    return [oris, oris_v2, dels, dels_v2]

def overlap_rate(triple_file, dstc3_test_file):

    def funct(classes, triples, tag):
        total = len(classes)
        for trp in triples:
            if trp in classes:
                classes.remove(trp)
        """
        if tag == 'del nror':
            for cls in classes:
                print(cls)
                input()
        #"""
        final = total - len(classes)
        print('========= {} overlap rate ========='.format(tag))
        print('Rate: {}/{}, {}'.format(final, total, float(final)/total))
        print('Triple Number: {}'.format(len(triples)))


    def freq(classes, triples, tag):
        total = len(classes)
        dic = defaultdict(int)
        for trp in classes:
            dic[trp] += 1
        num = 0
        for trp in triples:
            if trp in dic:
                if dic[trp] != 0:
                    num += dic[trp]
                    dic[trp] = 0
        print('========= {} frequency rate ========='.format(tag))
        print('Rate: {}/{}, {}'.format(num, total, float(num)/total))
        print('Triple Number: {}'.format(len(triples)))

    """
    with open(tmp_file, 'r') as f:
        lines = f.readlines()
    classes = [line.split('\t<=>\t')[1] for line in lines]
    print('Origin Test Number: {}'.format(len(classes)))
    classes = [cls.strip() for cls in classes if cls.strip() != '']
    print('Filter Test Number: {}'.format(len(classes)))
    classes = list(set(classes))
    print('Set Test Number: {}'.format(len(classes)))
    """

    tags = ['ori ror', 'ori nror', 'del ror', 'del nror']
    tests = read(dstc3_test_file)
    trps = read(triple_file)
    results = zip(copy.deepcopy(tests), copy.deepcopy(trps), tags)
    for (classes, triples, tag) in results:
        funct(classes, triples, tag)
    freq(copy.deepcopy(tests[0]), copy.deepcopy(trps[0]), 'ori')
    freq(copy.deepcopy(tests[2]), copy.deepcopy(trps[2]), 'del')

def compare_triples(file_name_1, file_name_2):
    _, _, _, tri1ples = read(file_name_1)
    _, _, _, tri2ples = read(file_name_2)
    results = list(set(tri1ples) - set(tri2ples))
    for i in results:
        print(i)

if __name__ == '__main__':
    dir_name = '/slfs1/users/zjz17/SLU/dstc3-st/manual-da/'
    ontology_file = '/slfs1/users/zjz17/SLU/dstc3-st/scripts/config/ontology_dstc3.json'
    """
    method_one_ont(
        dir_name + 'del/dstc3.seed',
        #dir_name + 'dstc3.test',
        ontology_file,
        dir_name + 'trp/m1.v2.3',
        3
    )
    #"""
    """
    method_thr_ont(
        dir_name + 'del/dstc2.train',
        dir_name + 'del/dstc3.seed',
        #dir_name + 'dstc3.test',
        ontology_file,
        dir_name + 'trp/no.m3.v2.3.tmp',
        3
    )
    #"""
    #"""
    overlap_rate(
        dir_name + 'trp/e3',
        dir_name + 'del/dstc3.test.triples.json'
    )
    #"""
    """
    compare_triples(
        dir_name + 'trp/m1.v2.1',
        dir_name + 'trp/m1'
    )
    #"""
    """
    EE(
        ontology_path = ontology_file,
        maxm = 3,
        save_file = dir_name + 'trp/e3',
        times = 3
    )
    #"""
