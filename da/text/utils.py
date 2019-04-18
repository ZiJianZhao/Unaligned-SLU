# -*- coding: utf-8 -*-

import argparse
import os, sys
import codecs, random
from collections import Counter, defaultdict
import json
import numpy as np
import torch
import copy

def generate_dstc2_old_value_test(test_file, save_file, slot):
    with open(test_file) as f:
        lines = f.readlines()
    results = []
    for line in lines:
        line_list = line.split('\t<=>\t')
        if line_list[1].strip() != '':
            triples = line_list[1].strip().split(';')
            for triple in triples:
                lis = triple.strip().split('-', 2)
                if (len(lis) == 3) and (lis[1] == slot):
                    results.append(line)
                    break
    results = list(set(results))
    if save_file is None:
        return results
    with open(save_file, 'w') as f:
        for line in results:
            f.write('{}'.format(line))


def generate_dstc2_new_value_test(class_train, test_file, save_file, slot, ontology_file):
    with open(class_train) as f:
        classes = f.readlines()
    results = []
    for cls in classes:
        lis = cls.strip().split('-', 2)
        if len(lis) == 3:
            if lis[1] == slot:
                results.append(lis[2])
    results = list(set(results))
    totals = json.loads(open(ontology_file).read())['informable'][slot]
    values = list(set(totals) - set(results))

    with open(test_file) as f:
        lines = f.readlines()
    results = []
    for line in lines:
        line_list = line.split('\t<=>\t')
        utterance = line_list[0].strip()
        new_triples = []
        if line_list[1].strip() != '':
            triples = line_list[1].strip().split(';')
            for triple in triples:
                lis = triple.strip().split('-', 2)
                if (len(lis) == 3) and (lis[1] == slot):
                    value = random.choice(values)
                    tmp = utterance.replace(lis[2], value)
                    if tmp == utterance:
                        pass
                    else:
                        utterance = tmp
                        triple = triple.replace(lis[2], value)
                new_triples.append(triple)
            new_triple = ';'.join(new_triples)
            if new_triple != line_list[1].strip():
                new_line = '{}\t<=>\t{}'.format(utterance, new_triple)
                results.append(new_line)
    results = list(set(results))
    with open(save_file, 'w') as f:
        for line in results:
            f.write('{}\n'.format(line))

def generate_dstc2_new_slot_test(test_file, save_file, slot, ontolgy_file):

    values = json.loads(open(ontology_file).read())['informable'][slot]
    lines = generate_dstc2_old_value_test(test_file, None, 'name')

    def new_example(value, utterance, triples):
        new_triples = []
        for triple in triples:
            lis = triple.strip().split('-', 2)
            if (len(lis) == 3) and (lis[1] == slot):
                utterance = utterance.replace(lis[2], value)
                triple = triple.replace(lis[2], value)
            new_triples.append(triple)

        new_triple = ';'.join(new_triples)
        new_line = '{}\t<=>\t{}'.format(utterance, new_triple)
        return new_line

    results = []
    for line in lines:
        flag = True
        line_list = line.split('\t<=>\t')
        utterance = line_list[0].strip()
        triples = line_list[1].strip().split(';')
        for triple in triples:
            lis = triple.strip().split('-', 2)
            if (len(lis) == 3) and (lis[1] == slot):
                if lis[2] not in utterance:
                    flag = False
        if flag:
            for value in values:
                new_line = new_example(value, utterance, triples)
                results.append(new_line)
    results = list(set(results))
    with open(save_file, 'w') as f:
        for line in results:
            f.write('{}\n'.format(line))

def generate_multi_triples_train(train_file, save_file):

    with open(train_file, 'r') as f:
        lines = f.readlines()

    results = []
    for line in lines:
        results.append(line.strip('\n'))
        lis = line.split('\t<=>\t')
        if lis[1].strip() != '':
            classes = lis[1].strip().split(';')
            if len(classes) >= 2:
                classes.reverse()
                cls = ';'.join(classes)
                tmp = '{}\t<=>\t{}'.format(lis[0], cls)
                results.append(tmp)

    with open(save_file, 'w') as f:
        for line in results:
            f.write('{}\n'.format(line))


def generate_multi_name_train(train_file, save_file, times=10):

    with open(train_file, 'r') as f:
        lines = f.readlines()

    results = []
    total = 0
    for line in lines:
        lis = line.split('\t<=>\t')
        if '-name-' in lis[1]:
            total += 1
            for i in range(times):
                results.append(line)
        else:
            results.append(line)
    print('Including name: {}'.format(total))

    random.shuffle(results)
    with open(save_file, 'w') as f:
        for line in results:
            f.write('{}'.format(line))

def file_triples(file_name, save_file, tri_file):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    triples = [line.split('\t<=>\t')[1].strip() for line in lines]
    triples = [triple for triple in triples if triple != '']
    #triples = list(set(triples))

    with open(save_file, 'w') as f:
        for triple in triples:
            f.write('{}\n'.format(triple))

    results = []
    for triple in triples:
        lis = triple.split(';')
        results.extend(lis)

    if tri_file is not None:
        results = list(set(results))
        with open(tri_file, 'w') as f:
            for triple in results:
                f.write('{}\n'.format(triple))

if __name__ == '__main__':
    root_dir = '/slfs1/users/zjz17/SLU/dstc3-st/'
    ontology_file = root_dir + 'traindev/scripts/config/ontology_dstc2.json'
    data_dir = 'manual-da/'

    file_name = root_dir + data_dir + 'dstc3.test'
    save_file = root_dir + data_dir + 'ori/dstc3.test.triples'
    tri_file = root_dir + data_dir + 'tmp.triples'
    file_triples(file_name, save_file, None)

    #train_file = root_dir + data_dir + 'train'
    #save_file = root_dir + data_dir + 'train.name'
    #generate_multi_name_train(train_file, save_file)

    #test_file = root_dir + data_dir + 'test'
    #save_file = root_dir + data_dir + 'tmp.new.value.test'
    #class_train = root_dir + data_dir + 'class.train'


    #generate_dstc2_old_value_test(test_file, save_file, 'food')
    #generate_dstc2_new_value_test(class_train, test_file, save_file, 'food', ontology_file)
    #generate_dstc2_new_slot_test(test_file, save_file, 'name', ontology_file)

