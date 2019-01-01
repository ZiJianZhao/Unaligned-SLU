# -*- coding: utf-8 -*-

import argparse
import os, sys
import codecs
import math
from collections import Counter, defaultdict
import json

root_dir = '../../../dstc3/'
ontology_path = root_dir + 'seed/scripts/config/ontology_dstc3.json'
train_data_dir = root_dir + 'seed/data/'
valid_data_dir = root_dir + 'seed/data/'
test_data_dir = root_dir + 'test/data/'
train_flist_path = root_dir + 'seed/scripts/config/dstc3_seed.flist'
valid_flist_path = root_dir + 'seed/scripts/config/dstc3_seed.flist'
test_flist_path = root_dir + 'test/scripts/config/dstc3_test.flist'
sz128_class_path = root_dir + 'classes/class.sz128'

manual_dir = root_dir + 'manual-da/'
one_best_live_dir = root_dir + '1best-live-da/'

dstc2_root_dir = '../../../dstc2-slu/'

# ======================== text processing ==================================

def process_sent(string):
    lis = string.strip().split()
    lis = [''.join(word.strip().split("'")) for word in lis]
    return lis

def process_word(word):
    word = ''.join(word.strip().split("'"))
    return word

# ======================== general ==================================

def slot2dic(string):
    str_lis = string.split('-', 2)
    if len(str_lis) == 1:
        return {'slots': [], 'act': str_lis[0]}
    elif len(str_lis) == 2:
        return {'slots': [['slot', str_lis[1]]], 'act': str_lis[0]}
    elif len(str_lis) == 3:
        return {'slots': [[str_lis[1], str_lis[2]]], 'act': str_lis[0]}
    else:
        raise Exception('Wrong slot string')

def get_classes_train():

    flists = read_flist(train_flist_path)
    classes = []
    for flist in flists:
        label_json = train_data_dir + flist + '/label.json'
        labels = read_class(label_json)
        for ls in labels:
            classes.extend(ls)

    classes = sorted(list(set(classes)))
    print('Class in train: {}'.format(len(classes)))
    return classes

def get_classes_all():
    classes = ['ack', 'affirm', 'bye', 'hello', 'negate', 'repeat', 
            'reqalts', 'reqmore', 'restart', 'thankyou']
    dic = json.loads(open(ontology_path).read())
    for key in dic:
        if key == 'requestable':
            for value in dic[key]:
                classes.append('request' + '-' +value)
        elif key == 'informable':
            for slot in dic[key]:
                for value in dic[key][slot]:
                    classes.append('inform' + '-' + slot + '-' + value)
                    classes.append('confirm' + '-' + slot + '-' + value)
                    classes.append('deny' + '-' + slot + '-' + value)
                classes.append('inform' + '-' + slot + '-' + 'dontcare')
        else:
            continue
    classes.append('inform-this-dontcare')
    #classes.append('inform-type-restaurant')
    #classes.append('confirm-type-restaurant')
    classes = sorted(list(set(classes)))
    print('Class in all: {}'.format(len(classes)))
    return classes

def read_class(label_json):
    dic = json.loads(open(label_json).read())
    turns = dic['turns']
    classes = []
    for dic in turns:
        labels = []
        sems = dic['semantics']['json']
        for sem in sems:
            slots = sem['slots']
            act = sem['act']
            if len(slots) == 0:
                labels.append(act)
            elif len(slots) == 1:
                for slot in slots:
                    if len(slot) == 1:
                        labels.append(act + '-' + slot[0])
                    elif len(slot) == 2:
                        if act == 'request':
                            labels.append(act + '-' + slot[1])
                        else:
                            labels.append(act + '-' + '-'.join(slot))
                    else:
                        raise Exception("Unexpected Situations")
            else:
                raise Exception("Unexpected Situations")
        classes.append(labels)
    return classes

def judge_classes(classes, classes_all, tag):
    lis = []
    for cls in classes:
        if cls not in classes_all:
            lis.append(cls)
    if len(lis) == 0:
        print('All contains all elements in {}'.format(tag))
    else:
        print('All lack some elements in {}:'.format(tag))
        print(';'.join(lis))

def get_classes(save_dir):

    class_train_save_path = save_dir + 'class.train'
    classes_train = get_classes_train()
    with open(class_train_save_path, 'w') as f:
        for cls in classes_train:
            f.write('{}\n'.format(cls))

    class_all_save_path = save_dir + 'class.all'
    classes_all = get_classes_all()
    with open(class_all_save_path, 'w') as f:
        for cls in classes_all:
            f.write('{}\n'.format(cls))

    judge_classes(classes_train, classes_all, 'train')

    """
    with codecs.open(sz128_class_path, 'r') as f:
        lines = f.readlines()
        classes = [line.strip() for line in lines]
        newes = []
        for cls in classes:
            if cls not in ['confirm-type-restaurant', 'inform-type-restaurant']:
                newes.append(cls)
        classes = newes

    judge_classes(classes, classes_all, 'sz128')
    """
    return classes_train, classes_all

def read_flist(filename):
    with codecs.open(filename, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

# ======================== 1best ==================================

def read_1best_live(log_json):
    dic = json.loads(open(log_json).read())
    turns = dic['turns']
    sents = []
    for dic in turns:
        sent = dic['input']['live']['asr-hyps'][0]['asr-hyp']
        sents.append(sent)
    return sents

def read_1best_batch(log_json):
    dic = json.loads(open(log_json).read())
    turns = dic['turns']
    sents = []
    for dic in turns:
        sent = dic['input']['batch']['asr-hyps'][0]['asr-hyp']
        sents.append(sent)
    return sents

# ======================== Manual ==================================
def read_manual(label_json):
    dic = json.loads(open(label_json).read())
    turns = dic['turns']
    sents = []
    for dic in turns:
        sent = dic['transcription']
        sents.append(sent)
    return sents

# ======================== All ==================================
def get_pairs(data_dir, flist_path, save_path, task):
    """for manual and 1best preprocessing"""

    flists = read_flist(flist_path)
    pairs = []
    for flist in flists:
        log_json = data_dir + flist + '/log.json'
        label_json = data_dir + flist + '/label.json'
        if task == 'manual':
            sents = read_manual(label_json)
        elif task == '1best-live':
            sents = read_1best_live(log_json)
        elif task == '1best-batch':
            sents = read_1best_batch(log_json)
        else:
            raise Exception("Wrong task in get_pairs")
        classes = read_class(label_json)
        assert len(sents) == len(classes)
        
        lis = list(zip(sents, classes))
        pairs.extend(lis)
            
    with open(save_path, 'w') as f:
        for (sent, labels) in pairs:
            f.write('{}\t<=>\t{}\n'.format(sent, ';'.join(labels)))
    return pairs

def get_all_pairs(task):

    if task == 'manual':
        save_dir = manual_dir
        dstc2_dir = dstc2_root_dir + 'manual/'
    elif task == '1best-live':
        save_dir = one_best_live_dir
        dstc2_dir = dstc2_root_dir + '1best-live/'
    else:
        raise Exception("Wrong task in get_all_pairs")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def count(pairs, tag):
        print('===========================')
        print('{} pairs: {}'.format(tag, len(pairs)))
        num_sent = 0
        num_label = 0
        for (sent, labels) in pairs:
            if sent.strip() == '':
                num_sent += 1
            if len(labels) == 0:
                num_label += 1
        print('{} pairs without sents: {}'.format(tag, num_sent))
        print('{} pairs without labels: {}'.format(tag, num_label))

    classes_train, classes_all = get_classes(save_dir)
    print('Classes saved')

    train_pairs = get_pairs(train_data_dir, train_flist_path, save_dir+'seed.train', task)
    count(train_pairs, 'Train')
    
    # ============================= dstc2 =======================================
    def f(n):
        save_path = save_dir + 'dstc2_seed_{}.train'.format(n)
        lines = open(dstc2_dir+'train', 'r').readlines()
        with open(save_path, 'w') as f:
            for line in lines:
                f.write('{}'.format(line))
            for i in range(n):
                for (sent, labels) in train_pairs:
                    f.write('{}\t<=>\t{}\n'.format(sent, ';'.join(labels)))
    f(0)
    f(1)
    f(5)
    f(10)

    def g(n):
        save_path = save_dir + 'dstc2_seed_{}.valid'.format(n)
        lines = open(dstc2_dir+'valid', 'r').readlines()
        with open(save_path, 'w') as f:
            for line in lines:
                f.write('{}'.format(line))
            for i in range(n):
                for (sent, labels) in train_pairs:
                    f.write('{}\t<=>\t{}\n'.format(sent, ';'.join(labels)))
    g(0)
    g(1)

    save_path = save_dir + 'dstc2_seed.class.train'
    classes = open(dstc2_dir+'class.train').readlines()
    classes = [cls.strip() for cls in classes]
    classes = list(set(classes+classes_train))
    with open(save_path, 'w') as f:
        for cls in classes:
            f.write('{}\n'.format(cls))

    save_path = save_dir + 'dstc2_seed.class.all'
    classes = open(dstc2_dir+'class.all').readlines()
    classes = [cls.strip() for cls in classes]
    classes = list(set(classes+classes_all))
    with open(save_path, 'w') as f:
        for cls in classes:
            f.write('{}\n'.format(cls))
    # ===========================================================================

    valid_pairs = get_pairs(valid_data_dir, valid_flist_path, save_dir+'seed.valid', task)
    count(valid_pairs, 'Valid')

    test_pairs = get_pairs(test_data_dir, test_flist_path, save_dir+'test', task)
    count(test_pairs, 'Test')

if __name__ == '__main__':
    get_all_pairs('manual')
    #print("============= 1best batch ===============")
    #get_all_pairs('1best-batch')
    """
    print("============= nbest live ===============")
    get_all_pairs('nbest-live')
    print("============= nbest batch ===============")
    get_all_pairs('nbest-batch')
    print("============= nbest ===============")
    get_test_json(test_data_dir, test_flist_path, n_best_dir+'test.json', 'nbest')
    print('Test jaon file saved')
    """
