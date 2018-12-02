# -*- coding: utf-8 -*-

import argparse
import os, sys
import codecs
import math
from collections import Counter, defaultdict
import json

root_dir = '../../dstc2-slu/'
ontology_path = root_dir + 'traindev/scripts/config/ontology_dstc2.json'
train_data_dir = root_dir + 'traindev/data/'
valid_data_dir = root_dir + 'traindev/data/'
test_data_dir = root_dir + 'test/data/'
train_flist_path = root_dir + 'traindev/scripts/config/dstc2_train.flist'
valid_flist_path = root_dir + 'traindev/scripts/config/dstc2_dev.flist'
test_flist_path = root_dir + 'test/scripts/config/dstc2_test.flist'
sz128_class_path = root_dir + 'classes/class.sz128'

manual_dir = root_dir + 'manual/'
one_best_live_dir = root_dir + '1best-live/'
one_best_batch_dir = root_dir + '1best-batch/'
n_best_live_dir = root_dir + 'nbest-live/'
n_best_batch_dir = root_dir + 'nbest-batch/'


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
    str_lis = string.split('-')
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

    """
    judge_classes(classes_train, classes_all, 'train')

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

# ======================== nbest ==================================

def read_nbest_live(log_json):
    dic = json.loads(open(log_json).read())
    turns = dic['turns']
    sents = []
    for dic in turns:
        sent = dic['input']['live']['asr-hyps']
        sents.append(sent)
    return sents

def read_nbest_batch(log_json):
    dic = json.loads(open(log_json).read())
    turns = dic['turns']
    sents = []
    for dic in turns:
        sent = dic['input']['batch']['asr-hyps']
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
        elif task == 'nbest-live':
            sents = read_nbest_live(log_json)
        elif task == 'nbest-batch':
            sents = read_nbest_batch(log_json)
        else:
            raise Exception("Wrong task in get_pairs")
        classes = read_class(label_json)
        assert len(sents) == len(classes)
        if task.startswith('nbest'):
            for i in range(len(sents)):
                pair = {'nbest': sents[i], 'label': ';'.join(classes[i])}
                pairs.append(pair)
        else:
            lis = list(zip(sents, classes))
            pairs.extend(lis)
            
    if task.startswith('nbest'):
        data = {'pairs': pairs}
        string = json.dumps(data, sort_keys=True, indent=4, separators=(',', ':'))
        with open(save_path, 'w') as f:
            f.write(string)
        return pairs
    else:
        with open(save_path, 'w') as f:
            for (sent, labels) in pairs:
                f.write('{}\t<=>\t{}\n'.format(sent, ';'.join(labels)))
        return pairs


def get_test_json(data_dir, flist_path, save_path, task):

    flists = read_flist(flist_path)
    sessions = {'sessions': []}
    for flist in flists:
        session = {}
        log_json = data_dir + flist + '/log.json'
        label_json = data_dir + flist + '/label.json'
        if task == 'manual':
            dic = json.loads(open(label_json).read())
        elif task in ['1best-live', '1best-batch', 'nbest-live', 'nbest-batch']:
            dic = json.loads(open(log_json).read())
        else:
            raise Exception("Wrong task in get_test_json")

        session['session-id'] = dic['session-id']
        session['turns'] = []
        lis = dic['turns']
        for dic in lis:
            if task == 'manual':
                sent = dic['transcription']
                asr_hyps = [{"asr-hyp": sent, "score": 1.0}]
            elif task == '1best-live':
                sent = dic['input']['live']['asr-hyps'][0]['asr-hyp']
                asr_hyps = [{"asr-hyp": sent, "score": 1.0}]
            elif task == '1best-batch':
                sent = dic['input']['batch']['asr-hyps'][0]['asr-hyp']
                asr_hyps = [{"asr-hyp": sent, "score": 1.0}]
            elif task == 'nbest-live':
                asr_hyps = dic['input']['live']['asr-hyps']
            elif task == 'nbest-batch':
                asr_hyps = dic['input']['batch']['asr-hyps']
            else:
                raise Exception("Wrong task in get_test_json")
            session['turns'].append({"asr-hyps": asr_hyps})
        sessions['sessions'].append(session)
    string = json.dumps(sessions, sort_keys=True, indent=4, separators=(',', ':'))
    with open(save_path, 'w') as f:
        f.write(string)

def get_all_pairs(task):

    if task == 'manual':
        save_dir = manual_dir
    elif task == '1best-live':
        save_dir = one_best_live_dir
    elif task == '1best-batch':
        save_dir = one_best_batch_dir
    elif task == 'nbest-live':
        save_dir = n_best_live_dir
    elif task == 'nbest-batch':
        save_dir = n_best_batch_dir
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

    def count_nbest(pairs, tag):
        print('===========================')
        print('{} pairs: {}'.format(tag, len(pairs)))
        dic = defaultdict(int)
        for pair in pairs:
            dic[len(pair['nbest'])] += 1
        print('{} pairs nbest number distribution:'.format(tag))
        print(dic)

    _, _ = get_classes(save_dir)
    print('Classes saved')

    train_pairs = get_pairs(train_data_dir, train_flist_path, save_dir+'train', task)
    if task.startswith('nbest'):
        count_nbest(train_pairs, 'Train')
    else:
        count(train_pairs, 'Train')

    valid_pairs = get_pairs(valid_data_dir, valid_flist_path, save_dir+'valid', task)
    if task.startswith('nbest'):
        count_nbest(valid_pairs, 'Valid')
    else:
        count(valid_pairs, 'Valid')

    test_pairs = get_pairs(test_data_dir, test_flist_path, save_dir+'test', task)
    if task.startswith('nbest'):
        count_nbest(test_pairs, 'Test')
    else:
        count(test_pairs, 'Test')

    get_test_json(test_data_dir, test_flist_path, save_dir+'test.json', task)
    print('Test json file saved.')

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
