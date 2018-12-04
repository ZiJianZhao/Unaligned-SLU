# -*- coding: utf-8 -*-

import argparse
import os, sys
import codecs
import math
from collections import Counter, defaultdict
import json

import math

root_dir = '../../dstc2-slu/'
ontology_path = root_dir + 'traindev/scripts/config/ontology_dstc2.json'
train_data_dir = root_dir + 'traindev/data/'
valid_data_dir = root_dir + 'traindev/data/'
test_data_dir = root_dir + 'test/data/'
train_flist_path = root_dir + 'traindev/scripts/config/dstc2_train.flist'
valid_flist_path = root_dir + 'traindev/scripts/config/dstc2_dev.flist'
test_flist_path = root_dir + 'test/scripts/config/dstc2_test.flist'
sz128_class_path = root_dir + 'classes/class.sz128'

wcn_dir = root_dir + 'wcn/'

# ======================== general ==================================

class2vars = {
        'addr': ['addr', 'address'],
        'pricerange': ['pricerange', 'price', 'priced', 'range'],
        'postcode': ['postcode', 'post', 'code'],
        'bye': ['bye', 'goodbye'],
        'thankyou': ['thankyou', 'thank', 'you'],
        'dontcare': ['dontcare', "doesn't", "does", "not", "don't", "dont", "care"],
        'phone': ['phone', 'number'], 
        'moderate': ['moderate', 'moderately']
        }

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

    slu_words = []
    for word in classes_all:
        word_lis = word.strip().split('-')
        for word in word_lis:
            slu_words.extend(word.strip().split())

    for key in class2vars:
        slu_words.extend(class2vars[key])

    slu_words = set(slu_words) - set(['the'])
    return list(slu_words)


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

def read_flist(filename):
    with codecs.open(filename, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

# ======================== 1best ==================================

def read_1best(log_json):
    dic = json.loads(open(log_json).read())
    turns = dic['turns']
    sents = []
    for dic in turns:
        sent = dic['input']['live']['asr-hyps'][0]['asr-hyp']
        #sent = dic['input']['batch']['asr-hyps'][0]['asr-hyp']
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


# ======================== CN ==================================
def read_cn(log_json):
    dic = json.loads(open(log_json).read())
    turns = dic['turns']
    cnets = []
    for dic in turns:
       cnet = dic['input']['batch']['cnet']
       cnets.append(cnet)
    return cnets

def class2words(classes):
    words = []
    for cls in classes:
        word_lis = cls.strip().split('-')
        for word in word_lis:
            word = word.strip()
            if word in class2vars:
                words.extend(class2vars[word])
            words.extend(word.split())
        words.extend(word_lis)
    return list(set(words))

def read_nbest_words(tag):
    filename = n_best_dir + tag
    interjections = set(['ah', 'uh', 'oh', 'um'])
   
    words = []
    pairs = json.loads(open(filename).read())['pairs']
    for pair in pairs:
        sents = pair['nbest']
        for sent in sents:
            words.extend(sent['asr-hyp'].strip().split())
    words = list(set(words) - interjections)
    words = {w : 0 for w in words}
    return words

def process_cnet(cnet, sent, ontology_words, classes, nbest_words):

    def extract_manual(cnet, sent, repeat=False):
        lis = sent.strip().split()
        dic = {word: [] for word in lis}
        for j in range(len(cnet)):
            for k in range(len(cnet[j]['arcs'])):
                word = cnet[j]['arcs'][k]['word']
                score = cnet[j]['arcs'][k]['score']
                if word in dic:
                    dic[word].append((j, k, score))
        ids = []
        pre = -1
        for word in lis:
            tmp = [tp for tp in dic[word] if tp[0] > pre]
            if len(tmp) > 0:
                if repeat:
                    ids.extend(tmp)
                else:
                    ids.append(tmp[0])
                pre = tmp[0][0]
        return ids

    if task == 'base':
        pass
    elif task == 'skip-rnn-manual-slu':
        _, lis = extract_unskip_steps(cnet, sent, classes)
        new_cnet = []
        for i in lis:
            new_cnet.append(cnet[i])
        cnet = new_cnet
    elif task == 'keep-nbest-words':  # remove interjections
        new_cnet = []
        for ct in cnet:
            new_arcs = []
            for arc in ct['arcs']:
                if arc['word'] in nbest_words:
                    if math.exp(arc['score']) > 0.001:
                        new_arcs.append(arc)
            if len(new_arcs) > 0:
                new_ct = ct
                new_ct['arcs'] = new_arcs
                new_cnet.append(new_ct)
        cnet = new_cnet
    elif task == 'all-sur-skip':
        pass
    elif task == 'all-sur-awsr':
        pass
    elif task == 'all-sur-all':
        pass
    elif task == 'manual-path':
        new_cnet = []
        ids = extract_manual(cnet, sent)
        step = 0
        for j in range(len(cnet)):
            new_cnet.append({"arcs": [], "end": cnet[j]['end'], "start": cnet[j]['start']})
            if step < len(ids) and j == ids[step][0]:
                new_cnet[-1]['arcs'].append(cnet[j]['arcs'][ids[step][1]])
                step += 1
            else:
                arc = sorted(cnet[j]['arcs'], key=lambda d: d['score'])[-1]
                new_cnet[-1]['arcs'].append(arc)
        cnet = new_cnet
    elif task == 'manual-path-norm':
        new_cnet = []
        ids = extract_manual(cnet, sent)
        step = 0
        for j in range(len(cnet)):
            new_cnet.append({"arcs": [], "end": cnet[j]['end'], "start": cnet[j]['start']})
            if step < len(ids) and j == ids[step][0]:
                new_cnet[-1]['arcs'].append(cnet[j]['arcs'][ids[step][1]])
                step += 1
            else:
                arc = sorted(cnet[j]['arcs'], key=lambda d: d['score'])[-1]
                new_cnet[-1]['arcs'].append(arc)
        cnet = new_cnet
        for j in range(len(cnet)):
           cnet[j]['arcs'][0]['score'] = 0.0
    elif task == 'manual-cnet-all':
        new_cnet = []
        ids = extract_manual(cnet, sent, True)
        ids = sorted(list(set([t[0] for t in ids])))
        for tp in ids:
            new_cnet.append(cnet[tp])
        cnet = new_cnet
    elif task == 'ontology-cnet-all':
        new_cnet = []
        tp = []
        for j in range(len(cnet)):
            for k in range(len(cnet[j]['arcs'])):
                word = cnet[j]['arcs'][k]['word']
                if word in ontology_words:
                    tp.append((j,k))
        ids = sorted(list(set([t[0] for t in tp])))
        for tp in ids:
            new_cnet.append(cnet[tp])
        cnet = new_cnet
    elif task == 'ontology-cnet-all-rescore':
        new_cnet = []
        ids = []
        for j in range(len(cnet)):
            for k in range(len(cnet[j]['arcs'])):
                word = cnet[j]['arcs'][k]['word']
                if word in ontology_words:
                    ids.append((j,k))
        ids = sorted(list(set(ids)))
        prev = -1
        for tp in ids:
            if tp[0] != prev:
                prev = tp[0]
                new_cnet.append(cnet[tp[0]])
                new_cnet[-1]['arcs'][tp[1]]['score'] = 0.0
            else:
                new_cnet[-1]['arcs'][tp[1]]['score'] = 0.0
        cnet = new_cnet
    elif task == 'manual-cnet-all-rescore':
        new_cnet = []
        ids = extract_manual(cnet, sent, True)
        ids = sorted(list(set(ids)))
        prev = -1
        for tp in ids:
            if tp[0] != prev:
                prev = tp[0]
                new_cnet.append(cnet[tp[0]])
                new_cnet[-1]['arcs'][tp[1]]['score'] = 0.0
            else:
                new_cnet[-1]['arcs'][tp[1]]['score'] = 0.0
        cnet = new_cnet
    elif task == 'manual-cnet':
        new_cnet = []
        ids = extract_manual(cnet, sent)
        for tp in ids:
            new_cnet.append(cnet[tp[0]])
        cnet = new_cnet
    elif task == 'manual-cnet-rescore':
        new_cnet = []
        ids = extract_manual(cnet, sent)
        for tp in ids:
            new_cnet.append(cnet[tp[0]])
            new_cnet[-1]['arcs'][tp[1]]['score'] = 0.0
        cnet = new_cnet
    elif task == 'partial':
        new_cnet = []
        for ct in cnet:
            flag = True
            for arc in ct['arcs']:
                if arc['word'] == '!null':
                    if arc['score'] > -0.04:
                        flag = False
                        break
            if flag:
                new_cnet.append(ct)
        cnet = new_cnet
    elif task == 'base-avg-prob':
        for j in range(len(cnet)):
            num = len(cnet[j]['arcs'])
            for k in range(num):
                cnet[j]['arcs'][k]['score'] = math.log(1.0/num)
    elif task == 'only-manual':
        lis = sent.strip().split()
        for j in range(len(cnet)):
            for k in range(len(cnet[j]['arcs'])):
                word = cnet[j]['arcs'][k]['word']
                if word not in lis:
                    cnet[j]['arcs'][k]['score'] = float('-inf')
    elif task == 'improve-manual':
        lis = sent.strip().split()
        for j in range(len(cnet)):
            for k in range(len(cnet[j]['arcs'])):
                word = cnet[j]['arcs'][k]['word']
                if word in lis:
                    cnet[j]['arcs'][k]['score'] = 0.0
    elif task == 'only-label-slu':
        lis = class2words(classes)
        for j in range(len(cnet)):
            for k in range(len(cnet[j]['arcs'])):
                word = cnet[j]['arcs'][k]['word']
                if word not in lis:
                    cnet[j]['arcs'][k]['score'] = float('-inf')
    elif task == 'improve-label-slu':
        lis = class2words(classes)
        for j in range(len(cnet)):
            for k in range(len(cnet[j]['arcs'])):
                word = cnet[j]['arcs'][k]['word']
                if word in lis:
                    cnet[j]['arcs'][k]['score'] = 0.0
    elif task == 'only-ontology':
        lis = ontology_words
        for j in range(len(cnet)):
            for k in range(len(cnet[j]['arcs'])):
                word = cnet[j]['arcs'][k]['word']
                if word not in lis:
                    cnet[j]['arcs'][k]['score'] = float('-inf')
    elif task == 'improve-ontology':
        lis = ontology_words
        for j in range(len(cnet)):
            for k in range(len(cnet[j]['arcs'])):
                word = cnet[j]['arcs'][k]['word']
                if word in lis:
                    cnet[j]['arcs'][k]['score'] = 0.0
    elif task == 'only-manual-partial':
        lis = sent.strip().split()
        for j in range(len(cnet)):
            for k in range(len(cnet[j]['arcs'])):
                word = cnet[j]['arcs'][k]['word']
                if word not in lis:
                    cnet[j]['arcs'][k]['score'] = float('-inf')
        new_cnet = []
        for j in range(len(cnet)):
            prob = 0.
            for k in range(len(cnet[j]['arcs'])):
                score = cnet[j]['arcs'][k]['score']
                prob += math.exp(score)
            if prob > 0.001:
                new_cnet.append(cnet[j])
        if len(new_cnet) == 0:
            new_cnet = cnet[0:1]
        cnet = new_cnet
    else:
        raise Exception('Undefined task type.')
    return cnet

def extract_unskip_steps(cnet, sent, classes):
    lis = sent.strip().split()
    words = class2words(classes)
    words.extend(lis)
    words = list(set(words))

    # just for speed up
    lis = [0] * len(cnet)
    dic = {word: 0 for word in words}
    for j in range(len(cnet)):
        for k in range(len(cnet[j]['arcs'])):
            word = cnet[j]['arcs'][k]['word']
            if word in dic:
                lis[j] = 1
    ids = []
    for i in range(len(lis)):
        if lis[i] == 1:
            ids.append(i)
    return lis, ids

def get_cn_pairs(data_dir, flist_path, save_path, ontology_words, tag):
    """for manual and 1best preprocessing"""

    if task == 'keep-nbest-words':  # remove interjections
        nbest_words = read_nbest_words(tag)
    else:
        nbest_words = None

    flists = read_flist(flist_path)
    pairs = {'pairs': []}
    cnet_num = 0.
    arcs_num = 0.
    num = 0.
    tt = 0.
    for flist in flists:
        log_json = data_dir + flist + '/log.json'
        label_json = data_dir + flist + '/label.json'
        cnets = read_cn(log_json)
        sents = read_manual(label_json)
        classes = read_class(label_json)
        assert len(cnets) == len(classes)
        for i in range(len(cnets)):

            cnet = process_cnet(cnets[i], sents[i], ontology_words, classes[i], nbest_words)
            num += 1
            cnet_num += len(cnet)
            arcs_num += sum([len(ct['arcs']) for ct in cnet])
            if len(cnet) == 0:
                tt += 1
            if task == 'all-sur-skip':
                lis, _ = extract_unskip_steps(cnets[i], sents[i], classes[i])
                pair = {'cnet': cnet, 'label': ';'.join(classes[i]), 'manual': sents[i], 'steps': lis}
            else:
                pair = {'cnet': cnet, 'label': ';'.join(classes[i]), 'manual': sents[i]}
            pairs['pairs'].append(pair)
    print(tag)
    print('cnet average length: {}/{}, {}'.format(cnet_num, num, cnet_num/num))
    print('arcs average number: {}/{}, {}'.format(arcs_num, cnet_num, arcs_num/cnet_num))

    string = json.dumps(pairs, sort_keys=True, indent=4, separators=(',', ':'))
    with open(save_path, 'w') as f:
        f.write(string)
    return pairs

def get_cn_test_json(data_dir, flist_path, save_path, ontology_words):

    if task == 'keep_nbest_words':
        nbest_words = read_nbest_words('test')
    else:
        nbest_words = None

    flists = read_flist(flist_path)
    sessions = {'sessions': []}
    for flist in flists:
        session = {}
        log_json = data_dir + flist + '/log.json'
        label_json = data_dir + flist + '/label.json'
        dic = json.loads(open(log_json).read())

        sents = read_manual(label_json)
        classes = read_class(label_json)

        session['session-id'] = dic['session-id']
        session['turns'] = []
        lis = dic['turns']
        for i in range(len(lis)):
            dic = lis[i]
            asr_hyps = dic['input']['batch']['cnet']

            asr_hyps = process_cnet(asr_hyps, sents[i], ontology_words, classes[i], nbest_words)

            session['turns'].append({"asr-hyps": asr_hyps})
        sessions['sessions'].append(session)
    string = json.dumps(sessions, sort_keys=True, indent=4, separators=(',', ':'))
    with open(save_path, 'w') as f:
        f.write(string)

def get_cn_all_pairs(tk):

    global task
    task = tk
    cn_dir = root_dir + 'wcn' + '-' + task + '/'
    if not os.path.exists(cn_dir):
        os.makedirs(cn_dir)
    save_dir = cn_dir

    print('CN preprocessing ...')

    ontology_words = get_classes(save_dir)
    print('Classes saved')

    train_pairs = get_cn_pairs(train_data_dir, train_flist_path, save_dir+'train', ontology_words, 'train')
    valid_pairs = get_cn_pairs(valid_data_dir, valid_flist_path, save_dir+'valid', ontology_words, 'valid')
    test_pairs = get_cn_pairs(test_data_dir, test_flist_path, save_dir+'test', ontology_words, 'test')

    get_cn_test_json(test_data_dir, test_flist_path, save_dir+'test.json', ontology_words)
    print('Test json file saved.')

def count_cnet(cnet, sent, ontology_words, classes, nbest_words):
    scores = []
    num = 0.
    lis = sent.strip().split()
    for j in range(len(cnet)):
        for k in range(len(cnet[j]['arcs'])):
            word = cnet[j]['arcs'][k]['word']
            if word in lis:
                scores.append((word,math.exp(cnet[j]['arcs'][k]['score'])))
                num += 1
    on_scores = []
    on_num = 0.
    for j in range(len(cnet)):
        for k in range(len(cnet[j]['arcs'])):
            word = cnet[j]['arcs'][k]['word']
            if word in ontology_words:
                on_scores.append((word,math.exp(cnet[j]['arcs'][k]['score'])))
                on_num += 1
    cha_scores = []
    cha_num = 0.
    for j in range(len(cnet)):
        for k in range(len(cnet[j]['arcs'])):
            word = cnet[j]['arcs'][k]['word']
            if word in ontology_words and word not in lis:
                cha_scores.append((word,math.exp(cnet[j]['arcs'][k]['score'])))
                cha_num += 1
    
    """
    if len(cha_scores) > 0:
        print("===========================")
        print(cnet)
        print("===========================")
        print(scores)
        print("===========================")
        print(on_scores)
        print("===========================")
        print(cha_scores)
        input()
    """
    return cnet, scores, num, on_scores, on_num, cha_scores, cha_num

def count_cn_pairs(data_dir, flist_path, save_path, ontology_words, tag):
    """for manual and 1best preprocessing"""

    nbest_words = read_nbest_words(tag)

    flists = read_flist(flist_path)
    pairs = {'pairs': []}
    cnet_num = 0.
    arcs_num = 0.
    num = 0.
    manual_scores = []
    manual_num = 0.
    onto_scores = []
    onto_num = 0.
    cha_scores = []
    cha_num = 0.
    for flist in flists:
        log_json = data_dir + flist + '/log.json'
        label_json = data_dir + flist + '/label.json'
        cnets = read_cn(log_json)
        sents = read_manual(label_json)
        classes = read_class(label_json)
        assert len(cnets) == len(classes)
        for i in range(len(cnets)):

            cnet, m_scores, m_num, o_scores, o_num, c_scores, c_num = count_cnet(cnets[i], sents[i], ontology_words, classes[i], nbest_words)
            manual_scores.extend(m_scores)
            manual_num += m_num
            onto_scores.extend(o_scores)
            onto_num += o_num
            cha_scores.extend(c_scores)
            cha_num += c_num

            num += 1
            cnet_num += len(cnet)
            arcs_num += sum([len(ct['arcs']) for ct in cnet])
    print(tag)
    print('cnet average length: {}/{}, {}'.format(cnet_num, num, cnet_num/num))
    print('arcs average number: {}/{}, {}'.format(arcs_num, cnet_num, arcs_num/cnet_num))
    print('manual average words: {}/{}, {}'.format(manual_num, num, manual_num / num))
    print('onto average words: {}/{}, {}'.format(onto_num, num, onto_num / num))
    print('cha average words: {}/{}, {}'.format(cha_num, num, cha_num / num))
    
    words = [s[0] for s in manual_scores if s[1] < 0.001]
    words = Counter(words)

    onto_words = [s[0] for s in onto_scores if s[1] < 0.001]
    onto_words = Counter(words)

    cha_words = [s[0] for s in cha_scores]
    cha_words = Counter(words)

    big = [1 for s in manual_scores if s[1] > 0.01]
    print('manual scores high number: {}/{}, {}'.format(len(big), len(manual_scores), len(big)/len(manual_scores)))

def count_cn_all_pairs(tk):

    global task
    task = tk
    cn_dir = root_dir + 'cn' + '-' + task + '/'
    save_dir = cn_dir

    ontology_words = get_classes(save_dir)
    print('Classes saved')
    train_pairs = count_cn_pairs(train_data_dir, train_flist_path, save_dir+'train', ontology_words, 'train')
    valid_pairs = count_cn_pairs(valid_data_dir, valid_flist_path, save_dir+'valid', ontology_words, 'valid')
    test_pairs = count_cn_pairs(test_data_dir, test_flist_path, save_dir+'test', ontology_words, 'test')


if __name__ == '__main__':
    print("============= cn ===============")
    get_cn_all_pairs('partial')
    #count_cn_all_pairs('only-manual')
