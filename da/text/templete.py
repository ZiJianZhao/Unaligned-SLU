# -*- coding: utf-8 -*-

import argparse
import os, sys, random
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

random.seed(1234)

root_dir = '../../../dstc2-slu/'
glove_path = '../../../../NLPData/glove.6B/glove.6B.100d.txt'
stop_words = ['the', 'all', ]

dstc2_templates = {
    'ack': ['okay', 'ok'],
    'hello': ['hello'],
    'affirm': ['yes'],
    'bye': ['goodbye', 'good bye', 'bye'],
    'negate': ['no'],
    'repeat': ['repeat that'],
    'reqalts': ['how about', 'what about', 'anything else', 'what else'],
    'reqmore': ['more'],
    'restart': ['start over'],
    'thankyou': ['thank you'],

    'request': ['', 'what is', 'may i have'],
    'inform': [''],
    'confirm': ['is it', 'is that', 'is there', 'is this'],
    'deny': ['not', 'hate', 'dont want', 'fuck'],

    'addr': ['address'],
    'area': ['area'],
    'food': ['food', 'type of food'],
    'name': ['name'],
    'phone': ['phone number'],
    'postcode': ['postcode', 'post code'],
    'pricerange': ['price range'],
    'signature': ['signature'],

    'food-XX': ['XX', 'XX food', 'serves XX food'],
    'name-XX': ['XX', 'for XX', 'of XX'],
    'area-XX': ['XX', 'in the XX part of town', 'in the XX'],
    'pricerange-XX': ['XX', 'XX restaurant'],

    'dontcare': ['dont care', 'any', 'do not care', 'doesnt matter']
    }

#'moderate': ['moderately']

dstc3_templates = {

    'hastv': ['does it has a tv', 'does it has a television'],
    'childrenallowed': ['does it allow children'],
    'price': ['the price'],
    'hasinternet': ['does it have an internet connection'],
    'near': ['near'],
    'type': ['type of'],

    'hastv-true': ['has a television', 'with a television', 'has a tv', 'with a tv'],
    'hastv-false': ['not has a television', 'not with a television', 'not has a tv', 'not with a tv'],
    'childrenallowed-true': ['allow children', 'children'],
    'childrenallowed-false': ['not allow children', 'with no children', 'dont want children'],
    'hasinternet-true': ['with internet connection', 'has internet connection', 'with internet', 'has internet'],
    'hasinternet-false': ['no internet connection', 'no internet'],
    'near-XX': ['XX'],
    'type-XX': ['XX'],

    'area-XX': ['XX', 'XX area'],
    'pricerange-XX': ['XX', 'XX price range'],

    'coffeeshop': ['coffee shop', 'coffee', 'cafe']
    }

def generate_new_templates(old_templates, new_templates):
    result = copy.deepcopy(new_templates)
    for key in old_templates:
        if key not in result:
            result[key] = old_templates[key]
    return result

def inform_XX_dontcare_rules(slots, templates):
    results = []
    values = templates['dontcare']
    for slot in slots:
        for utt in templates[slot]:
            for value in values:
                results.append((value+' '+utt, 'inform-'+slot+'-dontcare'))
    for value in values:
        results.append((value, 'inform-this-dontcare'))
    return results

def value_filling_rules(utt, triple, values, times=None):

    def get_values(triple, tag):
        position = triple.find(tag)
        if position > 0:
            slot = triple[:position].split('-')[-2]
            value = values[slot]
        else:
            value = ['hehe']
        return value

    xx_values = get_values(triple, 'XX')
    yy_values = get_values(triple, 'YY')

    results = []

    if times is None:
        for x in xx_values:
            for y in yy_values:
                rutt = utt.replace('XX', x).replace('YY', y)
                rtri = triple.replace('XX', x).replace('YY', y)
                results.append((rutt, rtri))
    else:
        for i in range(times):
            x = random.choice(xx_values)
            y = random.choice(yy_values)
            rutt = utt.replace('XX', x).replace('YY', y)
            rtri = triple.replace('XX', x).replace('YY', y)
            results.append((rutt, rtri))

    resuls = list(set(results))
    return results

def triple_to_utterance(triple, templates):
    lis = triple.split('-', 2)
    results = []
    if len(lis) == 1:
        results.extend(templates[lis[0]])
    elif len(lis) == 2:
        for act_utt in templates[lis[0]]:
            for slot_utt in templates[lis[1]]:
                results.append(act_utt + ' ' + slot_utt)
    else:
        if lis[2] == 'dontcare':
            assert lis[0] == 'inform'
            values = templates['dontcare']
            if lis[1] == 'this':
                results.extend(values)
            else:
                for utt in templates[lis[1]]:
                    for val in values:
                        results.append(val + ' ' + utt)
        elif lis[2] == 'true' or lis[2] == 'false':
            assert lis[0] == 'inform'
            values = templates[lis[1]+'-'+lis[2]]
            for utt in values:
                results.append(utt)
        else:
            for act_utt in templates[lis[0]]:
                for slot_utt in templates[lis[1]+'-XX']:
                    string = act_utt + ' ' + slot_utt
                    string = string.replace('XX', lis[2])
                    results.append(string)
    return results

def triples_to_utterances(triples, templates):
    triple_lis = triples.split(';')
    results = ['']
    for triple in triple_lis:
        tmp = triple_to_utterance(triple, templates)
        new = []
        for ix in results:
            for iy in tmp:
                new.append(ix + ' ' + iy)
        results = new
    return results

def save_utterance_triple(results, save_path, repeat=False):

    if not repeat:
        results = list(set(results))
    print('Lines number: {}'.format(len(results)))
    triples = []
    for (utt, tri) in results:
        triples.extend(tri.split(';'))
    triples = list(set(triples))

    random.shuffle(results)
    random.shuffle(triples)

    with open(save_path, 'w') as f:
        for (utt, tri) in results:
            f.write('{}\t<=>\t{}\n'.format(utt.strip(), tri))

    #with open(save_dir+'class.train', 'w') as f:
    #    for tri in triples:
    #        f.write('{}\n'.format(tri))


def file_triples_generation_rules(file_name, templates, save_path, repeat=True):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    triples = [line.split('\t<=>\t')[1].strip() for line in lines]
    triples = [triple for triple in triples if triple != '']
    if not repeat:
        triples = list(set(triples))
    #news = []
    #for triple in triples:
    #    news.extend(triple.split(';'))
    #triples = list(set(triples+news))
    results = []
    for triple in triples:
        utterances = triples_to_utterances(triple, templates)
        if repeat:
            utt = random.choice(utterances)
            results.append((utt, triple))
        else:
            for utt in utterances:
                results.append((utt, triple))

    save_utterance_triple(results, save_path, repeat)

def triple_combination_rules(ontology_path, templates, mode, save_path):

    single_acts = ['ack', 'hello', 'affirm', 'bye', 'negate', 'repeat', 'reqalts', 'reqmore', 'restart', 'thankyou']
    double_acts = ['request']
    triple_acts = ['inform', 'deny', 'confirm']

    ontology = json.loads(open(ontology_path).read())
    double_slots = ontology['requestable']
    triple_slots = list(ontology['informable'].keys())
    special_triple_slots = ['hastv', 'childrenallowed', 'hasinternet']

    values = ontology['informable']

    # single triple
    dontcares = inform_XX_dontcare_rules(triple_slots, templates)

    single_act_triples = []
    for act in single_acts:
        for utt in templates[act]:
            single_act_triples.append((utt, act))

    double_act_triples = []
    for act in double_acts:
        for act_utt in templates[act]:
            for slot in double_slots:
                for slot_utt in templates[slot]:
                    double_act_triples.append((act_utt+' '+slot_utt, act+'-'+slot))

    triple_act_triples = []
    for act in triple_acts:
        for act_utt in templates[act]:
            for slot in triple_slots:
                if slot in special_triple_slots:
                    slots = [slot+'-true', slot+'-false']
                    for st in slots:
                        for slot_utt in templates[st]:
                            utt = act_utt + ' ' + slot_utt
                            tri = act + '-' + st
                            triple_act_triples.append((utt, tri))
                else:
                    slot_XX = slot+'-XX'
                    for slot_utt in templates[slot_XX]:
                        utt = act_utt + ' ' + slot_utt
                        tri = act + '-' + slot_XX
                        tmp_triples = value_filling_rules(utt, tri, values)
                        triple_act_triples.extend(tmp_triples)

    single_triples = single_act_triples + double_act_triples + triple_act_triples

    def f(tri1s, tri2s):
        results = []
        max_num = max(len(tri1s), len(tri2s))
        min_num = min(len(tri1s), len(tri2s))
        while (len(results) < min_num):
            random.shuffle(tri1s)
            random.shuffle(tri2s)
            for i in range(0, max_num):
                ix = i % len(tri1s)
                iy = i % len(tri2s)
                x = tri1s[ix]
                y = tri2s[iy]
                if x[1] != y[1]:
                    results.append((x[0]+' '+y[0], x[1]+';'+y[1]))
        results = list(set(results))
        random.shuffle(results)
        results = results[:min_num]
        return results

    complex_triples = []
    triples = [single_act_triples, double_act_triples, triple_act_triples]
    for i in range(len(triples)):
        for j in range(i, len(triples)):
            tmp1 = copy.deepcopy(triples[i])
            tmp2 = copy.deepcopy(triples[j])
            tmp = f(tmp1, tmp2)
            complex_triples.extend(tmp)

    if mode == 1:
        results = dontcares + single_triples
    else:
        results = dontcares + single_triples + complex_triples

    save_utterance_triple(results, save_path)

def generate_templete(train_file, save_file):
    with open(train_file, 'r') as f:
        lines = f.readlines()
    results = []
    for line in lines:
        lis = line.split('\t<=>\t')
        utterance = lis[0].strip()
        triples = lis[1].strip()
        if triples == '':
            new_line = '{}\t<=>\t{}\t<=>\t{}'.format(utterance, ' ', ' ')
        else:
            class_string = random.choice(triples_to_utterances(triples, dstc2_templates))
            #class_string = triples2templete(triples)
            new_line = '{}\t<=>\t{}\t<=>\t{}'.format(utterance, triples, class_string)
        results.append(new_line)
    with open(save_file, 'w') as f:
        for line in results:
            f.write('{}\n'.format(line))


if __name__ == '__main__':
    dir_name = '/slfs1/users/zjz17/SLU/dstc3-st/'

    templates = generate_new_templates(dstc2_templates, dstc3_templates)

    """
    triple_combination_rules(
        dir_name + 'scripts/config/ontology_dstc3.json',
        templates,
        2,
        dir_name + 'manual/tpe2'
    )

    """
    file_triples_generation_rules(
        dir_name + 'manual/test',
        templates,
        dir_name + 'manual/tpe3v1',
        False
    )

"""
def templete(triple):
    lis = triple.strip().split('-', 2)
    if len(lis) == 3:
        value = lis[2]
    if triple == 'ack':
        result = 'okay'
    elif triple == 'hello':
        result = 'hello'
    elif triple == 'affirm':
        result = 'yes'
    elif triple == 'bye':
        result = 'good bye'
    elif triple == 'negate':
        result = 'no'
    elif triple == 'repeat':
        result = 'repeat that'
    elif triple == 'reqalts':
        result = 'how about'
    elif triple == 'reqmore':
        result = 'more'
    elif triple == 'restart':
        result = 'start over'
    elif triple == 'thankyou':
        result = 'thank you'
    elif triple == 'request-addr':
        result = 'the address'
    elif triple == 'request-area':
        result = 'the area'
    elif triple == 'request-food':
        result = 'type of food'
    elif triple == 'request-name':
        result = 'the name'
    elif triple == 'request-phone':
        result = 'the phone number'
    elif triple == 'request-postcode':
        result =  'the post code'
    elif triple == 'request-pricerange':
        result = 'price range'
    elif triple == 'request-signature':
        result = 'signature'
    elif triple.startswith('inform-food'):
        result = value + ' ' + 'food'
    elif triple.startswith('inform-name'):
        result = value + ' ' + 'name'
    elif triple.startswith('inform-area'):
        result = 'in the' + ' ' + value + ' ' + 'area'
    elif triple.startswith('inform-pricerange'):
        result = value + ' ' + 'price range'
    elif triple.startswith('deny-food'):
        result = 'not' + ' ' + value + ' ' + 'food'
    elif triple.startswith('deny-name'):
        result = 'not' + ' ' + value + ' ' + 'name'
    elif triple.startswith('deny-area'):
        result = 'not' + ' ' +  'in the' + ' ' + value + ' ' + 'area'
    elif triple.startswith('deny-pricerange'):
        result = 'not' + ' ' + value + ' ' + 'price range'
    elif triple.startswith('confirm-food'):
        result = 'is the food' + ' ' + value
    elif triple.startswith('confirm-name'):
        result = 'is the name' + ' ' + value
    elif triple.startswith('confirm-area'):
        result = 'is in the area of' + ' ' + value
    elif triple.startswith('confirm-pricerange'):
        result = 'is the price range' + ' ' + value
    elif triple == 'inform-this-dontcare':
        result = 'dont care'
    else:
        print(triple)
        raise Exception('Unknown triple')

    return result

def triples2templete(class_string):
    lis = class_string.strip().split(';')
    res = []
    for triple in lis:
        res.append(templete(triple))
    string = ' ; '.join(res)
    return string
"""

"""
if __name__ == '__main__':
    dir_name = root_dir + 'manual-templete/'

    train_file = dir_name + 'train'
    save_file = dir_name + 'train.templete'
    generate_templete(train_file, save_file)

    train_file = dir_name + 'valid'
    save_file = dir_name + 'valid.templete'
    generate_templete(train_file, save_file)

    train_file = dir_name + 'test'
    save_file = dir_name + 'test.templete'
    generate_templete(train_file, save_file)

    train_file = dir_name + 'old.value.test'
    save_file = dir_name + 'old.value.test.templete'
    generate_templete(train_file, save_file)

    train_file = dir_name + 'new.value.test'
    save_file = dir_name + 'new.value.test.templete'
    generate_templete(train_file, save_file)

    train_file = dir_name + 'new.slot.test'
    save_file = dir_name + 'new.slot.test.templete'
    generate_templete(train_file, save_file)
"""
