# -*- coding: utf-8 -*-

import argparse
import os, sys, random
import codecs
from collections import Counter
import json
import numpy as np
import torch
import copy
import difflib
import Levenshtein

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

    'request-food': ['type of food', 'what kind of food do they serve'],
    'request-addr': ['the address', 'whats the address', 'can i get the address'],
    'request-area': ['the area', 'what area'],
    'request-name': ['what was the name', 'the name'],
    'request-phone': ['what is the phone number', 'the phone number'],
    'request-postcode': ['post code', 'the postcode'],
    'request-pricerange': ['the price range', 'can i get the price range'],
    'request-signature': ['the signature'],

    'inform-food-[food]': ['[food]', '[food] food', 'serve [food] food'],
    'confirm-food-[food]': ['does it serve [food] food', 'is it [food]'],
    'deny-food-[food]': ['i dont want [food]', 'not [food] food'],

    'inform-name-[name]': ['[name]'],
    'confirm-name-[name]': ['is it [name]'],
    'deny-name-[name]': ['not [name]', 'hate [name]', 'fuck [name]'],

    'inform-area-[area]': ['in the [area] part of town', '[area]', 'in the [area]'],
    'confirm-area-[area]': ['is it in the [area] of town'],
    'deny-area-[area]': ['not in the [area]'],

    'inform-pricerange-cheap': ['cheap'],
    'inform-pricerange-moderate': ['moderately priced', 'moderate'],
    'inform-pricerange-expensive': ['expensive'],
    'confirm-pricerange-cheap': ['is it cheap'],
    'confirm-pricerange-moderate': ['is it moderately priced', 'is it moderate'],
    'confirm-pricerange-expensive': ['is it expensive'],
    'deny-pricerange-cheap': ['not cheap'],
    'deny-pricerange-moderate': ['no moderately priced', 'not moderate'],
    'deny-pricerange-expensive': ['not expensive'],

    'inform-this-dontcare': ['dont care', 'i dont mind', 'any', 'doesnt matter'],
    'inform-food-dontcare': ['any food', 'any type of food', 'any kind of food'],
    'inform-area-dontcare': ['any area', 'any part of town'],
    'inform-name-dontcare': ['any name'],
    'inform-pricerange-dontcare': ['dont care price range'],
    }

dstc3_templates = {

    'request-hastv': ['does it has a tv', 'does it has a television'],
    'request-childrenallowed': ['does it allow children'],
    'request-price': ['the price'],
    'request-hasinternet': ['does it have an internet connection'],
    'request-near': ['near'],
    'request-type': ['type of', 'what is the type'],

    'inform-hastv-true': ['has a television', 'with a television', 'television', 'tv'],
    'inform-hastv-false': ['no television', 'no tv'],
    'inform-childrenallowed-true': ['allows children', 'allows children', 'children allowed'],
    'inform-childrenallowed-false': ['not allow children', 'with no children'],
    'inform-hasinternet-true': ['with internet connection', 'has internet connection', 'with internet', 'has internet'],
    'inform-hasinternet-false': ['no internet connection', 'no internet'],

    'inform-hastv-dontcare': ['dont care tv', 'dont care television'],
    'inform-hasinternet-dontcare': ['dont care internet'],
    'inform-childrenallowed-dontcare': ['dont care children'],

    'inform-near-[near]': ['[near]'],
    'confirm-near-[near]': ['is it [near]'],
    'deny-near-[near]': ['not [near]'],

    'inform-area-[area]': ['[area]', '[area] area'],
    'confirm-area-[area]': ['is it in [area] area'],
    'deny-area-[area]': ['not [area] area'],

    'inform-type-restaurant': ['restaurant', 'looking for a restaurant'],
    'inform-type-pub': ['pub', 'looking for a pub'],
    'inform-type-coffeeshop': ['coffeeshop', 'coffee shop', 'cafe'],
    'confirm-type-restaurant': ['is it a restaurant'],
    'confirm-type-pub': ['pub', 'looking for a pub'],
    'confirm-type-coffeeshop': ['do they serve coffee', 'is it a coffee shop', 'is it a cafe'],

    'inform-pricerange-free': ['free', 'free price range'],
    'confirm-pricerange-free': ['is it free'],
    'deny-pricerange-free': ['not free'],

    'inform-near-dontcare': ['any where']
    }

def generate_new_templates(old_templates, new_templates):
    result = copy.deepcopy(new_templates)
    for key in old_templates:
        if key not in result:
            result[key] = old_templates[key]
    return result

def triple_to_utterance(triple, templates):
    triple_lis = triple.split(';')
    results = ['']
    for trp in triple_lis:
        tmp = templates[trp]
        new = []
        for ix in results:
            for iy in tmp:
                new.append(ix + ' ' + iy)
        results = new
    return results

def rank_similarest_template(triple, utterance, templates):
    results = triple_to_utterance(triple, templates)
    similarity = 0
    string = ''
    for line in results:
        tmp = difflib.SequenceMatcher(None, line, utterance).ratio()
        if tmp > similarity:
            similarity = tmp
            string = line
    return string

def new_template_file(file_name, save_file, templates):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    results = []
    for line in lines:
        utt, trp = line.strip().split('\t<=>\t')
        tmp = rank_similarest_template(trp, utt, templates)
        results.append((utt, trp, tmp))
    with open(save_file, 'w') as g:
        for (utt, trp, tmp) in results:
            g.write('{}\t<=>\t{}\t<=>\t{}\n'.format(utt.strip(), trp.strip(), tmp.strip()))

def new_class_template_file(file_name, save_file, templates):
    triples = json.loads(open(file_name).read())
    res = []
    for trp in triples:
        class_string = trp[0].strip()
        value_dict = trp[1]
        tmp = triple_to_utterance(class_string, templates)
        sample = random.choice(tmp)
        res.append((class_string, sample.strip(), value_dict))
        #for t in tmp:
        #    res.append((class_string, t.strip(), value_dict))
    string = json.dumps(res, sort_keys=False, indent=4, separators=(',', ':'))
    with open(save_file, 'w') as g:
        g.write(string)

def class_to_template_file(file_name, save_file):
    classes = json.loads(open(file_name).read())
    res = []
    for cls in classes:
        trp = cls[0].strip()
        utt = cls[1].strip()
        value_dict = cls[2]
        for key in value_dict:
            utt = utt.replace(key, value_dict[key])
            trp = trp.replace(key, value_dict[key])
        res.append((utt, trp))
    with open(save_file, 'w') as g:
        for (utt, trp) in res:
            g.write('{}\t<=>\t{}\n'.format(utt, trp))


"""
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
"""

if __name__ == '__main__':
    dir_name = '/slfs1/users/zjz17/SLU/dstc3-st/manual-da/'

    dstc3_templates = generate_new_templates(dstc2_templates, dstc3_templates)
    """
    new_template_file(
        dir_name + 'del/dstc3.valid',
        dir_name + 'tmp/dstc3.valid',
        dstc3_templates
    )
    #"""
    """
    new_class_template_file(
        dir_name + 'del/dstc3.test.triples.json',
        dir_name + 'tmp/dstc3.test.triples.json',
        dstc3_templates
    )
    #"""
    #"""
    class_to_template_file(
        dir_name + 'tmp/dstc3.test.triples.json',
        dir_name + 'tmp/decodes/template.rule.on.test'
    )
    #"""
