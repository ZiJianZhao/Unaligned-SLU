# -*- coding:utf-8 -*-

import os, codecs, random


root_dir = '/slfs1/users/zjz17/SLU/maps/'
word_dir = root_dir + 'word/'
char_dir = root_dir + 'char/'


def split_train_valid(word_dir, word_file, char_dir, char_file, ratio=0.95):

    word_file = os.path.join(word_dir, word_file)
    word_train = os.path.join(word_dir, 'train')
    word_valid = os.path.join(word_dir, 'valid')

    char_file = os.path.join(char_dir, char_file)
    char_train = os.path.join(char_dir, 'train')
    char_valid = os.path.join(char_dir, 'valid')

    with codecs.open(word_file, 'r', 'utf8') as f:
        lines = f.readlines()
    lis = list(range(len(lines)))
    random.shuffle(lis)

    def f(file_path, train_path, valid_path, lis):
        with codecs.open(file_path, 'r', 'utf8') as f:
            lines = f.readlines()
        num = int(len(lines)*ratio)
        train_lines = []
        for i in range(num):
            train_lines.append(lines[lis[i]])
        valid_lines = []
        for i in range(num, len(lines)):
            valid_lines.append(lines[lis[i]])
        with open(train_path, 'w') as g:
            for line in train_lines:
                g.write(line.strip()+'\n')
        with open(valid_path, 'w') as g:
            for line in valid_lines:
                g.write(line.strip()+'\n')

    f(word_file, word_train, word_valid, lis)
    f(char_file, char_train, char_valid, lis)

#split_train_valid(word_dir, 'origin_traindev', char_dir, 'origin_traindev')

def get_classes(root_dir, filename, savefile='class.train'):
    filename = os.path.join(root_dir, filename)
    savefile = os.path.join(root_dir, savefile)
    with codecs.open(filename, 'r', 'utf8') as f:
        lines = f.readlines()
    classes = []
    for line in lines:
        lis = line.strip().split(' => ')
        if len(lis) < 2:
            print(line)
            continue
        clses = lis[1].strip().split(';')
        classes.extend(clses)
    classes = sorted(list(set(classes)), key=lambda x: len(x.strip().split('-')))
    acts = []
    for cls in classes:
        acts.append(cls.strip().split('-')[0])
    acts = list(set(acts))
    print(acts)
    with open(savefile, 'w') as g:
        for cls in classes:
            g.write(cls+'\n')

#get_classes(word_dir, 'train')
#get_classes(char_dir, 'train')

def process_empty_label(root_dir, filename):
    filename = os.path.join(root_dir, filename)
    with codecs.open(filename, 'r', 'utf8') as f:
        lines = f.readlines()
    with open(filename, 'w') as g:
        for line in lines:
            g.write(line.strip()+' \n')

#process_empty_label(word_dir, 'train')
#process_empty_label(word_dir, 'valid')
#process_empty_label(word_dir, 'test')
#process_empty_label(char_dir, 'train')
#process_empty_label(char_dir, 'valid')
#process_empty_label(char_dir, 'test')


def judge_label_in_utt(root_dir, filename):
    filename = os.path.join(root_dir, filename)
    with codecs.open(filename, 'r', 'utf8') as f:
        lines = f.readlines()
    num = 0
    for line in lines:
        lis = line.strip().split(' => ')
        if len(lis) < 2:
            continue
        utt = lis[0].strip()
        classes = lis[1].strip().split(';')
        for cls in classes:
            cls = cls.strip().split('-')
            if len(cls) == 3:
                cls = cls[2]
                if cls not in utt:
                    num += 1
                    break
    print('num / total: {} / {}'.format(num, len(lines)))


#judge_label_in_utt(word_dir, 'train')
#judge_label_in_utt(word_dir, 'valid')
#judge_label_in_utt(word_dir, 'test')

def count_value_max_num(root_dir, filename):
    filename = os.path.join(root_dir, filename)
    with codecs.open(filename, 'r', 'utf8') as f:
        lines = f.readlines()
    num = 0
    for line in lines:
        lis = line.strip().split(' => ')
        if len(lis) < 2:
            continue
        utt = lis[0].strip()
        classes = lis[1].strip().split(';')
        for cls in classes:
            cls = cls.strip().split('-')
            if len(cls) == 3:
                cls_lis = cls[2].strip().split()
                if len(cls_lis) > num:
                    num = len(cls_lis)
    print('{} max num: {}'.format(filename, num))


count_value_max_num(word_dir, 'train')
count_value_max_num(word_dir, 'valid')
count_value_max_num(word_dir, 'test')

def judge_value_vocab_in_utt(root_dir, filename):
    filename = os.path.join(root_dir, filename)
    with codecs.open(filename, 'r', 'utf8') as f:
        lines = f.readlines()
    utts = []
    values = []
    for line in lines:
        lis = line.strip().split(' => ')
        if len(lis) < 2:
            continue
        utts.extend(lis[0].strip().split())
        classes = lis[1].strip().split(';')
        for cls in classes:
            cls = cls.strip().split('-')
            if len(cls) == 3:
                values.extend(cls[2].strip().split())
    utts = {k: 0 for k in utts}
    for val in values:
        if val not in utts:
            print(val)
    print('---------------')


#judge_value_vocab_in_utt(word_dir, 'train')
