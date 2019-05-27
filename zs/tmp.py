import codecs

def f(file_name, save_file):
    with codecs.open(file_name, 'r') as f:
        lines = f.readlines()
    triples = []
    for l in lines:
        utt, trp = l.split('\t<=>\t')
        trp = trp.strip()
        if trp == '':
            continue
        else:
            trps = trp.split(';')
            triples.extend(trps)
    triples = list(set(triples))
    with open(save_file, 'w') as g:
        for trp in triples:
            g.write('{}\n'.format(trp))

if __name__ == '__main__':
    f('dstc2train.3seed', 'dstc2.3.train.class')
