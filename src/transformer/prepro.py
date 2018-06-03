'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from collections import Counter
from hyperparams import Hyperparams as hp
import codecs
import numpy as np
import os
import tensorflow as tf
# import regex


def make_vocab(fpath, fname):
    '''Constructs vocabulary.

    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.

    Writes vocabulary line by line to `logdir/fname`
    '''
    text = codecs.open(fpath, 'r', 'utf-8').read()
    # text = regex.sub("[^\s\p{Latin}']", "", text)
    words = text.split()
    word2cnt = Counter(words)
    if not os.path.exists(hp.logdir): os.mkdir(hp.logdir)
    with codecs.open('{}/{}'.format(hp.logdir, fname), 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))


if __name__ == '__main__':
    make_vocab(hp.source_train, "de.vocab.tsv")
    make_vocab(hp.target_train, "en.vocab.tsv")
    print("Done")
