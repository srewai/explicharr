from collections import Counter
from hyperparams import Hyperparams as hp
import os


def make_vocab(fpath, fname):
    with open(fpath) as f: word2freq = Counter(f.read().split())
    with open(os.path.join(hp.logdir, fname), 'w') as f:
        print("<PAD>\t9999\n<UNK>\t9999\n<S>\t9999\n</S>\t9999", file= f)
        for word, freq in word2freq.most_common(len(word2freq)):
            print(word, "\t", freq, file= f)


if __name__ == '__main__':
    make_vocab(hp.source_train, "vocab.src")
    make_vocab(hp.target_train, "vocab.tgt")
