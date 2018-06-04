from hyperparams import Hyperparams as hp
import numpy as np
import os


def load(fname):
    with open(fname) as f:
        for s in f:
            yield s.split()


def load_vocab(fname):
    idx2word = []
    with open(fname) as f:
        for word, freq in map(str.split, f):
            if hp.min_cnt <= int(freq):
                idx2word.append(word)
    word2idx = {word: idx for idx, word in enumerate(idx2word)}
    return word2idx, idx2word


class DataLoader:

    def __init__(self, source, target):
        # TODO make target optional
        src2idx, idx2src = load_vocab(os.path.join(hp.logdir, "vocab.src"))
        tgt2idx, idx2tgt = load_vocab(os.path.join(hp.logdir, "vocab.tgt"))
        self._idx2src = idx2src
        self._idx2tgt = idx2tgt
        src, tgt = [], []
        for s, t in zip(load(source), load(target)):
            s.append("</S>")
            t.append("</S>")
            if max(len(s), len(t)) <= hp.max_len:
                src.append(np.fromiter((src2idx.get(w, src2idx["<UNK>"]) for w in s), dtype= np.int32))
                tgt.append(np.fromiter((tgt2idx.get(w, tgt2idx["<UNK>"]) for w in t), dtype= np.int32))
        print(len(src), "sentences loaded")
        x = np.full((len(src), hp.max_len), src2idx["<PAD>"], np.int32)
        y = np.full((len(tgt), hp.max_len), tgt2idx["<PAD>"], np.int32)
        for i, (s, t) in enumerate(zip(src, tgt)):
            x[i, 0:len(s)] = s
            y[i, 0:len(t)] = t
        self._x = x
        self._y = y

    @property
    def dim_src(self):
        return len(self._idx2src)

    @property
    def dim_tgt(self):
        return len(self._idx2tgt)

    @property
    def num_bat(self):
        return len(self._x) // hp.batch_size

    def batches(self):
        import tensorflow as tf
        return tf.data.Dataset.from_tensor_slices((self._x, self._y)) \
                              .shuffle(2**12) \
                              .repeat() \
                              .batch(hp.batch_size) \
                              .make_one_shot_iterator() \
                              .get_next()

    def batch_x(self):
        batch_size = hp.batch_size
        i, x = 0, self._x
        while i < len(x):
            j = i + batch_size
            yield x[i:j]
            i = j

    def idx2tgt(self, idxs):
        end = self._idx2tgt.index("</S>")
        tgt = []
        for idx in idxs:
            if idx == end: break
            tgt.append(self._idx2tgt[idx])
        return " ".join(tgt)
