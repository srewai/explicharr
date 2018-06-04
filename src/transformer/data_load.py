from hyperparams import Hyperparams as hp
import numpy as np
import os


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
        src2idx, idx2src = load_vocab(os.path.join(hp.logdir, "vocab.src"))
        tgt2idx, idx2tgt = load_vocab(os.path.join(hp.logdir, "vocab.tgt"))
        self._src2idx = src2idx
        self._idx2src = idx2src
        self._tgt2idx = tgt2idx
        self._idx2tgt = idx2tgt
        with open(source) as f: src = f.read().splitlines()
        with open(target) as f: tgt = f.read().splitlines()
        xs, ys, ss, ts = [], [], [], []
        for s, t in zip(src, tgt):
            x = s.split(); x.append("</S>")
            y = t.split(); y.append("</S>")
            if max(len(x), len(y)) <= hp.max_len:
                xs.append(np.fromiter((src2idx.get(w, src2idx["<UNK>"]) for w in x), dtype= np.int32))
                ys.append(np.fromiter((tgt2idx.get(w, tgt2idx["<UNK>"]) for w in y), dtype= np.int32))
                ss.append(s)
                ts.append(t)
        print(len(xs), "sentences loaded")
        self._s = ss
        self._t = ts
        xx = np.full((len(xs), hp.max_len), src2idx["<PAD>"], np.int32)
        yy = np.full((len(ys), hp.max_len), tgt2idx["<PAD>"], np.int32)
        for i, (x, y) in enumerate(zip(xs, ys)):
            xx[i, 0:len(x)] = x
            yy[i, 0:len(y)] = y
        self._x = xx
        self._y = yy

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
