from hyperparams import Hyperparams as hp


def load_vocab(fname):
    idx2word = []
    with open(fname) as f:
        for word, freq in map(str.split, f):
            if hp.min_cnt <= int(freq):
                idx2word.append(word)
    word2idx = {word: idx for idx, word in enumerate(idx2word)}
    return word2idx, idx2word


def create_data(src, tgt):
    import numpy as np
    import os
    src2idx, idx2src = load_vocab(os.path.join(hp.logdir, "vocab.src"))
    tgt2idx, idx2tgt = load_vocab(os.path.join(hp.logdir, "vocab.tgt"))
    xs, ys, ss, ts = [], [], [], []
    for st in zip(src, tgt):
        s, t = map(str.split, st)
        s.append("</S>")
        t.append("</S>")
        if max(len(s), len(t)) <= hp.max_len:
            xs.append(np.fromiter((src2idx.get(w, src2idx["<UNK>"]) for w in s), dtype= np.int32))
            ys.append(np.fromiter((tgt2idx.get(w, tgt2idx["<UNK>"]) for w in t), dtype= np.int32))
            ss.append(st[0])
            ts.append(st[1])
    xx = np.full((len(xs), hp.max_len), src2idx["<PAD>"], np.int32)
    yy = np.full((len(ys), hp.max_len), tgt2idx["<PAD>"], np.int32)
    for i, (x, y) in enumerate(zip(xs, ys)):
        xx[i, 0:len(x)] = x
        yy[i, 0:len(y)] = y
    return xx, yy, ss, ts


def load_data(fname_src, fname_tgt):
    with open(fname_src) as f: fs = f.read().splitlines()
    with open(fname_tgt) as f: ft = f.read().splitlines()
    x, y, _, _ = create_data(fs, ft)
    return x, y


def get_batch_data():
    import tensorflow as tf
    x, y = load_data(hp.source_train, hp.target_train)
    num_batch = len(x) // hp.batch_size
    x, y = tf.data.Dataset.from_tensor_slices((x, y)) \
        .shuffle(2**12) \
        .repeat() \
        .batch(hp.batch_size) \
        .make_one_shot_iterator() \
        .get_next()
    return x, y, num_batch
