#!/usr/bin/env python3


from hyperparams import Hyperparams as hp
import numpy as np

from data_load import DataLoader
dl = DataLoader(hp.source_test, hp.target_test)

from model import Model
m = Model(dl, training= False)

import tensorflow as tf
sess = tf.InteractiveSession()
checkpoint = tf.train.latest_checkpoint(hp.logdir)
tf.train.Saver().restore(sess, checkpoint)

with open(checkpoint + ".pred", 'w') as f:
    for x in dl.batch_x():
        pred = np.zeros_like(x, np.int32)
        for i in range(hp.max_len):
            pred[:,i] = sess.run(m.preds, {m.x: x, m.y: pred})[:,i]
        for p in pred:
            print(dl.idx2tgt(p), file= f)


def idx2tgt(idxs):
    end = dl._idx2tgt.index("</S>")
    tgt = []
    for idx in idxs:
        if idx == end: break
        tgt.append(dl._idx2tgt[idx])
    return " ".join(tgt)
