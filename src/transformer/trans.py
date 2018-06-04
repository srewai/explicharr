#!/usr/bin/env python3


from hyperparams import Hyperparams as hp
import numpy as np

from data_load import DataLoader
dl = DataLoader(hp.source_test, hp.target_test)

from model import Model
m = Model(dl, training= False)

import tensorflow as tf
sess = tf.InteractiveSession()
svr = tf.train.Saver(max_to_keep= 1e9)

for e in range(19):
    ckpt = "{}/m{:02d}".format(hp.logdir, e)
    svr.restore(sess, ckpt)
    with open(ckpt + ".pred", 'w') as f:
        for x in dl.batch_x():
            pred = np.zeros_like(x, np.int32)
            for i in range(hp.max_len):
                pred[:,i] = sess.run(m.preds, {m.x: x, m.y: pred})[:,i]
            for p in pred:
                print(dl.idx2tgt(p), file= f)
