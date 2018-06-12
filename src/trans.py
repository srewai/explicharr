#!/usr/bin/env python3


len_cap    = 2**6
batch_size = 2**6
ckpt       = None


from model import model, trans
from tqdm import tqdm
from utils import batch, PointedIndex, decode
import numpy as np
import tensorflow as tf


idx = PointedIndex(np.load("trial/data/idx.npy").item()['idx2tgt'])
src = np.load("trial/data/src_valid.npy")

m = model(training= False, len_cap= len_cap)
m.p = m.pred[:,-1]

saver = tf.train.Saver()
sess = tf.InteractiveSession()

ckpt = tf.train.latest_checkpoint("trial/model/")
saver.restore(sess, ckpt)

r = range(0, len(src) + batch_size, batch_size)
with open(ckpt + ".pred", 'w') as f:
    for i, j in zip(r, r[1:]):
        for p in trans(m, src[i:j])[:,1:]:
            print(decode(idx, p), file= f)
