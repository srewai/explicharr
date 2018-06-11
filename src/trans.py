#!/usr/bin/env python3


len_cap = 128
batch_size = 64
ckpt = None


from model import model
from tqdm import tqdm
from utils import batch, PointedIndex, decode
import numpy as np
import tensorflow as tf


idx = PointedIndex(np.load("trial/data/idx.npy").item()['idx2tgt'])
src = np.load("trial/data/src_valid.npy")
src = batch(src, batch_size, shuffle= False, repeat= False)

m = model(src= src, len_cap= len_cap, training= False)
m.z = m.pred[:,-1]

saver = tf.train.Saver(max_to_keep= None)
sess = tf.InteractiveSession()

ckpt = tf.train.latest_checkpoint("trial/model/")
saver.restore(sess, ckpt)

pad, end = idx(' '), idx("\n")
with open(ckpt + ".pred", 'w') as f:
    while True:
        try:
            w = sess.run(m.w)
            x = np.full((len(w), len_cap), end, dtype= np.int32)
            x[:,0] = pad
            for i in tqdm(range(1, len_cap), ncols= 70):
                z = sess.run(m.z, {m.w: w, m.x: x[:,:i]})
                if np.alltrue(z == end): break
                x[:,i] = z
            for i in x:
                print(decode(idx, i), file= f)
        except tf.errors.OutOfRangeError:
            break
