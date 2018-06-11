#!/usr/bin/env python3


trial = '01'
len_cap = 128
batch_size = 32
step_eval = 1000
step_save = 10000
ckpt = None


from model import model
from os.path import expanduser
from tqdm import tqdm
from utils import batch
import numpy as np
import tensorflow as tf

src_train = np.load("tmp/data/src_train.npy")
tgt_train = np.load("tmp/data/tgt_train.npy")
src_valid = np.load("tmp/data/src_valid.npy")
tgt_valid = np.load("tmp/data/tgt_valid.npy")

src, tgt = map(tf.to_int32, batch((src_train, tgt_train), batch_size= batch_size))
m = model(src= src, tgt= tgt, len_cap= len_cap
          , dim= 64, dim_mid= 128, num_layer= 3, num_head= 4)

src, tgt = batch((src_valid, tgt_valid), batch_size= batch_size)

# wtr = tf.summary.FileWriter("tmp/graph", tf.get_default_graph())
wtr = tf.summary.FileWriter(expanduser("~/cache/tensorboard-logdir/explicharr/trial" + trial))
svr = tf.train.Saver(max_to_keep= None)
sess = tf.InteractiveSession()

if ckpt:
    svr.restore(sess, ckpt)
else:
    tf.global_variables_initializer().run()

summ_up = tf.summary.scalar('loss_train', m.loss), m.step, m.up
summ_ev = tf.summary.scalar('loss_valid', m.loss)

while True:
    for _ in tqdm(range(step_save), ncols= 70):
        summ, step, _ = sess.run(summ_up)
        if not (step % step_eval):
            wtr.add_summary(summ, step)
            summ = sess.run(summ_ev, feed_dict= {m.src: src.eval(), m.tgt: tgt.eval()})
            wtr.add_summary(summ, step)
    svr.save(sess, save_path= "tmp/model/m", step)
