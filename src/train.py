#!/usr/bin/env python3


trial      = '01'
len_cap    = 2**7
batch_size = 2**5
step_eval  = 2**7
step_save  = 2**12
ckpt       = None


from model import model
from os.path import expanduser, join
from tqdm import tqdm
from utils import batch
import numpy as np
import tensorflow as tf

src_train = np.load("trial/data/src_train.npy")
tgt_train = np.load("trial/data/tgt_train.npy")
src_valid = np.load("trial/data/src_valid.npy")
tgt_valid = np.load("trial/data/tgt_valid.npy")

src, tgt = batch((src_train, tgt_train), batch_size= batch_size)
m = model(src= src, tgt= tgt, len_cap= len_cap)
src, tgt = batch((src_valid, tgt_valid), batch_size= batch_size)

path = expanduser("~/cache/tensorboard-logdir/explicharr/trial{}".format(trial))
wtr_train = tf.summary.FileWriter(join(path, 'train'))
wtr_valid = tf.summary.FileWriter(join(path, 'valid'))
# wtr = tf.summary.FileWriter(join(path, 'graph'), tf.get_default_graph())
saver = tf.train.Saver()
sess = tf.InteractiveSession()

if ckpt:
    saver.restore(sess, ckpt)
else:
    tf.global_variables_initializer().run()

summ_ev = tf.summary.scalar('loss', m.loss)
summ_up = summ_ev, m.step, m.up

while True:
    for _ in tqdm(range(step_save), ncols= 70):
        summ, step, _ = sess.run(summ_up)
        if not (step % step_eval):
            wtr_train.add_summary(summ, step)
            summ = sess.run(summ_ev, {m.src: src.eval(), m.tgt: tgt.eval()})
            wtr_valid.add_summary(summ, step)
    wtr_train.flush()
    wtr_valid.flush()
    saver.save(sess, "trial/model/m", step)
