#!/usr/bin/env python3


trial      = '00'
len_cap    = 2**6
batch_size = 2**6
step_eval  = 2**7
step_save  = 2**12
ckpt       = None


from model import model
from os.path import expanduser, join
from tqdm import tqdm
from utils import permute, batch
import numpy as np
import tensorflow as tf
tf.set_random_seed(0)


src_train = np.load("trial/data/src_train.npy")
tgt_train = np.load("trial/data/tgt_train.npy")
# src_valid = np.load("trial/data/src_valid.npy")
# tgt_valid = np.load("trial/data/tgt_valid.npy")

i = permute(len(src_train))
src_train = src_train[i]
tgt_train = tgt_train[i]
del i

src, tgt = batch((src_train, tgt_train), batch_size= batch_size)
m = model(src= src, tgt= tgt, len_cap= len_cap)

path = expanduser("~/cache/tensorboard-logdir/explicharr/trial{}".format(trial))
wtr_train = tf.summary.FileWriter(join(path, 'train'))
# wtr_valid = tf.summary.FileWriter(join(path, 'valid'))
# wtr = tf.summary.FileWriter(join(path, 'graph'), tf.get_default_graph())
saver = tf.train.Saver()
sess = tf.InteractiveSession()

if ckpt:
    saver.restore(sess, ckpt)
else:
    tf.global_variables_initializer().run()

step_up = m.step, m.up
summ_ev = tf.summary.merge((
    tf.summary.scalar('step_loss', m.loss)
    , tf.summary.scalar('step_acc', m.acc)))

while True:
    for _ in tqdm(range(step_save), ncols= 70):
        step, _ = sess.run(step_up)
        if not (step % step_eval):
            wtr_train.add_summary(sess.run(summ_ev), step)
    wtr_train.flush()
    saver.save(sess, "trial/model/m", step)
