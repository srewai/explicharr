#!/usr/bin/env python3


# ckpt = None
# epochs = range(20)

ckpt = "baseline/m19"
epochs = range(20, 40)


from hyperparams import Hyperparams as hp
from tqdm import tqdm

from utils import DataLoader
dl = DataLoader(hp.source_train, hp.target_train)

from model import Model
m = Model(dl, training= True)

import tensorflow as tf
sess = tf.InteractiveSession()

# wtr = tf.summary.FileWriter(hp.logdir, tf.get_default_graph())
wtr = tf.summary.FileWriter(hp.logdir)
svr = tf.train.Saver(max_to_keep= None)

if ckpt:
    svr.restore(sess, ckpt)
else:
    tf.global_variables_initializer().run()

summ_up = tf.summary.scalar('loss', m.mean_loss), m.train_op

print("training for {} batches per epoch".format(dl.num_bat))
step = 0
for epoch in epochs:
    for _ in tqdm(range(dl.num_bat), ncols= 70):
        summ, _ = sess.run(summ_up)
        step += 1
        if not (step % 32): wtr.add_summary(summ, step)
    svr.save(sess, save_path= "{}/m{:02d}".format(hp.logdir, epoch))

wtr.close()
sess.close()
