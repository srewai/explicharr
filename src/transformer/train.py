#!/usr/bin/env python3


from hyperparams import Hyperparams as hp
from tqdm import tqdm

from data_load import DataLoader
dl = DataLoader(hp.source_train, hp.target_train)

from model import Model
m = Model(dl, training= True)

import tensorflow as tf
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# wtr = tf.summary.FileWriter(hp.logdir, tf.get_default_graph())
wtr = tf.summary.FileWriter(hp.logdir)
svr = tf.train.Saver(max_to_keep= hp.num_epochs)

summ_up = tf.summary.scalar('loss', m.mean_loss), m.train_op

print("training for {} batches per epoch".format(dl.num_bat))
step = 0
for epoch in range(hp.num_epochs):
    for _ in tqdm(range(dl.num_bat), ncols= 70):
        summ, _ = sess.run(summ_up)
        step += 1
        if not (step % 32): wtr.add_summary(summ, step)
    svr.save(sess, save_path= "{}/m{}".format(hp.logdir, epoch), global_step= step)

wtr.close()
sess.close()
