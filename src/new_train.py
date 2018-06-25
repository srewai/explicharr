#!/usr/bin/env python3


trial = '02'
len_cap    = 2**8
batch_size = 2**6
step_eval  = 2**7
ckpt       = 'trial/model/02_17220'


from new_model import Transformer
from os.path import expanduser, join
from tqdm import tqdm
from util import PointedIndex
from util_io import decode
from util_np import np, permute
from util_tf import tf, batch
tf.set_random_seed(0)


path = expanduser("~/cache/tensorboard-logdir/explicharr")
src_train = np.load("trial/data/train_src.npy")
tgt_train = np.load("trial/data/train_tgt.npy")
assert src_train.shape[1] <= len_cap
assert tgt_train.shape[1] <= len_cap

i = permute(len(src_train))
src_train = src_train[i]
tgt_train = tgt_train[i]
del i

# for training
epoch = int(len(src_train) * (step_eval - 2) / step_eval / batch_size)
src, tgt = batch((src_train, tgt_train), batch_size= batch_size)
m = Transformer.new().prep(len_cap= len_cap, src= src, tgt= tgt)
# mtrain = m.forcing().post().train(warmup= epoch)
# minfer = m.autoreg().post()
minfer = m.autoreg(trainable= True).post().train(warmup= epoch)

# # for profiling
# m = Transformer.new().prep().autoreg(trainable= True).post().train()
# from util_tf import profile
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     profile(join(path, "graph"), sess, m.up
#             , {m.src: src_train[:batch_size]
#                , m.tgt: tgt_train[:batch_size,:-1]
#                , m.gold: tgt_train[:batch_size,1:]})

#############
# translate #
#############

src = np.load("trial/data/valid_src.npy")
idx = PointedIndex(np.load("trial/data/index_tgt.npy").item())

def trans(path, m= minfer, src= src, idx= idx, len_cap= len_cap, batch_size= batch_size):
    # todo remove tgt
    rng = range(0, len(src) + batch_size, batch_size)
    with open(path, 'w') as f:
        for i, j in zip(rng, rng[1:]):
            for p in m.pred.eval({m.src: src[i:j], m.tgt: src[i:j,:1], m.len_tgt: len_cap}):
                print(decode(idx, p), file= f)

############
# training #
############

saver = tf.train.Saver()
sess = tf.InteractiveSession()
wtr_autoreg = tf.summary.FileWriter(join(path, "trial{}".format(trial)))
# wtr_autoreg = tf.summary.FileWriter(join(path, "trial{}/autoreg".format(trial)))
# wtr_forcing = tf.summary.FileWriter(join(path, "trial{}/forcing".format(trial)))

if ckpt:
    saver.restore(sess, ckpt)
else:
    tf.global_variables_initializer().run()

summ_autoreg = tf.summary.merge((
    tf.summary.scalar('step_loss', minfer.loss)
    , tf.summary.scalar('step_acc', minfer.acc)))
# summ_forcing = tf.summary.merge((
#     tf.summary.scalar('step_loss', mtrain.loss)
#     , tf.summary.scalar('step_acc', mtrain.acc)))
# feed_eval = {mtrain.dropout.rate: 0}

# for _ in tqdm(range(epoch), ncols= 70):
#     sess.run(mtrain.up)
#     step = sess.run(mtrain.step)
#     if not (step % step_eval):
#         wtr_autoreg.add_summary(sess.run(summ_autoreg), step)
#         wtr_forcing.add_summary(sess.run(summ_forcing, feed_eval), step)

# for r in 4, 3, 3, 2, 2, 2, 2:
#     for _ in tqdm(range(epoch), ncols= 70):
#         if step % r:
#             sess.run(mtrain.up)
#         else:
#             s, g, p = sess.run((minfer.src, minfer.gold, minfer.prob))
#             sess.run(mtrain.up, {mtrain.src: s, mtrain.gold: g, mtrain.tgt_prob: p})
#         step = sess.run(mtrain.step)
#         if not step % step_eval:
#             wtr_autoreg.add_summary(sess.run(summ_autoreg), step)
#             wtr_forcing.add_summary(sess.run(summ_forcing, feed_eval), step)
#     saver.save(sess, "trial/model/{}_{}".format(trial, step), write_meta_graph= False)
#     trans("trial/pred/{}_{}".format(step, trial))

for _ in range(5):
    for _ in tqdm(range(epoch), ncols= 70):
        sess.run(minfer.up)
        step = sess.run(minfer.step)
        if not (step % step_eval):
            wtr_autoreg.add_summary(sess.run(summ_autoreg), step)
    saver.save(sess, "trial/model/{}_{}".format(trial, step), write_meta_graph= False)
    trans("trial/pred/{}_{}".format(step, trial))
