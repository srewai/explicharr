#!/usr/bin/env python3


trial = '01'
len_cap    = 2**8
batch_size = 2**6
step_eval  = 2**7
step_save  = 2**12
ckpt       = None


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

src, tgt = batch((src_train, tgt_train), batch_size= batch_size)
m = Transformer.new().prep(len_cap= len_cap, src= src, tgt= tgt)
mtrain = m.forcing().post().train()
# minfer = m.autoreg().post()

# # todo profiling
# from util_tf import profile
# m = model()
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     profile(join(path, "graph"), sess, m.up, {m.src: src_train[:batch_size], m.tgt: tgt_train[:batch_size]})

#############
# translate #
#############

# src = np.load("trial/data/valid_src.npy")
# tgt = np.load("trial/data/valid_tgt.npy")
# idx = PointedIndex(np.load("trial/data/index_tgt.npy").item())

# def trans(path, m, src= src, tgt= tgt, idx= idx, batch_size= batch_size):
#     # todo remove tgt
#     rng = range(0, len(src) + batch_size, batch_size)
#     with open(path, 'w') as f:
#         for i, j in zip(rng, rng[1:]):
#             for p in m.pred.eval({m.src: src[i:j], m.tgt: tgt[i:j]}):
#                 print(decode(idx, p), file= f)

############
# training #
############

saver = tf.train.Saver()
sess = tf.InteractiveSession()
wtr = tf.summary.FileWriter(join(path, "trial{}".format(trial)))

if ckpt:
    saver.restore(sess, ckpt)
else:
    tf.global_variables_initializer().run()

summ = tf.summary.merge((
    tf.summary.scalar('step_loss', mtrain.loss)
    , tf.summary.scalar('step_acc', mtrain.acc)))
feed_eval = {mtrain.dropout.rate: 0}

for _ in range(5):
    for _ in tqdm(range(step_save), ncols= 70):
        sess.run(mtrain.up)
        step = sess.run(mtrain.step)
        if not (step % step_eval):
            wtr.add_summary(sess.run(summ, feed_eval), step)
    # trans("trial/pred/{}_{}".format(step, trial), minfer)
saver.save(sess, "trial/model/m{}".format(trial), write_meta_graph= False)
