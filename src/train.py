#!/usr/bin/env python3


trial      = '01'
len_cap    = 2**8
batch_size = 2**6
ckpt       = None


from model import Transformer
from os.path import expanduser, join
from tqdm import tqdm
from util import PointedIndex
from util_io import decode
from util_np import np, permute
from util_tf import tf, batch

logdir = expanduser("~/cache/tensorboard-logdir/explicharr")
tf.set_random_seed(0)

###############
# preparation #
###############

src_train = np.load("trial/data/train_src.npy")
tgt_train = np.load("trial/data/train_tgt.npy")
src_valid = np.load("trial/data/valid_src.npy")
tgt_valid = np.load("trial/data/valid_tgt.npy")
assert src_train.shape[1] <= len_cap
assert tgt_train.shape[1] <= len_cap
assert src_valid.shape[1] <= len_cap
assert tgt_valid.shape[1] <= len_cap
epoch = len(src_train) // batch_size

# # for profiling
# from util_tf import profile
# m = Transformer.new().data()
# forcing = m.forcing(trainable= False)
# autoreg = m.autoreg(trainable= False)
# feed = {m.src_: src_train[:batch_size], m.tgt_: tgt_train[:batch_size]}
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     with tf.summary.FileWriter(join(logdir, "graph"), sess.graph) as wtr:
#         profile(sess, wtr, forcing.loss, feed, tag= 'forcing')
#         profile(sess, wtr, autoreg.loss, feed, tag= 'autoreg')

####################
# validation model #
####################

model = Transformer.new()
model_valid = model.data(*batch((src_valid, tgt_valid), batch_size), len_cap)
forcing_valid = model_valid.forcing(trainable= False)
autoreg_valid = model_valid.autoreg(trainable= False)

idx_tgt = PointedIndex(np.load("trial/data/index_tgt.npy").item())
def trans(path, m= autoreg_valid, src= src_valid, idx= idx_tgt, len_cap= len_cap, batch_size= batch_size):
    rng = range(0, len(src) + batch_size, batch_size)
    with open(path, 'w') as f:
        for i, j in zip(rng, rng[1:]):
            for p in m.pred.eval({m.src: src[i:j], m.tgt: src[i:j,:1], m.len_tgt: len_cap}):
                print(decode(idx, p), file= f)

##################
# training model #
##################

model_train = model.data(*batch((src_train, tgt_train), batch_size), len_cap)
forcing_train = model_train.forcing().train(warmup= epoch)
autoreg_train = model_train.autoreg().train(warmup= epoch)

############
# training #
############

saver = tf.train.Saver(max_to_keep= None)
sess = tf.InteractiveSession()
wtr = tf.summary.FileWriter(join(logdir, "trial{}".format(trial)))

if ckpt:
    saver.restore(sess, ckpt)
else:
    tf.global_variables_initializer().run()

step_eval = epoch // 32
summ = tf.summary.merge(
    (tf.summary.scalar('step_loss', autoreg_valid.loss)
     , tf.summary.scalar('step_acc', autoreg_valid.acc)))

forc = lambda: sess.run(forcing_train.up)
bptt = lambda: sess.run(autoreg_train.up)
def auto():
    s, g, p = sess.run(        (autoreg_train.src,    autoreg_train.gold,    autoreg_train.prob))
    sess.run(forcing_train.up, {forcing_train.src: s, forcing_train.gold: g, forcing_train.tgt_prob: p})

while True:
    for _ in tqdm(range(epoch), ncols= 70):
        # pick a training fn to run
        step = sess.run(forcing_train.step)
        if not step % step_eval: wtr.add_summary(sess.run(summ), step)
    saver.save(sess, "trial/model/{}_{}".format(trial, step), write_meta_graph= False)
    trans("trial/pred/{}_{}".format(trial, step))
