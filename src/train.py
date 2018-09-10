#!/usr/bin/env python3


trial      = 'm'
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

# from util_io import encode
# idx_src = PointedIndex(np.load("trial/data/index_src.npy").item())
# def auto(s, m= autoreg_valid, idx_src= idx_src, idx_tgt= idx_tgt, len_cap= len_cap):
#     src = np.array(encode(idx_src, s)).reshape(1, -1)
#     return decode(idx_tgt, m.pred.eval({m.src: src, m.tgt: src[:,:1], m.len_tgt: len_cap})[0])

##################
# training model #
##################

model_train = model.data(*batch((src_train, tgt_train), batch_size), len_cap)
forcing_train = model_train.forcing().train()

# # bptt training and free running
# autoreg_train = model_train.autoreg().train(warmup= epoch)
# bptt = lambda: sess.run(autoreg_train.up)
# def free():
#     s, g, p = sess.run(        (autoreg_train.src,    autoreg_train.gold,    autoreg_train.prob))
#     sess.run(forcing_train.up, {forcing_train.src: s, forcing_train.gold: g, forcing_train.tgt_prob: p})

############
# training #
############

saver = tf.train.Saver()
sess = tf.InteractiveSession()
wtr = tf.summary.FileWriter(join(logdir, "{}".format(trial)))

if ckpt:
    saver.restore(sess, "trial/model/{}{}".format(trial, ckpt))
else:
    tf.global_variables_initializer().run()

step_eval = epoch // 8
summ = tf.summary.merge(
    (tf.summary.scalar('step_loss', forcing_valid.loss)
     , tf.summary.scalar('step_acc', forcing_valid.acc)))

for _ in range(15):
    for _ in tqdm(range(12 * epoch), ncols= 70):
        sess.run(forcing_train.up)
        step = sess.run(forcing_train.step)
        if not step % step_eval: wtr.add_summary(sess.run(summ), step)
    saver.save(sess, "trial/model/{}{}".format(trial, step), write_meta_graph= False)
    trans("trial/pred/{}{}".format(trial, step))
