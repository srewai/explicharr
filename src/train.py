#!/usr/bin/env python3


trial      = '01'
len_cap    = 2**8
batch_size = 2**6
step_eval  = 2**7
ckpt       = None


from model import Transformer
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
src_valid = np.load("trial/data/valid_src.npy")
tgt_valid = np.load("trial/data/valid_tgt.npy")
assert src_train.shape[1] <= len_cap
assert tgt_train.shape[1] <= len_cap
assert src_valid.shape[1] <= len_cap
assert tgt_valid.shape[1] <= len_cap

# # for profiling
# from util_tf import profile
# m = Transformer.new().data().autoreg(trainable= False)
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     profile(join(path, "graph"), sess, m.acc
#             , {m.src: src_train[:batch_size]
#                , m.tgt: tgt_train[:batch_size,:-1]
#                , m.gold: tgt_train[:batch_size,1:]})

model = Transformer.new()
model_train = model.data(*batch((src_train, tgt_train), batch_size), len_cap)
model_valid = model.data(*batch((src_valid, tgt_valid), batch_size), len_cap)
forcing_train = model_train.forcing().train()
autoreg_train = model_train.autoreg().train()
autoreg_valid = model_valid.autoreg(trainable= False)

#############
# translate #
#############

idx_tgt = PointedIndex(np.load("trial/data/index_tgt.npy").item())

def trans(path, m= autoreg_valid, src= src_valid, idx= idx_tgt, len_cap= len_cap, batch_size= batch_size):
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
wtr = tf.summary.FileWriter(join(path, "trial{}".format(trial)))

if ckpt:
    saver.restore(sess, ckpt)
else:
    tf.global_variables_initializer().run()

summ = tf.summary.merge(
    (tf.summary.scalar('step_loss', autoreg_valid.loss)
     , tf.summary.scalar('step_acc', autoreg_valid.acc)))

epoch = len(src_train) // batch_size

# warmup only with teacher forcing
for _ in tqdm(range(epoch), ncols= 70):
    sess.run(forcing_train.up)
    step = sess.run(forcing_train.step)
    if not (step % step_eval):
        wtr.add_summary(sess.run(summ), step)

bptt = 6, 6, 4, 2, 4, 6, 6
auto = 3, 2, 2, 1, 2, 2, 3
# mixed teacher forcing, autoregressive, and backprop through time
for b, a in zip(bptt, auto):
    for _ in tqdm(range(epoch), ncols= 70):
        if not step % b:
            sess.run(autoreg_train.up)
        elif not step % a:
            s, g, p = sess.run(        (autoreg_train.src,    autoreg_train.gold,    autoreg_train.prob))
            sess.run(forcing_train.up, {forcing_train.src: s, forcing_train.gold: g, forcing_train.tgt_prob: p})
        else:
            sess.run(forcing_train.up)
        step = sess.run(forcing_train.step)
        if not step % step_eval:
            wtr.add_summary(sess.run(summ), step)
    saver.save(sess, "trial/model/{}_{}".format(trial, step), write_meta_graph= False)
    trans("trial/pred/{}_{}".format(step, trial))
