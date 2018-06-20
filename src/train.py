#!/usr/bin/env python3


trial = '01'
param = dict(dim= 256, dim_mid= 512, num_head= 4, num_layer= 2, dropout= 0.25)
len_cap    = 2**8
batch_size = 2**6
step_eval  = 2**7
step_save  = 2**12
ckpt       = None


from model import model
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

i = permute(len(src_train))
src_train = src_train[i]
tgt_train = tgt_train[i]
del i

# # for testing
# m = model(training= False, **param)

# # for profiling
# from util_tf import profile
# m = model(**param)
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     profile(join(path, "graph"), sess, m.up, {m.src: src_train[:batch_size], m.tgt: tgt_train[:batch_size]})

# for training
src, tgt = batch((src_train, tgt_train), batch_size= batch_size)
m = model(src= src, tgt= tgt, len_cap= len_cap, **param)

########################
# autoregressive model #
########################

m.p = m.pred[:,-1]
src = np.load("trial/data/valid_src.npy")
rng = range(0, len(src) + batch_size, batch_size)
idx = PointedIndex(np.load("trial/data/index_tgt.npy").item())

def write_trans(path, src= src, rng= rng, idx= idx, batch_size= batch_size):
    with open(path, 'w') as f:
        for i, j in zip(rng, rng[1:]):
            for p in trans(m, src[i:j])[:,1:]:
                print(decode(idx, p), file= f)

def trans(m, src, begin= 2, len_cap= 256):
    end = m.end.eval()
    w = m.w.eval({m.src: src, m.dropout: 0})
    x = np.full((len(src), len_cap), end, dtype= np.int32)
    x[:,0] = begin
    for i in range(1, len_cap):
        p = m.p.eval({m.w: w, m.x: x[:,:i], m.dropout: 0})
        if np.alltrue(p == end): break
        x[:,i] = p
    return x

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
    tf.summary.scalar('step_loss', m.loss)
    , tf.summary.scalar('step_acc', m.acc)))
feed_eval = {m.dropout: 0}

for _ in range(5):
    for _ in tqdm(range(step_save), ncols= 70):
        sess.run(m.up)
        step = sess.run(m.step)
        if not (step % step_eval):
            wtr.add_summary(sess.run(summ, feed_eval), step)
    saver.save(sess, "trial/model/m{}".format(trial), step, write_meta_graph= False)
    write_trans("trial/pred/{}_{}".format(step, trial))
