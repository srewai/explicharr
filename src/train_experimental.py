#!/usr/bin/env python3


trial = '02'
len_cap    = 2**8
batch_size = 2**6
step_eval  = 2**7
step_save  = 2**12
ckpt       = "trial/model/m00"


from model_experimental import model
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

# # for testing
# m = model(training= False)

# # for profiling
# from util_tf import profile
# m = model()
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     profile(join(path, "graph"), sess, m.up, {m.src: src_train[:batch_size], m.tgt: tgt_train[:batch_size]})

# for training
src, tgt = batch((src_train, tgt_train), batch_size= batch_size)
with tf.variable_scope(""):
    mtrain = model(len_cap= len_cap)
with tf.variable_scope("", reuse= True):
    minfer = model(len_cap= len_cap, src= src, tgt= tgt, training= False)

########################
# autoregressive model #
########################

def infer(m, src, len_cap, start= 2):
    x = np.empty((len(src), len_cap, int(m.logit.shape[-1])), np.float32)
    # todo this should be handled in the model
    a = np.zeros(int(m.x.shape[-1]), np.float32)
    a[start] = 1.0
    x[:,0] = a
    w = m.w.eval({m.src: src})
    for i in range(1, len_cap):
        x[:,i] = m.p.eval({m.w: w, m.x: x[:,:i]})
    return x

src = np.load("trial/data/valid_src.npy")
idx = PointedIndex(np.load("trial/data/index_tgt.npy").item())

def trans(path, m, src= src, idx= idx, batch_size= batch_size):
    rng = range(0, len(src) + batch_size, batch_size)
    with open(path, 'w') as f:
        for i, j in zip(rng, rng[1:]):
            for p in infer(m, src[i:j], len_cap= len_cap)[:,1:]:
                print(decode(idx, np.argmax(p, axis= -1)), file= f)

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
    tf.summary.scalar('step_loss', minfer.loss)
    , tf.summary.scalar('step_acc', minfer.acc)))

for _ in range(5):
    for _ in tqdm(range(step_save), ncols= 70):
        src, gold = sess.run((minfer.src, minfer.gold))
        x = infer(minfer, src, gold.shape[1])
        sess.run(mtrain.up, {mtrain.src: src, mtrain.gold: gold, mtrain.x: x})
        step = sess.run(mtrain.step)
        if not (step % step_eval):
            wtr.add_summary(sess.run(summ), step)
    trans("trial/pred/{}_{}".format(step, trial), minfer)
saver.save(sess, "trial/model/m{}".format(trial), write_meta_graph= False)
