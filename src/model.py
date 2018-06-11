from utils import Record
import numpy as np
import tensorflow as tf


def count(tensor, item, axis= None):
    return tf.to_int32(tf.reduce_sum(tf.to_float(tf.equal(tensor, item)), axis))


def sinusoid(n, t):
    a = (1e-8 ** (np.arange(n // 2) / n)).reshape(-1, 1) @ np.arange(t).reshape(1, -1)
    return np.concatenate((np.sin(a), np.cos(a)), -1).reshape(n, t).T


def multihead_attention(value, query, dense, dim, num_head= 8, mask= None, name= 'attention'):
    # value : b,s,k
    # query : b,t,q
    with tf.variable_scope(name):
        v = tf.stack(tf.split(dense(value, dim * num_head, name= 'v'), num_head, -1)) # h,b,s,d
        k = tf.stack(tf.split(dense(value, dim * num_head, name= 'k'), num_head, -1)) # h,b,s,d
        q = tf.stack(tf.split(dense(query, dim * num_head, name= 'q'), num_head, -1)) # h,b,t,d
        q = (dim ** -0.5) * tf.matmul(q, k, transpose_b= True)                        # h,b,t,s
        if mask is not None: q += tf.log(mask)
        return tf.concat(tf.unstack(tf.matmul(tf.nn.softmax(q), v)), -1)              # b,t,dh


# TODO efficient generation
def model(end= 1
          , src= None, dim_src= 256, len_src= None
          , tgt= None, dim_tgt= 256, len_tgt= None
          , dim= 512, dim_mid= 2048, len_cap= 512
          , num_layer= 6, num_head= 8
          , kinit= tf.orthogonal_initializer()
          , binit= tf.zeros_initializer()
          , act= tf.nn.relu
          , training= True
          , smooth= 0.1
          , dropout= 0.1
          , warmup= 4e3):
    # src : b,s
    # tgt : b,t
    # both should be padded at the end (with `end`)
    # tgt (or both) should be padded at the beginning
    assert not dim % 2 and not dim % num_head
    self = Record()
    # if `len_src` unspecified, trim to the maximum valid index among the batch
    with tf.variable_scope("src"):
        if src is None: src = tf.placeholder(tf.int32, (None, len_src), 'src')
        self.src = src
        if len_src is not None:
            shape = tf.shape(src)
            len_src = shape[1] - count(count(src, end, 0), shape[0]) + 1
            len_src = tf.minimum(len_src, len_cap)
        src = src[:,:len_src]
    # same for `len_tgt`, but with one less index
    with tf.variable_scope("tgt"):
        if tgt is None: tgt = tf.placeholder(tf.int32, (None, len_tgt), 'tgt')
        self.tgt = tgt
        if len_tgt is not None:
            shape = tf.shape(tgt)
            len_tgt = shape[1] - count(count(tgt, end, 0), shape[0])
            len_tgt = tf.minimum(len_tgt, len_cap)
        tgt, gold = tgt[:,:len_tgt], tgt[:,1:1+len_tgt]
    # prep
    if training and dropout is not None:
        with tf.variable_scope('training'):
            self.training = tf.placeholder_with_default(True, (), 'training')
            self.dropout = tf.placeholder_with_default(dropout, (), 'dropout')
            dropout = lambda x: tf.layers.dropout(x, self.dropout, training= self.training)
    else:
        dropout = lambda x: x

    norm = tf.contrib.layers.layer_norm
    dense = lambda x, d, **args: tf.layers.dense(
        inputs= x, units= d, kernel_initializer= kinit, bias_initializer= binit, **args)
    attention = lambda v, q, **args: multihead_attention(
        value= v, query= q, dense= dense, dim= dim // num_head, num_head= num_head, **args)
    # sinusoidal positional encoding
    pos = tf.constant(sinusoid(dim, len_cap), tf.float32, name= 'sinusoid')
    with tf.variable_scope('encode'):
        with tf.variable_scope('embed'):
            x = tf.gather(tf.get_variable('src', (dim_src, dim), tf.float32, kinit), src)
            x += pos[:len_src]
            x = dropout(x)
        for i in range(num_layer):
            with tf.variable_scope("layer{}".format(i)):
                x = norm(x + dropout(attention(x, x)))
                r = dense(x, dim_mid, activation= act, name= 'relu')
                x = norm(x + dropout(dense(r, dim))) # why no activation ????
    with tf.variable_scope('decode'):
        with tf.variable_scope('embed'):
            y = tf.gather(tf.get_variable('tgt', (dim_tgt, dim), tf.float32, kinit), tgt)
            y += pos[:len_tgt]
            y = dropout(y)
        with tf.variable_scope('mask'):
            mask = tf.linalg.LinearOperatorLowerTriangular(tf.ones((len_tgt, len_tgt))).to_dense()
        for i in range(num_layer):
            with tf.variable_scope("layer{}".format(i)):
                y = norm(y + dropout(attention(y, y, mask= mask, name= 'masked_attention')))
                y = norm(y + dropout(attention(x, y)))
                r = dense(y, dim_mid, activation= act, name= 'relu')
                y = norm(y + dropout(dense(r, dim))) # why no activation ????
        logit = dense(y, dim_tgt, name= 'logit')
    with tf.variable_scope('pred'): pred = self.pred = tf.to_int32(tf.argmax(logit, -1))
    with tf.variable_scope('acc'):
        mask = tf.to_float(tf.not_equal(gold, end))
        acc = self.acc = tf.reduce_sum(tf.to_float(tf.equal(pred, gold)) * mask) / tf.reduce_sum(mask)
    if training:
        with tf.variable_scope('loss'):
            loss = self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels= tf.gather(tf.eye(dim_tgt) * (1 - smooth) + (smooth / dim_tgt), gold)
                , logits= logit))
        with tf.variable_scope('training'):
            step = self.step = tf.train.get_or_create_global_step()
            s = tf.to_float(step)
            lr = self.lr = tf.placeholder_with_default(
                (dim ** -0.5) * tf.minimum(s ** -0.5, s * (warmup ** -1.5))
                , (), 'lr')
        up = self.up = tf.train.AdamOptimizer(
            lr, beta1= 0.9, beta2= 0.98, epsilon= 1e-9).minimize(loss, step)
    return self
