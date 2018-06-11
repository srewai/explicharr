from utils import Record
import numpy as np
import tensorflow as tf


def placeholder(x, dtype, shape):
    if x is None:
        return tf.placeholder(dtype, shape)
    else:
        return tf.placeholder_with_default(tf.cast(x, dtype), shape)


def count(tensor, item, axis= None):
    return tf.to_int32(tf.reduce_sum(tf.to_float(tf.equal(tensor, item)), axis))


def sinusoid(n, t):
    a = (1e-8 ** (np.arange(n // 2) / n)).reshape(-1, 1) @ np.arange(t).reshape(1, -1)
    return np.concatenate((np.sin(a), np.cos(a)), -1).reshape(n, t).T


def multihead_attention(value, query, dim= 64, num_head= 8, mask= None, name= 'attention'):
    # value : b,s,k
    # query : b,t,q
    # mask  :   t,s
    # ->    : b,t,dim*num_head
    dense = lambda x, d, name: tf.layers.dense(x, d, use_bias= False, name= name)
    split = lambda x: tf.split(x, num_head, -1)
    with tf.variable_scope(name):
        v = tf.stack(split(dense(value, dim * num_head, 'v'))) # h,b,s,d
        k = tf.stack(split(dense(value, dim * num_head, 'k'))) # h,b,s,d
        q = tf.stack(split(dense(query, dim * num_head, 'q'))) # h,b,t,d
        q = (dim ** -0.5) * tf.matmul(q, k, transpose_b= True) # h,b,t,s
        if mask is not None: q += tf.log(mask)
        return tf.concat(tf.unstack(tf.matmul(tf.nn.softmax(q), v)), -1)


def model(end= 1
          , src= None, dim_src= 256, len_src= None
          , tgt= None, dim_tgt= 256, len_tgt= None
          , dim= 512, dim_mid= 2048, len_cap= 512
          , num_layer= 6, num_head= 8
          , act= tf.nn.relu
          , training= True
          , smooth= 0.1
          , dropout= 0.1
          , warmup= 4e3):
    # src : ?, len_src
    # tgt : ?, len_tgt
    # both should be padded at the end (with `end`)
    # tgt (or both) should be padded at the beginning
    #
    # as an autoregressive model, this is a function : w, x -> z
    # encoded src, dense w : ?, len_src, dim
    # tgt history, index x : ?, len_tgt
    # current tgt, logit z : ?, dim_tgt
    assert not dim % 2 and not dim % num_head
    self = Record()
    # building blocks
    if training and dropout is not None:
        with tf.variable_scope('training/'):
            self.training = tf.placeholder_with_default(True, (), 'training')
            self.dropout = tf.placeholder_with_default(dropout, (), 'dropout')
            dropout = lambda x: tf.layers.dropout(x, self.dropout, training= self.training)
    else:
        dropout = lambda x: x
    norm = tf.contrib.layers.layer_norm
    dense = tf.layers.dense
    attention = lambda v, q, **args: multihead_attention(
        value= v, query= q, dim= dim // num_head, num_head= num_head, **args)
    # if `len_src` unspecified, trim to the maximum valid index among the batch
    with tf.variable_scope('src'):
        src = self.src = placeholder(src, tf.int32, (None, len_src))
        if len_src is None:
            shape = tf.shape(src)
            len_src = shape[1] - count(count(src, end, 0), shape[0]) + 1
        len_src = tf.minimum(len_src, len_cap)
        src = src[:,:len_src]
    # same for `len_tgt`, but with one less index
    with tf.variable_scope('tgt'):
        tgt = self.tgt = placeholder(tgt, tf.int32, (None, len_tgt))
        if len_tgt is None:
            shape = tf.shape(tgt)
            len_tgt = shape[1] - count(count(tgt, end, 0), shape[0])
        len_tgt = tf.minimum(len_tgt, len_cap)
        tgt, gold = tgt[:,:len_tgt], tgt[:,1:1+len_tgt]
    # embedding with sinusoidal positional encoding
    pos = tf.constant(sinusoid(dim, len_cap), tf.float32, name= 'sinusoid')
    with tf.variable_scope('source'):
        w = dropout(pos[:len_src] + tf.gather(tf.get_variable('emb', (dim_src, dim), tf.float32), src))
    with tf.variable_scope('target'):
        len_tgt = tf.shape(tgt)[1] # in case tgt is fed by user
        x = dropout(pos[:len_tgt] + tf.gather(tf.get_variable('emb', (dim_tgt, dim), tf.float32), tgt))
    # construction
    with tf.variable_scope('encode'):
        for i in range(num_layer):
            with tf.variable_scope("layer{}".format(i)):
                w = norm(w + dropout(attention(w, w)))
                h = dense(w, dim_mid, activation= act, name= 'relu')
                h = dense(h, dim) # why no activation ????
                w = norm(w + dropout(h))
    self.w, self.x = w, tgt
    with tf.variable_scope('decode'):
        with tf.variable_scope('mask'):
            mask = tf.linalg.LinearOperatorLowerTriangular(tf.ones((len_tgt, len_tgt))).to_dense()
        for i in range(num_layer):
            with tf.variable_scope("layer{}".format(i)):
                x = norm(x + dropout(attention(x, x, mask= mask, name= 'masked_attention')))
                x = norm(x + dropout(attention(w, x)))
                h = dense(x, dim_mid, activation= act, name= 'relu')
                h = dense(h, dim) # why no activation ????
                x = norm(x + dropout(h))
    logit = self.y = dense(x, dim_tgt, name= 'logit')
    self.z = self.y[:,-1]
    # done
    with tf.variable_scope('eval'):
        self.prob = tf.nn.softmax(logit)
        pred = self.pred = tf.to_int32(tf.argmax(logit, -1))
        mask = tf.to_float(tf.not_equal(gold, end))
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(pred, gold)) * mask) / tf.reduce_sum(mask)
    if training:
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels= tf.gather(tf.eye(dim_tgt) * (1 - smooth) + (smooth / dim_tgt), gold)
                , logits= logit))
        with tf.variable_scope('training/'):
            self.step = tf.train.get_or_create_global_step()
            step = tf.to_float(self.step + 1)
            self.lr = tf.placeholder_with_default(
                (dim ** -0.5) * tf.minimum(step ** -0.5, step * (warmup ** -1.5))
                , (), 'lr')
        self.up = tf.train.AdamOptimizer(
            self.lr, beta1= 0.9, beta2= 0.98, epsilon= 1e-9).minimize(self.loss, self.step)
    return self
