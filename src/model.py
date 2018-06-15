from utils import Record
import numpy as np
import tensorflow as tf


def placeholder(dtype, shape, x= None):
    """returns a placeholder with `dtype` and `shape`.

    if tensor `x` is given, converts and uses it as default.

    """

    if x is None:
        return tf.placeholder(dtype, shape)
    else:
        return tf.placeholder_with_default(tf.cast(x, dtype), shape)


def normalize(x, axis= -1, eps= 1e-16, name= "normalize"):
    """returns a tensor from `x` scaled and centered across `axis`."""
    with tf.variable_scope(name):
        mean, var = tf.nn.moments(x, axis, keep_dims=True)
        return (x - mean) * tf.rsqrt(var + eps)


def count(x, item, relation= tf.equal, axis= 0):
    """counts `item` in tensor `x` across `axis` by `relation`."""
    return tf.to_int32(tf.reduce_sum(tf.to_float(relation(x, item)), axis))


def sinusoid(time, dim, freq= 1e-4, name= 'sinusoid'):
    """returns a rank-2 tensor of shape `time, dim`, where each row
    corresponds to a time step and each column a sinusoid, with
    frequencies in a geometric progression from 1 to `freq`.

    """
    with tf.variable_scope(name):
        a = tf.reshape(
            freq ** ((2 / dim) * tf.range(dim // 2, dtype= tf.float32))
            , (-1, 1)) @ tf.reshape(
                tf.range(tf.convert_to_tensor(time, tf.float32), dtype= tf.float32)
                , (1, -1))
        return tf.transpose(tf.reshape(tf.concat((tf.sin(a), tf.cos(a)), -1), (dim, time)))


def multihead_attention(value, query, dim= 64, num_head= 8, bias= None, name= 'attention'):
    """computes multi-head attention from `value` and `query` tensors.

    with batch size `b`, time steps `s, t`, dimensions `k, q`

    - value : b,s,k
    - query : b,t,q

    the returned tensor has shape `b, t, dim * num_head`, and `bias`
    when supplied must have shape compatible to `num_head, b, t, s`.

    """
    dense = lambda x, d, name: tf.layers.dense(x, d, use_bias= False, name= name)
    split = lambda x: tf.split(x, num_head, -1)
    with tf.variable_scope(name):
        v = tf.stack(split(dense(value, dim * num_head, 'v'))) # h,b,s,d
        k = tf.stack(split(dense(value, dim * num_head, 'k'))) # h,b,s,d
        q = tf.stack(split(dense(query, dim * num_head, 'q'))) # h,b,t,d
        q = (dim ** -0.5) * tf.matmul(q, k, transpose_b= True) # h,b,t,s
        if bias is not None: q += bias
        # todo try square and normalize instead of softmax
        return tf.concat(tf.unstack(tf.matmul(tf.nn.softmax(q), v)), -1)


def model(training= True, share_embedding= True
          , end= 1
          , src= None, dim_src= 256
          , tgt= None, dim_tgt= 256
          , dim= 512,  dim_mid= 2048
          , num_head= 8, num_layer= 6
          , activation= tf.nn.relu
          , dropout= 0.1
          , smooth= 0.1
          , warmup= 4e3):
    # src : ?, s
    # tgt : ?, t
    # both should be padded at the end (with `end`)
    # tgt (or both) should be padded at the beginning
    #
    # as an autoregressive model, this is a function : w, x -> y
    # encoded src, dense w : ?, s, dim
    # tgt history, index x : ?, t
    # current tgt, logit y : ?, dim_tgt
    assert not dim % 2 and not dim % num_head
    self = Record(end= end)
    # building blocks
    if training and dropout is not None:
        with tf.variable_scope('training/'):
            self.training = tf.placeholder_with_default(True, (), 'training')
            self.dropout = tf.placeholder_with_default(dropout, (), 'dropout')
            # todo make only one dropout mask
            dropout = lambda x: tf.layers.dropout(x, self.dropout, training= self.training)
    else:
        dropout = lambda x: x
    attention = lambda v, q, **args: multihead_attention(
        value= v, query= q, dim= dim // num_head, num_head= num_head, **args)
    # trim `src` to the maximum valid index among the batch
    with tf.variable_scope('src'):
        src = self.src = placeholder(tf.int32, (None, None), src)
        len_src = count(count(src, end), tf.shape(src)[0], tf.not_equal) + 1
        src = src[:,:len_src]
    # same for `tgt`, but with one less index
    with tf.variable_scope('tgt'):
        tgt = self.tgt = placeholder(tf.int32, (None, None), tgt)
        len_tgt = count(count(tgt, end), tf.shape(tgt)[0], tf.not_equal)
        tgt, gold = tgt[:,:len_tgt], tgt[:,1:1+len_tgt]
    # source, target, and position embeddings
    init = tf.orthogonal_initializer()
    emb_src = tf.get_variable('emb_src', (dim_src, dim), tf.float32, init)
    emb_tgt = tf.get_variable('emb_tgt', (dim_tgt, dim), tf.float32, init)
    # construction
    with tf.variable_scope('encode'):
        with tf.variable_scope('embed'):
            w = dropout(tf.gather(normalize(emb_src), src) + sinusoid(len_src, dim))
        for i in range(num_layer):
            with tf.variable_scope("layer{}".format(i)):
                w = normalize(w + dropout(attention(w, w)))
                h = tf.layers.dense(w, dim_mid, activation, name= 'relu')
                h = tf.layers.dense(h, dim, name= 'linear')
                w = normalize(w + dropout(h))
    self.w, self.x = w, tgt
    with tf.variable_scope('decode'):
        with tf.variable_scope('embed'):
            len_tgt = tf.shape(tgt)[1] # in case tgt is fed by user
            x = dropout(tf.gather(normalize(emb_tgt), tgt) + sinusoid(len_tgt, dim))
        with tf.variable_scope('mask'):
            mask = tf.log(tf.linalg.LinearOperatorLowerTriangular(tf.ones((len_tgt, len_tgt))).to_dense())
        for i in range(num_layer):
            with tf.variable_scope("layer{}".format(i)):
                x = normalize(x + dropout(attention(x, x, bias= mask, name= 'masked_attention')))
                x = normalize(x + dropout(attention(w, x)))
                h = tf.layers.dense(x, dim_mid, activation, name= 'relu')
                h = tf.layers.dense(h, dim, name= 'linear')
                x = normalize(x + dropout(h))
    with tf.variable_scope('logit'):
        logit = tf.tensordot(x, tf.transpose(emb_tgt), 1) if share_embedding \
            else tf.layers.dense(x, dim_tgt)
        self.y = logit[:,-1]
    # done
    with tf.variable_scope('eval'):
        self.prob = tf.nn.log_softmax(logit)
        pred = self.pred = tf.to_int32(tf.argmax(logit, -1))
        self.acc = tf.reduce_mean(tf.to_float(tf.equal(pred, gold)))
    if training:
        with tf.variable_scope('loss'):
             self.loss = tf.reduce_mean(
                 tf.nn.softmax_cross_entropy_with_logits_v2(
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
