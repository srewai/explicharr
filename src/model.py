from util import Record
from util_tf import tf, placeholder, normalize
import numpy as np


def sinusoid(time, dim, freq= 1e-4, name= 'sinusoid', scale= True, array= False):
    """returns a rank-2 tensor of shape `time, dim`, where each row
    corresponds to a time step and each column a sinusoid, with
    frequencies in a geometric progression from 1 to `freq`.

    """
    if array:
        a = (freq ** ((2 / dim) * np.arange(dim // 2))).reshape(-1, 1) @ np.arange(time).reshape(1, -1)
        s = np.concatenate((np.sin(a), np.cos(a)), -1).reshape(dim, time)
        if scale: s *= dim ** -0.5
        return s.T
    with tf.variable_scope(name):
        a = tf.reshape(
            freq ** ((2 / dim) * tf.range(dim // 2, dtype= tf.float32))
            , (-1, 1)) @ tf.reshape(
                tf.range(tf.cast(time, tf.float32), dtype= tf.float32)
                , (1, -1))
        s = tf.reshape(tf.concat((tf.sin(a), tf.cos(a)), -1), (dim, time))
        if scale: s *= dim ** -0.5
        return tf.transpose(s)


def multihead_attention(value, query, dim= 64, num_head= 8, softmax= True, mask= None, name= 'attention'):
    """computes multi-head attention from `value` and `query` tensors.

    with batch size `b`, time steps `s, t`, dimensions `k, q`

    - value : b,s,k
    - query : b,t,q

    the returned tensor has shape `b, t, dim * num_head`, and `mask`
    when supplied must have shape compatible to `num_head, b, t, s`.

    """
    dense = lambda x, d, name: tf.layers.dense(x, d, use_bias= False, name= name)
    split = lambda x: tf.split(x, num_head, -1)
    # v : h,b,s,d
    # k : h,b,s,d
    # q : h,b,t,d
    # a : h,b,t,s
    with tf.variable_scope(name):
        q = tf.stack(split(dense(query, dim * num_head, 'q')))
        if softmax:
            v = tf.stack(split(dense(value, dim * num_head, 'v')))
            k = tf.stack(split(dense(value, dim * num_head, 'k')))
            a = tf.matmul(q, k, transpose_b= True)
            a *= (dim ** -0.5)
            if mask is not None: a += mask
            a = tf.nn.softmax(a)
        else:
            v = k = tf.stack(split(value))
            a = tf.matmul(q, k, transpose_b= True)
            if mask is not None: a *= mask
            a = tf.square(a)
            a /= tf.reduce_sum(a, -1, True) + 1e-8
        return tf.concat(tf.unstack(a @ v), -1)


def model(len_cap= None
          , src= None, dim_src= 256
          , tgt= None, dim_tgt= 256
          , dim= 256,  dim_mid= 512
          , num_head= 4, num_layer= 2
          , softmax= True
          , activation= tf.nn.relu
          , logit_share_embedding= False
          , training= True
          , dropout= 0.1
          , smooth= 0.1
          , warmup= 4e3
          , end= 1):
    """-> Record, with the following fields of tensors

    dropout : f32 ()              dropout rate, has no effect if not `training`
        end : i32 ()              end padding for `src` and `tgt`
        src : i32 (b, s)          source feed, in range `[0, dim_src)`
        tgt : f32 (b, t)          target feed, in range `[0, dim_tgt)`
      logit : f32 (b, t, dim_tgt) prediction on logit scale
       prob : f32 (b, t)          prediction on log scale
       pred : i32 (b, t)          prediction
        acc : f32 ()              accuracy

    and as an autoregressive model : w, x -> y

    w : f32  (b, s, dim)     encoded `src`
    x : f32  (b, ?, dim_tgt) target feed for the current prediction
    y : f32  (b, dim_tgt)    current prediction on logit scale

    and if `training`

    smooth : f32 () prediction smoothing
      loss : f32 () prediction loss
      step : i64 () global update step
        lr : f32 () learning rate for the current step
        up :        update operation

    setting `len_cap` makes it more efficient for training.  you won't
    be able to feed it longer sequences, but it doesn't affect any
    model parameters.

    """
    assert not dim % 2 and not dim % num_head
    self = Record()
    with tf.variable_scope('dropout'):
        self.dropout = placeholder(tf.float32, (), dropout)
        keep = 1.0 - self.dropout
    def dropout(x, keep= keep):
        with tf.variable_scope('dropout'):
            return tf.nn.dropout(x, keep, (tf.shape(x)[0], 1, dim))
    if not training: dropout = lambda x: x
    attention = lambda v, q, mask= None: multihead_attention(
        v, q, dim // num_head, num_head, softmax, mask)
    forward = lambda x, dim_mid= dim_mid, dim= dim: tf.layers.dense(
        tf.layers.dense(
            x, dim_mid, activation, name= 'relu')
        , dim, name= 'linear')
    nrd = lambda x, y: normalize(x + dropout(y))
    # trim `src` to the maximum valid index among the batch, plus one for padding
    count_not_all = lambda x: tf.reduce_sum(tf.to_int32(~ tf.reduce_all(x, 0)))
    with tf.variable_scope('src'):
        end = self.end = tf.constant(end, tf.int32, (), 'end')
        src = self.src = placeholder(tf.int32, (None, None), src)
        len_src = count_not_all(tf.equal(src, end)) + 1
        src = src[:,:len_src]
    # same for `tgt`, but with one less index
    with tf.variable_scope('tgt'):
        tgt = self.tgt = placeholder(tf.int32, (None, None), tgt)
        len_tgt = count_not_all(tf.equal(tgt, end))
        tgt, gold = tgt[:,:len_tgt], tgt[:,1:1+len_tgt]
    # embedding
    if len_cap: emb_pos = tf.constant(sinusoid(len_cap, dim, array= True), tf.float32, name= 'sinusoid')
    init = tf.orthogonal_initializer()
    with tf.variable_scope('emb_src'):
        pos = emb_pos[:len_src] if len_cap else sinusoid(len_src, dim)
        emb = tf.get_variable('emb', (dim_src, dim), tf.float32, init)
        w = dropout(pos + tf.gather(emb, src))
        # w = normalize(w) todo test if necessary
    self.x = tgt
    with tf.variable_scope('emb_tgt'):
        len_tgt = tf.shape(tgt)[1] # in case tgt is fed by user
        pos = emb_pos[:len_tgt] if len_cap else sinusoid(len_tgt, dim)
        emb = tf.get_variable('emb', (dim_tgt, dim), tf.float32, init)
        x = dropout(pos + tf.gather(emb, tgt))
        # x = normalize(x) todo test if necessary
    with tf.variable_scope('encode'):
        for i in range(num_layer):
            with tf.variable_scope("layer{}".format(i + 1)):
                with tf.variable_scope("attention"):
                    w = nrd(w, attention(w, w))
                with tf.variable_scope("forward"):
                    w = nrd(w, forward(w))
    self.w, self.x = w, tgt
    with tf.variable_scope('decode'):
        with tf.variable_scope('mask'):
            t = tf.shape(x)[1]
            mask = tf.linalg.LinearOperatorLowerTriangular(tf.ones((t, t))).to_dense()
            if softmax: mask = tf.log(mask)
        for i in range(num_layer):
            with tf.variable_scope("layer{}".format(i + 1)):
                with tf.variable_scope("causal_attention"):
                    x = nrd(x, attention(x, x, mask))
                with tf.variable_scope("attention"):
                    x = nrd(x, attention(w, x))
                with tf.variable_scope("forward"):
                    x = nrd(x, forward(x))
    # output
    with tf.variable_scope('logit'):
        logit = self.logit = tf.tensordot(x, tf.transpose(emb), 1) \
            if logit_share_embedding else tf.layers.dense(x, dim_tgt)
        self.y = logit[:,-1]
    # done
    with tf.variable_scope('eval'):
        self.prob = tf.nn.log_softmax(logit)
        self.pred = tf.to_int32(tf.argmax(logit, -1))
        self.acc = tf.reduce_mean(tf.to_float(tf.equal(gold, self.pred)))
    if training:
        # <experimental: approximate autoregressive>
        with tf.variable_scope('emb_tgt'):
            x = dropout(pos + tf.concat((tf.gather(emb, tgt[:,:1]), tf.tensordot(tf.nn.softmax(logit[:,:-1]), emb, 1)), 1))
        with tf.variable_scope('decode', reuse= True):
            for i in range(num_layer):
                with tf.variable_scope("layer{}".format(i + 1)):
                    with tf.variable_scope("causal_attention"):
                        x = nrd(x, attention(x, x, mask))
                    with tf.variable_scope("attention"):
                        x = nrd(x, attention(w, x))
                    with tf.variable_scope("forward"):
                        x = nrd(x, forward(x))
        with tf.variable_scope('logit', reuse= True):
            logit = tf.tensordot(x, tf.transpose(emb), 1) \
                if logit_share_embedding else tf.layers.dense(x, dim_tgt)
        # </experimental>
        with tf.variable_scope('loss'):
            smooth = self.smooth = tf.placeholder_with_default(smooth, (), 'smooth')
            shared = smooth / dim_tgt
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels= tf.one_hot(gold, dim_tgt, 1.0 - smooth + shared, shared)
                     , logits= logit))
        with tf.variable_scope('lr'):
            self.step = tf.train.get_or_create_global_step()
            step = tf.to_float(self.step + 1)
            self.lr = tf.placeholder_with_default(
                (dim ** -0.5) * tf.minimum(step ** -0.5, step * (warmup ** -1.5))
                , (), 'lr')
        self.up = tf.train.AdamOptimizer(self.lr, 0.9, 0.98, 1e-9).minimize(self.loss, self.step)
    return self
