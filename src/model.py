from record import Record
import numpy as np
import tensorflow as tf


def sinusoid(n, t):
    a = (1e-8 ** (np.arange(int(n / 2)) / n)).reshape(-1, 1) @ np.arange(t).reshape(1, -1)
    return np.concatenate((np.sin(a), np.cos(a)), -1).reshape(n, t).T


def norm_res_drop(x, res, dropout, training):
    return tf.contrib.layers.layer_norm(res + tf.layers.dropout(x, dropout, training= training))


def multihead_attention(value, query, dense, dim, num_head= 8, mask= None, name= 'attention'):
    # value : ?,s,k
    # query : ?,t,q
    with tf.variable_scope(name):
        v = tf.stack(tf.split(dense(value, dim * num_head, name= 'v'), num_head, -1)) # h,?,s,d
        k = tf.stack(tf.split(dense(value, dim * num_head, name= 'k'), num_head, -1)) # h,?,s,d
        q = tf.stack(tf.split(dense(query, dim * num_head, name= 'q'), num_head, -1)) # h,?,t,d
        q = (dim ** -0.5) * tf.matmul(q, k, transpose_b= True)                        # h,?,t,s
        if mask is not None: q += tf.log(mask)
        return tf.concat(tf.unstack(tf.matmul(tf.nn.softmax(q), v)), -1)              # ?,t,dh


def model(trainable= True
          , src= None, dim_src= 256, len_src= None
          , tgt= None, dim_tgt= 256, len_tgt= None
          , dim= 512, dim_mid= 2048, len_cap= 512
          , num_layer= 6, num_head= 8
          , kinit= tf.orthogonal_initializer()
          , binit= tf.zeros_initializer()
          , act= tf.nn.relu
          , smooth= 0.1
          , dropout= 0.1
          , warmup= 4e3):
    # TODO efficient generation
    assert not dim % 2 and not dim % num_head
    # placeholder for generation
    if src is None: src = tf.placeholder(tf.int32, (None, len_src), 'src')
    if tgt is None: tgt = tf.placeholder(tf.int32, (None, len_tgt), 'tgt')
    # prep
    if len_src is None: len_src = src.shape[-1]
    if len_tgt is None: len_tgt = tgt.shape[-1]
    len_tgt -= 1
    with tf.variable_scope('training'):
        training = tf.placeholder_with_default(True, (), 'training')
        dropout = tf.placeholder_with_default(dropout, (), 'dropout')
        step = tf.train.get_or_create_global_step()
        s = tf.to_float(step)
        lr = tf.placeholder_with_default(
            (dim ** -0.5) * tf.minimum(s ** -0.5, s * (warmup ** -1.5))
            , (), 'lr')
    nrd = lambda res, x: norm_res_drop(x, res, dropout, training)
    dense = lambda x, d, **args: tf.layers.dense(
        inputs= x, units= d, kernel_initializer= kinit, bias_initializer= binit, **args)
    attention = lambda v, q, **args: multihead_attention(
        value= v, query= q, dense= dense, dim= int(dim/num_head), num_head= num_head, **args)
    # sinusoidal positional encoding
    pos = tf.constant(sinusoid(dim, len_cap), tf.float32, name= 'sinusoid')
    with tf.variable_scope('encode'):
        with tf.variable_scope('embed'):
            x = tf.gather(tf.get_variable('src', (dim_src, dim), tf.float32, kinit), src)
            x += pos[:len_src]
            x = tf.layers.dropout(x, dropout, training= training)
        for i in range(num_layer):
            with tf.variable_scope("layer{}".format(i)):
                x = nrd(x, attention(x, x))
                r = dense(x, dim_mid, activation= act, name= 'relu')
                x = nrd(x, dense(r, dim)) # why no activation ????
    with tf.variable_scope('decode'):
        with tf.variable_scope('embed'):
            y = tf.gather(tf.get_variable('tgt', (dim_tgt, dim), tf.float32, kinit), tgt[:,:-1])
            y += pos[:len_tgt]
            y = tf.layers.dropout(y, dropout, training= training)
        with tf.variable_scope('mask'):
            mask = tf.linalg.LinearOperatorLowerTriangular(tf.ones((len_tgt, len_tgt))).to_dense()
        for i in range(num_layer):
            with tf.variable_scope("layer{}".format(i)):
                y = nrd(y, attention(y, y, mask= mask, name= 'masked_attention'))
                y = nrd(y, attention(x, y))
                r = dense(y, dim_mid, activation= act, name= 'relu')
                y = nrd(y, dense(r, dim)) # why no activation ????
        logit = dense(y, dim_tgt, name= 'logit')
    with tf.variable_scope('gold'): gold = tgt[:,1:]
    with tf.variable_scope('pred'): pred = tf.to_int32(tf.argmax(logit, -1))
    with tf.variable_scope('acc'):
        mask = tf.to_float(tf.not_equal(gold, 1))  # idx= 1 for padding
        acc = tf.reduce_sum(tf.to_float(tf.equal(pred, gold)) * mask) / tf.reduce_sum(mask)
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels= tf.gather(tf.eye(dim_tgt) * (1 - smooth) + (smooth / dim_tgt), gold)
            , logits= logit))
    up = tf.train.AdamOptimizer(lr, beta1= 0.9, beta2= 0.98, epsilon= 1e-9).minimize(loss, step)
    return Record(
        src= src, tgt= tgt, dropout= dropout, training= training
        , pred= pred, acc= acc, loss= loss, step= step, lr= lr, up= up)


m = model(dim_src= 256, len_src= 64, dim_tgt= 256, len_tgt= 64, dim= 128, dim_mid= 512, len_cap= 64, num_layer= 2, num_head= 4)
wtr = tf.summary.FileWriter("log", tf.get_default_graph())
