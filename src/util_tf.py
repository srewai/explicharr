from util import Record, identity
import tensorflow as tf


def profile(sess, wtr, run, feed_dict= None, prerun= 3, tag= 'flow'):
    for _ in range(prerun): sess.run(run, feed_dict)
    meta = tf.RunMetadata()
    sess.run(run, feed_dict, tf.RunOptions(trace_level= tf.RunOptions.FULL_TRACE), meta)
    wtr.add_run_metadata(meta, tag)


def batch(data, batch_size, shuffle= 1e4, repeat= True, fn= None, prefetch= 16, name= 'batch'):
    """returns a tensorflow dataset iterator from `data`."""
    with tf.variable_scope(name):
        ds = tf.data.Dataset.from_tensor_slices(data)
        if shuffle: ds = ds.shuffle(int(shuffle))
        if repeat:  ds = ds.repeat()
        ds = ds.batch(batch_size)
        if fn: ds = ds.map(fn)
        if prefetch: ds = ds.prefetch(prefetch)
        return ds.make_one_shot_iterator().get_next()


def placeholder(dtype, shape, x= None, name= None):
    """returns a placeholder with `dtype` and `shape`.

    if tensor `x` is given, converts and uses it as default.

    """
    if x is None: return tf.placeholder(dtype, shape, name)
    try:
        x = tf.convert_to_tensor(x, dtype)
    except ValueError:
        x = tf.cast(x, dtype)
    return tf.placeholder_with_default(x, shape, name)


def normalize(x, axis= -1, eps= 1e-8, name= 'normalize'):
    """returns a tensor from `x` scaled and centered across `axis`."""
    with tf.variable_scope(name):
        mean, var = tf.nn.moments(x, axis, keep_dims=True)
        return (x - mean) * tf.rsqrt(var + eps * eps)


class Normalize(Record):

    def __init__(self, dim
                 , gain_initializer= tf.ones_initializer()
                 , bias_initializer= tf.zeros_initializer()
                 , name= 'normalize'):
        with tf.variable_scope(name):
            self.name = name
            self.gain = tf.get_variable('gain', dim, initializer= gain_initializer)
            self.bias = tf.get_variable('bias', dim, initializer= bias_initializer)

    def __call__(self, x, axis= -1, eps= 1e-8, name= None):
        with tf.variable_scope(name or self.name):
            return normalize(x, axis, eps) * self.gain + self.bias


class Embed(Record):

    def __init__(self, dim, dim_out= None, name= 'embed'):
        if dim_out is None: dim_out = dim
        self.name = name
        self.kern = tf.get_variable(name, (dim, dim_out))

    def __call__(self, x, name= None):
        return tf.gather(self.kern, x, name= name or self.name)


class Dense(Record):

    def __init__(self, dim, dim_out= None, bias= True, name= 'dense'):
        if dim_out is None: dim_out = dim
        with tf.variable_scope(name):
            self.name = name
            self.kern = tf.get_variable('kern', (dim, dim_out))
            self.bias = tf.get_variable('bias', dim_out) if bias else None

    def __call__(self, x, act= None, name= None):
        with tf.variable_scope(name or self.name):
            x = tf.tensordot(x, self.kern, 1)
            if self.bias is not None:
                x += self.bias
            if act is not None:
                x = act(x)
        return x


class Forward(Record):

    def __init__(self, dim, dim_mid= None, dim_out= None, bias= True, name= 'forward'):
        if dim_mid is None: dim_mid = dim
        if dim_out is None: dim_out = dim
        with tf.variable_scope(name):
            self.name = name
            self.mid = Dense(dim, dim_mid, bias= bias, name= 'mid')
            self.out = Dense(dim_mid, dim_out, bias= bias, name= 'out')

    def __call__(self, x, act= None, name= None):
        with tf.variable_scope(name or self.name):
            return self.out(self.mid(x, act))


class Attention(Record):
    """computes multi-head attention from `query` and `value` tensors.

    with batch size `b`, time steps `t, s`, dimensions `q, v`

    - query : b,t,q
    - value : b,s,v

    the returned tensor has shape `b, t, dim`, and `mask` when
    supplied must have shape compatible to `num_head, b, t, s`.

    """

    def __init__(self, dim, dim_q= None, dim_v= None, softmax= True, name= 'attention'):
        if dim_q is None: dim_q = dim
        if dim_v is None: dim_v = dim
        self.dim, self.softmax = dim, softmax
        with tf.variable_scope(name):
            self.name = name
            self.q = Dense(dim_q, dim, bias= False, name= 'q')
            if softmax:
                self.v = Dense(dim_v, dim, bias= False, name= 'v')
                self.k = Dense(dim_v, dim, bias= False, name= 'k')

    def __call__(self, query, value, num_head= 1, mask= None, name= None):
        assert not self.dim % num_head
        if 1 < num_head:
            stack_split = lambda x: tf.stack(tf.split(x, num_head, -1))
        else:
            stack_split = identity
        # v : h,b,s,d
        # k : h,b,s,d
        # q : h,b,t,d
        # a : h,b,t,s
        with tf.variable_scope(name or self.name):
            q = stack_split(self.q(query))
            if self.softmax:
                v = stack_split(self.v(value))
                k = stack_split(self.k(value))
                a = tf.matmul(q, k, transpose_b= True)
                a *= (self.dim ** -0.5)
                if mask is not None: a += mask
                a = tf.nn.softmax(a)
            else:
                v = k = stack_split(value)
                a = tf.matmul(q, k, transpose_b= True)
                if mask is not None: a *= mask
                a = tf.square(a)
                a /= tf.reduce_sum(a, -1, True) + 1e-8
            x = a @ v
            if 1 < num_head:
                x = tf.concat(tf.unstack(x), -1)
        return x


class Dropout(Record):
    """dropout shape must be a tuple of None or 1 or a fixed known
    dimension, such as `(None, 1, 256)`.  when applied to a tensor,
    None will be filled, and the whole shape broadcast to fit.

    """

    def __init__(self, rate, shape= None, name= 'dropout'):
        self.shape = shape
        with tf.variable_scope(name):
            self.name = name
            self.rate = placeholder(tf.float32, (), rate, 'rate')
            self.keep = 1.0 - self.rate

    def __call__(self, x, name= None):
        with tf.variable_scope(name or self.name):
            if self.shape is not None:
                shape = tf.shape(x)
                shape = [s or shape[i] for i, s in enumerate(self.shape)]
            return tf.nn.dropout(x, self.keep, shape)


class Smooth(Record):
    """binary smoothing if dim is None or one-hot smoothing"""

    def __init__(self, rate, dim= None, name= 'smooth'):
        self.dim = dim
        with tf.variable_scope(name):
            self.name = name
            self.rate = placeholder(tf.float32, (), rate, 'rate')
            self.shared = self.rate / (dim or 2)
            self.smooth = 1.0 - self.rate
            if dim: self.smooth += self.shared

    def __call__(self, x, name= None):
        if self.dim:
            return tf.one_hot(x, self.dim, self.smooth, self.shared, name= name or self.name)
        else:
            with tf.variable_scope(name or self.name):
                return x * self.smooth + self.shared
