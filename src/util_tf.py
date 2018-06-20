import tensorflow as tf


def profile(path, sess, run, feed_dict= None, prerun= 3, tag= 'step'):
    for _ in range(prerun): sess.run(run, feed_dict)
    meta = tf.RunMetadata()
    sess.run(run, feed_dict, tf.RunOptions(trace_level= tf.RunOptions.FULL_TRACE), meta)
    with tf.summary.FileWriter(path, sess.graph) as wtr:
        wtr.add_run_metadata(meta, tag)


def batch(data, batch_size, shuffle= 1e4, repeat= True, name= 'batch'):
    """returns a tensorflow dataset iterator from `data`."""
    with tf.variable_scope(name):
        ds = tf.data.Dataset.from_tensor_slices(data)
        if shuffle: ds = ds.shuffle(int(shuffle))
        if repeat:  ds = ds.repeat()
        return ds.batch(batch_size) \
                 .make_one_shot_iterator() \
                 .get_next()


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


def normalize(x, axis= -1, eps= 1e-8
              , gain= True, gain_initializer= tf.ones_initializer()
              , bias= True, bias_initializer= tf.zeros_initializer()
              , name= 'normalize'):
    """returns a tensor from `x` scaled and centered across `axis`."""
    with tf.variable_scope(name):
        mean, var = tf.nn.moments(x, axis, keep_dims=True)
        x = (x - mean) * tf.rsqrt(var + eps * eps)
        if gain or bias: dim = x.shape[-1]
        if gain: x *= tf.get_variable('gain', dim, initializer= gain_initializer)
        if bias: x += tf.get_variable('bias', dim, initializer= bias_initializer)
        return x
