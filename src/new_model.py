from util import Record, identity
from new_util_tf import tf, placeholder, Normalize, Dense, Forward, Attention, Dropout, SmoothOnehot
import numpy as np


def sinusoid(time, dim, freq= 1e-4, name= 'sinusoid', scale= True, array= False):
    """returns a rank-2 tensor of shape `time, dim`, where each row
    corresponds to a time step and each column a sinusoid, with
    frequencies in a geometric progression from 1 to `freq`.

    """
    assert not dim % 2
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


class ForwardBlock(Record):

    def __init__(self, dim, dim_mid, name= 'forward'):
        self.name = name
        with tf.variable_scope(name) as name:
            self.forward = Forward(dim, dim_mid)
            self.normalize = Normalize(dim)

    def __call__(self, x, dropout, activation, name= None):
        with tf.variable_scope(name or self.name):
            y = dropout(self.forward(x, activation))
            return self.normalize(x + y)


class AttentionBlock(Record):

    def __init__(self, dim, softmax, name= 'attention'):
        self.name = name
        with tf.variable_scope(name) as name:
            self.attention = Attention(dim, softmax= softmax)
            self.normalize = Normalize(dim)

    def __call__(self, x, value, dropout, num_head, mask= None, name= None):
        with tf.variable_scope(name or self.name):
            y = dropout(self.attention(x, value, num_head, mask))
            return self.normalize(x + y)


class EncodeBlock(Record):

    def __init__(self, dim, dim_mid, num_head, activation, softmax, name):
        self.num_head = num_head
        self.activation = activation
        self.softmax = softmax
        self.name = name
        with tf.variable_scope(name) as name:
            self.attention = AttentionBlock(dim, softmax)
            self.forward = ForwardBlock(dim, dim_mid)

    def __call__(self, x, dropout, name= None):
        with tf.variable_scope(name or self.name):
            x = self.attention(x, x, dropout, self.num_head)
            x = self.forward(x, dropout, self.activation)
        return x


class DecodeBlock(Record):

    def __init__(self, dim, dim_mid, num_head, activation, softmax, name):
        self.num_head = num_head
        self.activation = activation
        self.softmax = softmax
        self.name = name
        with tf.variable_scope(name) as name:
            self.causal = AttentionBlock(dim, softmax, 'causal')
            self.attention = AttentionBlock(dim, softmax)
            self.forward = ForwardBlock(dim, dim_mid)

    def __call__(self, x, v, w, dropout, mask= None, name= None):
        with tf.variable_scope(name or self.name):
            x = self.causal(x, v, dropout, self.num_head, mask)
            x = self.attention(x, w, dropout, self.num_head)
            x = self.forward(x, dropout, self.activation)
        return x


# len_cap = 11
# src = tgt = None
# dim_src = 5
# dim_tgt = 7
# dim = 8
# dim_mid = 16
# num_head = 4
# num_layer = 2
# softmax = True
# activation = tf.nn.relu
# logit_share_embedding = False
# training = True
# dropout = 0.1
# warmup = 4e3
# end = 1


class Transformer(Record):
    """-> Record, with the following fields of tensors

       end : i32 ()              end padding for `src` and `tgt`
       src : i32 (b, s)          source feed, in range `[0, dim_src)`
       tgt : f32 (b, t)          target feed, in range `[0, dim_tgt)`
      pred : i32 (b, t)          prediction
     logit : f32 (b, t, dim_tgt) prediction on logit scale
    smooth : f32 ()              prediction smoothing
      loss : f32 ()              prediction loss
       acc : f32 ()              accuracy

    and if `training`

    dropout : f32 () dropout rate
       step : i64 () global update step
         lr : f32 () learning rate for the current step
         up :        update operation

    setting `len_cap` makes it more efficient for training.  you won't
    be able to feed it longer sequences, but it doesn't affect any
    model parameters.

    """

    @staticmethod
    def new(dim= 256
            , dim_src= 256
            , dim_tgt= 256
            , dim_mid= 512
            , num_layer= 2
            , num_head= 4
            , softmax= True
            , activation= tf.nn.relu
            , logit_share_embedding= False
            , dropout= 0.1
            , smooth= 0.1
            , end= 1):
        assert not dim % 2 and not dim % num_head
        init = tf.orthogonal_initializer()
        emb_src = tf.get_variable('emb_src', (dim_src, dim), tf.float32, init)
        emb_tgt = tf.get_variable('emb_tgt', (dim_tgt, dim), tf.float32, init)
        with tf.variable_scope('encode'):
            encode = tuple(EncodeBlock(
                dim, dim_mid, num_head, activation, softmax, "layer{}".format(i + 1))
                             for i in range(num_layer))
        with tf.variable_scope('decode'):
            decode = tuple(DecodeBlock(
                dim, dim_mid, num_head, activation, softmax, "layer{}".format(i + 1))
                        for i in range(num_layer))
        if logit_share_embedding:
            with tf.variable_scope('logit'):
                def logit(x, w= tf.transpose(emb_tgt)):
                    with tf.variable_scope('logit'):
                        return tf.tensordot(x, w, 1)
        else:
            logit = Dense(dim, dim_tgt, name= 'logit')
        return Transformer(
            end= tf.constant(end, tf.int32, (), 'end')
            , dropout= Dropout(dropout, (None, 1, dim))
            , smooth= SmoothOnehot(smooth, dim_tgt)
            , emb_src= emb_src
            , emb_tgt= emb_tgt
            , encode= encode
            , decode= decode
            , logit= logit)

    def prep(self, src= None, tgt= None, len_cap= None):
        dim = int(self.emb_tgt.shape[1])
        end = self.end
        count_not_all = lambda x: tf.reduce_sum(tf.to_int32(~ tf.reduce_all(x, 0)))
        # trim `src` to the maximum valid index among the batch, plus one for padding
        with tf.variable_scope('src'):
            src = placeholder(tf.int32, (None, None), src)
            len_src = count_not_all(tf.equal(src, end)) + 1
            src = src[:,:len_src]
        # same for `tgt`, but with one less index
        with tf.variable_scope('tgt'):
            tgt = placeholder(tf.int32, (None, None), tgt)
            len_tgt = count_not_all(tf.equal(tgt, end))
            tgt, gold = tgt[:,:len_tgt], tgt[:,1:1+len_tgt]
        # sinusoidal positional encoding
        if len_cap:
            position = lambda t, dim= dim, sinusoid= tf.constant(
                sinusoid(len_cap, dim, array= True), tf.float32, name= 'sinusoid'
            ): sinusoid[:t]
        else:
            position = lambda t, dim= dim: sinusoid(t, dim)
        return Transformer(src= src, tgt= tgt, gold= gold, position= position, **self)

    def autoreg(self, trainable= False):
        position, logit, dropout = self.position, self.logit, self.dropout if trainable else identity
        src, emb_src, encode = self.src, self.emb_src, self.encode
        tgt, emb_tgt, decode = self.tgt, self.emb_tgt, self.decode
        dim_tgt, dim = map(int, emb_tgt.shape)
        with tf.variable_scope('emb_src_autoreg'):
            w = tf.gather(emb_src, src)
            w = dropout(w + position(tf.shape(w)[1]))
        with tf.variable_scope('encode_autoreg'):
            for enc in encode:
                w = enc(w, dropout)
        with tf.variable_scope('decode_autoreg'):
            len_tgt = tf.shape(tgt)[1]
            pos = position(len_tgt)
            def autoreg(i, x, v, y):
                # i : ()              time step from 0 to t=len_tgt
                # x : (b, 1, dim_tgt) prob dist over x_i
                # v : (b, t, dim)     embeded x
                # y : (b, t, dim_tgt) logit over x one step ahead
                # todo find way around concat
                x = dropout(tf.tensordot(x, emb_tgt, 1) + pos[i])
                v = tf.concat((v, x), 1)
                for dec in decode: x = dec(x, v, w, dropout)
                x = logit(x)
                y = tf.concat((y, x), 1)
                x = tf.nn.softmax(x)
                return i + 1, x, v, y
            x = tf.one_hot(tgt[:,:1], dim_tgt) # todo try smoothing
            y = x[:,1:]
            _, _, _, y = tf.while_loop(
                lambda i, *_: i < len_tgt
                , autoreg
                , (0, x, tf.reshape(y, (tf.shape(y)[0], 0, dim)), y)
                , (tf.TensorShape(()), x.shape, tf.TensorShape((None, None, dim)), y.shape)
                , back_prop= trainable
                , name= 'autoreg')
        return Transformer(len_tgt= len_tgt, output= y, **self)

    def forcing(self):
        position, logit, dropout = self.position, self.logit, self.dropout
        src, emb_src, encode = self.src, self.emb_src, self.encode
        tgt, emb_tgt, decode = self.tgt, self.emb_tgt, self.decode
        with tf.variable_scope('emb_src_forcing'):
            w = tf.gather(emb_src, src)
            w = dropout(w + position(tf.shape(w)[1]))
        with tf.variable_scope('encode_forcing'):
            for enc in encode:
                w = enc(w, dropout)
        with tf.variable_scope('emb_tgt_forcing'):
            tgt_prob = tf.one_hot(tgt, int(emb_tgt.shape[0])) # todo try smoothing
            x = tf.tensordot(tgt_prob, emb_tgt, 1)
            x = dropout(x + position(tf.shape(x)[1]))
        with tf.variable_scope('decode_forcing'):
            with tf.variable_scope('mask'):
                mask = tf.linalg.LinearOperatorLowerTriangular(tf.ones((tf.shape(x)[1],)*2)).to_dense()
                if self.decode[0].softmax: mask = tf.log(mask)
            for dec in decode:
                x = dec(x, x, w, dropout, mask)
        with tf.variable_scope('logit_forcing'):
            y = logit(x)
        return Transformer(tgt_prob= tgt_prob, output= y, **self)

    def post(self):
        gold, output, smooth = self.gold, self.output, self.smooth
        with tf.variable_scope('pred'):
            prob = tf.nn.softmax(output)
            pred = tf.to_int32(tf.argmax(output, -1))
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= output, labels= smooth(gold)))
        with tf.variable_scope('acc'):
            acc = tf.reduce_mean(tf.to_float(tf.equal(gold, pred)))
        return Transformer(prob= prob, pred= pred, loss= loss, acc= acc, **self)

    def train(self, warmup= 4e3, beta1= 0.9, beta2= 0.98, epsilon= 1e-9):
        dim, loss = int(self.emb_tgt.shape[1]), self.loss
        with tf.variable_scope('lr'):
            s = tf.train.get_or_create_global_step()
            t = tf.to_float(s + 1)
            lr = placeholder(tf.float32, (), (dim ** -0.5) * tf.minimum(t ** -0.5, t * (warmup ** -1.5)), 'lr')
        up = tf.train.AdamOptimizer(lr, beta1, beta2, epsilon).minimize(loss, s)
        return Transformer(step= s, lr= lr, up= up, **self)
