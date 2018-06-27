from util import Record, identity
from util_tf import tf, placeholder, Normalize, Dense, Forward, Attention, Dropout, SmoothOnehot
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


class Sinusoid(Record):

    def __init__(self, dim, len_cap= None, name= 'sinusoid'):
        self.dim, self.name = dim, name
        self.pos = tf.constant(
            sinusoid(len_cap, dim, array= True), tf.float32, name= name
        ) if len_cap else None

    def __call__(self, time, name= None):
        with tf.variable_scope(name or self.name):
            return sinusoid(time, self.dim) if self.pos is None else self.pos[:time]


class ForwardBlock(Record):

    def __init__(self, dim, dim_mid, name= 'forward'):
        with tf.variable_scope(name):
            self.name = name
            self.forward = Forward(dim, dim_mid)
            self.normalize = Normalize(dim)

    def __call__(self, x, dropout, act, name= None):
        with tf.variable_scope(name or self.name):
            return self.normalize(x + dropout(self.forward(x, act)))


class AttentionBlock(Record):

    def __init__(self, dim, softmax, name= 'attention'):
        with tf.variable_scope(name):
            self.name = name
            self.attention = Attention(dim, softmax= softmax)
            self.normalize = Normalize(dim)

    def __call__(self, x, value, dropout, num_head, mask= None, name= None):
        with tf.variable_scope(name or self.name):
            return self.normalize(x + dropout(self.attention(x, value, num_head, mask)))


class EncodeBlock(Record):

    def __init__(self, dim, dim_mid, num_head, act, softmax, name):
        self.num_head, self.act, self.softmax = num_head, act, softmax
        with tf.variable_scope(name):
            self.name = name
            self.attention = AttentionBlock(dim, softmax)
            self.forward = ForwardBlock(dim, dim_mid)

    def __call__(self, x, dropout, name= None):
        with tf.variable_scope(name or self.name):
            x = self.attention(x, x, dropout, self.num_head)
            x = self.forward(x, dropout, self.act)
        return x


class DecodeBlock(Record):

    def __init__(self, dim, dim_mid, num_head, act, softmax, name):
        self.num_head, self.act, self.softmax = num_head, act, softmax
        with tf.variable_scope(name):
            self.name = name
            self.causal = AttentionBlock(dim, softmax, 'causal')
            self.attention = AttentionBlock(dim, softmax)
            self.forward = ForwardBlock(dim, dim_mid)

    def __call__(self, x, v, w, dropout, mask= None, name= None):
        with tf.variable_scope(name or self.name):
            x = self.causal(x, v, dropout, self.num_head, mask)
            x = self.attention(x, w, dropout, self.num_head)
            x = self.forward(x, dropout, self.act)
        return x


class Transformer(Record):
    """-> Record

    model = Transformer.new()
    model_train = model.data(src_train, tgt_train, len_cap)
    model_valid = model.data(src_valid, tgt_valid)

    forcing_train = model_train.forcing().train()
    forcing_valid = model_valid.forcing()

    autoreg_train = model_train.autoreg(trainable= True).train()
    autoreg_valid = model_valid.autoreg(trainable= False)

    """

    @staticmethod
    def new(end= 1
            , dim_src= 256, dim= 256
            , dim_tgt= 256, dim_mid= 512
            , num_layer= 2, num_head= 4
            , softmax= True
            , act= tf.nn.relu
            , logit_share_embedding= False
            , smooth= 0.1
            , dropout= 0.1):
        """-> Transformer with fields

            end : i32 ()
        emb_src : f32 (dim_src, dim)
        emb_tgt : f32 (dim_tgt, dim)
         encode : tuple EncodeBlock
         decode : tuple DecodeBlock
          logit : dim -> dim_tgt
         smooth : SmoothOnehot
        dropout : Dropout

        `end` is treated as the padding for both source and target.

        """
        assert not dim % 2 and not dim % num_head
        init = tf.orthogonal_initializer()
        emb_src = tf.get_variable('emb_src', (dim_src, dim), tf.float32, init)
        emb_tgt = tf.get_variable('emb_tgt', (dim_tgt, dim), tf.float32, init)
        with tf.variable_scope('encode'):
            encode = tuple(EncodeBlock(
                dim, dim_mid, num_head, act, softmax, "layer{}".format(i + 1))
                             for i in range(num_layer))
        with tf.variable_scope('decode'):
            decode = tuple(DecodeBlock(
                dim, dim_mid, num_head, act, softmax, "layer{}".format(i + 1))
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
            , emb_src= emb_src, encode= encode
            , emb_tgt= emb_tgt, decode= decode, logit= logit
            , smooth= SmoothOnehot(smooth, dim_tgt)
            , dropout= Dropout(dropout, (None, 1, dim)))

    def data(self, src= None, tgt= None, len_cap= None):
        """-> Transformer with new fields

            src_ : i32 (b, ?) source feed, in range `[0, dim_src)`
            tgt_ : i32 (b, ?) target feed, in range `[0, dim_tgt)`
             src : i32 (b, s) source with `end` trimmed among the batch
             tgt : i32 (b, t) target with `end` trimmed among the batch
            gold : i32 (b, t) target one step ahead
        position : Sinusoid

        setting `len_cap` makes it more efficient for training.  you
        won't be able to feed it longer sequences, but it doesn't
        affect any model parameters.

        """
        dim = int(self.emb_tgt.shape[1])
        end = self.end
        count_not_all = lambda x: tf.reduce_sum(tf.to_int32(~ tf.reduce_all(x, 0)))
        with tf.variable_scope('src'):
            src = src_ = placeholder(tf.int32, (None, None), src)
            len_src = count_not_all(tf.equal(src, end)) + 1
            src = src[:,:len_src]
        with tf.variable_scope('tgt'):
            tgt = tgt_ = placeholder(tf.int32, (None, None), tgt)
            len_tgt = count_not_all(tf.equal(tgt, end))
            tgt, gold = tgt[:,:len_tgt], tgt[:,1:1+len_tgt]
        return Transformer(
            position= Sinusoid(dim, len_cap)
            , src_= src_, src= src
            , tgt_= tgt_, tgt= tgt, gold= gold
            , **self)

    def autoreg(self, trainable= True):
        """-> Transformer with new fields, autoregressive

        len_tgt : i32 ()              steps to unfold aka t
         output : f32 (b, t, dim_tgt) prediction on logit scale
           prob : f32 (b, t, dim_tgt) prediction, soft
           pred : i32 (b, t)          prediction, hard
           loss : f32 ()              prediction loss
            acc : f32 ()              accuracy

        must be called after `data`.

        """
        position, logit, dropout = self.position, self.logit, self.dropout if trainable else identity
        src, emb_src, encode = self.src, self.emb_src, self.encode
        tgt, emb_tgt, decode = self.tgt, self.emb_tgt, self.decode
        dim_tgt, dim = map(int, emb_tgt.shape)
        with tf.variable_scope('emb_src_autoreg'):
            w = tf.gather(emb_src, src)
            w = dropout(w + position(tf.shape(w)[1]))
        with tf.variable_scope('encode_autoreg'):
            for enc in encode: w = enc(w, dropout)
        with tf.variable_scope('decode_autoreg'):
            with tf.variable_scope('init'):
                len_tgt = tf.shape(tgt)[1]
                pos = position(len_tgt)
                x = tf.one_hot(tgt[:,:1], dim_tgt) # todo try smoothing
                y = x[:,1:]
                v = tf.reshape(y, (tf.shape(y)[0], 0, dim))
            def autoreg(i, x, vs, y):
                # i : ()              time step from 0 to t=len_tgt
                # x : (b, 1, dim_tgt) prob dist over x_i
                # v : (b, t, dim)     embeded x
                # y : (b, t, dim_tgt) logit over x one step ahead
                with tf.variable_scope('emb_tgt'): x = dropout(tf.tensordot(x, emb_tgt, 1) + pos[i])
                us = []
                for dec, v in zip(decode, vs):
                    with tf.variable_scope('cache_v'):
                        v = tf.concat((v, x), 1)
                        us.append(v)
                    x = dec(x, v, w, dropout)
                x = logit(x)
                with tf.variable_scope('cache_y'): y = tf.concat((y, x), 1)
                # with tf.variable_scope('softmax'): x = tf.nn.softmax(x)
                with tf.variable_scope('hardmax'): x = tf.one_hot(tf.argmax(x, -1), dim_tgt)
                return i + 1, x, tuple(us), y
            _, _, _, y = tf.while_loop(
                lambda i, *_: i < len_tgt # todo stop when end is reached if not trainable
                , autoreg
                , (0, x, (v,) * len(decode), y)
                , (tf.TensorShape(()), x.shape, (tf.TensorShape((None, None, dim)),) * len(decode), y.shape)
                , back_prop= trainable
                , swap_memory= True
                , name= 'autoreg')
        return Transformer(len_tgt= len_tgt, output= y, **self)._pred()

    def forcing(self):
        """-> Transformer with new fields, teacher forcing

        tgt_prob : f32 (b, t, dim_tgt) soft target feed
          output : f32 (b, t, dim_tgt) prediction on logit scale
            prob : f32 (b, t, dim_tgt) prediction, soft
            pred : i32 (b, t)          prediction, hard
            loss : f32 ()              prediction loss
             acc : f32 ()              accuracy

        must be called after `data`.

        """
        position, logit, dropout = self.position, self.logit, self.dropout
        src, emb_src, encode = self.src, self.emb_src, self.encode
        tgt, emb_tgt, decode = self.tgt, self.emb_tgt, self.decode
        with tf.variable_scope('emb_src_forcing'):
            w = tf.gather(emb_src, src)
            w = dropout(w + position(tf.shape(w)[1]))
        with tf.variable_scope('emb_tgt_forcing'):
            tgt_prob = tf.one_hot(tgt, int(emb_tgt.shape[0])) # todo try smoothing and make tgt_prob optional
            x = tf.tensordot(tgt_prob, emb_tgt, 1)
            x = dropout(x + position(tf.shape(x)[1]))
        with tf.variable_scope('encode_forcing'):
            for enc in encode: w = enc(w, dropout)
        with tf.variable_scope('decode_forcing'):
            with tf.variable_scope('mask'):
                mask = tf.linalg.LinearOperatorLowerTriangular(tf.ones((tf.shape(x)[1],)*2)).to_dense()
                if self.decode[0].softmax: mask = tf.log(mask)
            for dec in decode: x = dec(x, x, w, dropout, mask)
        with tf.variable_scope('logit_forcing'):
            y = logit(x)
        return Transformer(tgt_prob= tgt_prob, output= y, **self)._pred()

    def _pred(self):
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
        """-> Transformer with new fields

        step : i64 () global update step
          lr : f32 () learning rate for the current step
          up :        update operation

        """
        dim, loss = int(self.emb_tgt.shape[1]), self.loss
        with tf.variable_scope('lr'):
            s = tf.train.get_or_create_global_step()
            t = tf.to_float(s + 1)
            lr = placeholder(tf.float32, (), (dim ** -0.5) * tf.minimum(t ** -0.5, t * (warmup ** -1.5)), 'lr')
        up = tf.train.AdamOptimizer(lr, beta1, beta2, epsilon).minimize(loss, s)
        return Transformer(step= s, lr= lr, up= up, **self)
