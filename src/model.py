from util import Record, identity
from util_tf import tf, placeholder, Normalize, Smooth, Dropout, Maxout, Linear, Affine, Forward, BiForward
from util_tf import Attention as Attention
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

    def __init__(self, dim, dim_mid, act, name= 'forward'):
        with tf.variable_scope(name):
            self.name = name
            self.forward = Forward(dim, dim, dim_mid, act)
            self.normalize = Normalize(dim)

    def __call__(self, x, dropout, name= None):
        with tf.variable_scope(name or self.name):
            return self.normalize(x + dropout(self.forward(x)))


class BiForwardBlock(Record):

    def __init__(self, dim, dim_mid, act, name= 'forward'):
        with tf.variable_scope(name):
            self.name = name
            self.forward = BiForward(dim, dim, dim, dim_mid, act)
            self.normalize = Normalize(dim)

    def __call__(self, x, w, dropout, name= None):
        with tf.variable_scope(name or self.name):
            return self.normalize(x + dropout(self.forward(x, w)))


class AttentionBlock(Record):

    def __init__(self, dim, num_head, name= 'attention'):
        with tf.variable_scope(name):
            self.name = name
            self.attention = Attention(dim, num_head= num_head)
            self.normalize = Normalize(dim)

    def __call__(self, x, value, dropout, mask= None, name= None):
        with tf.variable_scope(name or self.name):
            return self.normalize(x + dropout(self.attention(x, value, mask)))


class EncodeBlock(Record):

    def __init__(self, dim, dim_mid, num_head, act, name):
        with tf.variable_scope(name):
            self.name = name
            self.attention = AttentionBlock(dim, num_head)
            self.forward = ForwardBlock(dim, dim_mid, act)

    def __call__(self, x, dropout, name= None):
        with tf.variable_scope(name or self.name):
            return self.forward(self.attention(x, x, dropout), dropout)


class DecodeBlock(Record):

    def __init__(self, dim, dim_mid, num_head, act, name):
        with tf.variable_scope(name):
            self.name = name
            self.causal_attention = AttentionBlock(dim, num_head, 'causal_attention')
            self.attention = AttentionBlock(dim, num_head)
            self.forward = BiForwardBlock(dim, dim_mid, act)

    def __call__(self, x, v, w, dropout, mask= None, name= None):
        with tf.variable_scope(name or self.name):
            return self.forward(
                self.causal_attention(x, v, dropout, mask)
                , self.attention(x, w, dropout)
                , dropout)


# # original transformer
# class DecodeBlock(Record):
#     def __init__(self, dim, dim_mid, num_head, act, name):
#         with tf.variable_scope(name):
#             self.name = name
#             self.causal_attention = AttentionBlock(dim, num_head, 'causal_attention')
#             self.attention = AttentionBlock(dim, num_head)
#             self.forward = ForwardBlock(dim, dim_mid, act)
#     def __call__(self, x, v, w, dropout, mask= None, name= None):
#         with tf.variable_scope(name or self.name):
#             return self.forward(self.attention(self.causal_attention(x, v, dropout, mask), w, dropout), dropout)


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
            , logit_share_embedding= False
            , act= Maxout(2)
            , smooth= 0.1
            , dropout= 0.1):
        """-> Transformer with fields

            end : i32 ()
        emb_src : Linear
        emb_tgt : Linear
         encode : tuple EncodeBlock
         decode : tuple DecodeBlock
          logit : Affine
         smooth : Smooth
        dropout : Dropout

        `end` is treated as the padding for both source and target.

        """
        assert not dim % 2 and not dim % num_head
        emb_src = Linear(dim, dim_src, 'emb_src')
        # <--experiment
        emb_tgt = Linear(dim, dim_tgt, 'emb_tgt')
        # emb_tgt = Forward(dim, dim_tgt, dim_mid, act, 'emb_tgt')
        # experiment-->
        with tf.variable_scope('encode'):
            encode = tuple(EncodeBlock(
                dim, dim_mid, num_head, act, "layer{}".format(i + 1))
                           for i in range(num_layer))
        with tf.variable_scope('decode'):
            decode = tuple(DecodeBlock(
                dim, dim_mid, num_head, act, "layer{}".format(i + 1))
                        for i in range(num_layer))
        return Transformer(
            dim= dim, dim_tgt= dim_tgt
            , end= tf.constant(end, tf.int32, (), 'end')
            , emb_src= emb_src, encode= encode
            , emb_tgt= emb_tgt, decode= decode
            , logit= emb_tgt.transpose('logit') if logit_share_embedding else Affine(dim_tgt, dim, 'logit')
            , smooth= Smooth(smooth, dim_tgt)
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
        end, dim = self.end, self.dim
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
            , tgt_= tgt_, tgt= tgt
            , gold= gold
            , **self)

    def autoreg(self, trainable= True, random= False):
        """-> Transformer with new fields, autoregressive

        len_tgt : i32 ()              steps to unfold aka t
         output : f32 (b, t, dim_tgt) prediction on logit scale
           prob : f32 (b, t, dim_tgt) prediction, soft
           pred : i32 (b, t)          prediction, hard
           loss : f32 ()              prediction loss
            acc : f32 ()              accuracy

        must be called after `data`.

        """
        logit, smooth = self.logit, self.smooth
        position, dropout = self.position, self.dropout if trainable else identity
        src, emb_src, encode = self.src, self.emb_src, self.encode
        tgt, emb_tgt, decode = self.tgt, self.emb_tgt, self.decode
        dim, dim_tgt = self.dim, self.dim_tgt
        with tf.variable_scope('emb_src_autoreg'):
            w = emb_src.embed(src)
            w = dropout(w + position(tf.shape(w)[1]))
        with tf.variable_scope('encode_autoreg'):
            for enc in encode: w = enc(w, dropout)
        with tf.variable_scope('decode_autoreg'):
            with tf.variable_scope('init'):
                len_tgt = tf.shape(tgt)[1]
                pos = position(len_tgt)
                i = tf.constant(0)
                # <--experiment
                x = tgt[:,:1] # Linear
                # x = smooth(tgt[:,:1]) # Forward
                # experiment-->
                v = w[:,:0]
                y = tf.reshape(v, (tf.shape(v)[0], 0, dim_tgt))
                p = x[:,1:]
            def autoreg(i, x, vs, y, p):
                # i : ()              time step from 0 to t=len_tgt
                # x : (b, 1|,dim_tgt) x_i
                # v : (b, t, dim)     attention values
                # y : (b, t, dim_tgt) logit over x one step ahead
                # p : (b, t|,dim_tgt) predictions
                with tf.variable_scope('emb_tgt'):
                    # <--experiment
                    x = dropout(emb_tgt.embed(x) + pos[i]) # Linear
                    # x = dropout(emb_tgt(x) + pos[i]) # Forward
                    # experiment-->
                us = []
                for dec, v in zip(decode, vs):
                    with tf.variable_scope('cache_v'):
                        v = tf.concat((v, x), 1)
                        us.append(v)
                    x = dec(x, v, w, dropout)
                x = logit(x)
                with tf.variable_scope('cache_y'): y = tf.concat((y, x), 1)
                # <--experiment
                # Linear
                if random:
                    with tf.variable_scope('sample'):
                        x = tf.expand_dims(tf.multinomial(tf.squeeze(x, 1), 1), 1)
                else:
                    x = tf.argmax(x, -1, output_type= tf.int32, name= 'argmax')
                # x = tf.nn.softmax(x, name= 'softmax') # Forward
                # experiment-->
                with tf.variable_scope('cache_p'): p = tf.concat((p, x), 1)
                return i + 1, x, tuple(us), y, p
            _, _, _, y, p = tf.while_loop(
                lambda i, *_: i < len_tgt # todo stop when end is reached if not trainable
                , autoreg
                , (i, x, (v,)*len(decode), y, p)
                , (i.shape, x.shape, (v.shape,)*len(decode), tf.TensorShape((None, None, dim_tgt)), p.shape)
                , back_prop= trainable
                , swap_memory= True
                , name= 'autoreg')
        return Transformer(len_tgt= len_tgt, output= y, pred= p, **self)._eval()

    def forcing(self, trainable= True):
        """-> Transformer with new fields, teacher forcing

        tgt_prob : f32 (b, t, dim_tgt) soft target feed
          output : f32 (b, t, dim_tgt) prediction on logit scale
            prob : f32 (b, t, dim_tgt) prediction, soft
            pred : i32 (b, t)          prediction, hard
            loss : f32 ()              prediction loss
             acc : f32 ()              accuracy

        must be called after `data`.

        """
        logit, smooth = self.logit, self.smooth
        position, dropout = self.position, self.dropout if trainable else identity
        src, emb_src, encode = self.src, self.emb_src, self.encode
        tgt, emb_tgt, decode = self.tgt, self.emb_tgt, self.decode
        dim_tgt = self.dim_tgt
        with tf.variable_scope('emb_src_forcing'):
            w = emb_src.embed(src)
            w = dropout(w + position(tf.shape(w)[1]))
        with tf.variable_scope('encode_forcing'):
            for enc in encode: w = enc(w, dropout)
        with tf.variable_scope('emb_tgt_forcing'):
            # <--experiment
            x = emb_tgt.embed(tgt) # Linear
            tgt_prob = smooth(tgt) # Forward
            # x = emb_tgt(tgt_prob) # Forward
            # experiment-->
            x = dropout(x + position(tf.shape(x)[1]))
        with tf.variable_scope('decode_forcing'):
            with tf.variable_scope('mask'):
                mask = tf.linalg.LinearOperatorLowerTriangular(tf.ones((tf.shape(x)[1],)*2)).to_dense()
            for dec in decode: x = dec(x, x, w, dropout, mask)
        y = logit(x)
        p = tf.argmax(y, -1, output_type= tf.int32, name= 'pred')
        return Transformer(tgt_prob= tgt_prob, output= y, pred= p, **self)._eval()

    def _eval(self):
        gold, pred, output, smooth = self.gold, self.pred, self.output, self.smooth
        with tf.variable_scope('acc'):
            acc = tf.reduce_mean(tf.to_float(tf.equal(gold, pred)))
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= output, labels= smooth(gold)))
        return Transformer(prob= tf.nn.softmax(output, name= 'prob'), loss= loss, acc= acc, **self)

    def train(self, warmup= 4e3, beta1= 0.9, beta2= 0.98, epsilon= 1e-9):
        """-> Transformer with new fields

        step : i64 () global update step
          lr : f32 () learning rate for the current step
          up :        update operation

        """
        dim, loss = self.dim, self.loss
        with tf.variable_scope('lr'):
            s = tf.train.get_or_create_global_step()
            t = tf.to_float(s + 1)
            lr = (dim ** -0.5) * tf.minimum(t ** -0.5, t * (warmup ** -1.5))
        up = tf.train.AdamOptimizer(lr, beta1, beta2, epsilon).minimize(loss, s)
        return Transformer(step= s, lr= lr, up= up, **self)
