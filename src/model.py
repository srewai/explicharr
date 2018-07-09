from util import Record, identity
from util_tf import SquareAttention as Attention # experiment
from util_tf import tf, placeholder, Normalize, Smooth, Dropout, Linear, Affine, Forward
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


class EncodeBlock(Record):

    def __init__(self, dim, dim_mid, act, name):
        with tf.variable_scope(name):
            self.name = name
            with tf.variable_scope('att'):
                self.att = Attention(dim, layer= Forward, mid= dim_mid, act= act) # experiment
                self.norm_att = Normalize(dim)
            with tf.variable_scope('fwd'):
                self.fwd = Forward(dim, dim, dim_mid, act)
                self.norm_fwd = Normalize(dim)

    def __call__(self, x, dropout, name= None):
        with tf.variable_scope(name or self.name):
            with tf.variable_scope('att'): x = self.norm_att(x + dropout(self.att(x, x)))
            with tf.variable_scope('fwd'): x = self.norm_fwd(x + dropout(self.fwd(x)))
            return x


class DecodeBlock(Record):

    def __init__(self, dim, dim_mid, act, name):
        with tf.variable_scope(name):
            self.name = name
            with tf.variable_scope('att'):
                self.csl = Attention(dim, layer= Forward, mid= dim_mid, act= act, name= 'causal') # experiment
                self.att = Attention(dim, layer= Forward, mid= dim_mid, act= act) # experiment
                self.norm_att = Normalize(dim)
            with tf.variable_scope('fwd'):
                self.fwd = Forward(dim, dim, dim_mid, act)
                self.norm_fwd = Normalize(dim)

    def __call__(self, x, v, w, dropout, mask= None, name= None):
        with tf.variable_scope(name or self.name):
            with tf.variable_scope('att'): x = self.norm_att(x + dropout(self.att(x, w) + self.csl(x, v, mask)))
            with tf.variable_scope('fwd'): x = self.norm_fwd(x + dropout(self.fwd(x)))
            return x


# # original transformer
# from util_tf import SoftmaxAttention as Attention
# class DecodeBlock(Record):
#     def __init__(self, dim, dim_mid, act, name, num_head):
#         with tf.variable_scope(name):
#             self.name = name
#             with tf.variable_scope('csl'):
#                 self.csl = Attention(dim, num_head= num_head)
#                 self.norm_csl = Normalize(dim)
#             with tf.variable_scope('att'):
#                 self.att = Attention(dim, num_head= num_head)
#                 self.norm_att = Normalize(dim)
#             with tf.variable_scope('fwd'):
#                 self.fwd = Forward(dim, dim, dim_mid, act)
#                 self.norm_fwd = Normalize(dim)
#     def __call__(self, x, v, w, dropout, mask= None, name= None):
#         with tf.variable_scope(name or self.name):
#             with tf.variable_scope('csl'): x = self.norm_csl(x + dropout(self.csl(x, v, mask)))
#             with tf.variable_scope('att'): x = self.norm_att(x + dropout(self.att(x, w)))
#             with tf.variable_scope('fwd'): x = self.norm_fwd(x + dropout(self.fwd(x)))
#             return x


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
            , dim_tgt= 256, dim_mid= 512, num_layer= 2
            , logit_share_embedding= False
            , act= tf.nn.relu
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
        assert not dim % 2
        emb_src = Linear(dim, dim_src, 'emb_src')
        emb_tgt = Linear(dim, dim_tgt, 'emb_tgt')
        with tf.variable_scope('encode'):
            encode = tuple(EncodeBlock(dim, dim_mid, act, "layer{}".format(1+i)) for i in range(num_layer))
        with tf.variable_scope('decode'):
            decode = tuple(DecodeBlock(dim, dim_mid, act, "layer{}".format(1+i)) for i in range(num_layer))
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

    def autoreg(self, trainable= False, random= False, minimal= True):
        """-> Transformer with new fields, autoregressive

        len_tgt : i32 ()              steps to unfold aka t
         output : f32 (b, t, dim_tgt) prediction on logit scale
           prob : f32 (b, t, dim_tgt) prediction, soft
           pred : i32 (b, t)          prediction, hard
           loss : f32 ()              prediction loss
            acc : f32 ()              accuracy

        must be called after `data`.

        """
        assert not trainable or not random
        assert not trainable or not minimal
        end, dim_tgt, logit = self.end, self.dim_tgt, self.logit
        position, dropout = self.position, self.dropout if trainable else identity
        src, emb_src, encode = self.src, self.emb_src, self.encode
        tgt, emb_tgt, decode = self.tgt, self.emb_tgt, self.decode
        with tf.variable_scope('emb_src_autoreg'): w = position(tf.shape(src)[1]) + dropout(emb_src.embed(src))
        with tf.variable_scope('encode_autoreg'):
            for enc in encode: w = enc(w, dropout)
        with tf.variable_scope('decode_autoreg'):
            with tf.variable_scope('init'):
                len_tgt = tf.shape(tgt)[1]
                pos = position(len_tgt)
                i = tf.constant(0)
                x = tgt[:,:1]
                v = w[:,:0]
                y = tf.reshape(v, (tf.shape(v)[0], 0, dim_tgt))
                p = x[:,1:]
            def autoreg(i, x, vs, y, p):
                # i : ()              time step from 0 to t=len_tgt
                # x : (b, 1)          x_i
                # v : (b, t, dim)     attention values
                # y : (b, t, dim_tgt) logit over x one step ahead
                # p : (b, t)          predictions
                with tf.variable_scope('emb_tgt'): x = pos[i] + dropout(emb_tgt.embed(x))
                us = []
                for dec, v in zip(decode, vs):
                    with tf.variable_scope('cache_v'):
                        v = tf.concat((v, x), 1)
                        us.append(v)
                    x = dec(x, v, w, dropout)
                x = logit(x)
                with tf.variable_scope('cache_y'): y = tf.concat((y, x), 1)
                if random:
                    with tf.variable_scope('sample'):
                        x = tf.multinomial(tf.squeeze(x, 1), 1, output_dtype= tf.int32)
                else:
                    x = tf.argmax(x, -1, output_type= tf.int32, name= 'argmax')
                with tf.variable_scope('cache_p'): p = tf.concat((p, x), 1)
                return i + 1, x, tuple(us), y, p
            _, _, _, y, p = tf.while_loop(
                lambda i, x, *_: ((i < len_tgt) & ~ tf.reduce_all(tf.equal(x, end))) if minimal else (i < len_tgt)
                , autoreg
                , (i, x, (v,)*len(decode), y, p)
                , (i.shape, x.shape, (v.shape,)*len(decode), tf.TensorShape((None, None, dim_tgt)), p.shape)
                , back_prop= trainable
                , swap_memory= True
                , name= 'autoreg')
        return Transformer(len_tgt= len_tgt, output= y, pred= p, **self)._eval()

    def forcing(self, trainable= True):
        """-> Transformer with new fields, teacher forcing

          output : f32 (b, t, dim_tgt) prediction on logit scale
            prob : f32 (b, t, dim_tgt) prediction, soft
            pred : i32 (b, t)          prediction, hard
            loss : f32 ()              prediction loss
             acc : f32 ()              accuracy

        must be called after `data`.

        """
        logit, position, dropout = self.logit, self.position, self.dropout if trainable else identity
        src, emb_src, encode = self.src, self.emb_src, self.encode
        tgt, emb_tgt, decode = self.tgt, self.emb_tgt, self.decode
        with tf.variable_scope('emb_src_forcing'): w = position(tf.shape(src)[1]) + dropout(emb_src.embed(src))
        with tf.variable_scope('emb_tgt_forcing'): x = position(tf.shape(tgt)[1]) + dropout(emb_tgt.embed(tgt))
        with tf.variable_scope('encode_forcing'):
            for enc in encode: w = enc(w, dropout)
        with tf.variable_scope('decode_forcing'):
            with tf.variable_scope('mask'):
                mask = tf.linalg.LinearOperatorLowerTriangular(tf.ones((tf.shape(x)[1],)*2)).to_dense()
            for dec in decode: x = dec(x, x, w, dropout, mask)
        y = logit(x)
        p = tf.argmax(y, -1, output_type= tf.int32, name= 'pred')
        return Transformer(output= y, pred= p, **self)._eval()

    def _eval(self):
        gold, pred, output, smooth = self.gold, self.pred, self.output, self.smooth
        with tf.variable_scope('acc'):
            acc = tf.reduce_mean(tf.to_float(tf.equal(gold, pred)))
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= output, labels= smooth(gold)))
        with tf.variable_scope('prob'):
            prob = tf.nn.softmax(output, name= 'prob')
        return Transformer(prob= prob, loss= loss, acc= acc, **self)

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
