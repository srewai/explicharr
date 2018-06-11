from hyperparams import Hyperparams as hp
import tensorflow as tf


def smooth(x, alpha= 0.1):
    return ((1 - alpha) * x) + (alpha / x.shape[-1])


def embedding(inputs, vocab_size, num_units, scope):
    with tf.variable_scope(scope):
        embed = tf.get_variable(
            name= 'embed'
            , shape= (vocab_size, num_units)
            , dtype= tf.float32
            , initializer= init_orth)
        return tf.gather(params= embed, indices= indices)


def sinusoidal(num_units, max_len):
    # todo dynamic version
    assert not num_units % 2
    freq = 1e-8 ** (np.arange(int(num_units / 2)) / num_units)
    freq.shape = -1, 1
    time = np.arange(max_len)
    time.shape = 1, -1
    angle = freq @ time
    return np.concatenate((np.sin(angle), np.cos(angle)), axis= -1).reshape(-1, max_len).T


def norm_res_drop(x, res, dropout, training):
    return tf.contrib.layers.layer_norm(
        res + tf.layers.dropout(
            outputs, rate= dropout, training= training))


def feedforward(inputs, num_units= 2048, scope= "feedforward"):
    # inputs : n, t, d
    with tf.variable_scope(scope):
        # inner layer : n, t, num_units
        outputs = tf.layers.dense(
            name= "inner"
            , inputs= inputs
            , num_units= num_units
            , kernel_initializer= kinit
            , bias_initializer= binit
            , activation= tf.nn.relu)
        # readout layer : n, t, d
        outputs = tf.layers.dense(
            name= "outer"
            , inputs= outputs
            , num_units= hp.hidden_units
            , kernel_initializer= kinit
            , bias_initializer= binit
            , activation= None)
    return norm_res_drop(outputs, inputs, dropout, training)


def multihead_attention(
        scope= "multihead_attention"
        , query                 # [?, t, q]
        , value                 # [?, s, k]
        , num_units
        , num_heads= 8
        , dropout= 0.1
        , training= True
        , causal= False):
    assert not num_units % num_heads
    with tf.variable_scope(scope):
        q = tf.layers.dense(query, num_units, activation= tf.nn.relu) # (?, t, d)
        k = tf.layers.dense(value, num_units, activation= tf.nn.relu) # (?, s, d)
        v = tf.layers.dense(value, num_units, activation= tf.nn.relu) # (?, s, d)
        # multihead
        q = tf.stack(tf.split(q, num_heads, axis= -1)) # (h, ?, t, d/h)
        k = tf.stack(tf.split(k, num_heads, axis= -1)) # (h, ?, s, d/h)
        v = tf.stack(tf.split(v, num_heads, axis= -1)) # (h, ?, s, d/h)
        # dot product
        q = tf.matmul(q, k, transpose_b= True) # (h, ?, t, s)
        # scale
        q *= (num_units / num_heads) ** -0.5
        if causal: q += tf.log(tf.linalg.LinearOperatorLowerTriangular(tf.ones_like(q[0])).to_dense())
        # activation
        a = tf.nn.softmax(q) # (h, ?, t, s)
        # weighted sum
        outputs = tf.matmul(a, v) # (h, ?, t, d/h)
        # restore shape
        outputs = tf.concat(tf.unstack(outputs), axis= -1) # (?, t, d)
    return norm_res_drop(outputs, inputs, dropout, training)


class Model:
    def __init__(self, dl, training):
        dim_src = dl.dim_src
        dim_tgt = dl.dim_tgt

        if training:
            self.x, self.y = dl.batches()
        else: # inference
            self.x = tf.placeholder(tf.int32, shape=(None, hp.max_len))
            self.y = tf.placeholder(tf.int32, shape=(None, hp.max_len))

        self.position = tf.constant(
            name= "position"
            , value= sinusoidal(hp.hidden_units, hp.max_len)
            , dtype= tf.foat32)

        # Encoder
        with tf.variable_scope("encoder"):
            ## Embedding
            self.enc = embedding(
                scope= "enc_embed"
                , inputs= self.x
                , vocab_size=dim_src
                , num_units=hp.hidden_units)

            ## Positional Encoding
            self.enc += self.position

            ## Dropout
            self.enc = tf.layers.dropout(self.enc,
                                        rate=hp.dropout_rate,
                                        training= training)

            ## Blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.enc = multihead_attention(queries=self.enc,
                                                    keys=self.enc,
                                                    num_units=hp.hidden_units,
                                                    num_heads=hp.num_heads,
                                                    dropout_rate=hp.dropout_rate,
                                                    training=training,
                                                    causality=False)

                    ### Feed Forward
                    self.enc = feedforward(self.enc, num_units= 4 * hp.hidden_units)


        with tf.variable_scope("target"):
            # define decoder inputs
            s = dl._idx2tgt.index("<S>")
            self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1])*s, self.y[:, :-1]), -1)

        # Decoder
        with tf.variable_scope("decoder"):
            ## Embedding
            self.dec = embedding(
                scope= "dec_embed"
                , inputs= self.decoder_inputs
                , vocab_size= dim_tgt
                , num_units= hp.hidden_units)

            ## Positional Encoding todo fixme
            self.dec += self.position

            ## Dropout
            self.dec = tf.layers.dropout(self.dec, rate= hp.dropout_rate, training= training)

            ## Blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ## Multihead Attention ( self-attention)
                    self.dec = multihead_attention(queries=self.dec,
                                                    keys=self.dec,
                                                    num_units=hp.hidden_units,
                                                    num_heads=hp.num_heads,
                                                    dropout_rate=hp.dropout_rate,
                                                    training=training,
                                                    causality=True,
                                                    scope="self_attention")

                    ## Multihead Attention ( vanilla attention)
                    self.dec = multihead_attention(queries=self.dec,
                                                    keys=self.enc,
                                                    num_units=hp.hidden_units,
                                                    num_heads=hp.num_heads,
                                                    dropout_rate=hp.dropout_rate,
                                                    training=training,
                                                    causality=False,
                                                    scope="vanilla_attention")

                    ## Feed Forward
                    self.dec = feedforward(self.dec, num_units= 4 * hp.hidden_units)

        self.logits = tf.layers.dense(self.dec, dim_tgt, name= "logit")

        with tf.variable_scope("accuracy"):
            self.preds = tf.to_int32(tf.argmax(self.logits, axis= -1))
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))*self.istarget)/ (tf.reduce_sum(self.istarget))

        if training:
            # Loss
            with tf.variable_scope("loss"):
                # todo smooth once and gather
                alpha = 0.1
                self.y_smoothed = ((1 - alpha) * tf.one_hot(self.y, depth= dim_tgt)) + (alpha / dim_tgt)
                self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_smoothed)
                self.mean_loss = tf.reduce_mean(self.loss)

            # Training Scheme
            with tf.variable_scope("update"):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss)
