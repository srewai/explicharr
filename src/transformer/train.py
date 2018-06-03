from data_load import get_batch_data, load_vocab
from hyperparams import Hyperparams as hp
from modules import *
from tqdm import tqdm
import os
import tensorflow as tf


class Graph():
    def __init__(self, dim_src, dim_tgt, is_training=True):
        if is_training:
            self.x, self.y, self.num_batch = get_batch_data() # (N, T)
        else: # inference
            self.x = tf.placeholder(tf.int32, shape=(None, hp.max_len))
            self.y = tf.placeholder(tf.int32, shape=(None, hp.max_len))

        # Encoder
        with tf.variable_scope("encoder"):
            ## Embedding
            self.enc = embedding(self.x,
                                  vocab_size=dim_src,
                                  num_units=hp.hidden_units,
                                  scale=True,
                                  scope="enc_embed")

            ## Positional Encoding
            if hp.sinusoid:
                self.enc += positional_encoding(self.x,
                                  num_units=hp.hidden_units,
                                  zero_pad=False,
                                  scale=False,
                                  scope="enc_pe")
            else:
                self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                                  vocab_size=hp.max_len,
                                  num_units=hp.hidden_units,
                                  zero_pad=False,
                                  scale=False,
                                  scope="enc_pe")


            ## Dropout
            self.enc = tf.layers.dropout(self.enc,
                                        rate=hp.dropout_rate,
                                        training=tf.convert_to_tensor(is_training))

            ## Blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.enc = multihead_attention(queries=self.enc,
                                                    keys=self.enc,
                                                    num_units=hp.hidden_units,
                                                    num_heads=hp.num_heads,
                                                    dropout_rate=hp.dropout_rate,
                                                    is_training=is_training,
                                                    causality=False)

                    ### Feed Forward
                    self.enc = feedforward(self.enc, num_units=[4*hp.hidden_units, hp.hidden_units])


        with tf.variable_scope("target"):
            # define decoder inputs
            self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1])*2, self.y[:, :-1]), -1) # 2:<S>

        # Decoder
        with tf.variable_scope("decoder"):
            ## Embedding
            self.dec = embedding(self.decoder_inputs,
                                  vocab_size=dim_tgt,
                                  num_units=hp.hidden_units,
                                  scale=True,
                                  scope="dec_embed")

            ## Positional Encoding
            if hp.sinusoid:
                self.dec += positional_encoding(self.decoder_inputs,
                                  vocab_size=hp.max_len,
                                  num_units=hp.hidden_units,
                                  zero_pad=False,
                                  scale=False,
                                  scope="dec_pe")
            else:
                self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
                                  vocab_size=hp.max_len,
                                  num_units=hp.hidden_units,
                                  zero_pad=False,
                                  scale=False,
                                  scope="dec_pe")

            ## Dropout
            self.dec = tf.layers.dropout(self.dec,
                                        rate=hp.dropout_rate,
                                        training=tf.convert_to_tensor(is_training))

            ## Blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ## Multihead Attention ( self-attention)
                    self.dec = multihead_attention(queries=self.dec,
                                                    keys=self.dec,
                                                    num_units=hp.hidden_units,
                                                    num_heads=hp.num_heads,
                                                    dropout_rate=hp.dropout_rate,
                                                    is_training=is_training,
                                                    causality=True,
                                                    scope="self_attention")

                    ## Multihead Attention ( vanilla attention)
                    self.dec = multihead_attention(queries=self.dec,
                                                    keys=self.enc,
                                                    num_units=hp.hidden_units,
                                                    num_heads=hp.num_heads,
                                                    dropout_rate=hp.dropout_rate,
                                                    is_training=is_training,
                                                    causality=False,
                                                    scope="vanilla_attention")

                    ## Feed Forward
                    self.dec = feedforward(self.dec, num_units=[4*hp.hidden_units, hp.hidden_units])

        # Final linear projection
        with tf.variable_scope("linear_projection"):
            self.logits = tf.layers.dense(self.dec, dim_tgt)
            self.preds = tf.to_int32(tf.argmax(self.logits, axis= -1))
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))

        with tf.variable_scope("accuracy"):
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))*self.istarget)/ (tf.reduce_sum(self.istarget))

        if is_training:
            # Loss
            with tf.variable_scope("loss"):
                self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=dim_tgt))
                self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_smoothed)
                self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))

            # Training Scheme
            with tf.variable_scope("update"):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss)


if __name__ == '__main__':

    src2idx, idx2src = load_vocab(os.path.join(hp.logdir, "vocab.src"))
    tgt2idx, idx2tgt = load_vocab(os.path.join(hp.logdir, "vocab.tgt"))

    g = Graph(len(idx2src), len(idx2tgt), "train")

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # wtr = tf.summary.FileWriter(hp.logdir, tf.get_default_graph())
    wtr = tf.summary.FileWriter(hp.logdir)
    svr = tf.train.Saver(max_to_keep= None)

    summ_up = tf.summary.scalar('loss', g.mean_loss), g.train_op

    print("training for {} batches per epoch".format(g.num_batch))
    step = 0
    for epoch in range(hp.num_epochs):
        for _ in tqdm(range(g.num_batch), ncols= 70):
            summ, _ = sess.run(summ_up)
            step += 1
            if not (step % 32): wtr.add_summary(summ, step)
        svr.save(sess, save_path= "{}/m{}".format(hp.logdir, epoch), global_step= step)

    wtr.close()
    sess.close()
