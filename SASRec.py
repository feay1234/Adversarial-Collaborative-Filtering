from time import localtime, strftime

import tensorflow as tf
import numpy as np
import math
from keras_preprocessing.sequence import pad_sequences

# Default params
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', default="Video")
# parser.add_argument('--train_dir', default="default")
# parser.add_argument('--batch_size', default=128, type=int)
# parser.add_argument('--lr', default=0.001, type=float)
# parser.add_argument('--maxlen', default=50, type=int)
# parser.add_argument('--hidden_units', default=50, type=int)
# parser.add_argument('--num_blocks', default=2, type=int)
# parser.add_argument('--num_epochs', default=201, type=int)
# parser.add_argument('--num_heads', default=1, type=int)
# parser.add_argument('--dropout_rate', default=0.5, type=float)
# parser.add_argument('--l2_emb', default=0.0, type=float)

# Self-Attentive Sequential Recommendation
# https://github.com/kang205/SASRec
from tensorflow.python.layers.convolutional import Conv1D
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops import init_ops

from Recommender import Recommender
import numpy as np
import logging
import os
from multiprocessing import Process, Queue


class SASRec(Recommender):
    def __init__(self, usernum, itemnum, hidden_units=50, maxlen=50, num_blocks=2,
                 num_heads=1,
                 dropout_rate=0.5,
                 l2_emb=0.0, lr=0.001, reuse=None, args=None, eps=0.5, time_stamp=None):

        self.uNum = usernum + 1
        self.iNum = itemnum + 1
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.eps = eps
        self.args = args
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, maxlen))
        pos = self.pos
        neg = self.neg
        self.mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)


        with tf.variable_scope("SASRec", reuse=reuse):
            # with tf.name_scope("SASRec"):
            # sequence embedding, item embedding table
            self.emb, self.item_emb_table = embedding(self.input_seq,
                                                      vocab_size=itemnum + 1,
                                                      num_units=hidden_units,
                                                      zero_pad=True,
                                                      scale=True,
                                                      l2_reg=l2_emb,
                                                      scope="input_embeddings",
                                                      with_t=True,
                                                      reuse=reuse
                                                      )

            # self.delta_emb = tf.Variable(tf.zeros(shape=[itemnum + 1, hidden_units]), name='delta_emb', dtype=tf.float32, trainable=False)
            if args.mode == 1:
                self.delta_emb = tf.Variable(tf.zeros(shape=[1, self.args.batch_size, maxlen, hidden_units]),
                                             name='delta_emb', dtype=tf.float32, trainable=False)

            if args.mode == 2:
                self.delta_emb = tf.Variable(tf.zeros(shape=[itemnum + 1, hidden_units]),
                                             name='delta_emb', dtype=tf.float32, trainable=False)
                self.delta_pos_emb = tf.Variable(tf.zeros(shape=[maxlen, hidden_units]),
                                               name='delta_pos_emb', dtype=tf.float32, trainable=False)

                self.delta_q_denses, self.delta_k_denses, self.delta_v_denses, self.delta_ff1, self.delta_ff2 = [], [], [], [], []
                for i in range(num_blocks):
                    self.delta_q_denses.append(Dense(hidden_units, kernel_initializer=init_ops.zeros_initializer(), trainable=False))
                    self.delta_k_denses.append(Dense(hidden_units, kernel_initializer=init_ops.zeros_initializer(), trainable=False))
                    self.delta_v_denses.append(Dense(hidden_units, kernel_initializer=init_ops.zeros_initializer(), trainable=False))

                    self.delta_ff1.append(Conv1D(filters=hidden_units, kernel_size=1, activation=tf.nn.relu, use_bias=True, trainable=False))
                    self.delta_ff2.append(Conv1D(filters=hidden_units, kernel_size=1, activation=None, use_bias=True, trainable=False))




            # Positional Encoding
            self.pos_emb, self.pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=maxlen,
                num_units=hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )


            embed_input = self.emb + self.pos_emb

            # Dropout
            embed_input = tf.layers.dropout(embed_input,
                                            rate=dropout_rate,
                                            training=tf.convert_to_tensor(self.is_training))
            embed_input *= self.mask

            # Build blocks

            self.q_denses, self.k_denses, self.v_denses, self.feedforwards1, self.feedforwards2 = [], [], [], [], []
            for i in range(num_blocks):
                # with tf.variable_scope("num_blocks_%d" % i):
                with tf.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.q_denses.append(Dense(hidden_units, activation=None))
                    self.k_denses.append(Dense(hidden_units, activation=None))
                    self.v_denses.append(Dense(hidden_units, activation=None))

                    embed_input = multihead_attention(self.q_denses[i], self.k_denses[i], self.v_denses[i],
                                                      queries=normalize(embed_input),
                                                      keys=embed_input,
                                                      num_units=hidden_units,
                                                      num_heads=num_heads,
                                                      dropout_rate=dropout_rate,
                                                      is_training=self.is_training,
                                                      causality=True,
                                                      scope="self_attention")

                    # Feed forward

                    self.feedforwards1.apend(
                        Conv1D(filters=hidden_units, kernel_size=1, activation=tf.nn.relu, use_bias=True))
                    self.feedforwards2.apend(
                        Conv1D(filters=hidden_units, kernel_size=1, activation=None, use_bias=True))

                    ff_output = self.feedforwards1[i].apply(embed_input)
                    ff_output = tf.layers.dropout(ff_output, rate=dropout_rate, training=tf.convert_to_tensor(True))
                    ff_output = self.feedforwards2[i].apply(ff_output)
                    ff_output = tf.layers.dropout(ff_output, rate=dropout_rate, training=tf.convert_to_tensor(True))

                    # Residual Connection
                    embed_input += ff_output

                    embed_input *= self.mask

            embed_input = normalize(embed_input)

        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * maxlen])
        pos_emb = tf.nn.embedding_lookup(self.item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(self.item_emb_table, neg)
        seq_emb = tf.reshape(embed_input, [tf.shape(self.input_seq)[0] * maxlen, hidden_units])

        self.test_item = tf.placeholder(tf.int32, shape=(itemnum + 1))
        test_item_emb = tf.nn.embedding_lookup(self.item_emb_table, self.test_item)
        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], maxlen, itemnum + 1])
        self.test_logits = self.test_logits[:, -1, :]

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * maxlen])
        self.loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        # if reuse is None:
        #     tf.summary.scalar('auc', self.auc)
        #     self.global_step = tf.Variable(0, name='global_step', trainable=False)
        #     self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta2=0.98)
        #     self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        # else:
        #     tf.summary.scalar('test_auc', self.auc)
        tf.summary.scalar('auc', self.auc)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta2=0.98)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.merged = tf.summary.merge_all()

        if args.adver:
            #
            # self.output_adv, embed_seq_pos = self._create_inference_adv(self.pos, maxlen, hidden_units)
            self.output_adv = self._create_inference_adv(self.pos, maxlen, hidden_units)
            # self.output_neg_adv, embed_seq_neg = self._create_inference_adv(self.neg, maxlen, hidden_units)
            self.output_neg_adv = self._create_inference_adv(self.neg, maxlen, hidden_units)
            # self.result_adv = tf.clip_by_value(self.output_adv - self.output_neg_adv, -80.0, 1e8)

            self.adv_loss = tf.reduce_sum(
                - tf.log(tf.sigmoid(self.output_adv) + 1e-24) * istarget -
                tf.log(1 - tf.sigmoid(self.output_neg_adv) + 1e-24) * istarget
            ) / tf.reduce_sum(istarget)
            reg_adv_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.adv_loss += sum(reg_adv_losses)

        self._create_adversarial()

        # # initialized the save op
        #
        # if args.adver:
        #     self.ckpt_save_path = "Pretrain/%s/ASASREC/embed_%d/%s/" % (args.dataset, args.embed_size, time_stamp)
        #     self.ckpt_restore_path = "Pretrain/%s/SASREC/embed_%d/%s/" % (args.dataset, args.embed_size, time_stamp)
        # else:
        #     self.ckpt_save_path = "Pretrain/%s/SASREC/embed_%d/%s/" % (args.dataset, args.embed_size, time_stamp)
        #     self.ckpt_restore_path = 0 if args.restore is None else "Pretrain/%s/SASREC/embed_%d/%s/" % (args.dataset, args.embed_size, args.restore)
        #
        # if not os.path.exists(self.ckpt_save_path):
        #     os.makedirs(self.ckpt_save_path)
        # if self.ckpt_restore_path and not os.path.exists(self.ckpt_restore_path):
        #     os.makedirs(self.ckpt_restore_path)
        #
        # self.saver_ckpt = tf.train.Saver()
        #
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # config.allow_soft_placement = True
        # self.sess = tf.Session(config=config)
        #
        # self.sess.run(tf.initialize_all_variables())
        #
        # # restore the weights when pretrained
        # if args.restore is not None:
        #     ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.ckpt_restore_path + 'checkpoint'))
        #     if ckpt and ckpt.model_checkpoint_path:
        #         self.saver_ckpt.restore(self.sess, ckpt.model_checkpoint_path)
        #
        # # initialize the weights
        # else:
        #     logging.info("Initialized from scratch")

    def _create_inference_adv(self, item_input, maxlen, hidden_units):
        if self.args.mode == 1:
            emb = tf.reduce_sum(tf.nn.embedding_lookup(self.item_emb_table, item_input), 1)
            seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * maxlen, hidden_units])
            emb_plus_delta = emb + tf.reduce_sum(tf.nn.embedding_lookup(self.delta_emb, item_input), 1)
            return tf.reduce_sum(emb_plus_delta * seq_emb, -1)

        elif self.args.mode == 2:
            emb = tf.reduce_sum(tf.nn.embedding_lookup(self.item_emb_table, item_input), 1)
            emb_plus_delta = emb + tf.reduce_sum(tf.nn.embedding_lookup(self.delta_emb, item_input), 1)

            pos_emb = tf.reduce_sum(tf.nn.embedding_lookup(self.pos_emb_table, tf.tile(tf.expand_dims(tf.range(tf.shape(item_input)[1]), 0), [tf.shape(item_input)[0], 1]),), 1)
            pos_emb_plus_delta = pos_emb + tf.reduce_sum(tf.nn.embedding_lookup(self.delta_pos_emb, tf.tile(tf.expand_dims(tf.range(tf.shape(item_input)[1]), 0), [tf.shape(item_input)[0], 1]),), 1)

            embed_input = emb_plus_delta + pos_emb_plus_delta

            embed_input = tf.layers.dropout(embed_input,
                                            rate=self.dropout_rate,
                                            training=tf.convert_to_tensor(self.is_training))
            embed_input *= self.mask

            for i in range(self.num_blocks):

                embed_input = multihead_attention(self.q_denses[i], self.k_denses[i], self.v_denses[i],
                                                  queries=normalize(embed_input),
                                                  keys=embed_input,
                                                  delta_q_dense=self.delta_q_denses[i],
                                                  delta_k_dense=self.delta_k_denses[i],
                                                  delta_v_dense=self.delta_v_denses[i],
                                                  num_units=hidden_units,
                                                  num_heads=self.num_heads,
                                                  dropout_rate=self.dropout_rate,
                                                  is_training=self.is_training,
                                                  causality=True,
                                                  scope="self_adv_attention")

                ff_output = self.feedforwards1[i].apply(embed_input)
                ff_output += tf.reduce_sum(self.delta_ff1[i].apply(embed_input), 1)
                ff_output = tf.layers.dropout(ff_output, rate=self.dropout_rate, training=tf.convert_to_tensor(True))
                ff_output = self.feedforwards2[i].apply(ff_output)
                ff_output += tf.reduce_sum(self.delta_ff2[i].apply(ff_output), 1)
                ff_output = tf.layers.dropout(ff_output, rate=self.dropout_rate, training=tf.convert_to_tensor(True))

                # Residual Connection
                embed_input += ff_output

                embed_input *= self.mask

            embed_input = normalize(embed_input)
            embed_input = tf.reshape(embed_input, [tf.shape(self.input_seq)[0] * maxlen, hidden_units])
            return tf.reduce_sume(emb_plus_delta * embed_input, -1)



    def _create_adversarial(self):
        if self.args.mode == 1:
            self.grad_emb = tf.gradients(self.loss, [self.seq])
            self.grad_emb_dense = tf.stop_gradient(self.grad_emb)
            # self.grad_emb_dense = tf.truncated_normal(shape=[self.iNum, self.hidden_units], mean=0.0, stddev=0.01)
            self.update_emb = self.delta_emb.assign(tf.nn.l2_normalize(self.grad_emb_dense, 1) * self.eps)

        elif self.args.mode == 2:
            grad_emb = tf.gradients(self.loss, [self.emb])
            grad_emb_dense = tf.stop_gradient(grad_emb)
            self.update_emb = self.delta_emb.assign(tf.nn.l2_normalize(grad_emb_dense, 1) * self.eps)

            grad_pos_emb = tf.gradients(self.loss, [self.pos_emb])
            grad_pos_emb_dense = tf.stop_gradient(grad_pos_emb)
            self.update_pos_emb = self.delta_pos_emb.assign(tf.nn.l2_normalize(grad_pos_emb_dense, 1) * self.eps)

            self.update_q_denses, self.update_k_denses, self.update_v_denses = [], [], []
            self.update_ff1, self.update_ff2 = [], []

            for i in range(self.num_blocks):
                # TODO gradient over multihead as a whole or individual dense
                grad_q = tf.gradients(self.loss, [self.q_denses[i]])
                grad_k = tf.gradients(self.loss, [self.k_denses[i]])
                grad_v = tf.gradients(self.loss, [self.v_denses[i]])

                grad_q_dense = tf.stop_gradient(grad_q)
                grad_k_dense = tf.stop_gradient(grad_k)
                grad_v_dense = tf.stop_gradient(grad_v)

                self.update_q_denses.append(self.delta_q_denses[i].assign(tf.nn.l2_normalize(grad_q_dense, 1) * self.eps))
                self.update_k_denses.append(self.delta_k_denses[i].assign(tf.nn.l2_normalize(grad_k_dense, 1) * self.eps))
                self.update_v_denses.append(self.delta_v_denses[i].assign(tf.nn.l2_normalize(grad_v_dense, 1) * self.eps))

                grad_ff1 = tf.gradients(self.loss, [self.feedforwards1[i]])
                grad_ff2 = tf.gradients(self.loss, [self.feedforwards2[i]])

                grad_ff1_dense = tf.stop_gradient(grad_ff1)
                grad_ff2_dense = tf.stop_gradient(grad_ff2)

                self.update_ff1.append(self.delta_ff1[i].assign(tf.nn.l2_normalize(grad_ff1_dense, 1) * self.eps))
                self.update_ff2.append(self.delta_ff2[i].assign(tf.nn.l2_normalize(grad_ff2_dense, 1) * self.eps))






    def init(self, trainSeq, batch_size, sess):
        self.trainSeq = trainSeq
        self.sampler = WarpSampler(self.trainSeq, self.uNum, self.iNum, batch_size=batch_size, maxlen=self.maxlen,
                                   n_workers=3)
        self.sess = sess

        # self.saver_ckpt.save(self.sess, self.ckpt_save_path + 'weights', global_step=0)

    def rank(self, users, items):
        users = users[0]
        seq = pad_sequences([self.trainSeq[users[0]]], self.maxlen)

        score = self.sess.run(self.test_logits,
                              {self.u: users[0], self.input_seq: seq, self.test_item: range(self.iNum),
                               self.is_training: False})[0]
        # res = []
        # for i in items:
        #
        #     res.append(score)
        # return np.array(res)
        return score[items.flatten()]

    def save(self, path):
        pass

    def load_pre_train(self, pre):
        pass

    def get_train_instances(self, train):
        return None, None

    def train(self, x_train, y_train, batch_size):
        losses = []
        num_batch = int(len(self.trainSeq) / batch_size)

        for step in list(range(num_batch)):
            u, seq, pos, neg = self.sampler.next_batch()

            auc, loss, _ = self.sess.run([self.auc, self.loss, self.train_op],
                                         {self.u: u, self.input_seq: seq, self.pos: pos, self.neg: neg,
                                          self.is_training: True})

            if self.args.adver:
                self.sess.run([self.update_emb], {self.u: u, self.input_seq: seq, self.pos: pos, self.neg: neg,
                                                  self.is_training: True})

            losses.append(loss)

        return np.mean(losses)

    def get_params(self):
        return "_m%d" % (self.args.mode)


# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''


def positional_encoding(dim, sentence_length, dtype=tf.float32):
    encoded_vec = np.array([pos / np.power(10000, 2 * i / dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # with tf.name_scope(scope):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              l2_reg=0.0,
              scope="embedding",
              with_t=False,
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # with tf.name_scope(scope):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       # initializer=tf.contrib.layers.xavier_initializer(),
                                       regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    if with_t:
        return outputs, lookup_table
    else:
        return outputs


def multihead_attention(q_dense, k_dense, v_dense,
                        queries,
                        keys,
                        delta_q_dense=None,
                        delta_k_dense=None,
                        delta_v_dense=None,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None,
                        with_qk=False):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # with tf.name_scope(scope):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        # Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)




        # Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
        Q = q_dense.apply(queries)
        K = k_dense.apply(keys)
        V = v_dense.apply(keys)

        if delta_q_dense != None:
            Q += tf.reduce_sum(delta_q_dense.apply(queries), 1)
            K += tf.reduce_sum(delta_k_dense.apply(keys), 1)
            V += tf.reduce_sum(delta_v_dense.apply(keys), 1)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        # outputs = normalize(outputs) # (N, T_q, C)

    if with_qk:
        return Q, K
    else:
        return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                dropout_rate=0.2,
                is_training=True,
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # with tf.name_scope(scope):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Residual connection
        outputs += inputs

        # Normalize
        # outputs = normalize(outputs)

    return outputs


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(list(zip(*one_batch)))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=0):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
