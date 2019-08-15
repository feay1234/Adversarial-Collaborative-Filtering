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
        self.eps = eps
        self.args = args
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, maxlen))
        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

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
            self.delta_emb = tf.Variable(tf.zeros(shape=[1, self.args.batch_size, maxlen, hidden_units]), name='delta_emb', dtype=tf.float32, trainable=False)
            self.h = tf.constant(1.0, tf.float32, [hidden_units, 1], name="h")

            # Positional Encoding
            self.t, pos_emb_table = embedding(
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

            self.seq = self.emb + self.t

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            # Build blocks

            for i in range(num_blocks):
                # with tf.variable_scope("num_blocks_%d" % i):
                with tf.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=hidden_units,
                                                   num_heads=num_heads,
                                                   dropout_rate=dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[hidden_units, hidden_units],
                                           dropout_rate=dropout_rate, is_training=self.is_training)
                    self.seq *= mask

            self.seq = normalize(self.seq)

        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * maxlen])
        pos_emb = tf.nn.embedding_lookup(self.item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(self.item_emb_table, neg)
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * maxlen, hidden_units])

        self.test_item = tf.placeholder(tf.int32, shape=(itemnum+1))
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
        # emb = tf.nn.embedding_lookup(self.item_emb_table, item_input)
        emb = tf.reduce_sum(tf.nn.embedding_lookup(self.item_emb_table, item_input), 1)
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * maxlen, hidden_units])

        emb_plus_delta = emb + tf.reduce_sum(tf.nn.embedding_lookup(self.delta_emb, item_input),1)

        return tf.reduce_sum(emb_plus_delta * seq_emb, -1)

    def _create_adversarial(self):
        self.grad_emb = tf.gradients(self.loss, [self.seq])
        self.grad_emb_dense = tf.stop_gradient(self.grad_emb)
        # self.grad_emb_dense = tf.truncated_normal(shape=[self.iNum, self.hidden_units], mean=0.0, stddev=0.01)
        self.update_emb = self.delta_emb.assign(tf.nn.l2_normalize(self.grad_emb_dense, 1) * self.eps)


    def init(self, trainSeq, batch_size, sess):
        self.trainSeq = trainSeq
        self.sampler = WarpSampler(self.trainSeq, self.uNum, self.iNum, batch_size=batch_size, maxlen=self.maxlen, n_workers=3)
        self.sess = sess

        # self.saver_ckpt.save(self.sess, self.ckpt_save_path + 'weights', global_step=0)

    def rank(self, users, items):
        users = users[0]
        seq = pad_sequences([self.trainSeq[users[0]]], self.maxlen)

        score = self.sess.run(self.test_logits,
                              {self.u: users[0], self.input_seq: seq, self.test_item: range(self.iNum), self.is_training: False})[0]
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
            break

        return np.mean(losses)

    def get_params(self):
        return "_ml%d" % (self.maxlen)


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


def multihead_attention(queries,
                        keys,
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
        Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

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
