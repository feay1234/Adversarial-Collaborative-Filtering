import argparse

# -*- coding: utf-8 -*-
from Recommender import Recommender

"""
Created on Feb 26, 2017
@author: Weiping Song
"""
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
import numpy as np

from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences


def parseArgs_GRU4Rec():
    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--layer', default=1, type=int)
    parser.add_argument('--size', default=100, type=int)
    parser.add_argument('--epoch', default=3, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--test', default=2, type=int)
    parser.add_argument('--hidden_act', default='tanh', type=str)
    parser.add_argument('--final_act', default='softmax', type=str)
    parser.add_argument('--loss', default='cross-entropy', type=str)
    parser.add_argument('--dropout', default='0.5', type=float)

    return parser.parse_args()


class Args():
    is_training = True
    layers = 1
    rnn_size = 100
    n_epochs = 3
    batch_size = 512
    dropout_p_hidden = 1
    learning_rate = 0.001
    decay = 0.96
    decay_steps = 1e4
    sigma = 0
    init_as_normal = False
    reset_after_session = True
    time_key = 'Time'
    grad_cap = 0
    test_model = 2
    checkpoint_dir = './checkpoint'
    loss = 'bpr'
    final_act = 'linear'
    hidden_act = 'tanh'
    n_items = -1


class GRU4Rec(Recommender):


    def __init__(self, uNum, iNum, dim, batch_size):
        self.uNum = uNum
        self.iNum = iNum
        self.dim = dim
        args = Args()
        self.is_training = args.is_training

        self.layers = args.layers
        self.rnn_size = dim
        self.n_epochs = args.n_epochs
        self.batch_size = batch_size
        self.dropout_p_hidden = args.dropout_p_hidden
        self.learning_rate = args.learning_rate
        self.decay = args.decay
        self.decay_steps = args.decay_steps
        self.sigma = args.sigma
        self.init_as_normal = args.init_as_normal
        self.reset_after_session = args.reset_after_session
        self.grad_cap = args.grad_cap
        self.n_items = iNum

        if args.hidden_act == 'tanh':
            self.hidden_act = self.tanh
        elif args.hidden_act == 'relu':
            self.hidden_act = self.relu
        else:
            raise NotImplementedError

        if args.loss == 'cross-entropy':
            if args.final_act == 'tanh':
                self.final_activation = self.softmaxth
            else:
                self.final_activation = self.softmax
            self.loss_function = self.cross_entropy
        elif args.loss == 'bpr':
            if args.final_act == 'linear':
                self.final_activation = self.linear
            elif args.final_act == 'relu':
                self.final_activation = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.bpr
        elif args.loss == 'top1':
            if args.final_act == 'linear':
                self.final_activation = self.linear
            elif args.final_act == 'relu':
                self.final_activatin = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.top1
        else:
            raise NotImplementedError

        self.build_model()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        self.predict_state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in range(self.layers)]

    ########################ACTIVATION FUNCTIONS#########################
    def linear(self, X):
        return X

    def tanh(self, X):
        return tf.nn.tanh(X)

    def softmax(self, X):
        return tf.nn.softmax(X)

    def softmaxth(self, X):
        return tf.nn.softmax(tf.tanh(X))

    def relu(self, X):
        return tf.nn.relu(X)

    def sigmoid(self, X):
        return tf.nn.sigmoid(X)

    ############################LOSS FUNCTIONS######################
    def cross_entropy(self, yhat):
        return tf.reduce_mean(-tf.log(tf.diag_part(yhat) + 1e-24))

    def bpr(self, yhat):
        yhatT = tf.transpose(yhat)
        return tf.reduce_mean(-tf.log(tf.nn.sigmoid(tf.diag_part(yhat) - yhatT)))

    def top1(self, yhat):
        yhatT = tf.transpose(yhat)
        term1 = tf.reduce_mean(tf.nn.sigmoid(-tf.diag_part(yhat) + yhatT) + tf.nn.sigmoid(yhatT ** 2), axis=0)
        term2 = tf.nn.sigmoid(tf.diag_part(yhat) ** 2) / self.batch_size
        return tf.reduce_mean(term1 - term2)

    def build_model(self):

        self.X = tf.placeholder(tf.int32, [self.batch_size], name='input')
        self.Y = tf.placeholder(tf.int32, [self.batch_size], name='output')
        self.state = [tf.placeholder(tf.float32, [self.batch_size, self.rnn_size], name='rnn_state') for _ in
                      range(self.layers)]
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope('gru_layer'):
            sigma = self.sigma if self.sigma != 0 else np.sqrt(6.0 / (self.n_items + self.rnn_size))
            if self.init_as_normal:
                initializer = tf.random_normal_initializer(mean=0, stddev=sigma)
            else:
                initializer = tf.random_uniform_initializer(minval=-sigma, maxval=sigma)
            embedding = tf.get_variable('embedding', [self.n_items, self.rnn_size], initializer=initializer)
            softmax_W = tf.get_variable('softmax_w', [self.n_items, self.rnn_size], initializer=initializer)
            softmax_b = tf.get_variable('softmax_b', [self.n_items], initializer=tf.constant_initializer(0.0))

            cell = rnn_cell.GRUCell(self.rnn_size, activation=self.hidden_act)
            drop_cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_p_hidden)
            stacked_cell = rnn_cell.MultiRNNCell([drop_cell] * self.layers)

            inputs = tf.nn.embedding_lookup(embedding, self.X)
            output, state = stacked_cell(inputs, tuple(self.state))
            self.final_state = state

        '''
        Use other examples of the minibatch as negative samples.
        '''
        sampled_W = tf.nn.embedding_lookup(softmax_W, self.Y)
        sampled_b = tf.nn.embedding_lookup(softmax_b, self.Y)
        logits = tf.matmul(output, sampled_W, transpose_b=True) + sampled_b
        self.yhat = self.final_activation(logits)
        self.cost = self.loss_function(self.yhat)

        logits = tf.matmul(output, softmax_W, transpose_b=True) + softmax_b
        self.pred = self.final_activation(logits)

        self.lr = tf.maximum(1e-5, tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                              self.decay, staircase=True))

        '''
        Try different optimizers.
        '''
        # optimizer = tf.train.AdagradOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)
        # optimizer = tf.train.AdadeltaOptimizer(self.lr)
        # optimizer = tf.train.RMSPropOptimizer(self.lr)

        tvars = tf.trainable_variables()
        gvs = optimizer.compute_gradients(self.cost, tvars)
        if self.grad_cap > 0:
            capped_gvs = [(tf.clip_by_norm(grad, self.grad_cap), var) for grad, var in gvs]
        else:
            capped_gvs = gvs
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

    def init(self, trainSeq, df):
        self.trainSeq = trainSeq
        self.df = df

        self.offset_sessions = np.zeros(self.uNum + 1, dtype=np.int32)
        for u in self.trainSeq:
            self.offset_sessions[u] = len(self.trainSeq[u])

    def get_train_instances(self, train):
        return None, None


    def rank(self, users, items):
        seq = pad_sequences([self.trainSeq[users[0]]], self.batch_size)[0]
        fetches = [self.pred, self.final_state]
        feed_dict = {self.X: seq}
        for i in range(self.layers):
            feed_dict[self.state[i]] = self.predict_state[i]
        preds, self.predict_state = self.sess.run(fetches, feed_dict)
        return preds[-1][items]
    
    def save(self, path):
        super().save(path)

    def load_pre_train(self, pre):
        super().load_pre_train(pre)


    def train(self, x_train, y_train, batch_size):
        losses = []
        state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in range(self.layers)]
        session_idx_arr = np.arange(len(self.offset_sessions) - 1)
        iters = np.arange(self.batch_size)
        maxiter = iters.max()
        start = self.offset_sessions[session_idx_arr[iters]]
        end = self.offset_sessions[session_idx_arr[iters] + 1]
        finished = False
        while not finished:
            minlen = (end - start).min()
            out_idx = self.df.iid.values[start]
            for i in range(minlen - 1):
                in_idx = out_idx
                out_idx = self.df.iid.values[start + i + 1]
                # prepare inputs, targeted outputs and hidden states
                fetches = [self.cost, self.final_state, self.global_step, self.lr, self.train_op]
                feed_dict = {self.X: in_idx, self.Y: out_idx}
                for j in range(self.layers):
                    feed_dict[self.state[j]] = state[j]

                loss, state, step, lr, _ = self.sess.run(fetches, feed_dict)
                losses.append(loss)
            start = start + minlen - 1
            mask = np.arange(len(iters))[(end - start) <= 1]
            for idx in mask:
                maxiter += 1
                if maxiter >= len(self.offset_sessions) - 1:
                    finished = True
                    break
                iters[idx] = maxiter
                start[idx] = self.offset_sessions[session_idx_arr[maxiter]]
                end[idx] = self.offset_sessions[session_idx_arr[maxiter] + 1]
            if len(mask) and self.reset_after_session:
                for i in range(self.layers):
                    state[i][mask] = 0

        return "%.4f" % np.mean(losses)



# b = GRU4Rec(10, 10, 5, 3, 512)
# s = {i: [1, 2, 3, 4] for i in range(512)}
# b.init(s)
# print(b.rank([1, 1, 1], [5, 6]).shape)
# x, y = b.get_train_instances(None)
# print(b.train(x,y, 32))
# b.train(None, None, 512)
