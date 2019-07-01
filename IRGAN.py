import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
from keras.engine.saving import load_model
import math


class IRGAN():
    def __init__(self, uNum, iNum, dim, batch_size):
        self.uNum = uNum
        self.iNum = iNum
        self.dim = dim

        INIT_DELTA = 0.05

        self.generator = GEN(iNum, uNum, dim, lamda=0.0 / batch_size, param=None, initdelta=INIT_DELTA,
                             learning_rate=0.05)

        self.discriminator = DIS(iNum, uNum, dim, lamda=0.0 / batch_size, param=None, initdelta=INIT_DELTA,
                                 learning_rate=0.05)

        # self.discriminator = DIS(iNum, uNum, dim, lamda=0.1, param=None, initdelta=0.05, learning_rate=0.05)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def load_pre_train(self, path):
        pretrainModel = load_model(path)

        assign_P = self.generator.user_embeddings.assign(pretrainModel.get_layer("uEmb").get_weights()[0])
        assign_Q = self.generator.item_embeddings.assign(pretrainModel.get_layer("iEmb").get_weights()[0])
        self.sess.run([assign_P, assign_Q])

    def rank(self, users, items):
        user_batch_rating = self.sess.run(self.generator.all_rating, {self.generator.u: [users[0]]})
        # user_batch_rating = self.sess.run(self.discriminator.all_rating, {self.discriminator.u: [users[0]]})
        return user_batch_rating[0][items]

    def init(self, train):

        self.trainData = train
        self.user_pos_item = {}
        for (u, i) in train.keys():
            if u in self.user_pos_item:
                self.user_pos_item[u].append(i)
            else:
                self.user_pos_item[u] = [i]

    def get_train_instances(self, train):
        # We do not need need function for IRGAN
        return None, None

    def save(self, path):
        pass

    def train2(self, x_train, y_train, batch_size):

        x_train_d, y_train_d = self.generate_dns()
        for i in range(math.ceil(len(y_train_d) / batch_size)):
            _u = x_train_d[0][i * batch_size:(i * batch_size) + batch_size]
            _p = x_train_d[1][i * batch_size:(i * batch_size) + batch_size]
            _n = x_train_d[2][i * batch_size:(i * batch_size) + batch_size]
            _ = self.sess.run(self.discriminator.d_updates,
                           feed_dict={self.discriminator.u: _u, self.discriminator.pos: _p,
                                      self.discriminator.neg: _n})
        return 0

    def train(self, x_train, y_train, batch_size):
        # TODO support d_ and g_steps

        x_train_d, y_train_d = self.generate_for_d()
        for i in range(math.ceil(len(y_train_d) / batch_size)):
            _u = x_train_d[0][i * batch_size:(i * batch_size) + batch_size]
            _i = x_train_d[1][i * batch_size:(i * batch_size) + batch_size]
            _y = y_train_d[i * batch_size:(i * batch_size) + batch_size]
            _ = self.sess.run(self.discriminator.d_updates,
                              feed_dict={self.discriminator.u: _u, self.discriminator.i: _i,
                                         self.discriminator.label: _y})

        losses = []
        for u in self.user_pos_item:
            sample_lambda = 0.2
            pos = self.user_pos_item[u]

            rating = self.sess.run(self.generator.all_logits, {self.generator.u: u})
            exp_rating = np.exp(rating)
            prob = exp_rating / np.sum(exp_rating)  # prob is generator distribution p_\theta

            pn = (1 - sample_lambda) * prob
            pn[pos] += sample_lambda * 1.0 / len(pos)
            # Now, pn is the Pn in importance sampling, prob is generator distribution p_\theta

            sample = np.random.choice(np.arange(self.iNum), 2 * len(pos), p=pn)
            ###########################################################################
            # Get reward and adapt it with importance sampling
            ###########################################################################
            reward = self.sess.run(self.discriminator.reward, {self.discriminator.u: u, self.discriminator.i: sample})
            reward = reward * prob[sample] / pn[sample]
            ###########################################################################
            # Update G
            ###########################################################################
            loss, _ = self.sess.run([self.generator.gan_loss, self.generator.gan_updates],
                                    {self.generator.u: u, self.generator.i: sample, self.generator.reward: reward})
            losses.append(loss)
        return np.mean(losses)

    def generate_for_d(self):
        _u, _i, _y = [], [], []
        for u in self.user_pos_item:
            pos = self.user_pos_item[u]

            rating = self.sess.run(self.generator.all_rating, {self.generator.u: [u]})
            rating = np.array(rating[0]) / 0.2  # Temperature

            exp_rating = np.exp(rating)

            if np.sum(exp_rating) == float("inf"):
                # uniform distribution over all items
                neg = np.random.choice(np.arange(self.iNum), size=len(pos))
            else:
                prob = exp_rating / np.sum(exp_rating)
                neg = np.random.choice(np.arange(self.iNum), size=len(pos), p=prob)

            for i in range(len(pos)):
                _u.extend([u, u])
                _i.append(pos[i])
                _i.append(neg[i])
                _y.extend([1, 0])

        return [_u, _i], _y

    def generate_dns(self):
        user_input, pos_item_input, neg_item_input, labels = [], [], [], []
        for (u, i) in self.trainData.keys():
            # positive instance
            user_input.append(u)
            pos_item_input.append(i)
            # negative instances
            j = np.random.randint(self.iNum)
            while (u, j) in self.trainData:
                j = np.random.randint(self.iNum)
            neg_item_input.append(j)
            labels.append(1)

        idx = np.arange(len(user_input))
        np.random.shuffle(idx)

        return [np.array(user_input)[idx], np.array(pos_item_input)[idx], np.array(neg_item_input)[idx]], np.array(labels)


class GEN():
    def __init__(self, itemNum, userNum, emb_dim, lamda, param=None, initdelta=0.05, learning_rate=0.05):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.param = param
        self.initdelta = initdelta
        self.learning_rate = learning_rate
        self.g_params = []

        with tf.variable_scope('generator'):
            if self.param == None:
                self.user_embeddings = tf.Variable(
                    tf.random_uniform([self.userNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_embeddings = tf.Variable(
                    tf.random_uniform([self.itemNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
            else:
                self.user_embeddings = tf.Variable(self.param[0])
                self.item_embeddings = tf.Variable(self.param[1])

            self.g_params = [self.user_embeddings, self.item_embeddings]

        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)
        self.reward = tf.placeholder(tf.float32)

        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i)

        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1)
        self.i_prob = tf.gather(
            tf.reshape(tf.nn.softmax(tf.reshape(self.all_logits, [1, -1])), [-1]),
            self.i)

        self.gan_loss = -tf.reduce_mean(tf.log(self.i_prob) * self.reward) + self.lamda * (
            tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding))

        g_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.gan_updates = g_opt.minimize(self.gan_loss, var_list=self.g_params)

        # not efficient, but for policy gradient
        self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
                                    transpose_b=True)

        # used at test time
        self.embedding_p = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.embedding_q = tf.nn.embedding_lookup(self.item_embeddings, self.i)  # (b, embedding_size)
        self.predictor = tf.reduce_sum(tf.multiply(self.embedding_p, self.embedding_q), 1)

    def save_model(self, sess, filename):
        param = sess.run(self.g_params)
        pickle.dump(param, open(filename, 'w'))


class DIS():
    def __init__(self, itemNum, userNum, emb_dim, lamda, param=None, initdelta=0.05, learning_rate=0.05):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.param = param
        self.initdelta = initdelta
        self.learning_rate = learning_rate
        self.d_params = []

        with tf.variable_scope('discriminator'):
            if self.param == None:
                self.user_embeddings = tf.Variable(
                    tf.random_uniform([self.userNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_embeddings = tf.Variable(
                    tf.random_uniform([self.itemNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
            else:
                self.user_embeddings = tf.Variable(self.param[0])
                self.item_embeddings = tf.Variable(self.param[1])

        self.d_params = [self.user_embeddings, self.item_embeddings]

        # placeholder definition
        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)
        self.label = tf.placeholder(tf.float32)

        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i)

        self.pre_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), 1)
        self.pre_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label,
                                                                logits=self.pre_logits) + self.lamda * (
            tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding)
        )

        d_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.d_updates = d_opt.minimize(self.pre_loss, var_list=self.d_params)

        self.reward_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), 1)
        self.reward = 2 * (tf.sigmoid(self.reward_logits) - 0.5)

        # for test stage, self.u: [batch_size]
        self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False, transpose_b=True)

        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1)
        self.NLL = -tf.reduce_mean(tf.log(
            tf.gather(tf.reshape(tf.nn.softmax(tf.reshape(self.all_logits, [1, -1])), [-1]), self.i))
        )
        # for dns sample
        self.dns_rating = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1)

    def save_model(self, sess, filename):
        param = sess.run(self.d_params)
        pickle.dump(param, open(filename, 'w'))

class DIS2():
    def __init__(self, itemNum, userNum, emb_dim, lamda, param=None, initdelta=0.05, learning_rate=0.05):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.param = param
        self.initdelta = initdelta
        self.learning_rate = learning_rate
        self.d_params = []

        with tf.variable_scope('discriminator'):
            if self.param is None:
                self.user_embeddings = tf.Variable(
                    tf.random_uniform([self.userNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_embeddings = tf.Variable(
                    tf.random_uniform([self.itemNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
            else:
                self.user_embeddings = tf.Variable(self.param[0])
                self.item_embeddings = tf.Variable(self.param[1])

        self.d_params = [self.user_embeddings, self.item_embeddings]

        # placeholder definition
        self.u = tf.placeholder(tf.int32)
        self.pos = tf.placeholder(tf.int32)
        self.neg = tf.placeholder(tf.int32)

        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.pos_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.pos)
        self.neg_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.neg)

        # self.pre_logits = tf.sigmoid(
        #     tf.reduce_sum(tf.multiply(self.u_embedding, self.pos_embedding - self.neg_embedding),1))
        # self.pre_loss = -tf.reduce_mean(tf.log(self.pre_logits))
        #
        # d_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        # self.d_updates = d_opt.minimize(self.pre_loss, var_list=self.d_params)

        self.output = tf.multiply(self.u_embedding, self.pos_embedding)
        self.output_neg = tf.multiply(self.u_embedding, self.neg_embedding)

        # self.result = tf.clip_by_value(self.output - self.output_neg, -80.0, 1e8)
        self.result = self.output - self.output_neg
        # self.loss = tf.reduce_sum(tf.log(1 + tf.exp(-self.result))) # this is numerically unstable
        self.opt_loss = tf.reduce_sum(tf.nn.softplus(-self.result))

        self.d_updates = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.opt_loss)

        # loss to be omptimized
        # self.opt_loss = self.loss + self.reg * tf.reduce_mean(
        #     tf.square(embed_p_pos) + tf.square(embed_q_pos) + tf.square(embed_q_neg))  # embed_p_pos == embed_q_neg


        # for test stage, self.u: [batch_size]
        self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
                                    transpose_b=True)

        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1)
        # for dns sample
        self.dns_rating = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1)

    def save_model(self, sess, filename):
        param = sess.run(self.d_params)
        pickle.dump(param, open(filename, 'w'))
