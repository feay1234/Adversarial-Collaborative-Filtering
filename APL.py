import time
import pickle
import argparse
import numpy as np
import tensorflow as tf
import math

from keras.engine.saving import load_model

from BPR import BPR


# def parse_apl_args():
#     parser = argparse.ArgumentParser(description="Run APL.")
#     parser.add_argument('--input_path', nargs='?', default='./data/',
#                         help='Input data path.')
#     parser.add_argument('--loss_function', nargs='?', default='log',
#                         help='Choose a loss function from "log", "wgan" or "hinge".')
#     parser.add_argument('--epochs', type=int, default=200,
#                         help='Number of epochs.')
#     parser.add_argument('--batch_size', type=int, default=32,
#                         help='Batch size.')
#     parser.add_argument('--factors_num', type=int, default=20,
#                         help='Embedding size.')
#     parser.add_argument('--regs', nargs='?', default='[0, 0.05]',
#                         help="Regularization for generator and critic.")
#     parser.add_argument('--lr', type=float, default=0.05,
#                         help='Learning rate.')
#     parser.add_argument('--save_model', type=int, default=0,
#                         help='Whether to save the trained model.')
#     return parser.parse_args()


def init_param(shape):
    return tf.random_uniform([shape[0], shape[1]],
                             minval=-0.05, maxval=0.05, dtype=tf.float32)


def gumbel_softmax(logits, temperature=0.2):
    eps = 1e-20
    u = tf.random_uniform(tf.shape(logits), minval=0, maxval=1)
    gumbel_noise = -tf.log(-tf.log(u + eps) + eps)
    y = tf.log(logits + eps) + gumbel_noise
    return tf.nn.softmax(y / temperature)


class APL(BPR):
    def __init__(self, uNum, iNum, dim, args):
        self.uNum = uNum
        self.iNum = iNum
        self.dim = dim
        self.users_num = uNum
        self.items_num = iNum
        self.factors_num = dim
        self.lr = 0.05
        self.regs = eval('[0, 0.05]',)
        self.loss_function = 'log' #'Choose a loss function from "log", "wgan" or "hinge".')
        self.all_items = set(range(self.items_num))

        np.random.seed(2018)
        tf.set_random_seed(2018)

        self.u = tf.placeholder(tf.int32, name="user_holder")
        self.i = tf.placeholder(tf.int32, name="item_holder")
        self.g_params, self.c_params = self._def_params(None)
        self._build_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def load_pre_train(self, path):

        # pretrainModel = load_model(path)
        # assign_P = self.g_params[0].assign(pretrainModel.get_layer("uEmb").get_weights()[0])
        # assign_Q = self.g_params[1].assign(pretrainModel.get_layer("iEmb").get_weights()[0])
        # self.sess.run([assign_P, assign_Q])
        pass

    def _def_params(self, g_init_param=None):
        with tf.variable_scope("g_params"):
            if g_init_param is None:
                user_embeddings = tf.get_variable("g_user_embs",
                                                  initializer=init_param([self.users_num, self.factors_num]))
                item_embeddings = tf.get_variable("g_item_embs",
                                                  initializer=init_param([self.items_num, self.factors_num]))
            else:
                user_embeddings = tf.get_variable("g_user_embs", initializer=g_init_param[0])
                item_embeddings = tf.get_variable("g_item_embs", initializer=g_init_param[1])
        g_params = [user_embeddings, item_embeddings]

        with tf.variable_scope("c_params"):
            user_embeddings = tf.get_variable("c_user_embs",
                                              initializer=init_param([self.users_num, self.factors_num]))
            item_embeddings = tf.get_variable("c_item_embs",
                                              initializer=init_param([self.items_num, self.factors_num]))
        c_params = [user_embeddings, item_embeddings]
        return g_params, c_params


    def _build_graph(self):
        with tf.name_scope("generator"):
            with tf.variable_scope("g_params", reuse=True):
                user_embeddings = tf.get_variable(name="g_user_embs")
                item_embeddings = tf.get_variable(name="g_item_embs")

            with tf.name_scope("g_latent_vectors"):
                u_embedding = tf.nn.embedding_lookup(user_embeddings, self.u)

            with tf.name_scope("g_mf"):
                self.g_all_logits = tf.matmul(u_embedding, item_embeddings, transpose_b=True)
            self.g_l2_loss = tf.nn.l2_loss(u_embedding) + tf.nn.l2_loss(item_embeddings)

        with tf.name_scope("critic"):
            with tf.variable_scope("c_params", reuse=True):
                user_embeddings = tf.get_variable(name="c_user_embs")
                item_embeddings = tf.get_variable(name="c_item_embs")

            with tf.name_scope("real_item"):
                u_embedding = tf.nn.embedding_lookup(user_embeddings, self.u)
                i_embedding = tf.nn.embedding_lookup(item_embeddings, self.i)
                with tf.name_scope("real_mf"):
                    real_logits = tf.reduce_sum(tf.multiply(u_embedding, i_embedding), 1)
                self.c_l2_loss = tf.nn.l2_loss(u_embedding) + tf.nn.l2_loss(i_embedding)

            with tf.name_scope("fake_item"):
                u_embedding = tf.nn.embedding_lookup(user_embeddings, self.u)
                fake_one_hot = self.sampling()
                i_embedding = tf.matmul(fake_one_hot, item_embeddings)
                with tf.name_scope("fake_mf"):
                    fake_logits = tf.reduce_sum(tf.multiply(u_embedding, i_embedding), 1)
                self.c_l2_loss += tf.nn.l2_loss(u_embedding) + tf.nn.l2_loss(i_embedding)

        self.gen_loss, self.critic_loss = self._get_loss(real_logits, fake_logits)

        g_opt = tf.train.GradientDescentOptimizer(self.lr)
        self.gen_updates = g_opt.minimize(self.gen_loss, var_list=self.g_params)

        d_opt = tf.train.GradientDescentOptimizer(self.lr)
        self.critic_updates = d_opt.minimize(self.critic_loss, var_list=self.c_params)

        if self.loss_function == "wgan":
            with tf.control_dependencies([self.critic_updates]):
                with tf.name_scope("wgan_clip"):
                    self.critic_updates = [var.assign(tf.clip_by_value(var, -0.05, 0.05))
                                           for var in self.c_params]
        return

    def _get_loss(self, real_logits, fake_logits):
        y_ij = real_logits - fake_logits
        with tf.name_scope("g_loss"):
            gen_wgan_loss = -tf.reduce_mean(fake_logits) + self.regs[0] * self.g_l2_loss
            gen_log_loss = tf.reduce_mean(tf.log(tf.sigmoid(y_ij))) + self.regs[0] * self.g_l2_loss
            gen_hinge_loss = -tf.reduce_mean(tf.maximum(1 - y_ij, 0)) + self.regs[0] * self.g_l2_loss
        with tf.name_scope("c_loss"):
            critic_wgan_loss = tf.reduce_mean(-y_ij)
            critic_log_loss = -tf.reduce_mean(tf.log(tf.sigmoid(y_ij))) + self.regs[1] * self.c_l2_loss
            critic_hinge_loss = tf.reduce_mean(tf.maximum(1 - y_ij, 0)) + self.regs[1] * self.c_l2_loss

        loss_dict = {"log": (critic_log_loss, gen_log_loss),
                     "wgan": (critic_wgan_loss, gen_wgan_loss),
                     "hinge": (critic_hinge_loss, gen_hinge_loss)}
        if self.loss_function in loss_dict:
            c_loss, gen_loss = loss_dict[self.loss_function]
        else:
            print("The %s loss is invalid, log loss has been used!" % self.loss_function)
            c_loss, gen_loss = critic_log_loss, gen_log_loss
        return gen_loss, c_loss

    def sampling(self):
        self.training_flag = tf.placeholder(tf.bool)
        fake_one_hot = tf.cond(self.training_flag,
                               true_fn=self._gen_sampling,
                               false_fn=self._critic_sampling)
        return fake_one_hot

    def _gen_sampling(self):
        self.gen_p_aux = tf.placeholder(tf.float32)
        logits = tf.nn.softmax(self.g_all_logits)
        logits = (1 - 0.2) * logits + self.gen_p_aux
        logits = gumbel_softmax(logits, 0.2)
        return logits

    def _critic_sampling(self):
        logits = tf.nn.softmax(self.g_all_logits / 0.2)
        logits = gumbel_softmax(logits, 0.2)
        return logits




    def rank(self, users, items):
        pred = self.sess.run(self.g_all_logits, feed_dict={self.u: [users[0]]})
        return pred[0][items]

    def save(self, path):
        pass

    def init(self, train):

        self.user_pos_item = {i: [] for i in range(self.uNum)}
        user_input, pos_item_input, labels = [], [], []
        for (u, i) in train.keys():
            self.user_pos_item[u].append(i)
            user_input.append(u)
            pos_item_input.append(i)
            labels.append(1)

        self.x_train = [np.array(user_input), np.array(pos_item_input)]
        self.y_train = np.array(labels)

    def get_train_instances(self, train):
        idx = np.arange(len(self.x_train))
        np.random.shuffle(idx)
        return [self.x_train[0][idx], self.x_train[1][idx]], self.y_train

    def train(self, x_train, y_train, batch_size):
        losses = []
        for i in range(math.ceil(len(y_train) / batch_size)):
            print(i)
            _u = x_train[0][i * batch_size:(i * batch_size) + batch_size]
            _i = x_train[1][i * batch_size:(i * batch_size) + batch_size]
            _batch_size = len(_u)
            # _u = np.expand_dims(_u, -1)
            # _p = np.expand_dims(_p, -1)
            # _n = np.expand_dims(_n, -1)


            start_time = time.time()
            _ = self.sess.run([self.critic_updates],
                          feed_dict={self.u: _u, self.i: _i, self.training_flag: False})

            print(_)
            # print("training time of critic: %fs" % (time.time() - start_time))

            p_aux = np.zeros([_batch_size, self.iNum])
            for uid in range(len(_u)):
                p_aux[uid][self.user_pos_item[_u[uid]]] = 0.2 / len(self.user_pos_item[_u[uid]])

            _ = self.sess.run([self.gen_updates],
                                 feed_dict={self.u: _u, self.i: _i,
                                            self.gen_p_aux: p_aux, self.training_flag: True})
            print(_)
            print()

            # print("training time of generator: %fs" % (time.time() - start_time))

            # losses.append(loss)

        # return np.mean(losses)
        return 0

    def training(self):

        print("Dataset: %s\nUsers number: %d\nItems number: %d" % (self.data_set, self.users_num, self.items_num))
        print("Metric:\t\tPrecision@10\t\tRecall@10\t\tMAP@10\t\tNDCG@10")
        result = self.eval()
        buf = '\t'.join([str(x) for x in result])
        print("pre_trained:\t%s" % buf)

        for epoch in range(self.epochs):
            np.random.shuffle(self.data)
            print("epoch: %d" % epoch)

            train_size = len(self.data)
            index = 0

            train_size = len(self.data)
            index = 0
            start_time = time.time()
            while index + self.batch_size < train_size:
                input_user, input_item = get_batch_data(self.data, index, self.batch_size)

                p_aux = np.zeros([self.batch_size, self.items_num])
                for uid in range(len(input_user)):
                    p_aux[uid][self.user_pos_train[input_user[uid]]] = 0.2 / len(self.user_pos_train[input_user[uid]])

                index += self.batch_size
                self.sess.run([self.gen_updates],
                              feed_dict={self.u: input_user, self.i: input_item,
                                         self.gen_p_aux: p_aux, self.training_flag: True})

            print("training time of generator: %fs" % (time.time() - start_time))
            if epoch % 5 == 0:
                result = self.eval()
                buf = '\t'.join([str(x) for x in result])
                print("epoch %d: %s" % (epoch, buf))

            if self.save_model:
                params = self.sess.run(self.g_params)
                pickle.dump(params, open(self.save_model_dir + "%03d_gen_model.pkl" % epoch, "wb"))
        return


#
# if __name__ == "__main__":
#     args = parse_apl_args()
#     apl = APL(10, 5 ,6, args)
#
#     # apl.training()
