import tensorflow as tf
import pickle


class IRGAN():
    def __init__(self, uNum, iNum, dim, batch_size):
        self.uNum = uNum
        self.iNum = iNum
        self.dim = dim

        INIT_DELTA = 0.05

        self.generator = GEN(iNum, uNum, dim, lamda=0.0 / batch_size, param=None, initdelta=INIT_DELTA,
                             learning_rate=0.001)

        self.discriminator = DIS(iNum, uNum, dim, lamda=0.1 / batch_size, param=None, initdelta=INIT_DELTA,
                                 learning_rate=0.001)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())


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

        self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
                                    transpose_b=True)

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
