# A Dynamic Recurrent Neural Network model for Venue Recommendation
import numpy as np
from keras.models import Model
from keras.layers import Embedding, Input, SimpleRNN, Dot, Subtract, Activation
from keras.preprocessing import sequence
from Recommender import Recommender
import tensorflow as tf
import math

class DREAM(Recommender):
    def __init__(self, uNum, iNum, latent_dim, maxlen):

        self.uNum = uNum
        self.iNum = iNum
        self.latent_dim = latent_dim
        self.maxlen = maxlen

        self.positive_input = Input((1,), name='positive_input')
        self.negative_input = Input((1,), name='negative_input')
        self.user_checkin_sequence = Input((maxlen,), name='user_checkin_sequence')

        self.venue_embedding = Embedding(iNum + 1, self.latent_dim, mask_zero=True,
                                         name="venue_embedding")

        self.rnn = SimpleRNN(self.latent_dim, unroll=True, name="rnn_layer")

        self.positive_venue_embedding = self.venue_embedding(self.positive_input)
        self.negative_venue_embedding = self.venue_embedding(self.negative_input)
        self.hidden_layer = self.rnn(self.venue_embedding(self.user_checkin_sequence))

        pDot = Dot(axes=-1)([self.hidden_layer, self.positive_venue_embedding])
        nDot = Dot(axes=-1)([self.hidden_layer, self.negative_venue_embedding])

        diff = Subtract()([pDot, nDot])

        # Pass difference through sigmoid function.
        self.pred = Activation("sigmoid")(diff)

        self.model = Model(inputs=[self.user_checkin_sequence, self.positive_input, self.negative_input],
                           outputs=self.pred)

        self.model.compile(optimizer="adam", loss="binary_crossentropy")
        self.predictor = Model([self.user_checkin_sequence, self.positive_input], [pDot])

    def init(self, df):
        self.df = df

    def get_train_instances(self, train):
        checkins, positive_venues, negative_venues, labels = [], [], [], []

        for u in range(self.uNum):
            visited = self.df[self.df.uid == u].iid.tolist()
            checkin_ = []
            for v in visited[:-1]:
                checkin_.append(v)
                checkins.extend(sequence.pad_sequences([checkin_[:]], maxlen=self.maxlen))

            # start from the second venue in user's checkin sequence.
            visited = visited[1:]
            for i in range(len(visited)):
                positive_venues.append(visited[i])

                j = np.random.randint(self.iNum)
                # check if j is in training dataset or in user's sequence at state i or not
                while (u, j) in train or j in visited[:i]:
                    j = np.random.randint(self.iNum)

                negative_venues.append(j)
                labels.append(1)

        return [np.array(checkins), np.array(positive_venues), np.array(negative_venues)], np.array(labels)

    def rank(self, users, items):
        checkins = [self.df[self.df.uid == users[0]].iid.tolist()] * len(items)
        checkins = sequence.pad_sequences(checkins, maxlen=self.maxlen)
        return self.predictor.predict([checkins, items], batch_size=100, verbose=0)

    def load_pre_train(self, pre):
        super().load_pre_train(pre)

    def train(self, x_train, y_train, batch_size):
        hist = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        loss = hist.history['loss'][0]

        return loss

    def save(self, path):
        super().save(path)

    def get_params(self):
        return ""


class DREAM_TF(DREAM):
    def __init__(self, uNum, iNum, latent_dim, maxlen):
        self.uNum = uNum
        self.iNum = iNum
        self.dim = latent_dim
        self.maxlen = maxlen

        self.seq_input = tf.placeholder(tf.int32, shape=[None, self.maxlen], name="seq_input")
        self.item_input_pos = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_pos")
        self.item_input_neg = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_neg")

        self.embedding = tf.Variable(
            tf.truncated_normal(shape=[self.iNum, self.dim], mean=0.0, stddev=0.01),
            name='embedding', dtype=tf.float32)  # (items, embedding_size)

        self.rnn = tf.contrib.rnn.BasicRNNCell(self.dim)

        # embedding look up
        seq_emb = tf.nn.embedding_lookup(self.embedding, self.seq_input)
        pos_item_emb = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding, self.item_input_pos), 1)
        neg_item_emb = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding, self.item_input_neg), 1)

        outputs, states = tf.nn.dynamic_rnn(self.rnn, seq_emb, dtype=tf.float32)

        final_output = outputs[:, -1, :]

        # pos_score = tf.matmul(final_output , pos_item_emb , transpose_b=False)
        # neg_score = tf.matmul(final_output , neg_item_emb , transpose_b=False)
        h = tf.constant(1.0, tf.float32, [self.dim, 1], name="h")

        pos_score = tf.matmul(final_output * pos_item_emb, h)
        neg_score = tf.matmul(final_output * neg_item_emb, h)

        self.loss = tf.reduce_mean(-tf.log(tf.nn.sigmoid(pos_score-neg_score)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

        # predict_user_embed = tf.nn.embedding_lookup(user_embedding , self.X_predict)
        # self.predict = tf.matmul(predict_user_embed , item_embedding , transpose_b=True)
        self.predictor = pos_score


        self.sess = tf.Session() #create session
        self.sess.run(tf.global_variables_initializer())

    def rank(self, users, items):
        items = np.expand_dims(items, -1)
        checkins = [self.df[self.df.uid == users[0]].iid.tolist()] * len(items)
        checkins = sequence.pad_sequences(checkins, maxlen=self.maxlen)

        feed_dict = {self.seq_input: checkins, self.item_input_pos: items}
        pred = self.sess.run(self.predictor, feed_dict)
        return pred


    def train(self, x_train, y_train, batch_size):
        losses = []
        for i in range(np.math.ceil(len(y_train) / batch_size)):
            _s = x_train[0][i * batch_size:(i * batch_size) + batch_size]
            _p = x_train[1][i * batch_size:(i * batch_size) + batch_size]
            _n = x_train[2][i * batch_size:(i * batch_size) + batch_size]

            _p = np.expand_dims(_p, -1)
            _n = np.expand_dims(_n, -1)

            feed_dict = {self.seq_input: _s,
                         self.item_input_pos: _p,
                         self.item_input_neg: _n}

            _, loss = self.sess.run([self.optimizer, self.loss], feed_dict)

            losses.append(loss)

        return np.mean(losses)



#
# ## Define the shape of the tensor
# X = tf.placeholder(tf.float32, [None, 5])
# X_batch = np.random.rand(2,5)
# ## Define the network
# basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=10)
# outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
# init = tf.global_variables_initializer()
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     init.run()
#     outputs_val = outputs.eval(feed_dict={X: X_batch})
# print(outputs_val)
#
