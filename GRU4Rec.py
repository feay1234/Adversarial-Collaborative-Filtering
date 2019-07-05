from keras.optimizers import Adam

import keras
import keras.backend as K
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy
from keras.layers import Input, Dense, Dropout, GRU, Embedding
from keras_preprocessing.sequence import pad_sequences

from Recommender import Recommender


class GRU4Rec(Recommender):
    def __init__(self, uNum, iNum, dim, maxlen):
        self.uNum = uNum
        self.iNum = iNum
        self.dim = dim
        self.maxlen = maxlen
        inputs = Input(shape=(self.maxlen,))
        emb = Embedding(input_dim=self.iNum, output_dim=self.dim, name="iEmb", mask_zero=True)
        gru = GRU(self.dim, return_sequences=True)(emb(inputs))
        drop2 = Dropout(0.25)(gru)
        predictions = Dense(self.iNum + 1, activation='softmax')(drop2)
        self.model = Model(input=inputs, output=[predictions])
        self.model.compile(loss=categorical_crossentropy, optimizer=Adam())

    def init(self, trainSeq):
        self.trainSeq = trainSeq

    def load_pre_train(self, pre):
        super().load_pre_train(pre)

    def get_train_instances(self, train):
        seq, label = [], []
        for u in self.trainSeq:
            # positive instance
            seq.append(self.trainSeq[u][:-1])
            label.append(self.trainSeq[u][1:])

        seq = pad_sequences(seq, self.maxlen)
        label = pad_sequences(label, self.maxlen)
        label = to_categorical(label, self.iNum + 1)

        return seq, label

    def rank(self, users, items):
        seq = pad_sequences([self.trainSeq[users[0]]], self.maxlen)
        # predict next item, using the output of the last hidden state
        return self.model.predict(seq)[0][-1][items]

    def train(self, x_train, y_train, batch_size):

        hist = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0, shuffle=False)
        loss = hist.history['loss'][0]

        return "%.4f" % loss

    def save(self, path):
        super().save(path)


# b = GRU4Rec(10, 10, 5, 3)
# s = {1: [1, 2, 3, 4]}
# b.init(s)
# print(b.rank([1, 1, 1], [5, 6]).shape)
# x, y = b.get_train_instances(None)
# print(b.train(x,y, 32))
