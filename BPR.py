from keras.engine.saving import load_model
from keras.layers import Input, Embedding, Dot, Subtract, Activation, SimpleRNN, Flatten, Lambda
from keras.models import Model
from keras import backend as K
import numpy as np

import math
from MF import AdversarialMatrixFactorisation


def bpr_triplet_loss(X):
    positive_item_latent, negative_item_latent = X

    loss = 1 - K.log(K.sigmoid(positive_item_latent - negative_item_latent))

    return loss


def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)


class BPR():
    def __init__(self, uNum, iNum, dim):

        self.uNum = uNum
        self.iNum = iNum
        self.dim = dim

        self.userInput = Input(shape=(1,), dtype="int32")
        self.itemPosInput = Input(shape=(1,), dtype="int32")
        self.itemNegInput = Input(shape=(1,), dtype="int32")

        userEmbeddingLayer = Embedding(input_dim=uNum, output_dim=dim, name="uEmb")
        itemEmbeddingLayer = Embedding(input_dim=iNum, output_dim=dim, name="iEmb")

        self.uEmb = Flatten()(userEmbeddingLayer(self.userInput))
        self.pEmb = Flatten()(itemEmbeddingLayer(self.itemPosInput))
        self.nEmb = Flatten()(itemEmbeddingLayer(self.itemNegInput))

        pDot = Dot(axes=-1)([self.uEmb, self.pEmb])
        nDot = Dot(axes=-1)([self.uEmb, self.nEmb])

        diff = Subtract()([pDot, nDot])

        # lammbda_output = Lambda(bpr_triplet_loss, output_shape=(1,))
        # self.pred = lammbda_output([pDot, nDot])

        # Pass difference through sigmoid function.
        self.pred = Activation("sigmoid")(diff)

        self.model = Model(inputs=[self.userInput, self.itemPosInput, self.itemNegInput], outputs=self.pred)

        self.model.compile(optimizer="adam", loss="binary_crossentropy")
        # self.model.compile(optimizer="adam", loss=identity_loss)
        self.predictor = Model([self.userInput, self.itemPosInput], [pDot])

    def load_pre_train(self, pre):
        pretrainModel = load_model(pre)
        self.predictor.get_layer("uEmb").set_weights(pretrainModel.get_layer("uEmb").get_weights())
        self.predictor.get_layer("iEmb").set_weights(pretrainModel.get_layer("iEmb").get_weights())

    def save(self, path):
        self.model.save(path, overwrite=True)

    def rank(self, users, items):
        return self.predictor.predict([users, items], batch_size=100, verbose=0)

    def train(self, x_train, y_train, batch_size):
        # for i in range(math.ceil(len(y_train) / batch_size)):
        #     _u = x_train[0][i * batch_size:(i * batch_size) + batch_size]
        #     _p = x_train[1][i * batch_size:(i * batch_size) + batch_size]
        #     _n = x_train[2][i * batch_size:(i * batch_size) + batch_size]
        #     _batch_size = _u.shape[0]
        #     y = np.ones(_batch_size)
        #     hist = self.model.fit([_u, _p, _n], y, batch_size=_batch_size, epochs=1, verbose=0)
        hist = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0)
        loss = hist.history['loss'][0]

        return loss

    def get_train_instances(self, train):
        user_input, pos_item_input, neg_item_input, labels = [], [], [], []
        for (u, i) in train.keys():
            # positive instance
            user_input.append(u)
            pos_item_input.append(i)
            # negative instances
            j = np.random.randint(self.iNum)
            while (u, j) in train:
                j = np.random.randint(self.iNum)
            neg_item_input.append(j)
            labels.append(1)

        # idx = np.arange(len(user_input))
        # np.random.shuffle(idx)
        # return [np.array(user_input)[idx], np.array(pos_item_input)[idx], np.array(neg_item_input)[idx]], np.array(labels)
        return [np.array(user_input), np.array(pos_item_input), np.array(neg_item_input)], np.array(labels)

    def get_params(self):
        return ""


class AdversarialBPR(BPR, AdversarialMatrixFactorisation):
    def __init__(self, uNum, iNum, dim, weight, pop_percent):
        BPR.__init__(self, uNum, iNum, dim)

        self.weight = weight
        self.pop_percent = pop_percent

        self.uEncoder = Model(self.userInput, self.uEmb)
        self.iEncoder = Model(self.itemPosInput, self.pEmb)

        self.discriminator_u = self.generate_discriminator()
        self.discriminator_u.compile(optimizer="adam", loss="binary_crossentropy", metrics=['acc'])
        self.discriminator_u.trainable = False
        validity_u = self.discriminator_u(self.uEmb)

        self.discriminator_i = self.generate_discriminator()
        self.discriminator_i.compile(optimizer="adam", loss="binary_crossentropy", metrics=['acc'])
        self.discriminator_i.trainable = False
        validity_i = self.discriminator_i(self.pEmb)

        self.advModel = Model([self.userInput, self.itemPosInput, self.itemNegInput],
                              [self.pred, validity_u, validity_i])
        self.advModel.compile(optimizer="adam",
                              loss=["binary_crossentropy", "binary_crossentropy", "binary_crossentropy"],
                              metrics=['acc', 'acc', 'acc'], loss_weights=[1, self.weight, self.weight])

    def train(self, x_train, y_train, batch_size):
        for i in range(math.ceil(len(y_train) / batch_size)):
            _u = x_train[0][i * batch_size:(i * batch_size) + batch_size]
            _p = x_train[1][i * batch_size:(i * batch_size) + batch_size]
            _n = x_train[2][i * batch_size:(i * batch_size) + batch_size]
            _labels = y_train[i * batch_size: (i * batch_size) + batch_size]
            _batch_size = _u.shape[0]

            # sample mini-batch for User Discriminator

            idx = np.random.randint(0, len(self.popular_user_x), batch_size)
            _popular_user_x = self.popular_user_x[idx]

            idx = np.random.randint(0, len(self.rare_user_x), batch_size)
            _rare_user_x = self.rare_user_x[idx]

            _popular_user_x = self.uEncoder.predict(_popular_user_x)
            _rare_user_x = self.uEncoder.predict(_rare_user_x)

            d_loss_popular_user = self.discriminator_u.train_on_batch(_popular_user_x, np.ones(batch_size))
            d_loss_rare_user = self.discriminator_u.train_on_batch(_rare_user_x, np.zeros(batch_size))

            # sample mini-batch for Item Discriminator

            idx = np.random.randint(0, len(self.popular_item_x), batch_size)
            _popular_item_x = self.popular_item_x[idx]

            idx = np.random.randint(0, len(self.rare_item_x), batch_size)
            _rare_item_x = self.rare_item_x[idx]

            _popular_item_x = self.iEncoder.predict(_popular_item_x)
            _rare_item_x = self.iEncoder.predict(_rare_item_x)

            d_loss_popular_item = self.discriminator_i.train_on_batch(_popular_item_x, np.ones(batch_size))
            d_loss_rare_item = self.discriminator_i.train_on_batch(_rare_item_x, np.zeros(batch_size))

            # Sample mini-batch for adversarial model

            # Important: we need to swape label to confuse discriminator
            y_user = np.array([0 if i in self.popular_user_x else 1 for i in _u])
            y_item = np.array([0 if i in self.popular_item_x else 1 for i in _p])

            hist = self.advModel.fit([_u, _p, _n], [_labels, y_user, y_item], batch_size=batch_size, epochs=1,
                                     verbose=0)

        return hist
