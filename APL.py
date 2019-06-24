from keras.engine import Layer
from keras.engine.saving import load_model
from keras.initializers import RandomUniform
from keras.layers import Input, Embedding, Dot, Subtract, Activation, SimpleRNN, Flatten, Lambda, Dense, Multiply
from keras.models import Model
from keras import backend as K
import numpy as np
import math
from keras.utils import to_categorical
import tensorflow as tf
from MatrixFactorisation import AdversarialMatrixFactorisation
from scipy.special import softmax


class OnehotEmbedding(Layer):
    def __init__(self, input_num, output_num, **kwargs):
        self.input_num = input_num
        self.output_num = output_num
        # self.enableTranspose = enableTranspose
        super(OnehotEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.input_num, self.output_num),
                                      initializer='uniform',
                                      trainable=True)
        super(OnehotEmbedding, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # if self.enableTranspose:
        #     return K.dot(x, K.transpose(self.kernel))
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        # if self.enableTranspose:
        #     return (input_shape[0], input_shape[-1])
        # return (input_shape[0], self.Nembeddings)
        return (input_shape[0], self.output_num)


class APL():
    def __init__(self, uNum, iNum, dim, trainGeneratorOnly=False):

        self.uNum = uNum
        self.iNum = iNum
        self.dim = dim
        self.trainGeneratorOnly = trainGeneratorOnly

        def generator_gumbel_softmax(x):
            logits, aux = x
            logits = K.softmax(logits)
            logits = (1 - 0.2) * logits + aux

            # This one works
            tau = K.variable(0.2, name="temperature")
            eps = 1e-20
            U = K.random_uniform(K.shape(logits), minval=0, maxval=1)
            y = logits - K.log(-K.log(U + eps) + eps)
            y = K.softmax(y / tau)
            return y

        def discriminator_gumbel_softmax(logits):
            tau = K.variable(0.2, name="temperature")
            eps = 1e-20
            U = K.random_uniform(K.shape(logits), minval=0, maxval=1)
            y = logits - K.log(-K.log(U + eps) + eps)
            y = K.softmax(y / tau)
            return y

        def mul_emb(x):
            fakeEmb, emb = x
            return K.multiply(fakeEmb, emb)

        def mul_shape(x):
            return x[0], iNum, dim

        def gumbel_shape(x):
            return x[0], iNum

        def sum_shape(x):
            return x[0], dim

        userInput = Input(shape=(1,))
        posItemInput = Input(shape=(iNum,), name="pos_item")
        negItemInput = Input(shape=(iNum,), name="neg_item")

        userDEmbeddingLayer = Embedding(input_dim=uNum, output_dim=dim, name="uDEmb")
        itemDEmbeddingLayer = OnehotEmbedding(iNum, dim, name="iDEmb")

        uEmb = Flatten()(userDEmbeddingLayer(userInput))
        piEmb = itemDEmbeddingLayer(posItemInput)
        niEmb = itemDEmbeddingLayer(negItemInput)

        pDot = Dot(axes=-1)([uEmb, piEmb])
        nDot = Dot(axes=-1)([uEmb, niEmb])
        diff = Subtract()([pDot, nDot])
        # Pass difference through sigmoid function.
        pred = Activation("sigmoid")(diff)

        self.discriminator = Model([userInput, posItemInput, negItemInput], pred)
        self.discriminator.compile(optimizer="adam", loss="binary_crossentropy")
        self.discriminator.trainable = False

        fakeItemInput = Input(shape=(iNum,))
        gumbel_out = Lambda(discriminator_gumbel_softmax, output_shape=gumbel_shape, name="gumbel_softmax")(fakeItemInput)
        self.discriminator_gumbel_sampler = Model(fakeItemInput, gumbel_out)

        userGInput = Input(shape=(1,))
        realItemGInput = Input(shape=(iNum,))
        auxGInput = Input(shape=(iNum,))

        userGEmbeddingLayer = Embedding(input_dim=uNum, output_dim=dim, name="uEmb")
        itemGEmbeddingLayer = OnehotEmbedding(dim, iNum, name="iEmb")
        Gout = Flatten()(itemGEmbeddingLayer(userGEmbeddingLayer(userGInput)))


        fakeInput = Lambda(generator_gumbel_softmax, output_shape=gumbel_shape, name="gumbel_softmax")([Gout, auxGInput])

        validity = self.discriminator([userGInput, realItemGInput, fakeInput])

        self.generator = Model([userGInput, realItemGInput, auxGInput], validity)
        self.generator.compile(optimizer="adam", loss="binary_crossentropy")

        self.predictor = Model([userGInput], [Gout])

    def init(self, train):

        self.user_pos_item = {i:[] for i in range(self.uNum)}
        for (u, i) in train.keys():
            self.user_pos_item[u].append(i)

    def rank(self, users, items):
        # items = np.expand_dims(to_categorical(items, self.iNum), axis=-1)
        # items = to_categorical(items, self.iNum)
        # ranks = self.predictor.predict([users, items], batch_size=100, verbose=0)
        ranks = self.predictor.predict(users[:1], batch_size=1, verbose=0)
        # print(ranks.shape)
        # print(items)
        return ranks[0][items]

        # ranks = self.predictor.predict([users, items], batch_size=100, verbose=0)
        # ranks = self.predictor.predict(users, batch_size=100, verbose=0)
        # print(ranks)
        # return ranks[0][items]

    def load_pre_train(self, pre):
        pretrainModel = load_model(pre)
        self.predictor.get_layer("uEmb").set_weights(pretrainModel.get_layer("uEmb").get_weights())
        weight = np.transpose(pretrainModel.get_layer("iEmb").get_weights()[0])
        self.predictor.get_layer("iEmb").set_weights([weight])

    def save(self, path):
        print("TODO")
        # self.model.save(path, overwrite=True)

    def train(self, x_train, y_train, batch_size=32):

        # train discriminator
        for i in range(math.ceil(len(y_train) / batch_size)):
            _u = x_train[0][i * batch_size:(i * batch_size) + batch_size]
            real = x_train[1][i * batch_size:(i * batch_size) + batch_size]
            _labels = y_train[i * batch_size: (i * batch_size) + batch_size]

            # sample fake instances
            fake = self.predictor.predict(_u)
            fake = softmax(fake / 0.2)
            fake = self.discriminator_gumbel_sampler.predict(fake)

            real = to_categorical(real, self.iNum)

            self.discriminator.train_on_batch([_u, real, fake], _labels)

        # train generator
        for i in range(math.ceil(len(y_train) / batch_size)):
            _u = x_train[0][i * batch_size:(i * batch_size) + batch_size]
            real = x_train[1][i * batch_size:(i * batch_size) + batch_size]
            _batch_size = _u.shape[0]
            # swap label to confuse discriminator
            _labels = np.zeros(_batch_size)

            aux = np.zeros([_batch_size, self.iNum])
            for j in range(_batch_size):
                pos_items = self.user_pos_item[_u[j]]
                if len(pos_items) > 0:
                    aux[j][pos_items] = 0.2 / len(pos_items)

            # sample fake instances

            real = to_categorical(real, self.iNum)

            hist = self.generator.fit([_u, real, aux], _labels, batch_size=_batch_size, verbose=0)
        return hist

    # def train(self, x_train, y_train, batch_size=32):
    #
    #     for i in range(math.ceil(len(y_train) / batch_size)):
    #         # print(i)
    #         _u = x_train[0][i * batch_size:(i * batch_size) + batch_size]
    #         real = x_train[1][i * batch_size:(i * batch_size) + batch_size]
    #         fake = x_train[2][i * batch_size:(i * batch_size) + batch_size]
    #         _labels = y_train[i * batch_size: (i * batch_size) + batch_size]
    #         _batch_size = _u.shape[0]
    #
    #         # real = np.expand_dims(to_categorical(real, self.iNum), axis=-1)
    #         # fake = np.expand_dims(to_categorical(fake, self.iNum), axis=-1)
    #         real = to_categorical(real, self.iNum)
    #         fake = to_categorical(fake, self.iNum)
    #
    #         hist = self.discriminator.fit([_u, real, fake], _labels, batch_size=_batch_size, verbose=0)
    #         # print(hist.history, real, fake)
    #     return hist

    def get_train_instances(self, train):
        user_input, pos_item_input, labels = [], [], []
        for (u, i) in train.keys():
            # positive instance
            user_input.append(u)
            pos_item_input.append(i)
            labels.append(1)

        return [np.array(user_input), np.array(pos_item_input)], np.array(labels)

# from keras.datasets import mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape)

# uNum = 5000
# iNum = 5
# dim = 10
#
# apl = APL(uNum, iNum, dim)
#
# u = np.random.randint(0, uNum, size=2)
# i = np.random.randint(0, iNum, size=(2, iNum, 1))
# print(apl.rank(np.array([1]), [1,2,4]))
# print(apl.predictor.predict(np.array([1])).shape)
#
# print(apl.generator.predict(u).shape)
# print(apl.generator.predict(u))

# print(apl.advModel.predict([u,i]))

# i = np.random.randint(0,7, size=(2,7,1))
# y = np.random.randint(0,5, size=(2))
# y2 = np.random.randint(0,7, size=(2,7))
# y3 = np.random.randint(0,7, size=(2,7))
# print(apl.model.predict([u,i]).shape)
# # apl.model.fit([u,i], [y2, y3], batch_size=2)
# apl.model.fit([u,i], [y], batch_size=2)
# #
# # print(apl.model.predict([u,i])[1].shape)
# #
# #
