from keras.layers import Input, Embedding, Dot, Subtract, Activation, SimpleRNN, Flatten, Lambda, Dense, Multiply
from keras.models import Model
from keras import backend as K
import numpy as np
import math
from keras.utils import to_categorical
from MatrixFactorisation import AdversarialMatrixFactorisation
from keras.activations import softmax


class APL():
    def __init__(self, uNum, iNum, dim):

        self.uNum = uNum
        self.iNum = iNum
        self.dim = dim

        def gumbel_softmax(logits):
            tau = K.variable(0.2, name="temperature")
            eps = 1e-20
            # eps = 1
            U = K.random_uniform(K.shape(logits), minval=0, maxval=1)
            gumbel_noise = - K.log(-K.log(U + eps) + eps) # logits + gumbel noise
            y = K.log(logits + eps) + gumbel_noise
            y = K.softmax(y / tau)
            return K.expand_dims(y)

        def mul_emb(x):
            fakeEmb, emb = x
            return K.multiply(fakeEmb, emb)


        def mul_shape(x):
            return x[0], iNum, dim


        def gumbel_shape(x):
            return x[0], iNum, 1

        def sum_shape(x):
            return x[0], dim

        userGInput = Input(shape=(1,))
        realItemGInput = Input(shape=(1,))

        userGEmbeddingLayer = Embedding(input_dim=uNum, output_dim=dim, name="uEmb")
        itemGEmbeddingLayer = Dense(iNum, name="iGEmb")
        Gout = Flatten()(itemGEmbeddingLayer(userGEmbeddingLayer(userGInput)))

        fakeInput = Lambda(gumbel_softmax, output_shape=gumbel_shape, name="gumbel_softmax")(Gout)

        userDEmbeddingLayer = Embedding(input_dim=uNum, output_dim=dim, name="uDEmb")
        # itemDEmbeddingLayer = Embedding(input_dim=iNum, output_dim=dim, name="iDEmb")
        itemDEmbeddingLayer = Dense(dim, name="iDEmb")

        userInput = Input(shape=(1,))
        posItemInput = Input(shape=(iNum,1,), name="pos_item")
        negItemInput = Input(shape=(iNum,1,), name="neg_item")

        uEmb = Flatten()(userDEmbeddingLayer(userInput))
        piEmb = itemDEmbeddingLayer(posItemInput)
        piEmb = Lambda(lambda x : K.sum(x, axis=1), name="sum", output_shape=sum_shape)(piEmb)
        niEmb = itemDEmbeddingLayer(negItemInput)
        niEmb = Lambda(lambda x : K.sum(x, axis=1), name="sum2", output_shape=sum_shape)(niEmb)

        pDot = Dot(axes=-1)([uEmb, piEmb])
        nDot = Dot(axes=-1)([uEmb, niEmb])
        diff = Subtract()([pDot, nDot])
        # Pass difference through sigmoid function.
        pred = Activation("sigmoid")(diff)


        self.discriminator = Model([userInput, posItemInput, negItemInput], pred)
        self.discriminator.compile(optimizer="adam", loss="binary_crossentropy")
        self.discriminator.trainable = False

        validity = self.discriminator([userGInput, realItemGInput, fakeInput])

        self.advModel = Model([userGInput, realItemGInput], validity)
        self.advModel.compile(optimizer="adam", loss="binary_crossentropy")

        self.generator = Model([userGInput], fakeInput)

        self.predictor = Model([userInput,posItemInput], [pDot])

    def rank(self, users, items):
        items = np.expand_dims(to_categorical(items, self.iNum), axis=-1)
        return self.predictor.predict([users, items], batch_size=100, verbose=0)

    def train(self, x_train, y_train, batch_size=32):

        # train clitic
        for i in range(math.ceil(len(y_train) / batch_size)):
            _u = x_train[0][i * batch_size:(i * batch_size) + batch_size]
            real = x_train[1][i * batch_size:(i * batch_size) + batch_size]
            fake = self.generator.predict(_u)
            _labels = y_train[i * batch_size: (i * batch_size) + batch_size]
            _batch_size = _u.shape[0]

            real = np.expand_dims(to_categorical(real, self.iNum), axis=-1)
            fake = np.expand_dims(to_categorical(fake, self.iNum), axis=-1)

            self.discriminator.train_on_batch([_u, real, fake], _labels)

        for i in range(math.ceil(len(y_train) / batch_size)):
            _u = x_train[0][i * batch_size:(i * batch_size) + batch_size]
            real = x_train[1][i * batch_size:(i * batch_size) + batch_size]
            fake = self.generator.predict(_u)
            _labels = y_train[i * batch_size: (i * batch_size) + batch_size]
            _batch_size = _u.shape[0]

            real = np.expand_dims(to_categorical(real, self.iNum), axis=-1)
            fake = np.expand_dims(to_categorical(fake, self.iNum), axis=-1)

            hist = self.advModel.fit([_u, real, fake], _labels, batch_size=batch_size, verbose=0)
        return hist


    def get_train_instances(self, train):
        user_input, pos_item_input, labels = [], [], []
        for (u, i) in train.keys():
            # positive instance
            user_input.append(u)
            pos_item_input.append(i)
            # negative instances
            labels.append(1)

        return [np.array(user_input), np.array(pos_item_input)], np.array(labels)

# from keras.datasets import mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape)

# uNum = 5
# iNum = 7
# dim = 10
#
# apl = APL(uNum, iNum, dim)
# u = np.random.randint(0,5, size=2)
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