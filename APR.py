from keras.engine.saving import load_model
from keras.layers import Input, Embedding, Dot, Subtract, Activation, SimpleRNN, Flatten, Lambda, Add
from keras.models import Model
from keras import backend as K
import numpy as np

import math

from BPR import BPR
from MatrixFactorisation import AdversarialMatrixFactorisation


# Adversarial Personalized Ranking for Recommendation SIGIR 201
class APR(BPR):
    def __init__(self, uNum, iNum, dim, eps=0.5):
        self.uNum = uNum
        self.iNum = iNum
        self.dim = dim
        self.eps = eps

        self.model, self.predictor, self.uEncoder, self.iEncoder = self.generate_apr()
        self.adv_noise_model, self.get_gradients = self.generate_adv_noise()

    def train(self, x_train, y_train, batch_size):

        # train discriminator
        for i in range(math.ceil(len(y_train) / batch_size)):
            _u = x_train[0][i * batch_size:(i * batch_size) + batch_size]
            _p = x_train[1][i * batch_size:(i * batch_size) + batch_size]
            _n = x_train[2][i * batch_size:(i * batch_size) + batch_size]
            _batch_size = _u.shape[0]

            # Embedding
            _ue = self.uEncoder.predict(_u)
            _pe = self.iEncoder.predict(_p)
            _ne = self.iEncoder.predict(_n)

            x = [_u, _p, _n, _ue, _pe, _ne]
            y = np.ones(_batch_size)

            self.adv_noise_model.train_on_batch(x, y)

            uDelta, iDelta = self.get_gradients(x + [y, y, 0])

            # Delta
            _ud = uDelta[_u]
            _pd = iDelta[_p]
            _nd = iDelta[_n]

            # Normalise
            _ud = (_ud / np.linalg.norm(_ud)) * self.eps
            _pd = (_pd / np.linalg.norm(_pd)) * self.eps
            _nd = (_nd / np.linalg.norm(_nd)) * self.eps

            x = [_u, _p, _n, _ud, _pd, _nd]
            y = np.ones(_batch_size)

            hist = self.model.fit(x, y, batch_size=_batch_size, verbose=0)

        return hist

    def generate_apr(self):
        userInput = Input(shape=(1,), dtype="int32")
        itemPosInput = Input(shape=(1,), dtype="int32")
        itemNegInput = Input(shape=(1,), dtype="int32")

        uDelEmb = Input(shape=(self.dim,))
        pDelEmb = Input(shape=(self.dim,))
        nDelEmb = Input(shape=(self.dim,))

        userEmbeddingLayer = Embedding(input_dim=self.uNum, output_dim=self.dim, name="uEmb")
        itemEmbeddingLayer = Embedding(input_dim=self.iNum, output_dim=self.dim, name="iEmb")

        uEmb = Flatten()(userEmbeddingLayer(userInput))
        pEmb = Flatten()(itemEmbeddingLayer(itemPosInput))
        nEmb = Flatten()(itemEmbeddingLayer(itemNegInput))

        pDot = Dot(axes=-1)([uEmb, pEmb])
        nDot = Dot(axes=-1)([uEmb, nEmb])

        diff = Subtract()([pDot, nDot])
        loss = Activation("sigmoid")(diff)

        uPerturbedEmb = Add()([uEmb, uDelEmb])
        pPerturbedEmb = Add()([pEmb, pDelEmb])
        nPerturbedEmb = Add()([nEmb, nDelEmb])

        pPerturbedDot = Dot(axes=-1)([uPerturbedEmb, pPerturbedEmb])
        nPerturbedDot = Dot(axes=-1)([uPerturbedEmb, nPerturbedEmb])

        diffPerturbed = Subtract()([pPerturbedDot, nPerturbedDot])
        lossPerturbed = Activation("sigmoid")(diffPerturbed)

        loss = Add()([loss, lossPerturbed])

        model = Model(inputs=[userInput, itemPosInput, itemNegInput, uDelEmb, pDelEmb, nDelEmb], outputs=loss)
        model.compile(optimizer="adam", loss="binary_crossentropy")

        predictor = Model([userInput, itemPosInput], pDot)
        uEncoder = Model(userInput, uEmb)
        iEncoder = Model(itemPosInput, pEmb)

        return model, predictor, uEncoder, iEncoder

    def generate_adv_noise(self):
        userInput = Input(shape=(1,), dtype="int32", name="uInput")
        itemPosInput = Input(shape=(1,), dtype="int32", name="pInput")
        itemNegInput = Input(shape=(1,), dtype="int32", name="nInput")

        uRealEmb = Input(shape=(self.dim,), name="uRInput")
        pRealEmb = Input(shape=(self.dim,), name="pRInput")
        nRealEmb = Input(shape=(self.dim,), name="nRInput")

        userDeltaEmbeddingLayer = Embedding(input_dim=self.uNum, output_dim=self.dim, name="uDeltaEmb")
        itemDeltaEmbeddingLayer = Embedding(input_dim=self.iNum, output_dim=self.dim, name="iDeltaEmb")

        uDelEmb = Flatten()(userDeltaEmbeddingLayer(userInput))
        pDelEmb = Flatten()(itemDeltaEmbeddingLayer(itemPosInput))
        nDelEmb = Flatten()(itemDeltaEmbeddingLayer(itemNegInput))

        uEmb = Add()([uRealEmb, uDelEmb])
        pEmb = Add()([pRealEmb, pDelEmb])
        nEmb = Add()([nRealEmb, nDelEmb])

        pDot = Dot(axes=-1)([uEmb, pEmb])
        nDot = Dot(axes=-1)([uEmb, nEmb])

        diff = Subtract()([pDot, nDot])

        # Pass difference through sigmoid function.
        pred = Activation("sigmoid", name="sigmoid")(diff)

        model = Model(inputs=[userInput, itemPosInput, itemNegInput, uRealEmb, pRealEmb, nRealEmb], outputs=pred)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        # model.trainable = False

        weights = model.trainable_weights  # weight tensors
        gradients = model.optimizer.get_gradients(model.total_loss, weights)  # gradient tensors
        input_tensors = model.inputs + model.sample_weights + model.targets + [K.learning_phase()]
        get_gradients = K.function(inputs=input_tensors, outputs=gradients)

        return model, get_gradients


# uNum = 5
# iNum = 4
# dim = 3
# size = 10
#
# apr = APR(uNum, iNum, dim)
#
# u = np.random.randint(0, uNum, size)
# p = np.random.randint(0, iNum, size)
# n = np.random.randint(0, iNum, size)
#
# ue = np.random.rand(size, dim)
# pe = np.random.rand(size, dim)
# ne = np.random.rand(size, dim)
#
# x = [u, p, n, ue, pe, ne]
# inputs = x + [np.ones(len(u)), np.ones(len(u)), 0]
# grads = apr.get_gradients(inputs)
# print(grads[0])
# print(grads[0][[1,-1,1]])
