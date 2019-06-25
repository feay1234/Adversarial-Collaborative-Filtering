from keras.engine.saving import load_model
from keras.layers import Input, Embedding, Dot, Subtract, Activation, SimpleRNN, Flatten, Lambda
from keras.models import Model
from keras import backend as K
import numpy as np

import math

from BPR import BPR
from MatrixFactorisation import AdversarialMatrixFactorisation


class APR(BPR):
    def __init__(self, uNum, iNum, dim):
        BPR.__init__(self, uNum, iNum, dim)

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

        # Pass difference through sigmoid function.
        self.pred = Activation("sigmoid")(diff)

        self.model = Model(inputs=[self.userInput, self.itemPosInput, self.itemNegInput], outputs=self.pred)

        self.model.compile(optimizer="adam", loss="binary_crossentropy")
        self.predictor = Model([self.userInput, self.itemPosInput], [pDot])



