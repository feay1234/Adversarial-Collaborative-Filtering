import sys;

from keras.layers import Input, Embedding, Dot, Subtract, Activation, SimpleRNN, Flatten, Lambda
from keras.models import Model
from tqdm.autonotebook import tqdm
from keras import backend as K
import numpy as np
from keras_preprocessing.sequence import pad_sequences


def init_normal(shape, name=None):
    return K.random_normal_variable(shape, mean = 0.0, scale=0.01, name=name)

def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

def bpr_triplet_loss(X):
    positive_item_latent, negative_item_latent = X

    loss = 1 - K.log(K.sigmoid(positive_item_latent - negative_item_latent))

    return loss

class BPR():

    def __init__(self, uNum, iNum, dim):

        self.uNum = uNum
        self.iNum = iNum
        self.dim = dim

        userInput = Input(shape=(1,), dtype="int32")
        itemPosInput = Input(shape=(1,), dtype="int32")
        itemNegInput = Input(shape=(1,), dtype="int32")

        userEmbeddingLayer = Embedding(input_dim=uNum, output_dim=dim)
        itemEmbeddingLayer = Embedding(input_dim=iNum, output_dim=dim)


        uEmb = Flatten()(userEmbeddingLayer(userInput))
        pEmb = Flatten()(itemEmbeddingLayer(itemPosInput))
        nEmb = Flatten()(itemEmbeddingLayer(itemNegInput))


        pDot = Dot(axes = -1)([uEmb, pEmb])
        nDot = Dot(axes = -1)([uEmb, nEmb])

        # pDot = Lambda(self.getDot)([uEmb, pEmb])
        # nDot = Lambda(self.getDot)([uEmb, nEmb])

        diff = Subtract()([pDot, nDot])

        # dotDifferenceLayer = Lambda(self.getDotDifference, output_shape=self.getDotDifferenceShape) \
        #     ([pDot, nDot])

        # Pass difference through sigmoid function.
        pred = Activation("sigmoid")(diff)

        self.model = Model(inputs = [userInput, itemPosInput, itemNegInput], outputs = pred)

        def identity_loss(y_true, y_pred):
            return K.mean(y_pred - 0 * y_true)

        self.model.compile(optimizer = "adam", loss = "binary_crossentropy")
        # self.get_score = K.function([userInput, itemPosInput], [pDot])
        self.predictor = Model([userInput, itemPosInput], [pDot])

    # def predict(self, inp):
    #     return self.predictor.predict(inp)
        # return self.get_score(inp)[0].flatten()

    def getDot(self, x):
        user_latent, positive_item_latent = x
        return K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True)

    def getDotDifference(self, parameterMatrixList):
        positive_item_latent, negative_item_latent = parameterMatrixList

        return 1 - K.log(K.sigmoid(positive_item_latent - negative_item_latent))

        # return K.batch_dot(userEmbeddingMatrix, itemPositiveEmbeddingMatrix, axes=1) - K.batch_dot(userEmbeddingMatrix,
        #                                                                                            itemNegativeEmbeddingMatrix,
        #                                                                                            axes=1)

    def getDotDifferenceShape(self, shapeVectorList):
        positive_item_latent, negative_item_latent = shapeVectorList;
        return positive_item_latent[0], 1;

    def get_train_instances(self, train):
        user_input, pos_item_input, neg_item_input, labels = [], [], [], []
        for (u, i) in train.keys():
            # positive instance
            user_input.append(u)
            pos_item_input.append(i)
            # labels.append(1)
            # labels.append(train[u,i])
            # negative instances
            j = np.random.randint(self.iNum)
            while (u, j) in train:
                j = np.random.randint(self.iNum)
            neg_item_input.append(j)
            labels.append(1)

        return [np.array(user_input), np.array(pos_item_input), np.array(neg_item_input)], np.array(labels)
