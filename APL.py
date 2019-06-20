from keras.layers import Input, Embedding, Dot, Subtract, Activation, SimpleRNN, Flatten, Lambda, Dense
from keras.models import Model
from keras import backend as K
import numpy as np

from MatrixFactorisation import AdversarialMatrixFactorisation
from keras.activations import softmax


class APL():
    def __init__(self, uNum, iNum, dim):

        def gumbel_softmax(logits):
            tau = K.variable(0.2, name="temperature")
            eps = 1e-20
            U = K.random_uniform(K.shape(logits), 0, 1)
            #     gumbel_noise = - K.log(-K.log(U + eps) + eps) # logits + gumbel noise
            #     y = tf.log(logits + eps) + gumbel_noise
            y = logits - K.log(-K.log(U + eps) + eps)  # logits + gumbel noise
            y = softmax(K.reshape(y, (-1, iNum)) / tau)
            y = K.argmax(y, axis=-1)
            return y

        userInput = Input(shape=(1,), dtype="int32")
        itemInput = Input(shape=(1,), dtype="int32")

        userGEmbeddingLayer = Embedding(input_dim=uNum, output_dim=dim, name="uGEmb")
        itemGEmbeddingLayer = Dense(iNum, name="iGEmb")
        Gout = itemGEmbeddingLayer(userGEmbeddingLayer(userInput))

        fakeInput = Lambda(gumbel_softmax)(Gout)

        userDEmbeddingLayer = Embedding(input_dim=uNum, output_dim=dim, name="uDEmb")
        itemDEmbeddingLayer = Embedding(input_dim=iNum, output_dim=dim, name="iDEmb")

        uEmb = Flatten()(userDEmbeddingLayer(userInput))
        piEmb = Flatten()(itemDEmbeddingLayer(itemInput))
        niEmb = itemDEmbeddingLayer(fakeInput)

        pDot = Dot(axes=-1)([uEmb, piEmb])
        nDot = Dot(axes=-1)([uEmb, niEmb])
        diff = Subtract()([pDot, nDot])
        # Pass difference through sigmoid function.
        pred = Activation("sigmoid")(diff)
        model = Model([userInput, itemInput], pred)

        self.model.compile(optimizer="adam", loss="binary_crossentropy")
        self.predictor = Model([userInput, itemInput], [pDot])


    def get_train_instances(self, train):
        user_input, pos_item_input, labels = [], [], []
        for (u, i) in train.keys():
            # positive instance
            user_input.append(u)
            pos_item_input.append(i)
            # negative instances

        return [np.array(user_input), np.array(pos_item_input)], np.array(labels)


