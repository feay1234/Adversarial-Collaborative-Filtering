from keras.layers import Input, Embedding, Dot, Subtract, Activation, SimpleRNN, Flatten, Lambda, Dense, Multiply
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

        userInput = Input(shape=(1,))
        itemInput = Input(shape=(7,1,), name="item")

        userGEmbeddingLayer = Embedding(input_dim=uNum, output_dim=dim, name="uEmb")
        # itemGEmbeddingLayer = Embedding(input_dim=iNum, output_dim=dim, name="iEmb")
        itemGEmbeddingLayer = Dense(iNum, name="iGEmb")
        Gout = Flatten()(itemGEmbeddingLayer(userGEmbeddingLayer(userInput)))

        fakeInput = Lambda(gumbel_softmax, output_shape=gumbel_shape, name="gumbel_softmax")(Gout)

        userDEmbeddingLayer = Embedding(input_dim=uNum, output_dim=dim, name="uDEmb")
        # itemDEmbeddingLayer = Embedding(input_dim=iNum, output_dim=dim, name="iDEmb")
        itemDEmbeddingLayer = Dense(dim, name="iDEmb")

        uEmb = Flatten()(userDEmbeddingLayer(userInput))
        piEmb = itemDEmbeddingLayer(itemInput)
        piEmb = Lambda(lambda x : K.sum(x, axis=1), name="sum", output_shape=sum_shape)(piEmb)
        # niEmb = Lambda(mul_emb, output_shape=mul_shape)([fakeInput, itemDEmbeddingLayer.embeddings])
        # niEmb = Lambda(lambda x : K.sum(x, axis=1), name="sum", output_shape=sum_shape)(niEmb)
        # niEmb = Multiply()([fakeInput, itemDEmbeddingLayer.embeddings])
        niEmb = itemDEmbeddingLayer(fakeInput)
        niEmb = Lambda(lambda x : K.sum(x, axis=1), name="sum2", output_shape=sum_shape)(niEmb)

        pDot = Dot(axes=-1)([uEmb, piEmb])
        nDot = Dot(axes=-1)([uEmb, niEmb])
        diff = Subtract()([pDot, nDot])
        # Pass difference through sigmoid function.
        pred = Activation("sigmoid")(diff)
        self.model = Model([userInput, itemInput], [pred])

        self.model.compile(optimizer="adam", loss="binary_crossentropy")
        # self.predictor = Model([userInput], [Gout])


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

uNum = 5
iNum = 7
dim = 10

apl = APL(uNum, iNum, dim)
u = np.random.randint(0,5, size=2)
i = np.random.randint(0,7, size=(2,7,1))
y = np.random.randint(0,5, size=(2))
y2 = np.random.randint(0,7, size=(2,7))
y3 = np.random.randint(0,7, size=(2,7))
print(apl.model.predict([u,i]).shape)
# apl.model.fit([u,i], [y2, y3], batch_size=2)
apl.model.fit([u,i], [y], batch_size=2)
#
# print(apl.model.predict([u,i])[1].shape)
#
#