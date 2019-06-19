from keras.layers import Input, Embedding, Dot, Subtract, Activation, SimpleRNN, Flatten, Lambda
from keras.models import Model
from keras import backend as K
import numpy as np


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

        diff = Subtract()([pDot, nDot])

        # Pass difference through sigmoid function.
        pred = Activation("sigmoid")(diff)

        self.model = Model(inputs = [userInput, itemPosInput, itemNegInput], outputs = pred)

        def identity_loss(y_true, y_pred):
            return K.mean(y_pred - 0 * y_true)

        self.model.compile(optimizer = "adam", loss = "binary_crossentropy")
        self.predictor = Model([userInput, itemPosInput], [pDot])

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

        return [np.array(user_input), np.array(pos_item_input), np.array(neg_item_input)], np.array(labels)

class AdversarialBPR(BPR):

    def __init__(self, uNum, iNum, dim, weight, pop_percent):
        BPR.__init__(self, uNum, iNum, dim)

        self.weight = weight
        self.pop_percent = pop_percent

        # Define user input -- user index (an integer)
        userInput = Input(shape=(1,), dtype="int32")
        itemInput = Input(shape=(1,), dtype="int32")
        userAdvInput = Input(shape=(1,), dtype="int32")
        itemAdvInput = Input(shape=(1,), dtype="int32")

        userEmbeddingLayer = Embedding(input_dim=uNum, output_dim=dim)
        itemEmbeddingLayer = Embedding(input_dim=iNum, output_dim=dim)

        uEmb = Flatten()(userEmbeddingLayer(userInput))
        iEmb = Flatten()(itemEmbeddingLayer(itemInput))
        uAdvEmb = Flatten()(userEmbeddingLayer(userAdvInput))
        iAdvEmb = Flatten()(itemEmbeddingLayer(itemAdvInput))

        self.uEncoder = Model(userInput, uEmb)
        self.iEncoder = Model(itemInput, iEmb)

        self.discriminator_i = self.generate_discriminator()
        self.discriminator_i.compile(optimizer="adam", loss="binary_crossentropy", metrics=['acc'])
        self.discriminator_i.trainable = False
        validity = self.discriminator_i(iAdvEmb)
        # validity = self.discriminator_i(iEmb)

        self.discriminator_u = self.generate_discriminator()
        self.discriminator_u.compile(optimizer="adam", loss="binary_crossentropy", metrics=['acc'])
        self.discriminator_u.trainable = False
        validity_u = self.discriminator_u(uAdvEmb)
        # validity_u = self.discriminator_u(uEmb)

        pred = dot([uEmb, iEmb], axes=-1)

        self.model = Model([userInput, itemInput], pred)
        self.model.compile(optimizer="adam", loss="mean_squared_error", metrics=['mse'])

        self.advModel = Model([userInput, itemInput, userAdvInput, itemAdvInput], [pred, validity_u, validity])
        # self.advModel = Model([userInput, itemInput], [pred, validity_u, validity])
        self.advModel.compile(optimizer="adam",
                              loss=["mean_squared_error", "binary_crossentropy", "binary_crossentropy"],
                              metrics=['mse', 'acc', 'acc'], loss_weights=[1, self.weight, self.weight])
