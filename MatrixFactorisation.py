from keras.layers import Input, Embedding, dot, Add, Flatten, Lambda, Dense;
from keras.models import Model;
import numpy as np

class MatrixFactorization:
    def __init__(self, uNum, iNum, dim):
        # Define user input -- user index (an integer)
        userInput = Input(shape=(1,), dtype="int32")
        itemInput = Input(shape=(1,), dtype="int32")
        userEmbeddingLayer = Embedding(input_dim=uNum, output_dim=dim)
        itemEmbeddingLayer = Embedding(input_dim=iNum, output_dim=dim)

        uEmb = Flatten()(userEmbeddingLayer(userInput))
        iEmb = Flatten()(itemEmbeddingLayer(itemInput))

        pred = dot([uEmb, iEmb], axes=-1)

        self.model = Model([userInput, itemInput], pred)
        self.model.compile(optimizer="adam", loss="mean_squared_error", metrics=['mse'])


class AdversarialMatrixFactorisation:

    def __init__(self, uNum, iNum, dim, weight, pop_percent, mode=1):

        self.dim = dim
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
        self.discriminator_i.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
        self.discriminator_i.trainable = False
        validity = self.discriminator_i(iAdvEmb)

        if mode == 2:
            self.discriminator_u = self.generate_discriminator()
            self.discriminator_u.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
            self.discriminator_u.trainable = False
            validity_u = self.discriminator_u(uAdvEmb)


        pred = dot([uEmb, iEmb], axes=-1)

        self.model = Model([userInput, itemInput], pred)
        self.model.compile(optimizer="adam", loss="mean_squared_error", metrics=['mse'])


        if mode == 1:
            self.advModel = Model([userInput, itemInput, itemAdvInput], [pred, validity])
            self.advModel.compile(optimizer="adam", loss=["mean_squared_error", "binary_crossentropy"], metrics=['mse', 'acc'], loss_weights=[1, self.weight])
        else:
            self.advModel = Model([userInput, itemInput, userAdvInput, itemAdvInput], [pred, validity_u, validity])
            self.advModel.compile(optimizer="adam", loss=["mean_squared_error", "binary_crossentropy", "binary_crossentropy"], metrics=['mse', 'acc', 'acc'], loss_weights=[1, self.weight, self.weight])

    # def train(self, x_train, y_train, x_test, y_test):


    def generate_discriminator(self):

        itemInput = Input(shape=(self.dim,))
        hidden = Dense(self.dim, activation="relu")
        finalHidden = Dense(1, activation="sigmoid")

        pred = finalHidden(hidden(itemInput))

        return Model(itemInput, pred)

    def get_discriminator_train_data(self, x_train, x_test, batch_size):

        # print(x_train.shape)
        # print(x_train)
        # print(x_test)

        items = np.concatenate([x_train, x_test], axis=-1)
        popularity = {}
        for i in items:
            if i in popularity:
                popularity[i] += 1
            else:
                popularity[i] = 1

        popularity = {k: v for k, v in sorted(popularity.items(), key=lambda x: x[1])[::-1]}
        popularity = np.array(list(popularity.keys()))

        popular_x = popularity[:int(len(popularity) * self.pop_percent)]
        rare_x = popularity[int(len(popularity) * self.pop_percent):]
        popular_y = np.ones(batch_size)
        rare_y = np.zeros(batch_size)

        return popular_x, popular_y, rare_x, rare_y
