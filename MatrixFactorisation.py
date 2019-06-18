from keras.layers import Input, Embedding, Flatten, Lambda, Dense, dot
from keras.models import Model;
import numpy as np
import math
from tqdm import tqdm


class MatrixFactorization:
    def __init__(self, uNum, iNum, dim):

        self.uNum = uNum
        self.iNum = iNum

        # Define user input -- user index (an integer)
        userInput = Input(shape=(1,), dtype="int32")
        itemInput = Input(shape=(1,), dtype="int32")
        userEmbeddingLayer = Embedding(input_dim=uNum, output_dim=dim)
        itemEmbeddingLayer = Embedding(input_dim=iNum, output_dim=dim)

        uEmb = Flatten()(userEmbeddingLayer(userInput))
        iEmb = Flatten()(itemEmbeddingLayer(itemInput))

        pred = dot([uEmb, iEmb], axes=-1)

        self.model = Model([userInput, itemInput], pred)
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['mse'])

    def get_train_instances(self, train, num_negatives):
        user_input, item_input, labels = [], [], []
        for (u, i) in train.keys():
            # positive instance
            user_input.append(u)
            item_input.append(i)
            labels.append(1)
            # negative instances
            for t in range(num_negatives):
                j = np.random.randint(self.iNum)
                while (u, j) in train:
                    j = np.random.randint(self.iNum)
                user_input.append(u)
                item_input.append(j)
                labels.append(0)
        return user_input, item_input, labels


class AdversarialMatrixFactorisation(MatrixFactorization):
    def __init__(self, uNum, iNum, dim, weight, pop_percent):

        self.uNum = uNum
        self.iNum = iNum
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
        self.discriminator_i.compile(optimizer="adam", loss="binary_crossentropy", metrics=['acc'])
        self.discriminator_i.trainable = False
        validity = self.discriminator_i(iAdvEmb)

        self.discriminator_u = self.generate_discriminator()
        self.discriminator_u.compile(optimizer="adam", loss="binary_crossentropy", metrics=['acc'])
        self.discriminator_u.trainable = False
        validity_u = self.discriminator_u(uAdvEmb)

        pred = dot([uEmb, iEmb], axes=-1)

        self.model = Model([userInput, itemInput], pred)
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['acc'])

        self.advModel = Model([userInput, itemInput, userAdvInput, itemAdvInput], [pred, validity_u, validity])
        self.advModel.compile(optimizer="adam",
                              loss=["binary_crossentropy", "binary_crossentropy", "binary_crossentropy"],
                              metrics=['acc', 'acc', 'acc'], loss_weights=[1, self.weight, self.weight])

    def init(self, users, items):
        self.popular_user_x, self.rare_user_x = self.get_discriminator_train_data(
            users)

        self.popular_item_x, self.rare_item_x = self.get_discriminator_train_data(
            items)

    def train(self, x_train, y_train, batch_size):

        idx = np.random.randint(0, y_train.shape[0], batch_size)
        _x_train = [x_train[0][idx], x_train[1][idx]]
        _y_train = y_train[idx]

        # sample mini-batch for User Discriminator

        idx = np.random.randint(0, len(self.popular_user_x), batch_size)
        _popular_user_x = self.popular_user_x[idx]

        idx = np.random.randint(0, len(self.rare_user_x), batch_size)
        _rare_user_x = self.rare_user_x[idx]

        _popular_user_x = self.uEncoder.predict(_popular_user_x)
        _rare_user_x = self.uEncoder.predict(_rare_user_x)

        d_loss_popular_user = self.discriminator_u.train_on_batch(_popular_user_x, self.popular_user_y)
        d_loss_rare_user = self.discriminator_u.train_on_batch(_rare_user_x, self.rare_user_y)

        # sample mini-batch for Item Discriminator

        idx = np.random.randint(0, len(self.popular_item_x), batch_size)
        _popular_item_x = self.popular_item_x[idx]

        idx = np.random.randint(0, len(self.rare_item_x), batch_size)
        _rare_item_x = self.rare_item_x[idx]

        _popular_item_x = self.iEncoder.predict(_popular_item_x)
        _rare_item_x = self.iEncoder.predict(_rare_item_x)

        d_loss_popular_item = self.discriminator_i.train_on_batch(_popular_item_x, self.popular_item_y)
        d_loss_rare_item = self.discriminator_i.train_on_batch(_rare_item_x, self.rare_item_y)

        # Discriminator's loss
        # d_loss = 0.5 * np.add(d_loss_popular_user, d_loss_rare_user) + 0.5 * np.add(d_loss_popular_item,
        #                                                                             d_loss_rare_item)

        # Sample mini-batch for adversarial model

        idx = np.random.randint(0, len(self.popular_user_x), int(batch_size / 2))
        _popular_user_x = self.popular_user_x[idx]

        idx = np.random.randint(0, len(self.rare_user_x), int(batch_size / 2))
        _rare_user_x = self.rare_user_x[idx]

        idx = np.random.randint(0, len(self.popular_item_x), int(batch_size / 2))
        _popular_item_x = self.popular_item_x[idx]

        idx = np.random.randint(0, len(self.rare_item_x), int(batch_size / 2))
        _rare_item_x = self.rare_item_x[idx]
        #
        _popular_rare_user_x = np.concatenate([_popular_user_x, _rare_user_x])
        _popular_rare_item_x = np.concatenate([_popular_item_x, _rare_item_x])
        #
        _popular_rare_user_x = np.concatenate([_popular_user_x, _rare_user_x])
        _popular_rare_item_x = np.concatenate([_popular_item_x, _rare_item_x])

        _popular_rare_y = np.concatenate([np.zeros(int(batch_size / 2)), np.ones(int(batch_size / 2))])
        # Important: we need to swape label to confuse discriminator
        # _popular_rare_y = np.concatenate([np.ones(int(batch_size / 2)), np.zeros(int(batch_size / 2))])


        # Train adversarial model
        # hist = self.advModel.fit(_x_train + [_popular_rare_user_x, _popular_rare_item_x],
        #                                       [_y_train, _popular_rare_y, _popular_rare_y], batch_size=256, epochs=1, verbose=0)
        hist = self.advModel.fit(_x_train + [_popular_rare_user_x, _popular_rare_item_x],
                                 [_y_train, _popular_rare_y, _popular_rare_y], batch_size=batch_size, epochs=1,
                                 verbose=0)
        return hist

    # Fast but not effective
    def train2(self, x_train, y_train, batch_size):

        # idx = np.random.randint(0, y_train.shape[0], batch_size)
        # _x_train = [x_train[0][idx], x_train[1][idx]]
        # _y_train = y_train[idx]

        # sample mini-batch for User Discriminator

        idx = np.random.randint(0, len(self.popular_user_x), len(y_train))
        _popular_user_x = self.popular_user_x[idx]

        idx = np.random.randint(0, len(self.rare_user_x), len(y_train))
        _rare_user_x = self.rare_user_x[idx]

        _popular_user_x = self.uEncoder.predict(_popular_user_x)
        _rare_user_x = self.uEncoder.predict(_rare_user_x)

        # sample mini-batch for Item Discriminator

        idx = np.random.randint(0, len(self.popular_item_x), len(y_train))
        _popular_item_x = self.popular_item_x[idx]

        idx = np.random.randint(0, len(self.rare_item_x), len(y_train))
        _rare_item_x = self.rare_item_x[idx]

        _popular_item_x = self.iEncoder.predict(_popular_item_x)
        _rare_item_x = self.iEncoder.predict(_rare_item_x)

        # Sample mini-batch for adversarial model

        idx = np.random.randint(0, len(self.popular_user_x), int(len(y_train) / 2))
        half_popular_user_x = self.popular_user_x[idx]

        idx = np.random.randint(0, len(self.rare_user_x), int(len(y_train) / 2))
        half_rare_user_x = self.rare_user_x[idx]

        idx = np.random.randint(0, len(self.popular_item_x), int(len(y_train) / 2))
        half_popular_item_x = self.popular_item_x[idx]

        idx = np.random.randint(0, len(self.rare_item_x), int(len(y_train) / 2))
        half_rare_item_x = self.rare_item_x[idx]
        #
        _popular_rare_user_x = np.concatenate([half_popular_user_x, half_rare_user_x])[:len(y_train)]
        _popular_rare_item_x = np.concatenate([half_popular_item_x, half_rare_item_x])[:len(y_train)]
        _popular_rare_y = np.concatenate([np.zeros(int(len(y_train) / 2)), np.ones(int(len(y_train) / 2))])[
                          :len(y_train)]

        idx = np.random.randint(0, len(self.popular_user_x), len(y_train))
        _popular_rare_user_x = _popular_rare_user_x[idx]
        _popular_rare_item_x = _popular_rare_item_x[idx]
        _popular_rare_y = _popular_rare_y[idx]

        for i in tqdm(range(math.ceil(len(y_train) / batch_size))):
            start = i * batch_size
            end = start + batch_size

            d_loss_popular_user = self.discriminator_u.train_on_batch(_popular_user_x[start:end], np.ones(len(_popular_user_x[start:end])))
            d_loss_rare_user = self.discriminator_u.train_on_batch(_rare_user_x[start:end], np.zeros(len(_rare_user_x[start:end])))
            #
            d_loss_popular_user = self.discriminator_i.train_on_batch(_popular_item_x[start:end], np.ones(len(_popular_item_x[start:end])))
            d_loss_rare_user = self.discriminator_i.train_on_batch(_rare_item_x[start:end], np.zeros(len(_rare_item_x[start:end])))

            # # Train adversarial model
            hist = self.advModel.fit([x_train[0][start:end], x_train[1][start:end]] + [_popular_rare_user_x[start:end], _popular_rare_item_x[start:end]],
                                                [y_train[start:end], _popular_rare_y[start:end], _popular_rare_y[start:end]], verbose=0, batch_size=batch_size, shuffle=True)
        print(hist.history)

        return hist

    def generate_discriminator(self):

        itemInput = Input(shape=(self.dim,))
        hidden = Dense(self.dim, activation="relu")
        finalHidden = Dense(1, activation="sigmoid")

        pred = finalHidden(hidden(itemInput))

        return Model(itemInput, pred)

    def get_discriminator_train_data(self, items):

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
        # popular_y = np.ones(batch_size)
        # rare_y = np.zeros(batch_size)

        return popular_x, rare_x
