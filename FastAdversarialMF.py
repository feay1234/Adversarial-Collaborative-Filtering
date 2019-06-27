from keras.layers import Input, Embedding, Flatten, Lambda, Dense, dot
from keras.models import Model;
import numpy as np
from keras.optimizers import Adam

from MF import MatrixFactorization

from keras_adversarial import AdversarialModel, fix_names, n_choice
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling, AdversarialOptimizerScheduled, \
    AdversarialOptimizerAlternating


class FastAdversarialMF(MatrixFactorization):
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
        self.discriminator_i.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
        self.discriminator_i.trainable = False
        validity = self.discriminator_i(iAdvEmb)

        self.discriminator_u = self.generate_discriminator()
        self.discriminator_u.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
        self.discriminator_u.trainable = False
        validity_u = self.discriminator_u(uAdvEmb)

        pred = dot([uEmb, iEmb], axes=-1)
        # pred = merge([uEmb, iEmb], mode="concat")

        self.model = Model([userInput, itemInput], pred)
        # self.model.compile(optimizer="adam", loss="mean_squared_error", metrics=['mse'])

        # self.advModel = Model([userInput, itemInput, userAdvInput, itemAdvInput], [pred, validity_u, validity])
        # self.advModel.compile(optimizer="adam",
        #                       loss=["mean_squared_error", "binary_crossentropy", "binary_crossentropy"],
        #                       metrics=['mse', 'acc', 'acc'], loss_weights=[1, self.weight, self.weight])

        self.aae = Model([userInput, itemInput, userAdvInput, itemAdvInput],
                         fix_names([pred, validity_u, validity], ["xpred", "upred", "ipred"]))

        mf_params = self.uEncoder.trainable_weights + self.iEncoder.trainable_weights
        self.advModel = AdversarialModel(base_model=self.aae,
                                         player_params=[mf_params, self.discriminator_u.trainable_weights,
                                                        self.discriminator_i.trainable_weights],
                                         player_names=["mf", "disc_u", "disc_i"])

        self.advModel.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(),
                                          player_optimizers=[Adam(), Adam(), Adam()],
                                          loss={"upred": "binary_crossentropy", "ipred": "binary_crossentropy",
                                                "xpred": "mean_squared_error"},
                                          player_compile_kwargs=[{"loss_weights": {"upred": 1, "ipred": 1,
                                                                                   "xpred": 1}}] * 2)

        # model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
        #                           player_optimizers=[Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4)],
        #                           loss={"yfake": "binary_crossentropy", "yreal": "binary_crossentropy",
        #                                 "xpred": "mean_squared_error"},
        #                           player_compile_kwargs=[{"loss_weights": {"yfake": 1e-2, "yreal": 1e-2,
        #                                                                    "xpred": 1}}] * 2)

    def init(self, users, items):
        self.popular_user_x, self.rare_user_x = self.get_discriminator_train_data(users)
        self.popular_item_x, self.rare_item_x = self.get_discriminator_train_data(items)




    def train(self, x_train, y_train, batch_size):


        # sample batches for User Discriminator

        pop_idx = np.random.randint(0, len(self.popular_user_x), int(len(y_train) / 2))
        rare_idx = np.random.randint(0, len(self.rare_user_x), int(len(y_train) / 2))

        user_x = np.concatenate([self.popular_user_x[pop_idx], self.rare_user_x[rare_idx]])[:len(y_train)]
        user_y = np.concatenate([np.ones(int(len(y_train) / 2)), np.zeros(int(len(y_train) / 2))])[:len(y_train)]
        user_x = self.uEncoder.predict((user_x))

        # sample mini-batches for Item Discriminator

        pop_idx = np.random.randint(0, len(self.popular_item_x), int(len(y_train) / 2))
        rare_idx = np.random.randint(0, len(self.rare_item_x), int(len(y_train) / 2))

        item_x = np.concatenate([self.popular_item_x[pop_idx], self.rare_item_x[rare_idx]])[:len(y_train)]
        item_y = np.concatenate([np.ones(int(len(y_train) / 2)), np.zeros(int(len(y_train) / 2))])[:len(y_train)]
        item_x = self.iEncoder.predict((item_x))

        # Train adversarial model
        x = x_train + [user_x, item_x, user_x, item_x]
        y = [y_train, user_y, item_y, y_train, user_y[::-1], item_y[::-1]]

        history = self.advModel.fit(x=x, y=y, batch_size=batch_size, epochs=1, verbose=1)

        return history

    def generate_discriminator(self):

        itemInput = Input(shape=(self.dim,))
        hidden = Dense(self.dim, activation="relu")
        finalHidden = Dense(1, activation="sigmoid")

        pred = finalHidden(hidden(itemInput))

        return Model(itemInput, pred)

    def get_discriminator_train_data(self, x):

        popularity = {}
        for i in x:
            if i in popularity:
                popularity[i] += 1
            else:
                popularity[i] = 1

        popularity = {k: v for k, v in sorted(popularity.items(), key=lambda x: x[1])[::-1]}
        popularity = np.array(list(popularity.keys()))

        popular_x = popularity[:int(len(popularity) * self.pop_percent)]
        rare_x = popularity[int(len(popularity) * self.pop_percent):]

        return popular_x, rare_x

