

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Multiply, Concatenate, Flatten, Dropout
import numpy as np

from MatrixFactorisation import AdversarialMatrixFactorisation


class NeuMF():
    def __init__(self, num_users, num_items, mf_dim=10):

        layers = [mf_dim, mf_dim*2, mf_dim]

        num_layer = len(layers)  # Number of layers in the MLP
        # Input variables
        self.user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
        self.item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

        # Embedding layer
        self.MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user', input_length=1)
        self.MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item', input_length=1)

        self.MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = layers[0 ], name = "mlp_embedding_user", input_length=1)
        self.MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = layers[0 ], name = 'mlp_embedding_item', input_length=1)

        # MF part
        self.mf_user_latent = Flatten()(self.MF_Embedding_User(self.user_input))
        self.mf_item_latent = Flatten()(self.MF_Embedding_Item(self.item_input))
        mf_vector = Multiply()([self.mf_user_latent, self.mf_item_latent])

        # MLP part
        self.mlp_user_latent = Flatten()(self.MLP_Embedding_User(self.user_input))
        self.mlp_item_latent = Flatten()(self.MLP_Embedding_Item(self.item_input))
        mlp_vector = Concatenate()([self.mlp_user_latent, self.mlp_item_latent])
        for idx in range(1, num_layer):
            layer = Dense(layers[idx], activation='relu', name="layer%d" %idx)
            mlp_vector = layer(mlp_vector)

        predict_vector = Concatenate()([mf_vector, mlp_vector])

        # Final prediction layer
        self.pred = Dense(1, activation='linear', name="prediction")(predict_vector)

        self.model = Model(input=[self.user_input, self.item_input],
                      output=self.pred)

        self.model.compile(optimizer="adam", loss="mean_squared_error", metrics=['mse'])


class AdversarialNeuMF(NeuMF,AdversarialMatrixFactorisation):
    def __init__(self, num_users, num_items, mf_dim, weight, pop_percent):
        NeuMF.__init__(self, num_users, num_items, mf_dim=mf_dim)

        self.dim = mf_dim
        self.weight = weight
        self.pop_percent = pop_percent

        self.userMFAdvInput = Input(shape=(1,), dtype="int32")
        self.itemMFAdvInput = Input(shape=(1,), dtype="int32")
        self.userMLPAdvInput = Input(shape=(1,), dtype="int32")
        self.itemMLPAdvInput = Input(shape=(1,), dtype="int32")

        self.discriminator_mf_u = self.generate_discriminator()
        self.discriminator_mf_i = self.generate_discriminator()
        self.discriminator_mlp_u = self.generate_discriminator()
        self.discriminator_mlp_i = self.generate_discriminator()

        self.discriminator_mf_u.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
        self.discriminator_mf_i.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
        self.discriminator_mlp_u.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
        self.discriminator_mlp_i.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

        self.uMFEncoder = Model(self.user_input, self.mf_user_latent)
        self.iMFEncoder = Model(self.item_input, self.mf_item_latent)

        self.uMLPEncoder = Model(self.user_input, self.mlp_user_latent)
        self.iMLPEncoder = Model(self.item_input, self.mlp_item_latent)

        self.discriminator_mf_u.trainable = False
        self.discriminator_mf_i.trainable = False
        self.discriminator_mlp_u.trainable = False
        self.discriminator_mlp_i.trainable = False

        uMFAdvEmb = Flatten()(self.MF_Embedding_User(self.userMFAdvInput))
        iMFAdvEmb = Flatten()(self.MF_Embedding_Item(self.itemMFAdvInput))
        uMLPAdvEmb = Flatten()(self.MLP_Embedding_User(self.userMLPAdvInput))
        iMLPAdvEmb = Flatten()(self.MLP_Embedding_Item(self.itemMLPAdvInput))

        self.pred_u_mf_disc = self.discriminator_mf_u(uMFAdvEmb)
        self.pred_i_mf_disc = self.discriminator_mf_i(iMFAdvEmb)
        self.pred_u_mlp_disc = self.discriminator_mlp_u(uMLPAdvEmb)
        self.pred_i_mlp_disc = self.discriminator_mlp_i(iMLPAdvEmb)


        self.advModel = Model([self.user_input, self.item_input, self.userMFAdvInput, self.itemMFAdvInput, self.userMLPAdvInput, self.itemMLPAdvInput], [self.pred, self.pred_u_mf_disc, self.pred_i_mf_disc, self.pred_u_mlp_disc, self.pred_i_mlp_disc])
        self.advModel.compile(optimizer="adam", loss=["mean_squared_error", "binary_crossentropy", "binary_crossentropy", "binary_crossentropy", "binary_crossentropy"], metrics=['mse', 'acc', 'acc', 'acc', 'acc'], loss_weights=[1, self.weight, self.weight, self.weight, self.weight])

    def train(self, x_train, y_train, batch_size):

        idx = np.random.randint(0, y_train.shape[0], batch_size)
        _x_train = [x_train[0][idx], x_train[1][idx]]
        _y_train = y_train[idx]

        # sample mini-batch for User Discriminator

        idx = np.random.randint(0, len(self.popular_user_x), batch_size)
        _popular_user_x = self.popular_user_x[idx]

        idx = np.random.randint(0, len(self.rare_user_x), batch_size)
        _rare_user_x = self.rare_user_x[idx]

        _popular_mf_user_x = self.uMFEncoder.predict(_popular_user_x)
        _rare_mf_user_x = self.uMFEncoder.predict(_rare_user_x)

        _popular_mlp_user_x = self.uMLPEncoder.predict(_popular_user_x)
        _rare_mlp_user_x = self.uMLPEncoder.predict(_rare_user_x)


        d_loss_popular_mf_user = self.discriminator_mf_u.train_on_batch(_popular_mf_user_x, self.popular_user_y)
        d_loss_rare_mf_user = self.discriminator_mf_u.train_on_batch(_rare_mf_user_x, self.rare_user_y)

        d_loss_popular_mlp_user = self.discriminator_mlp_u.train_on_batch(_popular_mlp_user_x, self.popular_user_y)
        d_loss_rare_mlp_user = self.discriminator_mlp_u.train_on_batch(_rare_mlp_user_x, self.rare_user_y)

        # sample mini-batch for Item Discriminator

        idx = np.random.randint(0, len(self.popular_item_x), batch_size)
        _popular_item_x = self.popular_item_x[idx]

        idx = np.random.randint(0, len(self.rare_item_x), batch_size)
        _rare_item_x = self.rare_item_x[idx]

        _popular_mf_item_x = self.iMFEncoder.predict(_popular_item_x)
        _rare_mf_item_x = self.iMFEncoder.predict(_rare_item_x)
        _popular_mlp_item_x = self.iMLPEncoder.predict(_popular_item_x)
        _rare_mlp_item_x = self.iMLPEncoder.predict(_rare_item_x)

        d_loss_popular_mf_item = self.discriminator_mf_i.train_on_batch(_popular_mf_item_x, self.popular_item_y)
        d_loss_rare_mf_item = self.discriminator_mf_i.train_on_batch(_rare_mf_item_x, self.rare_item_y)

        d_loss_popular_mlp_item = self.discriminator_mlp_i.train_on_batch(_popular_mlp_item_x, self.popular_item_y)
        d_loss_rare_mlp_item = self.discriminator_mlp_i.train_on_batch(_rare_mlp_item_x, self.rare_item_y)

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

        _popular_rare_user_x = np.concatenate([_popular_user_x, _rare_user_x])
        _popular_rare_item_x = np.concatenate([_popular_item_x, _rare_item_x])

        _popular_rare_y = np.concatenate([np.zeros(int(batch_size / 2)), np.ones(int(batch_size / 2))])
        # Important: we need to swape label to confuse discriminator
        # _popular_rare_y = np.concatenate([np.ones(int(batch_size / 2)), np.zeros(int(batch_size / 2))])


        # Train adversarial model
        g_loss = self.advModel.train_on_batch(_x_train + [_popular_rare_user_x, _popular_rare_item_x, _popular_rare_user_x, _popular_rare_item_x],
                                              [_y_train, _popular_rare_y, _popular_rare_y, _popular_rare_y, _popular_rare_y])


# a = AdversarialMatrixFactorisation(1,1,1,1,1)

