import numpy as np
from keras import backend as K
from keras.models import Sequential, Model, load_model, save_model
from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten, SimpleRNN
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.preprocessing import sequence


# use separate embedding for dot operation
class DRCF(MLPDRBPR2):
    def __init__(self, num_users, num_items, layers = [20,10], reg_layers=[0,0], maxVenue =5, opt=Adam(), pre=None):
        MLPDRBPR2.__init__(self, num_users, num_items, layers,reg_layers,  maxVenue)
        # GDRBPR.__init__(self, num_users, num_items, int(layers[-1]), maxVenue)

        self.num_layer = len(layers)  # Number of layers in the MLP
        self.layers = layers

        # Concatenate MF and MLP parts
        predict_vector = merge([self.mf_vector, self.mlp_vector], mode="concat")
        predict_neg_vector = merge([self.mf_neg_vector, self.mlp_neg_vector], mode="concat")


        self.dense = Dense(1, init='lecun_uniform', name='prediction', activation='linear')
        prediction = self.dense(predict_vector)
        neg_prediction = self.dense(predict_neg_vector)


        def bpr_triplet_loss(X):
            pos, neg = X
            loss = 1 - K.log(K.sigmoid(pos - neg))

            return loss

        loss = merge(
            [prediction, neg_prediction],
            mode=bpr_triplet_loss,
            name='loss',
            output_shape=(1,))

        self.model = Model(
            input=[self.user_input, self.checkin_input, self.item_input, self.item_neg_input],
            output=[loss])

        def identity_loss(y_true, y_pred):
            return K.mean(y_pred - 0 * y_true)

        self.model.compile(loss=identity_loss, optimizer=opt)

        if pre != None:
            mf = MF(num_users, num_items, self.layers[-1])
            mf.model.load_weights(pre)

            self.model.get_layer("mf_user_embedding").set_weights(mf.model.get_layer("user_embedding").get_weights())
            self.model.get_layer("mlp_user_embedding").set_weights(mf.model.get_layer("user_embedding").get_weights())
            self.model.get_layer("dot_mf_user_embedding").set_weights(mf.model.get_layer("user_embedding").get_weights())
            self.model.get_layer("dot_mlp_user_embedding").set_weights(mf.model.get_layer("user_embedding").get_weights())

            self.model.get_layer("mf_item_embedding").set_weights(mf.model.get_layer("item_embedding").get_weights())
            self.model.get_layer("mlp_item_embedding").set_weights(mf.model.get_layer("item_embedding").get_weights())
            self.model.get_layer("dot_mf_item_embedding").set_weights(mf.model.get_layer("item_embedding").get_weights())
            self.model.get_layer("dot_mlp_item_embedding").set_weights(mf.model.get_layer("item_embedding").get_weights())


            del mf

    def rank(self, uid, vids, t):
        uids = [uid] * len(vids)


        mf_user_latents = self.model.get_layer('mf_user_embedding').get_weights()[0][uids]
        mf_item_latent = self.model.get_layer('mf_item_embedding').get_weights()[0][vids]

        mlp_user_latents = self.model.get_layer('mlp_user_embedding').get_weights()[0][uids]
        mlp_item_latent = self.model.get_layer('mlp_item_embedding').get_weights()[0][vids]

        dot_mf_user_latent = self.model.get_layer('dot_mf_user_embedding').get_weights()[0][uid]
        dot_mf_item_latent = self.model.get_layer('dot_mf_item_embedding').get_weights()[0][vids]

        dot_mlp_user_latent = self.model.get_layer('dot_mlp_user_embedding').get_weights()[0][uid]
        dot_mlp_item_latent = self.model.get_layer('dot_mlp_item_embedding').get_weights()[0][vids]

        user_checkins = np.array(self.df[self.df.uid == uid].vid)
        user_checkins = sequence.pad_sequences([user_checkins], maxlen=self.maxVenue)

        mf_checkin_latent = self.MF_RNN.predict(user_checkins)
        mf_checkin_latents = np.array([mf_checkin_latent[0]] * len(vids))
        dot_mf_checkin_latent = self.DOT_MF_RNN.predict(user_checkins)

        mlp_checkin_latent = self.MLP_RNN.predict(user_checkins)
        mlp_checkin_latents = np.array([mlp_checkin_latent[0]] * len(vids))
        dot_mlp_checkin_latent = self.DOT_MLP_RNN.predict(user_checkins)


        mf_dot = (np.dot(dot_mf_checkin_latent + dot_mf_user_latent, dot_mf_item_latent.T))
        mf_dot = mf_dot.reshape((mf_dot.shape[-1], 1))
        mlp_dot = (np.dot(dot_mlp_checkin_latent + dot_mlp_user_latent, dot_mlp_item_latent.T))
        mlp_dot = mlp_dot.reshape((mlp_dot.shape[-1], 1))


        mf = np.multiply(mf_checkin_latents, np.multiply(mf_user_latents, mf_item_latent))
        mf = np.concatenate((mf_dot, mf), axis=-1)
        mlp = np.concatenate((mlp_dot, mlp_checkin_latents, mlp_user_latents, mlp_item_latent), axis=-1)

        for idx in range(1, self.num_layer):
            kernel = self.model.get_layer('layer%d' % idx).get_weights()[0]
            bias = self.model.get_layer('layer%d' % idx).get_weights()[1]
            x = np.dot(mlp, kernel) + bias
            mlp = np.maximum(x, 0, x)

        concat = np.concatenate((mf, mlp), axis=-1)

        kernel = self.model.get_layer('prediction').get_weights()[0]
        bias = self.model.get_layer('prediction').get_weights()[1]

        scores = np.dot(concat, kernel) + bias
        return scores

