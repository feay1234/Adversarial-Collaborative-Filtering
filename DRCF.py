import numpy as np
from keras import backend as K
from keras.models import Sequential, Model, load_model, save_model
from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten, SimpleRNN, Lambda, \
    Concatenate, Dot, Add, Multiply
from keras.optimizers import Adam
from keras_preprocessing import sequence

from BPR import BPR

def init_normal(shape, name=None):
    return K.random_normal_variable(shape, 0.0, 0.01, name=name)

# use separate embedding for dot operation
class DRCF(BPR):


    def __init__(self, num_users, num_items, dim, maxlen =5, opt=Adam()):
        # latent_dim = layers[-1]
        self.uNum = num_users
        self.iNum = num_items
        self.maxlen = maxlen

        latent_dim = dim
        layers = [dim, dim * 3, dim * 2 , dim]

        # Input variables
        self.user_input = Input(shape=(1,), dtype='int32', name='user_input')
        self.item_input = Input(shape=(1,), dtype='int32', name='item_input')
        self.item_neg_input = Input(shape=(1,), dtype='int32', name='item_neg_input')
        self.checkin_input = Input(shape=(maxlen,), dtype='int32', name='checkin_input')

        self.MF_Embedding_User = Embedding(input_dim=num_users, output_dim=latent_dim, name='mf_user_embedding',
                                      init=init_normal, input_length=1)

        self.MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=latent_dim, name='mf_item_embedding',
                                      init=init_normal, input_length=1)

        self.MF_Embedding_Checkin = Embedding(input_dim=num_items, output_dim=latent_dim, name='mf_checkin_embedding',
                                      init=init_normal, input_length=maxlen)

        self.DOT_MF_Embedding_User = Embedding(input_dim=num_users, output_dim=latent_dim, name='dot_mf_user_embedding',
                                           init=init_normal, input_length=1)

        self.DOT_MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=latent_dim, name='dot_mf_item_embedding',
                                           init=init_normal, input_length=1)

        self.DOT_MF_Embedding_Checkin = Embedding(input_dim=num_items, output_dim=latent_dim, name='dot_mf_checkin_embedding',
                                              init=init_normal, input_length=maxlen)

        self.MF_RNN = Sequential()
        self.MF_RNN.add(self.MF_Embedding_Checkin)
        self.MF_RNN.add(SimpleRNN(latent_dim, unroll=True, name="mf_rnn_layer"))

        self.DOT_MF_RNN = Sequential()
        self.DOT_MF_RNN.add(self.DOT_MF_Embedding_Checkin)
        self.DOT_MF_RNN.add(SimpleRNN(latent_dim, unroll=True, name="mf_rnn_layer"))

        # Crucial to flatten an embedding vector!
        self.mf_user_latent = Flatten()(self.MF_Embedding_User(self.user_input))
        self.mf_item_latent = Flatten()(self.MF_Embedding_Item(self.item_input))
        self.mf_item_neg_latent = Flatten()(self.MF_Embedding_Item(self.item_neg_input))
        self.mf_dynamic_user_latent = Flatten()(Reshape((1, latent_dim))(self.MF_RNN(self.checkin_input)))

        self.mf_dot_user_latent = Flatten()(self.DOT_MF_Embedding_User(self.user_input))
        self.mf_dot_item_latent = Flatten()(self.DOT_MF_Embedding_Item(self.item_input))
        self.mf_dot_item_neg_latent = Flatten()(self.DOT_MF_Embedding_Item(self.item_neg_input))
        self.mf_dot_dynamic_user_latent = Flatten()(Reshape((1, latent_dim))(self.DOT_MF_RNN(self.checkin_input)))


        # Dot-product of user and item embeddings
        self.dot_vector = Dot(axes=-1)([Add()([self.mf_dot_dynamic_user_latent , self.mf_dot_user_latent]), self.mf_dot_item_latent])
        self.dot_neg_vector = Dot(axes=-1)([Add()([self.mf_dot_dynamic_user_latent , self.mf_dot_user_latent]), self.mf_dot_item_neg_latent])

        # Element-wise product of user and item embeddings
        self.mf_vector = Concatenate()([self.dot_vector, Multiply()([self.mf_dynamic_user_latent, self.mf_user_latent, self.mf_item_latent])])
        self.mf_neg_vector = Concatenate()([self.dot_neg_vector, Multiply()([self.mf_dynamic_user_latent, self.mf_user_latent, self.mf_item_neg_latent])])

        self.num_layer = len(layers)  # Number of layers in the MLP
        self.layers = layers

        self.MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2),
                                            name='mlp_user_embedding',
                                            init=init_normal, input_length=1)

        self.MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2), name='mlp_item_embedding',
                                            init=init_normal, input_length=1)

        self.MLP_Embedding_checkin = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2),
                                               name='mlp_checkin_embedding',
                                               init=init_normal, input_length=maxlen)

        self.DOT_MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2),
                                                name='dot_mlp_user_embedding',
                                                init=init_normal, input_length=1)
        self.DOT_MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2),
                                                name='dot_mlp_item_embedding',
                                                init=init_normal, input_length=1)

        self.DOT_MLP_Embedding_checkin = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2),
                                                   name='dot_mlp_checkin_embedding',
                                                   init=init_normal, input_length=maxlen)

        self.MLP_RNN = Sequential()
        self.MLP_RNN.add(self.MLP_Embedding_checkin)
        self.MLP_RNN.add(SimpleRNN(int(layers[0] / 2), unroll=True, name="mlp_rnn_layer"))

        self.DOT_MLP_RNN = Sequential()
        self.DOT_MLP_RNN.add(self.DOT_MLP_Embedding_checkin)
        self.DOT_MLP_RNN.add(SimpleRNN(int(layers[0] / 2), unroll=True, name="dot_mlp_rnn_layer"))

        # Crucial to flatten an embedding vector!
        self.mlp_user_latent = Flatten()(self.MLP_Embedding_User(self.user_input))
        self.mlp_item_latent = Flatten()(self.MLP_Embedding_Item(self.item_input))
        self.mlp_item_neg_latent = Flatten()(self.MLP_Embedding_Item(self.item_neg_input))
        self.mlp_dynamic_user_latent = Flatten()(Reshape((1, int(layers[0] / 2)))(self.MLP_RNN(self.checkin_input)))

        self.mlp_dot_user_latent = Flatten()(self.DOT_MLP_Embedding_User(self.user_input))
        self.mlp_dot_item_latent = Flatten()(self.DOT_MLP_Embedding_Item(self.item_input))
        self.mlp_dot_item_neg_latent = Flatten()(self.DOT_MLP_Embedding_Item(self.item_neg_input))
        self.mlp_dot_dynamic_user_latent = Flatten()(Reshape((1, int(layers[0] / 2)))(self.DOT_MLP_RNN(self.checkin_input)))

        # Dot-product of user and item embeddings
        self.mlp_dot_vector = Dot(axes=-1)([Add()([self.mlp_dot_dynamic_user_latent, self.mlp_dot_user_latent]), self.mlp_dot_item_latent])
        self.mlp_dot_neg_vector = Dot(axes=-1)([Add()([self.mlp_dot_dynamic_user_latent, self.mlp_dot_user_latent]), self.mlp_dot_item_neg_latent])

        # The 0-th layer is the concatenation of embedding layers
        self.mlp_vector = Concatenate()(
            [self.mlp_dot_vector, self.mlp_dynamic_user_latent, self.mlp_user_latent, self.mlp_item_latent])
        self.mlp_neg_vector = Concatenate()(
            [self.mlp_dot_neg_vector, self.mlp_dynamic_user_latent, self.mlp_user_latent, self.mlp_item_neg_latent])

        # MLP layers
        for idx in range(1, self.num_layer):
            layer = Dense(layers[idx], activation='relu', name='layer%d' % idx)
            self.mlp_vector = layer(self.mlp_vector)
            self.mlp_neg_vector = layer(self.mlp_neg_vector)


        # Concatenate MF and MLP parts
        predict_vector = Concatenate()([self.mf_vector, self.mlp_vector])
        predict_neg_vector = Concatenate()([self.mf_neg_vector, self.mlp_neg_vector])



        self.dense = Dense(1, init='lecun_uniform', name='prediction', activation='linear')
        prediction = self.dense(predict_vector)
        neg_prediction = self.dense(predict_neg_vector)


        def bpr_triplet_loss(X):
            pos, neg = X
            loss = 1 - K.log(K.sigmoid(pos - neg))

            return loss

        bprloss = Lambda(bpr_triplet_loss, output_shape=(1,))
        loss = bprloss([prediction, neg_prediction])

        self.model = Model(
            input=[self.user_input, self.checkin_input, self.item_input, self.item_neg_input],
            output=[loss])

        def identity_loss(y_true, y_pred):
            return K.mean(y_pred - 0 * y_true)

        self.model.compile(loss=identity_loss, optimizer=opt)

        self.predictor = Model([self.user_input, self.checkin_input, self.item_input], [prediction])

    def rank(self, users, items):
        checkins = [self.trainSeq[users[0][0]]] * len(items)
        checkins = sequence.pad_sequences(checkins, maxlen=self.maxlen)
        return self.predictor.predict([users, checkins, items], batch_size=512, verbose=0)
        # res = []
        # for i in items:
        #     res.append(self.predictor.predict([users, checkins, [i]], verbose=0))
        # return res



    def get_train_instances(self, train):
        users, checkins, positive_venues, negative_venues, labels = [], [], [], [], []

        for u in self.trainSeq:
            visited = self.trainSeq[u]
            checkin_ = []
            for v in visited[:-1]:
                checkin_.append(v)
                checkins.extend(sequence.pad_sequences([checkin_[:]], maxlen=self.maxlen))
                users.append(u)

            # start from the second venue in user's checkin sequence.
            for i in range(2, len(visited) + 1, 1):

                j = np.random.randint(self.iNum)
                # check if j is in training dataset or in user's sequence at state i or not
                while (u, j) in train or j in visited[:i]:
                    j = np.random.randint(self.iNum)

                negative_venues.append(j)

            for v in visited[1:]:
                positive_venues.append(v)
                labels.append(1)

        return [np.array(users), np.array(checkins), np.array(positive_venues), np.array(negative_venues)], np.array(labels)


    def init(self, trainSeq):
        self.trainSeq = trainSeq


    def get_params(self):
        return "_ml%d" % (self.maxlen)