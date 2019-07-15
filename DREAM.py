# A Dynamic Recurrent Neural Network model for Venue Recommendation
import numpy as np
from keras.models import Model
from keras.layers import Embedding, Input, SimpleRNN, Dot, Subtract, Activation
from keras.preprocessing import sequence
from Recommender import Recommender


class DREAM(Recommender):

    def __init__(self, uNum, vNum, latent_dim, maxVenue):

        self.uNum = uNum
        self.vNum = vNum
        self.latent_dim = latent_dim
        self.maxVenue = maxVenue

        self.positive_input = Input((1,), name='positive_input')
        self.negative_input = Input((1,), name='negative_input')
        self.user_checkin_sequence = Input((maxVenue,), name='user_checkin_sequence')

        self.venue_embedding = Embedding(vNum + 1, self.latent_dim, mask_zero=True,
                                         name="venue_embedding")

        self.rnn = SimpleRNN(self.latent_dim, unroll=True, name="rnn_layer")

        self.positive_venue_embedding = self.venue_embedding(self.positive_input)
        self.negative_venue_embedding = self.venue_embedding(self.negative_input)
        self.hidden_layer = self.rnn(self.venue_embedding(self.user_checkin_sequence))


        pDot = Dot(axes=-1)([self.hidden_layer, self.positive_venue_embedding])
        nDot = Dot(axes=-1)([self.hidden_layer, self.negative_venue_embedding])

        diff = Subtract()([pDot, nDot])

        # Pass difference through sigmoid function.
        self.pred = Activation("sigmoid")(diff)

        self.model = Model(inputs=[self.user_checkin_sequence, self.positive_input, self.negative_input], outputs=self.pred)

        self.model.compile(optimizer="adam", loss="binary_crossentropy")
        self.predictor = Model([self.user_checkin_sequence, self.positive_input], [pDot])


    def init(self, df):
        self.df = df

    def get_train_instances(self, train):
        checkins, positive_venues, negative_venues, labels = [], [], [], []

        for u in range(self.uNum):
            visited = self.df[self.df.uid == u].iid.tolist()
            checkin_ = []
            for v in visited[:-1]:
                checkin_.append(v)
                checkins.extend(sequence.pad_sequences([checkin_[:]], maxlen=self.maxVenue))

            # start from the second venue in user's checkin sequence.
            visited = visited[1:]
            for i in range(len(visited)):
                positive_venues.append(visited[i])

                j = np.random.randint(self.vNum)
                # check if j is in training dataset or in user's sequence at state i or not
                while (u, j) in train or j in visited[:i]:
                    j = np.random.randint(self.vNum)

                negative_venues.append(j)
                labels.append(1)

        return [np.array(checkins), np.array(positive_venues), np.array(negative_venues)], np.array(labels)

    def rank(self, users, items):
        checkins = [self.df[self.df.uid == users[0]].iid.tolist()] * len(items)
        checkins = sequence.pad_sequences(checkins, maxlen=self.maxVenue)
        return self.predictor.predict([checkins, items], batch_size=100, verbose=0)

    def load_pre_train(self, pre):
        super().load_pre_train(pre)

    def train(self, x_train, y_train, batch_size):
        hist = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        loss = hist.history['loss'][0]

        return loss

    def save(self, path):
        super().save(path)

    def get_params(self):
        return ""
