from deepctr.inputs import SparseFeat, VarLenSparseFeat, get_fixlen_feature_names, get_varlen_feature_names
from deepctr.models import DSIN

# [IJCAI 2019]Deep Session Interest Network for Click-Through Rate Prediction
import numpy as np
from keras_preprocessing import sequence

from Recommender import Recommender


class DeepSessionInterestNetwork(Recommender):
    def __init__(self, uNum, iNum, dim, maxlen):
        self.uNum = uNum
        self.iNum = iNum
        self.dim = dim
        self.maxlen = maxlen

        hash_flag = True
        self.feature_columns = [SparseFeat('user', self.uNum, hash_flag),
                                SparseFeat('item', self.iNum, hash_flag),
                                VarLenSparseFeat('sess_0_item', self.iNum, self.dim, use_hash=hash_flag,
                                                 embedding_name='item')]

        self.behavior_feature_list = ["item"]

        self.model = DSIN(self.feature_columns, self.behavior_feature_list, sess_max_count=1,
                          embedding_size=self.dim,
                          att_head_num=self.dim,
                          dnn_hidden_units=[self.dim, self.dim, self.dim], dnn_dropout=0.5)

        self.model.compile('adam', 'binary_crossentropy',
                           metrics=['acc'])

    def init(self, trainSeq):
        self.trainSeq = trainSeq

    def load_pre_train(self, pre):
        super().load_pre_train(pre)

    def get_params(self):
        super().get_params()

    def train(self, x_train, y_train, batch_size):
        history = self.model.fit(x, y, verbose=1, epochs=10, validation_split=0.5)
        hist = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0)
        loss = hist.history['loss'][0]

        return loss

    def get_train_instances(self, train):
        users, checkins, cand_venues, labels = [], [], [], []

        for u in self.trainSeq:
            visited = self.trainSeq[u]
            checkin_ = []
            for v in visited[:-1]:
                checkin_.append(v)
                checkins.extend(sequence.pad_sequences([checkin_[:]], maxlen=self.maxVenue))

            # start from the second venue in user's checkin sequence.
            visited = visited[1:]
            for i in range(len(visited)):
                cand_venues.append(visited[i])
                users.append(u)
                labels.append(1)
                j = np.random.randint(self.uNum)
                # check if j is in training dataset or in user's sequence at state i or not
                while (u, j) in train or j in visited[:i]:
                    j = np.random.randint(self.uNum)

                cand_venues.append(j)
                users.append(u)
                labels.append(0)

        sess_number = np.ones(len(labels))

        users = np.array(users)
        items = np.array(cand_venues)
        sess_item = np.array(checkins)
        labels = np.array(labels)

        feature_dict = {'user': users, 'item': items, 'score': labels, 'sess_0_item': sess_item}

        fixlen_feature_names = get_fixlen_feature_names(self.feature_columns)
        varlen_feature_names = get_varlen_feature_names(self.feature_columns)
        x = [feature_dict[name] for name in fixlen_feature_names] + [feature_dict[name] for name in
                                                                     varlen_feature_names]
        x += [sess_number]

        return x, labels

    def rank(self, users, items):
        super().rank(users, items)

    def save(self, path):
        super().save(path)

a = DeepSessionInterestNetwork(5,9,3, 2)
