from Recommender import Recommender
import numpy as np

class MostPopular(Recommender):

    def __init__(self, df):
        self.popular = df.groupby("iid").size().to_dict()

    def load_pre_train(self, pre):
        pass

    def get_train_instances(self, train):
        return None, None

    def save(self, path):
        pass

    def rank(self, users, items):
        score = []
        for i in items:
            if i in self.popular:
                score.append(self.popular[i])
            else:
                score.append(0)
        return score

    def get_params(self):
        return ""

    def train(self, x_train, y_train, batch_size):
        return 0

class MostRecentlyVisit(MostPopular):
    def __init__(self, df):
        self.df = df

    def rank(self, users, items):

        if len(self.df[self.df.uid == users[0]]) == 0:
            return np.zeros(len(items))

        mostRecentVenue = self.df[self.df.uid == users[0]].tail(1).iid.values[0]

        res = []
        for v in items:
            if v == mostRecentVenue:
                res.append(1)
            else:
                res.append(0)
        return res

class MostFrequentlyVisit(MostPopular):

    def __init__(self, df):
        self.df = df

    def rank(self, uid, vids):

        uid = uid[0]

        if len(self.df[self.df.uid == uid]) == 0:
            return np.zeros(len(vids))

        mostFreVenue = self.df[self.df.uid == uid].groupby("iid")['iid'].count().sort_values(ascending=False).index[0]

        res = []
        for v in vids:
            if v == mostFreVenue:
                res.append(1)
            else:
                res.append(0)
        return res

class AlreadyVisit(MostPopular):

    def __init__(self, df):
        self.df = df

    def rank(self, uid, vids):
        uid = uid[0]
        res = []
        for v in vids:
            if (uid, v) in self.df:
                res.append(1)
            else:
                res.append(0)
        return res
