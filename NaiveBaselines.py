from Recommender import Recommender
import numpy as np

# Sequential-based baselines are not relevant for new item recommendation tasks (i.e. test items are not interacted by users).

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
            if i[0] in self.popular:
                score.append([self.popular[i[0]]])
            else:
                score.append([0])
        return np.array(score)

    def get_params(self):
        return ""

    def train(self, x_train, y_train, batch_size):
        return 0

class MostRecentlyVisit(MostPopular):
    def __init__(self, df):
        self.df = df

    def rank(self, users, items):

        if len(self.df[self.df.uid == users[0][0]]) == 0:
            return np.zeros(len(items))

        mostRecentVenue = self.df[self.df.uid == users[0][0]].tail(1).iid.values[0]

        res = []
        for v in items:
            if v[0] == mostRecentVenue:
                res.append([1])
            else:
                res.append([0])
        return np.array(res)

class MostFrequentlyVisit(MostPopular):

    def __init__(self, df):
        self.df = df

    def rank(self, uid, vids):

        uid = uid[0][0]

        if len(self.df[self.df.uid == uid]) == 0:
            return np.zeros(len(vids))

        # mostFreVenue = self.df[self.df.uid == uid].groupby("iid")['iid'].count().sort_values(ascending=False).index[0]
        mostFreVenue = self.df[self.df.uid == uid].groupby("iid").count().to_dict()
        # print(mostFreVenue)
        # df.groupby("iid").size().to_dict()
        res = []
        for v in vids:
            if v[0] in mostFreVenue:
                res.append([mostFreVenue[v[0]]])
            else:
                res.append([0])
        print(sum(np.array(res)))
        return np.array(res)
