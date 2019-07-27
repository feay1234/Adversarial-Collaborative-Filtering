from Recommender import Recommender


class MostPopular(Recommender):

    def __init__(self, df):
        self.df
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
        pass
