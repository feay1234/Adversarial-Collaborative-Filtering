'''
Created on Aug 8, 2016
Processing datasets.

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)

        self.num_users, self.num_items = self.trainMatrix.shape

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()
        return mat


class RawDataset():

    def __init__(self, df):
        np.random.seed(1111)

        # pre-process
        df = df.groupby("iid").filter(lambda x: len(x) >= 10)
        # df = df.groupby("uid").filter(lambda x: len(x) >= 10)

        df.uid = df.uid.astype('category').cat.codes.values
        df.iid = df.iid.astype('category').cat.codes.values
        uNum = df.uid.max() + 1
        iNum = df.iid.max() + 1
        self.testRatings = df.groupby("uid").tail(1)[["uid", "iid"]].values.tolist()

        df = df[df.groupby("uid").cumcount(ascending=False) > 0]
        mat = sp.dok_matrix((uNum + 1, iNum + 1), dtype=np.float32)
        if "rating" in df.columns:
            for u, i, r in df[["uid", "iid", "rating"]].values.tolist():
                mat[u, i] = r
        else:
            for u, i in df[["uid", "iid"]].values.tolist():
                mat[u, i] = 1
        self.trainMatrix = mat

        negatives = []
        for u in range(uNum):
            neg = []
            for i in range(99):
                r = np.random.randint(0, iNum, 1)[0]
                while (u, r) in mat:
                    r = np.random.randint(0, iNum, 1)[0]
                neg.append(r)
            negatives.append(neg)

        self.testNegatives = negatives
        assert len(self.testRatings) == len(self.testNegatives)

        self.num_users, self.num_items = self.trainMatrix.shape

