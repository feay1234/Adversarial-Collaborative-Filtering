'''
Created on Aug 8, 2016
Processing datasets.

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
from collections import defaultdict

import scipy.sparse as sp
import numpy as np
import pandas as pd
import random


def getDataset(data, path, evalMode):
    # if data in ["ml-1m", "yelp", "pinterest-20"]:
    if data in ["brightkite", "fsq11", "yelp"]:
        dataset = Dataset(path + "data/" + data, 1)

    elif data == "ml-1m":
        # dataset = Dataset(path + "data/ml-1m", 0)
        names = ["uid", "iid", "rating", "timestamp"]
        train = pd.read_csv(path + "data/ml-1m.train.rating", sep="\t", names=names)
        test = pd.read_csv(path + "data/ml-1m.test.rating", sep="\t", names=names)
        df = train.append(test)
        dataset = PreProcessDataset(df)

    elif data == "yelp-he":
        dataset = Dataset(path + "data/yelp", 0)
        # names = ["uid", "iid", "rating", "timestamp"]
        # train = pd.read_csv(path+"data/yelp.train.rating", sep="\t", names=names)
        # test = pd.read_csv(path+"data/yelp.test.rating", sep="\t", names=names)
        # df = train.append(test)
        # dataset = PreProcessDataset(df)

    elif data == "pinterest-20":
        dataset = Dataset(path + "data/pinterest-20", 0)

    elif data in ["beauty", "steam", "video", "ml-sas"]:
        names = ["uid", "iid"]
        if data == "beauty":
            df = pd.read_csv(path + "data/Beauty.txt", sep=" ", names=names)
        elif data == "steam":
            df = pd.read_csv(path + "data/Steam.txt", sep=" ", names=names)
        elif data == "video":
            df = pd.read_csv(path + "data/Video.txt", sep=" ", names=names)
        else:
            df = pd.read_csv(path + "data/ml-1m.txt", sep=" ", names=names)
        dataset = PreProcessDataset(df, False)

    elif data == "steam":
        names = ["uid", "iid"]
        df = pd.read_csv(path + "data/Steam.txt", sep=" ", names=names)
        dataset = PreProcessDataset(df, False)

    elif data == "brightkite":
        columns = ["uid", "timestamp", "lat", "lng", "iid"]
        df = pd.read_csv(path + "data/brightkite.txt", names=columns, sep="\t")
        dataset = RawDataset(df)
    elif data == "test":
        columns = ["uid", "timestamp", "lat", "lng", "iid"]
        df = pd.read_csv(path + "data/brightkite.txt", names=columns, sep="\t", nrows=10000)
        dataset = PreProcessDataset(df)
    elif data == "gowalla":
        columns = ["uid", "timestamp", "lat", "lng", "iid"]
        df = pd.read_csv(path + "data/gowalla.txt", names=columns, sep="\t")
        dataset = RawDataset(df)

    return dataset


class Dataset(object):
    '''
    classdocs
    '''

    def get_all_negatives(self):
        res = []
        for u in self.trainSeq:
            gtItem = self.testRatings[u]
            cands = set(range(self.trainMatrix.shape[1])) - set(self.trainSeq[u])
            if gtItem in cands:
                cands.remove(gtItem)
            res.append(cands)
        return res

    def __init__(self, path, mode=0, evalMode="sample"):
        '''
        Constructor
        '''
        if mode == 0:
            print(path)
            self.trainMatrix, self.trainSeq, self.df = self.load_rating_file_as_matrix(path + ".train.rating")
            self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
            if evalMode == "sample":
                self.testNegatives = self.load_negative_file(path + ".test.negative")
            elif evalMode == "all":
                self.testNegatives = self.get_all_negatives()
        else:
            self.trainMatrix, self.trainSeq, self.df = self.load_rating_file_as_matrix(path + "Train")
            self.testRatings = self.load_rating_file_as_list(path + "Test")
            if evalMode == "sample":
                self.testNegatives = self.load_negative_file(path + "TestNegative")
            elif evalMode == "all":
                self.testNegatives = self.get_all_negatives()
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
                # print(len(negatives))
                # negativeList.append(negatives[:100])
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
        seq = defaultdict(list)
        columns = ["uid", "iid", "rating", "timestamp"]
        # if "ankita" in path:
        columns = ['uid', 'iid', 'rating', 'hour', 'day', 'month', 'timestamp']
        df = pd.read_csv(filename, names=columns, sep="\t")
        # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.sort_values(["uid", "timestamp"], inplace=True)

        for u, i in df[["uid", "iid"]].values.tolist():
            # mat[u, i] = 1.0
            seq[u].append(i)

        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()

        return mat, seq, df


class RawDataset():
    def __init__(self, df):
        # pre-process
        df = (df
              .merge(df.groupby('uid').iid.nunique().reset_index().rename(columns={'iid': 'num_uniq_vid'}),
                     on='uid', how='left')
              .merge(df.groupby('iid').uid.nunique().reset_index().rename(columns={'uid': 'num_uniq_uid'}),
                     on='iid', how='left'))
        df = df[(df.num_uniq_vid >= 10) & ((df.num_uniq_uid >= 10))]

        # index start at one and index zero is used for masking
        df.uid = df.uid.astype('category').cat.codes.values + 1
        df.iid = df.iid.astype('category').cat.codes.values + 1

        uNum = df.uid.nunique()
        iNum = df.iid.nunique()
        df.sort_values(["uid", "timestamp"], inplace=True)
        self.testRatings = df.groupby("uid").tail(1)[["uid", "iid"]].values.tolist()
        # for each user, remove last interaction from training set
        df = df.groupby("uid", as_index=False).apply(lambda x: x.iloc[:-1])

        mat = sp.dok_matrix((uNum + 1, iNum + 1), dtype=np.float32)
        seq = defaultdict(list)

        for u, i in df[["uid", "iid"]].values.tolist():
            mat[u, i] = 1.0
            seq[u].append(i)

        self.trainMatrix = mat
        self.trainSeq = seq
        self.df = df

        random.seed(2019)
        candidates = df.iid.tolist()

        negatives = []
        for u in range(uNum):
            neg = []
            for i in range(100):
                r = random.choice(candidates)
                while (u, r) in mat or self.testRatings[u] == r:
                    r = random.choice(candidates)
                neg.append(r)
            negatives.append(neg)

        self.testNegatives = negatives
        assert len(self.testRatings) == len(self.testNegatives)

        self.num_users, self.num_items = self.trainMatrix.shape


class PreProcessDataset():
    def __init__(self, df, doSort=True):

        # remove users who has less than 3
        # df = df.groupby("uid").filter(lambda x: len(x) >= 3)

        # index start at one and index zero is used for masking
        df.uid = df.uid.astype('category').cat.codes.values + 1
        df.iid = df.iid.astype('category').cat.codes.values + 1

        uNum = df.uid.nunique()
        iNum = df.iid.nunique()
        if doSort:
            df.sort_values(["uid", "timestamp"], inplace=True)

        self.testRatings = df.groupby("uid").tail(1)[["uid", "iid"]].values.tolist()
        # for each user, remove last interaction from training set
        df = df.groupby("uid", as_index=False).apply(lambda x: x.iloc[:-1])

        mat = sp.dok_matrix((uNum + 1, iNum + 1), dtype=np.float32)
        seq = defaultdict(list)

        for u, i in df[["uid", "iid"]].values.tolist():
            mat[u, i] = 1.0
            seq[u].append(i)

        self.trainMatrix = mat
        self.trainSeq = seq
        self.df = df

        random.seed(2019)
        candidates = df.iid.tolist()

        negatives = []
        for u in range(uNum):
            neg = []
            for i in range(100):
                r = random.choice(candidates)
                while (u, r) in mat or self.testRatings[u] == r:
                    r = random.choice(candidates)
                neg.append(r)
            negatives.append(neg)

        self.testNegatives = negatives
        assert len(self.testRatings) == len(self.testNegatives)

        self.num_users, self.num_items = self.trainMatrix.shape


class HeDataset(object):
    '''
    Loading the data file
        trainMatrix: load rating records as sparse matrix for class Data
        trianList: load rating records as list to speed up user's feature retrieval
        testRatings: load leave-one-out rating test for class Evaluate
        testNegatives: sample the items not rated by user
    '''

    def __init__(self, path, mode=0):
        '''
        Constructor
        '''
        if mode == 0:
            self.trainMatrix = self.load_training_file_as_matrix(path + ".train.rating")
            self.trainList = self.load_training_file_as_list(path + ".train.rating")
            self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
            self.testNegatives = self.load_negative_file(path + ".test.negative")

            # extract seq data
            names = ["uid", "iid", "rating", "timestamp"]
            self.df = pd.read_csv(path + ".train.rating", sep="\t", names=names)
            self.df.sort_values(["uid", "timestamp"], inplace=True)
            self.trainSeq = defaultdict(list)

            for u, i in self.df[["uid", "iid"]].values.tolist():
                self.trainSeq[u].append(i)


        elif mode == 1:
            self.trainMatrix = self.load_training_file_as_matrix(path + "Train")
            self.trainList = self.load_training_file_as_list(path + "Train")
            self.testRatings = self.load_rating_file_as_list(path + "Test")
            self.testNegatives = self.load_negative_file(path + "TestNegative")

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

    def load_training_file_as_matrix(self, filename):
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
        print("already load the trainMatrix...")
        return mat

    def load_training_file_as_list(self, filename):
        # Get number of users and items
        u_ = 0
        lists, items = [], []
        with open(filename, "r") as f:
            line = f.readline()
            index = 0
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                if u_ < u:
                    index = 0
                    lists.append(items)
                    items = []
                    u_ += 1
                index += 1
                # if index<300:
                items.append(i)
                line = f.readline()
        lists.append(items)
        print("already load the trainList...")
        return lists
