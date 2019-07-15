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


class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        # self.trainMatrix, self.trainSeq, self.df = self.load_rating_file_as_matrix(path + ".train.rating")
        # self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        # self.testNegatives = self.load_negative_file(path + ".test.negative")
        self.trainMatrix, self.trainSeq, self.df = self.load_rating_file_as_matrix(path + "Train")
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
                # print(len(negatives))
                negativeList.append(negatives[:100])
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

        mat = sp.dok_matrix((uNum+1, iNum+1), dtype=np.float32)
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
