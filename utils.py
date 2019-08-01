import numpy as np
import random
import torch
import pandas as pd

from Dataset import Dataset


def write2file(path, output):
    print(output)
    thefile = open(path, 'a')
    thefile.write("%s\n" % output)
    thefile.close()

def prediction2file(path, pred):
    thefile = open(path, 'w')
    for item in pred:
        thefile.write("%f\n" % item)
    thefile.close()

def set_seed(seed, cuda=False):

    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)

def getDataset(data, path, evalMode):
    # if data in ["ml-1m", "yelp", "pinterest-20"]:
    if data in ["brightkite", "fsq11", "yelp"]:
        columns = ['uid', 'iid', 'rating', 'hour', 'day', 'month', 'timestamp']
        train = pd.read_csv(path + "Train", names=columns, sep="\t")
        test = pd.read_csv(path + "Test")
        df = train.append(test)
        df.sort_values(["uid", "timestamp"], inplace=True)
        dataset = Dataset(df, evalMode)

    elif data in ["ml-1m", "yelp-he"]:
        names = ["uid", "iid", "rating", "timestamp"]
        train = pd.read_csv(path + "data/%s.train.rating" % data, sep="\t", names=names)
        test = pd.read_csv(path + "data/%s.test.rating" % data, sep="\t", names=names)
        df = train.append(test)
        dataset = Dataset(df, evalMode)

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
        dataset = Dataset(df, evalMode)

    elif data == "test":
        columns = ["uid", "timestamp", "lat", "lng", "iid"]
        df = pd.read_csv(path + "data/brightkite.txt", names=columns, sep="\t", nrows=10000)
        dataset = Dataset(df, evalMode)

    return dataset
