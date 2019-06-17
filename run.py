import argparse
import math
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from MatrixFactorisation import MatrixFactorization, AdversarialMatrixFactorisation
from NeuMF import NeuMF, AdversarialNeuMF


def parse_args():
    parser = argparse.ArgumentParser(description="Run Adversarial Collaborative Filtering")

    parser.add_argument('--path', type=str, help='Path to data', default="")

    parser.add_argument('--model', type=str,
                        help='Model Name: lstm', default="aneumf")

    parser.add_argument('--data', type=str,
                        help='Dataset name', default="ml-small")

    parser.add_argument('--d', type=int, default=10,
                        help='Dimension')

    parser.add_argument('--epochs', type=int, default=50,
                        help='Epoch number')

    parser.add_argument('--w', type=float, default=0.001,
                        help='Weight:')

    parser.add_argument('--pp', type=float, default=0.2,
                        help='Popularity Percentage:')

    parser.add_argument('--bs', type=int, default=256,
                        help='Batch Size:')

    return parser.parse_args()


if __name__ == '__main__':
    start = time()

    args = parse_args()

    path = args.path
    dataset = args.data
    modelName = args.model
    dim = args.d
    weight = args.w
    pop_percent = args.pp
    batch_size = args.bs
    epochs = args.epochs

    columns = ["uid", "iid", "rating", "timestamp"]

    if dataset == "ml-small":
        df = pd.read_csv(path+"data/ml-latest-small/ratings.csv", names=columns,
                         skiprows=1)
    elif dataset == "ml":
        df = pd.read_csv(path+"data/ml-20m/ratings.csv", names=columns,
                         skiprows=1)
    elif dataset == "dating":
        columns = ["uid", "iid", "rating"]
        df = pd.read_csv(path+"data/libimseti/ratings.dat", names=columns, sep=",")

    # Checkin data is not appropriate
    # elif dataset == "gowalla":
    #     columns = ["uid", "timestamp", "lat", "lng", "iid"]
    #     df = pd.read_csv(path+"data/gowalla/gowalla.txt.gz", names=columns)

    df.uid = df.uid.astype('category').cat.codes.values
    df.iid = df.iid.astype('category').cat.codes.values

    uNum = df.uid.max() + 1
    iNum = df.iid.max() + 1

    print("#Users: %d, #Items: %d" % (uNum, iNum))

    # Preparing dataset
    train, test = train_test_split(df, test_size=0.3, random_state=1111)

    sample = train
    x_train = [sample['uid'].values, sample['iid'].values]
    y_train = sample.rating.values
    sample = test
    x_test = [sample['uid'].values, sample['iid'].values]
    y_test = sample.rating.values

    # Initialise Model

    if modelName == "mf":
        ranker = MatrixFactorization(uNum, iNum, dim)
        runName = "%s_%s_d%d_%s" % (dataset, modelName, dim,
                                    datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
    elif modelName == "amf":
        ranker = AdversarialMatrixFactorisation(uNum, iNum, dim, weight, pop_percent, 1)
    elif modelName == "amf2":
        ranker = AdversarialMatrixFactorisation(uNum, iNum, dim, weight, pop_percent, 2)
        runName = "%s_%s_d%d_w%f_pp%f_%s" % (dataset, modelName, dim, weight, pop_percent,
                                             datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))

    elif modelName == "neumf":
        ranker = NeuMF(uNum, iNum, dim)
        runName = "%s_%s_d%d_%s" % (dataset, modelName, dim,
                                    datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
    elif modelName == "aneumf":
        ranker = AdversarialNeuMF(uNum, iNum, dim, weight, pop_percent)
        runName = "%s_%s_d%d_w%f_pp%f_%s" % (dataset, modelName, dim, weight, pop_percent,
                                             datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))

    # Trian model

    if "a" not in modelName:

        # history = ranker.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            print(epoch)
            t1 = time()
            for i in range(math.ceil(y_train.shape[0] / batch_size)):
                idx = np.random.randint(0, y_train.shape[0], batch_size)
                _x_train = [x_train[0][idx], x_train[1][idx]]
                _y_train = y_train[idx]
                ranker.model.train_on_batch(_x_train, _y_train)
            t2 = time()
            res = ranker.model.evaluate(x_test, y_test)
            output = "loss: %f, mse: %f, [%f.h]" % (res[0], res[1], (t2 - t1) / 3600)
            print(res)
            with open(path + "out/%s.res" % runName, "a") as myfile:
                myfile.write(output + "\n")
            print(output)

    else:

        ranker.init(x_train, x_test, batch_size)

        for epoch in range(epochs):
            print(epoch)
            t1 = time()
            for i in range(math.ceil(y_train.shape[0] / batch_size)):
                # sample mini-batch
                ranker.train(x_train, y_train, batch_size)

            t2 = time()

            res = ranker.model.evaluate(x_test, y_test)
            output = "loss: %f, mse: %f, [%f.h]" % (res[0], res[1], (t2 - t1) / 3600)
            with open(path + "out/%s.res" % runName, "a") as myfile:
                myfile.write(output + "\n")
            print(output)

            # TODO save results user item score for further analysis and save model when codes are stable
            # history = ranker.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=256)
