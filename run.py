import argparse
import math
from datetime import datetime
from time import time
from keras.models import load_model

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from BPR import BPR, AdversarialBPR
from Dataset import Dataset, RawDataset
from FastAdversarialMF import FastAdversarialMF
from MatrixFactorisation import MatrixFactorization, AdversarialMatrixFactorisation
from NeuMF import NeuMF, AdversarialNeuMF
from evaluation import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run Adversarial Collaborative Filtering")

    parser.add_argument('--path', type=str, help='Path to data', default="")

    parser.add_argument('--model', type=str,
                        help='Model Name: lstm', default="abpr")

    parser.add_argument('--data', type=str,
                        help='Dataset name', default="ml-small")

    parser.add_argument('--d', type=int, default=10,
                        help='Dimension')

    parser.add_argument('--epochs', type=int, default=10,
                        help='Epoch number')

    parser.add_argument('--w', type=float, default=0.001,
                        help='Weight:')

    parser.add_argument('--pp', type=float, default=0.2,
                        help='Popularity Percentage:')

    parser.add_argument('--bs', type=int, default=32,
                        help='Batch Size:')

    parser.add_argument('--pre', type=str, default="",
                        help='Pre-trained dir:')

    return parser.parse_args()


if __name__ == '__main__':
    start = time()

    args = parse_args()

    path = args.path
    data = args.data
    modelName = args.model
    dim = args.d
    weight = args.w
    pop_percent = args.pp
    batch_size = args.bs
    epochs = args.epochs
    pre = args.pre
    # pre = "h5/ml-small_bpr_d10_06-20-2019_14-49-38.h5"

    # num_negatives = 1
    topK = 10
    evaluation_threads = 1

    columns = ["uid", "iid", "rating", "timestamp"]

    # Loading data
    t1 = time()

    if data == "ml-1m":
        dataset = Dataset(path + "data/" + data)

    elif data == "ml-small":
        df = pd.read_csv(path + "data/ml-latest-small/ratings.csv", names=columns,
                         skiprows=1)
        dataset = RawDataset(df)
        # train, testRatings, testNegatives =
    elif data == "ml":
        df = pd.read_csv(path + "data/ml-20m/ratings.csv", names=columns,
                         skiprows=1)
        dataset = RawDataset(df)
    elif data == "dating":
        columns = ["uid", "iid", "rating"]
        df = pd.read_csv(path + "data/libimseti/ratings.dat", names=columns, sep=",")
    elif data == "brightkite":
        columns = ["uid", "timestamp", "lat", "lng", "iid"]
        df = pd.read_csv(path + "data/brightkite.txt", names=columns, sep="\t")
        dataset = RawDataset(df)
    elif data == "gowalla":
        columns = ["uid", "timestamp", "lat", "lng", "iid"]
        df = pd.read_csv(path + "data/gowalla.txt", names=columns, sep="\t")
        dataset = RawDataset(df)



    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    uNum, iNum = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, uNum, iNum, train.nnz, len(testRatings)))

    # Checkin data is not appropriate
    # elif data == "gowalla":
    #     columns = ["uid", "timestamp", "lat", "lng", "iid"]
    #     df = pd.read_csv(path+"data/gowalla/gowalla.txt.gz", names=columns)

    # df.uid = df.uid.astype('category').cat.codes.values
    # df.iid = df.iid.astype('category').cat.codes.values
    #
    # uNum = df.uid.max() + 1
    # iNum = df.iid.max() + 1

    # print("#Users: %d, #Items: %d" % (uNum, iNum))
    #
    # Preparing dataset
    # train, test = train_test_split(df, test_size=0.3, random_state=1111)
    #
    # sample = train
    # x_train = [sample['uid'].values, sample['iid'].values]
    # y_train = sample.rating.values
    # sample = test
    # x_test = [sample['uid'].values, sample['iid'].values]
    # y_test = sample.rating.values

    # Initialise Model

    if modelName == "mf":
        ranker = MatrixFactorization(uNum, iNum, dim)
        runName = "%s_%s_d%d_%s" % (data, modelName, dim,
                                    datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
    elif modelName == "bpr":
        ranker = BPR(uNum, iNum, dim)
        runName = "%s_%s_d%d_%s" % (data, modelName, dim,
                                    datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))

    elif modelName == "amf":
        ranker = AdversarialMatrixFactorisation(uNum, iNum, dim, weight, pop_percent)
        runName = "%s_%s_d%d_w%f_pp%f_%s" % (data, modelName, dim, weight, pop_percent,
                                             datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
        x_train, y_train = ranker.get_train_instances(train)
        ranker.init(x_train[0], x_train[1])

    elif modelName == "abpr":
        ranker = AdversarialBPR(uNum, iNum, dim, weight, pop_percent)
        runName = "%s_%s_d%d_w%f_pp%f_%s" % (data, modelName, dim, weight, pop_percent,
                                             datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
        x_train, y_train = ranker.get_train_instances(train)
        ranker.init(x_train[0], x_train[1])


    elif modelName == "amf2":
        ranker = FastAdversarialMF(uNum, iNum, dim, weight, pop_percent)
        runName = "%s_%s_d%d_w%f_pp%f_%s" % (data, modelName, dim, weight, pop_percent,
                                             datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
    elif modelName == "neumf":
        ranker = NeuMF(uNum, iNum, dim)
        runName = "%s_%s_d%d_%s" % (data, modelName, dim,
                                    datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))

    elif modelName == "aneumf":
        ranker = AdversarialNeuMF(uNum, iNum, dim, weight, pop_percent)
        runName = "%s_%s_d%d_w%f_pp%f_%s" % (data, modelName, dim, weight, pop_percent,
                                             datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
    # load pretrained
    # TODO only support BPR-based models
    if pre != "":
        pretrainModel = load_model(pre)
        runName = "%s_%s_pre_d%d_w%f_pp%f_%s" % (data, modelName, dim, weight, pop_percent,
                                                 datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
        ranker.predictor.get_layer("uEmb").set_weights(pretrainModel.get_layer("uEmb").get_weights())
        ranker.predictor.get_layer("iEmb").set_weights(pretrainModel.get_layer("iEmb").get_weights())

    print(runName)

    isAdvModel = ["amf", "aneumf", "abpr"]
    isPairwiseModel = True if modelName in ["bpr", "abpr"] else False

    # Init performance
    (hits, ndcgs) = evaluate_model(ranker.predictor if isPairwiseModel else ranker.model, testRatings, testNegatives,
                                   topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1

    # Training model
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        x_train, y_train = ranker.get_train_instances(train)

        if modelName in isAdvModel:

            if isPairwiseModel:
                for i in range(math.ceil(len(y_train) / batch_size)):
                    _u = x_train[0][i * batch_size:(i * batch_size) + batch_size]
                    _p = x_train[1][i * batch_size:(i * batch_size) + batch_size]
                    _n = x_train[2][i * batch_size:(i * batch_size) + batch_size]
                    _labels = y_train[i * batch_size: (i * batch_size) + batch_size]
                    _batch_size = _u.shape[0]

                    hist = ranker.train([_u, _p, _n], _labels, _batch_size)

            else:

                # for i in tqdm(range(math.ceil(len(labels) / batch_size))):
                for i in range(math.ceil(len(y_train) / batch_size)):
                    _u = x_train[0][i * batch_size:(i * batch_size) + batch_size]
                    _i = x_train[1][i * batch_size:(i * batch_size) + batch_size]
                    _labels = y_train[i * batch_size: (i * batch_size) + batch_size]
                    _batch_size = _u.shape[0]
                    #
                    hist = ranker.train([_u, _i], _labels, _batch_size)
                    #
                    #
                    # ranker.init(user_input, item_input)
                    # hist = ranker.train2([np.array(user_input), np.array(item_input)],  # input
                    #                         np.array(labels), batch_size)
        else:
            # Training
            hist = ranker.model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()

        (hits, ndcgs) = evaluate_model(ranker.predictor if isPairwiseModel else ranker.model, testRatings,
                                       testNegatives, topK, evaluation_threads)
        hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
        # hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist[0]

        output = 'Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' % (
            epoch, t2 - t1, hr, ndcg, loss, time() - t2)
        print(output)

        thefile = open(path + "out/" + runName + ".out", 'a')
        thefile.write("%s\n" % output)
        thefile.close()

        if ndcg > best_ndcg:
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch
            ranker.model.save(path + "h5/" + runName + ".h5", overwrite=True)

        # only save result file for the best model
        thefile = open(path + "out/" + runName + ".hr", 'w')
        for item in hits:
            thefile.write("%f\n" % item)
        thefile.close()

        thefile = open(path + "out/" + runName + ".ndcg", 'w')
        for item in ndcgs:
            thefile.write("%f\n" % item)
        thefile.close()

    output = "End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg)
    print(output)
    thefile = open(path + "out/" + runName + ".out", 'a')
    thefile.write("%s\n" % output)
    thefile.close()


    # Trian model
    #
    # if "a" not in modelName:
    #
    #     # history = ranker.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, shuffle=True)
    #
    #     for epoch in range(epochs):
    #         print(epoch)
    #         t1 = time()
    #         for i in range(math.ceil(y_train.shape[0] / batch_size)):
    #             idx = np.random.randint(0, y_train.shape[0], batch_size)
    #             _x_train = [x_train[0][idx], x_train[1][idx]]
    #             _y_train = y_train[idx]
    #             ranker.model.train_on_batch(_x_train, _y_train)
    #         t2 = time()
    #         res = ranker.model.evaluate(x_test, y_test)
    #         output = "loss: %.4f, mse: %.4f, [%.2f.h]" % (res[0], res[1], (t2 - t1) / 3600)
    #         print(res)
    #         with open(path + "out/%s.res" % runName, "a") as myfile:
    #             myfile.write(output + "\n")
    #         print(output)
    #
    # else:
    #
    #     ranker.init(x_train, x_test, batch_size)
    #
    #     for epoch in range(epochs):
    #         print(epoch)
    #         t1 = time()
    #         for i in range(math.ceil(y_train.shape[0] / batch_size)):
    #             # sample mini-batch
    #             ranker.train(x_train, y_train, batch_size)
    #
    #         t2 = time()
    #
    #         res = ranker.model.evaluate(x_test, y_test)
    #         output = "loss: %.4f, mse: %.4f, [%.2f.h]" % (res[0], res[1], (t2 - t1) / 3600)
    #         with open(path + "out/%s.res" % runName, "a") as myfile:
    #             myfile.write(output + "\n")
    #         print(output)
    #
    #         # TODO save results user item score for further analysis and save model when codes are stable
    #         # history = ranker.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=256)
