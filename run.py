import argparse
from datetime import datetime
from time import time
import numpy as np
import pandas as pd

from APR import APR
from APL import APL
from BPR import BPR, AdversarialBPR
from Dataset import Dataset, RawDataset
from FastAdversarialMF import FastAdversarialMF
from MF import MatrixFactorization, AdversarialMatrixFactorisation
from NeuMF import NeuMF, AdversarialNeuMF
from evaluation import evaluate_model
from utils import write2file, prediction2file


def parse_args():
    parser = argparse.ArgumentParser(description="Run Adversarial Collaborative Filtering")

    parser.add_argument('--path', type=str, help='Path to data', default="")

    parser.add_argument('--model', type=str,
                        help='Model Name: lstm', default="apl")

    parser.add_argument('--data', type=str,
                        help='Dataset name', default="pinterest-20")

    parser.add_argument('--d', type=int, default=10,
                        help='Dimension')

    parser.add_argument('--epochs', type=int, default=200,
                        help='Epoch number')

    parser.add_argument('--w', type=float, default=0.001,
                        help='Weight:')

    parser.add_argument('--pp', type=float, default=0.2,
                        help='Popularity Percentage:')

    parser.add_argument('--bs', type=int, default=512,
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
    # pre = "pinterest-20_bpr_d10_06-27-2019_11-43-42.last.h5"

    # num_negatives = 1
    topK = 10
    evaluation_threads = 1

    columns = ["uid", "iid", "rating", "timestamp"]

    # Loading data
    t1 = time()

    if data in ["ml-1m", "yelp", "pinterest-20"]:
        dataset = Dataset(path + "data/" + data)

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
        df = pd.read_csv(path + "data/gowalla.csv", names=columns, sep="\t")
        dataset = RawDataset(df)



    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    uNum, iNum = train.shape
    stat = "Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" % (time() - t1, uNum, iNum, train.nnz, len(testRatings))

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

    elif modelName == "apl":
        # args = parse_apl_args()
        ranker = APL(uNum, iNum, dim)
        runName = "%s_%s_d%d_%s" % (data, modelName, dim,
                                    datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
        ranker.init(train)

    elif modelName == "apr":
        # get APR's default params
        ranker = APR(uNum, iNum, dim)
        ranker.build_graph()
        runName = "%s_%s_d%d_%s" % (data, modelName, dim,
                                    datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))

    # load pretrained
    # TODO only support BPR-based models
    if pre != "":
        ranker.load_pre_train(path+"h5/"+pre)
        runName = "%s_%s_pre_d%d_%s" % (data, modelName, dim, datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))

    write2file(path + "out/" + runName + ".out", stat)
    write2file(path + "out/" + runName + ".out", runName)

    # Init performance
    (hits, ndcgs) = evaluate_model(ranker, testRatings, testNegatives,
                                   topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    output = 'Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg)
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    write2file(path + "out/" + runName + ".out", output)


    start = time()
    # Training model
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        x_train, y_train = ranker.get_train_instances(train)

        loss = ranker.train(x_train, y_train, batch_size)
        t2 = time()

        (hits, ndcgs) = evaluate_model(ranker, testRatings,
                                       testNegatives, topK, evaluation_threads)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        # hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist[0]

        output = 'Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' % (
            epoch, t2 - t1, hr, ndcg, loss, time() - t2)
        write2file(path + "out/" + runName + ".out", output)

        # TODO each mode save, TF and Keras
        if ndcg > best_ndcg:
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch
            ranker.save(path + "h5/" + runName + ".best.h5")

        # only save result file for the best model
        prediction2file(path + "out/" + runName + ".hr", hits)
        prediction2file(path + "out/" + runName + ".ndcg", ndcgs)
        # save current one
        ranker.save(path + "h5/" + runName + ".last.h5")

    output = "End. Best Iteration %d:  HR = %.4f, NDCG = %.4f, Total time = %.2f" % (best_iter, best_hr, best_ndcg, (time() - start) / 3600)
    write2file(path + "out/" + runName + ".out", output)



