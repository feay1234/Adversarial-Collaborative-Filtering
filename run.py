import argparse
from datetime import datetime
from time import time
import numpy as np
import pandas as pd

from APR import APR
from APL import APL
from BPR import BPR, AdversarialBPR
from Caser import CaserModel
from DRCF import DRCF
from DREAM import DREAM
from Dataset import Dataset, RawDataset, getDataset
from FastAdversarialMF import FastAdversarialMF
from GRU4Rec import GRU4Rec
from IRGAN import IRGAN
from MF import MatrixFactorization, AdversarialMatrixFactorisation
from NeuMF import NeuMF, AdversarialNeuMF
from SASRec import SASRec
from evaluation import evaluate_model
from utils import write2file, prediction2file, set_seed
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Run Adversarial Collaborative Filtering")

    parser.add_argument('--path', type=str, help='Path to data', default="")

    parser.add_argument('--model', type=str,
                        help='Model Name: lstm', default="sasrec")

    parser.add_argument('--data', type=str,
                        help='Dataset name', default="beauty")

    parser.add_argument('--d', type=int, default=64,
                        help='Dimension')

    parser.add_argument('--maxlen', type=int, default=50,
                        help='Maxlen')

    parser.add_argument('--epochs', type=int, default=10,
                        help='Epoch number')

    parser.add_argument('--w', type=float, default=0.001,
                        help='Weight:')

    parser.add_argument('--pp', type=float, default=0.2,
                        help='Popularity Percentage:')

    parser.add_argument('--bs', type=int, default=256,
                        help='Batch Size:')

    parser.add_argument('--pre', type=str, default="",
                        help='Pre-trained dir:')

    parser.add_argument('--mode', type=int, default=1,
                        help='mode')

    # parser.add_argument('--filter', type=int, default=2,
    #                     help='Filter Mode')

    parser.add_argument('--save_model', type=int, default=1,
                        help='Save model')

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
    maxlen = args.maxlen
    pre = args.pre
    mode = args.mode
    save_model = True if args.save_model == 1 else False
    # save_model = False
    # filterMode = args.filter

    # pre = "test_bpr-he_d10.best.h5"

    # num_negatives = 1
    topK = 10
    evaluation_threads = 1

    columns = ["uid", "iid", "rating", "timestamp"]

    # Loading data
    t1 = time()

    dataset = getDataset(data, path)

    train, trainSeq, df, testRatings, testNegatives = dataset.trainMatrix, dataset.trainSeq, dataset.df, dataset.testRatings, dataset.testNegatives
    uNum, iNum = train.shape
    uNum = max(uNum, len(testRatings))
    stat = "Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" % (
        time() - t1, uNum - 1, iNum - 1, len(df),
        len(testRatings))  # user and item index start at 1, not zero, so the exact number of users and items = num - 1

    # Initialise Model

    if modelName == "mf":
        ranker = MatrixFactorization(uNum, iNum, dim)
    elif modelName == "bpr":
        ranker = BPR(uNum, iNum, dim)

    elif modelName == "amf":
        ranker = AdversarialMatrixFactorisation(uNum, iNum, dim, weight, pop_percent)
        x_train, y_train = ranker.get_train_instances(train)
        ranker.init(x_train[0], x_train[1])

    elif modelName == "abpr":
        ranker = AdversarialBPR(uNum, iNum, dim, weight, pop_percent)
        x_train, y_train = ranker.get_train_instances(train)
        ranker.init(x_train[0], x_train[1])


    elif modelName == "amf2":
        ranker = FastAdversarialMF(uNum, iNum, dim, weight, pop_percent)
    elif modelName == "neumf":
        ranker = NeuMF(uNum, iNum, dim)

    elif modelName == "aneumf":
        ranker = AdversarialNeuMF(uNum, iNum, dim, weight, pop_percent)

    elif modelName == "apl":
        # args = parse_apl_args()
        ranker = APL(uNum, iNum, dim)
        ranker.init(train)

    elif modelName == "irgan":
        ranker = IRGAN(uNum, iNum, dim, batch_size)
        ranker.init(train)

    elif modelName == "apr":
        # get APR's default params
        ranker = APR(uNum, iNum, dim)
        ranker.build_graph()

    elif modelName == "bpr-he":
        ranker = APR(uNum, iNum, dim, False)
        ranker.build_graph()

    elif modelName == "sasrec":
        ranker = SASRec(uNum, iNum, dim, maxlen, testNegatives, mode)
        ranker.init(trainSeq)

    elif modelName == "drcf":
        ranker = DRCF(uNum, iNum, dim, maxlen)
        ranker.init(trainSeq)

    elif modelName == "gru4rec":
        ranker = GRU4Rec(uNum, iNum, dim, batch_size)
        ranker.init(df)

    elif modelName == "dream":
        ranker = DREAM(uNum, iNum, dim, maxlen)
        ranker.init(df)
    elif modelName == "caser":
        set_seed(2019, cuda=False)
        ranker = CaserModel(uNum, iNum, dim, maxlen, True)
        ranker.init(df)

    runName = "%s_%s_d%d%s_%s" % (data, modelName, dim, ranker.get_params(),
                                datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
    saveName = "%s_%s_d%d%s" % (data, modelName, dim, ranker.get_params())

    # load pretrained
    # TODO only support BPR-based models
    if pre != "":
        ranker.load_pre_train(path + "h5/" + pre)
        runName = "%s_%s_pre_d%d_%s" % (data, modelName, dim, datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))

    write2file(path + "out/" + runName + ".out", stat)
    write2file(path + "out/" + runName + ".out", runName)
    if pre != "":
        write2file(path + "out/" + runName + ".out", pre)

    # Init performance
    (hits, ndcgs) = evaluate_model(ranker, testRatings, testNegatives,
                                   topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    output = 'Init: HR = %f, NDCG = %f' % (hr, ndcg)
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

        output = 'Iteration %d [%.1f s]: HR = %f, NDCG = %f, loss = %.4f [%.1f s]' % (
            epoch, t2 - t1, hr, ndcg, loss, time() - t2)
        write2file(path + "out/" + runName + ".out", output)


        if ndcg > best_ndcg:

            best_hr, best_ndcg, best_iter = hr, ndcg, epoch
            if save_model:
                ranker.save(path + "h5/" + saveName + ".best.h5")

            # only save result file for the best model
            prediction2file(path + "out/" + runName + ".hr", hits)
            prediction2file(path + "out/" + runName + ".ndcg", ndcgs)

        if math.isnan(loss):
            break

        # save current one
        if save_model:
            ranker.save(path + "h5/" + saveName + ".last.h5")

    output = "End. Best Iteration %d:  HR = %.4f, NDCG = %.4f, Total time = %.2f" % (
        best_iter, best_hr, best_ndcg, (time() - start) / 3600)
    write2file(path + "out/" + runName + ".out", output)
