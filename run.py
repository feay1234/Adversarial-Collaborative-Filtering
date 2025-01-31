import argparse
from datetime import datetime
from time import time
import numpy as np

from APR import APR
from APL import APL
from BPR import BPR, AdversarialBPR
from Caser import CaserModel
from DRCF import DRCF
from DREAM import DREAM, DREAM_TF
from utils import getDataset
from FastAdversarialMF import FastAdversarialMF
from GRU4Rec import GRU4Rec
from IRGAN import IRGAN
from MF import MatrixFactorization, AdversarialMatrixFactorisation
from NaiveBaselines import MostPopular, AlreadyVisit, MostFrequentlyVisit, MostRecentlyVisit
from NeuMF import NeuMF, AdversarialNeuMF
from SASRec import SASRec
from evaluation import evaluate_model
from utils import write2file, prediction2file, set_seed
import math


def parse_args():
    parser = argparse.ArgumentParser(description="Run Adversarial Collaborative Filtering")

    parser.add_argument('--path', type=str, help='Path to data', default="")

    parser.add_argument('--opath', type=str, help='Path to output', default="test/")

    parser.add_argument('--model', type=str,
                        help='Model Name: lstm', default="bpr")

    parser.add_argument('--data', type=str,
                        help='Dataset name', default="test")

    parser.add_argument('--d', type=int, default=64,
                        help='Dimension')

    parser.add_argument('--verbose_eval', type=int, default=1,
                        help='Evaluate per X epochs.')

    parser.add_argument('--eval', type=str, default="all", help="DRCF evaluation mode or APR evaluation mode")

    parser.add_argument('--maxlen', type=int, default=10,
                        help='Maxlen')

    parser.add_argument('--epochs', type=int, default=10,
                        help='Epoch number')
    parser.add_argument('--adv_epochs', type=int, default=5,
                        help='Adversarial Epoch number')

    parser.add_argument('--w', type=float, default=0.001,
                        help='Weight:')

    parser.add_argument('--pp', type=float, default=0.2,
                        help='Popularity Percentage:')

    parser.add_argument('--bs', type=int, default=512,
                        help='Batch Size:')

    parser.add_argument('--pre', type=str, default="",
                        help='Pre-trained dir:')

    parser.add_argument('--mode', type=int, default=0,
                        help='mode')

    parser.add_argument('--ckpt', type=int, default=1,
                        help='Save the model per X epochs.')

    parser.add_argument('--save_model', type=int, default=1,
                        help='Save model')

    return parser.parse_args()


if __name__ == '__main__':
    start = time()

    args = parse_args()

    path = args.path
    opath = args.opath
    data = args.data
    modelName = args.model
    dim = args.d
    weight = args.w
    pop_percent = args.pp
    batch_size = args.bs
    epochs = args.epochs
    adv_epochs = args.adv_epochs
    maxlen = args.maxlen
    pre = args.pre
    mode = args.mode
    evalMode = args.eval
    verbose_eval = args.verbose_eval
    save_model = True if args.save_model == 1 else False
    # save_model = False
    # filterMode = args.filter

    # pre = "test_bpr-he_d10.last.h5"

    # num_negatives = 1
    topK = 100 if evalMode == "all" else 10
    evaluation_threads = 1

    # Loading data
    t1 = time()

    dataset = getDataset(data, path, evalMode)

    train, trainSeq, df, testRatings, testNegatives = dataset.trainMatrix, dataset.trainSeq, dataset.df, dataset.testRatings, dataset.testNegatives
    uNum, iNum = df.uid.max()+1, df.iid.max()+1
    # uNum = max(uNum, len(testRatings))

    stat = "Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" % (
        time() - t1, uNum , iNum , len(df),
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

    elif modelName in ["bpr-tf", "apr"]:
        ranker = APR(uNum, iNum, dim, False)
        runName = "%s_%s_d%d%s_%s" % (data, modelName, dim, ranker.get_params(),
                                      datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
        ranker.build_graph(path, opath, data, runName)

    elif modelName == "sasrec":
        # use mean
        maxlen = int(df.groupby("uid").size().mean())
        ranker = SASRec(uNum, iNum, dim, maxlen)
        ranker.init(trainSeq, batch_size)

    elif modelName == "drcf":
        ranker = DRCF(uNum, iNum, dim, maxlen)
        ranker.init(trainSeq)

    elif modelName == "gru4rec":
        ranker = GRU4Rec(uNum, iNum, dim, batch_size)
        ranker.init(df)

    elif modelName == "dream":
        ranker = DREAM(uNum, iNum, dim, maxlen)
        ranker.init(df)

    elif modelName == "dream-tf":
        ranker = DREAM_TF(uNum, iNum, dim, maxlen)
        ranker.init(df)

    elif modelName == "caser":
        set_seed(2019, cuda=False)
        maxlen = int(df.groupby("uid").size().mean())
        ranker = CaserModel(uNum, iNum, dim, maxlen, False)
        ranker.init(df, batch_size)

    elif modelName == "pop":
        ranker = MostPopular(df)

    elif modelName == "mrv":
        ranker = MostRecentlyVisit(df)

    elif modelName == "mfv":
        ranker = MostFrequentlyVisit(df)

    elif modelName == "av":
        ranker = AlreadyVisit(train)


    runName = "%s_%s_d%d%s_%s" % (data, modelName, dim, ranker.get_params(),
                                  datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
    saveName = "%s_%s_d%d%s" % (data, modelName, dim, ranker.get_params())

    # load pretrained
    # TODO only support BPR-based models
    if pre != "":
        ranker.load_pre_train(path + "h5/" + pre)
        runName = "%s_%s_%s.%s_d%d_%s" % (
            data, modelName, pre.split("_")[1], pre.split(".")[1], dim, datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))

    write2file(path + "out/" + opath, runName + ".out", stat)
    write2file(path + "out/" + opath, runName + ".out", runName)
    if pre != "":
        write2file(path + "out/" + opath, runName + ".out", pre)

    # Init performance
    (hits, ndcgs) = evaluate_model(ranker, testRatings, testNegatives, topK, evaluation_threads)

    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    # hr, ndcg = 0, 0
    output = 'Init: HR = %f, NDCG = %f' % (hr, ndcg)
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    write2file(path + "out/" + opath, runName + ".out", output)

    start = time()
    # Training model
    for epoch in range(epochs):

        if modelName == "apr":
            if epoch == adv_epochs:
                ranker = APR(uNum, iNum, dim, True)
                ranker.build_graph(path, opath, data, runName, True)



        t1 = time()
        # Generate training instances
        x_train, y_train = ranker.get_train_instances(train)

        loss = ranker.train(x_train, y_train, batch_size)
        t2 = time()

        if epoch % verbose_eval == 0:
            (hits, ndcgs) = evaluate_model(ranker, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()



        output = 'Iteration %d [%.1f s]: HR = %f, NDCG = %f, loss = %.4f [%.1f s]' % (
            epoch, t2 - t1, hr, ndcg, loss, time() - t2)
        write2file(path + "out/" + opath, runName + ".out", output)

        if ndcg > best_ndcg:

            best_hr, best_ndcg, best_iter = hr, ndcg, epoch
            if save_model:
                ranker.save(path + "h5/" + saveName + ".best.h5")

            # only save result file for the best model
            prediction2file(path + "out/" + opath, runName + ".hr", hits)
            prediction2file(path + "out/" + opath, runName + ".ndcg", ndcgs)

        if math.isnan(loss):
            break

        # save current one
        if save_model:
            ranker.save(path + "h5/" + saveName + ".last.h5")

        # we only need 1 epoch for naive baselines
        if modelName in ["pop", "mrv", "mfv", "av"]:
            break

    output = "End. Best Iteration %d:  HR = %.4f, NDCG = %.4f, Total time = %.2f" % (
        best_iter, best_hr, best_ndcg, (time() - start) / 3600)
    write2file(path + "out/" + opath, runName + ".out", output)
