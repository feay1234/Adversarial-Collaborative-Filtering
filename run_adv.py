import argparse

from SASRec import SASRec
from evaluation_adv import training, MF, sampling, init_eval_model, shuffle
from utils import write2file, prediction2file
from BPR import BPR
from Dataset import HeDataset
from time import time
from time import strftime
from time import localtime
import math
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Run AMF.")
    parser.add_argument('--path', nargs='?', default='',
                        help='Input data path.')
    parser.add_argument('--model', type=str,
                        help='Model Name', default="apr")
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Evaluate per X epochs.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs.')
    parser.add_argument('--adv_epoch', type=int, default=1,
                        help='Add APR in epoch X, when adv_epoch is 0, it\'s equivalent to pure AMF.\n '
                             'And when adv_epoch is larger than epochs, it\'s equivalent to pure MF model. ')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--dns', type=int, default=1,
                        help='number of negative sample for each positive in dns.')
    parser.add_argument('--reg', type=float, default=0,
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--reg_adv', type=float, default=1,
                        help='Regularization for adversarial loss')
    parser.add_argument('--restore', type=str, default=None,
                        help='The restore time_stamp for weights in \Pretrain')
    parser.add_argument('--ckpt', type=int, default=1,
                        help='Save the model per X epochs.')
    parser.add_argument('--task', nargs='?', default='',
                        help='Add the task name for launching experiments')
    parser.add_argument('--adv', nargs='?', default='grad',
                        help='Generate the adversarial sample by gradient method or random method')
    parser.add_argument('--eps', type=float, default=0.5,
                        help='Epsilon for adversarial weights.')
    return parser.parse_args()

if __name__ == '__main__':

    time_stamp = strftime('%Y_%m_%d_%H_%M_%S', localtime())

    # initilize arguments and logging
    args = parse_args()

    # initialize dataset
    if args.dataset in ["brightkite", "fsq11", "yelp"]:
        dataset = HeDataset(args.path + "data/" + args.dataset, mode=1)
    elif args.dataset in ["ml-1m", "pinterest-20"]:
        dataset = HeDataset(args.path + "data/" + args.dataset)
    elif args.dataset == "yelp-he":
        dataset = HeDataset(args.path + "data/yelp")

    print(dataset.num_users, dataset.num_items, len(dataset.testNegatives))

    if args.model == "bpr":
        runName = "%s_%s_d%d_%s" % (args.dataset, args.model, args.embed_size, time_stamp)
        write2file(args.path + "out/" + runName + ".out", runName)
        args.adver = 0
        # initialize MF_BPR models
        MF_BPR = MF(dataset.num_users, dataset.num_items, args)
        MF_BPR.build_graph()

        write2file(args.path + "out/" + runName + ".out", "Initialize MF_BPR")

        # start training
        training(MF_BPR, dataset, args, runName, epoch_start=0, epoch_end=args.epochs, time_stamp=time_stamp)

    elif args.model == "apr":
        runName = "%s_%s_d%d_e%f_l%f_%s" % (args.dataset, args.model, args.embed_size, args.eps, args.reg_adv, time_stamp)
        write2file(args.path + "out/" + runName + ".out", runName)

        args.adver = 0
        # initialize MF_BPR models
        MF_BPR = MF(dataset.num_users, dataset.num_items, args)
        MF_BPR.build_graph()

        write2file(args.path + "out/" + runName + ".out", "Initialize BPR")

        # start training
        training(MF_BPR, dataset, args, runName, epoch_start=0, epoch_end=args.adv_epoch - 1, time_stamp=time_stamp)

        args.adver = 1
        # instialize AMF model
        AMF = MF(dataset.num_users, dataset.num_items, args)
        AMF.build_graph()

        write2file(args.path + "out/" + runName + ".out", "Initialize APR")

        # start training
        training(AMF, dataset, args, runName, epoch_start=args.adv_epoch, epoch_end=args.epochs, time_stamp=time_stamp)

    else:
        runName = "%s_%s_d%d_%s" % (args.dataset, args.model, args.embed_size, time_stamp)
        if args.model == "bpe-keras":
            ranker = BPR(dataset.num_users, dataset.num_items, args.embed_size)
        elif args.model == "sasrec":

            maxlen = int(dataset.df.groupby("uid").size().mean())
            ranker = SASRec(dataset.num_users, dataset.num_items, args.embed_size, maxlen, dataset.testNegatives)
            ranker.init(dataset.trainSeq, args.batch_size)



        # samples = sampling(dataset)
        # samples = ()

        eval_feed_dicts = init_eval_model(ranker, dataset)

        # initialize the max_ndcg to memorize the best result
        max_ndcg = 0
        best_res = {}

        # train by epoch
        for epoch_count in range(args.epochs):

            # initialize for training batches
            # batches = shuffle(samples, args.batch_size, dataset, ranker)

            # user_input, item_input_pos, user_dns_list, item_dns_list = batches
            # item_input_neg = item_dns_list

            # for i in range(len(user_input)):
            #     hist = ranker.model.fit([user_input[i], item_input_pos[i], item_input_neg[i]], np.ones(len(user_input[i])), batch_size=args.batch_size, epochs=1, verbose=0)

            x_train, y_train = ranker.get_train_instances(dataset.trainMatrix)

            loss = ranker.train(x_train, y_train, args.batch_size)

            if epoch_count % args.verbose == 0:

                res = []
                for user in range(dataset.num_users):
                    user_input, item_input = eval_feed_dicts[user]
                    # u = np.full(len(item_input), user, dtype='int32')[:, None]
                    # predictions = ranker.rank(u , item_input)
                    # print(u)
                    predictions = ranker.rank(user_input[0], item_input)

                    neg_predict, pos_predict = predictions[:-1], predictions[-1]
                    position = (neg_predict >= pos_predict).sum()

                    # calculate from HR@1 to HR@100, and from NDCG@1 to NDCG@100, AUC
                    hr, ndcg, auc = [], [], []
                    K = 100
                    for k in range(1, K + 1):
                        hr.append(position < k)
                        ndcg.append(math.log(2) / math.log(position + 2) if position < k else 0)
                        auc.append(1 - (
                        position / len(neg_predict)))  # formula: [#(Xui>Xuj) / #(Items)] = [1 - #(Xui<=Xuj) / #(Items)]


                    res.append((hr,ndcg,auc))
                res = np.array(res)
                hr, ndcg, auc = (res.mean(axis=0)).tolist()
                hr, ndcg, auc = np.swapaxes((hr,ndcg, auc), 0, 1)[-1]
                res = "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f ACC = %.4f ACC_adv = %.4f [%.1fs], |P|=%.2f, |Q|=%.2f" % \
                      (epoch_count, 0, 0, hr, ndcg, loss,
                       0, 0, 0, 0)

                write2file(args.path + "out/" + runName + ".out", res)






