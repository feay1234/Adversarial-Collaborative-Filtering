from APL import APL
from APR import *
from Caser import CaserModel
from DRCF import DRCF
from DREAM import DREAM
from Dataset import Dataset, OriginalDataset
from GRU4Rec import GRU4Rec
from NaiveBaselines import MostPopular, MostFrequentlyVisit, MostRecentlyVisit
from NeuMF import NeuMF
from SASRec import SASRec
from utils import *




def parse_args():
    parser = argparse.ArgumentParser(description="Run AMF.")
    parser.add_argument('--path', nargs='?', default='',
                        help='Input data path.')
    parser.add_argument('--opath', nargs='?', default='aaa/',
                        help='Output path.')
    parser.add_argument('--dataset', nargs='?', default='fsq11-sort',
                        help='Choose a dataset.')
    parser.add_argument('--model', type=str,
                        help='Model Name', default="asasrec2")
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
    parser.add_argument('--ckpt', type=int, default=10,
                        help='Save the model per X epochs.')
    parser.add_argument('--task', nargs='?', default='',
                        help='Add the task name for launching experiments')
    parser.add_argument('--adv', nargs='?', default='grad',
                        help='Generate the adversarial sample by gradient method or random method')
    parser.add_argument('--eps', type=float, default=0.5,
                        help='Epsilon for adversarial weights.')
    parser.add_argument('--eval_mode', type=str, default="sample",
                        help='Eval mode: sample or all')
    return parser.parse_args()


if __name__ == '__main__':

    time_stamp = strftime('%Y_%m_%d_%H_%M_%S', localtime())

    # initilize arguments and logging
    args = parse_args()
    init_logging(args, time_stamp)

    # initialize dataset
    dataset = OriginalDataset(args.path + "data/" + args.dataset)

    if args.model == "bpr":
        runName = "%s_%s_d%d_%s" % (args.dataset, args.model, args.embed_size, time_stamp)
        write2file(args.path + "out/" + args.opath, runName + ".out", runName)
        args.adver = 0
        # initialize MF_BPR models
        MF_BPR = MF(dataset.num_users, dataset.num_items, args)
        MF_BPR.build_graph()

        write2file(args.path + "out/" + args.opath, runName + ".out", "Initialize MF_BPR")

        # start training
        training(MF_BPR, dataset, args, runName, epoch_start=0, epoch_end=args.epochs, time_stamp=time_stamp)

    elif args.model == "apr":
        runName = "%s_%s_d%d_e%f_l%f_%s" % (
        args.dataset, args.model, args.embed_size, args.eps, args.reg_adv, time_stamp)
        write2file(args.path + "out/" + args.opath, runName + ".out", runName)

        args.adver = 0
        # initialize MF_BPR models
        MF_BPR = MF(dataset.num_users, dataset.num_items, args)
        MF_BPR.build_graph()

        write2file(args.path + "out/" + args.opath, runName + ".out", "Initialize BPR")

        # start training
        training(MF_BPR, dataset, args, runName, epoch_start=0, epoch_end=args.adv_epoch - 1, time_stamp=time_stamp)

        args.adver = 1
        # instialize AMF model
        AMF = MF(dataset.num_users, dataset.num_items, args)
        AMF.build_graph()

        write2file(args.path + "out/" + args.opath, runName + ".out", "Initialize APR")

        # start training
        training(AMF, dataset, args, runName, epoch_start=args.adv_epoch, epoch_end=args.epochs, time_stamp=time_stamp)

    elif args.model == "apl":
        # TODO first train bpr to 1000
        # then train APL
        pass

    else:
        # initialize the max_ndcg to memorize the best result
        max_ndcg = -1
        best_res = {}

        eval_feed_dicts = init_eval_model(dataset, args)

        runName = "%s_%s_d%d_%s" % (args.dataset, args.model, args.embed_size, time_stamp)
        print(dataset.num_users, dataset.num_items)

        if args.model in ["sasrec", "asasrec", "asasrec2", "asasrec3"]:
            time_stamp = strftime('%Y_%m_%d_%H_%M_%S', localtime())
            args.adver = 0
            # maxlen = int(dataset.df.groupby("uid").size().mean())
            maxlen = min(int(dataset.df.groupby("uid").size().mean()), 50)
            print(maxlen, int(dataset.df.groupby("uid").size().mean()))
            if "asasrec" in args.model:
                runName = "%s_%s_d%d_ml%d_l%.2f_e%.2f_%s" % (
                    args.dataset, args.model, args.embed_size, maxlen, args.reg_adv, args.eps, time_stamp)

            write2file(args.path + "out/" + args.opath, runName + ".out", "Initialize SASREC")
            # ranker = SASRec(dataset.num_users, dataset.num_items, args.embed_size, maxlen, args=args,
            #                 time_stamp=time_stamp)

            # max_ndcg, best_res = run_normal_model(0, args.epochs if args.model == "sasrec" else args.adv_epoch - 1,
            #                                       max_ndcg, best_res, ranker, dataset, args, eval_feed_dicts, runName, time_stamp)

            if "asasrec" in args.model:
                tf.reset_default_graph()
                write2file(args.path + "out/" + args.opath, runName + ".out", "Initialize Adversarial_SASREC")
                args.adver = 1
                ranker = SASRec(dataset.num_users, dataset.num_items, args.embed_size, maxlen, args=args,
                                time_stamp=time_stamp)
                max_ndcg, best_res = run_normal_model(args.adv_epoch, args.epochs, max_ndcg, best_res, ranker, dataset,
                                                      args, eval_feed_dicts, runName, time_stamp)

        elif args.model == "apl":
            ranker = APL(dataset.num_users, dataset.num_items, args.embed_size)
            ranker.init(dataset.trainMatrix)
            # max_ndcg, best_res = run_normal_model(0, args.epochs, max_ndcg, best_res, ranker, dataset)

        elif args.model == "drcf":
            maxlen = 5
            runName = "%s_%s_d%d_ml%d_%s" % (args.dataset, args.model, args.embed_size, maxlen, time_stamp)
            ranker = DRCF(dataset.num_users, dataset.num_items, args.embed_size, maxlen)
            ranker.init(dataset.trainSeq)
            max_ndcg, best_res = run_keras_model(0, args.epochs, max_ndcg, best_res, ranker, args, dataset, eval_feed_dicts, runName)

        elif args.model == "gru4rec":
            ranker = GRU4Rec(dataset.num_users, dataset.num_items, args.embed_size, args.batch_size)
            ranker.init(dataset.df)
            max_ndcg, best_res = run_keras_model(0, args.epochs, max_ndcg, best_res, ranker, args, dataset, eval_feed_dicts, runName)

        elif args.model == "caser":
            isCUDA = torch.cuda.is_available()
            set_seed(2019, cuda=True if isCUDA else False)
            maxlen = min(int(dataset.df.groupby("uid").size().mean()), 50)
            ranker = CaserModel(dataset.num_users, dataset.num_items, args.embed_size, maxlen, isCUDA)
            ranker.init(dataset.df, args.batch_size)
            max_ndcg, best_res = run_keras_model(0, args.epochs, max_ndcg, best_res, ranker, args, dataset, eval_feed_dicts, runName)


        elif args.model == "neumf":
            ranker = NeuMF(dataset.num_users, dataset.num_items, args.embed_size)
            max_ndcg, best_res = run_keras_model(0, args.epochs, max_ndcg, best_res, ranker, args, dataset, eval_feed_dicts, runName)

        elif args.model == "dream":
            maxlen = 10
            ranker = DREAM(dataset.num_users, dataset.num_items, args.embed_size, 10)
            ranker.init(dataset.df)
            max_ndcg, best_res = run_keras_model(0, args.epochs, max_ndcg, best_res, ranker, args, dataset, eval_feed_dicts, runName)

        elif args.model == "pop":
            ranker = MostPopular(dataset.df)
            max_ndcg, best_res = run_keras_model(0,0, max_ndcg, best_res, ranker, args, dataset, eval_feed_dicts, runName)
        elif args.model == "mostvisit":
            ranker = MostFrequentlyVisit(dataset.df)
            max_ndcg, best_res = run_keras_model(0, 0, max_ndcg, best_res, ranker, args, dataset, eval_feed_dicts,
                                                 runName)
        elif args.model == "recentvisit":
            ranker = MostRecentlyVisit(dataset.df)
            max_ndcg, best_res = run_keras_model(0,0, max_ndcg, best_res, ranker, args, dataset, eval_feed_dicts, runName)

        output = "Epoch %d is the best epoch" % best_res['epoch']
        write2file(args.path + "out/" + args.opath, runName + ".out", output)
        for idx, (hr_k, ndcg_k, auc_k) in enumerate(np.swapaxes(best_res['result'], 0, 1)):
            res = "K = %d: HR = %.4f, NDCG = %.4f AUC = %.4f" % (idx + 1, hr_k, ndcg_k, auc_k)
            write2file(args.path + "out/" + args.opath, runName + ".out", res)
