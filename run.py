import pandas as pd
import numpy as np
from time import time
import math
from sklearn.model_selection import train_test_split
from Models.MatrixFactorisation import MatrixFactorization, AdversarialMatrixFactorisation
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run Adversarial Collaborative Filtering")

    parser.add_argument('--path', type=str, help='Path to data', default="")

    parser.add_argument('--model', type=str,
                        help='Model Name: lstm', default="amf2")

    parser.add_argument('--data', type=str,
                        help='Dataset name', default="TREC")

    parser.add_argument('--d', type=int, default=10,
                        help='Dimension')

    parser.add_argument('--epochs', type=int, default=50,
                        help='Epoch number')

    parser.add_argument('--w', type=float, default=0.1,
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
    df = pd.read_csv("../Adversarial-Collaborative-Filtering/data/ml-latest-small/ratings.csv", names=columns,
                     skiprows=1)

    df.uid = df.uid.astype('category').cat.codes.values
    df.iid = df.iid.astype('category').cat.codes.values

    uNum = df.uid.max() + 1
    iNum = df.iid.max() + 1

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
    elif modelName == "amf":
        ranker = AdversarialMatrixFactorisation(uNum, iNum, dim, weight, pop_percent, 1)
    elif modelName == "amf2":
        ranker = AdversarialMatrixFactorisation(uNum, iNum, dim, weight, pop_percent, 2)

    # Trian model

    if "a" not in modelName:

        history = ranker.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=256)

    else:

        #
        # if "2" in modelName:
        popular_user_x, popular_user_y, rare_user_x, rare_user_y = ranker.get_discriminator_train_data(x_train[0],
                                                                                                       x_test[0],
                                                                                                       batch_size)
        popular_item_x, popular_item_y, rare_item_x, rare_item_y = ranker.get_discriminator_train_data(x_train[1],
                                                                                                       x_test[1],
                                                                                                       batch_size)

        for epoch in range(epochs):

            for i in range(math.ceil(y_train.shape[0] / batch_size)):
                # sample mini-batch
                idx = np.random.randint(0, y_train.shape[0], batch_size)
                _x_train = [x_train[0][idx], x_train[1][idx]]
                _y_train = y_train[idx]

                # sample mini-batch for User Discriminator

                idx = np.random.randint(0, len(popular_user_x), batch_size)
                _popular_user_x = popular_user_x[idx]

                idx = np.random.randint(0, len(rare_user_x), batch_size)
                _rare_user_x = rare_user_x[idx]

                _popular_user_x = ranker.uEncoder.predict(_popular_user_x)
                _rare_user_x = ranker.uEncoder.predict(_rare_user_x)

                d_loss_popular_user = ranker.discriminator_u.train_on_batch(_popular_user_x, popular_user_y)
                d_loss_rare_user = ranker.discriminator_u.train_on_batch(_rare_user_x, rare_user_y)

                # sample mini-batch for Item Discriminator

                idx = np.random.randint(0, len(popular_item_x), batch_size)
                _popular_item_x = popular_item_x[idx]

                idx = np.random.randint(0, len(rare_item_x), batch_size)
                _rare_item_x = rare_item_x[idx]

                _popular_item_x = ranker.iEncoder.predict(_popular_item_x)
                _rare_item_x = ranker.iEncoder.predict(_rare_item_x)

                d_loss_popular_item = ranker.discriminator_i.train_on_batch(_popular_item_x, popular_item_y)
                d_loss_rare_item = ranker.discriminator_i.train_on_batch(_rare_item_x, rare_item_y)

                # Discriminator's loss
                d_loss = 0.5 * np.add(d_loss_popular_user, d_loss_rare_user) + 0.5 * np.add(d_loss_popular_item,
                                                                                            d_loss_rare_item)

                # Sample mini-batch for adversarial model

                idx = np.random.randint(0, len(popular_user_x), int(batch_size / 2))
                _popular_user_x = popular_user_x[idx]

                idx = np.random.randint(0, len(rare_user_x), int(batch_size / 2))
                _rare_user_x = rare_user_x[idx]

                idx = np.random.randint(0, len(popular_item_x), int(batch_size / 2))
                _popular_item_x = popular_item_x[idx]

                idx = np.random.randint(0, len(rare_item_x), int(batch_size / 2))
                _rare_item_x = rare_item_x[idx]

                _popular_rare_user_x = np.concatenate([_popular_user_x, _rare_user_x])
                _popular_rare_item_x = np.concatenate([_popular_item_x, _rare_item_x])

                _popular_rare_y = np.concatenate([np.zeros(int(batch_size / 2)), np.ones(int(batch_size / 2))])


                # Train adversarial model
                g_loss = ranker.advModel.train_on_batch(_x_train + [_popular_rare_user_x, _popular_rare_item_x],
                                                        [_y_train, _popular_rare_y, _popular_rare_y])

            res = ranker.model.evaluate(x_test, y_test)
            print(res)

            # history = ranker.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=256)
