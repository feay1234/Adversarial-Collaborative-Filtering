import pandas as pd
import sys

# convert He's dataset to seq-based dataset
def process_data(path, data):
    # names = ["uid", "iid", "rating", "timestamp"]
    names = ["uid", "iid", "rating", "hour", "day", "datetime"]
    train = pd.read_csv(path + "data/%s.train.rating" % data, sep="\t", names=names)
    test = pd.read_csv(path + "data/%s.test.rating" % data, sep="\t", names=names)
    df = train.append(test)

    # sort interactions for each user
    # df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values(['uid', 'datetime'])
    save2file(df, path, data, "-sort")

    # remove duplicate interactions for each user
    df_dup = df.drop_duplicates(['uid', 'iid'])
    df_dup = df_dup.sort_values(['uid', 'datetime'])
    save2file(df_dup, path, data, "-sort-dup")


def save2file(df, path, data, name):

    train = df.groupby("uid", as_index=False).apply(lambda x: x.iloc[:-1])
    test = df.groupby("uid").tail(1)

    assert len(df) == len(train) + len(test)

    # train[['uid', 'iid', 'rating', 'timestamp']].to_csv(path + "data/%s.train.rating" % (data+name), index=False, header=False, sep="\t")
    # test[['uid', 'iid', 'rating', 'timestamp']].to_csv(path + "data/%s.test.rating" % (data+name), index=False, header=False, sep="\t")

    train[['uid', 'iid', 'rating', 'datetime']].to_csv(path + "data/%s.train.rating" % (data+name), index=False, header=False, sep="\t")
    test[['uid', 'iid', 'rating', 'datetime']].to_csv(path + "data/%s.test.rating" % (data+name), index=False, header=False, sep="\t")

# print(sys.argv[1])
# print(sys.argv[2])
process_data(sys.argv[1], sys.argv[2])