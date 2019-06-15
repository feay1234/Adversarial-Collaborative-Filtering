import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from Models.MatrixFactorisation import MatrixFactorization

columns = ["uid", "iid", "rating", "timestamp"]
df = pd.read_csv("../Adversarial-Collaborative-Filtering/data/ml-latest-small/ratings.csv", names = columns , skiprows=1)

df.uid = df.uid.astype('category').cat.codes.values
df.iid = df.iid.astype('category').cat.codes.values

uNum = df.uid.max() + 1
iNum = df.iid.max() + 1


train, test = train_test_split(df, test_size=0.3, random_state=1111)


sample = train
x_train = [sample['uid'].values, sample['iid'].values]
y_train = sample.rating.values
sample = test
x_test = [sample['uid'].values, sample['iid'].values]
y_test = sample.rating.values

mf = MatrixFactorization(uNum, iNum, 10)

history = mf.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=256)