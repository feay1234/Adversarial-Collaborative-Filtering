import argparse

import pandas as pd

parser = argparse.ArgumentParser(description="Pre-processing data")
parser.add_argument('--inp', type=str, help='Path to data', default="")
parser.add_argument('--out', type=str, help='Path to data', default="")
args = parser.parse_args()

inp = args.inp
out = args.out
columns = ["uid", "timestamp", "lat", "lng", "iid"]

df = pd.read_csv(inp, names=columns, sep="\t")
tmp = df.groupby("iid").filter(lambda x: len(x) >= 10)
tmp.to_csv(out, index=False, sep="\t", header=False)
