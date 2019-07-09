import numpy as np
import random
import torch

def write2file(path, output):
    print(output)
    thefile = open(path, 'a')
    thefile.write("%s\n" % output)
    thefile.close()

def prediction2file(path, pred):
    thefile = open(path, 'w')
    for item in pred:
        thefile.write("%f\n" % item)
    thefile.close()

def set_seed(seed, cuda=False):

    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)


