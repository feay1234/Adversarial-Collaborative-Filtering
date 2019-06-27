import argparse


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


class History():
    def __init__(self, loss):
        self.history = {'loss':[loss]}