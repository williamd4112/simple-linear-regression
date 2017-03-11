import numpy as np
import random
import os, sys, csv, argparse

from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt

from linear_regressor import MLELinearRegressor
from common import *

def load_dataset(x_path, y_path):
    x_file = open(x_path, 'rb')
    y_file = open(y_path, 'rb')
    
    x_csv_reader = csv.reader(x_file, delimiter=',')
    y_csv_reader = csv.reader(y_file, delimiter=',')

    xs = np.matrix([x for x in x_csv_reader], dtype=np.float32)
    ys = np.matrix([y for y in y_csv_reader], dtype=np.float32)

    return xs, ys

def preprocess(Xs, basis, deg):
    if basis == 'poly':
        assert(deg > 0)
        return PolynomialFeatures(deg).fit_transform(Xs)
    elif basis == 'gaussian':
        raise NotImplementedError()
    elif basis == 'sigmoid':
        raise NotImplementedError()
    else:
        assert(False)

def plotModel_1d(x, y, w):
    plt.plot(x[:,1], y, "x")
    plt.plot(x[:,1], x * w, "r-")
    plt.show()


def main(args): 
    xs, ys = load_dataset(args.X, args.Y)
    assert(xs.shape[0] == ys.shape[0])
    
    xs_fi = np.matrix(preprocess(xs, args.basis, args.deg))
    w = train_sgd(xs_fi, ys, np.zeros((xs_fi.shape[1], 1)))
    plotModel_1d(xs_fi, ys, w)
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--X', help='data', required=True, type=str)
    parser.add_argument('--Y', help='ground truth', required=True, type=str)
    parser.add_argument('--algo', help='algorithm to perform',
            choices=['ml', 'map', 'bayes'], default='ml')
    parser.add_argument('--basis', help='basis function to perform',
            choices=['poly', 'gaussian', 'sigmoid'], default='poly')
    parser.add_argument('--deg', help='deg of poly', type=int, default=0)

    args = parser.parse_args()
    main(args)
