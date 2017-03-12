import numpy as np
import random
import os, sys, csv, argparse

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

from common import *

def plotModel_1d(x, x_phi, y, w):
    plt.plot(x[:], y, "x")
    plt.plot(x[:], x_phi * w, "r-")
    plt.show()

def main(args): 
    xs, ys = load_dataset(args.X, args.Y)
    N_train = int(0.8 * len(xs))
    N_test = len(xs) - N_train

    assert(xs.shape[0] == ys.shape[0])
    
    xs_phi = np.matrix(preprocess(xs, args.basis))
    xs_phi_train, ys_train = xs_phi[:N_train], ys[:N_train]
    xs_phi_test, ys_test = xs_phi[-N_test:], ys[-N_test:]
    
    K = args.K
    kf = KFold(n_splits=args.K)
    
    w_best = None
    loss_min = 1e9

    for train_index, validation_index in kf.split(xs_phi_train):
        w = train_sgd(xs_phi_train[train_index], ys_train[train_index], np.random.normal(0, 0.25, [xs_phi_train.shape[1], 1]), lr=0.0001, batch_size=1, max_epochs=args.epoch)
        loss_ = loss(xs_phi_train[validation_index], ys_train[validation_index], w)
        print 'Validation loss = %f' % loss_
        
        if loss_ < loss_min or w_best is None:
            w_best = w
            loss_min = loss

    loss_ = loss(xs_phi_test, ys_test, w_best)
    print 'Test loss = %f' % loss_  

    #plotModel_1d(xs, xs_fi, ys, w)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--X', help='data', required=True, type=str)
    parser.add_argument('--Y', help='ground truth', required=True, type=str)
    parser.add_argument('--K', help='k-fold', type=int, default=3)
    parser.add_argument('--epoch', help='k-fold', type=int, default=10000)
    parser.add_argument('--algo', help='algorithm to perform',
            choices=['ml', 'map', 'bayes'], default='ml')
    parser.add_argument('--basis', help='basis function to perform',
            choices=['poly', 'gaussian', 'sigmoid', 'custom'], default='poly')

    args = parser.parse_args()
    main(args)
