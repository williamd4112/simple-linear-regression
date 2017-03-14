from model_np import LinearModel
from util import load_dataset_csv
from plot import plot_3d
from preprocess import Preprocessor

import numpy as np
import random
import os, sys, csv, argparse

from sklearn.model_selection import KFold

def main(args):

    deg = 4
    
    def phi(x):
        return np.matrix(Preprocessor().gaussian(x, deg))

    xs, ys = load_dataset_csv(args.X, args.Y)
     
    n = len(xs)
    n_train = int(0.8 * n)

    shuffle_indices = range(n)
    np.random.shuffle(shuffle_indices)
    
    print('Preprocessing...')
    phi_xs = phi(xs)
    phi_xs_train, ys_train = phi_xs[:n_train], ys[:n_train]
    phi_xs_test, ys_test = phi_xs[n_train:], ys[n_train:]
 
    print('Training...')
    K = args.K
    kf = KFold(n_splits=args.K)

    model_best = None
    min_loss = 1e9
    
    model = LinearModel((len(phi_xs_train[0]),), lr=0.5) 
    for train_index, validation_index in kf.split(phi_xs_train):
        model.fit(phi_xs_train[train_index], ys_train[train_index], epochs=10, batch_size=1)
        loss = model.eval(phi_xs_train[validation_index], ys_train[validation_index])

        print('Validation loss = %f' % loss)

        if loss < min_loss:
            model_best = model
            min_loss = loss

    loss = model_best.eval(phi_xs_test, ys_test)
    print('Test loss = %f' % loss)
 

    def f(x):
        return np.round(np.clip(model_best.test(x), 0, 1081))
    

    print('Plotting...')
    plot_3d(f, phi, 0, 1081, 0, 1081)
     

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--X', help='data', required=True, type=str)
    parser.add_argument('--Y', help='ground truth', required=True, type=str)
    parser.add_argument('--K', help='k-fold', type=int, default=3)
    args = parser.parse_args()

    main(args)
