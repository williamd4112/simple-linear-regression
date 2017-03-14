from model_tf import LinearModel
from util import load_dataset_csv
from plot import plot_3d
from preprocess import Preprocessor

import numpy as np
import tensorflow as tf
import random
import os, sys, csv, argparse

from sklearn.model_selection import KFold

def main(args): 
    xs, ys = load_dataset_csv(args.X, args.Y)
   
    n = len(xs)
    n_train = int(0.8 * n)

    shuffle_indices = range(n)
    np.random.shuffle(shuffle_indices)
    
    print('Preprocessing...')
    deg = 512
    
    preprocessor = Preprocessor()
    means, sigmas = preprocessor.compute_gaussian_basis(preprocessor.normalize(xs, 1081), deg) 

    def phi(x):
        pre = Preprocessor()
        return pre.gaussian(pre.normalize(x, 1081), means, sigmas)

    phi_xs = phi(xs)
    phi_xs_train, ys_train = phi_xs[:n_train], ys[:n_train]
    phi_xs_test, ys_test = phi_xs[n_train:], ys[n_train:]
 
    print('Training...')
    K = args.K
    kf = KFold(n_splits=args.K)

    w_best = None
    min_loss = 1e9
    
    print(phi_xs_test.shape, phi_xs_train.shape)

    model = LinearModel((len(phi_xs_train[0]),), lr=0.5)
    with tf.Session() as sess: 
        for train_index, validation_index in kf.split(phi_xs_train):
            sess.run(tf.global_variables_initializer())
            model.fit(sess, phi_xs_train[train_index], ys_train[train_index], epoch=args.epoch, batch_size=128)
            loss, preds = model.eval(sess, phi_xs_train[validation_index], ys_train[validation_index])

            print('Validation loss = %f' % loss)
 
            if loss < min_loss:
                w_best = model.get_weight(sess)
                min_loss = loss

        model.set_weight(sess, w_best)
        loss, preds = model.eval(sess, phi_xs_test, ys_test)
        print('Test loss = %f' % loss)

        def f(x):
            print x.shape 
            return np.round(np.clip(model.test(sess, x), 0, 1081))

        print('Plotting...')
        plot_3d(f, phi, 0, 1081, 0, 1081)
     

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--X', help='data', required=True, type=str)
    parser.add_argument('--Y', help='ground truth', required=True, type=str)
    parser.add_argument('--K', help='k-fold', type=int, default=3)
    parser.add_argument('--epoch', help='epoch', type=int, default=10)
    args = parser.parse_args()

    main(args)
