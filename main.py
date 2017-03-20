from model_tf import LinearModel
from model_tf import RidgeLinearModel
from model_tf import BayesianLinearModel

from preprocess import Preprocessor

from util import load_dataset_csv, load_data, load_test_dataset_csv
from plot import plot_3d, plot_2d_map

import numpy as np
import tensorflow as tf
import random
import os, sys, csv, argparse

from sklearn.model_selection import KFold

import logging

def load_data(x_path, y_path, shuffle=True):
    xs, ys = load_dataset_csv(x_path, y_path)
   
    n = len(xs)
    shuffle_indices = range(n)
    np.random.shuffle(shuffle_indices)

    return xs, ys, n

'''
    Filter noise data
'''
def filter_data(x):
    xs_normalize_filtered = x
    xs_normalize_filtered = xs_normalize_filtered[xs_normalize_filtered[:, 0] > 0.23774283]
    xs_normalize_filtered = xs_normalize_filtered[xs_normalize_filtered[:, 0] < 0.89]
    xs_normalize_filtered = xs_normalize_filtered[xs_normalize_filtered[:, 1] > 0.120027752]
    return xs_normalize_filtered

'''
    Place additional gaussian basis
'''
def crafted_gaussian_feature(means, sigmas):
    means = np.vstack((means, [0.13876, 0.508788159], [0.46253469, 0.092506938], [0.6475, 0.185]))
    sigmas = np.vstack((sigmas, [0.285, 0.3], [0.5, 0.285], [0.2, 0.1]))
    return means, sigmas

def get_model(args, shape):
    if args.model == 'ml':
        return LinearModel(shape, optimizer=args.optimizer, lr=args.lr)
    elif args.model == 'map':
        logging.info('MAP hyperparameters [alpha: %f]' % args.alpha)
        return RidgeLinearModel(shape, optimizer=args.optimizer, lr=args.lr, alpha=args.alpha)
    elif args.model == 'bayes':
        logging.info('Bayes hyperparameters [m0: %f, s0: %f]' % (args.m0, args.s0))
        return BayesianLinearModel(shape, optimizer=args.optimizer, m0=args.m0, s0=args.s0, beta=args.beta)

def get_means_sigmas(args, x):
    if args.pre == 'kmeans':
        return Preprocessor().compute_gaussian_basis(x, deg=int(args.d), scale=args.scale)
    elif args.pre == 'grid':
        return Preprocessor().grid2d_means(np.min(x[:,0]), np.max(x[:,0]) , np.min(x[:,1]), np.max(x[:,1]), step=args.gsize, scale=args.scale)

def train(args, sess, model, phi_xs_train, ys_train):
    sess.run(tf.global_variables_initializer())
    model.fit(sess, phi_xs_train, ys_train, epoch=args.epoch, batch_size=args.batch_size)
    loss = model.eval(sess, phi_xs_train, ys_train)
    return loss

def train_cross_validation(args, sess, model, phi_xs_train, ys_train):
    kf = KFold(n_splits=args.K)

    w_best = None
    validation_loss = 0
     
    for train_index, validation_index in kf.split(phi_xs_train):
        sess.run(tf.global_variables_initializer())

        model.fit(sess, phi_xs_train[train_index], ys_train[train_index], epoch=args.epoch, batch_size=args.batch_size)
        loss = model.eval(sess, phi_xs_train[validation_index], ys_train[validation_index])

        logging.info('Validation loss = %f' % (loss))
        validation_loss += loss

    return validation_loss / float(args.K)

def test_model(args):
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.info('Loading data...')

    # Load test dataset    
    xs, n = load_test_dataset_csv(args.X)
 
    # Data preprocessing
    preprocessor = Preprocessor()
    rng = abs(args.max - args.min)
    xs_n = preprocessor.normalize(xs, rng)
    
    # Load means and sigmas
    logging.info('Loading mean data from %s' % (args.mean))
    means = np.load(args.mean)

    logging.info('Loading sigma data from %s' % (args.sigma))
    sigmas = np.load(args.sigma)

    # Setup preprocessing function
    def phi(x):
        pre = Preprocessor()
        return pre.gaussian(pre.normalize(x, rng), means, sigmas)

    logging.info('Preprocessing (d = %d)' % (len(means) + 1))
    phi_xs = phi(xs)
    phi_dim = len(phi_xs[0])
    model = get_model(args, (phi_dim,))
    logging.info('Using model %s' % (args.model))

    def f(x):
        return np.round(np.clip(model.test(sess, x), args.min, args.max))

    with tf.Session() as sess:
        assert args.output is not None

        logging.info('Loading model from %s' % (args.load))
        sess.run(tf.global_variables_initializer())

        model.load_from_file(sess, args.load)
        preds = f(phi_xs)

        logging.info('Save predictions at %s' % args.output)
        with open(args.output, 'w') as file:
            for pred in preds:
                file.write('%f\n' % pred)                    
     
def train_model(args):
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.info('Train model')
    logging.info('Loading data...')

    # Split data into training set and test set    
    xs, ys, n = load_data(args.X, args.Y, shuffle=True)
    n_train = int(args.frac * n)
     
    # Data preprocessing
    preprocessor = Preprocessor()
    rng = abs(args.max - args.min)
    xs_n = preprocessor.normalize(xs, rng)
    xs_n_filtered = xs_n
    
    if args.craft:
        xs_n_filtered = filter_data(xs_n_filtered)
    
    # Feature extraction
    logging.info('Computing means and sigmas (%s)...' % args.pre)
    means, sigmas = get_means_sigmas(args, xs_n_filtered)

    if args.craft:
        means, sigmas = crafted_gaussian_feature(means, sigmas)

    def phi(x):
        pre = Preprocessor()
        return pre.gaussian(pre.normalize(x, rng), means, sigmas)
    
    logging.info('Preprocessing... (d = %d; craft-feature %d)' % (means.shape[0], args.craft))
    phi_xs = phi(xs)
    phi_xs_train, ys_train = phi_xs[:n_train], ys[:n_train]
    phi_xs_test, ys_test = phi_xs[n_train:], ys[n_train:]
 
    phi_dim = len(phi_xs_train[0])
    model = get_model(args, (phi_dim,))
    logging.info('Using model %s (plot = %s)' % (args.model, args.plot))

    def f(x):
        return np.round(np.clip(model.test(sess, x), args.min, args.max))

    with tf.Session() as sess:
        logging.info('Training... (optimizer = %s)' % args.optimizer)
        if args.K <= 1:
            train_loss = train(args, sess, model, phi_xs_train, ys_train)
            logging.info('Training loss = %f' % train_loss)

            if n_train < n:
                test_loss = model.eval(sess, phi_xs_test, ys_test) 
                logging.info('Testing loss = %f' % test_loss)

            if args.output is not None:
                logging.info('Save model at %s' % args.output)
                model.save_to_file(sess, args.output)
                np.save(args.output + '-mean', means)
                np.save(args.output + '-sigma', sigmas)

            if args.plot is not None:
                logging.info('Plotting... (output = %s)' % args.fig)
                if args.plot == '3d':
                    plot_3d(f, phi, args.min, args.max, args.min, args.max, 0, 1081, args.fig)
                elif args.plot == '2d':
                    plot_2d_map(f, phi, args.min, args.max, args.min, args.max)
            
        else:
            validation_loss = train_cross_validation(args, sess, model, phi_xs_train, ys_train)
            log_filename = args.log
            with open(log_filename, 'w') as log_file:
                log_file.write('%s\t%s\n' % (log_filename, validation_loss))
             

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='train/test task', 
            choices=['train', 'test'], required=True, type=str, default='train')
    parser.add_argument('--X', help='data', required=True, type=str)
    parser.add_argument('--Y', help='ground truth', type=str)
    parser.add_argument('--K', help='k-fold', type=int, default=3)
    parser.add_argument('--epoch', help='epoch', type=int, default=50)
    parser.add_argument('--batch_size', help='batch size', type=int, default=128)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.5)
    parser.add_argument('--d', help='dimension of feature', type=float, default=512)
    parser.add_argument('--gsize', help='grid size', type=float, default=0.1)
    parser.add_argument('--scale', help='sigma scale', type=float, default=2.5)
    parser.add_argument('--min', help='minimum value of input space', type=float, default=0)
    parser.add_argument('--max', help='maximum value of input space', type=float, default=1081)
    parser.add_argument('--frac', help='fraction of training', type=float, default=0.8)
    parser.add_argument('--alpha', help='l2 penalty scale for map', type=float, default=0.01)
    parser.add_argument('--beta', help='beta (noise variance) for bayesian', type=float, default=1.0 / 0.2**2)
    parser.add_argument('--m0', help='m0 (mean) for bayesian', type=float, default=0.0)
    parser.add_argument('--s0', help='s0 (variance) for bayesian', type=float, default=2.0)
    parser.add_argument('--model', help='model',
            choices=['ml', 'map', 'bayes'], default='ml')
    parser.add_argument('--pre', help='preprocess approach',
            choices=['kmeans', 'grid'], default='grid')
    parser.add_argument('--optimizer', help='optimzier',
            choices=['ls', 'seq'], default='ls')
    parser.add_argument('--plot', help='enable plot', type=str, default=None)
    parser.add_argument('--craft', help='enable crafted features', type=bool, default=False)
    parser.add_argument('--fig', help='figure output', type=str, default=None)
    parser.add_argument('--log', help='log output', type=str, default='log.log')
    parser.add_argument('--output', help='output data, model or predictions', type=str, default=None)
    parser.add_argument('--load', help='model load', type=str, default=None)
    parser.add_argument('--mean', help='mean load', type=str, default=None)
    parser.add_argument('--sigma', help='sigma load', type=str, default=None)

    args = parser.parse_args()

    if args.task == 'train':
        train_model(args)
    else:
        test_model(args)
