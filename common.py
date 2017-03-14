import numpy as np
import csv

from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans as _kmeans
from scipy.cluster.vq import vq, whiten
from sklearn.preprocessing import PolynomialFeatures

from preprocess import Preprocessor

class LossFunction(object):
    def __init__(self):
        pass
    
    def eval(self, x, y, w):
        raise NotImplementedError()
    
    def gradient(self, x, y, w):
        raise NotImplementedError()

class MLEFunction(LossFunction):
    def __init__(self):
        super(MLEFunction, self).__init__()

    def gradient(self, x, y, w):
        return x.T * (x * w - y) / 2

class MAPFunction(MLEFunction):
    def __init__(self, alpha=0.01):
        super(MAPFunction, self).__init__()
        self.alpha = alpha

    def gradient(self, x, y, w):
        return super(MAPFunction, self).gradient(x, y, w) + self.alpha * w / len(x)


class Optimizer(object):
    def minimize(self, j):
        raise NotImplementedError()

class GradientDescentOptimizer(Optimizer):
    def __init__(self, lr=0.0001):
        super(GradientDescentOptimizer, self).__init__()
        self.lr = lr

    def minimize(self, j, x, y, w):
        return w - self.lr * j.gradient(x, y, w)

class SGDOptimizer(GradientDescentOptimizer):
    def __init__(self, lr=0.0001, batch_size=32, max_iter=100):
        super(SGDOptimizer, self).__init__(lr)
        self.batch_size = batch_size
        self.max_iter = max_iter

    def minimize(self, j, x, y, w):
        N = len(x)
        M = self.batch_size
        count = 0
        batch_indices = range(N)
        for i in xrange(0, N, M):
            i_end = max(i+M, N)
            x_batch = x[batch_indices[i:i_end]]
            y_batch = y[batch_indices[i:i_end]]
            w = super(SGDOptimizer, self).minimize(j, x_batch, y_batch, w)

            count += 1
            if count > self.max_iter:
                break
        return w

def load_dataset(x_path, y_path):
    x_file = open(x_path, 'rb')
    y_file = open(y_path, 'rb')
    
    x_csv_reader = csv.reader(x_file, delimiter=',')
    y_csv_reader = csv.reader(y_file, delimiter=',')

    xs = np.matrix([x for x in x_csv_reader], dtype=np.float32)
    ys = np.matrix([y for y in y_csv_reader], dtype=np.float32)

    return xs, ys

def preprocess(xs, basis):
    if basis == 'poly':
        return PolynomialFeatures(1).fit_transform(xs)
    elif basis == 'gaussian':
        '''
        #xs_normalize = np.matrix(whiten(xs))
        
        xs_normalize = xs / 1081.0
        xs_normalize_filtered = np.asarray(xs_normalize)
        #xs_normalize_filtered = xs_normalize_filtered[xs_normalize_filtered[:, 0] > 0.323774283]
        #xs_normalize_filtered = xs_normalize_filtered[xs_normalize_filtered[:, 1] > 0.370027752]
        #xs_normalize_filtered = whiten(xs_normalize_filtered)
        
        idx, means, dist = kmeans(xs_normalize_filtered, 256)
        sigmas = uniform_sigma(dist, [len(means), len(xs_normalize[0])])
        return (np.hstack((gaussian_basis(xs_normalize, means, sigmas), np.ones([len(xs), 1]))))
        '''
        return np.matrix(Preprocessor().gaussian(np.asarray(xs), 128))
    elif basis == 'sigmoid':
        raise NotImplementedError()

def kmeans(x, k):
    centroids, dist = _kmeans(x, k)
    idx, _ = vq(x,centroids)
    return idx, centroids, dist

def uniform_sigma(s, shape):
    return np.ones(shape) * s

def gaussian_kernel(x, means, sigmas):
    return np.exp(-np.sum(np.asarray(np.repeat(x, len(means), axis=0) - means)**2 / (2 * (sigmas**2)), axis=1))

def gaussian_basis(xs, means, sigmas):
    phi_x = np.zeros([len(xs), len(means)])
    for i in xrange(len(phi_x)):
        phi_x[i] = gaussian_kernel(xs[i], means, sigmas)
    return phi_x     

def score(ys, ys_):
    return np.sum(np.asarray(ys - ys_)**2) / (2 * len(ys))
             
def loss(xs, ys, w):
    return np.sum(np.asarray((xs * w) - ys)**2) / (2 * len(xs))

def train_sgd(J, xs, ys, w, lr=0.0001, batch_size=1, max_epochs=100000, max_iter=100000, verbose=False):
    N = len(xs) 
    M = batch_size
    optimizer = SGDOptimizer(lr=lr, batch_size=batch_size, max_iter=max_iter)

    last_mse = 0
    for epoch in xrange(max_epochs):
        w = optimizer.minimize(J, xs, ys, w)
        mse = loss(xs, ys, w)
        if epoch % 1 == 0:
            print 'Epoch %d: training loss = %f' % (epoch, mse)

        if np.abs(last_mse - mse) <= 1e-5:
            print 'Early stop'
            break                
        last_mse = mse

    return w


