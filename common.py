import numpy as np
import csv

from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans as _kmeans
from scipy.cluster.vq import vq, whiten
from sklearn.preprocessing import PolynomialFeatures

class LossFunction(object):
    def __init__(self):
        pass
    
    def eval(self, x, y, w):
        raise NotImplementedError()
    
    def gradient(self, x, y, w):
        raise NotImplementedError()

class MLEFunction(LossFunction):
    def __init__(self):
        super(LossFunction, self).__init__()

    def eval(self, x, y, w):
        return x * w  - y

    def gradient(self, x, y, w):
        return x.T * self.eval(x, y, w) / len(x)

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
        for i in xrange(0, N, M):
            i_end = max(i+M, N)
            x_batch = x[i:i_end]
            y_batch = y[i:i_end]
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
        xs_normalize = np.matrix((xs))
        idx, means, dist = kmeans(xs_normalize, 40)
        sigmas = uniform_sigma(dist, [len(means), len(xs_normalize[0])])
        return gaussian_basis(xs_normalize, means, sigmas)
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
             
def loss(xs, ys, w):
    return np.sum(np.asarray((xs * w) - ys)**2) / (2 * len(xs))

def train_sgd(xs, ys, w, lr=0.0001, batch_size=1, max_epochs=100000, max_iter=100000, verbose=False):
    N = len(xs) 
    M = batch_size
    J = MLEFunction()
    optimizer = SGDOptimizer(lr=lr, batch_size=batch_size, max_iter=max_iter)

    last_mse = 0
    for epoch in xrange(max_epochs):
        w = optimizer.minimize(J, xs, ys, w)
        mse = loss(xs, ys, w)
        if epoch % 10 == 0:
            print 'Epoch %d: training loss = %f' % (epoch, mse)

        if np.abs(last_mse - mse) <= 1e-8:
            print 'Early stop'
            break                
        last_mse = mse

    return w


