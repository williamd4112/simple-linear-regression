import numpy as np
import numpy.matlib
import tensorflow as tf
import csv

from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans as _kmeans
from scipy.cluster.vq import vq, whiten
from sklearn.preprocessing import PolynomialFeatures

def kmeans(x, k):
    centroids, dist = _kmeans(x, k)
    idx, _ = vq(x,centroids)
    return idx, centroids, dist

class Preprocessor(object):
    def polynomial(self, X, deg=1):
        return PolynomialFeatures(deg).fit_transform(X)

    def normalize(self, X, rng):
        return X / rng

    def grid2d_means(self, x_min, x_max, y_min, y_max, step=0.1, deg=4, scale=2.5):
        X = np.arange(x_min, x_max, step)
        Y = np.arange(y_min, y_max, step)
        X, Y = np.meshgrid(X, Y) 
        
        X, Y = X.flatten(), Y.flatten()
        means = np.array([X, Y], dtype=np.float32).T
        sigmas = np.ones([len(means), 2]) * (step * scale)
        return means, sigmas

    def compute_gaussian_basis(self, xs_normalize, deg=4, scale=2.5):
        xs_normalize_filtered = xs_normalize
        idx, means, dist = kmeans(xs_normalize_filtered, deg)
        sigmas = np.ones([len(means), len(xs_normalize[0])]) * (dist * scale)
        
        return means, sigmas
        
    def gaussian(self, X, means, sigmas):
        n = len(X)
        m = len(means)
        phi_x = np.zeros([n, m])

        for i in xrange(m):
            mean = np.matlib.repmat(means[i], n, 1)
            sigma = np.matlib.repmat(sigmas[i], n, 1)
           
            phi_x[:, i] = np.exp(-np.sum((np.square(X - mean) / (2 * np.square(sigma))), axis=1))
        
        return np.hstack((np.ones([n, 1]), phi_x))
 
if __name__ == '__main__':
    means, sigmas = Preprocessor().grid2d_means(0, 1081, 0, 1081, step=25)
    print means.shape, sigmas.shape

