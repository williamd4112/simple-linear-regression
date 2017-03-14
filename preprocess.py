import numpy as np
import numpy.matlib
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

    def gaussian(self, X, deg=4):
        xs_normalize = X / 1081
        xs_normalize_filtered = xs_normalize
        xs_normalize_filtered = xs_normalize_filtered[xs_normalize_filtered[:, 0] > 0.323774283]
        xs_normalize_filtered = xs_normalize_filtered[xs_normalize_filtered[:, 1] > 0.270027752]

        idx, means, dist = kmeans(xs_normalize_filtered, deg)

        # Hand design basis
        means = np.vstack((means, [0.13876, 0.508788159]))
        means = np.vstack((means, [0.46253469, 0.092506938]))

        sigmas = np.ones([len(means), len(xs_normalize[0])]) * (dist * 2.5)

        sigmas = np.vstack((sigmas, [0.285, 1.0], [1.0, 0.285]))

        n = len(xs_normalize)
        m = len(means)
        phi_x = np.zeros([n, m])

        for i in xrange(m):
            mean = np.matlib.repmat(means[i], n, 1)
            sigma = np.matlib.repmat(sigmas[i], n, 1)
           
            phi_x[:, i] = np.exp(-np.sum((np.square(xs_normalize - mean) / (2 * np.square(sigma))), axis=1))
        
        #return phi_x
        return np.hstack((np.ones([n, 1]), phi_x))
