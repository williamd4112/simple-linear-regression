from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)


from util import load_csv
from preprocess import Preprocessor
import sys
import numpy as np

# data generation
#data = vstack((rand(150,2) + array([.5,.5]),rand(150,2)))
data = load_csv(sys.argv[1])
data = np.asarray(data)
# computing K-Means with K = 2 (2 clusters)
K = 128
print data.shape
data_ = data
data_ = data_[data_[:, 0] > 350]
data_ = data_[data_[:, 1] > 300]
x = data_

print data.shape
#centroids,_ = kmeans(data_, K)
centroids,_ = Preprocessor().grid2d_means(np.min(x[:,0]), np.max(x[:,0]) , np.min(x[:,1]), np.max(x[:,1]), step=100.81, scale=1.0)
# assign each sample to a cluster
idx,_ = vq(data,centroids)
K = len(centroids)
from matplotlib.pyplot import cm 
color = iter(cm.rainbow(np.linspace(0,1,K)))
for i in xrange(K):
   c = next(color)
   plot(data[idx==i,0],data[idx==i,1], c=c)

plot(centroids[:,0],centroids[:,1],'s',markersize=8)
show()
