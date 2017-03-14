from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq

from common import load_dataset
import sys
import numpy as np

# data generation
#data = vstack((rand(150,2) + array([.5,.5]),rand(150,2)))
data, _ = load_dataset(sys.argv[1], sys.argv[2])
data = np.asarray(data)
# computing K-Means with K = 2 (2 clusters)
K = 128
print data.shape
data_ = data[data[:, 0] > 350]
data_ = data_[data_[:, 1] > 300]
print data.shape
centroids,_ = kmeans(data_, K)
# assign each sample to a cluster
idx,_ = vq(data,centroids)

# some plotting using numpy's logical indexing
#plot(data[idx==0,0],data[idx==0,1],'ob',
#     data[idx==1,0],data[idx==1,1],'or',
#     data[idx==2,0],data[idx==2,1],'og')

from matplotlib.pyplot import cm 
color = iter(cm.rainbow(np.linspace(0,1,K)))
for i in xrange(K):
   c = next(color)
   plot(data[idx==i,0],data[idx==i,1], c=c)

plot(centroids[:,0],centroids[:,1],'s',markersize=8)
show()
