from common import *
import sys
import numpy as np

k = 4
x, y = load_dataset(sys.argv[1], sys.argv[2])
idx, means, dist = kmeans(x, k)

sigmas = np.array([[100, 100]])
sigmas = np.repeat(sigmas, k, axis=0)

print x[0]
print means
print sigmas

phi_x = gaussian_kernel(x[0], means, sigmas)
print 'phi\n', phi_x

# Check gaussian
i = 0
tmp = np.zeros([k, 2])
error = np.zeros(k)
for mean, sigma in zip(means, sigmas):
    mu1 = mean[0]
    mu2 = mean[1]
    s1 = sigma[0]
    s2 = sigma[1]
    tmp[i] = [(x[0,0] - mu1)**2 / (2 * s1**2), (x[0,1] - mu2)**2 / (2 * s2**2)]
    error[i] = np.exp(-np.sum(tmp[i]))
    i += 1
print error
