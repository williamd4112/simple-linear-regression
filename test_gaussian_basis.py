from common import *
import sys
import numpy as np

from plot import *

k = 2
x, y = load_dataset(sys.argv[1], sys.argv[2])
idx, means, dist = kmeans(x, k)

sigmas = np.array([[200, 200]])
sigmas = np.repeat(sigmas, k, axis=0)

phi_x = gaussian_basis(x, means, sigmas)

print phi_x
print phi_x.shape

plot_2d_hist(phi_x[:, 0], phi_x[:, 1])
