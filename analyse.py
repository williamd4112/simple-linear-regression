from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

from common import load_dataset
import sys

# normal distribution center at x=0 and y=5
x, _ = load_dataset(sys.argv[1], sys.argv[2])
x = np.asarray(x)
x1 = np.squeeze(x[:, 0])
x2 = np.squeeze(x[:, 1])

plt.hist2d(x1, x2, bins=10)
plt.colorbar()
plt.show()
