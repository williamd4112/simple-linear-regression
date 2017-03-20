from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

# normal distribution center at x=0 and y=5
#x = np.random.randn(100000)
#y = np.random.randn(100000) + 5

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from util import load_csv

data = load_csv(sys.argv[1])
x = data[:,0]
y = data[:,1]

plt.hist2d(x, y, bins=40, norm=LogNorm())
plt.colorbar()
plt.show()
