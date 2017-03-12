from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

def plot_2d_hist(x1, x2, bins=10):
    plt.hist2d(x1, x2, bins=10, norm=LogNorm())
    plt.colorbar()
    plt.show()
