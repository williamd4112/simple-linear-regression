from matplotlib.colors import LogNorm

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


import numpy as np

def plot_1d(x, y, t):
    plt.plot(x[:], t, "x")
    plt.plot(x[:], y, "r-")
    plt.show()

def plot_2d_hist(x1, x2, bins=10):
    plt.hist2d(x1, x2, bins=10, norm=LogNorm())
    plt.colorbar()
    plt.show()

def plot_3d(model, phi, x_min, x_max, y_min, y_max):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.arange(x_min, x_max, 5)
    Y = np.arange(y_min, y_max, 5)
    X, Y = np.meshgrid(X, Y)
    
    x, y = np.reshape(X, len(X)**2), np.reshape(Y, len(Y)**2) 
    Z = model(np.matrix(phi(np.array([x, y], dtype=np.float32).T)))
    
    Z = np.reshape(Z, [len(X), len(X)])    

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, shade=True)

    # Customize the z axis.
    ax.set_zlim(0, 1081)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
