import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random

def generateSample(N, variance=100):
    X = np.matrix(range(N)).T + 1
    Y = np.matrix([random.random() * variance + i * 10 + 900 for i in range(len(X))]).T
    print X.shape, Y.shape
    return X, Y

def fitModel_gradient(x, y):
    N = len(x)
    w = np.zeros((x.shape[1], 1))
    print type(w), w.shape
    eta = 0.0001

    maxIteration = 100000
    for i in range(maxIteration):
        error = x * w - y
        gradient = x.T * error / N
        w = w - eta * gradient
    return w

def plotModel(x, y, w):
    plt.plot(x[:,1], y, "x")
    plt.plot(x[:,1], x * w, "r-")
    plt.show()

def test(N, variance, modelFunction):
    X, Y = generateSample(N, variance)
    with open('X.csv', 'w') as f:
        for x in X:
            f.write('%f\n' % x)
    with open('Y.csv', 'w') as f:
        for y in Y:
            f.write('%f\n' % y)

    X = np.hstack([np.matrix(np.ones(len(X))).T, X])
    w = modelFunction(X, Y)
    #plotModel(X, Y, w)


#test(50, 600, fitModel_gradient)
#test(50, 1000, fitModel_gradient)
test(100, 200, fitModel_gradient)
