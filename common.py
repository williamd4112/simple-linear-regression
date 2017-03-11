import numpy as np

class LossFunction(object):
    def __init__(self):
        pass
    
    def eval(self, x, y, w):
        raise NotImplementedError()
    
    def gradient(self, x, y, w):
        raise NotImplementedError()

class MLEFunction(LossFunction):
    def __init__(self):
        super(LossFunction, self).__init__()

    def eval(self, x, y, w):
        return x * w  - y

    def gradient(self, x, y, w):
        return x.T * self.eval(x, y, w) / len(x)

class Optimizer(object):
    def minimize(self, j):
        raise NotImplementedError()

class GradientDescentOptimizer(Optimizer):
    def __init__(self, lr=0.0001):
        super(GradientDescentOptimizer, self).__init__()
        self.lr = lr

    def minimize(self, j, x, y, w):
        return w - self.lr * j.gradient(x, y, w)

class SGDOptimizer(GradientDescentOptimizer):
    def __init__(self, lr=0.0001, batch_size=32):
        super(SGDOptimizer, self).__init__(lr)
        self.batch_size = batch_size

    def minimize(self, j, x, y, w):
        N = len(x)
        M = self.batch_size
        for i in xrange(0, N, M):
            i_end = max(i+M, N)
            x_batch = x[i:i_end]
            y_batch = y[i:i_end]
            w = super(SGDOptimizer, self).minimize(j, x_batch, y_batch, w)
        return w

def loss(xs, ys, w):
    return np.sum(np.asarray((xs * w) - ys)**2) / (2 * len(xs))

def train_sgd(xs, ys, w, lr=0.0001, batch_size=16, max_epochs=100000, verbose=False):
    N = len(xs) 
    M = batch_size
    J = MLEFunction()
    optimizer = SGDOptimizer(lr=lr, batch_size=batch_size)

    for epoch in xrange(max_epochs):
        w = optimizer.minimize(J, xs, ys, w)
        print 'Epoch %d: loss = %f' % (epoch, loss(xs, ys, w))
    return w


