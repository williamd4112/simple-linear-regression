import numpy as np

def loss(xs, ys, w):
    return np.sum(np.asarray((xs * w) - ys)**2) / (2 * len(xs))

def sgd(xs, ys, w, lr=0.0001, batch_size=32, max_epochs=100000, verbose=False):
    N = len(xs) 
    M = batch_size
    for epoch in xrange(max_epochs):
        for i in xrange(0, N, M):
            i_ = max(i+M, N)
            x = xs[i:i_]
            y = ys[i:i_]
            error = x * w - y
            gradient = x.T * error / M
            w = w - lr * gradient
        print 'Epoch %d: loss = %f' % (epoch, loss(xs, ys, w))
    return w


