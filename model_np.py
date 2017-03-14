import numpy as np

def mse(ys, ts):
    return np.sum(np.asarray(ys - ts)**2) / (2 * len(ys))

class LinearModel(object):
    def __init__(self, shape, lr=1e-6):
        self.lr = lr
        self.w = np.zeros(shape).T
    
    def _optimize(self, x_batch, t_batch):
        grad = self.lr * (x_batch.T * (x_batch * self.w - t_batch)) / 2.0
        self.w = self.w - grad

    def fit(self, x_, t_, epochs=1000, batch_size=1):
        N = len(x_) 
        M = batch_size

        for epoch in xrange(epochs):
            batch_indices = range(N)
            loss = 0
            np.random.shuffle(batch_indices)
            for i in xrange(0, N, M):
                i_end = max(i+M, N)
                x_batch = x_[batch_indices[i:i_end]]
                t_batch = t_[batch_indices[i:i_end]]

                self._optimize(x_batch, t_batch)
            
            loss = self.eval(x_, t_)        
            if epoch % 1 == 0:
                print 'Epoch %d: training loss = %f' % (epoch, loss)

    def test(self, x_):
        return x_ * self.w

    def eval(self, x_, t_):
        return mse(self.test(x_), t_)         

class RidgeLinearModel(LinearModel):
    def __init__(self, shape, lr=1e-6, lamb=1e-5):
        super(RidgeLinearModel, self).__init__(shape, lr)
        self.lamb = lamb
    
    def _optimize(self, x_batch, t_batch):
        self.w = self.w - self.lr * (x_batch.T * (x_batch * self.w - t_batch)) / 2.0 + 2 * self.lamb * self.w 
 


if __name__ == '__main__':
    from util import generate_1d_dataset
    from preprocess import Preprocessor
    from plot import plot_1d
 
    n = 100
    n_train = int(0.8 * n)

    xs, ys = generate_1d_dataset(n, 100)
    
    phi_xs = Preprocessor().polynomial(xs)
    phi_xs_train, ys_train = phi_xs[:n_train], ys[:n_train]
    phi_xs_test, ys_test = phi_xs[n_train:], ys[n_train:]

    model = LinearModel((2,), lr=1e-5)
    model.fit(phi_xs_train, ys_train, epochs=1000)
    loss = model.eval(phi_xs_test, ys_test)
    print('loss = %f' % loss)
    plot_1d(xs[n_train:], model.test(phi_xs_test), ys_test)        
 
