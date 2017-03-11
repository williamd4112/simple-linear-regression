import numpy as np

# For testing
from sklearn.model_selection import cross_val_score
from sklearn import linear_model

def loss(xs, ys, w):
    m = len(xs)
    return np.sum((xs.dot(w) - ys)**2) / (2 * m)

class LinearRegressor(object):
    def __init__(self, x_shape, y_shape):
        pass
 
    def train(self, x, y):
        '''
        Fit linear model in X (data), Y (ground truth)
 
        '''   
        raise NotImplementedError()
    
    def test(self, x, y):
        '''
        Test linear model in X (data), Y (ground truth)
 
        '''   
        raise NotImplementedError()

class MLELinearRegressor(LinearRegressor):
    def __init__(self, x_shape, y_shape):
        super(LinearRegressor, self).__init__()
        
        self.w = np.random.normal(0, 0.0, x_shape)

        # Learning rate
        self.lr = 0.01

    def train(self, xs, ys):
        '''
        xs: N*d ndarray
        ys: N column vector
        '''
        N = xs.shape[0]
        M = 1
        for i in xrange(0, M*1000, M):
            x_batch = xs[i:i+M]
            y_batch = ys[i:i+M]
        
            self.w = self.w - self.lr * x_batch.T.dot(x_batch.dot(self.w) - y_batch) / N
            print loss(x_batch, y_batch, self.w)
        return self.w

 
    def test(self, xs, ys):
        #return cross_val_score(self.model, xs, ys, cv=10)
        pass

