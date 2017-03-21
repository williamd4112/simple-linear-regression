import numpy as np
import logging

from tqdm import *

class LinearModel(object):
    def __init__(self, shape, optimizer='seq', lr=0.01, clip_min=0, clip_max=1081, is_round=True):
        '''
        param shape: phi_x shape
        '''
        self.optimizer = optimizer
        self.lr = lr
        self.shape = shape
        self.learning_rate_decay = 0.99
        
        # w is Mx1 column vector
        self.w = np.asmatrix(self._init_weight(shape + (1,)))
        
        # Setup model
        self._setup_model()

        # Setup optimizer (vary from models)
        self._setup_optimizer()
    
    def get_weight(self, sess):
        return self.w
    
    def set_weight(self, sess, w):
        self.w = w

    def save(sess):
        self.w_save = self.get_weight(sess)
   
    def save_to_file(self, sess, filename):
        np.save(filename, self.get_weight(sess))
    
    def load(sess):
        self.set_weight(sess, self.w_save)

    def load_from_file(self, sess, filename):
        w = np.load(filename)
        self.set_weight(sess, w)

    def fit(self, sess, x_, t_, epoch=10, batch_size=1):
        lr = self.lr
        if self.optimizer == 'seq':
            batch_indices = range(len(x_))
            for epoch in xrange(epoch):
                np.random.shuffle(batch_indices)
                for i in tqdm(range(0, len(x_), batch_size)):
                    i_ = min(i + batch_size, len(x_))
                    self._optimize(sess, x_[batch_indices[i:i_]], t_[batch_indices[i:i_]], lr=lr)
                loss = self.eval(sess, x_, t_)
           
                if epoch % 1 == 0:
                    logging.info('Epoch %d: training loss = %f (lr = %f)' % (epoch, loss, lr))
                lr = lr * self.learning_rate_decay
        else:
            self._optimize(sess, x_, t_)

    def eval(self, sess, x_, t_):
        return np.sum(np.asarray(t_ - self.test(sess, x_))**2) / (2 * len(x_))
                               
    def test(self, sess, x_):
        return np.asmatrix(x_) * self.w

    def reset(self, sess):
        self.set_weight(sess, np.asmatrix(self._init_weight(self.shape + (1,))))

    def _init_weight(self, shape):
        return np.zeros(shape)
    
    def _setup_model(self):
        '''
            Linear model doesn't need extra setup
        '''        

    def _setup_optimizer(self):
        '''
            Numpy do not need
        '''

    def _optimize(self, sess, x_, t_, lr=None):
        if lr == None:
            lr = self.lr
        self.w = self.w - lr * x_.T * (self.test(sess, x_) - t_) / (2 * len(x_))

class RidgeLinearModel(LinearModel):
    def __init__(self, shape, optimizer='seq', alpha=0.01, mean=0.0, var=0.8, lr=0.01, clip_min=0, clip_max=1081, is_round=True):
        self.alpha = alpha
        self.mean = mean
        self.var = var
        super(RidgeLinearModel, self).__init__(shape, optimizer, lr, clip_min, clip_max, is_round)

    def _optimize(self, sess, x_, t_, lr=None):
        if lr == None:
            lr = self.lr
        self.w = self.w - lr * (x_.T * (self.test(sess, x_) - t_) / (2 * len(x_))) - self.alpha * np.sum(self.w) / (2 * len(x_))

        
