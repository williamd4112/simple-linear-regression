import numpy as np
import tensorflow as tf

from sklearn.model_selection import KFold
import logging

class LinearModel(object):
    def __init__(self, shape, optimizer='seq', lr=0.01, clip_min=0, clip_max=1081, is_round=True):
        '''
        param shape: phi_x shape
        '''
        self.optimizer = optimizer
        self.lr = lr

        # x is NxM matrix
        self.x = tf.placeholder(tf.float32, shape=(None,) + shape)

        # t is Nx1 column vector
        self.t = tf.placeholder(tf.float32, shape=(None,) + (1,))
        
        # w is Mx1 column vector
        self.w_assign = tf.placeholder(tf.float32, shape=shape + (1,))
        self.w = tf.Variable(self._init_weight(shape + (1,)), dtype=tf.float32)
        self.w_assign_op = tf.assign(self.w, self.w_assign)
        
        # y is Nx1 column vector
        self.y = tf.matmul(self.x, self.w)

        # MSE evaluation
        self.loss = tf.reduce_mean(tf.pow(self.y - self.t, 2)) / 2.0

        # Setup model
        self._setup_model()

        # Setup optimizer (vary from models)
        self._setup_optimizer()
    
    def get_weight(self, sess):
        return sess.run(self.w)
    
    def set_weight(self, sess, w):
        sess.run(self.w_assign_op, feed_dict={self.w_assign: w})

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
        if self.optimizer == 'seq':
            batch_indices = range(len(x_))
            for epoch in xrange(epoch):
                np.random.shuffle(batch_indices)
                for i in xrange(0, len(x_), batch_size):
                    i_ = max(i, len(x_))
                    self._optimize(sess, x_[batch_indices[i:i_]], t_[batch_indices[i:i_]])
                    loss = self.eval(sess, x_, t_)
                 
                if epoch % 1 == 0:
                    logging.info('Epoch %d: training loss = %f' % (epoch, loss))
        else:
            self._optimize(sess, x_, t_)

    def eval(self, sess, x_, t_):
        return sess.run(self.loss, feed_dict={self.x: x_,
                                              self.t: t_})
    def test(self, sess, x_):
        return sess.run(self.y, feed_dict={self.x: x_})

    def _init_weight(self, shape):
        return np.zeros(shape)
    
    def _setup_model(self):
        '''
        Linear model doesn't need extra setup
        '''        

    def _setup_optimizer(self): 
        if self.optimizer == 'seq':
            self.optimize_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        else:
            # W_ml is calculated with solving normal equation
            xt = tf.transpose(self.x)
            x_xt = tf.matmul(xt, self.x)
            x_xt_inv = tf.matrix_inverse(x_xt)
            x_xt_inv_xt = tf.matmul(x_xt_inv, xt)

            self.w_ml = tf.matmul(x_xt_inv_xt, self.t)
            self.optimize_op = tf.assign(self.w, self.w_ml)

    def _optimize(self, sess, x_, t_):
        sess.run(self.optimize_op, feed_dict={self.x: x_,
                                              self.t: t_})
         

class RidgeLinearModel(LinearModel):
    def __init__(self, shape, optimizer='seq', alpha=0.01, mean=0.0, var=0.8, lr=0.01, clip_min=0, clip_max=1081, is_round=True):
        self.alpha = alpha
        self.mean = mean
        self.var = var
        super(RidgeLinearModel, self).__init__(shape, optimizer, lr, clip_min, clip_max, is_round)

    def _init_weight(self, shape):
        return np.random.normal(self.mean, self.var, shape)

    def _setup_optimizer(self):
        self.ridge_loss = tf.reduce_mean(tf.pow(self.y - self.t, 2) + self.alpha * tf.reduce_sum(tf.square(self.w))) / 2.0
        self.optimize_op = tf.train.AdamOptimizer(self.lr).minimize(self.ridge_loss) 

class BayesianLinearModel(LinearModel):
    def __init__(self, shape, m0, s0, beta, optimizer='ls', clip_min=0, clip_max=1081, is_round=True):
        # Setup mean vector
        if np.isscalar(m0):
            self.m0 = tf.ones(shape + (1,), dtype=tf.float32) * m0
        else:
            self.m0 = m0
        
        # Setup covariance matrix
        if np.isscalar(s0):
            self.s0 = 1.0 / s0 * tf.identity(np.identity(shape[0], dtype=np.float32))
        else:
            self.s0 = s0

        # Setup variance of noise gaussian distribution
        self.beta = beta
        
        super(BayesianLinearModel, self).__init__(shape, optimizer, 0.0, clip_min, clip_max, is_round)
 
    def _setup_model(self):
        # Setup mn, sn
        xt = tf.transpose(self.x)
        self.sn = tf.matrix_inverse(tf.matrix_inverse(self.s0) + (self.beta * tf.matmul(xt, self.x)))
        self.mn = tf.matmul(self.sn, tf.matmul(tf.matrix_inverse(self.s0), self.m0) + self.beta * tf.matmul(xt, self.t))

    def _setup_optimizer(self):
        self.optimize_op = tf.assign(self.w, self.mn)

class ModelTrainer(object):
    def __init__(self, model_name, model_phi):
        self.model_name = model_name
        self.model_phi = model_phi

    def get_model(self, cfg, shape):
        if self.model_name == 'ml':
            return LinearModel(shape, optimizer=cfg['optimizer'], lr=cfg['lr'])
        elif self.model_name == 'map':
            logging.info('MAP hyperparameters [alpha: %f]' % cfg['alpha'])
            return RidgeLinearModel(shape, optimizer=cfg['optimizer'], lr=cfg['lr'], alpha=cfg['alpha'])
        elif self.model_name == 'bayes':
            logging.info('Bayes hyperparameters [m0: %f, s0: %f]' % (cfg['m0'], cfg['s0']))
            return BayesianLinearModel(shape, optimizer=cfg['optimizer'], m0=cfg['m0'], s0=cfg['s0'], beta=cfg['beta'])
 
    def train_cross_validation(self, sess, X, Y, n_train, K, cfg):
        # Preprocess data
        phi_xs = self.model_phi(X)
        phi_xs_train, ys_train = phi_xs[:n_train], Y[:n_train]

        kf = KFold(n_splits=K)

        # Init model
        model = self.get_model(cfg, (len(phi_xs_train[0]),))

        logging.info('Model [%s]' % self.model_name)        

        # Train with K-fold
        model_mean_loss = 0
        for train_index, validation_index in kf.split(X):
            print train_index, phi_xs_train.shape, X.shape
            sess.run(tf.global_variables_initializer())
            model.fit(sess, phi_xs_train[train_index], ys_train[train_index], epoch=cfg['epoch'], batch_size=cfg['batch_size'])
            loss = model.eval(sess, phi_xs_train[validation_index], ys_train[validation_index])
            
            model_mean_loss += loss

        # Calculate mean of validation loss
        model_mean_loss /= float(K)

        return model_mean_loss



'''
class BayesianLinearModel(LinearModel):
    def __init__(self, shape, m0, s0, beta, optimizer='seq', clip_min=0, clip_max=1081, is_round=True):
        self.w = np.zeros(shape + (1,))

        if np.isscalar(m0):
            self.m0 = np.ones(shape + (1,), dtype=np.float32) * m0
        else:
            self.m0 = m0
        
        if np.isscalar(s0):
            self.s0 = 1.0 / s0 * np.identity(shape[0], dtype=np.float32)
        else:
            self.s0 = s0
        print self.s0
        self.mn = self.m0
        self.sn = self.s0
        self.beta = beta

        print 'w', self.w.shape
        print 'm0', self.m0.shape
        print 's0', self.s0.shape

    def get_weight(self, sess):
        return self.w

    def set_weight(self, sess, w):
        self.w = w

    def _optimize(self, sess, x_, t_):
        x_mat = np.asmatrix(x_)
        t_mat = np.asmatrix(t_)
        
        self.sn = np.linalg.inv(np.linalg.inv(self.s0) + self.beta * (x_mat.T * x_mat))
        self.mn = self.sn * (np.linalg.inv(self.s0) * self.m0 + self.beta * x_mat.T * t_mat)
        self.s0 = self.sn
        self.m0 = self.mn

        self.w = self.mn

    def fit(self, sess, x_, t_, epoch=10, batch_size=1):
        batch_indices = range(len(x_))
        for epoch in xrange(epoch):
            np.random.shuffle(batch_indices)
            for i in xrange(0, len(x_), batch_size):
                i_ = max(i, len(x_))
                self._optimize(sess, x_[batch_indices[i:i_]], t_[batch_indices[i:i_]])
                loss = self.eval(sess, x_, t_)
             
            if epoch % 1 == 0:
                print 'Epoch %d: training loss = %f' % (epoch, loss)

    def eval(self, sess, x_, t_):
        return np.sum((self.test(sess, x_) - np.asarray(t_))**2) / (2 * len(x_))

    def test(self, sess, x_):
        return np.asarray(np.asmatrix(x_) * np.asmatrix(self.w))
'''    

if __name__ == '__main__':
    from preprocess import Preprocessor
    from util import generate_1d_dataset
    from plot import plot_1d

    N = 100
    N_train = int(0.2 * N)
    xs, ys = generate_1d_dataset(N, variance=100)

    phi_xs = Preprocessor().polynomial(xs)
   
    model = BayesianLinearModel((2,), m0=0.0, s0=2.0, beta=1/0.2**2)
    with tf.Session() as sess:
        model.fit(sess, phi_xs[:N_train], ys[:N_train])
        print model.eval(sess, phi_xs[N_train:], ys[N_train:])
        pred = model.test(sess, phi_xs)
        plot_1d(xs, pred, ys)
        
