import numpy as np
import tensorflow as tf

class LinearModel(object):
    def __init__(self, shape, optimizer='sgd', lr=0.01, clip_min=0, clip_max=1081, is_round=True):
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
        self.y = (tf.matmul(self.x, self.w))

        # Setup MSE evaluation
        self.loss = tf.reduce_mean(tf.pow(self.y - self.t, 2)) / 2.0

        self._setup_optimizer()
    
    def get_weight(self, sess):
        return sess.run(self.w)
    
    def set_weight(self, sess, w):
        sess.run(self.w_assign_op, feed_dict={self.w_assign: w})         

    def fit(self, sess, x_, t_, epoch=1000, batch_size=1000):
        if self.optimizer == 'sgd':
            batch_indices = range(len(x_))
            for epoch in xrange(epoch):
                np.random.shuffle(batch_indices)
                for i in xrange(0, len(x_), batch_size):
                    i_ = max(i, len(x_))
                    self._optimize(sess, x_[batch_indices[i:i_]], t_[batch_indices[i:i_]])
                    loss = self.eval(sess, x_, t_)
                 
                if epoch % 1 == 0:
                    print 'Epoch %d: training loss = %f' % (epoch, loss)
        else:
            self._optimize(sess, x_, t_)

    def eval(self, sess, x_, t_):
        return sess.run(self.loss, feed_dict={self.x: x_,
                                              self.t: t_})
    def test(self, sess, x_):
        return sess.run([self.y], feed_dict={self.x: x_})

    def _init_weight(self, shape):
        return np.zeros(shape)

    def _setup_optimizer(self): 
        if self.optimizer == 'sgd':
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
    def __init__(self, shape, optimizer='sgd', alpha=0.01, mean=0.0, var=0.8, lr=0.01, clip_min=0, clip_max=1081, is_round=True):
        self.alpha = alpha
        self.mean = mean
        self.var = var
        super(RidgeLinearModel, self).__init__(shape, optimizer, lr, clip_min, clip_max, is_round)

    def _init_weight(self, shape):
        return np.random.normal(self.mean, self.var, shape)

    def _setup_optimizer(self):
        self.ridge_loss = tf.reduce_mean(tf.pow(self.y - self.t, 2) + self.alpha * tf.reduce_sum(tf.square(self.w)) / 2.0)
        self.optimize_op = tf.train.AdamOptimizer(self.lr).minimize(self.ridge_loss) 
    

if __name__ == '__main__':
    from util import generate_1d_dataset
    from preprocess import Preprocessor
    from plot import plot_1d

    model = LinearModel((2,))
    
    n = 1000
    n_train = int(0.8 * n)

    xs, ys = generate_1d_dataset(n, 400)
    xs_train, ys_train = xs[:n_train], ys[:n_train]
    xs_test, ys_test = xs[n_train:], ys[n_train:]

    phi_xs_train = Preprocessor().polynomial(xs_train)
    phi_xs_test = Preprocessor().polynomial(xs_test)

    with tf.Session() as sess:
        model.fit(sess, phi_xs_train, ys_train)
        loss, preds = model.eval(sess, phi_xs_test, ys_test)
        print('loss = %f' % loss)
    plot_1d(xs_test, preds, ys_test)        
    
