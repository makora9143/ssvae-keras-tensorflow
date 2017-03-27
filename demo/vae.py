# coding: utf-8

try:
    import gpu_config
    gpu_config.set_tensorflow([0])
except ImportError:
    print "no gpu"

import numpy as np

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from keras.metrics import categorical_crossentropy
from keras.metrics import categorical_accuracy

import matplotlib.pyplot as plt


class M2VAE(object):
    def __init__(self):
        self.nb_epoch = 100
        self.batch_size = 100
        self.z_dim = 50
        self.n_classes = 10
        self.image_size = 784
        self.learning_rate = 0.0003
        self.momentum = 0.9
        self.init_flg = False

        self.q = 'gaussian'
        self.p = 'bernoulli'

        self.sess = tf.Session()
        K.set_session(self.sess)

        self.build_model()

    def build_model(self):
        self.nn_q_z_given_xy()
        self.nn_q_y_given_x()
        self.nn_p_x_given_yz()


    def nn_q_z_given_xy(self):
        # $ q_\phi (z| x, y)$ 
        self.encoder_z_given_xy = Sequential()
        self.encoder_z_given_xy.add(Dense(500, input_dim=self.image_size + self.n_classes))
        self.encoder_z_given_xy.add(Activation('softplus'))
        self.encoder_z_given_xy.add(BatchNormalization())
        self.encoder_z_given_xy.add(Dense(500))
        self.encoder_z_given_xy.add(Activation('softplus'))
        self.encoder_z_given_xy.add(BatchNormalization())

        self.encoder_z_given_xy_dense1 = Sequential([Dense(self.z_dim, input_dim=500)])
        self.encoder_z_given_xy_dense2 = Sequential([Dense(self.z_dim, input_dim=500)])

    def nn_q_y_given_x(self):
        # $q_\phi (y|x)$
        self.encoder_y_given_x = Sequential()
        self.encoder_y_given_x.add(Dense(500, input_dim=self.image_size))
        self.encoder_y_given_x.add(Activation('softplus'))
        #self.encoder_y_given_x.add(BatchNormalization())
        self.encoder_y_given_x.add(Dense(500))
        self.encoder_y_given_x.add(Activation('softplus'))
        #self.encoder_y_given_x.add(BatchNormalization())
        self.encoder_y_given_x.add(Dense(self.n_classes, activation='softmax'))

    def nn_p_x_given_yz(self):
        # $p_\theta (x | y, z)$
        self.decoder_x_given_yz = Sequential()
        self.decoder_x_given_yz.add(Dense(500, input_dim=self.z_dim + self.n_classes))
        self.decoder_x_given_yz.add(Activation('softplus'))
        self.decoder_x_given_yz.add(BatchNormalization())
        self.decoder_x_given_yz.add(Dense(500))
        self.decoder_x_given_yz.add(Activation('softplus'))
        self.decoder_x_given_yz.add(BatchNormalization())
        self.decoder_x_given_yz.add(Dense(self.image_size, activation='sigmoid'))

    def _encode_z_given_xy(self, x_ph, y_ph):
        xy_l = tf.concat([x_ph, y_ph], axis=1)
        h_l = self.encoder_z_given_xy(xy_l)
        mean = self.encoder_z_given_xy_dense1(h_l)
        log_var2 = self.encoder_z_given_xy_dense2(h_l)
        return mean, log_var2

    def infer(self, x, y=None):
        x_ph = tf.placeholder(tf.float32, [None, self.image_size])
        y_ph = tf.placeholder(tf.float32, [None, self.n_classes])
        z_op = self._encode_z_given_xy(x_ph, y_ph)
        if y is None:
            y = self.classify(x)
        return self.sess.run(z_op, feed_dict={x_ph: x, y_ph: y, K.learning_phase(): 0})

    def _encode_y_given_x(self, x_ph):
        return self.encoder_y_given_x(x_ph)

    def classify(self, x):
        x_ph = tf.placeholder(tf.float32, [None, self.image_size])
        y_op = self._encode_y_given_x(x_ph)
        return self.sess.run(y_op, feed_dict={x_ph: x, K.learning_phase(): 1})

    def _sampling_z(self, mean, log_var2):
        epsilon = tf.random_normal((tf.shape(mean)[0], self.z_dim), mean=0., stddev=1.)
        z = tf.add(mean, tf.multiply(tf.exp(0.5 * log_var2), epsilon))
        return z

    def _decode_x_given_zy(self, z, y):
        zy = tf.concat([z, y], axis=1)
        if self.p == 'bernoulli':
            dec_mean = self.decoder_x_given_yz(zy)
            return dec_mean

    def generate(self, z=None, y=None):
        z_ph = tf.placeholder(tf.float32, [None, self.z_dim])
        y_ph = tf.placeholder(tf.float32, [None, self.n_classes])
        x = self._decode_x_given_zy(z_ph, y_ph)
        if y is None:
            y = []
            for i in range(10):
                for j in range(10):
                    a = [0] * 10
                    a[i] = 1
                    y.append(a)
            num = 100
        else:
            num = y.shape[0]
        if z is None:
            z = np.random.normal(loc=0., scale=1., size=(num, self.z_dim))
        return self.sess.run(x, feed_dict={z_ph: z, y_ph: y, K.learning_phase(): 1})

    def accuracy(self, x, t):
        x_ph = tf.placeholder(tf.float32, shape=[None, self.image_size])
        t_ph = tf.placeholder(tf.float32, shape=[None, self.n_classes])
        y_op = self._encode_y_given_x(x_ph)
        acc_value = categorical_accuracy(t_ph, y_op)
        return self.sess.run(acc_value, feed_dict={x_ph: x, t_ph: t, K.learning_phase(): 1})

    def create_unlabeled_data(self, x):
        new_x = tf.tile(x, [self.n_classes, 1])
        new_y = np.zeros((self.batch_size * self.n_classes, self.n_classes))
        for i in range(self.n_classes):
            for j in range(self.batch_size):
                new_y[i * self.batch_size + j][i] = 1
        return new_x, new_y

    def bernoulli_log_likelihood(self, x, y):
        return tf.reduce_sum(y * tf.log(x+1e-12) + (1 - y) * tf.log(1 - x+1e-12), axis=1)

    def gaussian_kl_divergence(self, mean, ln_var2):
        var2 = tf.exp(ln_var2)
        kld = tf.reduce_sum(mean * mean + var2 - ln_var2 - 1, axis=1) * 0.5
        return kld

    def py_log_likelihood(self, y):
        return tf.ones((tf.shape(y)[0],)) * tf.log(1. / self.n_classes)

    def categorical_log_likelihood(self, x, y):
        # $\log p(y)$ 
        return -categorical_crossentropy(y, x)

    def set_optimizer(self, loss_op):
        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.momentum)
        gvs = optimizer.compute_gradients(loss_op)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        train_step = optimizer.apply_gradients(capped_gvs)
        return train_step

    def reconstruct(self, x, y=None, N=50000):
        x_tilde = x
        unlabeled_flg = False
        if y is None:
            x_tilde, y = self.create_unlabeled_data(x)
            unlabeled_flg = True

        enc_mean, enc_var2 = self._encode_z_given_xy(x_tilde, y)
        pred = self._encode_y_given_x(x)

        z = self._sampling_z(enc_mean, enc_var2)

        dec_mean = self._decode_x_given_zy(z, y)

        ll = self.bernoulli_log_likelihood(dec_mean, x_tilde)
        kld = self.gaussian_kl_divergence(enc_mean, enc_var2)

        logpy = self.py_log_likelihood(y)

        elbo = ll + logpy - kld
        if unlabeled_flg:
            elbo = tf.transpose(tf.reshape(elbo, (self.n_classes, self.batch_size)))
            elbo = pred * (elbo - tf.log(pred + 1e-12))

        elbo = tf.reduce_sum(elbo) / self.batch_size
        loss_y = None
        if not unlabeled_flg:
            loss_y = self.categorical_log_likelihood(pred, y)
        return elbo, loss_y

    def initialize(self):
        if not self.init_flg:
            self.init = tf.global_variables_initializer()
            self.sess.run(self.init)
        self.init_flg = True


    def train(self, unlabeled_x, labeled_x, labeled_y, validation_x, validation_y):
        x_l_ph = tf.placeholder(tf.float32, shape=[None, self.image_size])
        x_u_ph = tf.placeholder(tf.float32, shape=[None, self.image_size])
        y_l_ph = tf.placeholder(tf.float32, shape=[None, self.n_classes])

        N = unlabeled_x.shape[0] + labeled_x.shape[0]

        elbo_l, loss_y = self.reconstruct(x_l_ph, y_l_ph, N) 

        elbo_u, _ = self.reconstruct(x_u_ph)

        J_alpha = elbo_l + elbo_u + N * 0.1 * loss_y

        train_step = self.set_optimizer(-J_alpha)

        self.initialize()


        for i in range(self.nb_epoch):
            idx = np.random.permutation(range(unlabeled_x.shape[0]))
            elbo_ls = []
            elbo_us = []
            for j in range(unlabeled_x.shape[0] / self.batch_size):
                unlabeled_batch_x = unlabeled_x[idx[j*self.batch_size: (j+1)*self.batch_size]]
                _, e_l, e_u  = self.sess.run([train_step, elbo_l, elbo_u],
                               feed_dict={x_l_ph: labeled_x,
                                          x_u_ph: unlabeled_batch_x,
                                          y_l_ph: labeled_y, 
                                          K.learning_phase(): 1
                                         })
                elbo_ls.append(e_l)
                elbo_us.append(e_u)

            acc = self.accuracy(validation_x, validation_y)
            print ("Epoch: %d/%d, ELBO(labeled): %g, ELBO(unlabeled): %g, acc: %g" % 
                    (i+1, self.nb_epoch, np.mean(elbo_ls), np.mean(elbo_us), np.mean(acc)))

    def save(self, filepath="model.ckpt"):
        saver = tf.train.Saver()
        saver.save(self.sess, filepath)

    def load(self, filepath="./model.ckpt"):
        saver = tf.train.Saver()
        saver.restore(self.sess, filepath) 


class M1VAE(object):
    def __init__(self):
        self.nb_epoch = 1000
        self.batch_size = 100
        self.z_dim = 50
        self.image_size = 784
        self.learning_rate = 0.0003
        self.momentum = 0.9
        self.init_flg = False

        self.q = 'gaussian'
        self.p = 'bernoulli'

        self.sess = tf.Session()
        K.set_session(self.sess)

        self.build_model()

    def build_model(self):
        self.nn_q_z_given_x()
        self.nn_p_x_given_z()


    def nn_q_z_given_x(self):
        # $ q_\phi (z| x)$ 
        self.encoder_z_given_x = Sequential()
        self.encoder_z_given_x.add(Dense(500, input_dim=self.image_size))
        self.encoder_z_given_x.add(Activation('softplus'))
        self.encoder_z_given_x.add(Dense(500))
        self.encoder_z_given_x.add(Activation('softplus'))

        self.encoder_z_given_xy_dense1 = Dense(self.z_dim)
        self.encoder_z_given_xy_dense2 = Dense(self.z_dim)

    def nn_p_x_given_z(self):
        # $p_\theta (x | z)$
        self.decoder_x_given_z = Sequential()
        self.decoder_x_given_z.add(Dense(500, input_dim=self.z_dim))
        self.decoder_x_given_z.add(Activation('softplus'))
        self.decoder_x_given_z.add(Dense(500))
        self.decoder_x_given_z.add(Activation('softplus'))
        self.decoder_x_given_z.add(Dense(self.image_size, activation='sigmoid'))

    def _encode_z_given_xy(self, x_ph):
        h_l = self.encoder_z_given_x(x_ph)
        mean = self.encoder_z_given_x_dense1(h_l)
        log_var2 = self.encoder_z_given_x_dense2(h_l)
        return mean, log_var2

    def infer(self, x):
        x_ph = tf.placeholder(tf.float32, [None, self.image_size])
        z_op = self._encode_z_given_x(x_ph)
        return self.sess.run(z_op, feed_dict={x_ph: x, K.learning_phase(): 1})

    def _sampling_z(self, mean, log_var2):
        epsilon = tf.random_normal((tf.shape(mean)[0], self.z_dim), mean=0., stddev=1.)
        z = tf.add(mean, tf.multiply(tf.exp(0.5 * log_var2), epsilon))
        return z

    def _decode_x_given_zy(self, z):
        if self.p == 'bernoulli':
            dec_mean = self.decoder_x_given_yz(z)
            return dec_mean

    def generate(self):
        z_ph = tf.placeholder(tf.float32, [None, self.z_dim])
        self.initialize()
        x = self._decode_x_given_z(z_ph)
        z = np.random.normal(loc=0., scale=1., size=(100, self.z_dim))
        return self.sess.run(x, feed_dict={z_ph: z, K.learning_phase(): 1})

    def bernoulli_log_likelihood(self, x, y):
        return tf.reduce_sum(y * tf.log(x+1e-12) + (1 - y) * tf.log(1 - x+1e-12), axis=1)

    def gaussian_kl_divergence(self, mean, ln_var2):
        var2 = tf.exp(ln_var2)
        kld = tf.reduce_sum(mean * mean + var2 - ln_var2 - 1, axis=1) * 0.5
        return kld

    def set_optimizer(self, loss_op):
        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.momentum)
        gvs = optimizer.compute_gradients(loss_op)
        capped_gvs = [(tf.clip_by_value(grad, -1., 5.), var) for grad, var in gvs]
        train_step = optimizer.apply_gradients(capped_gvs)
        return train_step

    def reconstruct(self, x):
        x_tilde = x

        enc_mean, enc_var2 = self._encode_z_given_x(x_tilde)

        z = self._sampling_z(enc_mean, enc_var2)

        dec_mean = self._decode_x_given_z(z)

        ll = self.bernoulli_log_likelihood(dec_mean, x_tilde)
        kld = self.gaussian_kl_divergence(enc_mean, enc_var2)

        elbo = ll - kld

        elbo = tf.reduce_sum(elbo) / self.batch_size
        return elbo

    def initialize(self):
        if not self.init_flg:
            self.init = tf.global_variables_initializer()
            self.sess.run(self.init)
        self.init_flg = True

    def train(self, unlabeled_x, validation_x):
        x_l_ph = tf.placeholder(tf.float32, shape=[None, self.image_size])
        y_l_ph = tf.placeholder(tf.float32, shape=[None, self.n_classes])

        elbo_l = self.reconstruct(x_l_ph) 

        train_step = self.set_optimizer(-elbo_l)

        self.initialize()

        for i in range(self.nb_epoch):
            idx = np.random.permutation(range(unlabeled_x.shape[0]))
            elbo_ls = []
            for j in range(unlabeled_x.shape[0] / self.batch_size):
                unlabeled_batch_x = unlabeled_x[idx[j*self.batch_size: (j+1)*self.batch_size]]
                _, e_l, e_u  = self.sess.run([train_step, elbo_l, elbo_u],
                               feed_dict={x_l_ph: labeled_x,
                                          K.learning_phase(): 1
                                         })
                elbo_ls.append(e_l)

            acc = self.accuracy(validation_x, validation_y)
            print ("Epoch: %d/%d, ELBO(labeled): %g" % 
                    (i+1, self.nb_epoch, np.mean(elbo_ls)))


    def save(self, filepath="model.ckpt"):
        saver = tf.train.Saver()
        saver.save(self.sess, filepath)

    def load(self, filepath="./model.ckpt"):
        saver = tf.train.Saver()
        saver.restore(self.sess, filepath) 
