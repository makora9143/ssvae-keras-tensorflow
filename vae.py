# coding: utf-8

import collections

import gpu_config
gpu_config.set_tensorflow([1])

import numpy as np

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from keras.metrics import categorical_crossentropy
from keras.metrics import categorical_accuracy

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class M2VAE(object):
    def __init__(self):
        self.nb_epoch = 1000
        self.batch_size = 100
        self.z_dim = 50
        self.n_classes = 10
        self.image_size = 784
        self.learning_rate = 0.00005
        self.momentum = 0.9

        self.q = 'gaussian'
        self.p = 'bernoulli'

        self.sess = tf.Session()
        K.set_session(self.sess)

        self.build_model()

    def build_model(self):
        self.nn_q_z_given_xy()
        self.nn_q_y_given_x()
        self.nn_p_x_given_yz()

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def nn_q_z_given_xy(self):
        # $ q_\phi (z| x, y)$ 
        self.encoder_z_given_xy = Sequential()
        self.encoder_z_given_xy.add(Dense(500, input_dim=self.image_size + self.n_classes))
        self.encoder_z_given_xy.add(BatchNormalization())
        self.encoder_z_given_xy.add(Activation('softplus'))
        self.encoder_z_given_xy.add(Dense(500))
        self.encoder_z_given_xy.add(BatchNormalization())
        self.encoder_z_given_xy.add(Activation('softplus'))

        self.encoder_z_given_xy_dense1 = Dense(z_dim)
        self.encoder_z_given_xy_dense2 = Dense(z_dim)

    def nn_q_y_given_x(self):
        # $q_\phi (y|x)$
        self.encoder_y_given_x = Sequential()
        self.encoder_y_given_x.add(Dense(500, input_dim=self.image_size))
        self.encoder_y_given_x.add(BatchNormalization())
        self.encoder_y_given_x.add(Activation('softplus'))
        self.encoder_y_given_x.add(Dense(500))
        self.encoder_y_given_x.add(BatchNormalization())
        self.encoder_y_given_x.add(Activation('softplus'))
        self.encoder_y_given_x.add(Dense(self.n_classes, activation='softmax'))

    def nn_p_x_given_yz(self):
        # $p_\theta (x | y, z)$
        self.decoder_x_given_yz = Sequential()
        self.decoder_x_given_yz.add(Dense(500, input_dim=z_dim + self.n_classes))
        self.decoder_x_given_yz.add(BatchNormalization())
        self.decoder_x_given_yz.add(Activation('softplus'))
        self.decoder_x_given_yz.add(Dense(500))
        self.decoder_x_given_yz.add(BatchNormalization())
        self.decoder_x_given_yz.add(Activation('softplus'))
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
        return self.sess.run(z_op, feed_dict={x_ph: x, y_ph: y, K.learning_phase(): 1})

    def _encode_y_given_x(self, x_ph):
        return self.encoder_y_given_x(x_ph)

    def classify(self, x):
        x_ph = tf.placeholder(tf.float32, [None, self.image_size])
        y_op = self._encode_y_given_x(x_ph)
        return self.sess.run(y_op, feed_dict={x_ph: x, K.learning_phase(): 1})

    def _sampling_z(self, mean, log_var2):
        epsilon = tf.random_normal((self.batch_size, self.z_dim), mean=0., stddev=1.)
        z = tf.add(mean, tf.multiply(tf.exp(0.5 * log_var2), epsilon))
        return z

    def _decode_x_given_zy(self, z, y):
        zy = tf.concat([z, y], axis=1)
        if self.p == 'bernoulli':
            dec_mean = decoder_x_given_yz(zy_l)
            return dec_mean

    def generate(self, label=None, random=True):
        z_ph = tf.placeholder(tf.float32, [None, self.z_dim])
        y_ph = tf.placeholder(tf.float32, [None, self.n_classes])
        zy = tf.concat([z_ph, y_ph], axis=1)
        x = self._decode_x_given_zy(zy)
        z = None
        y = None
        return self.sess.run(x, feed_dict={z_ph: z, y_ph: y, K.learning_phase(): 1}

    def accuracy(self, x, t):
        x_ph = tf.placeholder(tf.float32, shape=[None, self.image_size])
        t_ph = tf.placeholder(tf.float32, shape=[None, self.n_classes])
        y_op = self._encode_y_given_x(x_ph)
        acc_value = categorical_accuracy(t_ph, y_op)
        return self.sess.run(acc_value, feed_dict={x_ph: x, t_ph: t})

    def create_unlabeled_data(self, x):
        new_x = tf.tile(x, [self.n_classes, 1])
        new_y = np.zeros((batch_size * self.n_classes, self.n_classes))
        for i in range(self.n_classes):
            for j in range(batch_size):
                new_y[i * batch_size + j][i] = 1
        return new_x, new_y

    def bernoulli_log_likelihood(x, y):
        return tf.reduce_sum(y * tf.log(x+1e-12) + (1 - y) * tf.log(1 - x+1e-12), axis=1)

    def gaussian_kl_divergence(mean, ln_var2):
        var2 = tf.exp(ln_var2)
        kld = tf.reduce_sum(mean * mean + var2 - ln_var2 - 1, axis=1) * 0.5
        return kld

    def py_log_likelihood(y):
        return tf.ones((tf.shape(y)[0],)) * tf.log(1. / self.n_classes)

    def categorical_log_likelihood(x, y):
        # $\log p(y)$ 
        return -categorical_crossentropy(y, x)

    def set_optimizer(self, loss_op):
        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.momentum)
        gvs = optimizer.compute_gradients(loss_op)
        capped_gvs = [(tf.clip_by_value(grad, -1., 5.), var) for grad, var in gvs]
        return optimizer.apply_gradients(capped_gvs)

    def train(self, unlabeled_x, labeled_x, labeled_y, validation_x, validation_y):
        x_l_ph = tf.placeholder(tf.float32, shape=[None, self.image_size])
        x_u_ph = tf.placeholder(tf.float32, shape=[None, self.image_size])
        y_l_ph = tf.placeholder(tf.float32, shape=[None, self.n_classes])

        x_u_ph2, self.y_u_ph = create_unlabeled_data(x_u_ph)
        enc_mean_l, self.enc_var2_l = self._encode_z_given_xy(x_l_ph, y_l_ph)
        enc_mean_u, self.enc_var2_u = self._encode_z_given_xy(x_u_ph2, y_u_ph)

        # ### $y$の推論
        pred_l = _encode_y_given_x(x_l_ph)
        pred_u = _encode_y_given_x(x_u_ph)

        # ## サンプリング
        z_l = self._sampling_z(self.enc_mean_l, self.enc_var2_l)
        z_u = self._sampling_z(self.enc_mean_u, self.enc_var2_u)


        # ## 生成(復元) 
        self.dec_mean_l = self._decode_x_given_zy(self.z_l, self.y_l_ph)
        self.dec_mean_u = self._decode_x_given_zy(self.z_u, self.y_u_ph)
        ll_l = self.bernoulli_log_likelihood(dec_mean_l, x_l_ph)
        ll_u = self.bernoulli_log_likelihood(dec_mean_u, x_u_ph2)

        kld_l = self.gaussian_kl_divergence(enc_mean_l, enc_var_l)
        kld_u = self.gaussian_kl_divergence(enc_mean_u, enc_var_u)

        logpy_l = self.py_log_likelihood(y_l_ph)
        logpy_u = self.py_log_likelihood(y_u_ph)

        # ### 目的関数

        elbo_l = tf.reduce_sum(ll_l + logpy_l - kld_l) / batch_size

        elbo_u_xy = tf.transpose(tf.reshape(ll_u + logpy_u - kld_u, (self.n_classes, self.batch_size)))
        elbo_u = tf.reduce_sum(pred_u * (elbo_u_xy - tf.log(pred_u + 1e-12))) / batch_size

        J = elbo_l + elbo_u

        loss_y = categorical_log_likelihood(pred_l, y_l_ph)

        N = unlabeled_x.shape[0] + labeled_x.shape[0]

        J_alpha = J + N * 0.1 * loss_y

        train_step = self.set_optimizer(-J_alpha)

        for i in range(nb_epoch):
            idx = np.random.permutation(range(unlabeled_size))
            elbo_ls = []
            elbo_us = []
            loss_ys = []
            for j in range(unlabeled_size / batch_size):
                unlabeled_batch_x = unlabeled_x[idx[j*batch_size: (j+1)*batch_size]]
                _, e_l, e_u, l_y = sess.run([train_step, elbo_l, elbo_u, loss_y],
                               feed_dict={x_l_ph: labeled_x,
                                          x_u_ph: unlabeled_batch_x,
                                          y_l_ph: labeled_y, 
                                          K.learning_phase(): 1
                                         })
                elbo_ls.append(e_l)
                elbo_us.append(e_u)
                loss_ys.append(l_y)

            acc = self.accuracy(validation_x, validation_y)
            print ("Epoch: %d/%d, ELBO(labeled): %g, ELBO(unlabeled): %g, logp(y|x): %g, acc: %g" % 
                    (i+1, nb_epoch, np.mean(elbo_ls), np.mean(elbo_us), np.mean(loss_ys), acc))


