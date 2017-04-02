# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape


class VAE(object):
    def __init__(self):
        self.batch_size = 100
        self.z_dim = 2

        self.trained_flg = False

        # encode
        self.q_net = Sequential()
        self.q_net.add(Dense(500, activation='relu', input_dim=794))
        self.q_net.add(Dense(500, activation='relu'))

        self.q_net_mean = Sequential([Dense(self.z_dim, input_dim=500)])
        self.q_net_log_var2 = Sequential([Dense(self.z_dim, input_dim=500)])

        # decode
        self.p_net = Sequential()
        self.p_net.add(Dense(500, activation='relu', input_dim=self.z_dim+10))
        self.p_net.add(Dense(500, activation='relu'))
        self.p_net.add(Dense(784, activation='sigmoid'))

    def _encode(self, x_ph, y_ph):
        xy_ph = tf.concat([x_ph, y_ph], axis=-1)
        q_h = self.q_net(xy_ph)
        q_mean = self.q_net_mean(q_h)
        q_log_var2 = self.q_net_log_var2(q_h)
        return q_mean, q_log_var2

    def infer(self, x, y):
        x_ph = tf.placeholder(tf.float32, [None, 784])
        y_ph = tf.placeholder(tf.float32, [None, 10])

        q_mean, q_log_var2 = self._encode(x_ph, y_ph)
        with tf.Session() as sess:
            sess = self.initialize(sess)
            result = sess.run(q_mean, feed_dict={x_ph: x, y_ph: y})
        return result

    def _decode(self, z_ph, y_ph):
        zy_ph = tf.concat([z_ph, y_ph], axis=-1)
        p_mean = self.p_net(zy_ph)
        return p_mean

    def generate(self, z, y):
        z_ph = tf.placeholder(tf.float32, [None, self.z_dim])
        y_ph = tf.placeholder(tf.float32, [None, 10])

        p_mean = self._decode(z_ph, y_ph)
        with tf.Session() as sess:
            sess = self.initialize(sess)
            result = sess.run(p_mean, feed_dict={z_ph: z, y_ph: y})
        return result

    def _reconstruct(self, x_ph, y_ph):
        q_mean, q_log_var2 = self._encode(x_ph, y_ph)

        noise = tf.random_normal([tf.shape(x_ph)[0], self.z_dim])
        z = tf.add(q_mean, tf.multiply(tf.sqrt(tf.exp(q_log_var2)), noise))

        p_mean = self._decode(z, y_ph)

        return p_mean

    def reconstruct(self, x, y):
        x_ph = tf.placeholder(tf.float32, [None, 784])
        y_ph = tf.placeholder(tf.float32, [None, 10])

        p_mean = self._reconstruct(x_ph, y_ph)

        with tf.Session() as sess:
            sess = self.initialize(sess)
            result = sess.run(p_mean, feed_dict={x_ph: x, y_ph: y})
        return result

    def initialize(self, sess):
        if self.trained_flg:
            saver = tf.train.Saver()
            saver.restore(sess, self.filepath)
        else:
            sess.run(tf.global_variables_initializer())
        return sess

    def log_likelihood(self, x, recon_x):
        log_p_given_z = tf.reduce_sum(x * tf.log(recon_x + 1e-12) + \
                        (1 - x) * tf.log(1 - recon_x + 1e-12), axis=-1)
        return log_p_given_z

    def kl_divergence(self, mean, log_var2):
        D_KL = 0.5 * tf.reduce_sum(1 + log_var2 - mean **2 - tf.exp(log_var2), axis=-1)
        return D_KL

    def train(self, mnist, save=False, filepath='./cvae.ckpt'):
        x_ph = tf.placeholder(tf.float32, [None, 784])
        y_ph = tf.placeholder(tf.float32, [None, 10])

        q_mean, q_log_var2 = self._encode(x_ph, y_ph)

        noise = tf.random_normal([self.batch_size, self.z_dim])
        z = tf.add(q_mean, tf.multiply(tf.sqrt(tf.exp(q_log_var2)), noise))

        p_mean = self._decode(z, y_ph)

        log_p_given_z = self.log_likelihood(x_ph, p_mean)
        D_KL = self.kl_divergence(q_mean, q_log_var2)

        low_bound = tf.reduce_mean(log_p_given_z + D_KL)
        train_vae = tf.train.AdamOptimizer(0.01).minimize(-low_bound)

        self.trained_flg = save
        self.filepath = filepath

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(5):
                ave_loss = []
                for j in range(mnist.train.images.shape[0] / self.batch_size):
                    batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)
                    a = sess.run([train_vae, low_bound], feed_dict={x_ph: batch_xs, y_ph: batch_ys})
                    ave_loss.append(a[1])
                result = np.mean(ave_loss)
                print i+1, result
            if self.trained_flg:
                saver = tf.train.Saver()
                saver.save(sess, filepath)

    def load(self, filepath):
        self.filepath = filepath
        self.trained_flg = True



