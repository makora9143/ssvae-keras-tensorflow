# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers import Conv2D, Deconv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.metrics import categorical_crossentropy



class VAE(object):
    def __init__(self, mode='mnist'):
        self.batch_size = 16
        self.epochs = 300

        self.trained_flg = False
        self.build_model(mode)

    def build_model(self, mode):
        if mode == 'mnist':
            self.nb_classes = 10
            self.z_dim = 15
            self.img_row, self.img_col = 28, 28
            self.build_mnist_model()
        elif mode == 'hiragana':
            self.nb_classes = 70
            self.z_dim = 150
            self.img_row, self.img_col = 32, 32
            self.build_hiragana_model()
        else:
            return 'error'

    def build_hiragana_model(self):
        K.clear_session()
        # encode
        def my_init(shape, name=None):
            return K.random_normal(shape, stddev=0.1)
        with tf.variable_scope('encoder', reuse=True):
            self.q_net = Sequential()
            self.q_net.add(Reshape((32, 32, 1), input_shape=(32*32,), name='hiragana_encode_reshape'))
            self.q_net.add(Conv2D(32, 3, padding='same', activation='relu', name='hiragana_encode_conv1'))
            self.q_net.add(Conv2D(32, 5, strides=(2, 2), padding='same', activation='relu', name='hiragana_encode_conv2'))
            self.q_net.add(Conv2D(64, 3, padding='same', activation='relu', name='hiragana_encode_conv3'))
            self.q_net.add(Conv2D(64, 5, strides=(2, 2), padding='same', activation='relu', name='hiragana_encode_conv4'))
            self.q_net.add(Conv2D(128, 3, padding='same', activation='relu', name='hiragana_encode_conv5'))
            self.q_net.add(Conv2D(128, 5, strides=(2, 2), padding='same', activation='relu', name='hiragana_encode_conv6'))
            self.q_net.add(Flatten(name='hiragana_encode_flatten'))
            self.q_net.add(Dense(1024, activation='relu', name='hiragana_encode_dense'))

            self.q_net_mean = Sequential([Dense(150, input_dim=1024+70, name='hiragana_encode_dense_mean')])
            self.q_net_log_var2 = Sequential([Dense(150, input_dim=1024+70, name='hiragana_encode_dense_var')])

        # decode
        with tf.variable_scope('decoder', reuse=True):
            self.p_net = Sequential()
            self.p_net.add(Dense(1024, activation='relu', input_dim=150+70, name='hiragana_decode_dense1'))
            self.p_net.add(Dense(4*4*128, activation='relu', name='hiragana_decode_dense2'))
            self.p_net.add(Reshape((4, 4, 128), name='hiragana_decode_reshape'))

            self.p_net.add(Conv2D(128, 3, padding='same', activation='relu', name='hiragana_decode_conv1'))
            self.p_net.add(Deconv2D(64, 5, strides=(2, 2), padding='same', activation='relu', name='hiragana_decode_conv2'))
            self.p_net.add(Conv2D(64, 3, padding='same', activation='relu', name='hiragana_decode_conv3'))
            self.p_net.add(Deconv2D(32, 5, strides=(2, 2), padding='same', activation='relu', name='hiragana_decode_conv4'))
            self.p_net.add(Conv2D(32, 3, padding='same', activation='relu', name='hiragana_decode_conv5'))
            self.p_net.add(Deconv2D(1, 5, strides=(2, 2), padding='same', activation='sigmoid', name='hiragana_decode_conv6'))
            self.p_net.add(Flatten(name='hiragana_decode_flatten'))

        # cnn
        with tf.variable_scope('cnn', reuse=True):
            self.cnn = Sequential()
            self.cnn.add(Reshape((32, 32, 1), input_shape=(32*32,), name='hiragana_cnn_reshape'))
            self.cnn.add(Conv2D(32, 3, kernel_initializer=my_init, padding='same', activation='relu', name='hiragana_cnn_conv1'))
            self.cnn.add(Conv2D(32, 3, kernel_initializer=my_init, padding='same', activation='relu', name='hiragana_cnn_conv2'))
            self.cnn.add(MaxPooling2D(name='hiragana_cnn_maxpool1'))
            self.cnn.add(Dropout(0.5, name='hiragana_cnn_dropout1'))

            self.cnn.add(Conv2D(64, 3, padding='same', kernel_initializer=my_init, activation='relu', name='hiragana_cnn_conv3'))
            self.cnn.add(Conv2D(64, 3, padding='same', kernel_initializer=my_init, activation='relu', name='hiragana_cnn_conv4'))
            self.cnn.add(MaxPooling2D(name='hiragana_cnn_maxpool2'))
            self.cnn.add(Dropout(0.5, name='hiragana_cnn_dropout2'))

            self.cnn.add(Flatten(name='hiragana_cnn_flatten'))
            self.cnn.add(Dense(256, activation='relu', kernel_initializer=my_init, name='hiragana_cnn_dense1'))
            self.cnn.add(Dropout(0.5, name='hiragana_cnn_dropout3'))
            self.cnn.add(Dense(70, activation='softmax', kernel_initializer=my_init, name='hiragana_cnn_dense2'))

    def build_mnist_model(self):

        K.clear_session()
        self.q_net = Sequential()
        self.q_net.add(Reshape((28, 28, 1), input_shape=(784,), name='mnist_encode_reshape'))
        self.q_net.add(Conv2D(32, 5, strides=(2, 2), padding='same', activation='relu', name='mnist_encode_conv1'))
        self.q_net.add(Conv2D(64, 5, strides=(2, 2), padding='same', activation='relu', name='mnist_encode_conv2'))
        self.q_net.add(Flatten(name='mnist_encode_flatten'))
        self.q_net.add(Dense(1024, activation='relu', name='mnist_encode_dense'))

        self.q_net_mean = Sequential([Dense(15, input_dim=1024+10, name='mnist_encode_dense_mean')])
        self.q_net_log_var2 = Sequential([Dense(15, input_dim=1024+10, name='mnist_encode_dense_var')])

        # decode
        self.p_net = Sequential()
        self.p_net.add(Dense(1024, activation='relu', input_dim=15+10, name='mnist_decode_dense1'))
        self.p_net.add(Dense(7*7*64, activation='relu', name='mnist_decode_dense2'))
        self.p_net.add(Reshape((7, 7, 64), name='mnist_decode_reshape'))

        self.p_net.add(Deconv2D(32, 5, strides=(2, 2), padding='same', activation='relu', name='mnist_decode_conv1'))
        self.p_net.add(Deconv2D(1, 5, strides=(2, 2), padding='same', activation='sigmoid', name='mnist_decode_conv2'))
        self.p_net.add(Flatten(name='mnist_decode_flatten'))

        # cnn
        self.cnn = Sequential()
        self.cnn.add(Reshape((28, 28, 1), input_shape=(784,), name='mnist_cnn_reshape'))
        self.cnn.add(Conv2D(32, 5, padding='same', activation='relu', name='mnist_cnn_conv1'))
        self.cnn.add(MaxPooling2D(name='mnist_cnn_maxpooling1'))
        self.cnn.add(Conv2D(64, 5, padding='same', activation='relu', name='mnist_cnn_conv2'))
        self.cnn.add(MaxPooling2D(name='mnist_cnn_maxpooling2'))
        self.cnn.add(Flatten(name='mnist_cnn_flatten'))
        self.cnn.add(Dense(1024, activation='relu', name='mnist_cnn_dense1'))
        self.cnn.add(Dropout(0.5, name='mnist_cnn_dropout'))
        self.cnn.add(Dense(10, activation='softmax', name='mnist_cnn_dense2'))

    def _encode(self, x_ph, y_ph):
        q_h = self.q_net(x_ph)
        xy_ph = tf.concat([q_h, y_ph], axis=-1)
        q_mean = self.q_net_mean(xy_ph)
        q_log_var2 = self.q_net_log_var2(xy_ph)
        return q_mean, q_log_var2

    def infer(self, x, y):
        x_ph = tf.placeholder(tf.float32, [None, self.img_row*self.img_col])
        y_ph = tf.placeholder(tf.float32, [None, self.nb_classes])

        q_mean, q_log_var2 = self._encode(x_ph, y_ph)
        with tf.Session() as sess:
            sess = self.initialize(sess)
            K.set_session(sess)
            result = sess.run(q_mean, feed_dict={x_ph: x, y_ph: y})
        return result

    def _decode(self, z_ph, y_ph):
        zy_ph = tf.concat([z_ph, y_ph], axis=-1)
        p_mean = self.p_net(zy_ph)
        return p_mean

    def generate(self, z, y):
        z_ph = tf.placeholder(tf.float32, [None, self.z_dim])
        y_ph = tf.placeholder(tf.float32, [None, self.nb_classes])

        p_mean = self._decode(z_ph, y_ph)
        with tf.Session() as sess:
            sess = self.initialize(sess)
            K.set_session(sess)
            result = sess.run(p_mean, feed_dict={z_ph: z, y_ph: y})
        return result

    def _reconstruct(self, x_ph, y_ph):
        q_mean, q_log_var2 = self._encode(x_ph, y_ph)

        noise = tf.random_normal([tf.shape(x_ph)[0], self.z_dim])
        z = tf.add(q_mean, tf.multiply(tf.sqrt(tf.exp(q_log_var2)), noise))

        p_mean = self._decode(z, y_ph)

        return p_mean

    def reconstruct(self, x, y):
        x_ph = tf.placeholder(tf.float32, [None, self.img_row*self.img_col])
        y_ph = tf.placeholder(tf.float32, [None, self.nb_classes])

        p_mean = self._reconstruct(x_ph, y_ph)

        with tf.Session() as sess:
            sess = self.initialize(sess)
            K.set_session(sess)
            result = sess.run(p_mean, feed_dict={x_ph: x, y_ph: y})
        return result

    def initialize(self, sess):
        if self.trained_flg:
            #saver = tf.train.Saver()
            #saver.restore(sess, self.filepath)
            return self.sess
        else:
            sess.run(tf.global_variables_initializer())
        return sess

    def predict(self, x):
        x_ph = tf.placeholder(tf.float32, [None, self.img_row*self.img_col])
        pred = self.cnn(x_ph)

        with tf.Session() as sess:
            sess = self.initialize(sess)
            result = sess.run(pred, feed_dict={x_ph: x, K.learning_phase(): 0})
        return result

    def log_likelihood(self, x, recon_x):
        log_p_given_z = tf.reduce_sum(x * tf.log(recon_x + 1e-12) + \
                        (1 - x) * tf.log(1 - recon_x + 1e-12), axis=-1)
        return log_p_given_z

    def kl_divergence(self, mean, log_var2):
        D_KL = 0.5 * tf.reduce_sum(1 + log_var2 - mean **2 - tf.exp(log_var2), axis=-1)
        return D_KL

    def train(self, X_train, y_train, X_test, y_test, save=False, filepath='./cvae.ckpt'):
        x_ph = tf.placeholder(tf.float32, [None, self.img_col*self.img_row])
        y_ph = tf.placeholder(tf.float32, [None, self.nb_classes])

        q_mean, q_log_var2 = self._encode(x_ph, y_ph)

        noise = tf.random_normal([self.batch_size, self.z_dim])
        z = tf.add(q_mean, tf.multiply(tf.sqrt(tf.exp(q_log_var2)), noise))

        p_mean = self._decode(z, y_ph)

        log_p_given_z = self.log_likelihood(x_ph, p_mean)
        D_KL = self.kl_divergence(q_mean, q_log_var2)

        low_bound = tf.reduce_mean(log_p_given_z + D_KL)
        train_vae = tf.train.AdamOptimizer(0.0003).minimize(-low_bound)

        # cnn
        pred = self.cnn(x_ph)
        loss = tf.reduce_mean(categorical_crossentropy(y_ph, pred))
        train_cnn = tf.train.AdadeltaOptimizer(1.0).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y_ph,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.trained_flg = save
        self.filepath = filepath

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(self.epochs):
                ave_loss = []
                ave_cnn = []
                idx = np.random.permutation(range(X_train.shape[0]))
                for j in range(X_train.shape[0] / self.batch_size):
                    batch_xs, batch_ys = X_train[j*self.batch_size: (j+1)*self.batch_size], y_train[j*self.batch_size: (j+1)*self.batch_size]
                    _, vae_loss, _, cnn_loss = sess.run([train_vae, low_bound, train_cnn, loss],
                            feed_dict={x_ph: batch_xs, y_ph: batch_ys, K.learning_phase(): 1})
                    ave_loss.append(vae_loss)
                    ave_cnn.append(cnn_loss)
                result = np.mean(ave_loss)
                cnn_result = np.mean(ave_cnn)
                print i+1, result, cnn_result
                print("test accuracy %g" % sess.run(accuracy, feed_dict={x_ph: X_test, y_ph: y_test, K.learning_phase(): 0}))
            if self.trained_flg:
                saver = tf.train.Saver()
                saver.save(sess, filepath)

    def load(self, filepath):
        self.filepath = filepath
        self.trained_flg = True
        self.sess = tf.Session()
        #saver = tf.train.import_meta_graph(filepath)
        #saver.restore(self.sess, tf.train.latest_checkpoint('./'))
        saver = tf.train.Saver()
        saver.restore(self.sess, filepath)

    def close(self):
        self.sess.close()
