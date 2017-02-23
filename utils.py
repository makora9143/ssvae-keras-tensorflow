# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import collections

from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split


def create_semisupervised_data(dataset='mnist', label_nums=100):
    # 半教師ありのデータセット作成
    if dataset == 'mnist':
        mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
        images = np.concatenate([mnist.train.images, mnist.validation.images, mnist.test.images])
        labels = np.concatenate([mnist.train.labels, mnist.validation.labels, mnist.test.labels])
        train_x, test_x, train_y, test_y = train_test_split(images, labels, 
                                                            test_size=2./7)
        unlabeled_x, labeled_x, unlabeled_y, labeled_y = train_test_split(
                                                            train_x, train_y, 
                                                            test_size=float(label_nums)/50000)
        validation_x, test_x, validation_y, test_y = train_test_split(
                                                            test_x, test_y, 
                                                            test_size=0.5)
        unlabeled_size = train_x.shape[0] - label_nums
        collections.Counter(np.argmax(labeled_y, axis=1))
        return (unlabeled_x, labeled_x, labeled_y), (validation_x, validation_y), (test_x, test_y)


def plot(imgs):
    final = np.zeros((28 * 10, 28 * 10))
    for i in range(10):
        for j in range(10):
            final[i * 28: (i+1) * 28, j*28: (j+1)* 28] = imgs[i*10 + j].reshape((28,28))
    plt.figure(figsize=(5,5))
    plt.imshow(final)
    plt.gray()
    plt.show()
