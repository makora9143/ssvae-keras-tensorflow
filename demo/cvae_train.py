import cvae

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)

model = cvae.VAE()

model.train(mnist, save=True)

