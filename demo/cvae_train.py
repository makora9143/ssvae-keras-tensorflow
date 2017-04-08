import cvae
import numpy as np
import scipy.misc
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

nb_classes = 72
# input image dimensions
img_rows, img_cols = 32, 32
# img_rows, img_cols = 127, 128

ary = np.load("hiragana.npz")['arr_0'].reshape([-1, 127, 128]).astype(np.float32) / 15
X_train = np.zeros([nb_classes * 160, img_rows, img_cols], dtype=np.float32)
for i in range(nb_classes * 160):
    X_train[i] = scipy.misc.imresize(ary[i], (img_rows, img_cols), mode='F')
    # X_train[i] = ary[i]
Y_train = np.repeat(np.arange(nb_classes), 160)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)

X_train = X_train.reshape(X_train.shape[0], 32*32)
X_test = X_test.reshape(X_test.shape[0], 32*32)
input_shape = (img_rows, img_cols, 1)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)
model = cvae.VAE()

model.train(X_train, Y_train, X_test, Y_test, save=True)

