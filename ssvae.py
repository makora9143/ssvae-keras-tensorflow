
# coding: utf-8

# In[2]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from sklearn import preprocessing
import gpu_config
gpu_config.set_tensorflow([1])
import tensorflow as tf
import numpy as np
import collections
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from keras.metrics import categorical_crossentropy

from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
sess = tf.InteractiveSession()
K.set_session(sess)


# <img  src="images/Screen Shot 0029-02-22 at 13.55.59.png" width=300px align="left"/>

# ### hyperparameter

# In[3]:

nb_epoch = 1000
batch_size = 100
z_dim = 50
labeled_size = 100


# # モデルの定義

# <img  src="images/Screen Shot 0029-02-22 at 13.57.53.png"width=500px align="left"/>

# ## Encoder

# $ q_\phi (z| x, y)$ 

# In[4]:

encoder_z_given_xy = Sequential()
encoder_z_given_xy.add(Dense(500, input_dim=794))
encoder_z_given_xy.add(BatchNormalization())
encoder_z_given_xy.add(Activation('softplus'))
encoder_z_given_xy.add(Dense(500))
encoder_z_given_xy.add(BatchNormalization())
encoder_z_given_xy.add(Activation('softplus'))

encoder_z_given_xy_dense1 = Dense(z_dim)
encoder_z_given_xy_dense2 = Dense(z_dim)


# $q_\phi (y|x)$

# In[5]:

encoder_y_given_x = Sequential()
encoder_y_given_x.add(Dense(500, input_dim=784))
encoder_y_given_x.add(BatchNormalization())
encoder_y_given_x.add(Activation('softplus'))
encoder_y_given_x.add(Dense(500, input_dim=784))
encoder_y_given_x.add(BatchNormalization())
encoder_y_given_x.add(Activation('softplus'))
encoder_y_given_x.add(Dense(10, activation='softmax'))


# ## Decoder 

# $p_\theta (x | y, z)$

# In[6]:

decoder_x_given_yz = Sequential()
decoder_x_given_yz.add(Dense(500, input_dim=z_dim+10))
decoder_x_given_yz.add(BatchNormalization())
decoder_x_given_yz.add(Activation('softplus'))
decoder_x_given_yz.add(Dense(500))
decoder_x_given_yz.add(BatchNormalization())
decoder_x_given_yz.add(Activation('softplus'))
decoder_x_given_yz.add(Dense(784, activation='sigmoid'))


# # 処計算

# In[7]:

def create_unlabeled_data(x):
    new_x = tf.tile(x, [10, 1])
    new_y = np.zeros((batch_size * 10, 10))
    for i in range(10):
        for j in range(batch_size):
            new_y[i * batch_size + j][i] = 1
    return new_x, new_y


# In[8]:

x_l_ph = tf.placeholder(tf.float32, shape=[None, 784])
x_u_ph = tf.placeholder(tf.float32, shape=[None, 784])
y_l_ph = tf.placeholder(tf.float32, shape=[None, 10])

x_u_ph2, y_u_ph = create_unlabeled_data(x_u_ph)


# ## 推論

# ### $z$の推論 

# ラベルあり

# In[9]:

xy_l = tf.concat([x_l_ph, y_l_ph], axis=1)
h_l = encoder_z_given_xy(xy_l)
enc_mean_l = encoder_z_given_xy_dense1(h_l)
enc_var_l = encoder_z_given_xy_dense2(h_l)


# ラベルなし

# In[10]:

xy_u = tf.concat([x_u_ph2, y_u_ph], axis=1)
h_u = encoder_z_given_xy(xy_u)
enc_mean_u = encoder_z_given_xy_dense1(h_u)
enc_var_u = encoder_z_given_xy_dense2(h_u)


# ### $y$の推論

# ラベルあり

# In[11]:

pred_l = encoder_y_given_x(x_l_ph)


# ラベルなし

# In[12]:

pred_u = encoder_y_given_x(x_u_ph)


# ## サンプリング

# ラベルあり

# In[13]:

epsilon_l = tf.random_normal((batch_size, z_dim), mean=0., stddev=1.)
z_l = tf.add(enc_mean_l, tf.multiply(tf.exp(0.5 * enc_var_l), epsilon_l))


# ラベルなし

# In[14]:

epsilon_u = tf.random_normal((batch_size*10, z_dim), mean=0., stddev=1.)
z_u = tf.add(enc_mean_u, tf.multiply(tf.exp(0.5 * enc_var_u), epsilon_u))


# ## 生成(復元) 

# ラベルあり

# In[15]:

zy_l = tf.concat([z_l, y_l_ph], axis=1)
dec_mean_l = decoder_x_given_yz(zy_l)


# ラベルなし

# In[16]:

zy_u = tf.concat([z_u, y_u_ph], axis=1)
dec_mean_u = decoder_x_given_yz(zy_u)


# ## 変分下限

# ラベルあり
# $$
# {\cal L}_{labeled}(x, y) = {\mathbb E}_{z \sim q_{\phi}(z|x, y)} \biggl[ \ln p_{\theta}(x|y,z) + \ln p_{\theta}(y)\biggr]- D_{KL}(q_{\phi}(z|x,y)\|p_{\theta}(z))
# $$

# ラベルなし
# $$
# {\cal L}_{unlabeled}(x) = {\mathbb E}_{y \sim q_{\phi}(y|x)}\biggl[{\cal L}_{labeled}(x,y) - \ln q_{\phi}(y|x)\biggr]
# $$

# ### 再構成誤差 

# In[17]:

def bernoulli_log_likelihood(x, y):
    return tf.reduce_sum(y * tf.log(x+1e-12) + (1 - y) * tf.log(1 - x+1e-12), axis=1)


# In[18]:

ll_l = bernoulli_log_likelihood(dec_mean_l, x_l_ph)
ll_u = bernoulli_log_likelihood(dec_mean_u, x_u_ph2)


# ### KLダイバージェンス

# In[19]:

def gaussian_kl_divergence(mean, ln_var2):
    var2 = tf.exp(ln_var2)
    kld = tf.reduce_sum(mean * mean + var2 - ln_var2 - 1, axis=1) * 0.5
    return kld


# In[20]:

kld_l = gaussian_kl_divergence(enc_mean_l, enc_var_l)
kld_u = gaussian_kl_divergence(enc_mean_u, enc_var_u)


# ### $\log p(y)$ 

# In[21]:

logpy_l = tf.ones((tf.shape(y_l_ph)[0],)) * tf.log(1 / 10.)
logpy_u = tf.ones((tf.shape(y_u_ph)[0],)) * tf.log(1 / 10.)


# ### 目的関数

# In[22]:

elbo_l = tf.reduce_sum(ll_l + logpy_l - kld_l) / batch_size
elbo_u = tf.reduce_sum(pred_u * (tf.transpose(tf.reshape(ll_u + logpy_u - kld_u, (10, 100))) - tf.log(pred_u + 1e-12))) / batch_size

J = elbo_l + elbo_u


# In[23]:

def categorical_log_likelihood(x, y):
    return -categorical_crossentropy(y, x)
loss_y = categorical_log_likelihood(pred_l, y_l_ph)


# In[24]:

J_alpha = J + 50000 * 0.1 * loss_y


# # その他

# ## Optimizer 

# In[25]:

adam =  tf.train.AdamOptimizer(0.00005, beta1=0.9)
gvs = adam.compute_gradients(-J_alpha)
capped_gvs = [(tf.clip_by_value(grad, -1., 5.), var) for grad, var in gvs]
train_step = adam.apply_gradients(capped_gvs)


# ## 生成用 

# In[26]:

z_ph = tf.placeholder(tf.float32, [None, z_dim])
sample_zy_ph = tf.concat([z_ph, y_l_ph], axis=1)
generated_x = decoder_x_given_yz(sample_zy_ph)


# In[27]:

from keras.metrics import categorical_accuracy as accuracy
x_test_ph = tf.placeholder(tf.float32, shape=[None, 784])
y_test_ph = tf.placeholder(tf.float32, shape=[None, 10])
acc_value = accuracy(y_test_ph, encoder_y_given_x(x_test_ph))


# ## 半教師ありのデータセット作成

# In[28]:


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
images = np.concatenate([mnist.train.images, mnist.validation.images, mnist.test.images])
labels = np.concatenate([mnist.train.labels, mnist.validation.labels, mnist.test.labels])
train_x, test_x, train_y, test_y = train_test_split(images, labels, 
                                                    test_size=20000./images.shape[0])
unlabeled_x, labeled_x, unlabeled_y, labeled_y = train_test_split(
                                                    train_x, train_y, 
                                                    test_size=float(labeled_size)/train_x.shape[0])
validation_x, test_x, validation_y, test_y = train_test_split(
                                                    test_x, test_y, 
                                                    test_size=0.5)
unlabeled_size = train_x.shape[0] - labeled_size


# In[29]:

collections.Counter(np.argmax(labeled_y, axis=1))


# # 実行

# In[30]:

sess.run(tf.global_variables_initializer())


# In[31]:

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

    acc = sess.run(acc_value, feed_dict={
                            x_test_ph: validation_x, 
                            y_test_ph: validation_y,
                            K.learning_phase(): 1})
    print "Epoch: %d/%d, ELBO(labeled): %g, ELBO(unlabeled): %g, logp(y|x): %g, acc: %g" % (i+1, nb_epoch, np.mean(elbo_ls), np.mean(elbo_us), np.mean(loss_ys), acc)


# In[57]:

def plot(imgs):
    final = np.zeros((28 * 10, 28 * 10))
    for i in range(10):
        for j in range(10):
            final[i * 28: (i+1) * 28, j*28: (j+1)* 28] = imgs[i*10 + j].reshape((28,28))
    plt.figure(figsize=(5,5))
    plt.imshow(final)
    plt.gray()
    plt.show()


# In[89]:

mean, var = sess.run([enc_mean_l, enc_var_l], feed_dict={x_l_ph: test_x[:1], y_l_ph: test_y[:1], K.learning_phase(): 1})
test_z = mean + var * np.random.normal(loc=0., scale=1., size=(100, z_dim))
y_lable = [[0,0,0,0,0,1,0,0,0,0]]*100


# In[90]:

reconstruction = sess.run(dec_mean_l, feed_dict={x_l_ph: mnist.test.images[:100], y_l_ph: mnist.test.labels[:100], K.learning_phase(): 1})
result = sess.run(generated_x, feed_dict={z_ph: test_z, y_l_ph: y_lable, K.learning_phase(): 1})


# In[88]:

plot(result)


# In[91]:

plot(result)


# In[85]:

plot(mnist.test.images[:100])
plot(reconstruction)


# In[ ]:



