import utils
import vae

(unlabeled_x, labeled_x, labeled_y), (validation_x, validation_y), (test_x, test_y) = utils.create_semisupervised_data()

#m2 = vae.M2VAE()
#m2.train(unlabeled_x, labeled_x, labeled_y, validation_x, validation_y)
#m2.save(filepath='demo.ckpt')
#
m2 = vae.M2VAE()
m2.load(filepath='./demo.ckpt')
import numpy as np
print np.mean(np.argmax(m2.classify(test_x), axis=1) == np.argmax(test_y, axis=1))

