import utils
import vae

(unlabeled_x, labeled_x, labeled_y), (validation_x, validation_y), (test_x, test_y) = utils.create_semisupervised_data()

m2 = vae.M2VAE()
print m2.generate()
#m2.train(unlabeled_x, labeled_x, labeled_y, validation_x, validation_y)
#m2.save()
#
#m2 = vae.M2VAE()
#m2.load()
