
import tensorflow as tf
import numpy as np
import cPickle as pkl
from sklearn.manifold import TSNE
from ATDA import ATDA
from utils import *
import tensorlayer as tl
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# Process MNIST

mnist_train = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.float32)
mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.float32)
mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)


# Load MNIST-M
mnistm = pkl.load(open('mnistm_data.pkl'))
mnistm_train = mnistm['train']/255.
mnistm_test = mnistm['test']/255.
mnistm_valid = mnistm['valid']/255.
print mnist_train.shape
print mnist_test.shape
print mnistm_train.shape
print mnistm_test.shape
print mnist.train.labels.shape
print mnist.test.labels.shape

#imshow_grid(mnist_train)
#imshow_grid(mnistm_train)

'''
# Create a mixed dataset for TSNE visualization
num_test = 500
combined_test_imgs = np.vstack([mnist_test[:num_test], mnistm_test[:num_test]])
combined_test_labels = np.vstack([mnist.test.labels[:num_test], mnist.test.labels[:num_test]])
combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]),
        np.tile([0., 1.], [num_test, 1])])

'''

with tf.Session() as sess:
        model=ATDA(sess=sess)
        model.create_model()
        model.fit_ATDA(source_train=mnist_train, y_train=mnist.train.labels,
                       target_val=mnistm_test, y_val=mnist.test.labels,
                       target_data=mnistm_train,target_label=mnist.train.labels)
