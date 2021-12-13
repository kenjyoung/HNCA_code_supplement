import pickle as pkl
import numpy as np
from mnist import read_labels_from_file, read_images_from_file

mnist_dir = "data/MNIST/"

train_labels = np.asarray(read_labels_from_file(mnist_dir+"train-labels-idx1-ubyte"))
test_labels = np.asarray(read_labels_from_file(mnist_dir+"t10k-labels-idx1-ubyte"))

train_images = np.asarray(read_images_from_file(mnist_dir+"train-images-idx3-ubyte"))/255
test_images = np.asarray(read_images_from_file(mnist_dir+"t10k-images-idx3-ubyte"))/255

train_images = np.reshape(train_images, (train_images.shape[0],train_images.shape[1]*train_images.shape[2]))
test_images = np.reshape(test_images, (test_images.shape[0],test_images.shape[1]*test_images.shape[2]))

with open(mnist_dir+"full_data", "wb") as f:
	pkl.dump({'train_img' : train_images,
			  'test_img'  : test_images,
			  'train_lbl' : train_labels,
			  'test_lbl'  : test_labels},f)

small_train_images = train_images[0:1000]
small_train_labels = train_labels[0:1000]
small_test_images = test_images[0:100]
small_test_labels = test_labels[0:100]

with open(mnist_dir+"small_data", "wb") as f:
	pkl.dump({'train_img' : small_train_images,
			  'test_img'  : small_test_images,
			  'train_lbl' : small_train_labels,
			  'test_lbl'  : small_test_labels},f)
	