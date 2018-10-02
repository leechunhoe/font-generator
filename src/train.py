import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os

import numpy as np

import matplotlib.pyplot as plt

_DIR_X = "img/Arial Unicode/"
_DIR_Y = "img/UnGungseo/"

SIZE = 250

X = []
Y = []

largest = 0

for filename in os.listdir(_DIR_X):
	if filename.endswith(".bmp"):
		im = Image.open(_DIR_X + filename)
		p = np.array(im)


for filename in os.listdir(_DIR_Y):
	if filename.endswith(".bmp"):
		im = Image.open(_DIR_Y + filename)
		p = np.array(im)




# print(plt.__version__)

# fashion_mnist = keras.datasets.fashion_mnist

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# plt.figure()
# plt.imshow(p)
# plt.colorbar()
# plt.grid(False)
# plt.show()