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

def normalise(raw_matrix):
	background_matrix = np.zeros((SIZE, SIZE))
	rx = raw_matrix.shape[0]
	ry = raw_matrix.shape[1]
	x1 = (SIZE) // 2 - (rx // 2)
	x2 = x1 + rx
	y1 = (SIZE) // 2 - (ry // 2)
	y2 = y1 + ry
	background_matrix[x1:x2, y1:y2] = raw_matrix
	return background_matrix

def get_image_vector(filename, directory):
	if filename.endswith(".bmp"):
		image = Image.open(directory + filename)
		image_matrix = normalise(np.array(image))
		return image_matrix.flatten()
	else:
		return None

for filename in os.listdir(_DIR_X):
	x = get_image_vector(filename, _DIR_X)
	X.append(x)

for filename in os.listdir(_DIR_Y):
	y = get_image_vector(filename, _DIR_Y)
	Y.append(y)



# print(Y[0])

# plt.figure()
# plt.imshow(Y[90])
# plt.colorbar()
# plt.grid(False)
# plt.show()

