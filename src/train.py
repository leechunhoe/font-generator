import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os

import numpy as np

import matplotlib.pyplot as plt

_DIR_X = "img/Songti_SC_Regular/"
_DIR_Y = "img/Gungseouche/"

SIZE = 250

X_TRAIN = []
Y_TRAIN = []

X_PREDICT = []
Y_PREDICT = []

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

def draw(image_matrix):
	plt.figure()
	plt.imshow(image_matrix)
	plt.colorbar()
	plt.grid(False)
	plt.show()

# Get file name of training and prediction datasets

y_names = []
train_names = []
predict_names = []

for filename in os.listdir(_DIR_Y):
	y_names.append(filename)

for filename in os.listdir(_DIR_X):
	if filename in y_names:
		train_names.append(filename)
	else:
		predict_names.append(filename)

# Load training dataset
for filename in train_names:
	x = get_image_vector(filename, _DIR_X)
	y = get_image_vector(filename, _DIR_Y)
	X_TRAIN.append(x)
	Y_TRAIN.append(y)

print("X_TRAIN loaded: " + str(len(X_TRAIN)))
print("Y_TRAIN loaded: " + str(len(Y_TRAIN)))

# train: 5013
# predict: 25750

# TODO: Train neural network

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(250, 250)),
	keras.layers.Dense(255, activation=tf.nn.relu),
	keras.layers.Dense(250 * 250, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

test_loss, test_acc = model.evaluate(X_TRAIN, Y_TRAIN)

print('Test accuracy:', test_acc)

model.fit(X_TRAIN, Y_TRAIN, epochs=5)

for filename in predict_names:
	x = get_image_vector(filename, _DIR_X)
	X_PREDICT.append(x)

predictions = model.predict(X_PREDICT)

print(predictions[300])

# TODO: 4. Predict from X_PREDICT



























