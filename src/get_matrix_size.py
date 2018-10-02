# Purpose: To get suitable size of X, Y matrix

import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os

import numpy as np

import matplotlib.pyplot as plt

_DIR_X = "img/Arial Unicode/"
_DIR_Y = "img/UnGungseo/"

X = []
Y = []

largest = 0

for filename in os.listdir(_DIR_X):
	if filename.endswith(".bmp"):
		im = Image.open(_DIR_X + filename)
		p = np.array(im)
		l = p.shape[0]
		w = p.shape[1]
		if l > largest:
			largest = l
		if w > largest:
			largest = w

for filename in os.listdir(_DIR_Y):
	if filename.endswith(".bmp"):
		im = Image.open(_DIR_Y + filename)
		p = np.array(im)
		l = p.shape[0]
		w = p.shape[1]
		if l > largest:
			largest = l
		if w > largest:
			largest = w

print("largest size = " + largest)