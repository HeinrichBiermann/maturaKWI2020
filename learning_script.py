#import all necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import cv2
import os
import matplotlib.pyplot as plt

#argument parser for command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
#ap.add_argument("-m", "--model", required=True,
    #help="path to output model file")
args = vars(ap.parse_args())

#initialize important constants
IMAGE_DIMS = (64, 64, 3)
iteration = 0

#initialize labels and data lists
labels = []
data = []

#getting the directory
BASE_DIR = args["dataset"]
CONS_DIR = os.path.join(BASE_DIR, "consolidated")
CONS_LIST = os.listdir(CONS_DIR)

#[INFO]shows the amount of categories found
print("[INFO]Found " + str(len(CONS_LIST)) + " categories.")

#constructing the labels and data list
for category in CONS_LIST:
	SPECIES_DIR = os.path.join(CONS_DIR, str(category))
	SPECIES_LIST = os.listdir(SPECIES_DIR)
	amount = len(SPECIES_LIST)
	for num in range(amount):
		labels.append(category)

	#[INFO]shows the processing progress
	iteration = iteration + 1
	n = iteration / 20
	if iteration == 1 or n.is_integer() == True:
		print("[INFO]Processing category " + str(iteration) + " ...")

    #constructing the data list containing all the bird images
	for bird in SPECIES_LIST:
		imagePath = os.path.join(SPECIES_DIR, str(bird))
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
		np_image = img_to_array(image)
		data.append(np_image)

labels = np.array(labels)
data = np.array(data, dtype="float") / 255.0
print(labels[0])

#binarize the testLabels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

plt.imshow(data[0])
plt.axis("off")
plt.show()

(trainData, testData, trainLabels, testLabels) = train_test_split(data,
	labels, test_size = 0.2, random_state = 42)
