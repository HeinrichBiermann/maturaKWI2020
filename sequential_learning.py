#import all necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from keras.utils import to_categorical
import numpy as np
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

#defining the neural network, 1st convolution
model = Sequential()
model.add(Conv2D(32, (3,3), padding = "same", activation = "relu",
	input_shape = (64, 64, 1)))
model.add(BatchNormalization(axis= -1))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

#convoltion * 2 followed by pooling
model.add(Conv2D(64, (3,3), padding = "same", activation = "relu"))
model.add(BatchNormalization(axis= -1))
model.add(Conv2D(64, (5,5), padding = "same", activation = "relu"))
model.add(BatchNormalization(axis= -1))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

#convoltion * 2 followed by pooling
model.add(Conv2D(128, (3,3), padding = "same", activation = "relu"))
model.add(BatchNormalization(axis= -1))
model.add(Conv2D(128, (5,5), padding = "same", activation = "relu"))
model.add(BatchNormalization(axis= -1))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

#idk what this does, further documentation needed
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization(axis= -1))
model.add(Dropout(0.5))

#softmax nn which does the classification
model.add(Dense(210, activation="softmax"))
model.summary()

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
	if str(category) = ".DS_Store":
		continue
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
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		np_image = img_to_array(image)
		data.append(np_image)

#converting labels and data to numpy arrays
labels = np.array(labels)
data = np.array(data, dtype="float32") / 255.0

#binarize the testLabels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

#train-test split
(train_data, test_data, train_labels, test_labels) = train_test_split(data,
	labels, test_size = 0.2, random_state = 42)

#compiling, running and testing the model
model.compile(loss = "categorical_crossentropy", optimizer = "sgd",
	metrics = ["accuracy"])
model.fit(train_data, train_labels, batch_size = 100, epochs = 10, verbose = 1)
test_loss, test_acc = model.evaluate(test_data, test_labels)
print("Test acc: ", test_acc)
