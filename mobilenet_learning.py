#import all necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
import pickle
import argparse
import cv2
import os

#argument parser for command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "path to input dataset")
ap.add_argument("-m", "--model", required = True,
	help = "path to model output")
ap.add_argument("-l", "--labelbin",
	help = "path to labelbinarizer output")
ap.add_argument("-e", "--epochs", default = 6,
	help = "number of epochs")
args = vars(ap.parse_args())

#initialize important constants, directories and defaults
IMAGE_DIMS = (128, 128, 3)
MODEL_PATH = os.path.join(args["model"],
	"model_" + str(datetime.date.today()))
if args["labelbin"] != None:
	LB_PATH = os.path.join(args["labelbin"],
		"labelbin_" + str(datetime.date.today()))
iteration = 0

#initialize labels and data lists
labels = []
data = []

#getting the directory
BASE_DIR = args["dataset"]
CONS_DIR = os.path.join(BASE_DIR, "consolidated")
CONS_LIST = os.listdir(CONS_DIR)
category_count = len(CONS_LIST)

#[INFO]shows the amount of categories found
print("Found ", category_count, " categories.")

#call for MobileNet weights
stock_mobilenet = MobileNet(weights = "imagenet", include_top = False,
	input_shape = IMAGE_DIMS, dropout = 0.02)
stock_mobilenet.trainable = False #Freeze MobileNet weights

#adding the MobileNet convolutions to the model
model = Sequential()
model.add(stock_mobilenet)

#softmax NN which does the classification
model.add(Flatten())
model.add(Activation("relu"))
model.add(Dense(210, activation= "softmax"))
model.summary()

#defining image data generator for augmentation
datagen = ImageDataGenerator(rotation_range = 20, width_shift_range = 0.15,
	height_shift_range = 0.15, shear_range = 0.05, zoom_range = 0.2,
	horizontal_flip = True)

#constructing the labels and data list
for category in CONS_LIST:
	if str(category) == ".DS_Store":
		continue
	SPECIES_DIR = os.path.join(CONS_DIR, str(category))
	SPECIES_LIST = os.listdir(SPECIES_DIR)
	amount = len(SPECIES_LIST)
	for num in range(amount):
		labels.append(category)

	#[INFO]shows the processing progress
	iteration = iteration + 1
	n = iteration / 40
	if iteration == 1 or n.is_integer() == True:
		print("Processing category " + str(iteration) + " ...")

    #constructing the data list containing all the bird images
	for bird in SPECIES_LIST:
		if str(bird) == ".DS_Store":
			continue
		imagePath = os.path.join(SPECIES_DIR, str(bird))
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		np_image = img_to_array(image)
		data.append(np_image)

#converting labels and data to numpy arrays
print("Processing labels & data arrays...")
labels = np.array(labels)
data = np.array(data, dtype="float32") / 255.0

#binarize the testLabels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

#train-test split
(train_data, test_data, train_labels, test_labels) = train_test_split(data,
	labels, test_size = 0.1, random_state = 42)

#compiling, running and testing the model
print("Compiling model...")
datagen.fit(train_data)
model.compile(loss = "categorical_crossentropy", optimizer = "sgd",
	metrics = ["accuracy"])
model.fit(datagen.flow(train_data, train_labels, batch_size = 10),
	steps_per_epoch = len(train_data) / 10,
	epochs = int(args["epochs"]), verbose = 1)
test_loss, test_acc = model.evaluate(test_data, test_labels)
print("Test acc: ", test_acc)

#saving model to specified location on disk
print("Saving model to disk...")
model.save(MODEL_PATH, save_format = "h5")

#saving labelbinarizer to specified location on disk
if args["labelbin"] != None:
	print("Saving labelbinarizer to disk...")
	location = open(LB_PATH, "wb")
	location.write(pickle.dumps(lb))
	location.close()
