#import all necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
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
IMAGE_DIMS = (69, 69, 3)

#initialize labels and data lists
labels = []
data = []

#getting the directory
BASE_DIR = args["dataset"]
CONS_DIR = os.path.join(BASE_DIR, "consolidated")
CONS_LIST = os.listdir(CONS_DIR)

#constructing the labels and data list
for category in CONS_LIST:
    labels.append(category)
    SPECIES_DIR = os.path.join(CONS_DIR, str(category))
    SPECIES_LIST = os.listdir(SPECIES_DIR)

    #specimen = os.path.join(SPECIES_DIR, str(SPECIES_LIST[0]))
    #specimen = cv2.imread(specimen)
    #cv2.imshow("", specimen)

    #constructing the data list containing all the bird images
    for bird in SPECIES_LIST:
        imagePath = os.path.join(SPECIES_DIR, str(bird))
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        #np_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        np_image = img_to_array(image)
        data.append(np_image)

#showing that the order of the labels and dataset is correct
print(labels[0])
plt.imshow(data[0])
plt.axis("off")
plt.show()
