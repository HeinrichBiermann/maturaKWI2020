#import all necessary libraries
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import pickle
import cv2
import os

#argument parser for command line arguments
ap = argparse.ArgumentParser()
ap.addargument("-m", "--model", required = True,
    help = "path to already trained model")
ap.addargument("-l", "--labelbin", required = True,
    help = "path to defined labelbinarizer file")
ap.addargument("-i", "--image", required = True,
    help = "path to image to be classified")
args = vars(ap.parse_args())

#initialize important constants
IMAGE_DIMS = (128, 128, 3)

#load image in case of single input
image = cv2.imread(args["image"])
copy = image.copy()

#image pre-processing for classification
image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
image = image.astype("float32") / 255.0
np_image = img_to_array(image)

#load the trained MobileNet and the label binarizer
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

#classify the input image
probs = model.predict(np_image)[0]
prediction = probs.argmax(axis = 0)
label = lb.classes_[prediction]

#display image with class and probability
