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
ap.add_argument("-m", "--model", required = False,
    help = "path to already trained model, else default is used")
ap.add_argument("-l", "--labelbin", required = False,
    help = "path to defined labelbinarizer file, else default is used")
ap.add_argument("-i", "--image", required = False,
    help = "path to single image to be classified")
ap.add_argument("-f", "--folder", required = False,
    help = "path to folder containing images to be classified")
args = vars(ap.parse_args())

#raise custom error when neither an image nor a folder were specified
class MissingArgsError(Exception):
    """Raised when neither an image nor a folder were specified"""
    pass

if args["image"] == None and args["folder"] == None:
    raise MissingArgsError

#initialize important constants, directories and defaults
IMAGE_DIMS = (128, 128, 3)
DEF_MODEL = "/Users/arthur/ocv_install/matura/workspace/git/models"
DEF_MODEL = DEF_MODEL + "/def_2020-09-12"
DEF_LABELBIN = "/Users/arthur/ocv_install/matura/workspace/git/labelbins"
DEF_LABELBIN = DEF_LABELBIN + "/labelbin_2020-09-20"
FOLDER_DIR = args["folder"]

#load the trained MobileNet and the label binarizer
if args["model"] == None:
    model = load_model(DEF_MODEL)
else:
    model = load_model(args["model"])

if args["labelbin"] == None:
    lb = pickle.loads(open(DEF_LABELBIN, "rb").read())
else:
    lb = pickle.loads(open(args["labelbin"], "rb").read())

#define processing, classification and display function
def classify(image, model, labelbinarizer):
    "perform a classification operation on an image"

    #processing the image before feeding it to the model
    processed = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    processed = processed.astype("float32") / 255.0
    processed = img_to_array(processed)
    processed = np.expand_dims(processed, axis = 0)

    #classify the input image
    predictions = model.predict(processed)[0]
    likeliest = np.argmax(predictions)
    label = lb.classes_[likeliest]
    proba = predictions[likeliest]
    print(str(label) + ", " + str(proba))

    #display image with class and probability
    display = str(label) + " " + str(int(proba * 100)) + "%"
    cv2.putText(image, display, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 0, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

#handling single image inputs
if args["image"] != None:
    image = cv2.imread(args["image"])
    classify(image, model, lb)

#handling a folder containing multiple images
if FOLDER_DIR != None:
    FOLDER_LIST = os.listdir(FOLDER_DIR)
    for bird in FOLDER_LIST:
        if str(bird) == ".DS_Store":
            continue
        imagePath = os.path.join(FOLDER_DIR, str(bird))
        image = cv2.imread(imagePath)
        classify(image, model, lb)
