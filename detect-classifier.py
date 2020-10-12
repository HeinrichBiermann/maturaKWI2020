#import all necessary libraries
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import argparse
import pickle
import cv2
import os

#argument parser for command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model",
    help = "path to already trained model, else default is used")
ap.add_argument("-l", "--labelbin",
    help = "path to defined labelbinarizer file, else default is used")
ap.add_argument("-i", "--image",
    help = "path to single image to be classified")
ap.add_argument("-f", "--folder",
    help = "path to folder containing images to be classified")
ap.add_argument("-c", "--minconf", type = float, default = 0.95,
    help = "minimum probability to sort out weaker detections")
ap.add_argument("-b", "--boxes", type = int, default = 1,
    help = "set the number of detection boxes to be displayed")
args = vars(ap.parse_args())

#raise custom error when neither an image nor a folder were specified
class MissingArgsError(Exception):
    """Raised when neither an image nor a folder were specified"""
    pass

if args["image"] == None and args["folder"] == None:
    raise MissingArgsError

#initialize important constants, directories and defaults
IMAGE_DIMS = (128, 128, 3)
DEF_MODEL = "/Users/arthur/ocv_install/matura/workspace/git"
DEF_MODEL = DEF_MODEL + "/models/def_2020-10-10"
DEF_LABELBIN = "/Users/arthur/ocv_install/matura/workspace/git/labelbins"
DEF_LABELBIN = DEF_LABELBIN + "/labelbin_2020-09-20"
FOLDER_DIR = args["folder"]
BOXES = args["boxes"]

WIDTH = 400
RESCALE = 1.5
WIN_STEP = 16
ROI_SIZE = (128, 128)

#load the trained MobileNet and the label binarizer
if args["model"] == None:
    model = load_model(DEF_MODEL)
else:
    model = load_model(args["model"])

if args["labelbin"] == None:
    lb = pickle.loads(open(DEF_LABELBIN, "rb").read())
else:
    lb = pickle.loads(open(args["labelbin"], "rb").read())

#define sliding window function
def sliding_window(image, step, size):
    for y in range(0, image.shape[0] - size[1], step):
        for x in range(0, image.shape[1] - size[0], step):
            yield (x, y, image[y:y + size[0], x:x + size[1]])

#define image pyramid function
def img_pyramid(image, scale = 1.5, minSize = (40, 40)):
    yield image
    while True:
        width = int(image.shape[1] / scale)
        height = int(image.shape[0] / scale)
        dims = (width, height)
        image = cv2.resize(image, dims)
        if width < minSize[1] or height < minSize[0]:
            break
        yield image

def detect_classify(original, model, labelbinarizer):
    coords = []
    labels = []
    probs = []

    scale = WIDTH / original.shape[1]
    dims = (WIDTH, int(original.shape[0] * scale))
    original = cv2.resize(original, dims)

    pyramid = img_pyramid(original, RESCALE, ROI_SIZE)
    for image in pyramid:
        scale = WIDTH / float(image.shape[1])

        for (x, y, roi) in sliding_window(image, WIN_STEP, ROI_SIZE):
            x = int(x * scale)
            y = int(y * scale)
            w = int(ROI_SIZE[0] * scale)
            h = int(ROI_SIZE[1] * scale)

            roi = roi.astype("float32") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis = 0)

            predictions = model.predict(roi)[0]
            likeliest = np.argmax(predictions)
            label = lb.classes_[likeliest]
            proba = predictions[likeliest]

            if proba > args["minconf"]:
                labels.append(label)
                coords.append((x, y, x + w, y + h))
                probs.append(proba)

    final_label = max(set(labels), key = labels.count)

    if BOXES == 1:
        coords_dict = dict(zip(labels, coords))
        probs_dict = dict(zip(labels, probs))
        box = coords_dict[final_label]
        acc = probs_dict[final_label]
        print(str(final_label) + ", " + str(acc))

        (startX, startY, endX, endY) = box
        cv2.rectangle(original, (startX, startY), (endX, endY), (0, 0, 255), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        display = str(final_label) + " " + str(int(acc * 100)) + "%"
        cv2.putText(original, display, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    else:
        data = pd.DataFrame()
        data["Label"] = labels
        data["Coordinate"] = coords
        data["Probability"] = probs
        data.set_index("Label", inplace = True)
        detects = pd.DataFrame(data.loc[final_label])
        detects.sort_values(by = ["Probability"], inplace = True,
            ascending = False)
        boxes = detects.iloc[:BOXES]["Coordinate"]
        acc = detects.iloc[0]["Probability"]
        print(str(final_label) + ", " + str(acc))

        for box in boxes:
            (startX, startY, endX, endY) = box
            cv2.rectangle(original, (startX, startY), (endX, endY),
                (255, 180, 0), 2)

        display = str(final_label) + " " + str(int(acc * 100)) + "%"
        cv2.putText(original, display, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    return original

#handling single image inputs
if args["image"] != None:
    print("!!!Single input mode!!!")
    original = cv2.imread(args["image"])
    processed = detect_classify(original, model, lb)
    cv2.imshow("Output", processed)
    cv2.waitKey(0)

#handling a folder containing multiple images
if FOLDER_DIR != None:
    print("!!!Multiple input mode!!!")
    FOLDER_LIST = os.listdir(FOLDER_DIR)
    BIRDS_LIST = []
    for bird in FOLDER_LIST:
        if str(bird) == ".DS_Store":
            continue
        imagePath = os.path.join(FOLDER_DIR, str(bird))
        original = cv2.imread(imagePath)
        processed = detect_classify(original, model, lb)
        BIRDS_LIST.append(processed)

    for image in BIRDS_LIST:
        cv2.imshow("Output", image)
        cv2.waitKey(0)
