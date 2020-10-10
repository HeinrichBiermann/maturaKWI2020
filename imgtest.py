import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = False)
ap.add_argument("-e", "--epochs", required = False)
args = vars(ap.parse_args())

def sliding_window(image, step, size):
    for y in range(0, image.shape[0] - size[1], step):
        for x in range(0, image.shape[1] - size[0], step):
            yield (x, y, image[y:y + size[0], x:x + size[1]])

def img_pyramid(image, scale = 1.5, minSize = (40, 40)):
    yield image
    while True:
        width = int(image.shape[1] // scale)
        height = int(image.shape[0] // scale)
        dims = (width, height)
        image = cv2.resize(image, dims)
        if width < minSize[1] or height < minSize[0]:
            break
        yield image


epochs = args["epochs"]
print(epochs)

image = cv2.imread(args["image"])
#cv2.imshow("", image)
#cv2.waitKey(0)
print(image.shape)

pyramid = img_pyramid(image)
for thing in pyramid:
    cv2.imshow("", thing)
    cv2.waitKey(0)
for val in sliding_window(image, 20, (40, 40)):
    print(val)
