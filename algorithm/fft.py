# import dependencies
import cv2
import numpy
from imutils import paths
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix # confusion_matrix - to check the correct
import datetime
import os
import sys

INPUT_FOLDER = os.path.join('static', 'input_images')


# function for display image
def display(title, img, max_size=200000):
    scale = numpy.sqrt(min(1.0, float(max_size)/(img.shape[0]*img.shape[1]))) # rescale image
    shape = (int(scale*img.shape[1]), int(scale*img.shape[0])) # reshape image
    img = cv2.resize(img, shape) # resize image
    cv2.imshow(title, img) # show image

# compute and classify image if blur or not, return img_fft, result
def evaluate(img, thresh):
    img_gry = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # convert image to grayscale
    # apply fft formula
    rows, cols = img_gry.shape
    crow, ccol = int(rows/2), int(cols/2)
    f = numpy.fft.fft2(img_gry)
    fshift = numpy.fft.fftshift(f)
    fshift[crow-75:crow+75, ccol-75:ccol+75] = 0
    f_ishift = numpy.fft.ifftshift(fshift)
    img_fft = numpy.fft.ifft2(f_ishift)
    img_fft = 20*numpy.log(numpy.abs(img_fft))
    result = numpy.mean(img_fft)
    return img_fft, result, result < thresh

# helper function for morphing image
def morphology(msk):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    msk = cv2.erode(msk, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, kernel)
    msk[msk < 128] = 0
    msk[msk > 127] = 255
    return msk

# removing border function
def remove_border(msk, width=50):
    dh, dw = map(lambda i: i//width, msk.shape)
    h, w = msk.shape
    msk[:dh, :] = 255
    msk[h-dh:, :] = 255
    msk[:, :dw] = 255
    msk[:, w-dw:] = 255
    return msk

# function for mask blurring
def blur_mask(img, thresh):
    msk, val, blurry = evaluate(img, thresh)
    msk = cv2.convertScaleAbs(255-(255*msk/numpy.max(msk)))
    msk[msk < 50] = 0
    msk[msk > 127] = 255
    msk = remove_border(msk)
    msk = morphology(msk)
    result = numpy.sum(msk)/(255.0*msk.size)
    return msk, result, blurry


def detect_blur(image, threshold):
    image = cv2.imread(os.path.join(INPUT_FOLDER, image))
    msk, val, blurry = evaluate(image, int(threshold))
    return msk, val, blurry
