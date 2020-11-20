import numpy as np
import cv2 as cv
import sys
import os
import time
#from algorithm import deblur

INPUT_FOLDER = os.path.join('static', 'input_images')

def compute_matches(image1,image2):

    img1 = cv.imread(os.path.join(INPUT_FOLDER, image1),0)
    # Match descriptors and measure elapsed time.
    start_time = time.time()
    img2 = cv.imread(os.path.join(INPUT_FOLDER, image2),0)
    #img2 = deblur.deblurme(image2)

    fast = cv.FastFeatureDetector_create()
    brisk = cv.BRISK_create()
    # find and draw the keypoints
    kp1 = fast.detect(img1,None)
    kp2 = fast.detect(img2,None)

    kp1, desc1 = brisk.compute(img1,kp1)
    kp2, desc2 = brisk.compute(img2,kp2)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=False)

    matches = bf.match(desc1,desc2)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw matches.
    result = cv.drawMatches(img1,kp1,img2,kp2, matches[:100], None, flags=2)
    return [result, len(matches), elapsed_time*1000]

def compute_keypoints(image1):
    img1 = cv.imread(os.path.join(INPUT_FOLDER, image1),0)

    start_time = time.time()

    fast = cv.FastFeatureDetector_create()
    #brisk = cv.BRISK_create()
    kp1 = fast.detect(img1,None)
    #kp1, desc1 = brisk.compute(img1,kp1)
    #result = cv.drawKeypoints(img1, kp1, None, color=(255,0,0))
    end_time = time.time()
    elapsed_time = end_time - start_time

    return [result, len(kp1), elapsed_time*1000]
