import numpy as np
import cv2 as cv
import sys
import os
import time
from datetime import datetime
import hashlib

INPUT_FOLDER = os.path.join('static', 'input_images')
OUTPUT_FOLDER = os.path.join('static', 'output_images')

def compute_matches(image1,image2):


    img1 = cv.imread(image1,0)
    img2 = cv.imread(image2,0)


    # Match descriptors and measure elapsed time.
    start_time = time.time()

    fast = cv.FastFeatureDetector_create()
    brisk = cv.BRISK_create()
    # find and draw the keypoints
    kp1 = fast.detect(img1,None)
    kp2 = fast.detect(img2,None)

    kp1, desc1 = brisk.compute(img1,kp1)
    kp2, desc2 = brisk.compute(img2,kp2)

    # create BFMatcher object
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    matches = bf.match(desc1,desc2)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw matches.
    result = cv.drawMatches(img1,kp1,img2,kp2,matches[:100],None, flags=2)

    now = datetime.now() # current date and time
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    digest = hashlib.md5()
    digest.update(date_time.encode('utf-8'))
    output_path = os.path.join(OUTPUT_FOLDER, digest.hexdigest()+"matching_output.jpg")
    cv.imwrite(output_path, result)
    time.sleep(1)
    return [output_path, len(matches), elapsed_time*1000]

def compute_keypoints(image1):
    img1 = cv.imread(os.path.join(INPUT_FOLDER, image1),0)

    start_time = time.time()

    fast = cv.FastFeatureDetector_create()
    brisk = cv.BRISK_create()
    kp1 = fast.detect(img1,None)
    kp1, desc1 = brisk.compute(img1,kp1)
    result = cv.drawKeypoints(img1, kp1, None, color=(255,0,0))
    end_time = time.time()
    elapsed_time = end_time - start_time

    return [result, len(kp1), elapsed_time*1000]
