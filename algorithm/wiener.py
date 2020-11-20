from skimage import color, data, restoration
from scipy.signal import convolve2d, gaussian
from numpy.fft import fft2, ifft2

import cv2 as cv
import numpy as np



# def wiener_filter(img, kernel, K):
#     kernel /= np.sum(kernel)
#     dummy = np.copy(img)
#     dummy = fft2(dummy)
#     kernel = fft2(kernel, s = img.shape)
#     kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
#     dummy = dummy * kernel
#     dummy = np.abs(ifft2(dummy))
#     return dummy
#
# def gaussian_kernel(kernel_size = 3):
#     h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
#     h = np.dot(h, h.transpose())
#     h /= np.sum(h)
#     return h
#
#
# def deblur():
#     img_orig = cv.imread('blur_3.jpg')
#     img_orig = color.rgb2gray(img_orig)
#
#     img = cv.imread('blur_3.jpg')
#     img = color.rgb2gray(img)
#     psf = np.ones((5, 5)) / 25
#     img = convolve2d(img, psf, 'same')
#
#     #Deconvolve an image using Richardson-Lucy deconvolution algorithm
#     # img_noisy = img.copy()
#     # img_noisy += (np.random.poisson(lam=25, size=img.shape) - 10) / 255.
#     # richardson_lucy = restoration.richardson_lucy(img_noisy, psf, iterations=30)
#
#     #deconvolve a noisy version of an image using Wiener and unsupervised Wiener algorithms
#     img += 0.1 * img.std() * np.random.standard_normal(img.shape)
#     #unsupervised_wiener, _ = restoration.unsupervised_wiener(img, psf)
#     wiener = restoration.wiener(img, psf, 1, clip=False)
#
#
#     # sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#     # sharpen = cv.filter2D(deconvolved, -1, sharpen_kernel)
#
#     # kernel = gaussian_kernel(3)
#     # filtered_img = wiener_filter(sharpen, kernel, K = 30)
#
#
#     cv.imshow('Original',img_orig)
#     #cv.imshow('richardson_lucy',richardson_lucy)
#     #cv.imshow('unsupervised_wiener',unsupervised_wiener)
#     cv.imshow('wiener',wiener)
#
#     cv.waitKey(0)
#
# deblur()
