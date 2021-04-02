# Author: Emirhan Gocturk
# Description: Erosion and dilation 
#               using functions

import cv2
import numpy as np

img = cv2.imread('sample.png', 0)

kernel  = np.ones((5,5), np.uint8)

imgErosion = cv2.erode(img, kernel, iterations=1)
imgDilation = cv2.dilate(img, kernel, iterations=1)

cv2.imshow('Input', img)
cv2.imshow('Erosion', imgErosion)
cv2.imshow('Dilation', imgDilation)
imgFinal = cv2.erode(img, kernel, iterations=1)
imgFinal = cv2.dilate(imgFinal, kernel, iterations=1)
cv2.imshow('Final', imgFinal)

cv2.waitKey(0)