# Author: Emirhan Gocturk
# Description: Median filter with functions

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# read image
img = io.imread('sample.png')
median = cv2.medianBlur(img, 5)
compare = np.concatenate((img, median), axis=1)

cv2.imshow('img', compare)
cv2.waitKey(0)
cv2.destroyAllWindows