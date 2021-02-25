# Author: Emirhan Gocturk
# Description: Histogram

import cv2
import numpy as np
from matplotlib import pyplot as plt

from skimage import exposure

img = cv2.imread('sample.png', cv2.IMREAD_GRAYSCALE)

dst = cv2.calcHist(img, [0], None, [256], [0,256])

plt.hist(img.ravel(),256,[0,256])
plt.xlabel('intensity value') 
plt.ylabel('number of pixels') 
plt.title('Histogram with functions')
plt.show()

p2, p98 = np.percentile(img, (2, 98))
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

plt.hist(img_rescale.ravel(),256,[0,256])
plt.xlabel('intensity value') 
plt.ylabel('number of pixels') 
plt.title('Histogram with functions')
plt.show()

cv2.imwrite("resultFunctions.png", img_rescale)