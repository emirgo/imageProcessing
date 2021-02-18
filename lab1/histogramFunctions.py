# Histogram using builtin fuctions

import cv2
import numpy as np
from matplotlib import pyplot as plt
  
# image path 
path = r'sample.jpg'

# using imread()   
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
cv2.imshow('Sample image',img)

dst = cv2.calcHist(img, [0], None, [256], [0,256])

plt.hist(img.ravel(),256,[0,256])
plt.title('Histogram for gray scale image')
plt.show()