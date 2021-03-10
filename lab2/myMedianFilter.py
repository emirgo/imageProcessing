# Author: Emirhan Gocturk
# Description: Median filter with functions

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image

img = io.imread('sample.png')

data = np.array(img)

filter_size = 5

temp = []
indexer = filter_size // 2
data_final = []
data_final = np.zeros((len(data),len(data[0])))

# Iterate X dimension of the image
for i in range(len(data)):
    # Iterate Y dimension of the image (per X dimension)
    for j in range(len(data[0])):
        # Iterate over the kernel (filter size)
        for z in range(filter_size):
            # Check if dimension X + Z (a range of filter) is less than 0
            # Or is it X + Z - indexer (half of the filter size) is at the edge
            if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                for c in range(filter_size):
                    temp.append(0)
            else:
                # Same checks for Y dimension done as it was for X
                if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                    temp.append(0)
                else:
                    # 
                    for k in range(filter_size):
                        temp.append(data[i + z - indexer][j + k - indexer])

        temp.sort()
        data_final[i][j] = temp[len(temp) // 2]
        temp = []

final_image = Image.fromarray(data_final)
final_image.show()
