import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Read the image for erosion
img = cv2.imread("sample.png",0)

#Acquire size of the image
m, n = img.shape 

# EROSION SECTION
# Kernel size
k = 5
SE= np.ones((k,k), dtype=np.uint8)
constant = (k-1)//2
# New image variable for eroded version
# of the sample image
imgErode = np.zeros((m,n), dtype=np.uint8)
# Erosion
for i in range(constant, m-constant):
  for j in range(constant,n-constant):
    temp = img[i-constant:i+constant+1, j-constant:j+constant+1]
    product = temp*SE
    imgErode[i, j] = np.min(product)

# Write to file
cv2.imwrite("eroded.png", imgErode)



# DILATION SECTION
imgDilate = np.zeros((m, n), dtype=np.uint8)
SED = np.array([[0,1,0], [1,1,1],[0,1,0]])
constant = 1

for i in range(constant, m-constant):
  for j in range(constant, n-constant):
    temp = img[i-constant:i+constant+1, j-constant:j+constant+1]
    product = temp*SED
    imgDilate[i, j] = np.max(product)

cv2.imwrite("dilated.png", imgDilate)



# Combined
imgDilate = np.zeros((m, n), dtype=np.uint8)
SED = np.array([[0,1,0], [1,1,1],[0,1,0]])
constant = 1

for i in range(constant, m-constant):
  for j in range(constant, n-constant):
    temp = imgErode[i-constant:i+constant+1, j-constant:j+constant+1]
    product = temp*SED
    imgDilate[i, j] = np.max(product)

cv2.imwrite("erosiondilated.png", imgDilate)