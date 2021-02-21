# Author: Emirhan Gocturk
# Description: Histogram

import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from skimage import io

img = io.imread('sample.png')

# To ascertain total numbers of rows and 
# columns of the image, size of the image 
x, y = img.shape 

# empty list to store the count 
# of each intensity value 
count =[] 

# empty list to store intensity 
# value 
r = [] 

# loop to traverse each intensity 
# value 
for k in range(0, 256): 
	r.append(k) 
	count1 = 0
	
	# loops to traverse each pixel in 
	# the image 
	for i in range(x): 
		for j in range(y): 
			if img[i, j]== k: 
				count1+= 1
	count.append(count1) 

# plotting the histogram 
plt.stem(r, count, markerfmt=" ") 
plt.xlabel('intensity value') 
plt.ylabel('number of pixels') 
plt.title('Histogram without functions') 
plt.show()


############################################
#############STRETCHING SECTION#############
############################################
min = 0
max = 0
threshold = 100

out = np.zeros((img.shape[0], img.shape[1]))

k = np.zeros(256)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        k[img[i][j]] += 1

for i in range(0, len(k)):
    if k[i] > threshold:
        min = i
        break

for i in range(len(k)-1,-1,-1):
    if k[i] > threshold:
        max = i
        break

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        out[i][j]=(( ((img[i][j]))-min )/(max-min))*255
        if out[i, j] > 255:
            out[i, j] = 255
        if out[i, j] < 0:
            out[i, j] = 0

cv2.imwrite("resultTest.png", out)


# To ascertain total numbers of rows and 
# columns of the image, size of the image 
x, y = out.shape 

# empty list to store the count 
# of each intensity value 
count =[] 

# empty list to store intensity 
# value 
r = [] 

# loop to traverse each intensity 
# value 
for k in range(0, 256): 
	r.append(k) 
	count1 = 0
	
	# loops to traverse each pixel in 
	# the image 
	for i in range(x): 
		for j in range(y): 
			if img[i, j]== k: 
				count1+= 1
	count.append(count1) 

# plotting the histogram 
plt.stem(r, count, markerfmt=" ") 
plt.xlabel('intensity value') 
plt.ylabel('number of pixels') 
plt.title('Histogram stretching without functions') 
plt.show()