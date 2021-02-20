# Author: Emirhan Gocturk
# Description: Histogram

import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

img = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE) 

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

