from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import cv2

img = io.imread('sample.png')

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