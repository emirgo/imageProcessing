# Author: Emirhan Gocturk
# Description: Camera capture
import cv2

# initialize the camera
cam = cv2.VideoCapture(0)
ret, image = cam.read()

if ret:
    cv2.imshow('testImage',image)
    cv2.waitKey(0)
    cv2.destroyWindow('testImage')
    cv2.imwrite('testImage.jpg',image)
cam.release()