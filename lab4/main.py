import argparse
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import RPi.GPIO as GPIO

camRead = 0
# Store circle count
circleCount = 0

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


def find_hough_circles(image, edge_image, r_min, r_max, delta_r, num_thetas, bin_threshold, post_process = True):
  #image size
  img_height, img_width = edge_image.shape[:2]
  
  # R and Theta ranges
  dtheta = int(360 / num_thetas)
  
  ## Thetas is bins created from 0 to 360 degree with increment of the dtheta
  thetas = np.arange(0, 360, step=dtheta)
  
  ## Radius ranges from r_min to r_max 
  rs = np.arange(r_min, r_max, step=delta_r)
  
  # Calculate Cos(theta) and Sin(theta) it will be required later
  cos_thetas = np.cos(np.deg2rad(thetas))
  sin_thetas = np.sin(np.deg2rad(thetas))
  
  # Evaluate and keep ready the candidate circles dx and dy for different delta radius
  # based on the the parametric equation of circle.
  # x = x_center + r * cos(t) and y = y_center + r * sin(t),  
  # where (x_center,y_center) is Center of candidate circle with radius r. t in range of [0,2PI)
  circle_candidates = []
  for r in rs:
    for t in range(num_thetas):
      #instead of using pre-calculated cos and sin theta values you can calculate here itself by following
      #circle_candidates.append((r, int(r*cos(2*pi*t/num_thetas)), int(r*sin(2*pi*t/num_thetas))))
      #but its better to pre-calculate and use it here.
      circle_candidates.append((r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))
  
  # Hough Accumulator, we are using defaultdic instead of standard dict as this will initialize for key which is not 
  # aready present in the dictionary instead of throwing exception.
  accumulator = defaultdict(int)
  
  for y in range(img_height):
    for x in range(img_width):
      if edge_image[y][x] != 0: #white pixel
        # Found an edge pixel so now find and vote for circle from the candidate circles passing through this pixel.
        for r, rcos_t, rsin_t in circle_candidates:
          x_center = x - rcos_t
          y_center = y - rsin_t
          accumulator[(x_center, y_center, r)] += 1 #vote for current candidate
  
  # Output image with detected lines drawn
  output_img = image.copy()
  # Output list of detected circles. A single circle would be a tuple of (x,y,r,threshold) 
  out_circles = []
  
  # Sort the accumulator based on the votes for the candidate circles 
  for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
    x, y, r = candidate_circle
    current_vote_percentage = votes / num_thetas
    if current_vote_percentage >= bin_threshold: 
      # Shortlist the circle for final result
      out_circles.append((x, y, r, current_vote_percentage))
      print(x, y, r, current_vote_percentage)
      
  
  # Post process the results, can add more post processing later.
  if post_process :
    pixel_threshold = 5
    postprocess_circles = []
    for x, y, r, v in out_circles:
      # Exclude circles that are too close of each other
      # all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc, v in postprocess_circles)
      # Remove nearby duplicate circles based on pixel_threshold
      if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for xc, yc, rc, v in postprocess_circles):
        postprocess_circles.append((x, y, r, v))
    out_circles = postprocess_circles
  

  # SQUARE SECTION
  output_img = cv2.GaussianBlur(output_img, (5, 5), 0)
  squares = []
  for gray in cv2.split(output_img):
    for thrs in range(0, 255, 26):
      if thrs == 0:
        bin = cv2.Canny(gray, 0, 50, apertureSize=5)
        bin = cv2.dilate(bin, None)
      else:
        retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
      
      contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
      for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
        if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
          cnt = cnt.reshape(-1, 2)
          max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
          if max_cos < 0.1:
            squares.append(cnt)

  cv2.drawContours(output_img, squares, -1, (0, 255, 0), 2)
  cv2.imshow('squares', output_img)
  cv2.waitKey(0)
    
  # Draw shortlisted circles on the output image
  for x, y, r, v in out_circles:
    output_img = cv2.circle(output_img, (x,y), r, (255, 0, 0), 2)
    circleCount += 1
    

  cv2.imshow('result', output_img)
  cv2.waitKey(0)
  
  return output_img, out_circles



img_path = "sample.png"
r_min = 10
r_max = 200
delta_r = 1
num_thetas = 100
bin_threshold = 0.4
min_edge_threshold = 100
max_edge_threshold = 200


input_img = cv2.imread(img_path)

# If camera is enabled, use camera
if camRead:
  # initialize the camera
  cam = cv2.VideoCapture(0)
  ret, input_img = cam.read()

  if ret:
      cv2.imshow('testImage',input_img)
      cv2.waitKey(0)
      cv2.destroyWindow('testImage')

#Edge detection on the input image
edge_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
#ret, edge_image = cv2.threshold(edge_image, 120, 255, cv2.THRESH_BINARY_INV)
edge_image = cv2.Canny(edge_image, min_edge_threshold, max_edge_threshold)

cv2.imshow('Edge Image', edge_image)
cv2.waitKey(0)

if edge_image is not None:

    print ("Detecting Hough Circles Started!")
    circle_img, circles = find_hough_circles(input_img, edge_image, r_min, r_max, delta_r, num_thetas, bin_threshold)
    
    cv2.imshow('Detected Circles', circle_img)
    cv2.waitKey(0)
    
    circle_file = open('circles_list.txt', 'w')
    circle_file.write('x ,\t y,\t Radius,\t Threshold \n')
    for i in range(len(circles)):
        circle_file.write(str(circles[i][0]) + ' , ' + str(circles[i][1]) + ' , ' + str(circles[i][2]) + ' , ' + str(circles[i][3]) + '\n')
    circle_file.close()
    
    if circle_img is not None:
        cv2.imwrite("circles_img.png", circle_img)

    servoPIN = 17
    PWM = 50
    # Servo section
    GPIO.setMode(GPIO.BCM)
    GPIO.setup(servoPIN, GPIO.OUT)

    p = GPIO.PWM(servoPIN, PWM)

    p.start(2.5)

    rotateDir = True
    rotateCounter = 2.5
    for i in range(circleCount):
      if rotateDir:
        rotateCounter += 2.5
        p.ChangeDutyCycle(rotateCounter)
        if rotateCounter == 12.5:
          rotateDir = False
      else:
        rotateCounter -= 2.5
        p.ChangeDutyCycle(rotateCounter)

        if rotateCounter <= 2.5:
          rotateDir = True

      time.sleep(0.5)

    p.stop()
    GPIO.cleanup()
    
else:
    print ("Error in input image!")
        
print ("Detecting Hough Circles Complete!")



