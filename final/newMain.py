import cv2
import numpy as np
from sklearn import cluster
import RPi.GPIO as GPIO
import time

SDI   = 11
RCLK  = 13
SRCLK = 15

biggestDiceVal = 0
segCode = [0x3f,0x06,0x5b,0x4f,0x66,0x6d,0x7d,0x07,0x7f,0x6f,0x77,0x7c,0x39,0x5e,0x79,0x71,0x80]

def hc595_shift(dat):
    for bit in range(0, 8): 
        GPIO.output(SDI, 0x80 & (dat << bit))
        GPIO.output(SRCLK, GPIO.HIGH)
        time.sleep(0.001)
        GPIO.output(SRCLK, GPIO.LOW)
    GPIO.output(RCLK, GPIO.HIGH)
    time.sleep(0.001)
    GPIO.output(RCLK, GPIO.LOW)

# setup GPIO
GPIO.setmode(GPIO.BOARD)    #Number GPIOs by its physical location
GPIO.setup(SDI, GPIO.OUT)
GPIO.setup(RCLK, GPIO.OUT)
GPIO.setup(SRCLK, GPIO.OUT)
GPIO.output(SDI, GPIO.LOW)
GPIO.output(RCLK, GPIO.LOW)
GPIO.output(SRCLK, GPIO.LOW)

params = cv2.SimpleBlobDetector_Params()

params.filterByInertia
params.minInertiaRatio = 0.6

detector = cv2.SimpleBlobDetector_create(params)

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    frameBlurred = cv2.medianBlur(frame, 7)
    frameGray = cv2.cvtColor(frameBlurred, cv2.COLOR_BGR2GRAY)
    dots = detector.detect(frameGray)

    # Get dots
    X = []
    for b in dots:
        pos = b.pt

        if pos != None:
            X.append(pos)
    X = np.asarray(X)

    diceVal = []
    if len(X) > 0:
        # cluster dots
        clustering = cluster.DBSCAN(eps=40, min_samples=0).fit(X)

        numDiceVal = max(clustering.labels_) + 1

        diceVal = []

        # centroid calculation for diceVal
        for i in range(numDiceVal):
            XdiceVal = X[clustering.labels_ == i]

            centroidDiceVal = np.mean(XdiceVal, axis=0)

            diceVal.append([len(XdiceVal), *centroidDiceVal])
    else:
        diceVal = []
    
    biggestDiceVal = 0
    # Dots
    for b in dots:
        pos = b.pt
        r = b.size/2

        cv2.circle(frame, (int(pos[0]), int(pos[1])),
                   int(r), (0, 255, 242), 2)

        # Overlay diceVal number
        for d in diceVal:
            # Get textsize for text centering
            textsize = cv2.getTextSize(
                str(d[0]), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]
            
            if d[0] > biggestDiceVal:
                biggestDiceVal = d[0]

            cv2.putText(frame, str(d[0]),
                        (int(d[1] - textsize[0] / 2),
                        int(d[2] + textsize[1] / 2)),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
        # to be printed on seven segment
        print(biggestDiceVal)
        hc595_shift(segCode[biggestDiceVal])

        cv2.imshow("frame", frame)

        res = cv2.waitKey(1)

        # Stop if the user presses "q"
        if res & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows
GPIO.cleanup()