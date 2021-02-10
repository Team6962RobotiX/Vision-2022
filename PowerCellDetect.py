import cv2
import numpy as np
import math
from videocaptureasync import VideoCaptureAsync as vc
from networktables import NetworkTables
import threading
from cscore import *

cond = threading.Condition()
notified = [False]

def connectionListener(connected, info):
    print(info, '; Connected=%s' % connected)   
    with cond:
        notified[0] = True
        cond.notify()

NetworkTables.initialize(server='10.69.62.2')
NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)
NetworkTables.startClientTeam(6962)

with cond:
    print("Waiting")
    if not notified[0]:
        cond.wait()

simulate = True
sd = NetworkTables.getTable('SmartDashboard')

def callback(x):
    global H_low, H_high, S_low, S_high, V_low, V_high
    H_low = cv2.getTrackbarPos('low H', 'controls')
    H_high = cv2.getTrackbarPos('high H', 'controls')
    S_low = cv2.getTrackbarPos('low S', 'controls')
    S_high = cv2.getTrackbarPos('high S', 'controls')
    V_low = cv2.getTrackbarPos('low V', 'controls')
    V_high = cv2.getTrackbarPos('high V', 'controls')

cv2.namedWindow('controls', 2)
cv2.resizeWindow('controls', 550, 10)

H_low = 20
H_high = 50
S_low = 90
S_high = 255
V_low = 115
V_high = 255

cv2.createTrackbar('low H', 'controls', H_low, 179, callback)
cv2.createTrackbar('high H', 'controls', H_high, 179, callback)
cv2.createTrackbar('low S', 'controls', S_low, 255, callback)
cv2.createTrackbar('high S', 'controls', S_high, 255, callback)
cv2.createTrackbar('low V', 'controls', V_low, 255, callback)
cv2.createTrackbar('high V', 'controls', V_high, 255, callback)

def find_circles(image):
    blur_size = 13
    kernel_size = 5
    circle_thresh = 0.2

    result = image.copy()
    height, width, channels = image.shape
    circles = []

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    largeKernel = np.ones((kernel_size * 2, kernel_size * 2), np.uint8)

    blur = cv2.GaussianBlur(image, (blur_size, blur_size), 1)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower_hsv = np.array([H_low, S_low, V_low])
    upper_hsv = np.array([H_high, S_high, V_high])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    open_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, largeKernel)

    canny = cv2.Canny(open_mask, 0, 255)
    closed_canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(closed_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        hull_list = []
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            x, y, w, h = cv2.boundingRect(hull)
            r = (w + h) / 4
            isCircle = True
            for j in range(len(hull)):
                dist = math.sqrt(((x + w / 2) - hull[j][0][0]) ** 2 + ((y + h / 2) - hull[j][0][1]) ** 2)
                if (dist * (1 - circle_thresh) > r) or (dist * (1 + circle_thresh) < r):
                    isCircle = False
            if isCircle:
                circles.append([int((x + w / 2)), int((y + h / 2)), int(r)])
                hull_list.append(hull)

    # Displaying:
    '''for circle in circles:
        cv2.circle(result, (circle[0], circle[1]), circle[2], (255, 100, 75), int(circle[2] / 15))
        cv2.circle(result, (circle[0], circle[1]), 1, (255, 100, 75), int(circle[2] / 15))
    cv2.imshow("result",result)'''
    cv2.imshow("reult",closed_canny)
    return circles


#Running Code
'''cs = CameraServer.getInstance()
cs.enableLogging()'''
cvSource = CvSource("first", VideoMode(VideoMode.PixelFormat.kMJPEG, 640,480,30))#cs.putVideo("Ball", 640, 480)
print("HI")
name = MjpegServer(name = "name", port = 1182)
name.setSource(cvSource)
print("HI AGAIN")

cap = vc(src=1)
cap.start()
while True:
    ret, frame = cap.read()
    circles = find_circles(frame)
    for circle in circles:
        cv2.circle(frame, (circle[0], circle[1]), circle[2], (255, 100, 75), int(circle[2] / 15))
        cv2.circle(frame, (circle[0], circle[1]), 1, (255, 100, 75), int(circle[2] / 15))
        sd.putNumber("ballx",circle[0])
    cvSource.putFrame(frame)
    cv2.imshow("result",frame)
    cv2.waitKey(16)