import numpy as np
import cv2
import math
import gc
cv2.useOptimized()
################### WEIRD RASPBERRY PI STUFF ###################

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

cvSource = CvSource("first", VideoMode(VideoMode.PixelFormat.kMJPEG, 640, 480, 30))  # cs.putVideo("Ball", 640, 480)

name = MjpegServer(name="name", port=1182)
name.setSource(cvSource)

cap = vc(src=0)
cap.start()
################### WEIRD RASPBERRY PI STUFF ###################


HSV_low = [15, 80, 100]
HSV_high = [50, 255, 255]

blur_size = 5
kernel_size = 5
circle_threshold = 0.21


# Called when the trackbars change
'''def callback(x):
    global HSV_low, HSV_high, kernel_size, circle_threshold
    HSV_low[0] = cv2.getTrackbarPos('low H', 'controls')
    HSV_high[0] = cv2.getTrackbarPos('high H', 'controls')
    HSV_low[1] = cv2.getTrackbarPos('low S', 'controls')
    HSV_high[1] = cv2.getTrackbarPos('high S', 'controls')
    HSV_low[2] = cv2.getTrackbarPos('low V', 'controls')
    HSV_high[2] = cv2.getTrackbarPos('high V', 'controls')
    kernel_size = cv2.getTrackbarPos('Kernel Size', 'controls')
    circle_threshold = cv2.getTrackbarPos('Circle Threshold', 'controls') / 100
    if circle_threshold is 0:
        circle_threshold = 1


# Window for trackbars
cv2.namedWindow('controls', 2)
cv2.resizeWindow('controls', 550, 10);

# Defining trackbars
cv2.createTrackbar('low H', 'controls', HSV_low[0], 179, callback)
cv2.createTrackbar('high H', 'controls', HSV_high[0], 179, callback)
cv2.createTrackbar('low S', 'controls', HSV_low[1], 255, callback)
cv2.createTrackbar('high S', 'controls', HSV_high[1], 255, callback)
cv2.createTrackbar('low V', 'controls', HSV_low[2], 255, callback)
cv2.createTrackbar('high V', 'controls', HSV_high[2], 255, callback)
cv2.createTrackbar('Kernel Size', 'controls', kernel_size, 16, callback)
cv2.createTrackbar('Circle Threshold', 'controls', int(circle_threshold * 100), 99, callback)

'''
# Find circle function
# Returns array of circles with accuracy (0 - 1), 1 meaning 100% a circle and 0 meaning 0% a circle
# Return Example:
# [[x, y, r, accuracy], [x, y, r, accuracy], [x, y, r, accuracy]...]
def find_circles(image):
    # Check for if we don't have video
    if image is None:
        return []

    # Array of circles to be returned
    result_circles = []

    # Kernels for GaussianBlur and morphologyEx
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    kernel2x = np.ones((kernel_size * 2, kernel_size * 2), np.uint8)

    # Blur image and convert to HSV
    blur = cv2.GaussianBlur(image, (blur_size, blur_size), 1)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a color mask and MORPH_OPEN to remove noise
    lower_hsv = np.array(HSV_low)
    upper_hsv = np.array(HSV_high)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    open_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2x)

    # Open_mask, but colored. Used for debug
    #colored_mask = cv2.bitwise_and(blur, blur, mask=open_mask)

    # Find edges of open_mask and MORPH_CLOSE to connect nearby edges and fill smaller holes
    canny = cv2.Canny(open_mask, 0, 255)
    closed_canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

    # Create a list of contours from the closed_canny
    contours, hierarchy = cv2.findContours(closed_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:  # Make sure we have contours to manipulate
        circle_contours = []
        for i in range(len(contours)):
            # Convex hull to make all contours convex
            convex_contour = cv2.convexHull(contours[i])

            # Get bounding box of contour
            x, y, w, h = cv2.boundingRect(convex_contour)

            # Calculate radius of contour based on bounding box
            r = (w + h) / 4

            is_circle = True
            accuracy = 1

            # Loop through contour's vertices
            for j in range(len(convex_contour)):
                # Calculate distance between the vertex and the center of the contour
                dist = math.sqrt(
                    ((x + w / 2) - convex_contour[j][0][0]) ** 2 + ((y + h / 2) - convex_contour[j][0][1]) ** 2)

                # Adjust accuracy value
                if accuracy > 1 - (abs(dist - r) / (r * circle_threshold)):
                    accuracy = 1 - (abs(dist - r) / (r * circle_threshold))

                # Check if distance is a similar length to the radius within the threshold
                if (dist * (1 - circle_threshold) > r) or (dist * (1 + circle_threshold) < r):
                    is_circle = False
                    break

            # If it passed, it gets added to an array
            if is_circle:
                result_circles.append([int((x + w / 2)), int((y + h / 2)), int(r), accuracy])
                circle_contours.append(convex_contour)

    # Show the colored mask, used for debug
    #cv2.imshow("colored_mask", colored_mask)

    # Return circles
    return result_circles


# Make sure capture is open and loaded
while True:
    # Read the frame of capture
    ret, frame = cap.read()

    # find circles
    circles = find_circles(frame)

    # Loop through circles and display them
    for circle in circles:
        cv2.circle(frame, (circle[0], circle[1]), circle[2], (255, 100, 75), int(circle[2] / 15))
        cv2.circle(frame, (circle[0], circle[1]), 1, (255, 100, 75), int(circle[2] / 15))
        cv2.putText(frame, str(round((circle[3]) * 100)) + "%", (circle[0] + 5, circle[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, circle[2] / 120, (255, 100, 75), int(circle[2] / 60), cv2.LINE_AA)

    # Display result frame
    cv2.imshow("result", frame)
    cv2.waitKey(1)
    gc.collect()

# Unload everything
cap.release()
cv2.destroyAllWindows()
