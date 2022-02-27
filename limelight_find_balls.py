import cv2
import numpy as np
import math

# Put a bunch of easy-to-tweak values up here in global

# Color to find (switch for different pipelines)
COLOR_TO_FIND = "red"

# Minimum certainty to be classified as a ball, 1 is 100% certain
MINIMUM_CERTAINTY_PERCENTAGE = 0.8

# Minimum radius to be classified as a ball, in pixels
MINIMUM_PIXELS_RADIUS = 4

# Low and high HSV ranges for blue balls. Ranging from (0 - 180), (0 - 255), and (0 - 255)
BLUE_HSV_LOW = (80, 220, 10)
BLUE_HSV_HIGH = (110, 255, 160)

# Low and high HSV ranges for red balls. Ranging from (0 - 180), (0 - 255), and (0 - 255)
RED_HSV_LOW_1 = (175, 240, 32)
RED_HSV_HIGH_1 = (180, 255, 255)

# Same thing as above, but for the other side of the hue spectrum for red
RED_HSV_LOW_2 = (0, RED_HSV_LOW_1[1], RED_HSV_LOW_1[2])
RED_HSV_HIGH_2 = (5, RED_HSV_HIGH_1[1], RED_HSV_HIGH_1[2])

# colors we know about
COLOR_RED = "red"
COLOR_BLUE = "blue"


# runPipeline() is called every frame by Limelight's backend.
def runPipeline(image, llrobot):
    global COLOR_TO_FIND

    # defaults to send back to the robot if nothing is found
    largest_contour = np.array([[]])
    llpython = [0, 0, 0, 0, 0, 0, 0, 0]

    if image is None:
        return largest_contour, image, llpython

    # find ball contours
    contours = find_ball_contours(image, COLOR_TO_FIND)

    # return if none found
    if len(contours) == 0:
        return largest_contour, image, llpython

    # draw the contours on our image
    cv2.drawContours(image, contours, -1, 255, 2)

    # find and record the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # get the un-rotated bounding box that surrounds the contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # record some custom data to send back to the robot
    llpython = [1, x, y, w, h, 9, 8, 7]

    # return the largest contour for the Limelight crosshair, the modified image, and custom robot data
    return largest_contour, image, llpython


# finds likely ball contours
def find_ball_contours(image, color):
    global MINIMUM_CERTAINTY_PERCENTAGE
    global MINIMUM_PIXELS_RADIUS
    global COLOR_RED
    global COLOR_BLUE

    matching_contours = []

    kernel = np.ones((3, 3), np.uint8)

    # convert the Image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # apply a Gaussian blur
    blur = cv2.GaussianBlur(hsv, (5, 5), cv2.BORDER_DEFAULT)

    # find color masks
    if COLOR_BLUE == color:
        masks = find_masks_blue(blur)
    elif COLOR_RED == color:
        masks = find_masks_red(blur)

    masked = cv2.morphologyEx(masks, cv2.MORPH_OPEN, kernel)

    # find the contours matching the mask
    contours, hierarchy = cv2.findContours(
        masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2:]

    if len(contours) == 0:
        return matching_contours

    # loop over the contours, looking for circles
    for contour in contours:
        convex_contour = cv2.convexHull(contour)

        (x, y), radius = cv2.minEnclosingCircle(convex_contour)
        x = int(x)
        y = int(y)
        radius = int(radius)

        # check whether the # of pixels is too small
        if radius < MINIMUM_PIXELS_RADIUS:
            continue

        # checks whether this contour is likely to be a target
        certainty = get_certainty(
            convex_contour, cv2.contourArea(convex_contour), cv2.contourArea(contour), radius
        )

        if certainty > MINIMUM_CERTAINTY_PERCENTAGE:
            # write the certainty % on the image for debugging
            cv2.putText(
                image,
                str(round(certainty, 2)),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                .5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

            matching_contours.append(contour)

    return matching_contours


# finds red masks
def find_masks_red(blur):
    global RED_HSV_LOW_1
    global RED_HSV_HIGH_1
    global RED_HSV_LOW_2
    global RED_HSV_HIGH_2

    return cv2.inRange(blur, RED_HSV_LOW_1, RED_HSV_HIGH_1) | cv2.inRange(blur, RED_HSV_LOW_2, RED_HSV_HIGH_2)


# finds blue masks
def find_masks_blue(blur):
    global BLUE_HSV_LOW
    global BLUE_HSV_HIGH

    return cv2.inRange(blur, BLUE_HSV_LOW, BLUE_HSV_HIGH)


# gets a percentage indicator for how likely this is a target
def get_certainty(convex_contour, convex_area, area, radius):
    return (get_certainty_area(convex_area, radius)
            + get_certainty_perimeter(convex_contour, radius)
            + get_certainty_solidity(convex_area, area, radius)) / 3


# gets a percentage indicator for how likely this perimeter matches desired
def get_certainty_perimeter(convex_contour, radius):
    certainty = cv2.arcLength(convex_contour, True) / (math.pi * 2 * radius)
    certainty = 1 / certainty if certainty > 1 else certainty

    return certainty ** 2


# gets a percentage indicator for how likely this area matches desired
def get_certainty_area(convex_area, radius):
    certainty = convex_area / (math.pi * (radius ** 2))
    certainty = 1 / certainty if certainty > 1 else certainty

    return certainty


# gets a percentage indicator for how likely the solidity of this contour matches desired
def get_certainty_solidity(convex_area, area, radius):
    certainty = 1 - ((convex_area - area) / (math.pi * (radius ** 2)))

    return certainty ** 2
