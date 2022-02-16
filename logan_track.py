import numpy as np
import cv2
import math
import gc
#import cProfile

 # 0.0 gets all shades of black, gray, and white
red_grayness = 0.5
blue_grayness = 0.20

# 1.0 gets all similar hues of red or blue
red_variance = 0.1
blue_variance = 0.4

upper_brightness = 200
lower_darkness = 20

cv2.useOptimized()

#def mask_color(r, g, b, varience, grayness):

def find_circles(image, certainty_thresh, min_radius):
    if image is None:
        return {}

    output = {
        "circles": [],
        "contours": [],
        "mask": []
    }



    kernel = np.ones((6, 6), np.uint8)
    #close_kernel = np.ones((15, 15), np.uint8)

    image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)

    b, g, r = image[:, :, 0].astype(int), image[:, :, 1].astype(int), image[:, :, 2].astype(int)

    red_mask = (r > b) & (r > g) & (abs(g - b) < r * red_variance) & ((r - np.fmax(g, b)) > r * red_grayness) & ((r + g + b) / 3 < upper_brightness) & ((r + g + b) / 3 > lower_darkness)
    blue_mask = (b > r) & (b > g) & (abs(g - r) < b * blue_variance) & ((b - np.fmax(r, g)) > b * blue_grayness) & ((r + g + b) / 3 < upper_brightness) & ((r + g + b) / 3 > lower_darkness)

    red_masked = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    blue_masked = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    red_masked[red_mask] = 255
    blue_masked[blue_mask] = 255

    # red_masked = cv2.morphologyEx(red_masked, cv2.MORPH_CLOSE, kernel)
    # blue_masked = cv2.morphologyEx(blue_masked, cv2.MORPH_CLOSE, kernel)
    red_masked = cv2.morphologyEx(red_masked, cv2.MORPH_OPEN, kernel)
    blue_masked = cv2.morphologyEx(blue_masked, cv2.MORPH_OPEN, kernel)


    output["mask"] = red_masked | blue_masked
    # red_masked = image & red_masked
    # blue_masked = image & blue_masked
    #
    # colored_mask = red_masked | blue_masked

    # red_closed_mask = cv2.morphologyEx(red_masked, cv2.MORPH_CLOSE, close_kernel)
    # blue_closed_mask = cv2.morphologyEx(blue_masked, cv2.MORPH_CLOSE, close_kernel)

    #red_canny = cv2.Canny(red_masked, 0, 255)
    #blue_canny = cv2.Canny(blue_masked, 0, 255)

    red_contours, red_hierarchy = cv2.findContours(
        red_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    blue_contours, blue_hierarchy = cv2.findContours(
        blue_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    contours = red_contours + blue_contours

    output["contours"] = contours

    if len(contours) != 0:
        for contour in contours:
            convex_contour = cv2.convexHull(contour)
            #contour = contours[i]

            area = cv2.contourArea(contour)
            convex_area = cv2.contourArea(convex_contour)

            (x, y), r = cv2.minEnclosingCircle(convex_contour)
            x = int(x)
            y = int(y)
            r = int(r)

            if r < min_radius:
                continue

            perimeter_certainty = cv2.arcLength(
                convex_contour, True) / (math.pi * 2 * r)
            perimeter_certainty = 1 / \
                perimeter_certainty if perimeter_certainty > 1 else perimeter_certainty
            perimeter_certainty = perimeter_certainty ** 2

            area_certainty = convex_area / (math.pi * (r ** 2))
            area_certainty = 1 / area_certainty if area_certainty > 1 else area_certainty

            solidity_certainty = 1 - \
                ((convex_area - area) / (math.pi * (r ** 2)))
            solidity_certainty = solidity_certainty ** 2

            certainty = (area_certainty + perimeter_certainty + solidity_certainty) / 3

            if certainty < certainty_thresh:
                continue

            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            color = cv2.mean(image, mask=mask)

            if (color[0] > color[2]):
                team = "blue"
            else:
                team = "red"

            certainty = (area_certainty +
                         perimeter_certainty + solidity_certainty) / 3

            if certainty > certainty_thresh:
                output["circles"].append({
                    "x": x,
                    "y": y,
                    "r": r,
                    "certainty": certainty,
                    "area_certainty": area_certainty,
                    "perimeter_certainty": perimeter_certainty,
                    "solidity_certainty": solidity_certainty,
                    "convex_contour": convex_contour,
                    "contour": contour,
                    "color": color,
                    "team": team
                })

    return output


cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('redBall.mp4')

while True:
    ret, frame = cap.read()

    #frame = cv2.resize(frame, (960, 540))

    data = find_circles(frame, 0.8, 30)
    circles = data["circles"]
    #frame = data["mask"]

    #cv2.drawContours(frame, data["contours"], -1, (100, 255, 75), 3)

    for i in range(len(circles)):
        circle_color = circles[i]["color"]
        #print(circles[i]["team"] + ": " + str(circle_color))
        # cv2.circle(frame, (circles[i]["x"], circles[i]["y"]),
        #            circles[i]["r"], circle_color, -1)
        if circles[i]["team"] == "blue":
            cv2.circle(frame, (circles[i]["x"], circles[i]["y"]),
                       circles[i]["r"], (50, 50, 150), 10)
            cv2.putText(frame, str(round((circles[i]["certainty"]) * 100)) + "%", (circles[i]["x"] + 5, circles[i]["y"] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, circles[i]["r"] / 120, (50, 50, 150), int(circles[i]["r"] / 60), cv2.LINE_AA)
        else:
            cv2.circle(frame, (circles[i]["x"], circles[i]["y"]),
                       circles[i]["r"], (100, 75, 50), 10)
            cv2.putText(frame, str(round((circles[i]["certainty"]) * 100)) + "%", (circles[i]["x"] + 5, circles[i]["y"] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, circles[i]["r"] / 120, (100, 75, 50), int(circles[i]["r"] / 60), cv2.LINE_AA)

    #print(len(data["circles"]))
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow("output", frame)
    cv2.resizeWindow('output', 1920, 1080)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

    gc.collect()
    #cv2.destroyAllWindows()
