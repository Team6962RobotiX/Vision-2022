def find_balls(image):
    """ Returns a list of balls in an image like this:
    [
        {
            "x": 16,
            "y": 18,
            "r": 6,
            "certainty": 0.93,
            "team": "blue"
        },
        {
            "x": 12,
            "y": 24,
            "r": 18,
            "certainty": 0.98,
            "team": "red"
        }, ...
    ]
    (requires cv2, numpy as np, and math)
    """

    # Minimum certainty to be classified as a ball, 1 is 100% certain
    min_certainty = 0.8

    # Minimum radius to be classified as a ball, in pixels
    min_radius = 4

    # Low and high hsv ranges for red balls. Ranging from (0 - 180), (0 - 255), and (0 - 255)
    red_low = (175, 240, 32)
    red_high = (180, 255, 255)

    # Same thing as above, but for the other side of the hue spectrum
    red_low2 = (0, red_low[1], red_low[2])
    red_high2 = (5, red_high[1], red_high[2])

    # Same thing as above, but for blue balls
    blue_low = (80, 220, 10)
    blue_high = (110, 255, 160)



    if image is None:
        return []

    output = []

    kernel = np.ones((3, 3), np.uint8)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    blur = cv2.GaussianBlur(hsv, (5, 5), cv2.BORDER_DEFAULT)

    red_masked = cv2.inRange(blur, red_low, red_high) | cv2.inRange(blur, red_low2, red_high2)
    blue_masked = cv2.inRange(blur, blue_low, blue_high)

    red_masked = cv2.morphologyEx(red_masked, cv2.MORPH_OPEN, kernel)
    blue_masked = cv2.morphologyEx(blue_masked, cv2.MORPH_OPEN, kernel)

    red_contours, red_hierarchy = cv2.findContours(
        red_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    blue_contours, blue_hierarchy = cv2.findContours(
        blue_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    contours = red_contours + blue_contours

    if len(contours) != 0:
        for contour in contours:

            convex_contour = cv2.convexHull(contour)
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

            if certainty < min_certainty:
                continue

            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            color = cv2.mean(image, mask = mask)

            if (color[0] > color[2]):
                team = "blue"
            else:
                team = "red"

            if certainty > min_certainty:
                output.append({
                    "x": x,
                    "y": y,
                    "r": r,
                    "certainty": certainty,
                    "team": team
                })

    return output
