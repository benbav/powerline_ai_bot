import cv2
import numpy as np
from preprocess_images import process_image
import mss
import time

# need to find snake heads next along with food and the snake bodies


# code for taking screenshot
def take_screenshot():
    count = 0

    # time.sleep(3)
    with mss.mss() as sct:
        # Capture a single screenshot
        monitor = {"top": 200, "left": 100, "width": 600, "height": 600}
        screenshot = sct.shot(output=f"garbo_pics/SCREENSHOT{count}.png")
        count += 1

    print(f"Screenshot saved as SCREENSHOT{count}.png")


""" while True:
    time.sleep(3)
    take_screenshot()
 """


def find_game_over_screen(img, threshold=0.8):
    """
    Find the game over screen in a full screen image.
    Parameters:
        img (numpy.ndarray): The NumPy array representing the full screen image.
        threshold (float): The similarity threshold for template matching. Default is 0.8.
    Returns:
        bool: True if the game over screen is found, False otherwise.
    """

    # Load the template image in grayscale
    # template = cv2.imread("pics/game_over_screen.png", cv2.IMREAD_COLOR)
    template = cv2.imread("pics/game_over2.png", cv2.IMREAD_COLOR)
    template = process_image(template)

    # Match the template in the full screen image

    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

    # show image
    """ cv2.imshow("template", result)
    cv2.waitKey(0) """

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # print(f"Max Val: {max_val}, Threshold: {threshold}")

    # If a match is found above the threshold, draw a purple rectangle and return True
    if max_val >= threshold:
        return True
    else:
        return False


"""
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,  # Minimum distance between detected circles
        param1=50,  # Upper threshold for edge detection
        param2=30,  # Threshold for circle center detection
        minRadius=10,  # 12  # Minimum radius of the circles
        maxRadius=20,  # 20  # Maximum radius of the circles
    )
"""


def find_food(processed_img):  # after putting in image through process_image
    # Apply Gaussian blur to reduce noise and improve circle detection
    # gray_blurred = cv2.GaussianBlur(processed_img, (3, 3), 1)
    gray_blurred = processed_img
    # cv2.imshow("blurred", gray_blurred)
    # cv2.waitKey(1000)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=5,  # Minimum distance between detected circles
        param1=80,  # Upper threshold for edge detection
        param2=20,  # Threshold for circle center detection
        minRadius=10,  # 12  # Minimum radius of the circles
        maxRadius=20,  # 20  # Maximum radius of the circles
    )

    if circles is not None:
        circle_coordinates = []
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, radius = circle
            # print(f"Radius: {radius}")
            circle_coordinates.append((x, y))
        print("found", len(circle_coordinates), "food")
        # print average radius

        return circle_coordinates
    else:
        return None


# tweak to get all the giant clumps of food
# but also make sure it works on original single food after tweaking
# on full snakes screen
# test usage:

# test_img = "pics/food_clump.png"
# # test_img = "pics/full_snakes_screen.png"
# circles_found = find_food(process_image(cv2.imread(test_img)))
# if circles_found:
#     print("Circles found:")
#     test = cv2.imread(test_img)
#     processed = process_image(cv2.imread(test_img))
#     for x, y in circles_found:
#         cv2.circle(test, (x, y), 10, (20, 255, 255), 3)
#     cv2.imshow("food found", test)
#     cv2.waitKey(0)
# else:
#     print("No circles found.")


# tweak params to get food clumps - look at processed image
