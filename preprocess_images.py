import cv2
import numpy as np


def process_image(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection with higher thresholds
    edges = cv2.Canny(gray_img, threshold1=200, threshold2=250, apertureSize=3)

    # Perform dilation to thicken the edges
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    return dilated_edges


def process_image1(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # line_img = cv2.Canny(gray_img, 115, 250)
    line_img = cv2.Canny(gray_img, threshold1=100, threshold2=150, apertureSize=3)
    return line_img


def process_image2(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection with higher thresholds
    edges = cv2.Canny(gray_img, threshold1=200, threshold2=250, apertureSize=3)

    # Perform dilation to thicken the edges
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Get image dimensions
    height, width = img.shape[:2]

    height = height + 80

    # Calculate box dimensions (e.g., 20% of image width and height)
    box_width = int(0.2 * width)
    box_height = int(0.2 * height)

    # Calculate box position just below the center
    x = int((width - box_width) / 2)
    y = int((height + box_height) / 2)

    # Draw a white rectangle just below the center of the image
    cv2.rectangle(dilated_edges, (x, y), (x + box_width, y + box_height), (0, 0, 0), -1)

    return dilated_edges


"""
# loop to show screen
import mss
import numpy as np

with mss.mss() as sct:
    while "Screen capturing":
        # last_time = time.time()
        monitor = {"top": 200, "left": 800, "width": 600, "height": 600}
        img = np.array(sct.grab(monitor))
        img = process_image(img)  # Process the captured image if needed

        # show image
        cv2.imshow("image", img)
        # Display the image with bounding rectangles for snakes and food pellets
        cv2.imshow("Snake and Food Detection", img)

        # resize image
        img = cv2.resize(img, (200, 200))

        # print(f"fps: {1 / (time.time() - last_time)}")

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
        # can later add in another function?? to do image show
        # print(f"fps: {1 / (time.time() - last_time)}")
"""
