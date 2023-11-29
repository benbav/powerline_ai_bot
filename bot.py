import cv2
import time
import random
import pyautogui
from game_env import GameEnvironment
import numpy as np

env = GameEnvironment(height=100, width=100)
env.reset()
frame_count = 0
start_time = time.time()
move_timer = time.time()

margin = 30

# its so close to being cold... just have to bug test a few weird features
# I don't want it to hug the wall forever - add a timer when to turn


def scan_left_right(img, y, x, margin):
    # Calculate left and right boundaries within the margin
    x_left = max(x - margin, 0)
    x_right = min(x + margin, img.shape[1])

    # Extract the region of interest (ROI)
    roi = img[y - margin : y + margin, x_left:x_right]

    # Count black pixels in the left and right halves
    black_pixels_left = np.sum(roi[:, : roi.shape[1] // 2] == 0)
    black_pixels_right = np.sum(roi[:, roi.shape[1] // 2 :] == 0)

    # Determine the direction with the most black pixels
    if black_pixels_left > black_pixels_right:
        # print("more black pixels left")
        return "left"
    else:
        # print("more black pixels right")
        return "right"


def scan_up_down(img, y, x, margin):
    # Calculate top and bottom boundaries within the margin
    y_top = max(y - margin, 0)
    y_bottom = min(y + margin, img.shape[0])

    # Extract the region of interest (ROI)
    roi = img[y_top:y_bottom, x - margin : x + margin]

    # Count black pixels in the top and bottom halves
    black_pixels_top = np.sum(roi[: roi.shape[0] // 2, :] == 0)
    black_pixels_bottom = np.sum(roi[roi.shape[0] // 2 :, :] == 0)

    # Determine the direction with the most black pixels
    if black_pixels_top > black_pixels_bottom:
        return "up"
    else:
        return "down"


while True:
    img = env.capture_screen()
    time_elapsed = time.time() - start_time

    move_mapping = {
        "up": ("up", env.snake_head_y - margin, env.snake_head_x),  # check if wall above
        "down": ("down", env.snake_head_y + margin, env.snake_head_x),  # check if wall below
        "left": ("left", env.snake_head_y, env.snake_head_x - margin),  # check if wall left
        "right": ("right", env.snake_head_y, env.snake_head_x + margin),  # check if wall right
    }

    direction_map = {
        "up": "w",
        "down": "s",
        "left": "a",
        "right": "d",
    }

    logic_mapping = {
        "up": scan_left_right(img, env.snake_head_y, env.snake_head_x, margin),
        "down": scan_left_right(img, env.snake_head_y, env.snake_head_x, margin),
        "left": scan_up_down(img, env.snake_head_y, env.snake_head_x, margin),
        "right": scan_up_down(img, env.snake_head_y, env.snake_head_x, margin),
    }

    # if there is a wall in front then turn
    direction, y, x = move_mapping[env.direction]

    if img[y, x] == 255 or (time.time() - move_timer > 6):
        print("found", env.direction, "wall")

        safe_direction = logic_mapping[env.direction]

        print("turning", direction, "pressing", direction_map[safe_direction])
        pyautogui.press(direction_map[safe_direction])

        env.direction = safe_direction
        move_timer = 0

        cv2.circle(img, (x, y), 5, (255, 255, 255), 3)

    # cv2.circle(img, (env.snake_head_x, env.snake_head_y), 5, (255, 255, 255), 3)
    # cv2.imshow("Game Screen", img)
    # cv2.waitKey(1)

    # frame_count += 1
    # fps = frame_count / time_elapsed
    # print("FPS:", fps)
    # frame_count = 0
    # start_time = time.time()

# see if we can make the image smaller to make it faster
