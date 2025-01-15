import cv2
import time
import random
import pyautogui
from game_env import GameEnvironment
import numpy as np
from multiprocessing import Pool
from check_range import scan_left_right, scan_up_down

env = GameEnvironment(height=100, width=100)
env.reset()
frame_count = 0
start_time = time.time()
# 
front_margin = 5
margin = 30
follow_margin = 20

# Global variable to store the result of wall detection
found_wall = False


def scan_wall_parallel(args):
    img, y, x, margin = args
    results = [
        scan_left_right(img, y, x, margin),
        scan_left_right(img, y, x, margin),
        scan_up_down(img, y, x, margin),
        scan_up_down(img, y, x, margin),
    ]
    return results


while True:
    img = env.capture_screen()
    time_elapsed = time.time() - start_time

    move_mapping = {
        "up": ("up", env.snake_head_y - margin, env.snake_head_x),  # check if wall above
        "down": ("down", env.snake_head_y + margin, env.snake_head_x),  # check if wall below
        "left": ("left", env.snake_head_y, env.snake_head_x - margin),  # check if wall left
        "right": ("right", env.snake_head_y, env.snake_head_x + margin),  # check if wall right
        # check if immediately in front of head
        "up": ("up", env.snake_head_y - front_margin, env.snake_head_x),  # check if wall above
        "down": ("up", env.snake_head_y + front_margin, env.snake_head_x),  # check if wall below
        "left": ("up", env.snake_head_y, env.snake_head_x - front_margin),  # check if wall left
        "right": ("up", env.snake_head_y, env.snake_head_x + front_margin),  # check if wall right
    }

    direction_map = {
        "up": "w",
        "down": "s",
        "left": "a",
        "right": "d",
    }

    # this checks the pixel directly in front of direction
    direction, y, x = move_mapping[env.direction]

    # print(direction, y, x)
    print(img[y, x])

    # implement multiprocessing with this
    # if there is white
    if img[y, x] == 255 and not found_wall:
        results = scan_wall_parallel(img, env.snake_head_y, env.snake_head_x, margin)

        logic_mapping = {
            "up": results[2],
            "down": results[3],
            "left": results[0],
            "right": results[1],
        }

        print("found", env.direction, "wall")

        safe_direction = logic_mapping[env.direction]

        print("turning", safe_direction, "pressing", direction_map[safe_direction])
        pyautogui.press(direction_map[safe_direction])

        env.direction = safe_direction

        move_timer = time.time()  # Reset the timer

        # cv2.circle(img, (x, y), 5, (255, 255, 255), 3)

    # Reset found_wall when the snake is no longer in front of a wall
    if img[y, x] != 255:
        found_wall = False

    # cv2.circle(img, (env.snakse_head_x, env.snake_head_y), 5, (255, 255, 255), 3)
    # this is test up to find snakes approaching from back right
    # cv2.circle(img, (env.snake_head_x + follow_margin, env.snake_head_y - 20), 5, (255, 255, 255), 3)
    cv2.imshow("Game Screen", img)
    cv2.waitKey(1)
