import numpy as np
import mss
import time
import pyautogui
import subprocess
from preprocess_images import process_image
import cv2
from RL.find_parts import find_food
import random


class GameEnvironment:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Define the position of the snake's head
        self.snake_head_x = self.width
        # self.snake_head_y = self.height + 80
        self.snake_head_y = self.height - 20
        self.direction = "up"

        def open_powerline_in_browser():
            applescript = """
            tell application "Safari"
            activate
            if (count of windows) is 0 then
                make new document
            end if
            set URL of document 1 to "https://powerline.io/"
            set bounds of front window to {600, 300, 1300, 800}  -- Change the coordinates and size as needed
            end tell
            """
            # Run the AppleScript using subprocess to open Safari at {0, 300, 760, 800}
            subprocess.run(["osascript", "-e", applescript], check=True)

        open_powerline_in_browser()

    def capture_screen(self):
        with mss.mss() as sct:
            monitor = {"top": 550, "left": 900, "width": self.width, "height": self.height}  # for smaller screen

            # monitor = {"top": 400, "left": 800, "width": self.width, "height": self.height} used 300 for height and width
            # self.snake_head_x = self.width
            # self.snake_head_y = self.height + 80
            # this is where id grab less screen
            img = np.array(sct.grab(monitor))
            img = process_image(img)
            return img

    def reset(self):
        print("RESETING GAME STATE")
        pyautogui.click(1200, 500)  # click on right screen
        time.sleep(1)
        pyautogui.write("NOT A BOT ")
        pyautogui.press("enter")
        time.sleep(1)
        self.current_screen = self.capture_screen()  # Capture the initial screen
        self.start_time = time.time()  # Set the start time when the episode starts
        self.direction = "up"
        return self.current_screen  # Return the initial screen as the state
