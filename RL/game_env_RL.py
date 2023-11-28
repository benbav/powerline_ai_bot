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
        self.current_screen = None  # Add a variable to store the current screen
        self.start_time = None  # Variable to store the start time
        self.eaten_food = 0

        # Define the position of the snake's head
        self.snake_head_x = self.width
        self.snake_head_y = self.height + 80
        self.direction = "up"

        def open_powerline_in_browser():
            # Define the AppleScript to open Safari and move/resize the window
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
            # Run the AppleScript using subprocess was at {0, 300, 760, 800}
            subprocess.run(["osascript", "-e", applescript], check=True)

        open_powerline_in_browser()

    def capture_screen(self):
        with mss.mss() as sct:
            monitor = {"top": 400, "left": 800, "width": self.width, "height": self.height}
            img = np.array(sct.grab(monitor))
            img = process_image(img)
            return img

    def reset(self):
        # time.sleep(1)  # give server a break
        print("RESETING GAME STATE")
        # Reset the environment to the initial state
        pyautogui.click(1200, 500)  # click on right screen
        time.sleep(1)
        pyautogui.write("NOT A BOT ")
        pyautogui.press("enter")
        time.sleep(1)
        self.current_screen = self.capture_screen()  # Capture the initial screen
        self.start_time = time.time()  # Set the start time when the episode starts
        self.food_pellets = []
        self.direction = "up"
        return self.current_screen  # Return the initial screen as the state

    def move_to_avoid_walls(self):
        # Dictionary mapping current direction to possible directions and key presses
        possible_moves = {
            "up": {"left": ("a", "left"), "right": ("d", "right")},
            "down": {"left": ("a", "left"), "right": ("d", "right")},
            "left": {"up": ("w", "up"), "down": ("s", "down")},
            "right": {"up": ("w", "up"), "down": ("s", "down")},
        }

        current_direction = self.direction
        new_direction = random.choice(list(possible_moves[current_direction].keys()))

        key_press, turn_text = possible_moves[current_direction][new_direction]

        pyautogui.press(key_press)  # Press the key to move in the chosen direction
        self.direction = new_direction
        print(f"Turning {turn_text}")

    def step(self, action):
        reward = 0
        if action == 0:  # 'W' key
            pyautogui.press("w")
            # press w and d
        elif action == 1:  # 'A' key
            pyautogui.press("a")
        elif action == 2:  # 'S' key
            pyautogui.press("s")
        elif action == 3:  # 'D' key
            pyautogui.press("d")
        # elif action == 4:  # Do Nothing (No keypress)
        #    pass

        new_screen = self.capture_screen()

        food_pellets = find_food(self.capture_screen())

        if self.food_pellets:
            print("FOUND FOOD PELLETS", self.food_pellets)
            margin = 60  # margin to check around center of food to confirm if eaten
            for x, y in self.food_pellets:
                if abs(self.snake_head_x - x) <= margin and abs(self.snake_head_y - y) <= margin:
                    # print("EATING FOOD")
                    # cv2.circle(
                    #     new_screen, (x, y), 10, (255, 255, 255), 3
                    # )  # can take this out later if it slows it down too much
                    self.eaten_food += 1
                    self.reward += 1000  # Add a positive reward for eating food
                    self.food_pellets.remove((x, y))  # do i need this?

        # Calculate the time elapsed since the episode started
        time_elapsed = time.time() - self.start_time
        # Calculate the reward based on the time elapsed (you can adjust the coefficients)
        reward += time_elapsed * 0.01  # Adjust the coefficient as needed

        done = False  # Modify this based on game termination conditions

        return new_screen, reward, done


# env = GameEnvironment(height=300, width=300)
# time.sleep(1)
# env.reset()

# frame_count = 0
# start_time = time.time()

# while True:
#     img = env.capture_screen()
#     time_elapsed = time.time() - start_time

#     # draw giant dot on snake head
#     # cv2.circle(img, (env.snake_head_x, env.snake_head_y), 10, (255, 255, 255), 3)

#     if find_food(img):
#         coords = find_food(img)
#         # print("found", len(coords), "food")
#         margin = 60  # Adjust this margin as needed
#         for x, y in coords:
#             if abs(env.snake_head_x - x) <= margin and abs(env.snake_head_y - y) <= margin:
#                 # print("EATING FOOD")
#                 env.eaten_food += 1

#             cv2.circle(img, (x, y), 10, (255, 255, 255), 3)
#             # cv2.imshow("Game Screen", img)
#         # print("TOTAL FOOD EATEN", env.eaten_food)

#     frame_count += 1
#     if time_elapsed >= 1.0:  # Update FPS every second
#         fps = frame_count / time_elapsed
#         print("FPS:", fps)
#         frame_count = 0
#         start_time = time.time()

#     cv2.imshow("Game Screen", img)
#     cv2.waitKey(1)


# it can detect food but is has trouble with big clumps of food together
# its having trouble deteting the food in teh actual trianing loop - maybe separate out the find_food function?
#
