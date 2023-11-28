import tensorflow as tf
import numpy as np
from RL.find_parts import find_game_over_screen
from preprocess_images import process_image
import subprocess
import time
import mss
import cv2
from game_env import GameEnvironment

# TODO : add in object detection for food and change reward function to match - currently working on..
# also try without frame skipping


# Define the neural network architecture (e.g., a simple feedforward network)
def create_model(input_shape, num_actions):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_actions, activation="linear"),
        ]
    )
    return model


# Define the RL algorithm (e.g., Deep Q-Network - DQN)
class DQNAgent:
    def __init__(self, input_shape, num_actions):
        self.model = create_model(input_shape, num_actions)
        self.target_model = create_model(input_shape, num_actions)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.loss_fn = tf.keras.losses.Huber()
        self.num_actions = num_actions

    def set_model_weights(self, loaded_model):
        self.model.set_weights(loaded_model.get_weights())
        self.target_model.set_weights(loaded_model.get_weights())

    def select_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.choice(self.num_actions)
        else:
            q_values = self.model.predict(state, verbose=0)  # verbose = 0 disables printing
            return np.argmax(q_values)

    def train(self, replay_buffer, batch_size, gamma):
        if len(replay_buffer) < batch_size:
            return

        # Extract transitions from the replay buffer into separate lists
        states, actions, rewards, next_states, dones = zip(*replay_buffer)

        # Convert these lists into NumPy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Calculate target Q-values
        target_values = rewards + gamma * np.max(self.target_model.predict(next_states, verbose=0), axis=1)
        target_values[dones] = rewards[dones]

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            selected_action_values = tf.reduce_sum(q_values * tf.one_hot(actions, self.num_actions), axis=1)
            loss = self.loss_fn(target_values, selected_action_values)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


# Initialize the RL environment and agents
env = GameEnvironment(height=300, width=300)

# Hyperparameters
input_shape = (env.height * 2, env.height * 2)  # Adjust to match your game state size
# print("input shape", input_shape)
num_actions = 4  # Number of possible actions (e.g., W, A, S, D)
epsilon_initial = 0.5
epsilon_final = 0.1
epsilon_decay_steps = 5000
replay_buffer_size = 2000  # 100000
batch_size = 64
gamma = 0.99  # Discount factor

agent = DQNAgent(input_shape, num_actions)

agent.set_model_weights(tf.keras.models.load_model("dqn_model.h5"))

replay_buffer = []

# Training loops
total_steps = 0
epsilon = epsilon_initial

# Hyperparameters for the training loop

max_episodes = 1000  # Adjust the number of episodes as needed
target_update_frequency = 10000  # Frequency to update the target network

# Define the frame skip factor (e.g., skip every 3 frames)
# frame_skip = 1

cv2.namedWindow("Game Screen")

for episode in range(max_episodes):
    state = env.reset()
    episode_reward = 0
    frame_count = 0  # Initialize frame counter

    while True:
        # Measure the time before selecting an action
        start_time = time.time()

        if find_game_over_screen(img=env.capture_screen(), threshold=0.29):
            print("updating model weights")
            agent.target_model.set_weights(agent.model.get_weights())  # update weights every time it loses
            print(f"Game over. Episode {episode}: Total Reward = {episode_reward}")
            print("total food eaten:", len(env.food_pellets))
            break

        # Take the same action for frame_skip frames
        # if frame_count % frame_skip == 0:
        #     action = agent.select_action(np.expand_dims(state, axis=0), epsilon)

        # with no frame skipping
        action = agent.select_action(np.expand_dims(state, axis=0), epsilon)

        new_state, reward, done = env.step(action)
        # print("reward", reward)
        replay_buffer.append((state, action, reward, new_state, done))

        # Train the agent every frame_skip frames
        # if frame_count % frame_skip == 0 and len(replay_buffer) > batch_size:
        #     agent.train(replay_buffer, batch_size, gamma)

        agent.train(replay_buffer, batch_size, gamma)

        state = new_state
        episode_reward += reward
        total_steps += 1
        frame_count += 1
        epsilon = max(epsilon_final, epsilon_initial - total_steps / epsilon_decay_steps)

        # Measure the time after selecting an action and calculate the time taken
        # end_time = time.time()
        # time_taken = end_time - start_time

        # current_screen = env.capture_screen()
        # cv2.imshow("Game Screen", new_state)
        # cv2.waitKey(1)  # Keep the windowa open

        # Print the time taken for model output
        # print(f"Time taken for model output: {time_taken} seconds")

    # if total_steps % target_update_frequency == 0:
    #     agent.target_model.set_weights(agent.model.get_weights())


# Destroy the window when the loop ends
cv2.destroyAllWindows()

# Compile the agent's model before saving it
agent.model.compile(
    optimizer="adam",  # Replace with your optimizer
    loss="mean_squared_error",  # Replace with your loss function
    metrics=["accuracy"],
)  # Replace with your desired metrics

# Save the trained model after the batch
print("Saving model")
agent.model.save("dqn_model.h5")

# maybe keep screen brightness at a constant? (4 bars)
