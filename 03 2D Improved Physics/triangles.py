from math import dist
from typing import final
import gym
from gym import spaces
from gym.spaces import MultiDiscrete, Box
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

import keyboard

class OctopusEnv(gym.Env):
    """
    A simple 2D environment simulating an octopus as a triangle that can move and rotate.
    
    Translational Dynamics (Decoupled):
      x_dot = v_x
      y_dot = v_y
      [v_x_dot, v_y_dot] = (u_f * [cos(theta), sin(theta)] - c * [v_x, v_y]) / m
      
    Rotational Dynamics:
      theta_dot = omega
      omega_dot = (u_r - c_theta * omega) / I
      
    The state vector is [x, y, theta, v_x, v_y, omega, distance to food].
    """
    
    def __init__(self, epsilon=0.0):
        super(OctopusEnv, self).__init__()
        # Dynamics parameters
        self.mass = 10.0                   # Mass of the object
        self.inertia = 1.0                # Moment of inertia
        self.drag_coefficient = 50       # Linear drag coefficient for translation (c)
        self.rotational_damping = 20     # Rotational damping coefficient (c_theta)
        self.dt = 0.05                    # Time step for integration
        
        # Action space: [u_f, u_r]
        #self.action_space = Box(low=-10, high=10.0, shape=(2,), dtype=np.float32)
        self.nA = 3
        self.action_space = MultiDiscrete([self.nA, self.nA])
        self.AtoPush = {0: -100, 1: 0, 2: 100}
        self.AtoTurn = {0: -100, 1: 0, 2: 100}
        
        # self.action_space = MultiDiscrete([self.nA, self.nA])
        # self.AtoPush = {0: -100, 1: 100}
        # self.AtoTurn = {0: -100, 1: 100}

        # Observation space:
        self.nHist = 1 # Number of historical observations timesteps to include in a single observation
        #self.nObs = 8 # Number of observation features
        self.xlim = (-10, 10)
        self.ylim = (-10, 10)
        # x, y, theta, v_x, v_y, omega, prev_dist_to_food, prev_u_f, prev_u_r
        self.obs_max = np.array([self.xlim[1], self.ylim[1], 2*np.pi, 10, 10, 10, np.sqrt(2*20**2), self.nA-1, self.nA-1, 1])
        self.nObs = len(self.obs_max)
        # Expand obs_max into a 2D array of shape (nHist, nObs)
        self.obs_max = np.tile(self.obs_max, (self.nHist, 1))

        self.obs_min = np.array([self.xlim[0], self.ylim[0], 0, -10, -10, -10, 0, 0, 0, 0])
        # Expand obs_min into a 2D array of shape (nHist, nObs)
        self.obs_min = np.tile(self.obs_min, (self.nHist, 1))

        self.hidden_mask = np.array([0]*2 + [1]*8)
        self.hidden_mask = np.tile(self.hidden_mask, (self.nHist, 1))
        
        self.observation_space = Box(low=self.obs_min, high=self.obs_max,
                                     shape=(self.nHist, self.nObs), dtype=np.float32)
        
        # Print the observation and action spaces details
        print("Observation space:", self.observation_space)
        print("Action space:", self.action_space)


        self.state = None
        self.food_position = None
        self.min_food_dist = 0.7
        self.max_food_dist = 10
        self.food_dist = self.min_food_dist
        self.timestep = 0
        self.time_limit = 100
        self.run_avg_duration = self.time_limit
        self.eps = epsilon
        
        self.reset()
        
        self.border = 5.0
        self.initialize_rendering()

    def normalize_state(self, state):
        return (state - self.obs_min) / (self.obs_max - self.obs_min)
    
    def check_state_bounds(self):
        # Check if values from self.state are in between self.obs_min and self.obs_max
        state_values_in_range = np.all(self.obs_min <= self.state) and np.all(self.state <= self.obs_max)
        if not state_values_in_range:
            print(f"State out of bounds: {self.state}")
            print("Specifically:")
            # loop over all values and check if they are in range
            for i, (min_val, max_val) in enumerate(zip(self.obs_min, self.obs_max)):
                if not min_val <= self.state[i] <= max_val:
                    print(f"Value {self.state[i]} at index {i} is out of bounds [{min_val}, {max_val}]")
            raise ValueError("State out of bounds")

    def step(self, action):
        # Unpack current state
        #x, y, theta, v_x, v_y, omega, prev_dist_to_food, prev_u_f, prev_u_r = self.state
        x, y, theta, v_x, v_y, omega, prev_dist_to_food, _, _, _ = self.state
        action_turn, action_push = action

        # Epsilon greedy exploration
        if np.random.rand() < self.eps:
            action_turn = np.random.choice([0, 1, 2])
        if np.random.rand() < self.eps:
            action_push = np.random.choice([0, 1, 2])

        u_r = self.AtoTurn[action_turn]
        u_f = self.AtoPush[action_push]

        # Translational dynamics (decoupled)
        # Thrust is applied along the current heading (theta)

        thrust = np.array([np.cos(theta), np.sin(theta)]) * u_f
        #thrust = np.array([np.cos(theta_next), np.sin(theta_next)]) * u_f

        # Compute acceleration: a = (thrust - drag) / mass
        accel = (thrust - self.drag_coefficient * np.array([v_x, v_y])) / self.mass

        # Euler integration for translation
        x_next = x + v_x * self.dt
        y_next = y + v_y * self.dt
        v_next = np.array([v_x, v_y]) + accel * self.dt
        # Set limit to velocity
        v_max = 10.0
        v_next = np.clip(v_next, -v_max, v_max)
        
        # Rotational dynamics
        theta_next = theta + omega * self.dt
        theta_next = theta_next % (2 * np.pi)  # Wrap angle to [0, 2*pi)
        omega_next = omega + ((u_r - self.rotational_damping * omega) / self.inertia) * self.dt
        # Set limit to angular velocity
        ang_v_max = 10.0
        omega_next = np.clip(omega_next, -ang_v_max, ang_v_max)

        dist_to_food = np.linalg.norm(np.array([x_next, y_next]) - self.food_position)

        #self.state = np.array([x_next, y_next, theta_next, v_next[0], v_next[1], omega_next, dist_to_food, u_f, u_r], dtype=np.float32)
        counter = self.timestep / self.time_limit
        self.state = np.array([x_next, y_next, theta_next, v_next[0], v_next[1], omega_next, dist_to_food, action_turn, action_push, counter], dtype=np.float32)
        self.timestep += 1
        self.state_hist[self.timestep + self.nHist - 1] = self.normalize_state(self.state)

        reward = -1

        # initial_distance = np.linalg.norm(self.food_position) # Starts at (0,0)
        # final_distance = np.linalg.norm(np.array([x, y]) - self.food_position)
        # reward += 1/self.time_limit * (initial_distance - final_distance)/initial_distance

        #reward = - 0.1*dist_to_food - 0.1/(self.time_limit - self.timestep + 1)

        # dist_diff = prev_dist_to_food - dist_to_food
        # reward +=  1 * dist_diff #if dist_diff < 0 else 0.0

        # Idle penalty
        #is_idle = np.linalg.norm(np.array([x, y]) - np.array([x_next, y_next])) < 1e-3
        #reward -= 1.0 if is_idle else 0.0

        done = False
        
        # b = self.border
        # if x_next < -b or x_next > b or y_next < -b or y_next > b:
        #     done = True
        #     initial_distance = np.linalg.norm(self.food_position) # Starts at (0,0)
        #     final_distance = np.linalg.norm(np.array([x, y]) - self.food_position)
        #     reward -= 1.0

        if dist_to_food < 0.5:
            done = True
            reward = 0
        elif self.timestep >= self.time_limit:
            done = True
            #reward -= dist_to_food
            # initial_distance = np.linalg.norm(self.food_position) # Starts at (0,0)
            # final_distance = np.linalg.norm(np.array([x, y]) - self.food_position)
            # reward += (initial_distance - final_distance)/initial_distance
            #reward -= 2.0

        info = {}
        if done:
            # Updating running average of episode duration
            self.run_avg_duration = 0.9 * self.run_avg_duration + 0.1 * self.timestep
            info = {"run_avg_duration": self.run_avg_duration}

        # if done:
        #     # Append the complete self.state_hist at tge end of a csv file with historical data
        #     with open("historical_data.csv", "a") as f:
        #         if self.timestep == self.time_limit:
        #             np.savetxt(f, self.state_hist[self.nHist:], delimiter=",")
        #         else:
        #             np.savetxt(f, self.state_hist[self.nHist:self.timestep + self.nHist], delimiter=",")

        s = self.state_hist[self.timestep:self.timestep + self.nHist, :] # Skip x, y colums
        
        # Apply hidden mask to agent observation
        s = s * self.hidden_mask
        return s, reward, done, info

    def reset(self):
        # Initialize the state:
        #self.food_position = np.random.uniform(-5.0, 5.0, size=(2,))
        # options = np.array([(1, 1)])#, (3,-3)])#, (-3,3), (-3,-3)])
        # self.food_position = options[np.random.choice(len(options))]
        run_avg_threshold = 50
        if (self.food_dist < self.max_food_dist) and (self.run_avg_duration < run_avg_threshold):
            # Curriculum learning: Start with food closer to the octopus and increase distance gradually
            # as it solves the episodes faster
            self.food_dist = min(1.1 * self.food_dist, self.max_food_dist)
            print("> Increasing food distance to:", np.round(self.food_dist, 2))
            self.run_avg_duration = self.time_limit
        rand_dist = np.random.uniform(self.min_food_dist, self.food_dist)
        rand_angle = np.random.uniform(0, 2*np.pi)
        self.food_position = np.array([rand_dist*np.cos(rand_angle),
                                       rand_dist*np.sin(rand_angle)])

        # x, y, theta, v_x, v_y, omega, prev_dist_to_food, prev_u_f, prev_u_r
        state_dim = 10
        self.state = np.zeros((state_dim), dtype=np.float32)
        self.state[6] = np.linalg.norm(self.food_position)
        self.state[-2:] = -1, -1 # Initial previous actions

        # -nHist: zeros | 
        # .             |
        # .             |> nHist: Observed backward on first step
        # .             |
        # -1 zeros      |
        # 0 Initial state
        # 1 First state after taking 1st action
        self.state_hist = np.zeros((self.time_limit + self.nHist, state_dim), dtype=np.float32)
        self.state_hist[self.nHist - 1] = self.normalize_state(self.state) 

        self.timestep = 0
        s = self.state_hist[self.timestep:self.timestep + self.nHist, :] # Skip x, y colums

        # Apply hidden mask to agent observation
        s = s * self.hidden_mask

        return s
    
    def update_exploration(self, new_exploration_rate):
        self.eps = new_exploration_rate

    def initialize_rendering(self):
        """
        Set up persistent Matplotlib figure and axis for rendering.
        """
        plt.ion()  # Interactive mode on
        self.fig, self.ax = plt.subplots()
        # Adjust these limits based on expected simulation domain.
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title("Octopus Simulation")

        # Border of the simulation area.
        b = self.border
        self.ax.plot([-b, b, b, -b, -b], [-b, -b, b, b, -b], 'k-')
        
        # Initialize a blue dot for the octopus' position.
        self.octopus_point, = self.ax.plot([], [], 'bo', markersize=8)
        # Initialize an arrow for the heading.
        self.heading_arrow = self.ax.arrow(0, 0, 0, 0, head_width=0.1, head_length=0.2, fc='b', ec='b')
        # Placeholder for food (red dot).
        self.food_plot, = self.ax.plot([], [], "ro")
        
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()

    def render(self, mode='human'):
        """
        Renders the environment using matplotlib.
        The octopus is represented as a blue dot with an arrow indicating its orientation.
        """
        #x, y, theta, v_x, v_y, omega, _, _, _ = self.state
        x, y, theta, v_x, v_y, omega, distance_to_food, _, _, _ = self.state
        
        # Update the octopus position.
        self.octopus_point.set_data([x], [y])
        
        # Remove the previous heading arrow.
        if self.heading_arrow in self.ax.patches:
            self.heading_arrow.remove()
            
        # Draw a new arrow representing the heading.
        arrow_length = 1.0  # Length of the arrow
        dx = arrow_length * np.cos(theta)
        dy = arrow_length * np.sin(theta)
        self.heading_arrow = self.ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.15, fc='b', ec='b')

        # Update the food position.
        self.food_plot.set_data([self.food_position[0]], [self.food_position[1]])

        # Update title with distance_to_food
        self.ax.set_title(f"Octopus Simulation (Distance to food: {distance_to_food:.2f})")

        # Update the figure.
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close(self.fig)

# Example usage for testing the decoupled model and rendering
if __name__ == '__main__':#
    env = OctopusEnv(epsilon=0.0)
    state = env.reset()
    
    # Apply a constant force for 20 steps, then zero out the forces.
    for i in range(1000):

        # Poll for keys without waiting
        # if keyboard.is_pressed('left'):
        #     action = [0, 1]
        # elif keyboard.is_pressed('right'):
        #     action = [2, 1]
        # elif keyboard.is_pressed('up'):
        #     action = [1, 2]
        # elif keyboard.is_pressed('down'):
        #     action = [1, 0]
        # else:
        #     action = [1, 1]  # default or idle action

        # Sample random actions for testing
        action = env.action_space.sample()

        state, reward, done, info = env.step(action)
        print(i, np.around(state, 2), np.around(action, 2), np.round(reward, 3))
        env.render()
        sleep(0.0)
        if done:
            state = env.reset()
    
    env.close()

