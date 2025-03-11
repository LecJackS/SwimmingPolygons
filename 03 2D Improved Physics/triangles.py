from math import dist
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

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
    
    def __init__(self):
        super(OctopusEnv, self).__init__()
        # Dynamics parameters
        self.mass = 1.0                   # Mass of the object
        self.inertia = 1.0                # Moment of inertia
        self.drag_coefficient = 100       # Linear drag coefficient for translation (c)
        self.rotational_damping = 100     # Rotational damping coefficient (c_theta)
        self.dt = 0.02                    # Time step for integration
        
        # Action space: [u_f, u_r]
        self.action_space = spaces.Box(low=-10, high=10.0, shape=(2,), dtype=np.float32)
        
        # Observation space: [x, y, theta, v_x, v_y, omega]
        self.nA = 9
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.nA,), dtype=np.float32)
        
        self.state = None
        self.food_position = None
        self.timestep = 0
        self.time_limit = 100
        
        self.reset()
        self.border = 5.0
        self.initialize_rendering()

    def step(self, action):
        # Unpack current state
        x, y, theta, v_x, v_y, omega, _, prev_u_f, prev_u_r = self.state
        u_f, u_r = np.clip(action, self.action_space.low, self.action_space.high)

        u_f *= 20
        u_r *= 20
        

        # Rotational dynamics
        theta_next = theta + omega * self.dt
        omega_next = omega + ((u_r - self.rotational_damping * omega) / self.inertia) * self.dt

        # Translational dynamics (decoupled)
        # Thrust is applied along the current heading (theta)

        #thrust = np.array([np.cos(theta), np.sin(theta)]) * u_f # TODO: Change this again<<<
        thrust = np.array([np.cos(theta_next), np.sin(theta_next)]) * u_f

        # Compute acceleration: a = (thrust - drag) / mass
        accel = (thrust - self.drag_coefficient * np.array([v_x, v_y])) / self.mass

        
        # Euler integration for translation
        x_next = x + v_x * self.dt
        y_next = y + v_y * self.dt
        v_next = np.array([v_x, v_y]) + accel * self.dt
        
        
        
        dist_to_food = np.linalg.norm(np.array([x_next, y_next]) - self.food_position)


        self.state = np.array([x_next, y_next, theta_next, v_next[0], v_next[1], omega_next, dist_to_food, u_f, u_r], dtype=np.float32)
        
        # Placeholder reward: negative distance from the origin (modify as needed)
        
        reward = -1 #/(self.time_limit - self.timestep)
        done = False
        self.timestep += 1

        b = self.border
        if x_next < -b or x_next > b or y_next < -b or y_next > b:
            done = True
            #reward -= dist_to_food

        if dist_to_food < 0.5:
            done = True
            reward += 10.0

        if self.timestep >= self.time_limit:
            done = True
            reward -= dist_to_food

        info = {}
        
        return self.state, reward, done, info

    def reset(self):
        # Initialize the state: [x, y, theta, v_x, v_y, omega]
        self.state = np.zeros((self.nA,), dtype=np.float32)
        self.timestep = 0
        self.food_position = np.array([2,2]) #np.random.uniform(-2.0, 2.0, size=(2,))
        return self.state

    def initialize_rendering(self):
        """
        Set up persistent Matplotlib figure and axis for rendering.
        """
        plt.ion()  # Interactive mode on
        self.fig, self.ax = plt.subplots()
        # Adjust these limits based on expected simulation domain.
        self.ax.set_xlim(-10.0, 10.0)
        self.ax.set_ylim(-10.0, 10.0)
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
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def render(self, mode='human'):
        """
        Renders the environment using matplotlib.
        The octopus is represented as a blue dot with an arrow indicating its orientation.
        """
        x, y, theta, v_x, v_y, omega, _, _, _ = self.state
        
        # Update the octopus position.
        self.octopus_point.set_data(x, y)
        
        # Remove the previous heading arrow.
        if self.heading_arrow in self.ax.patches:
            self.heading_arrow.remove()
            
        # Draw a new arrow representing the heading.
        arrow_length = 1.0  # Length of the arrow
        dx = arrow_length * np.cos(theta)
        dy = arrow_length * np.sin(theta)
        self.heading_arrow = self.ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.15, fc='b', ec='b')

        # Update the food position.
        self.food_plot.set_data(self.food_position[0], self.food_position[1])

        
        # Update the figure.
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close(self.fig)

# Example usage for testing the decoupled model and rendering
if __name__ == '__main__':
    env = OctopusEnv()
    state = env.reset()
    
    # Apply a constant force for 20 steps, then zero out the forces.
    for i in range(100):
        #action = np.array([1.0, 0.5], dtype=np.float32) if i < 20 or i > 50 else np.array([0.0, 0.0], dtype=np.float32)
        # Sample random actions for testing
        action = np.random.uniform(env.action_space.low, env.action_space.high)
        # Use a short sleep for the initial phase and longer later, to see the effect.
        #sleep(0.0 if i < 20 else 1.0)
        state, reward, done, info = env.step(action)
        print(i, np.around(state, 2))
        env.render()
        if done:
            state = env.reset()
    
    env.close()

