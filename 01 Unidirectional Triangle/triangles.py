import numpy as np
import gym
import matplotlib.pyplot as plt
from gym import spaces

class SwimmingAgentEnv(gym.Env):
    """
    A custom environment for a swimming agent (fish-like triangle with a hinge).
    The nose of the fish is defined by (x_position, y_position). From the nose, two segments 
    of fixed length are drawn backward. Their angular separation is defined by hinge_angle.
    An auxiliary function computes the x_velocity adjustment based on the difference in area of 
    the triangle defined by these segments, mimicking fluid displacement effects.
    """
    def __init__(self, render_every=1):
        super(SwimmingAgentEnv, self).__init__()
        
        # --- Improvement 4: Cache constant values ---
        self.pi = np.pi
        
        self.observation_space_description = ["x_position", "y_position", "velocity_x", "velocity_y", "orientation", "hinge_angle", "prev_action"]
        self.obs_dim = len(self.observation_space_description)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        # Action: Change hinge angle (continuous)
        self.action_space = spaces.Box(low=-0.2, high=0.2, shape=(1,), dtype=np.float32)

        self.segment_length = 0.1  # fixed segment length
        self.timestep_count = 0
        self.speed_decay = 0.99  # speed decay factor
        
        # --- Improvement 2: Set up persistent Matplotlib figure and axis for faster updates ---
        plt.ion()  # enable interactive mode
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.5, 0.5)
        self.ax.set_aspect('equal', adjustable='box')
        self.poly = None  # will hold the polygon for the fish
        self.line_upper, = self.ax.plot([], [], "k")  # upper segment line
        self.line_lower, = self.ax.plot([], [], "k")  # lower segment line
        self.ax.set_title("Agent Visualization")
        
        # Render frequency parameter: only update visualization every 'render_every' steps.
        self.render_every = render_every
        self.render_step_counter = 0

        self.reset()
    
    def reset(self):
        """
        Resets the environment to the initial state.
        """
        self.state = np.array([0.0] * self.obs_dim, dtype=np.float32)
        self.timestep_count = 0

        # Initial fish area
        self.prev_area = self.compute_fish_area()
        return self.state

    # --- Improvement 1: Helper method to compute endpoints ---
    def _compute_endpoints(self):
        """
        Computes the endpoints of the two segments (upper and lower) from the fish's nose.
        Returns (upper_x, upper_y, lower_x, lower_y).
        """
        nose_x, nose_y = self.state[0], self.state[1]
        hinge_angle = self.state[5]
        angle_upper = self.pi - hinge_angle / 2.0
        angle_lower = self.pi + hinge_angle / 2.0
        
        upper_x = nose_x + self.segment_length * np.cos(angle_upper)
        upper_y = nose_y + self.segment_length * np.sin(angle_upper)
        lower_x = nose_x + self.segment_length * np.cos(angle_lower)
        lower_y = nose_y + self.segment_length * np.sin(angle_lower)
        
        return upper_x, upper_y, lower_x, lower_y

    def compute_fish_area(self):
        """
        Computes the area of the triangle defined by the fish's nose and its two backward segments.
        """
        nose_x, nose_y = self.state[0], self.state[1]
        upper_x, upper_y, lower_x, lower_y = self._compute_endpoints()
        area = 0.5 * abs((upper_x - nose_x) * (lower_y - nose_y) - (lower_x - nose_x) * (upper_y - nose_y))
        return area

    def compute_x_velocity_adjustment(self, current_area):
        """
        Computes an adjustment to the x_velocity based on the difference in area.
        """
        area_diff = self.prev_area - current_area

        # --- Improvement 3: Using conditional computation for adjustment ---
        if area_diff < 0:
            adjustment = area_diff  # backward drag
        else:
            adjustment = area_diff * 2.0   # forward drag
        return adjustment

    def step(self, action):
        """
        Executes an action and updates the environment.
        """
        self.timestep_count += 1

        hinge_angle_change = action[0]
        # Clip the new hinge angle between 0 and π/2.
        self.state[5] = np.clip(self.state[5] + hinge_angle_change, 0, self.pi/2)
        
        current_area = self.compute_fish_area()
        x_velocity_adjustment = self.compute_x_velocity_adjustment(current_area)
        self.prev_area = current_area

        # Use x_velocity_adjustment as the new x velocity.
        velocity_x = x_velocity_adjustment
        velocity_y = 0.0   # no vertical change
        
        # Speed decay
        
        self.state[2] *= self.speed_decay
        self.state[3] *= self.speed_decay

        self.state[2] += velocity_x
        self.state[3] += velocity_y

        self.state[0] += self.state[2]
        self.state[1] += self.state[3]
        
        #reward = velocity_x # - abs(hinge_angle_change) * 0.01 #penalization for sudden changes (needs to be tuned)
        
        penalty = 0
        # if prev action has different sign of current action, add penalty
        if self.state[6] * action[0] < 0:
            penalty = -0.5

        # Update previous action
        self.state[6] = action[0]

        # Check if the fish has reached the right boundary
        finish_line = 1.0
        done = self.state[0] >= finish_line or self.timestep_count >= 1e3

        #print("Timestep: ", self.timestep_count)

        reward = -1 + penalty

        if done:
            #reward = - self.timestep_count / 1000 - abs(finish_line - self.state[0])
            #reward = - abs(finish_line - self.state[0]) * 100
            reward = 1

        return self.state, reward, done, {}

    def render(self):
        """
        Visualizes the fish as a triangle using persistent plotting.
        """
        self.render_step_counter += 1
        if self.render_step_counter % self.render_every != 0:
            return
        
        # Compute current endpoints
        nose_x, nose_y = self.state[0], self.state[1]
        upper_x, upper_y, lower_x, lower_y = self._compute_endpoints()
        
        # Prepare triangle points (nose, upper, lower)
        triangle_points = np.array([
            [nose_x, nose_y],
            [upper_x, upper_y],
            [lower_x, lower_y]
        ])
        
        # Update the polygon representing the fish
        # if self.poly is None:
        #     self.poly = self.ax.fill(triangle_points[:, 0], triangle_points[:, 1], "b")[0]
        # else:
        #     self.poly.set_xy(triangle_points)
        
        # Update lines for the segments from the nose
        self.line_upper.set_data([nose_x, upper_x], [nose_y, upper_y])
        self.line_lower.set_data([nose_x, lower_x], [nose_y, lower_y])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# --- Main simulation loop using improved rendering frequency ---
if __name__ == "__main__":
    env = SwimmingAgentEnv(render_every=1)  # Render every 2 steps to reduce overhead
    state = env.reset()
    
    while True:
        action = env.action_space.sample()
        print(f"Action: {action}")
        state, reward, done, _ = env.step(action)
        env.render()
        
        print(f"State: {state}, Reward: {reward:.4f}")
        print(f"X position: {state[0]:.4f}, Y position: {state[1]:.4f}")
        print("-" * 50)
        
        if done:
            break

    # Keep the window open after simulation finishes
    #plt.ioff()  # turn off interactive mode
    plt.show()
