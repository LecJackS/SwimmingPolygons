from time import sleep
from click import pause
import numpy as np
import gym
import matplotlib.pyplot as plt
from gym import spaces

import numpy as np
from torch import cos_

class SwimmerTriangle:
    """
    A class representing a swimming triangle (or fish) with a controllable hinge angle 
    and global rotation. The triangle is defined by three vertices:
      - The "nose" (anchor point) at the origin in local coordinates.
      - Two rear vertices whose positions depend on the segment length and the hinge angle.
    The global shape is obtained by applying a rotation matrix (using the global orientation) 
    and then translating by the current position.
    """
    
    def __init__(self, 
                 position=np.array([0.0, 0.0]), 
                 velocity=np.array([0.0, 0.0]),
                 segment_length=0.1, 
                 hinge_angle=0.0, 
                 global_orientation=0.0):
        """
        Initializes the SwimmerTriangle.
        
        Parameters:
        - position: A 2D numpy array representing the global position of the nose.
        - velocity: A 2D numpy array representing the initial velocity.
        - segment_length: The length from the nose to the rear vertices.
        - hinge_angle: The initial internal hinge angle (in radians) that determines the spread.
        - global_orientation: The initial global orientation (in radians) of the swimmer.
        """
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.segment_length = segment_length
        self.hinge_angle = hinge_angle  # internal angle controlling the spread of the rear vertices
        self.global_orientation = global_orientation  # overall rotation of the triangle
        
    def compute_local_points(self):
        """
        Computes the vertices of the triangle in the local (body-fixed) coordinate system.
        
        Returns:
            A numpy array of shape (3,2) containing the local coordinates of:
              - The nose (always at [0,0]).
              - The upper rear vertex.
              - The lower rear vertex.
              
        The rear vertices are computed based on the hinge angle. For example, one can define:
          upper_vertex = (-L, L * tan(hinge_angle/2))
          lower_vertex = (-L, -L * tan(hinge_angle/2))
        """
        L = self.segment_length
        # SOH CAH TOA: tan(angle) = opposite / adjacent
        half_spread = np.tan(self.hinge_angle / 2.0) * L
        
        # Local coordinates:
        nose = np.array([0.0, 0.0])
        upper = np.array([-L, half_spread])
        lower = np.array([-L, -half_spread])
        
        return np.array([nose, upper, lower])
    
    
    def get_global_vertices(self):
        """
        Transforms the local vertices to global coordinates using the current global_orientation 
        and translation by the position.
        
        Returns:
            A numpy array of shape (3,2) with the global coordinates of the triangle's vertices.
        """
        # Get local vertices
        local_points = self.compute_local_points()
        
        # Construct the rotation matrix for the global orientation.
        theta = self.global_orientation
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        # Apply the rotation to all local vertices and translate by the global position.
        global_points = (rotation_matrix @ local_points.T).T + self.position
        
        return global_points
    
    def update_state(self, delta_hinge, delta_rotation, delta_position):
        """
        Updates the swimmer's internal state given control inputs.
        
        Parameters:
        - delta_hinge: Change to be added to the hinge angle.
        - delta_rotation: Change to be added to the global orientation.
        - delta_position: A 2D numpy array to be added to the current position.
        
        This method updates the hinge angle, global orientation, and position.
        """
        self.hinge_angle += delta_hinge
        self.hinge_angle = np.clip(self.hinge_angle, -np.pi / 2, np.pi / 2)

        self.global_orientation += delta_rotation
        self.global_orientation = np.mod(self.global_orientation, 2 * np.pi)

        self.position += np.array(delta_position, dtype=np.float32)
        
    def get_state(self):
        """
        Returns the current state of the swimmer as a dictionary.
        """
        return {
            "position": self.position,
            "velocity": self.velocity,
            "hinge_angle": self.hinge_angle,
            "global_orientation": self.global_orientation,
            "global_vertices": self.get_global_vertices()
        }



class SwimmingAgentEnv(gym.Env):
    """
    A custom environment for a swimming agent using the SwimmerTriangle class.
    The environment now supports 2D movement:
      - At each reset, a random food position is sampled.
      - The agent (swimmer) can adjust its internal hinge angle and rotate globally.
      - The swimmer’s geometry is managed by a dedicated SwimmerTriangle object.
      - Thrust is computed based on the change in the swimmer’s area (from its global vertices),
        and the resulting 2D velocity is updated accordingly.
    """
    def __init__(self, render_every=1):
        super(SwimmingAgentEnv, self).__init__()
        self.pi = np.pi
        
        # Observation: [x_pos, y_pos, v_x, v_y, global_orientation, hinge_angle, prev_rotation_action, prev_hinge_action, food_x, food_y, countdown]
        self.obs_dim = 12
        # Set observation bounds loosely here; adjust as needed.
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        
        # Action: [delta_hinge, delta_rotation] (both continuous)
        self.nA = 3
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.nA,), dtype=np.float32)
        
        self.segment_length = 0.1
        self.speed_decay = 0.5
        self.timestep_count = 0
        self.episode_max_duration = 100
        self.time_penalty = -1/self.episode_max_duration
        
        self.swimmer = SwimmerTriangle(
            position=np.array([0.2, 0.2]),
            velocity=np.array([0.0, 0.0]),
            segment_length=self.segment_length,
            hinge_angle=0.0,
            global_orientation=0.0
        )

        self.prev_action = np.zeros(self.nA)
        
        # Food position (2D target): sample at reset.
        self.food_position = np.array([0.0, 0.0])
        
        self.initialize_rendering()
        
        self.render_every = render_every
        self.render_step_counter = 0

        self.reset()


    def initialize_rendering(self):
        """
        Set up persistent Matplotlib figure and axis for rendering.
        """
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-1.1, 1.1)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_aspect('equal', adjustable='box')
        # For drawing the swimmer, we use a line (with the triangle closed)...
        self.line, = self.ax.plot([], [], "b-")
        # Draw velocity vector in blue starting from the nose
        self.vel_arrow = self.ax.arrow(0, 0, 0, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
        # ...and a red dot for the food.
        self.food_plot, = self.ax.plot([], [], "ro")
        self.ax.set_title("Agent Visualization")
        
    
    def reset(self):
        """
        Resets the environment:
          - Resets the timestep counter.
          - Resets the swimmer to initial state.
          - Samples a new food position.
          - Returns the initial observation.
        """
        self.timestep_count = 0
        self.swimmer = SwimmerTriangle(
            position=np.array([np.random.uniform(0, 1), np.random.uniform(0, 1)]),
            velocity=np.array([0.0, 0.0]),
            segment_length=self.segment_length,
            hinge_angle=0.0,
            global_orientation=0.0
        )
        self.prev_action = np.zeros(self.nA)

        # Sample food position
        #self.food_position = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
        #self.food_position = np.array([0.9, 0.0])
        self.food_position = np.array([np.random.uniform(0, 1), np.random.uniform(0, 1)])
        return self._get_obs()
    
    def _get_obs(self):
        """
        Constructs the observation vector from the swimmer’s state and the food position.
        Observation:
          [x_pos, y_pos, v_x, v_y,
          global_orientation, hinge_angle,
          prev_rotation_action_0, prev_rotation_action_1, prev_hinge_action,
          food_x, food_y, countdown]
        """
        state = self.swimmer.get_state()  # returns a dict with position, velocity, hinge_angle, and global_orientation
        pos = state["position"]
        vel = state["velocity"]
        orientation = state["global_orientation"]
        hinge = state["hinge_angle"]

        # Concatenate pos, vel, orientation, hinge, prev_action, and food position.
        obs = np.concatenate([pos.flatten(),
                              vel.flatten(),
                              [orientation, hinge],
                              self.prev_action.flatten(),
                              self.food_position.flatten(),
                              [(self.episode_max_duration - self.timestep_count)/self.episode_max_duration]])
        return obs
    
    def compute_fish_area(self):
        """
        Computes the area of the swimmer's triangle using its global vertices.
        """
        vertices = self.swimmer.get_global_vertices()  # shape (3,2)
        nose = vertices[0]
        upper = vertices[1]
        lower = vertices[2]
        area = 0.5 * abs((upper[0] - nose[0]) * (lower[1] - nose[1]) - (lower[0] - nose[0]) * (upper[1] - nose[1]))
        return area
    
    def compute_thrust(self, current_area, prev_area):
        """
        Computes a thrust magnitude based on the difference in area.
          - If the area decreases, it produces a backward drag.
          - If the area increases, it produces a stronger forward thrust.
        """
        area_diff = prev_area - current_area
        if area_diff < 0:
            thrust = 0 #area_diff       # backward drag
        else:
            thrust = area_diff * 10.0   # forward thrust (scaled)
        return thrust
    
    def compute_angle_from_vector(self, vector):
        """
        Computes the angle (in radians) of a 2D vector relative to the x-axis.
        """
        return np.arctan2(vector[1], vector[0])
    
    def _distance_to_food(self):
        """
        Returns the Euclidean distance from the swimmer to the food.
        """
        return np.linalg.norm(self.swimmer.position - self.food_position)
    
    def step(self, action):
        """
        Executes an action and updates the environment:
          - Action is a 2D vector: [delta_hinge, delta_rotation].
          - The swimmer updates its hinge angle and global orientation.
          - The thrust is computed from the change in the swimmer's area.
          - The 2D velocity is updated with a decay factor and the computed thrust along the forward direction.
          - The position is updated by integrating the velocity.
          - A penalty is applied if the hinge change reverses direction abruptly.
          - The episode ends if the swimmer reaches the food's x position or after a timestep limit.
        """
        self.timestep_count += 1
        delta_rotation = self.compute_angle_from_vector(action[:2])

        self.swimmer.update_state

        delta_hinge = action[2]
        
        delta_hinge /= 5 # Only values between -0.2 and 0.2 are allowed for the hinge angle
        delta_rotation /= 16 # Only values between -0.2 and 0.2 are allowed for the rotation
        
        # Save current area for thrust computation.
        prev_area = self.compute_fish_area()
        
        # Update swimmer: change hinge angle and global orientation.
        self.swimmer.update_state(delta_hinge, delta_rotation, delta_position=np.array([0.0, 0.0]))
        
        # Compute new area and derive thrust.
        current_area = self.compute_fish_area()
        thrust = self.compute_thrust(current_area, prev_area)
        
        # Compute the forward direction (based on the swimmer's global orientation).
        forward_direction = np.array([np.cos(self.swimmer.global_orientation), np.sin(self.swimmer.global_orientation)])
        thrust_vector = thrust * forward_direction
        
        # Update velocity with decay and thrust.
        self.swimmer.velocity *= self.speed_decay
        self.swimmer.velocity += thrust_vector
        
        # Update position using the new velocity.
        self.swimmer.position += self.swimmer.velocity
        
        # Penalize abrupt reversal of hinge action.
        change_rot_penalty = -0.005 * delta_rotation
        change_hing_penalty = 0 if np.sign(self.prev_action[2]) == np.sign(action[2]) else -0.008 * abs(self.prev_action[2] - action[2])

        penalty = change_rot_penalty + change_hing_penalty
        activity_reward = 10 * thrust

        self.prev_action = action.copy()
        
        dist_to_food = self._distance_to_food()

        # Define done condition: reached food, timestep limit reached or out of bounds.
        out_of_bounds_x = self.swimmer.position[0] < -1.0 or self.swimmer.position[0] > 1.0
        out_of_bounds_y = self.swimmer.position[1] < -1.0 or self.swimmer.position[1] > 1.0

        food_eaten = dist_to_food < 0.02
        done = food_eaten or (self.timestep_count >= self.episode_max_duration) #or out_of_bounds_x or out_of_bounds_y

        # Compute reward: constant per timestep plus penalty; bonus on reaching the target.
        reward = self.time_penalty + penalty + activity_reward # TODO: Remove this later on
        if done:
            reward += 10.0 * food_eaten - dist_to_food
        
        return self._get_obs(), reward, done, {}
    
    def render(self):
        """
        Renders the swimmer and the food using persistent Matplotlib plotting.
        """
        self.render_step_counter += 1
        if self.render_step_counter % self.render_every != 0:
            return
        
        # Get the global vertices of the swimmer and close the triangle.
        vertices = self.swimmer.get_global_vertices()
        triangle = np.vstack([vertices, vertices[0]])
        
        # Update the line data for the swimmer.
        self.line.set_data(triangle[:, 0], triangle[:, 1])
        
        # Update the food position (red dot).
        self.food_plot.set_data(self.food_position[0], self.food_position[1])

        # Draw velocity vector in blue starting from the nose
        nose = vertices[0]
        self.vel_arrow.remove()
        self.vel_arrow = self.ax.arrow(nose[0], nose[1], self.swimmer.velocity[0], self.swimmer.velocity[1], head_width=0.03, head_length=0.04,
                                       fc='w', ec='b',
                                       label="Velocity")

        # Set title as triangle position and distance to food
        dist_to_food = self._distance_to_food()
        global_pos = self.swimmer.position
        self.ax.set_title(f"Agent at ({np.round(global_pos, 2)}) - dist to food: {dist_to_food:.3f})")

        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# --- Main simulation loop using improved rendering frequency ---
if __name__ == "__main__":
    env = SwimmingAgentEnv(render_every=1)  # Render every 2 steps to reduce overhead
    state = env.reset()
    
    while True:
        print("-" * 50)
        action = env.action_space.sample()
        print(f"Action: {action}")
        state, reward, done, _ = env.step(action)
        env.render()
        
        #print(f"State: {state}, Reward: {reward:.4f}")
        #print(f"X position: {state[0]:.4f}, Y position: {state[1]:.4f}")
        
        
        #env.pretty_print_state(state)
        #sleep(0.5)
        
        
        if done:
            break

    # Keep the window open after simulation finishes
    #plt.ioff()  # turn off interactive mode
    plt.show()
