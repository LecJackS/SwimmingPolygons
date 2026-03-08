"""V4 environment: decoupled fish dynamics with Gymnasium API.

This version keeps the V3-style physics/reward flow but provides:
- Gymnasium-native step/reset signatures.
- Flat observation vector with relative target encoding only.
- FishConfig/FishState structures for future multi-fish extensibility.
- Single-agent identity helpers (`fish_0`) for future PettingZoo adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Tuple

import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class FishConfig:
    mass: float = 10.0
    inertia: float = 1.0
    drag_coefficient: float = 50.0
    rotational_damping: float = 20.0
    dt: float = 0.05
    max_speed: float = 10.0
    max_angular_speed: float = 10.0
    body_length: float = 0.8
    body_width: float = 0.45


@dataclass
class FishState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    theta: float = 0.0
    omega: float = 0.0
    prev_turn_action: int = 1
    prev_push_action: int = 1


class OctopusEnv(gym.Env):
    """Single-fish environment with future-ready internal structure."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        epsilon: float = 0.0,
        render_mode: str | None = None,
        fish_config: FishConfig | None = None,
        enable_curriculum: bool = True,
        fixed_food_distance: float | None = None,
        time_limit: int = 100,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.eps = float(epsilon)
        self.fish_config = fish_config or FishConfig()
        self.enable_curriculum = bool(enable_curriculum)
        self.fixed_food_distance = float(fixed_food_distance) if fixed_food_distance is not None else None
        if self.fixed_food_distance is not None and self.fixed_food_distance <= 0:
            raise ValueError("fixed_food_distance must be > 0.")
        self.time_limit = int(time_limit)
        if self.time_limit <= 0:
            raise ValueError("time_limit must be > 0.")

        # Adapter-ready identity convention for future multi-agent wrappers.
        self.primary_agent_id = "fish_0"

        self.n_actions = 3
        self.action_space = MultiDiscrete([self.n_actions, self.n_actions])
        self.turn_values = np.array([-100.0, 0.0, 100.0], dtype=np.float32)
        self.push_values = np.array([-100.0, 0.0, 100.0], dtype=np.float32)

        # Observation:
        # [rel_x, rel_y, rel_dist, cos(theta), sin(theta),
        #  vx, vy, omega, prev_turn, prev_push, progress]
        self.obs_low = np.array([-1.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32)
        self.obs_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)

        self.border = 5.0
        self.success_distance = 0.5

        self.min_food_dist = 0.7
        self.max_food_dist = 10.0
        self.food_dist = self.min_food_dist
        if self.fixed_food_distance is not None:
            self.food_dist = self.fixed_food_distance
        self.run_avg_duration = float(self.time_limit)

        self.timestep = 0
        self.food_position = np.zeros(2, dtype=np.float32)
        self.fish_state = FishState()

        self.fig = None
        self.ax = None
        self.fish_plot = None
        self.food_plot = None

    def get_agent_ids(self) -> Tuple[str]:
        return (self.primary_agent_id,)

    def _clip_velocity(self, velocity: np.ndarray) -> np.ndarray:
        speed = float(np.linalg.norm(velocity))
        if speed <= self.fish_config.max_speed or speed == 0.0:
            return velocity
        return velocity * (self.fish_config.max_speed / speed)

    def _sample_food_position(self) -> None:
        run_avg_threshold = 50.0
        if self.fixed_food_distance is not None:
            rand_dist = float(self.fixed_food_distance)
        else:
            if self.enable_curriculum and (self.food_dist < self.max_food_dist) and (self.run_avg_duration < run_avg_threshold):
                self.food_dist = min(1.1 * self.food_dist, self.max_food_dist)
                self.run_avg_duration = float(self.time_limit)
                print("> Increasing food distance to:", np.round(self.food_dist, 2))

            rand_dist = float(self.np_random.uniform(self.min_food_dist, self.food_dist))
        rand_angle = float(self.np_random.uniform(0.0, 2.0 * math.pi))
        self.food_position = np.array(
            [rand_dist * math.cos(rand_angle), rand_dist * math.sin(rand_angle)],
            dtype=np.float32,
        )

    def _target_relative_vector(self) -> np.ndarray:
        return self.food_position - self.fish_state.position

    def _get_obs(self) -> np.ndarray:
        rel = self._target_relative_vector()
        dist = float(np.linalg.norm(rel))
        cfg = self.fish_config

        rel_scale = max(self.max_food_dist, 1.0)
        rel_x = float(np.clip(rel[0] / rel_scale, -1.0, 1.0))
        rel_y = float(np.clip(rel[1] / rel_scale, -1.0, 1.0))
        rel_dist = float(np.clip(dist / rel_scale, 0.0, 1.0))

        vx = float(np.clip(self.fish_state.velocity[0] / cfg.max_speed, -1.0, 1.0))
        vy = float(np.clip(self.fish_state.velocity[1] / cfg.max_speed, -1.0, 1.0))
        omega = float(np.clip(self.fish_state.omega / cfg.max_angular_speed, -1.0, 1.0))

        prev_turn = float(np.clip(self.fish_state.prev_turn_action - 1, -1, 1))
        prev_push = float(np.clip(self.fish_state.prev_push_action - 1, -1, 1))
        progress = float(np.clip(self.timestep / self.time_limit, 0.0, 1.0))

        return np.array(
            [
                rel_x,
                rel_y,
                rel_dist,
                math.cos(self.fish_state.theta),
                math.sin(self.fish_state.theta),
                vx,
                vy,
                omega,
                prev_turn,
                prev_push,
                progress,
            ],
            dtype=np.float32,
        )

    def _compute_next_state(self, turn_idx: int, push_idx: int) -> FishState:
        cfg = self.fish_config
        state = self.fish_state
        torque = float(self.turn_values[turn_idx])
        thrust_magnitude = float(self.push_values[push_idx])

        thrust_vector = np.array(
            [math.cos(state.theta), math.sin(state.theta)],
            dtype=np.float32,
        ) * thrust_magnitude
        acceleration = (thrust_vector - cfg.drag_coefficient * state.velocity) / cfg.mass

        next_position = state.position + state.velocity * cfg.dt
        next_velocity = state.velocity + acceleration * cfg.dt
        next_velocity = self._clip_velocity(next_velocity.astype(np.float32))

        next_theta = (state.theta + state.omega * cfg.dt) % (2.0 * math.pi)
        next_omega = state.omega + ((torque - cfg.rotational_damping * state.omega) / cfg.inertia) * cfg.dt
        next_omega = float(np.clip(next_omega, -cfg.max_angular_speed, cfg.max_angular_speed))

        return FishState(
            position=next_position.astype(np.float32),
            velocity=next_velocity.astype(np.float32),
            theta=float(next_theta),
            omega=next_omega,
            prev_turn_action=turn_idx,
            prev_push_action=push_idx,
        )

    def _compute_reward_flags(self, dist_to_food: float) -> Tuple[float, bool, bool]:
        reward = -1.0
        terminated = False
        truncated = False

        if dist_to_food < self.success_distance:
            terminated = True
            reward = 0.0
        elif self.timestep >= self.time_limit:
            truncated = True

        if terminated or truncated:
            self.run_avg_duration = 0.9 * self.run_avg_duration + 0.1 * self.timestep

        return reward, terminated, truncated

    def _build_info(self, dist_to_food: float, terminated: bool) -> Dict[str, float | bool | str]:
        return {
            "agent_id": self.primary_agent_id,
            "distance_to_food": dist_to_food,
            "run_avg_duration": float(self.run_avg_duration),
            "food_dist_limit": float(self.food_dist),
            "success": bool(terminated),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.timestep = 0
        self._sample_food_position()

        self.fish_state = FishState(
            position=np.zeros(2, dtype=np.float32),
            velocity=np.zeros(2, dtype=np.float32),
            theta=float(self.np_random.uniform(0.0, 2.0 * math.pi)),
            omega=0.0,
            prev_turn_action=1,
            prev_push_action=1,
        )

        obs = self._get_obs()
        info = {
            "agent_id": self.primary_agent_id,
            "distance_to_food": float(np.linalg.norm(self._target_relative_vector())),
            "food_dist_limit": float(self.food_dist),
        }
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        if action.size != 2:
            raise ValueError(f"Expected action of size 2, got shape {action.shape}")

        turn_idx = int(np.clip(action[0], 0, self.n_actions - 1))
        push_idx = int(np.clip(action[1], 0, self.n_actions - 1))

        if self.np_random.random() < self.eps:
            turn_idx = int(self.np_random.integers(0, self.n_actions))
        if self.np_random.random() < self.eps:
            push_idx = int(self.np_random.integers(0, self.n_actions))

        self.fish_state = self._compute_next_state(turn_idx, push_idx)
        self.timestep += 1

        dist_to_food = float(np.linalg.norm(self._target_relative_vector()))
        reward, terminated, truncated = self._compute_reward_flags(dist_to_food)
        obs = self._get_obs()
        info = self._build_info(dist_to_food, terminated)
        return obs, reward, terminated, truncated, info

    def _initialize_rendering(self) -> None:
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-self.border, self.border)
        self.ax.set_ylim(-self.border, self.border)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.plot(
            [-self.border, self.border, self.border, -self.border, -self.border],
            [-self.border, -self.border, self.border, self.border, -self.border],
            "k-",
            linewidth=1,
        )
        self.fish_plot, = self.ax.plot([], [], "b-", linewidth=2)
        self.food_plot, = self.ax.plot([], [], "ro", markersize=6)

    def _fish_vertices(self) -> np.ndarray:
        cfg = self.fish_config
        local_points = np.array(
            [
                [cfg.body_length * 0.5, 0.0],
                [-cfg.body_length * 0.5, cfg.body_width * 0.5],
                [-cfg.body_length * 0.5, -cfg.body_width * 0.5],
            ],
            dtype=np.float32,
        )
        cos_t = math.cos(self.fish_state.theta)
        sin_t = math.sin(self.fish_state.theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
        return (rot @ local_points.T).T + self.fish_state.position

    def render(self):
        if self.render_mode != "human":
            return

        if self.fig is None:
            self._initialize_rendering()

        vertices = self._fish_vertices()
        triangle = np.vstack([vertices, vertices[0]])
        self.fish_plot.set_data(triangle[:, 0], triangle[:, 1])
        self.food_plot.set_data([self.food_position[0]], [self.food_position[1]])

        dist = float(np.linalg.norm(self._target_relative_vector()))
        self.ax.set_title(
            f"V4 Fish ({self.primary_agent_id}) | step={self.timestep}/{self.time_limit} | dist={dist:.2f}"
        )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None
            self.ax = None


if __name__ == "__main__":
    env = OctopusEnv(epsilon=0.1, render_mode="human")
    obs, info = env.reset()
    print("Initial observation:", np.round(obs, 3))
    print("Initial info:", info)
    print("Agent IDs:", env.get_agent_ids())

    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
