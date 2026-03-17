"""V6 environment: articulated 3-segment fish with drag-driven locomotion.

This version uses a built-in swim gait controller so the policy outputs
high-level drive and steer commands, while propulsion and turning emerge from
joint motion plus anisotropic drag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Sequence, Tuple

import gymnasium as gym
from gymnasium.spaces import Box
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle, Wedge
import numpy as np


def _as_float32_array(values, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if shape is not None:
        array = array.reshape(shape)
    return array


def _cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _wrap_angle(theta: float) -> float:
    return float((theta + math.pi) % (2.0 * math.pi) - math.pi)


def _rotation(theta: float) -> np.ndarray:
    return np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float32,
    )


def _body_frame(vector: np.ndarray, theta: float) -> np.ndarray:
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return np.array(
        [cos_t * vector[0] + sin_t * vector[1], -sin_t * vector[0] + cos_t * vector[1]],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class FishMorphology:
    segment_lengths: tuple[float, float, float] = (0.72, 0.54, 0.38)
    segment_front_widths: tuple[float, float, float] = (0.22, 0.16, 0.10)
    segment_back_widths: tuple[float, float, float] = (0.18, 0.12, 0.05)


@dataclass(frozen=True)
class FishDynamicsConfig:
    mass: float = 1.0
    inertia: float = 1.6
    dt: float = 0.05
    max_integration_dt: float = 0.01
    segment_parallel_drag: float = 2.0
    segment_perp_drag: float = 30.0
    segment_angular_drag: float = 0.16
    root_rotational_drag: float = 0.9
    steering_torque_gain: float = 3.0
    gait_propulsion_gain: float = 2.5
    max_speed: float = 10.0
    max_angular_speed: float = 8.0


@dataclass(frozen=True)
class FishActuationConfig:
    joint_limit: float = 0.9
    joint_torque_limit: float = 12.0
    joint_kp: float = 24.0
    joint_kd: float = 6.0
    joint_passive_damping: float = 2.0
    joint_inertia: float = 0.18
    joint_max_speed: float = 8.0
    action_effort_penalty_scale: float = 0.006
    joint_limit_penalty_scale: float = 0.04


@dataclass(frozen=True)
class FishRenderConfig:
    segment_colors: tuple[str, str, str] = ("#1d6fa5", "#2f92bf", "#7fd0e6")
    joint_color: str = "#163b52"
    food_color: str = "#f05d23"


@dataclass(frozen=True)
class FishPreset:
    name: str
    morphology: FishMorphology
    dynamics: FishDynamicsConfig
    actuation: FishActuationConfig
    render: FishRenderConfig


EEL_3SEG_PRESET = FishPreset(
    name="eel_3seg",
    morphology=FishMorphology(),
    dynamics=FishDynamicsConfig(),
    actuation=FishActuationConfig(),
    render=FishRenderConfig(),
)


@dataclass
class ArticulatedFishState:
    root_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    root_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    root_theta: float = 0.0
    root_omega: float = 0.0
    joint_angles: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    joint_velocities: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    prev_action: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))


class ArticulatedFishEnv(gym.Env):
    """Single articulated fish environment with preset-ready internals."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        epsilon: float = 0.0,
        render_mode: str | None = None,
        fish_preset: FishPreset | str | None = None,
        time_limit: int = 600,
        food_count: int = 48,
        food_capture_radius: float = 0.45,
        pellet_reward: float = 1.0,
        step_cost: float = 0.002,
        sensor_radius: float = 4.5,
        sensor_ring_edges: Sequence[float] = (1.5, 3.0, 4.5),
        sensor_num_sectors: int = 12,
        show_sensor_overlay: bool = True,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.eps = float(epsilon)
        self.fish_preset = self._resolve_preset(fish_preset)
        self.time_limit = int(time_limit)
        if self.time_limit <= 0:
            raise ValueError("time_limit must be > 0.")
        self.food_count = int(food_count)
        if self.food_count <= 0:
            raise ValueError("food_count must be > 0.")
        self.food_capture_radius = float(food_capture_radius)
        if self.food_capture_radius <= 0.0:
            raise ValueError("food_capture_radius must be > 0.")
        self.pellet_reward = float(pellet_reward)
        self.step_cost = float(step_cost)
        if self.step_cost < 0.0:
            raise ValueError("step_cost must be >= 0.")
        self.sensor_radius = float(sensor_radius)
        if self.sensor_radius <= 0.0:
            raise ValueError("sensor_radius must be > 0.")
        self.sensor_ring_edges = np.asarray(sensor_ring_edges, dtype=np.float32).reshape(-1)
        if self.sensor_ring_edges.size == 0:
            raise ValueError("sensor_ring_edges must be non-empty.")
        if np.any(self.sensor_ring_edges <= 0.0):
            raise ValueError("sensor_ring_edges values must be > 0.")
        if np.any(self.sensor_ring_edges[:-1] >= self.sensor_ring_edges[1:]):
            raise ValueError("sensor_ring_edges must be strictly increasing.")
        if float(self.sensor_ring_edges[-1]) > self.sensor_radius + 1e-6:
            raise ValueError("sensor_ring_edges must lie within sensor_radius.")
        self.sensor_num_sectors = int(sensor_num_sectors)
        if self.sensor_num_sectors <= 0:
            raise ValueError("sensor_num_sectors must be > 0.")
        self.sensor_bin_count = int(self.sensor_ring_edges.size * self.sensor_num_sectors)
        self.show_sensor_overlay = bool(show_sensor_overlay)

        self.primary_agent_id = "fish_0"
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.obs_low = np.concatenate(
            [
                np.zeros(self.sensor_bin_count, dtype=np.float32),
                np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32),
            ]
        )
        self.obs_high = np.ones(self.sensor_bin_count + 10, dtype=np.float32)
        self.observation_space = Box(low=self.obs_low, high=self.obs_high, dtype=np.float32)

        self.border = 12.0
        self.playable_half_extent = self.border - 0.75
        self.food_min_spacing = 0.75
        self.food_min_spawn_distance = 1.0
        self.base_step_penalty = -float(self.step_cost)

        self.timestep = 0
        self.food_positions = np.zeros((self.food_count, 2), dtype=np.float32)
        self.food_eaten_episode = 0
        self.fish_state = ArticulatedFishState()
        self.swim_phase = 0.0
        self.last_reward = 0.0
        self.last_nearest_food_distance = float("nan")
        self.last_visible_food_count = 0
        self.last_sensor_bins = np.zeros(self.sensor_bin_count, dtype=np.float32)
        self.last_sensor_active_bins: list[int] = []
        self.last_reward_breakdown: Dict[str, float | bool] = {
            "pellets_eaten_this_step": 0,
            "pellet_reward_total": 0.0,
            "step_cost": float(self.step_cost),
            "total_reward": 0.0,
            "terminated": False,
            "truncated": False,
        }
        self.last_dynamics_debug: Dict[str, Any] = self._empty_dynamics_debug()

        self.fig = None
        self.ax = None
        self.segment_patches: list[Polygon] = []
        self.sensor_patches: list[Wedge] = []
        self.sensor_legend_artists: list[Any] = []
        self.food_scatter = None
        self.joint_plot = None

    def _resolve_preset(self, fish_preset: FishPreset | str | None) -> FishPreset:
        if fish_preset is None:
            return EEL_3SEG_PRESET
        if isinstance(fish_preset, FishPreset):
            return fish_preset
        if fish_preset == "eel_3seg":
            return EEL_3SEG_PRESET
        raise ValueError(f"Unsupported fish preset: {fish_preset}")

    def get_agent_ids(self) -> Tuple[str]:
        return (self.primary_agent_id,)

    def _empty_dynamics_debug(self) -> Dict[str, Any]:
        return {
            "joint_target_angles": np.zeros(2, dtype=np.float32),
            "joint_torques": np.zeros(2, dtype=np.float32),
            "joint_acceleration": np.zeros(2, dtype=np.float32),
            "joint_limit_ratio": np.zeros(2, dtype=np.float32),
            "drive_command": 0.0,
            "steer_command": 0.0,
            "drive_level": 0.0,
            "gait_amplitude": 0.0,
            "gait_frequency": 0.0,
            "steer_curvature": 0.0,
            "swim_phase_prev": 0.0,
            "swim_phase_next": 0.0,
            "segment_centers": np.zeros((3, 2), dtype=np.float32),
            "segment_angles": np.zeros(3, dtype=np.float32),
            "segment_velocities": np.zeros((3, 2), dtype=np.float32),
            "segment_angular_velocities": np.zeros(3, dtype=np.float32),
            "segment_drag_forces": np.zeros((3, 2), dtype=np.float32),
            "segment_drag_torques": np.zeros(3, dtype=np.float32),
            "joint_positions": np.zeros((2, 2), dtype=np.float32),
            "total_force": np.zeros(2, dtype=np.float32),
            "total_torque": 0.0,
            "steering_torque_bias": 0.0,
            "gait_propulsion_force": np.zeros(2, dtype=np.float32),
            "root_acceleration": np.zeros(2, dtype=np.float32),
            "root_angular_acceleration": 0.0,
            "energy_proxy": 0.0,
        }

    def _sample_food_candidate(self) -> np.ndarray:
        return self.np_random.uniform(
            low=-self.playable_half_extent,
            high=self.playable_half_extent,
            size=2,
        ).astype(np.float32)

    def _is_food_position_valid(
        self,
        candidate: np.ndarray,
        existing_positions: np.ndarray,
        *,
        exclude_index: int | None = None,
        require_spacing: bool = True,
    ) -> bool:
        if float(np.linalg.norm(candidate - self.fish_state.root_position)) < self.food_min_spawn_distance:
            return False
        if not require_spacing:
            return True
        for idx, existing in enumerate(existing_positions):
            if exclude_index is not None and idx == exclude_index:
                continue
            if float(np.linalg.norm(candidate - existing)) < self.food_min_spacing:
                return False
        return True

    def _sample_food_position(
        self,
        existing_positions: np.ndarray,
        *,
        exclude_index: int | None = None,
    ) -> np.ndarray:
        fallback_with_spawn_clearance = None
        last_candidate = None
        for _ in range(64):
            candidate = self._sample_food_candidate()
            last_candidate = candidate
            if float(np.linalg.norm(candidate - self.fish_state.root_position)) >= self.food_min_spawn_distance:
                fallback_with_spawn_clearance = candidate
            if self._is_food_position_valid(
                candidate,
                existing_positions,
                exclude_index=exclude_index,
                require_spacing=True,
            ):
                return candidate
        if fallback_with_spawn_clearance is not None:
            return fallback_with_spawn_clearance.astype(np.float32)
        if last_candidate is not None:
            return last_candidate.astype(np.float32)
        return self._sample_food_candidate()

    def _spawn_food_field(self) -> None:
        positions = np.zeros((self.food_count, 2), dtype=np.float32)
        for idx in range(self.food_count):
            positions[idx] = self._sample_food_position(positions[:idx], exclude_index=None)
        self.food_positions = positions

    def _respawn_food_indices(self, indices: np.ndarray) -> None:
        for idx in np.flatnonzero(indices):
            self.food_positions[idx] = self._sample_food_position(self.food_positions, exclude_index=int(idx))

    def _food_relative_vectors(self) -> np.ndarray:
        return self.food_positions - self.fish_state.root_position

    def _nearest_food_vector(self) -> np.ndarray:
        relative = self._food_relative_vectors()
        if relative.size == 0:
            return np.zeros(2, dtype=np.float32)
        distances = np.linalg.norm(relative, axis=1)
        return relative[int(np.argmin(distances))].astype(np.float32)

    def _nearest_food_distance(self) -> float:
        relative = self._food_relative_vectors()
        if relative.size == 0:
            return float("nan")
        return float(np.min(np.linalg.norm(relative, axis=1)))

    def _compute_polar_food_sensor(self) -> tuple[np.ndarray, int, list[int], float]:
        relative_world = self._food_relative_vectors()
        if relative_world.size == 0:
            bins = np.zeros(self.sensor_bin_count, dtype=np.float32)
            return bins, 0, [], float("nan")

        relative_body = np.asarray(
            [_body_frame(vector, self.fish_state.root_theta) for vector in relative_world],
            dtype=np.float32,
        )
        distances = np.linalg.norm(relative_body, axis=1)
        visible_mask = distances <= self.sensor_radius
        visible_count = int(np.count_nonzero(visible_mask))
        nearest_distance = float(np.min(distances)) if distances.size else float("nan")
        counts = np.zeros((self.sensor_ring_edges.size, self.sensor_num_sectors), dtype=np.float32)
        if visible_count > 0:
            visible_vectors = relative_body[visible_mask]
            visible_distances = distances[visible_mask]
            sector_size = 2.0 * math.pi / float(self.sensor_num_sectors)
            for vector, distance in zip(visible_vectors, visible_distances, strict=False):
                ring_index = int(np.searchsorted(self.sensor_ring_edges, distance, side="left"))
                if ring_index >= self.sensor_ring_edges.size:
                    continue
                angle = math.atan2(float(vector[1]), float(vector[0]))
                sector_index = int(math.floor(((angle + 0.5 * sector_size) % (2.0 * math.pi)) / sector_size))
                sector_index = min(max(sector_index, 0), self.sensor_num_sectors - 1)
                counts[ring_index, sector_index] += 1.0

        normalized = np.minimum(counts, 3.0) / 3.0
        flat = normalized.reshape(-1).astype(np.float32)
        active_bins = np.flatnonzero(flat > 0.0).astype(int).tolist()
        return flat, visible_count, active_bins, nearest_distance

    def _segment_angles(self, state: ArticulatedFishState | None = None) -> np.ndarray:
        state = state or self.fish_state
        q0, q1 = state.joint_angles.astype(np.float32)
        return np.array(
            [state.root_theta, state.root_theta + q0, state.root_theta + q0 + q1],
            dtype=np.float32,
        )

    def _segment_geometry(self, state: ArticulatedFishState | None = None) -> Dict[str, np.ndarray]:
        state = state or self.fish_state
        morphology = self.fish_preset.morphology
        angles = self._segment_angles(state)
        lengths = np.asarray(morphology.segment_lengths, dtype=np.float32)

        centers = np.zeros((3, 2), dtype=np.float32)
        joint_positions = np.zeros((2, 2), dtype=np.float32)
        centers[0] = state.root_position.astype(np.float32)

        heading0 = np.array([math.cos(float(angles[0])), math.sin(float(angles[0]))], dtype=np.float32)
        joint_positions[0] = centers[0] - 0.5 * lengths[0] * heading0

        heading1 = np.array([math.cos(float(angles[1])), math.sin(float(angles[1]))], dtype=np.float32)
        centers[1] = joint_positions[0] - 0.5 * lengths[1] * heading1
        joint_positions[1] = joint_positions[0] - lengths[1] * heading1

        heading2 = np.array([math.cos(float(angles[2])), math.sin(float(angles[2]))], dtype=np.float32)
        centers[2] = joint_positions[1] - 0.5 * lengths[2] * heading2

        return {
            "centers": centers,
            "angles": angles,
            "joint_positions": joint_positions,
        }

    def _kinematic_prediction(self, state: ArticulatedFishState, dt: float) -> ArticulatedFishState:
        return ArticulatedFishState(
            root_position=(state.root_position + state.root_velocity * dt).astype(np.float32),
            root_velocity=state.root_velocity.astype(np.float32).copy(),
            root_theta=float(state.root_theta + state.root_omega * dt),
            root_omega=float(state.root_omega),
            joint_angles=(state.joint_angles + state.joint_velocities * dt).astype(np.float32),
            joint_velocities=state.joint_velocities.astype(np.float32).copy(),
            prev_action=state.prev_action.astype(np.float32).copy(),
        )

    def _segment_angular_velocities(self, state: ArticulatedFishState) -> np.ndarray:
        qd0, qd1 = state.joint_velocities.astype(np.float32)
        return np.array(
            [state.root_omega, state.root_omega + qd0, state.root_omega + qd0 + qd1],
            dtype=np.float32,
        )

    def _compute_segment_kinematics(self, state: ArticulatedFishState) -> Dict[str, np.ndarray]:
        geometry_now = self._segment_geometry(state)
        centers = geometry_now["centers"]
        angles = geometry_now["angles"]
        joint_positions = geometry_now["joint_positions"]
        lengths = np.asarray(self.fish_preset.morphology.segment_lengths, dtype=np.float32)

        omega_segments = self._segment_angular_velocities(state)
        normals = np.array(
            [[-math.sin(float(angle)), math.cos(float(angle))] for angle in angles],
            dtype=np.float32,
        )

        segment_velocities = np.zeros((3, 2), dtype=np.float32)
        segment_velocities[0] = state.root_velocity.astype(np.float32)

        joint0_velocity = (
            segment_velocities[0] - 0.5 * lengths[0] * float(omega_segments[0]) * normals[0]
        ).astype(np.float32)
        segment_velocities[1] = (
            joint0_velocity - 0.5 * lengths[1] * float(omega_segments[1]) * normals[1]
        ).astype(np.float32)
        joint1_velocity = (
            joint0_velocity - lengths[1] * float(omega_segments[1]) * normals[1]
        ).astype(np.float32)
        segment_velocities[2] = (
            joint1_velocity - 0.5 * lengths[2] * float(omega_segments[2]) * normals[2]
        ).astype(np.float32)

        return {
            "centers": centers,
            "angles": angles,
            "joint_positions": joint_positions,
            "segment_velocities": segment_velocities.astype(np.float32),
            "segment_angular_velocities": omega_segments.astype(np.float32),
        }

    def _clip_root_velocity(self, velocity: np.ndarray) -> np.ndarray:
        speed = float(np.linalg.norm(velocity))
        max_speed = float(self.fish_preset.dynamics.max_speed)
        if speed <= max_speed or speed == 0.0:
            return velocity.astype(np.float32)
        return (velocity * (max_speed / speed)).astype(np.float32)

    def _clip_joint_velocities(self, joint_velocities: np.ndarray) -> np.ndarray:
        limit = float(self.fish_preset.actuation.joint_max_speed)
        return np.clip(joint_velocities, -limit, limit).astype(np.float32)

    def _clamp_joint_state(
        self,
        joint_angles: np.ndarray,
        joint_velocities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        limit = float(self.fish_preset.actuation.joint_limit)
        clipped_angles = np.clip(joint_angles, -limit, limit).astype(np.float32)
        clipped_velocities = joint_velocities.astype(np.float32).copy()
        for idx in range(clipped_angles.size):
            if clipped_angles[idx] >= limit and clipped_velocities[idx] > 0.0:
                clipped_velocities[idx] = 0.0
            if clipped_angles[idx] <= -limit and clipped_velocities[idx] < 0.0:
                clipped_velocities[idx] = 0.0
        return clipped_angles, clipped_velocities

    def _compute_gait_targets(
        self,
        action: np.ndarray,
        dt: float,
        swim_phase: float,
    ) -> Dict[str, np.ndarray | float]:
        actuation = self.fish_preset.actuation
        drive = float(np.clip(action[0], -1.0, 1.0))
        steer = float(np.clip(action[1], -1.0, 1.0))

        # Treat minimum drive as an idle command so rest/decay probes stay meaningful.
        if drive <= -1.0 + 1e-6:
            drive_level = 0.0
            amplitude = 0.0
            frequency = 0.0
            phase_next = float(swim_phase)
            oscillation_head = 0.0
            oscillation_tail = 0.0
        else:
            drive_level = 0.5 * (drive + 1.0)
            amplitude = 0.08 + 0.62 * drive_level
            frequency = 2.5 + 2.0 * drive_level
            phase_next = float((swim_phase + (2.0 * math.pi * frequency * dt)) % (2.0 * math.pi))
            oscillation_head = math.sin(phase_next)
            oscillation_tail = math.sin(phase_next - (math.pi / 2.0))

        curvature = 0.55 * steer
        target_angles = np.clip(
            np.array(
                [
                    0.65 * curvature + 0.35 * amplitude * oscillation_head,
                    0.25 * curvature + 0.75 * amplitude * oscillation_tail,
                ],
                dtype=np.float32,
            ),
            -float(actuation.joint_limit),
            float(actuation.joint_limit),
        ).astype(np.float32)
        return {
            "drive_command": drive,
            "steer_command": steer,
            "drive_level": float(drive_level),
            "gait_amplitude": float(amplitude),
            "gait_frequency": float(frequency),
            "steer_curvature": float(curvature),
            "swim_phase_prev": float(swim_phase),
            "swim_phase_next": float(phase_next),
            "target_angles": target_angles,
        }

    def _compute_external_wrench(self, state: ArticulatedFishState) -> Dict[str, Any]:
        morphology = self.fish_preset.morphology
        dynamics = self.fish_preset.dynamics
        kinematics = self._compute_segment_kinematics(state)
        centers = np.asarray(kinematics["centers"], dtype=np.float32)
        angles = np.asarray(kinematics["angles"], dtype=np.float32)
        segment_velocities = np.asarray(kinematics["segment_velocities"], dtype=np.float32)
        segment_angular_velocities = np.asarray(kinematics["segment_angular_velocities"], dtype=np.float32)

        segment_drag_forces = np.zeros((3, 2), dtype=np.float32)
        segment_drag_torques = np.zeros(3, dtype=np.float32)
        total_force = np.zeros(2, dtype=np.float32)
        total_torque = 0.0

        lengths = np.asarray(morphology.segment_lengths, dtype=np.float32)
        areas = lengths * 0.5 * (
            np.asarray(morphology.segment_front_widths, dtype=np.float32)
            + np.asarray(morphology.segment_back_widths, dtype=np.float32)
        )

        for idx in range(3):
            tangent = np.array([math.cos(float(angles[idx])), math.sin(float(angles[idx]))], dtype=np.float32)
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
            velocity = segment_velocities[idx]
            v_parallel = float(np.dot(velocity, tangent))
            v_perp = float(np.dot(velocity, normal))
            force = (
                -dynamics.segment_parallel_drag * areas[idx] * v_parallel * tangent
                - dynamics.segment_perp_drag * areas[idx] * v_perp * normal
            ).astype(np.float32)
            torque_drag = float(-dynamics.segment_angular_drag * areas[idx] * segment_angular_velocities[idx])
            segment_drag_forces[idx] = force
            segment_drag_torques[idx] = torque_drag
            total_force += force
            total_torque += _cross2d(centers[idx] - state.root_position, force) + torque_drag

        total_torque += float(-dynamics.root_rotational_drag * state.root_omega)
        return {
            "segment_centers": centers,
            "segment_angles": angles,
            "segment_velocities": segment_velocities,
            "segment_angular_velocities": segment_angular_velocities,
            "segment_drag_forces": segment_drag_forces,
            "segment_drag_torques": segment_drag_torques.astype(np.float32),
            "joint_positions": np.asarray(kinematics["joint_positions"], dtype=np.float32),
            "total_force": total_force.astype(np.float32),
            "total_torque": float(total_torque),
        }

    def _compute_actuation(
        self,
        state: ArticulatedFishState,
        action: np.ndarray,
        dt: float,
        swim_phase: float,
    ) -> Dict[str, np.ndarray | float]:
        actuation = self.fish_preset.actuation
        gait = self._compute_gait_targets(action, dt, swim_phase)
        target_angles = np.asarray(gait["target_angles"], dtype=np.float32)
        torque_cmd = (
            actuation.joint_kp * (target_angles - state.joint_angles)
            - actuation.joint_kd * state.joint_velocities
        )
        torque_cmd = np.clip(torque_cmd, -actuation.joint_torque_limit, actuation.joint_torque_limit).astype(np.float32)
        joint_acceleration = (
            torque_cmd - actuation.joint_passive_damping * state.joint_velocities
        ) / float(actuation.joint_inertia)
        joint_velocity_next = self._clip_joint_velocities(
            state.joint_velocities + joint_acceleration.astype(np.float32) * float(dt)
        )
        joint_angle_next, joint_velocity_next = self._clamp_joint_state(
            state.joint_angles + joint_velocity_next * float(dt),
            joint_velocity_next,
        )
        joint_limit_ratio = np.abs(joint_angle_next) / max(float(actuation.joint_limit), 1e-6)
        return {
            "target_angles": target_angles,
            "joint_torques": torque_cmd,
            "joint_acceleration": joint_acceleration.astype(np.float32),
            "joint_velocity_next": joint_velocity_next.astype(np.float32),
            "joint_angle_next": joint_angle_next.astype(np.float32),
            "joint_limit_ratio": joint_limit_ratio.astype(np.float32),
            "drive_command": float(gait["drive_command"]),
            "steer_command": float(gait["steer_command"]),
            "drive_level": float(gait["drive_level"]),
            "gait_amplitude": float(gait["gait_amplitude"]),
            "gait_frequency": float(gait["gait_frequency"]),
            "steer_curvature": float(gait["steer_curvature"]),
            "swim_phase_prev": float(gait["swim_phase_prev"]),
            "swim_phase_next": float(gait["swim_phase_next"]),
        }

    def get_dynamics_breakdown(
        self,
        action,
        *,
        state: ArticulatedFishState | None = None,
        dt: float | None = None,
        swim_phase: float | None = None,
    ) -> Dict[str, Any]:
        state = state or self.fish_state
        action = np.clip(_as_float32_array(action, shape=(2,)), -1.0, 1.0)
        dt = float(self.fish_preset.dynamics.dt if dt is None else dt)
        swim_phase = float(self.swim_phase if swim_phase is None else swim_phase)
        actuation = self._compute_actuation(state, action, dt, swim_phase)
        predicted_state = ArticulatedFishState(
            root_position=state.root_position.astype(np.float32).copy(),
            root_velocity=state.root_velocity.astype(np.float32).copy(),
            root_theta=float(state.root_theta),
            root_omega=float(state.root_omega),
            joint_angles=actuation["joint_angle_next"].astype(np.float32).copy(),
            joint_velocities=actuation["joint_velocity_next"].astype(np.float32).copy(),
            prev_action=action.astype(np.float32).copy(),
        )
        wrench = self._compute_external_wrench(predicted_state)
        dynamics = self.fish_preset.dynamics
        forward_direction = np.array(
            [math.cos(predicted_state.root_theta), math.sin(predicted_state.root_theta)],
            dtype=np.float32,
        )
        gait_propulsion_magnitude = float(
            dynamics.gait_propulsion_gain * float(actuation["drive_level"]) * float(np.mean(np.abs(actuation["joint_velocity_next"])))
        )
        gait_propulsion_force = (gait_propulsion_magnitude * forward_direction).astype(np.float32)
        steering_torque_bias = float(
            -dynamics.steering_torque_gain * float(actuation["steer_curvature"]) * max(float(actuation["drive_level"]), 0.0)
        )
        total_force = wrench["total_force"].astype(np.float32) + gait_propulsion_force
        total_torque = float(wrench["total_torque"]) + steering_torque_bias
        root_acceleration = total_force / float(dynamics.mass)
        root_angular_acceleration = float(total_torque / float(dynamics.inertia))
        energy_proxy = float(np.mean(np.abs(actuation["joint_torques"])) / max(self.fish_preset.actuation.joint_torque_limit, 1e-6))
        return {
            "joint_target_angles": actuation["target_angles"].astype(np.float32),
            "joint_torques": actuation["joint_torques"].astype(np.float32),
            "joint_acceleration": actuation["joint_acceleration"].astype(np.float32),
            "joint_limit_ratio": actuation["joint_limit_ratio"].astype(np.float32),
            "drive_command": float(actuation["drive_command"]),
            "steer_command": float(actuation["steer_command"]),
            "drive_level": float(actuation["drive_level"]),
            "gait_amplitude": float(actuation["gait_amplitude"]),
            "gait_frequency": float(actuation["gait_frequency"]),
            "steer_curvature": float(actuation["steer_curvature"]),
            "swim_phase_prev": float(actuation["swim_phase_prev"]),
            "swim_phase_next": float(actuation["swim_phase_next"]),
            "joint_velocity_next": actuation["joint_velocity_next"].astype(np.float32),
            "joint_angle_next": actuation["joint_angle_next"].astype(np.float32),
            "segment_centers": wrench["segment_centers"].astype(np.float32),
            "segment_angles": wrench["segment_angles"].astype(np.float32),
            "segment_velocities": wrench["segment_velocities"].astype(np.float32),
            "segment_angular_velocities": wrench["segment_angular_velocities"].astype(np.float32),
            "segment_drag_forces": wrench["segment_drag_forces"].astype(np.float32),
            "segment_drag_torques": wrench["segment_drag_torques"].astype(np.float32),
            "joint_positions": wrench["joint_positions"].astype(np.float32),
            "total_force": total_force.astype(np.float32),
            "total_torque": total_torque,
            "steering_torque_bias": steering_torque_bias,
            "gait_propulsion_force": gait_propulsion_force.astype(np.float32),
            "root_acceleration": root_acceleration.astype(np.float32),
            "root_angular_acceleration": float(root_angular_acceleration),
            "energy_proxy": energy_proxy,
        }

    def _integrate_substep(
        self,
        state: ArticulatedFishState,
        action: np.ndarray,
        swim_phase: float,
        dt: float,
    ) -> tuple[ArticulatedFishState, float, Dict[str, Any]]:
        dynamics = self.fish_preset.dynamics
        dynamics_debug = self.get_dynamics_breakdown(action, state=state, dt=dt, swim_phase=swim_phase)
        next_root_velocity = self._clip_root_velocity(
            state.root_velocity + np.asarray(dynamics_debug["root_acceleration"], dtype=np.float32) * dt
        )
        next_root_omega = float(
            np.clip(
                state.root_omega + float(dynamics_debug["root_angular_acceleration"]) * dt,
                -dynamics.max_angular_speed,
                dynamics.max_angular_speed,
            )
        )
        next_root_theta = _wrap_angle(state.root_theta + next_root_omega * dt)
        next_root_position = (state.root_position + next_root_velocity * dt).astype(np.float32)

        next_state = ArticulatedFishState(
            root_position=next_root_position,
            root_velocity=next_root_velocity.astype(np.float32),
            root_theta=float(next_root_theta),
            root_omega=float(next_root_omega),
            joint_angles=np.asarray(dynamics_debug["joint_angle_next"], dtype=np.float32).copy(),
            joint_velocities=np.asarray(dynamics_debug["joint_velocity_next"], dtype=np.float32).copy(),
            prev_action=np.clip(_as_float32_array(action, shape=(2,)), -1.0, 1.0).astype(np.float32),
        )
        return next_state, float(dynamics_debug["swim_phase_next"]), dynamics_debug

    def get_reward_breakdown(
        self,
        *,
        pellets_eaten_this_step: int,
        truncated: bool,
    ) -> Dict[str, float | bool | int]:
        pellet_reward_total = float(self.pellet_reward * pellets_eaten_this_step)
        total_reward = float(pellet_reward_total - self.step_cost)
        return {
            "pellets_eaten_this_step": int(pellets_eaten_this_step),
            "pellet_reward_total": pellet_reward_total,
            "step_cost": float(self.step_cost),
            "total_reward": total_reward,
            "terminated": False,
            "truncated": bool(truncated),
        }

    def set_debug_state(
        self,
        position,
        velocity,
        theta: float,
        omega: float,
        food_positions,
        timestep: int = 0,
        joint_angles=None,
        joint_velocities=None,
        prev_action=None,
        swim_phase: float = 0.0,
    ) -> None:
        self.fish_state = ArticulatedFishState(
            root_position=_as_float32_array(position, shape=(2,)).copy(),
            root_velocity=_as_float32_array(velocity, shape=(2,)).copy(),
            root_theta=float(theta),
            root_omega=float(omega),
            joint_angles=_as_float32_array(joint_angles if joint_angles is not None else [0.0, 0.0], shape=(2,)).copy(),
            joint_velocities=_as_float32_array(joint_velocities if joint_velocities is not None else [0.0, 0.0], shape=(2,)).copy(),
            prev_action=_as_float32_array(prev_action if prev_action is not None else [0.0, 0.0], shape=(2,)).copy(),
        )
        food_positions_array = np.asarray(food_positions, dtype=np.float32).reshape(-1, 2)
        if food_positions_array.shape[0] != self.food_count:
            raise ValueError(f"Expected {self.food_count} food positions, got {food_positions_array.shape[0]}.")
        self.food_positions = food_positions_array.copy()
        self.timestep = int(timestep)
        self.swim_phase = float(swim_phase)
        self.last_reward = 0.0
        self.food_eaten_episode = 0
        self.last_nearest_food_distance = self._nearest_food_distance()
        self.last_sensor_bins = np.zeros(self.sensor_bin_count, dtype=np.float32)
        self.last_visible_food_count = 0
        self.last_sensor_active_bins = []
        self.last_reward_breakdown = {
            "pellets_eaten_this_step": 0,
            "pellet_reward_total": 0.0,
            "step_cost": float(self.step_cost),
            "total_reward": 0.0,
            "terminated": False,
            "truncated": False,
        }
        self.last_dynamics_debug = self._empty_dynamics_debug()

    def get_debug_snapshot(self) -> Dict[str, Any]:
        obs = self._get_obs()
        nearest_vector = self._nearest_food_vector()
        geometry = self._segment_geometry(self.fish_state)
        return {
            "position": self.fish_state.root_position.astype(np.float32).copy(),
            "velocity": self.fish_state.root_velocity.astype(np.float32).copy(),
            "theta": float(self.fish_state.root_theta),
            "omega": float(self.fish_state.root_omega),
            "root_position": self.fish_state.root_position.astype(np.float32).copy(),
            "root_velocity": self.fish_state.root_velocity.astype(np.float32).copy(),
            "root_theta": float(self.fish_state.root_theta),
            "root_omega": float(self.fish_state.root_omega),
            "joint_angles": self.fish_state.joint_angles.astype(np.float32).copy(),
            "joint_velocities": self.fish_state.joint_velocities.astype(np.float32).copy(),
            "prev_action": self.fish_state.prev_action.astype(np.float32).copy(),
            "food_positions": self.food_positions.astype(np.float32).copy(),
            "nearest_food_vector": nearest_vector.astype(np.float32).copy(),
            "nearest_food_distance": float(self.last_nearest_food_distance),
            "sensor_bins": self.last_sensor_bins.astype(np.float32).copy(),
            "visible_food_count": int(self.last_visible_food_count),
            "sensor_active_bins": list(self.last_sensor_active_bins),
            "food_eaten_episode": int(self.food_eaten_episode),
            "timestep": int(self.timestep),
            "swim_phase": float(self.swim_phase),
            "observation": obs.copy(),
            "segment_centers": geometry["centers"].astype(np.float32).copy(),
            "segment_angles": geometry["angles"].astype(np.float32).copy(),
            "joint_positions": geometry["joint_positions"].astype(np.float32).copy(),
            "reward_breakdown": dict(self.last_reward_breakdown),
            "dynamics_breakdown": {
                key: (np.asarray(value, dtype=np.float32).copy() if isinstance(value, np.ndarray) else value)
                for key, value in self.last_dynamics_debug.items()
            },
        }

    def _get_obs(self) -> np.ndarray:
        sensor_bins, visible_food_count, active_bins, nearest_food_distance = self._compute_polar_food_sensor()
        self.last_sensor_bins = sensor_bins.astype(np.float32).copy()
        self.last_visible_food_count = int(visible_food_count)
        self.last_sensor_active_bins = list(active_bins)
        self.last_nearest_food_distance = float(nearest_food_distance)
        root_velocity_body = _body_frame(self.fish_state.root_velocity, self.fish_state.root_theta)
        dynamics = self.fish_preset.dynamics
        actuation = self.fish_preset.actuation

        proprioception = np.array(
            [
                np.clip(root_velocity_body[0] / dynamics.max_speed, -1.0, 1.0),
                np.clip(root_velocity_body[1] / dynamics.max_speed, -1.0, 1.0),
                np.clip(self.fish_state.root_omega / dynamics.max_angular_speed, -1.0, 1.0),
                np.clip(self.fish_state.joint_angles[0] / actuation.joint_limit, -1.0, 1.0),
                np.clip(self.fish_state.joint_angles[1] / actuation.joint_limit, -1.0, 1.0),
                np.clip(self.fish_state.joint_velocities[0] / actuation.joint_max_speed, -1.0, 1.0),
                np.clip(self.fish_state.joint_velocities[1] / actuation.joint_max_speed, -1.0, 1.0),
                np.clip(self.fish_state.prev_action[0], -1.0, 1.0),
                np.clip(self.fish_state.prev_action[1], -1.0, 1.0),
                np.clip(self.timestep / self.time_limit, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return np.concatenate([sensor_bins.astype(np.float32), proprioception], dtype=np.float32)

    def _compute_next_state(self, action: np.ndarray) -> tuple[ArticulatedFishState, float]:
        dynamics = self.fish_preset.dynamics
        total_dt = float(dynamics.dt)
        max_substep_dt = max(1e-4, min(float(dynamics.max_integration_dt), total_dt))
        substeps = max(1, int(math.ceil(total_dt / max_substep_dt)))
        substep_dt = total_dt / substeps
        next_state = ArticulatedFishState(
            root_position=self.fish_state.root_position.astype(np.float32).copy(),
            root_velocity=self.fish_state.root_velocity.astype(np.float32).copy(),
            root_theta=float(self.fish_state.root_theta),
            root_omega=float(self.fish_state.root_omega),
            joint_angles=self.fish_state.joint_angles.astype(np.float32).copy(),
            joint_velocities=self.fish_state.joint_velocities.astype(np.float32).copy(),
            prev_action=np.clip(_as_float32_array(action, shape=(2,)), -1.0, 1.0).astype(np.float32),
        )
        next_swim_phase = float(self.swim_phase)
        dynamics_debug = self._empty_dynamics_debug()
        for _ in range(substeps):
            next_state, next_swim_phase, dynamics_debug = self._integrate_substep(
                next_state,
                next_state.prev_action,
                next_swim_phase,
                substep_dt,
            )
        self.last_dynamics_debug = dynamics_debug
        return next_state, next_swim_phase

    def _compute_reward_flags(self, pellets_eaten_this_step: int) -> Tuple[float, bool, bool]:
        truncated = bool(self.timestep >= self.time_limit)
        reward_breakdown = self.get_reward_breakdown(
            pellets_eaten_this_step=int(pellets_eaten_this_step),
            truncated=truncated,
        )
        reward = float(reward_breakdown["total_reward"])
        terminated = bool(reward_breakdown["terminated"])
        truncated = bool(reward_breakdown["truncated"])
        self.last_reward = reward
        self.last_reward_breakdown = reward_breakdown
        return reward, terminated, truncated

    def _build_info(self, *, truncated: bool) -> Dict[str, Any]:
        return {
            "agent_id": self.primary_agent_id,
            "nearest_food_distance": float(self.last_nearest_food_distance),
            "last_reward": float(self.last_reward),
            "reward_breakdown": dict(self.last_reward_breakdown),
            "food_eaten_this_step": int(self.last_reward_breakdown["pellets_eaten_this_step"]),
            "food_eaten_episode": int(self.food_eaten_episode),
            "visible_food_count": int(self.last_visible_food_count),
            "sensor_active_bins": list(self.last_sensor_active_bins),
            "truncated": bool(truncated),
            "fish_preset": self.fish_preset.name,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.timestep = 0
        self.swim_phase = 0.0
        initial_theta = float(self.np_random.uniform(-math.pi, math.pi))

        self.fish_state = ArticulatedFishState(
            root_position=np.zeros(2, dtype=np.float32),
            root_velocity=np.zeros(2, dtype=np.float32),
            root_theta=initial_theta,
            root_omega=0.0,
            joint_angles=np.zeros(2, dtype=np.float32),
            joint_velocities=np.zeros(2, dtype=np.float32),
            prev_action=np.zeros(2, dtype=np.float32),
        )
        self.food_eaten_episode = 0
        self._spawn_food_field()
        self.last_reward = 0.0
        self.last_nearest_food_distance = self._nearest_food_distance()
        self.last_visible_food_count = 0
        self.last_sensor_bins = np.zeros(self.sensor_bin_count, dtype=np.float32)
        self.last_sensor_active_bins = []
        self.last_reward_breakdown = {
            "pellets_eaten_this_step": 0,
            "pellet_reward_total": 0.0,
            "step_cost": float(self.step_cost),
            "total_reward": 0.0,
            "terminated": False,
            "truncated": False,
        }
        self.last_dynamics_debug = self._empty_dynamics_debug()

        obs = self._get_obs()
        info = {
            "agent_id": self.primary_agent_id,
            "nearest_food_distance": float(self.last_nearest_food_distance),
            "food_eaten_episode": int(self.food_eaten_episode),
            "visible_food_count": int(self.last_visible_food_count),
            "sensor_active_bins": list(self.last_sensor_active_bins),
            "fish_preset": self.fish_preset.name,
        }
        return obs, info

    def step(self, action):
        action = np.clip(_as_float32_array(action, shape=(2,)), -1.0, 1.0)
        if self.eps > 0.0:
            mask = self.np_random.random(2) < self.eps
            if np.any(mask):
                random_action = self.np_random.uniform(-1.0, 1.0, size=2).astype(np.float32)
                action = np.where(mask, random_action, action).astype(np.float32)

        self.fish_state, self.swim_phase = self._compute_next_state(action)
        self.timestep += 1

        relative = self._food_relative_vectors()
        distances = np.linalg.norm(relative, axis=1)
        eaten_mask = distances <= self.food_capture_radius
        pellets_eaten_this_step = int(np.count_nonzero(eaten_mask))
        if pellets_eaten_this_step > 0:
            self.food_eaten_episode += pellets_eaten_this_step
            self._respawn_food_indices(eaten_mask)
        reward, terminated, truncated = self._compute_reward_flags(pellets_eaten_this_step)
        obs = self._get_obs()
        info = self._build_info(truncated=truncated)
        return obs, reward, terminated, truncated, info

    def _initialize_rendering(self) -> None:
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.ax.set_xlim(-self.border, self.border)
        self.ax.set_ylim(-self.border, self.border)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.plot(
            [-self.border, self.border, self.border, -self.border, -self.border],
            [-self.border, -self.border, self.border, self.border, -self.border],
            color="#23303d",
            linewidth=1.0,
        )
        for color in self.fish_preset.render.segment_colors:
            patch = Polygon(
                np.zeros((4, 2), dtype=np.float32),
                closed=True,
                facecolor=color,
                edgecolor="#0d2533",
                linewidth=1.0,
            )
            self.ax.add_patch(patch)
            self.segment_patches.append(patch)
        if self.show_sensor_overlay:
            sector_width_deg = 360.0 / float(self.sensor_num_sectors)
            for ring_index, outer_radius in enumerate(self.sensor_ring_edges):
                inner_radius = 0.0 if ring_index == 0 else float(self.sensor_ring_edges[ring_index - 1])
                for _ in range(self.sensor_num_sectors):
                    patch = Wedge(
                        center=(0.0, 0.0),
                        r=float(outer_radius),
                        theta1=-0.5 * sector_width_deg,
                        theta2=0.5 * sector_width_deg,
                        width=float(outer_radius - inner_radius),
                        facecolor="#35d4ff",
                        edgecolor="none",
                        alpha=0.0,
                        zorder=0.2,
                    )
                    self.ax.add_patch(patch)
                    self.sensor_patches.append(patch)
            self._add_sensor_legend()
        self.food_scatter = self.ax.scatter([], [], s=20, c=self.fish_preset.render.food_color, zorder=2.5)
        self.joint_plot, = self.ax.plot([], [], "o", color=self.fish_preset.render.joint_color, markersize=4)

    def _segment_polygon(self, center: np.ndarray, angle: float, length: float, front_width: float, back_width: float) -> np.ndarray:
        local_points = np.array(
            [
                [0.5 * length, 0.5 * front_width],
                [0.5 * length, -0.5 * front_width],
                [-0.5 * length, -0.5 * back_width],
                [-0.5 * length, 0.5 * back_width],
            ],
            dtype=np.float32,
        )
        return (_rotation(angle) @ local_points.T).T + center

    def _sensor_bin_alpha(self, intensity: float) -> float:
        intensity = float(np.clip(intensity, 0.0, 1.0))
        if intensity <= 0.0:
            return 0.0
        return 0.04 + (0.28 * intensity)

    def _add_sensor_legend(self) -> None:
        if self.ax is None:
            return

        panel_x = 0.02
        panel_y = 0.70
        panel_w = 0.32
        panel_h = 0.26
        overlay_color = "#35d4ff"
        text_color = "#d9f6ff"
        z_order = 4.5

        panel = Rectangle(
            (panel_x, panel_y),
            panel_w,
            panel_h,
            transform=self.ax.transAxes,
            facecolor=(0.03, 0.09, 0.14, 0.84),
            edgecolor=(0.26, 0.71, 0.85, 0.40),
            linewidth=1.0,
            clip_on=False,
            zorder=z_order,
        )
        self.ax.add_patch(panel)
        self.sensor_legend_artists.append(panel)

        legend_rows = [
            ("Local Food Sensor", 0.93, 9, "bold"),
            ("R1: 0-1.5", 0.875, 8, None),
            ("R2: 1.5-3.0", 0.84, 8, None),
            ("R3: 3.0-4.5", 0.805, 8, None),
            ("12 sectors", 0.77, 8, None),
            ("sector 0 = forward", 0.735, 8, None),
            ("sectors advance CCW", 0.702, 8, None),
        ]
        for label, y_pos, font_size, font_weight in legend_rows:
            artist = self.ax.text(
                panel_x + 0.02,
                y_pos,
                label,
                transform=self.ax.transAxes,
                color=text_color,
                fontsize=font_size,
                fontweight=font_weight,
                ha="left",
                va="center",
                zorder=z_order + 0.1,
            )
            self.sensor_legend_artists.append(artist)

        arrow = FancyArrowPatch(
            (panel_x + 0.18, 0.735),
            (panel_x + 0.28, 0.735),
            transform=self.ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=12.0,
            linewidth=1.0,
            color=overlay_color,
            zorder=z_order + 0.2,
            clip_on=False,
        )
        self.ax.add_patch(arrow)
        self.sensor_legend_artists.append(arrow)

        swatch_y = panel_y + 0.03
        swatch_w = 0.038
        swatch_h = 0.028
        swatch_gap = 0.062
        swatch_start_x = panel_x + 0.025
        swatch_labels = ["0", "1", "2", "3+"]
        swatch_intensities = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]

        for index, (label, intensity) in enumerate(zip(swatch_labels, swatch_intensities)):
            swatch_x = swatch_start_x + (index * swatch_gap)
            swatch = Rectangle(
                (swatch_x, swatch_y),
                swatch_w,
                swatch_h,
                transform=self.ax.transAxes,
                facecolor=overlay_color,
                edgecolor=(0.26, 0.71, 0.85, 0.85),
                linewidth=0.8,
                alpha=self._sensor_bin_alpha(intensity),
                clip_on=False,
                zorder=z_order + 0.2,
            )
            self.ax.add_patch(swatch)
            self.sensor_legend_artists.append(swatch)
            label_artist = self.ax.text(
                swatch_x + (0.5 * swatch_w),
                swatch_y - 0.012,
                label,
                transform=self.ax.transAxes,
                color=text_color,
                fontsize=7,
                ha="center",
                va="top",
                zorder=z_order + 0.2,
            )
            self.sensor_legend_artists.append(label_artist)

    def render(self):
        if self.render_mode != "human":
            return
        if self.fig is None:
            self._initialize_rendering()

        geometry = self._segment_geometry(self.fish_state)
        morphology = self.fish_preset.morphology
        for idx, patch in enumerate(self.segment_patches):
            polygon = self._segment_polygon(
                geometry["centers"][idx],
                float(geometry["angles"][idx]),
                float(morphology.segment_lengths[idx]),
                float(morphology.segment_front_widths[idx]),
                float(morphology.segment_back_widths[idx]),
            )
            patch.set_xy(polygon)

        self.food_scatter.set_offsets(self.food_positions)
        self.joint_plot.set_data(geometry["joint_positions"][:, 0], geometry["joint_positions"][:, 1])
        if self.sensor_patches:
            sector_width = 2.0 * math.pi / float(self.sensor_num_sectors)
            theta_base = float(self.fish_state.root_theta)
            sensor_bins = self.last_sensor_bins.reshape(self.sensor_ring_edges.size, self.sensor_num_sectors)
            for ring_index, outer_radius in enumerate(self.sensor_ring_edges):
                inner_radius = 0.0 if ring_index == 0 else float(self.sensor_ring_edges[ring_index - 1])
                for sector_index in range(self.sensor_num_sectors):
                    patch = self.sensor_patches[ring_index * self.sensor_num_sectors + sector_index]
                    center_angle = theta_base + (sector_index * sector_width)
                    patch.set_center((float(self.fish_state.root_position[0]), float(self.fish_state.root_position[1])))
                    patch.set_radius(float(outer_radius))
                    patch.set_width(float(outer_radius - inner_radius))
                    patch.theta1 = math.degrees(center_angle - 0.5 * sector_width)
                    patch.theta2 = math.degrees(center_angle + 0.5 * sector_width)
                    intensity = float(sensor_bins[ring_index, sector_index])
                    patch.set_alpha(self._sensor_bin_alpha(intensity))

        nearest_food_distance = self.last_nearest_food_distance
        joint_text = ", ".join(f"{value:+.2f}" for value in self.fish_state.joint_angles)
        self.ax.set_title(
            f"V6 Foraging ({self.fish_preset.name}) | step={self.timestep}/{self.time_limit} "
            f"| eaten={self.food_eaten_episode} | nearest={nearest_food_distance:.2f} "
            f"| visible={self.last_visible_food_count} | joints=[{joint_text}]"
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
            self.segment_patches = []
            self.sensor_patches = []
            self.sensor_legend_artists = []
            self.food_scatter = None
            self.joint_plot = None


OctopusEnv = ArticulatedFishEnv


if __name__ == "__main__":
    env = ArticulatedFishEnv(epsilon=0.1, render_mode="human")
    obs, info = env.reset()
    print("Initial observation:", np.round(obs, 3))
    print("Initial info:", info)
    print("Agent IDs:", env.get_agent_ids())

    for _ in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
