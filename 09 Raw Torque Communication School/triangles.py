"""V9 environment: raw-torque articulated schooling with local communication."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict

from gymnasium.spaces import Box, Dict as DictSpace, Discrete
from gymnasium.utils import seeding
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle, Wedge
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


RED_TEAM = 0
BLUE_TEAM = 1
TEAM_NAMES = {RED_TEAM: "red", BLUE_TEAM: "blue"}
TEAM_COLORS = {RED_TEAM: "#d1495b", BLUE_TEAM: "#2f6fe4"}


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


def _normalize_message_token(value: Any, *, num_tokens: int) -> int:
    try:
        token = int(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        token = 0
    return int(np.clip(token, 0, num_tokens - 1))


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
    max_speed: float = 10.0
    max_angular_speed: float = 8.0


@dataclass(frozen=True)
class FishActuationConfig:
    joint_limit: float = 0.9
    joint_torque_limit: float = 12.0
    joint_passive_damping: float = 2.0
    joint_inertia: float = 0.18
    joint_max_speed: float = 8.0


@dataclass(frozen=True)
class FishRenderConfig:
    red_segment_colors: tuple[str, str, str] = ("#8f2533", "#c03d50", "#f08f98")
    blue_segment_colors: tuple[str, str, str] = ("#1d458c", "#2f6fe4", "#8eb4ff")
    joint_color: str = "#163b52"
    red_food_color: str = "#f05d23"
    blue_food_color: str = "#4f7cff"
    edible_food_sensor_color: str = "#35d4ff"
    non_edible_food_sensor_color: str = "#ff8a80"
    teammate_sensor_color: str = "#6fdc6f"
    opponent_sensor_color: str = "#f2c14e"
    teammate_message_sensor_color: str = "#7ff2ff"
    opponent_message_sensor_color: str = "#d3a4ff"


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
    prev_message_token: int = 0
    applied_joint_torque: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))


class CommunicatingSchoolEnv(MultiAgentEnv):
    """Raw-torque multi-agent schooling task with one shared trainable policy."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    CHANNEL_ORDER = (
        "edible_food",
        "non_edible_food",
        "teammate",
        "opponent",
        "teammate_message",
        "opponent_message",
    )

    def __init__(
        self,
        epsilon: float = 0.0,
        render_mode: str | None = None,
        render_profile: str = "fast",
        render_engine: str = "auto",
        fish_preset: FishPreset | str | None = None,
        time_limit: int = 300,
        num_red_fish: int = 10,
        num_blue_fish: int = 10,
        num_red_pellets: int = 48,
        num_blue_pellets: int = 48,
        food_capture_radius: float = 0.45,
        pellet_reward: float = 1.0,
        step_cost: float = 0.002,
        sector_radius: float = 5.0,
        sector_num: int = 6,
        communication_radius: float | None = None,
        num_message_tokens: int = 4,
        reward_mode: str = "forage",
        history_length: int = 8,
        actuator_time_constant: float = 0.10,
        show_sensor_overlay: bool = True,
        focus_agent_id: str = "fish_0",
        mute_received_messages: bool = False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.render_profile = str(render_profile).strip().lower()
        self.render_engine = str(render_engine).strip().lower()
        self.eps = float(epsilon)
        self.fish_preset = self._resolve_preset(fish_preset)
        self.time_limit = int(time_limit)
        self.num_red_fish = int(num_red_fish)
        self.num_blue_fish = int(num_blue_fish)
        self.num_red_pellets = int(num_red_pellets)
        self.num_blue_pellets = int(num_blue_pellets)
        self.num_fish = self.num_red_fish + self.num_blue_fish
        self.food_count = self.num_red_pellets + self.num_blue_pellets
        self.food_capture_radius = float(food_capture_radius)
        self.pellet_reward = float(pellet_reward)
        self.step_cost = float(step_cost)
        self.sector_radius = float(sector_radius)
        self.communication_radius = float(communication_radius if communication_radius is not None else sector_radius)
        self.sector_num = int(sector_num)
        self.num_message_tokens = int(num_message_tokens)
        self.reward_mode = str(reward_mode).strip().lower()
        self.history_length = int(history_length)
        self.actuator_time_constant = float(actuator_time_constant)
        self.show_sensor_overlay = bool(show_sensor_overlay)
        self.mute_received_messages = bool(mute_received_messages)
        if self.time_limit <= 0:
            raise ValueError("time_limit must be > 0.")
        if self.reward_mode not in {"forage", "locomotion_debug"}:
            raise ValueError("reward_mode must be 'forage' or 'locomotion_debug'.")
        if self.num_red_fish <= 0:
            raise ValueError("num_red_fish must be > 0.")
        if self.reward_mode == "forage":
            if self.num_blue_fish <= 0:
                raise ValueError("num_blue_fish must be > 0 in forage mode.")
            if self.num_red_pellets <= 0 or self.num_blue_pellets <= 0:
                raise ValueError("num_red_pellets and num_blue_pellets must be > 0 in forage mode.")
        else:
            if self.num_blue_fish < 0:
                raise ValueError("num_blue_fish must be >= 0 in locomotion_debug mode.")
            if self.num_red_pellets < 0 or self.num_blue_pellets < 0:
                raise ValueError("pellet counts must be >= 0 in locomotion_debug mode.")
        if self.food_capture_radius <= 0.0:
            raise ValueError("food_capture_radius must be > 0.")
        if self.pellet_reward <= 0.0:
            raise ValueError("pellet_reward must be > 0.")
        if self.step_cost < 0.0:
            raise ValueError("step_cost must be >= 0.")
        if self.sector_radius <= 0.0:
            raise ValueError("sector_radius must be > 0.")
        if self.communication_radius <= 0.0:
            raise ValueError("communication_radius must be > 0.")
        if self.sector_num != 6:
            raise ValueError("V9 keeps a fixed 6-sector exteroception contract.")
        if self.num_message_tokens != 4:
            raise ValueError("V9 keeps a fixed 4-token communication contract.")
        if self.history_length <= 0:
            raise ValueError("history_length must be > 0.")
        if self.actuator_time_constant < 0.0:
            raise ValueError("actuator_time_constant must be >= 0.")
        if self.render_profile not in {"fast", "full"}:
            raise ValueError("render_profile must be 'fast' or 'full'.")
        if self.render_engine not in {"auto", "blit", "safe"}:
            raise ValueError("render_engine must be 'auto', 'blit', or 'safe'.")

        self.possible_agents = [f"fish_{idx}" for idx in range(self.num_fish)]
        self.agents = list(self.possible_agents)
        self._agent_ids = set(self.possible_agents)
        self.agent_index = {agent_id: idx for idx, agent_id in enumerate(self.possible_agents)}
        self.focus_agent_id = focus_agent_id if focus_agent_id in self._agent_ids else self.possible_agents[0]
        self.agent_team_index = {
            agent_id: (RED_TEAM if idx < self.num_red_fish else BLUE_TEAM)
            for idx, agent_id in enumerate(self.possible_agents)
        }

        self.action_space = DictSpace(
            {
                "motion": Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                "message": Discrete(self.num_message_tokens),
            }
        )
        context_low = np.zeros(3, dtype=np.float32)
        context_high = np.ones(3, dtype=np.float32)
        history_low = np.full(self.history_length * 9, -1.0, dtype=np.float32)
        history_high = np.ones(self.history_length * 9, dtype=np.float32)
        obs_low = np.concatenate([np.zeros(36, dtype=np.float32), context_low, history_low])
        obs_high = np.concatenate([np.ones(36, dtype=np.float32), context_high, history_high])
        self.observation_space = Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.observation_spaces = {agent_id: self.observation_space for agent_id in self.possible_agents}
        self.action_spaces = {agent_id: self.action_space for agent_id in self.possible_agents}

        self.border = 12.0
        self.playable_half_extent = self.border - 0.75
        self.food_min_spacing = 0.75
        self.food_min_spawn_distance = 1.0
        self.spawn_ring_radius = 2.4
        self.min_spawn_separation = 1.2

        self.np_random, _ = seeding.np_random(None)
        self.timestep = 0
        self.food_positions = np.zeros((self.food_count, 2), dtype=np.float32)
        self.food_team_indices = self._default_food_team_indices()
        self.fish_states = {agent_id: ArticulatedFishState() for agent_id in self.possible_agents}
        self.last_message_tokens = {agent_id: 0 for agent_id in self.possible_agents}
        self.agent_food_eaten_episode = {agent_id: 0 for agent_id in self.possible_agents}
        self.team_food_eaten_episode = {RED_TEAM: 0, BLUE_TEAM: 0}
        self.last_rewards = {agent_id: 0.0 for agent_id in self.possible_agents}
        self.last_reward_breakdown = {
            agent_id: self.get_reward_breakdown(agent_id=agent_id, food_eaten_this_step=0, truncated=False)
            for agent_id in self.possible_agents
        }
        self.last_dynamics_debug = {agent_id: self._empty_dynamics_debug() for agent_id in self.possible_agents}
        self.control_histories = {
            agent_id: np.zeros((self.history_length, 9), dtype=np.float32)
            for agent_id in self.possible_agents
        }
        self.last_observation_channels = {
            agent_id: {name: np.zeros(self.sector_num, dtype=np.float32) for name in self.CHANNEL_ORDER}
            for agent_id in self.possible_agents
        }
        self.last_visible_counts = {
            agent_id: {"edible_food": 0, "non_edible_food": 0, "teammate": 0, "opponent": 0}
            for agent_id in self.possible_agents
        }
        self.last_sensor_active_bins = {
            agent_id: {name: [] for name in self.CHANNEL_ORDER}
            for agent_id in self.possible_agents
        }
        self.last_nearest_food_distance = {agent_id: float("nan") for agent_id in self.possible_agents}
        self.last_capture_distance = {agent_id: float("nan") for agent_id in self.possible_agents}
        self.last_joint_limit_occupancy = {agent_id: 0.0 for agent_id in self.possible_agents}
        self.last_joint_zero_crossings = {agent_id: 0 for agent_id in self.possible_agents}

        self.fig = None
        self.ax = None
        self.segment_patches: dict[str, list[Polygon]] = {}
        self.message_texts: dict[str, Any] = {}
        self.sensor_patches: dict[str, list[Wedge]] = {name: [] for name in self.CHANNEL_ORDER}
        self.sensor_legend_artists: list[Any] = []
        self.red_food_scatter = None
        self.blue_food_scatter = None
        self.joint_plots: dict[str, Any] = {}
        self.status_text = None
        self._border_artist = None
        self._render_background = None
        self._render_background_size: tuple[int, int] | None = None
        self.render_backend_name: str | None = None
        self.render_blit_bbox = None
        self.render_background_valid = False
        self.render_force_full_redraw = True
        self._render_suppress_draw_event_capture = False
        self._draw_event_cid: int | None = None
        self._resize_event_cid: int | None = None
        self._close_event_cid: int | None = None

    def _resolve_preset(self, fish_preset: FishPreset | str | None) -> FishPreset:
        if fish_preset is None:
            return EEL_3SEG_PRESET
        if isinstance(fish_preset, FishPreset):
            return fish_preset
        if fish_preset == "eel_3seg":
            return EEL_3SEG_PRESET
        raise ValueError(f"Unsupported fish preset: {fish_preset}")

    def _default_food_team_indices(self) -> np.ndarray:
        return np.concatenate(
            [
                np.full(self.num_red_pellets, RED_TEAM, dtype=np.int64),
                np.full(self.num_blue_pellets, BLUE_TEAM, dtype=np.int64),
            ]
        )

    def _normalized_low_level_snapshot(self, state: ArticulatedFishState) -> np.ndarray:
        dynamics = self.fish_preset.dynamics
        actuation = self.fish_preset.actuation
        theta = float(state.root_theta)
        body_velocity = _body_frame(state.root_velocity.astype(np.float32), theta)
        return np.array(
            [
                np.clip(body_velocity[0] / dynamics.max_speed, -1.0, 1.0),
                np.clip(body_velocity[1] / dynamics.max_speed, -1.0, 1.0),
                np.clip(state.root_omega / dynamics.max_angular_speed, -1.0, 1.0),
                np.clip(state.joint_angles[0] / actuation.joint_limit, -1.0, 1.0),
                np.clip(state.joint_angles[1] / actuation.joint_limit, -1.0, 1.0),
                np.clip(state.joint_velocities[0] / actuation.joint_max_speed, -1.0, 1.0),
                np.clip(state.joint_velocities[1] / actuation.joint_max_speed, -1.0, 1.0),
                np.clip(state.applied_joint_torque[0] / actuation.joint_torque_limit, -1.0, 1.0),
                np.clip(state.applied_joint_torque[1] / actuation.joint_torque_limit, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _initialize_control_history(self, agent_id: str) -> None:
        snapshot = self._normalized_low_level_snapshot(self.fish_states[agent_id])
        self.control_histories[agent_id] = np.broadcast_to(snapshot.reshape(1, -1), (self.history_length, snapshot.size)).copy()

    def _append_control_history(self, agent_id: str) -> None:
        snapshot = self._normalized_low_level_snapshot(self.fish_states[agent_id])
        history = self.control_histories[agent_id]
        history[:-1] = history[1:]
        history[-1] = snapshot

    def _joint_zero_crossings(self, previous: np.ndarray, current: np.ndarray) -> int:
        prev_sign = np.sign(previous)
        curr_sign = np.sign(current)
        crossings = (prev_sign != 0.0) & (curr_sign != 0.0) & (prev_sign != curr_sign)
        return int(np.count_nonzero(crossings))

    def get_agent_ids(self) -> tuple[str, ...]:
        return tuple(self.possible_agents)

    def get_agent_team_index(self, agent_id: str) -> int:
        return int(self.agent_team_index[agent_id])

    def get_agent_team_name(self, agent_id: str) -> str:
        return TEAM_NAMES[self.get_agent_team_index(agent_id)]

    def set_focus_agent(self, agent_id: str) -> None:
        if agent_id not in self._agent_ids:
            raise ValueError(f"Unknown focus agent: {agent_id}")
        self.focus_agent_id = agent_id

    def _empty_dynamics_debug(self) -> Dict[str, Any]:
        return {
            "desired_joint_torque": np.zeros(2, dtype=np.float32),
            "applied_joint_torque": np.zeros(2, dtype=np.float32),
            "joint_acceleration": np.zeros(2, dtype=np.float32),
            "joint_limit_ratio": np.zeros(2, dtype=np.float32),
            "segment_centers": np.zeros((3, 2), dtype=np.float32),
            "segment_angles": np.zeros(3, dtype=np.float32),
            "segment_velocities": np.zeros((3, 2), dtype=np.float32),
            "segment_angular_velocities": np.zeros(3, dtype=np.float32),
            "segment_drag_forces": np.zeros((3, 2), dtype=np.float32),
            "segment_drag_torques": np.zeros(3, dtype=np.float32),
            "joint_positions": np.zeros((2, 2), dtype=np.float32),
            "total_force": np.zeros(2, dtype=np.float32),
            "total_torque": 0.0,
            "root_acceleration": np.zeros(2, dtype=np.float32),
            "root_angular_acceleration": 0.0,
            "mean_abs_applied_torque": 0.0,
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
        fish_positions: np.ndarray,
        *,
        exclude_index: int | None = None,
        require_spacing: bool = True,
    ) -> bool:
        if fish_positions.size:
            if float(np.min(np.linalg.norm(fish_positions - candidate, axis=1))) < self.food_min_spawn_distance:
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
        fish_positions: np.ndarray,
        *,
        exclude_index: int | None = None,
    ) -> np.ndarray:
        fallback_with_clearance = None
        last_candidate = None
        for _ in range(64):
            candidate = self._sample_food_candidate()
            last_candidate = candidate
            if fish_positions.size == 0 or float(np.min(np.linalg.norm(fish_positions - candidate, axis=1))) >= self.food_min_spawn_distance:
                fallback_with_clearance = candidate
            if self._is_food_position_valid(
                candidate,
                existing_positions,
                fish_positions,
                exclude_index=exclude_index,
                require_spacing=True,
            ):
                return candidate
        if fallback_with_clearance is not None:
            return fallback_with_clearance.astype(np.float32)
        if last_candidate is not None:
            return last_candidate.astype(np.float32)
        return self._sample_food_candidate()

    def _spawn_food_field(self) -> None:
        positions = np.zeros((self.food_count, 2), dtype=np.float32)
        fish_positions = self._all_root_positions()
        for idx in range(self.food_count):
            positions[idx] = self._sample_food_position(positions[:idx], fish_positions)
        self.food_positions = positions
        self.food_team_indices = self._default_food_team_indices()

    def _respawn_food_indices(self, indices: np.ndarray) -> None:
        fish_positions = self._all_root_positions()
        for idx in np.flatnonzero(indices):
            self.food_positions[idx] = self._sample_food_position(
                self.food_positions,
                fish_positions,
                exclude_index=int(idx),
            )

    def _all_root_positions(self) -> np.ndarray:
        return np.asarray(
            [self.fish_states[agent_id].root_position for agent_id in self.possible_agents],
            dtype=np.float32,
        )

    def _food_relative_vectors(self, agent_id: str, *, edible: bool) -> np.ndarray:
        team_index = self.get_agent_team_index(agent_id)
        mask = self.food_team_indices == team_index if edible else self.food_team_indices != team_index
        positions = self.food_positions[mask]
        if positions.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return positions - self.fish_states[agent_id].root_position

    def _peer_relative_vectors(self, agent_id: str, *, teammate: bool) -> np.ndarray:
        origin = self.fish_states[agent_id].root_position
        team_index = self.get_agent_team_index(agent_id)
        peers = []
        for other_id in self.possible_agents:
            if other_id == agent_id:
                continue
            same_team = self.get_agent_team_index(other_id) == team_index
            if same_team != teammate:
                continue
            peers.append(self.fish_states[other_id].root_position - origin)
        if not peers:
            return np.zeros((0, 2), dtype=np.float32)
        return np.asarray(peers, dtype=np.float32)

    def _message_relative_vectors(self, agent_id: str, *, teammate: bool) -> tuple[np.ndarray, np.ndarray]:
        origin = self.fish_states[agent_id].root_position
        team_index = self.get_agent_team_index(agent_id)
        vectors = []
        tokens = []
        for other_id in self.possible_agents:
            if other_id == agent_id:
                continue
            same_team = self.get_agent_team_index(other_id) == team_index
            if same_team != teammate:
                continue
            vectors.append(self.fish_states[other_id].root_position - origin)
            tokens.append(self.last_message_tokens[other_id] / max(self.num_message_tokens - 1, 1))
        if not vectors:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)
        return np.asarray(vectors, dtype=np.float32), np.asarray(tokens, dtype=np.float32)

    def _nearest_edible_food_distance(self, agent_id: str) -> float:
        relative = self._food_relative_vectors(agent_id, edible=True)
        if relative.size == 0:
            return float("nan")
        return float(np.min(np.linalg.norm(relative, axis=1)))

    def _sector_indices_from_body_coords(self, body_x: np.ndarray, body_y: np.ndarray) -> np.ndarray:
        sector_size = 2.0 * math.pi / float(self.sector_num)
        angles = np.arctan2(body_y, body_x)
        sector_indices = np.floor(((angles + (0.5 * sector_size)) % (2.0 * math.pi)) / sector_size).astype(np.int64)
        return np.clip(sector_indices, 0, self.sector_num - 1)

    def _aggregate_sector_counts(
        self,
        sector_indices: np.ndarray,
        active_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
        num_rows = int(sector_indices.shape[0])
        bins = np.zeros((num_rows, self.sector_num), dtype=np.float32)
        visible_counts = np.zeros(num_rows, dtype=np.int32)
        active_bins: list[list[int]] = [[] for _ in range(num_rows)]
        for row_idx in range(num_rows):
            row_mask = np.asarray(active_mask[row_idx], dtype=bool)
            visible_counts[row_idx] = int(np.count_nonzero(row_mask))
            if not np.any(row_mask):
                continue
            counts = np.bincount(np.asarray(sector_indices[row_idx, row_mask], dtype=np.int64), minlength=self.sector_num).astype(np.float32)
            row_bins = np.minimum(counts, 2.0) / 2.0
            bins[row_idx] = row_bins
            active_bins[row_idx] = np.flatnonzero(row_bins > 0.0).astype(int).tolist()
        return bins, visible_counts, active_bins

    def _aggregate_message_means(
        self,
        sector_indices: np.ndarray,
        active_mask: np.ndarray,
        token_values: np.ndarray,
    ) -> tuple[np.ndarray, list[list[int]]]:
        num_rows = int(sector_indices.shape[0])
        bins = np.zeros((num_rows, self.sector_num), dtype=np.float32)
        active_bins: list[list[int]] = [[] for _ in range(num_rows)]
        for row_idx in range(num_rows):
            row_mask = np.asarray(active_mask[row_idx], dtype=bool)
            if not np.any(row_mask):
                continue
            row_indices = np.asarray(sector_indices[row_idx, row_mask], dtype=np.int64)
            row_tokens = np.asarray(token_values[row_idx, row_mask], dtype=np.float32)
            totals = np.bincount(row_indices, weights=row_tokens, minlength=self.sector_num).astype(np.float32)
            counts = np.bincount(row_indices, minlength=self.sector_num).astype(np.float32)
            row_bins = np.divide(
                totals,
                np.maximum(counts, 1.0),
                out=np.zeros(self.sector_num, dtype=np.float32),
                where=counts > 0.0,
            )
            bins[row_idx] = row_bins
            active_bins[row_idx] = np.flatnonzero(row_bins > 0.0).astype(int).tolist()
        return bins, active_bins

    def _compute_observation_bundle(self, agent_ids: list[str] | tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
        if agent_ids is None:
            agent_ids = list(self.possible_agents)
        else:
            agent_ids = list(agent_ids)
        if not agent_ids:
            return {}

        agent_indices = np.asarray([self.agent_index[agent_id] for agent_id in agent_ids], dtype=np.int64)
        positions_all = self._all_root_positions()
        theta_all = np.asarray([self.fish_states[agent_id].root_theta for agent_id in self.possible_agents], dtype=np.float32)
        omega_all = np.asarray([self.fish_states[agent_id].root_omega for agent_id in self.possible_agents], dtype=np.float32)
        velocity_all = np.asarray([self.fish_states[agent_id].root_velocity for agent_id in self.possible_agents], dtype=np.float32)
        joint_angles_all = np.asarray([self.fish_states[agent_id].joint_angles for agent_id in self.possible_agents], dtype=np.float32)
        joint_velocities_all = np.asarray([self.fish_states[agent_id].joint_velocities for agent_id in self.possible_agents], dtype=np.float32)
        prev_message_tokens_all = np.asarray(
            [self.fish_states[agent_id].prev_message_token for agent_id in self.possible_agents],
            dtype=np.float32,
        )
        team_indices_all = np.asarray([self.agent_team_index[agent_id] for agent_id in self.possible_agents], dtype=np.int64)
        message_token_values = np.asarray(
            [self.last_message_tokens[agent_id] for agent_id in self.possible_agents],
            dtype=np.float32,
        ) / max(self.num_message_tokens - 1, 1)

        positions = positions_all[agent_indices]
        theta = theta_all[agent_indices]
        cos_theta = np.cos(theta).astype(np.float32)
        sin_theta = np.sin(theta).astype(np.float32)
        teams = team_indices_all[agent_indices]

        if self.food_count > 0:
            food_relative = self.food_positions[None, :, :] - positions[:, None, :]
            food_body_x = (cos_theta[:, None] * food_relative[:, :, 0]) + (sin_theta[:, None] * food_relative[:, :, 1])
            food_body_y = (-sin_theta[:, None] * food_relative[:, :, 0]) + (cos_theta[:, None] * food_relative[:, :, 1])
            food_distance_sq = (food_body_x * food_body_x) + (food_body_y * food_body_y)
            food_distance = np.sqrt(food_distance_sq, out=np.zeros_like(food_distance_sq))
            food_sector_indices = self._sector_indices_from_body_coords(food_body_x, food_body_y)
            food_visible_mask = food_distance <= float(self.sector_radius)
            edible_food_mask = self.food_team_indices[None, :] == teams[:, None]
            non_edible_food_mask = np.logical_not(edible_food_mask)
            edible_food_bins, visible_edible_food_count, edible_food_active = self._aggregate_sector_counts(
                food_sector_indices,
                food_visible_mask & edible_food_mask,
            )
            non_edible_food_bins, visible_non_edible_food_count, non_edible_food_active = self._aggregate_sector_counts(
                food_sector_indices,
                food_visible_mask & non_edible_food_mask,
            )
            edible_food_distance = np.where(edible_food_mask, food_distance, np.inf)
            nearest_edible_food_distance = np.min(edible_food_distance, axis=1)
            nearest_edible_food_distance = np.where(
                np.isfinite(nearest_edible_food_distance),
                nearest_edible_food_distance,
                np.nan,
            ).astype(np.float32)
        else:
            edible_food_bins = np.zeros((len(agent_ids), self.sector_num), dtype=np.float32)
            non_edible_food_bins = np.zeros((len(agent_ids), self.sector_num), dtype=np.float32)
            visible_edible_food_count = np.zeros(len(agent_ids), dtype=np.int32)
            visible_non_edible_food_count = np.zeros(len(agent_ids), dtype=np.int32)
            edible_food_active = [[] for _ in agent_ids]
            non_edible_food_active = [[] for _ in agent_ids]
            nearest_edible_food_distance = np.full(len(agent_ids), np.nan, dtype=np.float32)

        fish_relative = positions_all[None, :, :] - positions[:, None, :]
        fish_body_x = (cos_theta[:, None] * fish_relative[:, :, 0]) + (sin_theta[:, None] * fish_relative[:, :, 1])
        fish_body_y = (-sin_theta[:, None] * fish_relative[:, :, 0]) + (cos_theta[:, None] * fish_relative[:, :, 1])
        fish_distance_sq = (fish_body_x * fish_body_x) + (fish_body_y * fish_body_y)
        fish_distance = np.sqrt(fish_distance_sq, out=np.zeros_like(fish_distance_sq))
        fish_sector_indices = self._sector_indices_from_body_coords(fish_body_x, fish_body_y)
        fish_visible_mask = fish_distance <= float(self.sector_radius)
        message_visible_mask = fish_distance <= float(self.communication_radius)
        same_team_mask = teams[:, None] == team_indices_all[None, :]
        self_mask = np.ones((len(agent_ids), self.num_fish), dtype=bool)
        self_mask[np.arange(len(agent_ids)), agent_indices] = False
        teammate_mask = same_team_mask & self_mask
        opponent_mask = np.logical_not(same_team_mask) & self_mask
        teammate_bins, visible_teammate_count, teammate_active = self._aggregate_sector_counts(
            fish_sector_indices,
            fish_visible_mask & teammate_mask,
        )
        opponent_bins, visible_opponent_count, opponent_active = self._aggregate_sector_counts(
            fish_sector_indices,
            fish_visible_mask & opponent_mask,
        )

        if self.mute_received_messages:
            teammate_message_bins = np.zeros((len(agent_ids), self.sector_num), dtype=np.float32)
            opponent_message_bins = np.zeros((len(agent_ids), self.sector_num), dtype=np.float32)
            teammate_message_active = [[] for _ in agent_ids]
            opponent_message_active = [[] for _ in agent_ids]
        else:
            token_matrix = np.broadcast_to(message_token_values[None, :], fish_sector_indices.shape)
            teammate_message_bins, teammate_message_active = self._aggregate_message_means(
                fish_sector_indices,
                message_visible_mask & teammate_mask,
                token_matrix,
            )
            opponent_message_bins, opponent_message_active = self._aggregate_message_means(
                fish_sector_indices,
                message_visible_mask & opponent_mask,
                token_matrix,
            )

        selected_velocity = velocity_all[agent_indices]
        root_velocity_body = np.stack(
            [
                (cos_theta * selected_velocity[:, 0]) + (sin_theta * selected_velocity[:, 1]),
                (-sin_theta * selected_velocity[:, 0]) + (cos_theta * selected_velocity[:, 1]),
            ],
            axis=1,
        ).astype(np.float32)
        selected_joint_angles = joint_angles_all[agent_indices]
        selected_joint_velocities = joint_velocities_all[agent_indices]
        selected_prev_message_tokens = prev_message_tokens_all[agent_indices]
        observations: dict[str, np.ndarray] = {}
        for row_idx, agent_id in enumerate(agent_ids):
            self.last_observation_channels[agent_id] = {
                "edible_food": edible_food_bins[row_idx],
                "non_edible_food": non_edible_food_bins[row_idx],
                "teammate": teammate_bins[row_idx],
                "opponent": opponent_bins[row_idx],
                "teammate_message": teammate_message_bins[row_idx],
                "opponent_message": opponent_message_bins[row_idx],
            }
            self.last_visible_counts[agent_id] = {
                "edible_food": int(visible_edible_food_count[row_idx]),
                "non_edible_food": int(visible_non_edible_food_count[row_idx]),
                "teammate": int(visible_teammate_count[row_idx]),
                "opponent": int(visible_opponent_count[row_idx]),
            }
            self.last_sensor_active_bins[agent_id] = {
                "edible_food": list(edible_food_active[row_idx]),
                "non_edible_food": list(non_edible_food_active[row_idx]),
                "teammate": list(teammate_active[row_idx]),
                "opponent": list(opponent_active[row_idx]),
                "teammate_message": list(teammate_message_active[row_idx]),
                "opponent_message": list(opponent_message_active[row_idx]),
            }
            self.last_nearest_food_distance[agent_id] = float(nearest_edible_food_distance[row_idx])
            context = np.array(
                [
                    float(teams[row_idx]),
                    float(selected_prev_message_tokens[row_idx]) / max(self.num_message_tokens - 1, 1),
                    np.clip(self.timestep / self.time_limit, 0.0, 1.0),
                ],
                dtype=np.float32,
            )
            observations[agent_id] = np.concatenate(
                [
                    edible_food_bins[row_idx],
                    non_edible_food_bins[row_idx],
                    teammate_bins[row_idx],
                    opponent_bins[row_idx],
                    teammate_message_bins[row_idx],
                    opponent_message_bins[row_idx],
                    context,
                    self.control_histories[agent_id].reshape(-1),
                ]
            ).astype(np.float32, copy=False)
        return observations

    def _sector_index(self, vector: np.ndarray) -> int:
        sector_size = 2.0 * math.pi / float(self.sector_num)
        angle = math.atan2(float(vector[1]), float(vector[0]))
        sector_index = int(math.floor(((angle + 0.5 * sector_size) % (2.0 * math.pi)) / sector_size))
        return min(max(sector_index, 0), self.sector_num - 1)

    def _sector_counts(
        self,
        relative_world: np.ndarray,
        *,
        theta: float,
        radius: float,
    ) -> tuple[np.ndarray, int, list[int]]:
        if relative_world.size == 0:
            return np.zeros(self.sector_num, dtype=np.float32), 0, []
        relative_body = np.asarray([_body_frame(vector, theta) for vector in relative_world], dtype=np.float32)
        distances = np.linalg.norm(relative_body, axis=1)
        visible_mask = distances <= radius
        counts = np.zeros(self.sector_num, dtype=np.float32)
        for vector in relative_body[visible_mask]:
            counts[self._sector_index(vector)] += 1.0
        normalized = np.minimum(counts, 2.0) / 2.0
        active_bins = np.flatnonzero(normalized > 0.0).astype(int).tolist()
        return normalized.astype(np.float32), int(np.count_nonzero(visible_mask)), active_bins

    def _message_sectors(
        self,
        relative_world: np.ndarray,
        token_values: np.ndarray,
        *,
        theta: float,
        radius: float,
    ) -> tuple[np.ndarray, list[int]]:
        if self.mute_received_messages or relative_world.size == 0:
            return np.zeros(self.sector_num, dtype=np.float32), []
        relative_body = np.asarray([_body_frame(vector, theta) for vector in relative_world], dtype=np.float32)
        distances = np.linalg.norm(relative_body, axis=1)
        visible_mask = distances <= radius
        totals = np.zeros(self.sector_num, dtype=np.float32)
        counts = np.zeros(self.sector_num, dtype=np.float32)
        for vector, token_value in zip(relative_body[visible_mask], token_values[visible_mask], strict=False):
            idx = self._sector_index(vector)
            totals[idx] += float(np.clip(token_value, 0.0, 1.0))
            counts[idx] += 1.0
        means = np.divide(
            totals,
            np.maximum(counts, 1.0),
            out=np.zeros_like(totals),
            where=counts > 0.0,
        )
        active_bins = np.flatnonzero(means > 0.0).astype(int).tolist()
        return means.astype(np.float32), active_bins

    def _segment_angles(self, state: ArticulatedFishState) -> np.ndarray:
        q0, q1 = state.joint_angles.astype(np.float32)
        return np.array(
            [state.root_theta, state.root_theta + q0, state.root_theta + q0 + q1],
            dtype=np.float32,
        )

    def _segment_geometry(self, state: ArticulatedFishState) -> Dict[str, np.ndarray]:
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

        joint0_velocity = segment_velocities[0] - 0.5 * lengths[0] * float(omega_segments[0]) * normals[0]
        segment_velocities[1] = joint0_velocity - 0.5 * lengths[1] * float(omega_segments[1]) * normals[1]
        joint1_velocity = joint0_velocity - lengths[1] * float(omega_segments[1]) * normals[1]
        segment_velocities[2] = joint1_velocity - 0.5 * lengths[2] * float(omega_segments[2]) * normals[2]

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

    def _compute_external_wrench(self, state: ArticulatedFishState, *, include_debug: bool = True) -> Dict[str, Any]:
        morphology = self.fish_preset.morphology
        dynamics = self.fish_preset.dynamics
        kinematics = self._compute_segment_kinematics(state)
        centers = np.asarray(kinematics["centers"], dtype=np.float32)
        angles = np.asarray(kinematics["angles"], dtype=np.float32)
        segment_velocities = np.asarray(kinematics["segment_velocities"], dtype=np.float32)
        segment_angular_velocities = np.asarray(kinematics["segment_angular_velocities"], dtype=np.float32)

        segment_drag_forces = np.zeros((3, 2), dtype=np.float32) if include_debug else None
        segment_drag_torques = np.zeros(3, dtype=np.float32) if include_debug else None
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
            if include_debug:
                segment_drag_forces[idx] = force
                segment_drag_torques[idx] = torque_drag
            total_force += force
            total_torque += _cross2d(centers[idx] - state.root_position, force) + torque_drag

        total_torque += float(-dynamics.root_rotational_drag * state.root_omega)
        if not include_debug:
            return {
                "total_force": total_force.astype(np.float32),
                "total_torque": float(total_torque),
            }
        return {
            "segment_centers": centers,
            "segment_angles": angles,
            "segment_velocities": segment_velocities,
            "segment_angular_velocities": segment_angular_velocities,
            "segment_drag_forces": segment_drag_forces.astype(np.float32),
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
        *,
        include_debug: bool = True,
    ) -> Dict[str, np.ndarray | float]:
        actuation = self.fish_preset.actuation
        desired_torque = (
            np.clip(_as_float32_array(action, shape=(2,)), -1.0, 1.0) * float(actuation.joint_torque_limit)
        ).astype(np.float32)
        alpha = float(dt / (self.actuator_time_constant + dt)) if self.actuator_time_constant > 0.0 else 1.0
        applied_torque_next = (
            state.applied_joint_torque + alpha * (desired_torque - state.applied_joint_torque)
        ).astype(np.float32)
        joint_acceleration = (
            applied_torque_next - actuation.joint_passive_damping * state.joint_velocities
        ) / float(actuation.joint_inertia)
        joint_velocity_next = self._clip_joint_velocities(
            state.joint_velocities + joint_acceleration.astype(np.float32) * float(dt)
        )
        joint_angle_next, joint_velocity_next = self._clamp_joint_state(
            state.joint_angles + joint_velocity_next * float(dt),
            joint_velocity_next,
        )
        result = {
            "desired_joint_torque": desired_torque.astype(np.float32),
            "applied_joint_torque": applied_torque_next.astype(np.float32),
            "joint_velocity_next": joint_velocity_next.astype(np.float32),
            "joint_angle_next": joint_angle_next.astype(np.float32),
            "joint_acceleration": joint_acceleration.astype(np.float32),
            "joint_limit_ratio": (
                np.abs(joint_angle_next) / max(float(actuation.joint_limit), 1e-6)
            ).astype(np.float32),
        }
        if not include_debug:
            return result
        return result

    def _compute_substep_dynamics(
        self,
        state: ArticulatedFishState,
        action: np.ndarray,
        dt: float,
    ) -> Dict[str, np.ndarray | float]:
        actuation = self._compute_actuation(state, action, dt, include_debug=False)
        predicted_state = ArticulatedFishState(
            root_position=state.root_position,
            root_velocity=state.root_velocity,
            root_theta=state.root_theta,
            root_omega=state.root_omega,
            joint_angles=np.asarray(actuation["joint_angle_next"], dtype=np.float32),
            joint_velocities=np.asarray(actuation["joint_velocity_next"], dtype=np.float32),
            prev_action=state.prev_action,
            prev_message_token=state.prev_message_token,
            applied_joint_torque=np.asarray(actuation["applied_joint_torque"], dtype=np.float32),
        )
        wrench = self._compute_external_wrench(predicted_state, include_debug=False)
        dynamics = self.fish_preset.dynamics
        return {
            "desired_joint_torque": np.asarray(actuation["desired_joint_torque"], dtype=np.float32),
            "applied_joint_torque": np.asarray(actuation["applied_joint_torque"], dtype=np.float32),
            "joint_velocity_next": np.asarray(actuation["joint_velocity_next"], dtype=np.float32),
            "joint_angle_next": np.asarray(actuation["joint_angle_next"], dtype=np.float32),
            "joint_acceleration": np.asarray(actuation["joint_acceleration"], dtype=np.float32),
            "joint_limit_ratio": np.asarray(actuation["joint_limit_ratio"], dtype=np.float32),
            "root_acceleration": (np.asarray(wrench["total_force"], dtype=np.float32) / float(dynamics.mass)).astype(np.float32),
            "root_angular_acceleration": float(float(wrench["total_torque"]) / float(dynamics.inertia)),
        }

    def get_dynamics_breakdown(
        self,
        agent_id: str,
        action,
        *,
        state: ArticulatedFishState | None = None,
        dt: float | None = None,
    ) -> Dict[str, Any]:
        state = state or self.fish_states[agent_id]
        action = np.clip(_as_float32_array(action, shape=(2,)), -1.0, 1.0)
        dt = float(self.fish_preset.dynamics.dt if dt is None else dt)
        actuation = self._compute_actuation(state, action, dt, include_debug=True)
        predicted_state = ArticulatedFishState(
            root_position=state.root_position.astype(np.float32).copy(),
            root_velocity=state.root_velocity.astype(np.float32).copy(),
            root_theta=float(state.root_theta),
            root_omega=float(state.root_omega),
            joint_angles=actuation["joint_angle_next"].astype(np.float32).copy(),
            joint_velocities=actuation["joint_velocity_next"].astype(np.float32).copy(),
            prev_action=action.astype(np.float32).copy(),
            prev_message_token=int(state.prev_message_token),
            applied_joint_torque=actuation["applied_joint_torque"].astype(np.float32).copy(),
        )
        wrench = self._compute_external_wrench(predicted_state, include_debug=True)
        dynamics = self.fish_preset.dynamics
        total_force = wrench["total_force"].astype(np.float32)
        total_torque = float(wrench["total_torque"])
        root_acceleration = total_force / float(dynamics.mass)
        root_angular_acceleration = float(total_torque / float(dynamics.inertia))
        return {
            "desired_joint_torque": actuation["desired_joint_torque"].astype(np.float32),
            "applied_joint_torque": actuation["applied_joint_torque"].astype(np.float32),
            "joint_acceleration": actuation["joint_acceleration"].astype(np.float32),
            "joint_limit_ratio": actuation["joint_limit_ratio"].astype(np.float32),
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
            "root_acceleration": root_acceleration.astype(np.float32),
            "root_angular_acceleration": float(root_angular_acceleration),
            "mean_abs_applied_torque": float(
                np.mean(np.abs(actuation["applied_joint_torque"])) / max(self.fish_preset.actuation.joint_torque_limit, 1e-6)
            ),
        }

    def _integrate_substep(
        self,
        agent_id: str,
        state: ArticulatedFishState,
        action: np.ndarray,
        dt: float,
        prev_message_token: int,
    ) -> ArticulatedFishState:
        dynamics = self.fish_preset.dynamics
        dynamics_step = self._compute_substep_dynamics(state, action, dt)
        next_root_velocity = self._clip_root_velocity(
            state.root_velocity + np.asarray(dynamics_step["root_acceleration"], dtype=np.float32) * dt
        )
        next_root_omega = float(
            np.clip(
                state.root_omega + float(dynamics_step["root_angular_acceleration"]) * dt,
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
            joint_angles=np.asarray(dynamics_step["joint_angle_next"], dtype=np.float32).copy(),
            joint_velocities=np.asarray(dynamics_step["joint_velocity_next"], dtype=np.float32).copy(),
            prev_action=np.clip(_as_float32_array(action, shape=(2,)), -1.0, 1.0).astype(np.float32),
            prev_message_token=int(prev_message_token),
            applied_joint_torque=np.asarray(dynamics_step["applied_joint_torque"], dtype=np.float32).copy(),
        )
        return next_state

    def _compute_next_state(
        self,
        agent_id: str,
        motion_action: np.ndarray,
        message_token: int,
    ) -> ArticulatedFishState:
        dynamics = self.fish_preset.dynamics
        total_dt = float(dynamics.dt)
        max_substep_dt = max(1e-4, min(float(dynamics.max_integration_dt), total_dt))
        substeps = max(1, int(math.ceil(total_dt / max_substep_dt)))
        substep_dt = total_dt / substeps
        next_state = ArticulatedFishState(
            root_position=self.fish_states[agent_id].root_position.astype(np.float32).copy(),
            root_velocity=self.fish_states[agent_id].root_velocity.astype(np.float32).copy(),
            root_theta=float(self.fish_states[agent_id].root_theta),
            root_omega=float(self.fish_states[agent_id].root_omega),
            joint_angles=self.fish_states[agent_id].joint_angles.astype(np.float32).copy(),
            joint_velocities=self.fish_states[agent_id].joint_velocities.astype(np.float32).copy(),
            prev_action=np.clip(_as_float32_array(motion_action, shape=(2,)), -1.0, 1.0).astype(np.float32),
            prev_message_token=int(message_token),
            applied_joint_torque=self.fish_states[agent_id].applied_joint_torque.astype(np.float32).copy(),
        )
        for _ in range(substeps):
            next_state = self._integrate_substep(
                agent_id,
                next_state,
                next_state.prev_action,
                substep_dt,
                message_token,
            )
        return next_state

    def get_reward_breakdown(
        self,
        *,
        agent_id: str,
        food_eaten_this_step: int,
        truncated: bool,
    ) -> Dict[str, float | bool | int | str]:
        metrics = self._current_motion_metrics(agent_id)
        pellet_reward_total = float(self.pellet_reward * food_eaten_this_step) if self.reward_mode == "forage" else 0.0
        if self.reward_mode == "forage":
            total_reward = float(pellet_reward_total - self.step_cost)
        else:
            forward_term = 0.70 * max(float(metrics["forward_velocity_norm"]), 0.0)
            lateral_term = -0.15 * abs(float(metrics["lateral_velocity_norm"]))
            angular_term = -0.15 * abs(float(metrics["angular_velocity_norm"]))
            torque_term = -0.01 * float(metrics["mean_abs_applied_torque_norm"])
            joint_limit_term = -0.02 * float(metrics["mean_joint_limit_ratio_sq"])
            total_reward = float(forward_term + lateral_term + angular_term + torque_term + joint_limit_term)
        return {
            "agent_id": agent_id,
            "agent_team": self.get_agent_team_name(agent_id),
            "reward_mode": self.reward_mode,
            "food_eaten_this_step": int(food_eaten_this_step),
            "pellet_reward_total": pellet_reward_total,
            "step_cost": float(self.step_cost if self.reward_mode == "forage" else 0.0),
            "forward_velocity": float(metrics["forward_velocity"]),
            "lateral_velocity": float(metrics["lateral_velocity"]),
            "angular_velocity": float(metrics["angular_velocity"]),
            "forward_velocity_norm": float(metrics["forward_velocity_norm"]),
            "lateral_velocity_norm": float(metrics["lateral_velocity_norm"]),
            "angular_velocity_norm": float(metrics["angular_velocity_norm"]),
            "mean_abs_applied_torque": float(metrics["mean_abs_applied_torque"]),
            "mean_abs_applied_torque_norm": float(metrics["mean_abs_applied_torque_norm"]),
            "mean_joint_limit_ratio": float(metrics["mean_joint_limit_ratio"]),
            "mean_joint_limit_ratio_sq": float(metrics["mean_joint_limit_ratio_sq"]),
            "total_reward": total_reward,
            "terminated": False,
            "truncated": bool(truncated),
        }

    def _current_motion_metrics(self, agent_id: str) -> dict[str, float]:
        state = self.fish_states[agent_id]
        dynamics = self.fish_preset.dynamics
        actuation = self.fish_preset.actuation
        body_velocity = _body_frame(state.root_velocity.astype(np.float32), float(state.root_theta))
        joint_limit_ratio = np.abs(state.joint_angles.astype(np.float32)) / max(float(actuation.joint_limit), 1e-6)
        return {
            "forward_velocity": float(body_velocity[0]),
            "lateral_velocity": float(body_velocity[1]),
            "angular_velocity": float(state.root_omega),
            "forward_velocity_norm": float(np.clip(body_velocity[0] / dynamics.max_speed, -1.0, 1.0)),
            "lateral_velocity_norm": float(np.clip(body_velocity[1] / dynamics.max_speed, -1.0, 1.0)),
            "angular_velocity_norm": float(np.clip(state.root_omega / dynamics.max_angular_speed, -1.0, 1.0)),
            "mean_abs_applied_torque": float(np.mean(np.abs(state.applied_joint_torque))),
            "mean_abs_applied_torque_norm": float(
                np.mean(np.abs(state.applied_joint_torque)) / max(float(actuation.joint_torque_limit), 1e-6)
            ),
            "mean_joint_limit_ratio": float(np.mean(joint_limit_ratio)),
            "mean_joint_limit_ratio_sq": float(np.mean(np.square(joint_limit_ratio))),
        }

    def _spawn_school_states(self) -> None:
        angle_offset = float(self.np_random.uniform(-math.pi, math.pi))
        sector_angle = math.pi / float(max(self.num_fish, 2))
        min_required_radius = float(self.min_spawn_separation / max(2.0 * math.sin(sector_angle), 1e-6))
        spawn_radius = float(max(self.spawn_ring_radius, min_required_radius))
        self.fish_states = {}
        self.last_message_tokens = {agent_id: 0 for agent_id in self.possible_agents}
        for idx, agent_id in enumerate(self.possible_agents):
            angle = angle_offset + ((2.0 * math.pi * idx) / float(self.num_fish))
            position = np.array(
                [spawn_radius * math.cos(angle), spawn_radius * math.sin(angle)],
                dtype=np.float32,
            )
            theta = float(self.np_random.uniform(-math.pi, math.pi))
            self.fish_states[agent_id] = ArticulatedFishState(
                root_position=position,
                root_velocity=np.zeros(2, dtype=np.float32),
                root_theta=theta,
                root_omega=0.0,
                joint_angles=np.zeros(2, dtype=np.float32),
                joint_velocities=np.zeros(2, dtype=np.float32),
                prev_action=np.zeros(2, dtype=np.float32),
                prev_message_token=0,
                applied_joint_torque=np.zeros(2, dtype=np.float32),
            )
            self._initialize_control_history(agent_id)
            self.last_capture_distance[agent_id] = float("nan")
            self.last_joint_limit_occupancy[agent_id] = 0.0
            self.last_joint_zero_crossings[agent_id] = 0

    def _compute_agent_obs(self, agent_id: str) -> np.ndarray:
        return self._compute_observation_bundle([agent_id])[agent_id]

    def _get_obs_dict(self) -> dict[str, np.ndarray]:
        return self._compute_observation_bundle(self.possible_agents)

    def _build_info_dict(
        self,
        *,
        truncated: bool,
        food_eaten_by_agent: dict[str, int] | None = None,
    ) -> dict[str, dict[str, Any]]:
        food_eaten_by_agent = food_eaten_by_agent or {agent_id: 0 for agent_id in self.possible_agents}
        infos: dict[str, dict[str, Any]] = {}
        for agent_id in self.possible_agents:
            visible_counts = self.last_visible_counts[agent_id]
            metrics = self._current_motion_metrics(agent_id)
            infos[agent_id] = {
                "agent_id": agent_id,
                "agent_team": self.get_agent_team_name(agent_id),
                "reward_mode": self.reward_mode,
                "focus_agent_id": self.focus_agent_id,
                "nearest_food_distance": float(self.last_nearest_food_distance[agent_id]),
                "capture_distance_this_step": float(self.last_capture_distance[agent_id]),
                "reward_breakdown": dict(self.last_reward_breakdown[agent_id]),
                "food_eaten_this_step": int(food_eaten_by_agent.get(agent_id, 0)),
                "food_eaten_episode": int(self.agent_food_eaten_episode[agent_id]),
                "food_eaten_episode_red": int(self.team_food_eaten_episode[RED_TEAM]),
                "food_eaten_episode_blue": int(self.team_food_eaten_episode[BLUE_TEAM]),
                "visible_food_count": int(visible_counts["edible_food"] + visible_counts["non_edible_food"]),
                "visible_edible_food_count": int(visible_counts["edible_food"]),
                "visible_non_edible_food_count": int(visible_counts["non_edible_food"]),
                "visible_teammate_count": int(visible_counts["teammate"]),
                "visible_opponent_count": int(visible_counts["opponent"]),
                "sensor_active_bins": {key: list(value) for key, value in self.last_sensor_active_bins[agent_id].items()},
                "emitted_message_token": int(self.last_message_tokens[agent_id]),
                "forward_velocity": float(metrics["forward_velocity"]),
                "lateral_velocity": float(metrics["lateral_velocity"]),
                "angular_velocity": float(metrics["angular_velocity"]),
                "forward_velocity_norm": float(metrics["forward_velocity_norm"]),
                "lateral_velocity_norm": float(metrics["lateral_velocity_norm"]),
                "angular_velocity_norm": float(metrics["angular_velocity_norm"]),
                "mean_abs_applied_torque": float(metrics["mean_abs_applied_torque"]),
                "mean_abs_applied_torque_norm": float(metrics["mean_abs_applied_torque_norm"]),
                "mean_joint_limit_ratio": float(metrics["mean_joint_limit_ratio"]),
                "joint_velocity_zero_crossings": int(self.last_joint_zero_crossings[agent_id]),
                "truncated": bool(truncated),
                "fish_preset": self.fish_preset.name,
            }
        return infos

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        self.agents = list(self.possible_agents)
        self.timestep = 0
        self.agent_food_eaten_episode = {agent_id: 0 for agent_id in self.possible_agents}
        self.team_food_eaten_episode = {RED_TEAM: 0, BLUE_TEAM: 0}
        self._spawn_school_states()
        self._spawn_food_field()
        self.last_rewards = {agent_id: 0.0 for agent_id in self.possible_agents}
        self.last_reward_breakdown = {
            agent_id: self.get_reward_breakdown(agent_id=agent_id, food_eaten_this_step=0, truncated=False)
            for agent_id in self.possible_agents
        }
        self.last_dynamics_debug = {agent_id: self._empty_dynamics_debug() for agent_id in self.possible_agents}
        obs = self._get_obs_dict()
        infos = self._build_info_dict(truncated=False)
        return obs, infos

    def _normalize_action(self, action: Any) -> dict[str, Any]:
        motion = np.zeros(2, dtype=np.float32)
        message = 0
        if isinstance(action, dict):
            motion = np.clip(_as_float32_array(action.get("motion", [0.0, 0.0]), shape=(2,)), -1.0, 1.0)
            message = _normalize_message_token(action.get("message", 0), num_tokens=self.num_message_tokens)
        else:
            array = np.asarray(action, dtype=np.float32).reshape(-1)
            if array.size >= 2:
                motion = np.clip(array[:2], -1.0, 1.0).astype(np.float32)
            if array.size >= 3:
                message = _normalize_message_token(array[2], num_tokens=self.num_message_tokens)
        if self.eps > 0.0:
            if bool(self.np_random.random() < self.eps):
                motion = self.np_random.uniform(-1.0, 1.0, size=2).astype(np.float32)
            if bool(self.np_random.random() < self.eps):
                message = int(self.np_random.integers(0, self.num_message_tokens))
        return {"motion": motion.astype(np.float32), "message": int(message)}

    def _resolve_step_actions(self, action_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
        resolved: dict[str, dict[str, Any]] = {}
        for agent_id in self.possible_agents:
            resolved[agent_id] = self._normalize_action(
                action_dict.get(agent_id, {"motion": np.zeros(2, dtype=np.float32), "message": 0})
            )
        return resolved

    def _food_capture_assignments(self) -> tuple[dict[str, int], np.ndarray, dict[str, float]]:
        if self.food_count == 0:
            return {agent_id: 0 for agent_id in self.possible_agents}, np.zeros(0, dtype=bool), {
                agent_id: float("nan") for agent_id in self.possible_agents
            }
        fish_positions = self._all_root_positions()
        distances = np.linalg.norm(
            fish_positions[:, None, :] - self.food_positions[None, :, :],
            axis=2,
        )
        captured_mask = np.zeros(self.food_count, dtype=bool)
        food_eaten_by_agent = {agent_id: 0 for agent_id in self.possible_agents}
        capture_distances: dict[str, list[float]] = {agent_id: [] for agent_id in self.possible_agents}
        for food_index in range(self.food_count):
            food_team = int(self.food_team_indices[food_index])
            capturers = [
                idx
                for idx, agent_id in enumerate(self.possible_agents)
                if self.get_agent_team_index(agent_id) == food_team and distances[idx, food_index] <= self.food_capture_radius
            ]
            if not capturers:
                continue
            best_idx = int(min(capturers, key=lambda idx: float(distances[idx, food_index])))
            captured_mask[food_index] = True
            best_agent_id = self.possible_agents[best_idx]
            food_eaten_by_agent[best_agent_id] += 1
            capture_distances[best_agent_id].append(float(distances[best_idx, food_index]))
        mean_capture_distance = {
            agent_id: (float(np.mean(values)) if values else float("nan"))
            for agent_id, values in capture_distances.items()
        }
        return food_eaten_by_agent, captured_mask, mean_capture_distance

    def step(self, action_dict: dict[str, Any]):
        actions = self._resolve_step_actions(action_dict)
        next_states: dict[str, ArticulatedFishState] = {}
        previous_joint_velocities = {
            agent_id: self.fish_states[agent_id].joint_velocities.astype(np.float32).copy()
            for agent_id in self.possible_agents
        }

        for agent_id, action in actions.items():
            next_state = self._compute_next_state(
                agent_id,
                action["motion"],
                int(action["message"]),
            )
            next_states[agent_id] = next_state

        self.fish_states = next_states
        self.last_dynamics_debug = {agent_id: self._empty_dynamics_debug() for agent_id in self.possible_agents}
        self.last_message_tokens = {agent_id: int(action["message"]) for agent_id, action in actions.items()}
        self.timestep += 1
        actuation = self.fish_preset.actuation
        for agent_id in self.possible_agents:
            self.last_joint_zero_crossings[agent_id] = self._joint_zero_crossings(
                previous_joint_velocities[agent_id],
                self.fish_states[agent_id].joint_velocities,
            )
            self.last_joint_limit_occupancy[agent_id] = float(
                np.mean(np.abs(self.fish_states[agent_id].joint_angles) / max(float(actuation.joint_limit), 1e-6))
            )
            self._append_control_history(agent_id)

        food_eaten_by_agent, eaten_mask, capture_distances = self._food_capture_assignments()
        self.last_capture_distance = capture_distances
        if np.any(eaten_mask):
            for agent_id, eaten_count in food_eaten_by_agent.items():
                self.agent_food_eaten_episode[agent_id] += int(eaten_count)
                self.team_food_eaten_episode[self.get_agent_team_index(agent_id)] += int(eaten_count)
            self._respawn_food_indices(eaten_mask)

        truncated = bool(self.timestep >= self.time_limit)
        self.last_reward_breakdown = {
            agent_id: self.get_reward_breakdown(
                agent_id=agent_id,
                food_eaten_this_step=int(food_eaten_by_agent[agent_id]),
                truncated=truncated,
            )
            for agent_id in self.possible_agents
        }
        self.last_rewards = {
            agent_id: float(self.last_reward_breakdown[agent_id]["total_reward"])
            for agent_id in self.possible_agents
        }

        obs = self._get_obs_dict()
        infos = self._build_info_dict(truncated=truncated, food_eaten_by_agent=food_eaten_by_agent)
        rewards = {agent_id: float(self.last_rewards[agent_id]) for agent_id in self.possible_agents}
        terminateds = {agent_id: False for agent_id in self.possible_agents}
        terminateds["__all__"] = False
        truncateds = {agent_id: truncated for agent_id in self.possible_agents}
        truncateds["__all__"] = truncated
        return obs, rewards, terminateds, truncateds, infos

    def set_debug_state(
        self,
        *,
        agent_states: dict[str, dict[str, Any]],
        food_positions,
        food_team_indices=None,
        timestep: int = 0,
        focus_agent_id: str | None = None,
        last_message_tokens: dict[str, int] | None = None,
        food_eaten_episode_by_agent: dict[str, int] | None = None,
    ) -> None:
        self.agents = list(self.possible_agents)
        self.timestep = int(timestep)
        self.agent_food_eaten_episode = {agent_id: 0 for agent_id in self.possible_agents}
        self.team_food_eaten_episode = {RED_TEAM: 0, BLUE_TEAM: 0}
        self.fish_states = {}
        for agent_id in self.possible_agents:
            data = agent_states.get(agent_id, {})
            prev_message_token = _normalize_message_token(
                data.get("prev_message_token", 0),
                num_tokens=self.num_message_tokens,
            )
            self.fish_states[agent_id] = ArticulatedFishState(
                root_position=_as_float32_array(data.get("position", [0.0, 0.0]), shape=(2,)).copy(),
                root_velocity=_as_float32_array(data.get("velocity", [0.0, 0.0]), shape=(2,)).copy(),
                root_theta=float(data.get("theta", 0.0)),
                root_omega=float(data.get("omega", 0.0)),
                joint_angles=_as_float32_array(data.get("joint_angles", [0.0, 0.0]), shape=(2,)).copy(),
                joint_velocities=_as_float32_array(data.get("joint_velocities", [0.0, 0.0]), shape=(2,)).copy(),
                prev_action=_as_float32_array(data.get("prev_action", [0.0, 0.0]), shape=(2,)).copy(),
                prev_message_token=prev_message_token,
                applied_joint_torque=_as_float32_array(data.get("applied_joint_torque", [0.0, 0.0]), shape=(2,)).copy(),
            )
            if food_eaten_episode_by_agent and agent_id in food_eaten_episode_by_agent:
                self.agent_food_eaten_episode[agent_id] = int(food_eaten_episode_by_agent[agent_id])
            team_index = self.get_agent_team_index(agent_id)
            self.team_food_eaten_episode[team_index] += int(self.agent_food_eaten_episode[agent_id])
            raw_history = np.asarray(data.get("control_history", []), dtype=np.float32).reshape(-1, 9)
            if raw_history.shape == (self.history_length, 9):
                self.control_histories[agent_id] = raw_history.copy()
            else:
                self._initialize_control_history(agent_id)
            self.last_joint_limit_occupancy[agent_id] = float(
                np.mean(np.abs(self.fish_states[agent_id].joint_angles) / max(float(self.fish_preset.actuation.joint_limit), 1e-6))
            )
            self.last_joint_zero_crossings[agent_id] = 0
            self.last_capture_distance[agent_id] = float("nan")
        food_positions_array = np.asarray(food_positions, dtype=np.float32).reshape(-1, 2)
        if food_positions_array.shape[0] != self.food_count:
            raise ValueError(f"Expected {self.food_count} food positions, got {food_positions_array.shape[0]}.")
        self.food_positions = food_positions_array.copy()
        if food_team_indices is None:
            self.food_team_indices = self._default_food_team_indices()
        else:
            raw_indices = np.asarray(food_team_indices).reshape(-1)
            if raw_indices.shape[0] != self.food_count:
                raise ValueError(f"Expected {self.food_count} food team indices, got {raw_indices.shape[0]}.")
            normalized = np.zeros(self.food_count, dtype=np.int64)
            for idx, value in enumerate(raw_indices):
                if isinstance(value, str):
                    normalized[idx] = RED_TEAM if value.lower() == "red" else BLUE_TEAM
                else:
                    normalized[idx] = int(value)
            self.food_team_indices = normalized
        self.last_message_tokens = {
            agent_id: _normalize_message_token(
                0 if last_message_tokens is None else last_message_tokens.get(agent_id, 0),
                num_tokens=self.num_message_tokens,
            )
            for agent_id in self.possible_agents
        }
        if focus_agent_id is not None:
            self.set_focus_agent(focus_agent_id)
        self.last_rewards = {agent_id: 0.0 for agent_id in self.possible_agents}
        self.last_reward_breakdown = {
            agent_id: self.get_reward_breakdown(agent_id=agent_id, food_eaten_this_step=0, truncated=False)
            for agent_id in self.possible_agents
        }
        self.last_dynamics_debug = {agent_id: self._empty_dynamics_debug() for agent_id in self.possible_agents}
        self._get_obs_dict()

    def get_debug_snapshot(self, agent_id: str | None = None) -> Dict[str, Any]:
        agent_id = agent_id or self.focus_agent_id
        if agent_id not in self._agent_ids:
            raise ValueError(f"Unknown agent id: {agent_id}")
        obs = self._compute_agent_obs(agent_id)
        state = self.fish_states[agent_id]
        geometry = self._segment_geometry(state)
        dynamics_breakdown = self.get_dynamics_breakdown(
            agent_id,
            state.prev_action,
            state=state,
            dt=self.fish_preset.dynamics.dt,
        )
        return {
            "agent_id": agent_id,
            "agent_team": self.get_agent_team_name(agent_id),
            "focus_agent_id": self.focus_agent_id,
            "root_position": state.root_position.astype(np.float32).copy(),
            "root_velocity": state.root_velocity.astype(np.float32).copy(),
            "root_theta": float(state.root_theta),
            "root_omega": float(state.root_omega),
            "joint_angles": state.joint_angles.astype(np.float32).copy(),
            "joint_velocities": state.joint_velocities.astype(np.float32).copy(),
            "prev_action": state.prev_action.astype(np.float32).copy(),
            "prev_message_token": int(state.prev_message_token),
            "applied_joint_torque": state.applied_joint_torque.astype(np.float32).copy(),
            "food_positions": self.food_positions.astype(np.float32).copy(),
            "food_team_indices": self.food_team_indices.astype(np.int64).copy(),
            "nearest_food_distance": float(self.last_nearest_food_distance[agent_id]),
            "capture_distance_this_step": float(self.last_capture_distance[agent_id]),
            "observation_channels": {
                key: value.astype(np.float32).copy()
                for key, value in self.last_observation_channels[agent_id].items()
            },
            "visible_counts": dict(self.last_visible_counts[agent_id]),
            "sensor_active_bins": {key: list(value) for key, value in self.last_sensor_active_bins[agent_id].items()},
            "food_eaten_episode_red": int(self.team_food_eaten_episode[RED_TEAM]),
            "food_eaten_episode_blue": int(self.team_food_eaten_episode[BLUE_TEAM]),
            "food_eaten_episode": int(self.agent_food_eaten_episode[agent_id]),
            "timestep": int(self.timestep),
            "observation": obs.copy(),
            "control_history": self.control_histories[agent_id].astype(np.float32).copy(),
            "joint_velocity_zero_crossings": int(self.last_joint_zero_crossings[agent_id]),
            "mean_joint_limit_ratio": float(self.last_joint_limit_occupancy[agent_id]),
            "motion_metrics": dict(self._current_motion_metrics(agent_id)),
            "segment_centers": geometry["centers"].astype(np.float32).copy(),
            "segment_angles": geometry["angles"].astype(np.float32).copy(),
            "joint_positions": geometry["joint_positions"].astype(np.float32).copy(),
            "reward_breakdown": dict(self.last_reward_breakdown[agent_id]),
            "last_message_tokens": dict(self.last_message_tokens),
            "all_root_positions": {
                other_id: self.fish_states[other_id].root_position.astype(np.float32).copy()
                for other_id in self.possible_agents
            },
            "dynamics_breakdown": {
                key: (np.asarray(value, dtype=np.float32).copy() if isinstance(value, np.ndarray) else value)
                for key, value in dynamics_breakdown.items()
            },
        }

    def _initialize_rendering(self) -> None:
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.render_backend_name = self._resolve_render_backend_name()
        self.render_force_full_redraw = True
        self._clear_render_background(force_full_redraw=True)
        self._connect_render_events()
        self.ax.set_xlim(-self.border, self.border)
        self.ax.set_ylim(-self.border, self.border)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self._border_artist, = self.ax.plot(
            [-self.border, self.border, self.border, -self.border, -self.border],
            [-self.border, -self.border, self.border, self.border, -self.border],
            color="#23303d",
            linewidth=1.0,
        )
        self.segment_patches = {}
        self.message_texts = {}
        self.joint_plots = {}
        for agent_id in self.possible_agents:
            patches: list[Polygon] = []
            palette = (
                self.fish_preset.render.red_segment_colors
                if self.get_agent_team_index(agent_id) == RED_TEAM
                else self.fish_preset.render.blue_segment_colors
            )
            for color in palette:
                patch = Polygon(
                    np.zeros((4, 2), dtype=np.float32),
                    closed=True,
                    facecolor=color,
                    edgecolor="#0d2533",
                    linewidth=1.0,
                    animated=True,
                    zorder=3.0,
                )
                self.ax.add_patch(patch)
                patches.append(patch)
            self.segment_patches[agent_id] = patches
            if self.render_profile == "full":
                joint_plot, = self.ax.plot(
                    [],
                    [],
                    "o",
                    color=self.fish_preset.render.joint_color,
                    markersize=3,
                    zorder=3.1,
                    animated=True,
                )
                self.joint_plots[agent_id] = joint_plot
                message_text = self.ax.text(
                    0.0,
                    0.0,
                    "0",
                    color=TEAM_COLORS[self.get_agent_team_index(agent_id)],
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    zorder=4.0,
                    animated=True,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=(0.03, 0.09, 0.14, 0.82), edgecolor="none"),
                )
                self.message_texts[agent_id] = message_text

        if self.render_profile == "full" and self.show_sensor_overlay:
            channel_edges = np.linspace(0.0, self.sector_radius, len(self.CHANNEL_ORDER) + 1, dtype=np.float32)
            sector_width_deg = 360.0 / float(self.sector_num)
            for channel_index, channel_name in enumerate(self.CHANNEL_ORDER):
                outer_radius = float(channel_edges[channel_index + 1])
                inner_radius = float(channel_edges[channel_index])
                for _ in range(self.sector_num):
                    patch = Wedge(
                        center=(0.0, 0.0),
                        r=outer_radius,
                        theta1=-0.5 * sector_width_deg,
                        theta2=0.5 * sector_width_deg,
                        width=max(outer_radius - inner_radius, 1e-6),
                        facecolor=self._channel_color(channel_name),
                        edgecolor="none",
                        alpha=0.0,
                        animated=True,
                        zorder=0.3 + (0.01 * channel_index),
                    )
                    self.ax.add_patch(patch)
                    self.sensor_patches[channel_name].append(patch)
            self._add_sensor_legend()

        self.red_food_scatter = self.ax.scatter([], [], s=22, c=self.fish_preset.render.red_food_color, zorder=2.0)
        self.red_food_scatter.set_animated(True)
        self.blue_food_scatter = self.ax.scatter([], [], s=22, c=self.fish_preset.render.blue_food_color, zorder=2.0)
        self.blue_food_scatter.set_animated(True)
        self.status_text = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            color="#d9f6ff",
            fontsize=9,
            ha="left",
            va="top",
            animated=True,
            zorder=5.3,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=(0.03, 0.09, 0.14, 0.82), edgecolor="none"),
        )
        self._refresh_render_background()

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

    def _resolve_render_backend_name(self) -> str:
        if self.fig is None:
            return ""
        return self.fig.canvas.__class__.__name__.lower()

    def _compute_render_blit_bbox(self):
        if self.fig is None or self.ax is None:
            return None
        backend_name = self.render_backend_name or self._resolve_render_backend_name()
        if self.render_engine == "safe" or "tk" in backend_name:
            return self.fig.bbox
        return self.ax.bbox

    def _clear_render_background(self, *, force_full_redraw: bool = False) -> None:
        self._render_background = None
        self.render_blit_bbox = None
        self.render_background_valid = False
        if force_full_redraw:
            self.render_force_full_redraw = True

    def _capture_render_background(self) -> None:
        if self.fig is None:
            return
        if not hasattr(self.fig.canvas, "copy_from_bbox"):
            self._clear_render_background()
            return
        self.render_backend_name = self._resolve_render_backend_name()
        self.render_blit_bbox = self._compute_render_blit_bbox()
        if self.render_blit_bbox is None:
            self._clear_render_background()
            return
        self._render_background = self.fig.canvas.copy_from_bbox(self.render_blit_bbox)
        self._render_background_size = tuple(self.fig.canvas.get_width_height())
        self.render_background_valid = True
        self.render_force_full_redraw = False

    def _on_draw_event(self, event) -> None:
        if self.fig is None or event.canvas is not self.fig.canvas:
            return
        self.render_backend_name = self._resolve_render_backend_name()
        self._render_background_size = tuple(self.fig.canvas.get_width_height())
        if self._render_suppress_draw_event_capture or self.render_engine == "safe":
            self._clear_render_background()
            return
        self.render_background_valid = False
        self._capture_render_background()

    def _on_resize_event(self, event) -> None:
        if self.fig is None or event.canvas is not self.fig.canvas:
            return
        self._clear_render_background(force_full_redraw=True)

    def _on_close_event(self, event) -> None:
        if self.fig is None or event.canvas is not self.fig.canvas:
            return
        self._disconnect_render_events()
        self._reset_render_context()

    def _connect_render_events(self) -> None:
        if self.fig is None:
            return
        self._disconnect_render_events()
        canvas = self.fig.canvas
        self._draw_event_cid = canvas.mpl_connect("draw_event", self._on_draw_event)
        self._resize_event_cid = canvas.mpl_connect("resize_event", self._on_resize_event)
        self._close_event_cid = canvas.mpl_connect("close_event", self._on_close_event)

    def _disconnect_render_events(self) -> None:
        if self.fig is None:
            self._draw_event_cid = None
            self._resize_event_cid = None
            self._close_event_cid = None
            return
        canvas = self.fig.canvas
        if self._draw_event_cid is not None:
            canvas.mpl_disconnect(self._draw_event_cid)
            self._draw_event_cid = None
        if self._resize_event_cid is not None:
            canvas.mpl_disconnect(self._resize_event_cid)
            self._resize_event_cid = None
        if self._close_event_cid is not None:
            canvas.mpl_disconnect(self._close_event_cid)
            self._close_event_cid = None

    def _canvas_supports_blit(self) -> bool:
        if self.fig is None:
            return False
        canvas = self.fig.canvas
        if not all(hasattr(canvas, attr) for attr in ("restore_region", "blit", "copy_from_bbox")):
            return False
        supports_blit = getattr(canvas, "supports_blit", None)
        return bool(True if supports_blit is None else supports_blit)

    def _can_use_cached_blit(self) -> bool:
        return bool(
            self.render_engine != "safe"
            and self.fig is not None
            and self.ax is not None
            and self.render_background_valid
            and self._render_background is not None
            and self.render_blit_bbox is not None
            and not self.render_force_full_redraw
            and self._canvas_supports_blit()
            and len(self._dynamic_render_artists()) > 0
        )

    def _add_sensor_legend(self) -> None:
        if self.ax is None:
            return
        panel = Rectangle(
            (0.02, 0.62),
            0.38,
            0.33,
            transform=self.ax.transAxes,
            facecolor=(0.03, 0.09, 0.14, 0.84),
            edgecolor=(0.26, 0.71, 0.85, 0.40),
            linewidth=1.0,
            clip_on=False,
            zorder=5.0,
        )
        self.ax.add_patch(panel)
        self.sensor_legend_artists.append(panel)
        title = self.ax.text(
            0.04,
            0.925,
            "Focus Fish Channels",
            transform=self.ax.transAxes,
            color="#d9f6ff",
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=5.1,
        )
        self.sensor_legend_artists.append(title)
        rows = [
            ("edible food", self._channel_color("edible_food")),
            ("non-edible food", self._channel_color("non_edible_food")),
            ("teammates", self._channel_color("teammate")),
            ("opponents", self._channel_color("opponent")),
            ("teammate msg", self._channel_color("teammate_message")),
            ("opponent msg", self._channel_color("opponent_message")),
        ]
        for row_idx, (label, color) in enumerate(rows):
            y_pos = 0.885 - (0.035 * row_idx)
            swatch = Rectangle(
                (0.04, y_pos - 0.012),
                0.02,
                0.02,
                transform=self.ax.transAxes,
                facecolor=color,
                edgecolor="none",
                alpha=self._sensor_bin_alpha(1.0),
                clip_on=False,
                zorder=5.1,
            )
            self.ax.add_patch(swatch)
            self.sensor_legend_artists.append(swatch)
            artist = self.ax.text(
                0.07,
                y_pos,
                label,
                transform=self.ax.transAxes,
                color="#d9f6ff",
                fontsize=8,
                ha="left",
                va="center",
                zorder=5.1,
            )
            self.sensor_legend_artists.append(artist)
        notes = [
            "6 sectors, sector 0 = forward",
            "sectors advance CCW",
            "tokens: 0 silence, 1..3 learned",
        ]
        for idx, note in enumerate(notes):
            text = self.ax.text(
                0.04,
                0.675 - (0.03 * idx),
                note,
                transform=self.ax.transAxes,
                color="#d9f6ff",
                fontsize=8,
                ha="left",
                va="center",
                zorder=5.1,
            )
            self.sensor_legend_artists.append(text)
        arrow = FancyArrowPatch(
            (0.23, 0.675),
            (0.33, 0.675),
            transform=self.ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=12.0,
            linewidth=1.0,
            color=self.fish_preset.render.edible_food_sensor_color,
            zorder=5.2,
            clip_on=False,
        )
        self.ax.add_patch(arrow)
        self.sensor_legend_artists.append(arrow)

    def _refresh_render_background(self) -> None:
        if self.fig is None or self.ax is None:
            return
        self.render_backend_name = self._resolve_render_backend_name()
        self.fig.canvas.draw()
        if not self.render_background_valid and self.render_engine != "safe":
            self._capture_render_background()

    def _dynamic_render_artists(self) -> list[Any]:
        artists: list[Any] = []
        if self.render_profile == "full" and self.show_sensor_overlay:
            for patches in self.sensor_patches.values():
                artists.extend(patches)
        if self.red_food_scatter is not None:
            artists.append(self.red_food_scatter)
        if self.blue_food_scatter is not None:
            artists.append(self.blue_food_scatter)
        for patches in self.segment_patches.values():
            artists.extend(patches)
        if self.render_profile == "full":
            artists.extend(self.joint_plots.values())
            artists.extend(self.message_texts.values())
        if self.status_text is not None:
            artists.append(self.status_text)
        return artists

    def _perform_safe_redraw(self) -> None:
        if self.fig is None:
            return
        self.render_backend_name = self._resolve_render_backend_name()
        self.fig.canvas.draw()
        artists = self._dynamic_render_artists()
        if self.ax is not None:
            for artist in artists:
                self.ax.draw_artist(artist)
        if self._canvas_supports_blit():
            present_bbox = self._compute_render_blit_bbox()
            if present_bbox is not None:
                self.fig.canvas.blit(present_bbox)
        self.fig.canvas.flush_events()
        if self.render_engine == "safe":
            self._clear_render_background()
            self.render_force_full_redraw = False
            self._render_background_size = tuple(self.fig.canvas.get_width_height())
            return
        if not self.render_background_valid:
            self._capture_render_background()

    def _reset_render_context(self) -> None:
        self.fig = None
        self.ax = None
        self.segment_patches = {}
        self.message_texts = {}
        self.sensor_patches = {name: [] for name in self.CHANNEL_ORDER}
        self.sensor_legend_artists = []
        self.red_food_scatter = None
        self.blue_food_scatter = None
        self.joint_plots = {}
        self.status_text = None
        self._border_artist = None
        self._render_background = None
        self._render_background_size = None
        self.render_backend_name = None
        self.render_blit_bbox = None
        self.render_background_valid = False
        self.render_force_full_redraw = True
        self._render_suppress_draw_event_capture = False
        self._draw_event_cid = None
        self._resize_event_cid = None
        self._close_event_cid = None

    def _update_render_artists(self) -> None:
        morphology = self.fish_preset.morphology
        for agent_id in self.possible_agents:
            geometry = self._segment_geometry(self.fish_states[agent_id])
            for idx, patch in enumerate(self.segment_patches[agent_id]):
                polygon = self._segment_polygon(
                    geometry["centers"][idx],
                    float(geometry["angles"][idx]),
                    float(morphology.segment_lengths[idx]),
                    float(morphology.segment_front_widths[idx]),
                    float(morphology.segment_back_widths[idx]),
                )
                patch.set_xy(polygon)
            if self.render_profile == "full":
                self.joint_plots[agent_id].set_data(geometry["joint_positions"][:, 0], geometry["joint_positions"][:, 1])
                text_pos = self.fish_states[agent_id].root_position + np.array([0.0, 0.45], dtype=np.float32)
                self.message_texts[agent_id].set_position((float(text_pos[0]), float(text_pos[1])))
                self.message_texts[agent_id].set_text(str(self.last_message_tokens[agent_id]))

        red_positions = self.food_positions[self.food_team_indices == RED_TEAM]
        blue_positions = self.food_positions[self.food_team_indices == BLUE_TEAM]
        self.red_food_scatter.set_offsets(red_positions if red_positions.size else np.zeros((0, 2), dtype=np.float32))
        self.blue_food_scatter.set_offsets(blue_positions if blue_positions.size else np.zeros((0, 2), dtype=np.float32))

        if self.render_profile == "full" and self.show_sensor_overlay and self.focus_agent_id in self._agent_ids:
            focus_state = self.fish_states[self.focus_agent_id]
            channel_edges = np.linspace(0.0, self.sector_radius, len(self.CHANNEL_ORDER) + 1, dtype=np.float32)
            sector_width = 2.0 * math.pi / float(self.sector_num)
            theta_base = float(focus_state.root_theta)
            channel_data = self.last_observation_channels[self.focus_agent_id]
            for channel_index, channel_name in enumerate(self.CHANNEL_ORDER):
                outer_radius = float(channel_edges[channel_index + 1])
                inner_radius = float(channel_edges[channel_index])
                bins = channel_data[channel_name]
                for sector_index in range(self.sector_num):
                    center_angle = theta_base + (sector_index * sector_width)
                    patch = self.sensor_patches[channel_name][sector_index]
                    patch.set_center((float(focus_state.root_position[0]), float(focus_state.root_position[1])))
                    patch.set_radius(outer_radius)
                    patch.set_width(max(outer_radius - inner_radius, 1e-6))
                    patch.theta1 = math.degrees(center_angle - 0.5 * sector_width)
                    patch.theta2 = math.degrees(center_angle + 0.5 * sector_width)
                    patch.set_alpha(self._sensor_bin_alpha(float(bins[sector_index])))

        if self.status_text is not None:
            focus_agent_id = self.focus_agent_id if self.focus_agent_id in self._agent_ids else self.possible_agents[0]
            focus_team = self.get_agent_team_name(focus_agent_id)
            focus_counts = self.last_visible_counts[focus_agent_id]
            self.status_text.set_text(
                f"V9 | step={self.timestep}/{self.time_limit} | red={self.team_food_eaten_episode[RED_TEAM]} "
                f"blue={self.team_food_eaten_episode[BLUE_TEAM]} | focus={focus_agent_id}({focus_team}) "
                f"food={focus_counts['edible_food'] + focus_counts['non_edible_food']} "
                f"team={focus_counts['teammate']} opp={focus_counts['opponent']}"
            )

    def render(self):
        if self.render_mode != "human":
            return
        if self.fig is None:
            self._initialize_rendering()
        self._update_render_artists()
        current_canvas_size = tuple(self.fig.canvas.get_width_height())
        if self._render_background_size != current_canvas_size:
            self._clear_render_background(force_full_redraw=True)
            self._render_background_size = current_canvas_size
        if not self._can_use_cached_blit():
            self._perform_safe_redraw()
            return
        if self._render_background is not None and self.render_blit_bbox is not None:
            self.fig.canvas.restore_region(self._render_background)
            for artist in self._dynamic_render_artists():
                self.ax.draw_artist(artist)
            self.fig.canvas.blit(self.render_blit_bbox)
            self.fig.canvas.flush_events()
            return
        self._perform_safe_redraw()

    def close(self):
        if self.fig is not None:
            self._disconnect_render_events()
            plt.ioff()
            plt.close(self.fig)
            self._reset_render_context()


    def _channel_color(self, channel_name: str) -> str:
        render = self.fish_preset.render
        mapping = {
            "edible_food": render.edible_food_sensor_color,
            "non_edible_food": render.non_edible_food_sensor_color,
            "teammate": render.teammate_sensor_color,
            "opponent": render.opponent_sensor_color,
            "teammate_message": render.teammate_message_sensor_color,
            "opponent_message": render.opponent_message_sensor_color,
        }
        return mapping[channel_name]


OctopusEnv = CommunicatingSchoolEnv


if __name__ == "__main__":
    env = CommunicatingSchoolEnv(epsilon=0.1, render_mode="human")
    obs, info = env.reset(seed=0)
    print("Initial agents:", list(obs.keys()))
    print("Observation shape:", obs["fish_0"].shape)
    for _ in range(200):
        actions = {agent_id: env.action_space.sample() for agent_id in env.get_agent_ids()}
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        env.render()
        if terminateds["__all__"] or truncateds["__all__"]:
            obs, info = env.reset()
    env.close()
