"""V9 environment: muscle-activation articulated schooling with local communication."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import math
from typing import Any, Dict

from gymnasium.spaces import Box, Dict as DictSpace, Discrete
from gymnasium.utils import seeding
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


RED_TEAM = 0
BLUE_TEAM = 1
TEAM_NAMES = {RED_TEAM: "red", BLUE_TEAM: "blue"}
TEAM_COLORS = {RED_TEAM: "#d1495b", BLUE_TEAM: "#2f6fe4"}

SCRIPTED_WAVE_AMPLITUDE = 0.95
SCRIPTED_WAVE_PHASE_RATE = 0.34
SCRIPTED_WAVE_PHASE_DELTA = math.pi / 2.0
SCRIPTED_WAVE_FORWARD_REFERENCE = 0.15
SCRIPTED_WAVE_ANGULAR_REFERENCE = 1.5
PROPULSION_PROGRESS_WINDOW_STEPS = 10
PROPULSION_SATURATION_THRESHOLD = 0.75

_RENDER_PLT = None
_RENDER_PATCH_TYPES: dict[str, Any] | None = None


def _get_render_matplotlib() -> tuple[Any, dict[str, Any]]:
    global _RENDER_PLT, _RENDER_PATCH_TYPES
    if _RENDER_PLT is None or _RENDER_PATCH_TYPES is None:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle, Wedge

        _RENDER_PLT = plt
        _RENDER_PATCH_TYPES = {
            "FancyArrowPatch": FancyArrowPatch,
            "Polygon": Polygon,
            "Rectangle": Rectangle,
            "Wedge": Wedge,
        }
    return _RENDER_PLT, dict(_RENDER_PATCH_TYPES)


def _canonical_training_phase(training_phase: str, *, reward_mode: str | None = None) -> str:
    phase = str(training_phase).strip().lower()
    if phase == "locomotion_only":
        phase = "locomotion_self"
    if phase == "forage_full" and str(reward_mode).strip().lower() == "locomotion_debug":
        phase = "locomotion_self"
    return phase


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
    segment_lengths: tuple[float, ...]
    segment_front_widths: tuple[float, ...]
    segment_back_widths: tuple[float, ...]


@dataclass(frozen=True)
class FishDynamicsConfig:
    mass: float = 1.0
    inertia: float = 1.6
    dt: float = 0.05
    max_integration_dt: float = 0.01
    segment_parallel_drag: float = 3.5
    segment_perp_drag: float = 28.0
    segment_angular_drag: float = 0.16
    root_rotational_drag: float = 1.8
    body_linear_drag: float = 1.0
    max_speed: float = 10.0
    max_angular_speed: float = 8.0


@dataclass(frozen=True)
class FishActuationConfig:
    joint_limit: float = 0.9
    joint_torque_limit: float = 12.0
    joint_passive_stiffness: float = 10.0
    joint_passive_damping: float = 3.0
    joint_soft_limit_start_ratio: float = 0.70
    joint_soft_limit_stiffness: float = 18.0
    joint_soft_limit_damping: float = 2.0
    joint_inertia: float = 0.18
    joint_max_speed: float = 8.0


@dataclass(frozen=True)
class FishRenderConfig:
    red_segment_colors: tuple[str, ...]
    blue_segment_colors: tuple[str, ...]
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


_EEL_HEAD_LENGTH = 0.72
_EEL_LENGTH_TAPER = 0.82
_EEL_FRONT_WIDTH_ANCHORS = np.asarray([0.22, 0.16, 0.10], dtype=np.float32)
_EEL_BACK_WIDTH_ANCHORS = np.asarray([0.18, 0.12, 0.05], dtype=np.float32)
_EEL_RED_SEGMENT_COLORS = ("#8f2533", "#c03d50", "#f08f98")
_EEL_BLUE_SEGMENT_COLORS = ("#1d458c", "#2f6fe4", "#8eb4ff")
_REFERENCE_DYNAMICS = FishDynamicsConfig()
_REFERENCE_ACTUATION = FishActuationConfig()


def _interpolate_profile(anchors: np.ndarray, count: int) -> tuple[float, ...]:
    if count <= 0:
        return ()
    if count == 1:
        return (float(anchors[0]),)
    positions = np.linspace(0.0, 1.0, count, dtype=np.float32)
    anchor_positions = np.linspace(0.0, 1.0, anchors.shape[0], dtype=np.float32)
    return tuple(float(value) for value in np.interp(positions, anchor_positions, anchors))


def _hex_to_rgb(color: str) -> np.ndarray:
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Unsupported hex color: {color}")
    return np.array([int(value[idx : idx + 2], 16) for idx in range(0, 6, 2)], dtype=np.float32)


def _rgb_to_hex(rgb: np.ndarray) -> str:
    clipped = np.clip(np.round(rgb), 0.0, 255.0).astype(np.uint8)
    return "#" + "".join(f"{channel:02x}" for channel in clipped.tolist())


def _interpolate_palette(colors: tuple[str, ...], count: int) -> tuple[str, ...]:
    if count <= 0:
        return ()
    if count == len(colors):
        return tuple(colors)
    if count == 1:
        return (colors[0],)
    anchors = np.stack([_hex_to_rgb(color) for color in colors], axis=0)
    positions = np.linspace(0.0, 1.0, count, dtype=np.float32)
    anchor_positions = np.linspace(0.0, 1.0, anchors.shape[0], dtype=np.float32)
    interpolated = np.stack(
        [np.interp(positions, anchor_positions, anchors[:, channel_idx]) for channel_idx in range(3)],
        axis=1,
    )
    return tuple(_rgb_to_hex(rgb) for rgb in interpolated)


def _body_planform_area(morphology: FishMorphology) -> float:
    lengths = np.asarray(morphology.segment_lengths, dtype=np.float32)
    widths = 0.5 * (
        np.asarray(morphology.segment_front_widths, dtype=np.float32)
        + np.asarray(morphology.segment_back_widths, dtype=np.float32)
    )
    return float(np.sum(lengths * widths))


def _build_eel_morphology(num_body_segments: int) -> FishMorphology:
    if num_body_segments < 2:
        raise ValueError("num_body_segments must be >= 2.")
    lengths = tuple(float(_EEL_HEAD_LENGTH * (_EEL_LENGTH_TAPER ** idx)) for idx in range(num_body_segments))
    return FishMorphology(
        segment_lengths=lengths,
        segment_front_widths=_interpolate_profile(_EEL_FRONT_WIDTH_ANCHORS, num_body_segments),
        segment_back_widths=_interpolate_profile(_EEL_BACK_WIDTH_ANCHORS, num_body_segments),
    )


_REFERENCE_MORPHOLOGY = _build_eel_morphology(3)
_REFERENCE_BODY_AREA = _body_planform_area(_REFERENCE_MORPHOLOGY)
_REFERENCE_TOTAL_LENGTH = float(np.sum(np.asarray(_REFERENCE_MORPHOLOGY.segment_lengths, dtype=np.float32)))


def make_eel_preset(num_body_segments: int) -> FishPreset:
    morphology = _build_eel_morphology(int(num_body_segments))
    total_length = float(np.sum(np.asarray(morphology.segment_lengths, dtype=np.float32)))
    body_area = _body_planform_area(morphology)
    mass_scale = body_area / max(_REFERENCE_BODY_AREA, 1e-6)
    length_scale = total_length / max(_REFERENCE_TOTAL_LENGTH, 1e-6)
    return FishPreset(
        name=f"eel_{int(num_body_segments)}seg",
        morphology=morphology,
        dynamics=replace(
            _REFERENCE_DYNAMICS,
            mass=float(_REFERENCE_DYNAMICS.mass * mass_scale),
            inertia=float(_REFERENCE_DYNAMICS.inertia * mass_scale * (length_scale ** 2)),
        ),
        actuation=_REFERENCE_ACTUATION,
        render=FishRenderConfig(
            red_segment_colors=_interpolate_palette(_EEL_RED_SEGMENT_COLORS, int(num_body_segments)),
            blue_segment_colors=_interpolate_palette(_EEL_BLUE_SEGMENT_COLORS, int(num_body_segments)),
        ),
    )


EEL_3SEG_PRESET = make_eel_preset(3)
DEFAULT_V9_FISH_PRESET = make_eel_preset(5)


@dataclass
class ArticulatedFishState:
    root_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    root_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    root_theta: float = 0.0
    root_omega: float = 0.0
    joint_angles: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    joint_velocities: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    prev_action: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    prev_message_token: int = 0
    joint_activation: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    applied_joint_torque: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


class CommunicatingSchoolEnv(MultiAgentEnv):
    """Muscle-activation multi-agent schooling task with one shared trainable policy."""

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
        epsilon: float | None = None,
        motion_epsilon: float | None = None,
        message_epsilon: float = 0.0,
        render_mode: str | None = None,
        render_profile: str = "fast",
        render_engine: str = "auto",
        num_body_segments: int = 5,
        fish_preset: FishPreset | str | None = None,
        time_limit: int = 300,
        num_red_fish: int = 10,
        num_blue_fish: int = 10,
        num_red_pellets: int = 48,
        num_blue_pellets: int = 48,
        food_capture_radius: float = 0.45,
        pellet_reward: float = 1.0,
        step_cost: float = 0.002,
        food_respawn_mode: str = "respawn",
        forage_timeout_mode: str = "fixed_time_limit",
        forage_idle_timeout_steps: int = 500,
        forage_time_context_mode: str = "episode_progress",
        sector_radius: float = 5.0,
        sector_num: int = 6,
        communication_radius: float | None = None,
        num_message_tokens: int = 4,
        reward_mode: str = "forage",
        training_phase: str = "forage_full",
        observation_profile: str = "full_v9",
        history_length: int = 8,
        activation_time_constant: float = 0.12,
        joint_passive_stiffness: float | None = None,
        joint_soft_limit_start_ratio: float | None = None,
        joint_soft_limit_stiffness: float | None = None,
        joint_soft_limit_damping: float | None = None,
        body_linear_drag: float | None = None,
        propulsion_near_limit_weight: float = -0.22,
        propulsion_saturation_weight: float = -0.10,
        propulsion_torque_weight: float = -0.05,
        swim_assist_start_weight: float = 0.0,
        show_sensor_overlay: bool = True,
        focus_agent_id: str = "fish_0",
        mute_received_messages: bool = False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.render_profile = str(render_profile).strip().lower()
        self.render_engine = str(render_engine).strip().lower()
        resolved_motion_epsilon = motion_epsilon if motion_epsilon is not None else (0.0 if epsilon is None else epsilon)
        self.motion_epsilon = float(resolved_motion_epsilon)
        self.message_epsilon = float(message_epsilon)
        self.eps = float(self.motion_epsilon)
        self.num_body_segments = int(num_body_segments)
        if self.num_body_segments < 2:
            raise ValueError("num_body_segments must be >= 2.")
        self.num_joints = self.num_body_segments - 1
        base_preset = self._resolve_preset(fish_preset)
        self.num_body_segments = int(len(base_preset.morphology.segment_lengths))
        self.num_joints = self.num_body_segments - 1
        if joint_passive_stiffness is not None:
            base_preset = replace(
                base_preset,
                actuation=replace(base_preset.actuation, joint_passive_stiffness=float(joint_passive_stiffness)),
            )
        if joint_soft_limit_start_ratio is not None:
            base_preset = replace(
                base_preset,
                actuation=replace(base_preset.actuation, joint_soft_limit_start_ratio=float(joint_soft_limit_start_ratio)),
            )
        if joint_soft_limit_stiffness is not None:
            base_preset = replace(
                base_preset,
                actuation=replace(base_preset.actuation, joint_soft_limit_stiffness=float(joint_soft_limit_stiffness)),
            )
        if joint_soft_limit_damping is not None:
            base_preset = replace(
                base_preset,
                actuation=replace(base_preset.actuation, joint_soft_limit_damping=float(joint_soft_limit_damping)),
            )
        if body_linear_drag is not None:
            base_preset = replace(
                base_preset,
                dynamics=replace(base_preset.dynamics, body_linear_drag=float(body_linear_drag)),
            )
        self.fish_preset = base_preset
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
        self.food_respawn_mode = str(food_respawn_mode).strip().lower()
        self.forage_timeout_mode = str(forage_timeout_mode).strip().lower()
        self.forage_idle_timeout_steps = int(forage_idle_timeout_steps)
        self.forage_time_context_mode = str(forage_time_context_mode).strip().lower()
        self.sector_radius = float(sector_radius)
        self.communication_radius = float(communication_radius if communication_radius is not None else sector_radius)
        self.sector_num = int(sector_num)
        self.num_message_tokens = int(num_message_tokens)
        self.reward_mode = str(reward_mode).strip().lower()
        self.training_phase = _canonical_training_phase(training_phase, reward_mode=self.reward_mode)
        self.teacher_phase_enabled = self.training_phase == "locomotion_teacher"
        self.observation_profile = str(observation_profile).strip().lower()
        self.history_length = int(history_length)
        self.activation_time_constant = float(activation_time_constant)
        self.propulsion_near_limit_weight = float(propulsion_near_limit_weight)
        self.propulsion_saturation_weight = float(propulsion_saturation_weight)
        self.propulsion_torque_weight = float(propulsion_torque_weight)
        self.swim_assist_start_weight = float(swim_assist_start_weight)
        self.show_sensor_overlay = bool(show_sensor_overlay)
        self.mute_received_messages = bool(mute_received_messages)
        if self.time_limit <= 0:
            raise ValueError("time_limit must be > 0.")
        if self.reward_mode not in {"forage", "locomotion_debug"}:
            raise ValueError("reward_mode must be 'forage' or 'locomotion_debug'.")
        if self.training_phase not in {
            "forage_full",
            "locomotion_teacher",
            "locomotion_self",
            "locomotion_propulsion_easy",
            "locomotion_propulsion_robust",
        }:
            raise ValueError(
                "training_phase must be forage_full, locomotion_teacher, locomotion_self, locomotion_propulsion_easy, or locomotion_propulsion_robust."
            )
        if self.observation_profile not in {"full_v9", "minimal_wave"}:
            raise ValueError("observation_profile must be 'full_v9' or 'minimal_wave'.")
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
        if self.food_respawn_mode not in {"respawn", "deplete"}:
            raise ValueError("food_respawn_mode must be 'respawn' or 'deplete'.")
        if self.forage_timeout_mode not in {"fixed_time_limit", "reset_on_food"}:
            raise ValueError("forage_timeout_mode must be 'fixed_time_limit' or 'reset_on_food'.")
        if self.forage_idle_timeout_steps <= 0:
            raise ValueError("forage_idle_timeout_steps must be > 0.")
        if self.forage_time_context_mode not in {"episode_progress", "idle_budget_remaining"}:
            raise ValueError("forage_time_context_mode must be 'episode_progress' or 'idle_budget_remaining'.")
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
        if self.activation_time_constant < 0.0:
            raise ValueError("activation_time_constant must be >= 0.")
        if self.motion_epsilon < 0.0:
            raise ValueError("motion_epsilon must be >= 0.")
        if self.message_epsilon < 0.0:
            raise ValueError("message_epsilon must be >= 0.")
        if self.swim_assist_start_weight < 0.0:
            raise ValueError("swim_assist_start_weight must be >= 0.")
        if not np.isfinite(self.propulsion_near_limit_weight):
            raise ValueError("propulsion_near_limit_weight must be finite.")
        if not np.isfinite(self.propulsion_saturation_weight):
            raise ValueError("propulsion_saturation_weight must be finite.")
        if not np.isfinite(self.propulsion_torque_weight):
            raise ValueError("propulsion_torque_weight must be finite.")
        if not (0.0 <= float(self.fish_preset.actuation.joint_soft_limit_start_ratio) < 1.0):
            raise ValueError("joint_soft_limit_start_ratio must be in [0, 1).")
        if float(self.fish_preset.actuation.joint_soft_limit_stiffness) < 0.0:
            raise ValueError("joint_soft_limit_stiffness must be >= 0.")
        if float(self.fish_preset.actuation.joint_soft_limit_damping) < 0.0:
            raise ValueError("joint_soft_limit_damping must be >= 0.")
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
                "motion": Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32),
                "message": Discrete(self.num_message_tokens),
            }
        )
        context_low = np.zeros(3, dtype=np.float32)
        context_high = np.ones(3, dtype=np.float32)
        history_feature_count = self._control_history_feature_count()
        history_low = np.full(self.history_length * history_feature_count, -1.0, dtype=np.float32)
        history_high = np.ones(self.history_length * history_feature_count, dtype=np.float32)
        obs_low_parts = [np.zeros(36, dtype=np.float32), context_low, history_low]
        obs_high_parts = [np.ones(36, dtype=np.float32), context_high, history_high]
        if self.teacher_phase_enabled:
            obs_low_parts.append(np.full(2, -1.0, dtype=np.float32))
            obs_high_parts.append(np.ones(2, dtype=np.float32))
        obs_low = np.concatenate(obs_low_parts)
        obs_high = np.concatenate(obs_high_parts)
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
        self.food_active_mask = np.zeros(self.food_count, dtype=bool)
        self.steps_since_last_food = 0
        self.episode_end_reason = "none"
        self.fish_states = {agent_id: self._empty_fish_state() for agent_id in self.possible_agents}
        self.last_message_tokens = {agent_id: 0 for agent_id in self.possible_agents}
        self.last_motion_commands = {agent_id: self._zero_joint_vector() for agent_id in self.possible_agents}
        self.agent_food_eaten_episode = {agent_id: 0 for agent_id in self.possible_agents}
        self.team_food_eaten_episode = {RED_TEAM: 0, BLUE_TEAM: 0}
        self.last_rewards = {agent_id: 0.0 for agent_id in self.possible_agents}
        self.swim_assist_weight = 0.0
        self.swim_assist_state = "off"
        self.set_swim_assist_weight(self.swim_assist_start_weight)
        self.position_progress_history: dict[str, deque[float]] = {}
        self.body_forward_velocity_history: dict[str, deque[float]] = {}
        self.last_reward_breakdown = {
            agent_id: self.get_reward_breakdown(agent_id=agent_id, food_eaten_this_step=0, truncated=False, terminated=False)
            for agent_id in self.possible_agents
        }
        self.last_dynamics_debug = {agent_id: self._empty_dynamics_debug() for agent_id in self.possible_agents}
        self.control_histories = {
            agent_id: np.zeros((self.history_length, history_feature_count), dtype=np.float32)
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
        self.last_joint_limit_high = {agent_id: False for agent_id in self.possible_agents}
        self.last_joints_quiet = {agent_id: False for agent_id in self.possible_agents}
        self.last_negative_forward_velocity = {agent_id: False for agent_id in self.possible_agents}
        self.last_activation_sign_changes_step = {agent_id: self._zero_joint_int_vector() for agent_id in self.possible_agents}
        self.activation_sign_changes_episode = {agent_id: self._zero_joint_int_vector() for agent_id in self.possible_agents}

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
            preset = make_eel_preset(self.num_body_segments)
        elif isinstance(fish_preset, FishPreset):
            preset = fish_preset
        elif isinstance(fish_preset, str):
            preset_name = fish_preset.strip().lower()
            if preset_name in {"eel", "default"}:
                preset = make_eel_preset(self.num_body_segments)
            elif preset_name.startswith("eel_") and preset_name.endswith("seg") and preset_name[4:-3].isdigit():
                preset = make_eel_preset(int(preset_name[4:-3]))
            else:
                raise ValueError(f"Unsupported fish preset: {fish_preset}")
        else:
            raise ValueError(f"Unsupported fish preset: {fish_preset}")
        segment_count = len(preset.morphology.segment_lengths)
        if segment_count != self.num_body_segments:
            raise ValueError(
                f"Fish preset segment count mismatch: requested {self.num_body_segments}, preset provides {segment_count}."
            )
        if len(preset.morphology.segment_front_widths) != segment_count or len(preset.morphology.segment_back_widths) != segment_count:
            raise ValueError("Fish morphology width tuples must match segment count.")
        if len(preset.render.red_segment_colors) != segment_count or len(preset.render.blue_segment_colors) != segment_count:
            raise ValueError("Fish render palettes must match segment count.")
        return preset

    def _zero_joint_vector(self) -> np.ndarray:
        return np.zeros(self.num_joints, dtype=np.float32)

    def _zero_joint_int_vector(self) -> np.ndarray:
        return np.zeros(self.num_joints, dtype=np.int32)

    def _empty_fish_state(self) -> ArticulatedFishState:
        return ArticulatedFishState(
            root_position=np.zeros(2, dtype=np.float32),
            root_velocity=np.zeros(2, dtype=np.float32),
            root_theta=0.0,
            root_omega=0.0,
            joint_angles=self._zero_joint_vector(),
            joint_velocities=self._zero_joint_vector(),
            prev_action=self._zero_joint_vector(),
            prev_message_token=0,
            joint_activation=self._zero_joint_vector(),
            applied_joint_torque=self._zero_joint_vector(),
        )

    def _coerce_joint_vector(self, values: Any, *, default: float = 0.0) -> np.ndarray:
        if values is None:
            return np.full(self.num_joints, float(default), dtype=np.float32)
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if array.size == self.num_joints:
            return array.astype(np.float32, copy=True)
        if array.size == 0:
            return np.full(self.num_joints, float(default), dtype=np.float32)
        raise ValueError(f"Expected {self.num_joints} joint values, got {array.size}.")

    def _default_food_team_indices(self) -> np.ndarray:
        return np.concatenate(
            [
                np.full(self.num_red_pellets, RED_TEAM, dtype=np.int64),
                np.full(self.num_blue_pellets, BLUE_TEAM, dtype=np.int64),
            ]
        )

    def _forage_uses_idle_timeout(self) -> bool:
        return self.reward_mode == "forage" and self.forage_timeout_mode == "reset_on_food"

    def _forage_uses_idle_budget_context(self) -> bool:
        return self.reward_mode == "forage" and self.forage_time_context_mode == "idle_budget_remaining"

    def _remaining_idle_budget_fraction(self) -> float:
        remaining = 1.0 - (float(self.steps_since_last_food) / max(float(self.forage_idle_timeout_steps), 1.0))
        return float(np.clip(remaining, 0.0, 1.0))

    def _time_context_feature(self) -> float:
        if self._forage_uses_idle_budget_context():
            return self._remaining_idle_budget_fraction()
        return float(np.clip(self.timestep / max(float(self.time_limit), 1.0), 0.0, 1.0))

    def _active_food_positions(self) -> np.ndarray:
        if self.food_count == 0 or self.food_active_mask.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return self.food_positions[self.food_active_mask]

    def _active_food_team_indices(self) -> np.ndarray:
        if self.food_count == 0 or self.food_active_mask.size == 0:
            return np.zeros(0, dtype=np.int64)
        return self.food_team_indices[self.food_active_mask]

    def _remaining_food_counts(self) -> dict[str, int]:
        if self.food_count == 0 or self.food_active_mask.size == 0:
            return {"total": 0, "red": 0, "blue": 0}
        active_team_indices = self._active_food_team_indices()
        return {
            "total": int(np.count_nonzero(self.food_active_mask)),
            "red": int(np.count_nonzero(active_team_indices == RED_TEAM)),
            "blue": int(np.count_nonzero(active_team_indices == BLUE_TEAM)),
        }

    def _all_food_depleted(self) -> bool:
        remaining = self._remaining_food_counts()
        return (
            self.reward_mode == "forage"
            and self.food_respawn_mode == "deplete"
            and self.food_count > 0
            and remaining["total"] == 0
        )

    def set_swim_assist_weight(self, weight: float) -> None:
        requested = max(float(weight), 0.0)
        if self.reward_mode != "forage":
            requested = 0.0
        self.swim_assist_weight = requested
        if requested <= 0.0:
            self.swim_assist_state = "off"
        elif requested < max(self.swim_assist_start_weight, 1e-9):
            self.swim_assist_state = "fading"
        else:
            self.swim_assist_state = "on"

    def get_swim_assist_state(self) -> dict[str, float | str]:
        return {
            "swim_assist_state": str(self.swim_assist_state),
            "swim_assist_weight": float(self.swim_assist_weight),
            "swim_assist_start_weight": float(self.swim_assist_start_weight),
        }

    def set_motion_epsilon(self, value: float) -> None:
        self.motion_epsilon = max(float(value), 0.0)
        self.eps = float(self.motion_epsilon)

    def set_message_epsilon(self, value: float) -> None:
        self.message_epsilon = max(float(value), 0.0)

    def set_action_epsilons(self, *, motion_epsilon: float | None = None, message_epsilon: float | None = None) -> None:
        if motion_epsilon is not None:
            self.set_motion_epsilon(float(motion_epsilon))
        if message_epsilon is not None:
            self.set_message_epsilon(float(message_epsilon))

    def get_action_epsilon_state(self) -> dict[str, float]:
        return {
            "motion_epsilon": float(self.motion_epsilon),
            "message_epsilon": float(self.message_epsilon),
        }

    def _normalized_low_level_snapshot(self, state: ArticulatedFishState) -> np.ndarray:
        dynamics = self.fish_preset.dynamics
        actuation = self.fish_preset.actuation
        theta = float(state.root_theta)
        body_velocity = _body_frame(state.root_velocity.astype(np.float32), theta)
        joint_angle_features = np.clip(state.joint_angles.astype(np.float32) / actuation.joint_limit, -1.0, 1.0)
        joint_velocity_features = np.clip(state.joint_velocities.astype(np.float32) / actuation.joint_max_speed, -1.0, 1.0)
        joint_activation_features = np.clip(state.joint_activation.astype(np.float32), -1.0, 1.0)
        if self.observation_profile == "minimal_wave":
            return np.concatenate(
                [
                    np.array(
                        [
                            np.clip(body_velocity[0] / dynamics.max_speed, -1.0, 1.0),
                            np.clip(state.root_omega / dynamics.max_angular_speed, -1.0, 1.0),
                        ],
                        dtype=np.float32,
                    ),
                    joint_angle_features,
                    joint_activation_features,
                ]
            )
        return np.concatenate(
            [
                np.array(
                    [
                        np.clip(body_velocity[0] / dynamics.max_speed, -1.0, 1.0),
                        np.clip(body_velocity[1] / dynamics.max_speed, -1.0, 1.0),
                        np.clip(state.root_omega / dynamics.max_angular_speed, -1.0, 1.0),
                    ],
                    dtype=np.float32,
                ),
                joint_angle_features,
                joint_velocity_features,
                joint_activation_features,
            ]
        )

    def _control_history_feature_count(self) -> int:
        return (3 + (3 * self.num_joints)) if self.observation_profile == "full_v9" else (2 + (2 * self.num_joints))

    def _teacher_phase(self, step_idx: int | None = None) -> float:
        phase_step = self.timestep if step_idx is None else int(step_idx)
        return SCRIPTED_WAVE_PHASE_RATE * float(max(phase_step, 0))

    def _teacher_phase_features(self, step_idx: int | None = None) -> np.ndarray:
        phase = self._teacher_phase(step_idx)
        return np.array([math.sin(phase), math.cos(phase)], dtype=np.float32)

    def _scripted_target_activation(self, step_idx: int | None = None) -> np.ndarray:
        phase = self._teacher_phase(step_idx)
        phase_offsets = np.arange(self.num_joints, dtype=np.float32) * float(SCRIPTED_WAVE_PHASE_DELTA)
        activations = SCRIPTED_WAVE_AMPLITUDE * np.sin(phase - phase_offsets)
        return np.clip(np.asarray(activations, dtype=np.float32), -1.0, 1.0)

    def _debug_forward_progress(self, metrics: dict[str, float]) -> float:
        return float(np.clip(max(float(metrics["forward_velocity"]), 0.0) / SCRIPTED_WAVE_FORWARD_REFERENCE, 0.0, 1.0))

    def _debug_lateral_penalty(self, metrics: dict[str, float]) -> float:
        return float(np.clip(abs(float(metrics["lateral_velocity"])) / SCRIPTED_WAVE_FORWARD_REFERENCE, 0.0, 1.0))

    def _debug_angular_penalty(self, metrics: dict[str, float]) -> float:
        return float(np.clip(abs(float(metrics["angular_velocity"])) / SCRIPTED_WAVE_ANGULAR_REFERENCE, 0.0, 1.0))

    def _teacher_imitation_reward(self, agent_id: str) -> tuple[float, np.ndarray]:
        state = self.fish_states[agent_id]
        target_activation = self._scripted_target_activation(self.timestep - 1)
        imitation_error = float(
            np.mean(
                np.square(
                    np.clip(state.joint_activation.astype(np.float32), -1.0, 1.0) - target_activation
                )
            )
        )
        return float(np.clip(1.0 - imitation_error, 0.0, 1.0)), target_activation.astype(np.float32)

    def _oscillation_bonus(self, agent_id: str) -> float:
        change_map = getattr(self, "last_activation_sign_changes_step", {})
        sign_changes = np.asarray(change_map.get(agent_id, self._zero_joint_int_vector()), dtype=np.float32)
        return float(np.clip(np.sum(sign_changes) / max(float(self.num_joints), 1.0), 0.0, 1.0))

    def _reset_propulsion_history(self, agent_id: str) -> None:
        state = self.fish_states[agent_id]
        body_velocity = _body_frame(state.root_velocity.astype(np.float32), float(state.root_theta))
        self.position_progress_history[agent_id] = deque(
            [float(state.root_position[0])],
            maxlen=PROPULSION_PROGRESS_WINDOW_STEPS + 1,
        )
        self.body_forward_velocity_history[agent_id] = deque(
            [float(body_velocity[0])],
            maxlen=PROPULSION_PROGRESS_WINDOW_STEPS,
        )

    def _append_propulsion_history(self, agent_id: str) -> None:
        state = self.fish_states[agent_id]
        body_velocity = _body_frame(state.root_velocity.astype(np.float32), float(state.root_theta))
        if agent_id not in self.position_progress_history or agent_id not in self.body_forward_velocity_history:
            self._reset_propulsion_history(agent_id)
        self.position_progress_history[agent_id].append(float(state.root_position[0]))
        self.body_forward_velocity_history[agent_id].append(float(body_velocity[0]))

    def _propulsion_saturation_penalty(self, agent_id: str) -> float:
        commands = np.abs(np.asarray(self.last_motion_commands[agent_id], dtype=np.float32))
        return float(np.mean(commands > PROPULSION_SATURATION_THRESHOLD))

    def _propulsion_progress_terms(self, agent_id: str, metrics: dict[str, float]) -> dict[str, float | str]:
        dt = float(self.fish_preset.dynamics.dt)
        window_denominator = max(float(PROPULSION_PROGRESS_WINDOW_STEPS) * dt * SCRIPTED_WAVE_FORWARD_REFERENCE, 1e-6)
        if self.training_phase == "locomotion_propulsion_easy":
            if agent_id not in self.position_progress_history:
                self._reset_propulsion_history(agent_id)
            position_history = self.position_progress_history[agent_id]
            displacement = float(position_history[-1] - position_history[0]) if len(position_history) >= 2 else 0.0
            positive_window_progress = float(np.clip(max(displacement, 0.0) / window_denominator, 0.0, 1.0))
            backward_window_progress = float(np.clip(max(-displacement, 0.0) / window_denominator, 0.0, 1.0))
            positive_forward_speed = float(
                np.clip(max(float(self.fish_states[agent_id].root_velocity[0]), 0.0) / SCRIPTED_WAVE_FORWARD_REFERENCE, 0.0, 1.0)
            )
            progress_frame = "world_x"
        else:
            if agent_id not in self.body_forward_velocity_history:
                self._reset_propulsion_history(agent_id)
            forward_velocity_history = self.body_forward_velocity_history[agent_id]
            mean_body_forward_velocity = float(np.mean(np.asarray(forward_velocity_history, dtype=np.float32)))
            positive_window_progress = float(
                np.clip(max(mean_body_forward_velocity, 0.0) / SCRIPTED_WAVE_FORWARD_REFERENCE, 0.0, 1.0)
            )
            backward_window_progress = float(
                np.clip(max(-mean_body_forward_velocity, 0.0) / SCRIPTED_WAVE_FORWARD_REFERENCE, 0.0, 1.0)
            )
            positive_forward_speed = float(
                np.clip(max(float(metrics["forward_velocity"]), 0.0) / SCRIPTED_WAVE_FORWARD_REFERENCE, 0.0, 1.0)
            )
            progress_frame = "body_forward"
        return {
            "positive_window_progress": float(positive_window_progress),
            "backward_window_progress": float(backward_window_progress),
            "positive_forward_speed": float(positive_forward_speed),
            "progress_frame": progress_frame,
        }

    def _joint_limit_efficiency_terms(self, state: ArticulatedFishState) -> dict[str, float]:
        actuation = self.fish_preset.actuation
        limit = max(float(actuation.joint_limit), 1e-6)
        joint_limit_ratio = np.abs(state.joint_angles.astype(np.float32)) / limit
        excess = np.clip((joint_limit_ratio - 0.65) / max(1.0 - 0.65, 1e-6), 0.0, 1.0).astype(np.float32)
        return {
            "near_limit_penalty": float(np.mean(np.square(excess))),
            "fraction_near_limit_joints": float(np.mean(joint_limit_ratio > 0.70)),
            "mean_joint_limit_excess": float(np.mean(excess)),
        }

    def _soft_limit_torque(self, state: ArticulatedFishState) -> tuple[np.ndarray, np.ndarray]:
        actuation = self.fish_preset.actuation
        limit = max(float(actuation.joint_limit), 1e-6)
        start_ratio = float(actuation.joint_soft_limit_start_ratio)
        limit_ratio = np.abs(state.joint_angles.astype(np.float32)) / limit
        excess = np.clip((limit_ratio - start_ratio) / max(1.0 - start_ratio, 1e-6), 0.0, 1.0).astype(np.float32)
        direction = np.sign(state.joint_angles.astype(np.float32))
        restoring = (
            -float(actuation.joint_soft_limit_stiffness) * np.square(excess) * direction
        ).astype(np.float32)
        outward_velocity = np.maximum(state.joint_velocities.astype(np.float32) * direction, 0.0).astype(np.float32)
        damping = (
            -float(actuation.joint_soft_limit_damping) * np.square(excess) * outward_velocity * direction
        ).astype(np.float32)
        return (restoring + damping).astype(np.float32), excess.astype(np.float32)

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

    def _sign_change_counts(self, previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        prev_sign = np.sign(previous)
        curr_sign = np.sign(current)
        changed = ((prev_sign != 0.0) & (curr_sign != 0.0) & (prev_sign != curr_sign)).astype(np.int32)
        return changed

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
            "desired_joint_activation": self._zero_joint_vector(),
            "joint_activation": self._zero_joint_vector(),
            "active_joint_torque": self._zero_joint_vector(),
            "passive_joint_torque": self._zero_joint_vector(),
            "soft_limit_joint_torque": self._zero_joint_vector(),
            "soft_limit_excess": self._zero_joint_vector(),
            "net_joint_torque": self._zero_joint_vector(),
            "applied_joint_torque": self._zero_joint_vector(),
            "joint_acceleration": self._zero_joint_vector(),
            "joint_limit_ratio": self._zero_joint_vector(),
            "segment_centers": np.zeros((self.num_body_segments, 2), dtype=np.float32),
            "segment_angles": np.zeros(self.num_body_segments, dtype=np.float32),
            "segment_velocities": np.zeros((self.num_body_segments, 2), dtype=np.float32),
            "segment_angular_velocities": np.zeros(self.num_body_segments, dtype=np.float32),
            "segment_drag_forces": np.zeros((self.num_body_segments, 2), dtype=np.float32),
            "segment_drag_torques": np.zeros(self.num_body_segments, dtype=np.float32),
            "joint_positions": np.zeros((self.num_joints, 2), dtype=np.float32),
            "total_force": np.zeros(2, dtype=np.float32),
            "total_torque": 0.0,
            "body_linear_drag_force": np.zeros(2, dtype=np.float32),
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
        self.food_active_mask = np.ones(self.food_count, dtype=bool)
        self.steps_since_last_food = 0
        self.episode_end_reason = "none"

    def _respawn_food_indices(self, indices: np.ndarray) -> None:
        fish_positions = self._all_root_positions()
        for idx in np.flatnonzero(indices):
            self.food_positions[idx] = self._sample_food_position(
                self.food_positions,
                fish_positions,
                exclude_index=int(idx),
            )
            self.food_active_mask[int(idx)] = True

    def _all_root_positions(self) -> np.ndarray:
        return np.asarray(
            [self.fish_states[agent_id].root_position for agent_id in self.possible_agents],
            dtype=np.float32,
        )

    def _mouth_position(self, state: ArticulatedFishState) -> np.ndarray:
        angle = float(state.root_theta)
        heading = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        front_length = float(self.fish_preset.morphology.segment_lengths[0])
        return (state.root_position + (0.5 * front_length * heading)).astype(np.float32)

    def _all_mouth_positions(self) -> np.ndarray:
        return np.asarray([self._mouth_position(self.fish_states[agent_id]) for agent_id in self.possible_agents], dtype=np.float32)

    def _food_relative_vectors(self, agent_id: str, *, edible: bool) -> np.ndarray:
        team_index = self.get_agent_team_index(agent_id)
        active_team_indices = self._active_food_team_indices()
        mask = active_team_indices == team_index if edible else active_team_indices != team_index
        positions = self._active_food_positions()[mask]
        if positions.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return positions - self._mouth_position(self.fish_states[agent_id])

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
        mouth_positions_all = self._all_mouth_positions()
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
        mouth_positions = mouth_positions_all[agent_indices]
        theta = theta_all[agent_indices]
        cos_theta = np.cos(theta).astype(np.float32)
        sin_theta = np.sin(theta).astype(np.float32)
        teams = team_indices_all[agent_indices]

        active_food_positions = self._active_food_positions()
        active_food_team_indices = self._active_food_team_indices()
        if active_food_positions.shape[0] > 0:
            food_relative = active_food_positions[None, :, :] - mouth_positions[:, None, :]
            food_body_x = (cos_theta[:, None] * food_relative[:, :, 0]) + (sin_theta[:, None] * food_relative[:, :, 1])
            food_body_y = (-sin_theta[:, None] * food_relative[:, :, 0]) + (cos_theta[:, None] * food_relative[:, :, 1])
            food_distance_sq = (food_body_x * food_body_x) + (food_body_y * food_body_y)
            food_distance = np.sqrt(food_distance_sq, out=np.zeros_like(food_distance_sq))
            food_sector_indices = self._sector_indices_from_body_coords(food_body_x, food_body_y)
            food_visible_mask = food_distance <= float(self.sector_radius)
            edible_food_mask = active_food_team_indices[None, :] == teams[:, None]
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
                    self._time_context_feature(),
                ],
                dtype=np.float32,
            )
            obs_parts = [
                edible_food_bins[row_idx],
                non_edible_food_bins[row_idx],
                teammate_bins[row_idx],
                opponent_bins[row_idx],
                teammate_message_bins[row_idx],
                opponent_message_bins[row_idx],
                context,
                self.control_histories[agent_id].reshape(-1),
            ]
            if self.teacher_phase_enabled:
                obs_parts.append(self._teacher_phase_features())
            observations[agent_id] = np.concatenate(obs_parts).astype(np.float32, copy=False)
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
        cumulative_joint_angles = np.cumsum(state.joint_angles.astype(np.float32), dtype=np.float32)
        return np.concatenate(
            [
                np.array([float(state.root_theta)], dtype=np.float32),
                (float(state.root_theta) + cumulative_joint_angles).astype(np.float32),
            ]
        )

    def _segment_geometry(self, state: ArticulatedFishState) -> Dict[str, np.ndarray]:
        morphology = self.fish_preset.morphology
        angles = self._segment_angles(state)
        lengths = np.asarray(morphology.segment_lengths, dtype=np.float32)
        centers = np.zeros((self.num_body_segments, 2), dtype=np.float32)
        joint_positions = np.zeros((self.num_joints, 2), dtype=np.float32)
        centers[0] = state.root_position.astype(np.float32)

        headings = np.asarray(
            [[math.cos(float(angle)), math.sin(float(angle))] for angle in angles],
            dtype=np.float32,
        )
        mouth_position = (centers[0] + (0.5 * lengths[0] * headings[0])).astype(np.float32)
        trailing_joint = centers[0] - (0.5 * lengths[0] * headings[0])
        if self.num_joints > 0:
            joint_positions[0] = trailing_joint

        for segment_idx in range(1, self.num_body_segments):
            centers[segment_idx] = trailing_joint - (0.5 * lengths[segment_idx] * headings[segment_idx])
            if segment_idx < self.num_joints:
                trailing_joint = trailing_joint - (lengths[segment_idx] * headings[segment_idx])
                joint_positions[segment_idx] = trailing_joint

        return {
            "centers": centers,
            "angles": angles,
            "joint_positions": joint_positions,
            "mouth_position": mouth_position,
        }

    def _segment_angular_velocities(self, state: ArticulatedFishState) -> np.ndarray:
        cumulative_joint_velocities = np.cumsum(state.joint_velocities.astype(np.float32), dtype=np.float32)
        return np.concatenate(
            [
                np.array([float(state.root_omega)], dtype=np.float32),
                (float(state.root_omega) + cumulative_joint_velocities).astype(np.float32),
            ]
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

        segment_velocities = np.zeros((self.num_body_segments, 2), dtype=np.float32)
        segment_velocities[0] = state.root_velocity.astype(np.float32)
        trailing_joint_velocity = segment_velocities[0] - (0.5 * lengths[0] * float(omega_segments[0]) * normals[0])
        for segment_idx in range(1, self.num_body_segments):
            segment_velocities[segment_idx] = trailing_joint_velocity - (
                0.5 * lengths[segment_idx] * float(omega_segments[segment_idx]) * normals[segment_idx]
            )
            if segment_idx < self.num_joints:
                trailing_joint_velocity = trailing_joint_velocity - (
                    lengths[segment_idx] * float(omega_segments[segment_idx]) * normals[segment_idx]
                )

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

        segment_drag_forces = np.zeros((self.num_body_segments, 2), dtype=np.float32) if include_debug else None
        segment_drag_torques = np.zeros(self.num_body_segments, dtype=np.float32) if include_debug else None
        total_force = np.zeros(2, dtype=np.float32)
        total_torque = 0.0

        lengths = np.asarray(morphology.segment_lengths, dtype=np.float32)
        areas = lengths * 0.5 * (
            np.asarray(morphology.segment_front_widths, dtype=np.float32)
            + np.asarray(morphology.segment_back_widths, dtype=np.float32)
        )

        for idx in range(self.num_body_segments):
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
        body_linear_drag_force = (-float(dynamics.body_linear_drag) * state.root_velocity.astype(np.float32)).astype(np.float32)
        total_force += body_linear_drag_force
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
            "body_linear_drag_force": body_linear_drag_force.astype(np.float32),
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
        desired_activation = np.clip(_as_float32_array(action, shape=(self.num_joints,)), -1.0, 1.0).astype(np.float32)
        alpha = float(dt / (self.activation_time_constant + dt)) if self.activation_time_constant > 0.0 else 1.0
        activation_next = (
            state.joint_activation + alpha * (desired_activation - state.joint_activation)
        ).astype(np.float32)
        active_torque = (float(actuation.joint_torque_limit) * activation_next).astype(np.float32)
        passive_torque = (
            (-float(actuation.joint_passive_stiffness) * state.joint_angles.astype(np.float32))
            - (float(actuation.joint_passive_damping) * state.joint_velocities.astype(np.float32))
        ).astype(np.float32)
        soft_limit_torque, soft_limit_excess = self._soft_limit_torque(state)
        net_torque = (active_torque + passive_torque + soft_limit_torque).astype(np.float32)
        limit = float(actuation.joint_limit)
        blocked_positive = (state.joint_angles >= limit) & (net_torque > 0.0)
        blocked_negative = (state.joint_angles <= -limit) & (net_torque < 0.0)
        blocked_mask = blocked_positive | blocked_negative
        net_torque[blocked_mask] = 0.0
        joint_acceleration = net_torque / float(actuation.joint_inertia)
        joint_velocity_next = self._clip_joint_velocities(
            state.joint_velocities + joint_acceleration.astype(np.float32) * float(dt)
        )
        joint_angle_next, joint_velocity_next = self._clamp_joint_state(
            state.joint_angles + joint_velocity_next * float(dt),
            joint_velocity_next,
        )
        result = {
            "desired_joint_activation": desired_activation.astype(np.float32),
            "joint_activation": activation_next.astype(np.float32),
            "active_joint_torque": active_torque.astype(np.float32),
            "passive_joint_torque": passive_torque.astype(np.float32),
            "soft_limit_joint_torque": soft_limit_torque.astype(np.float32),
            "soft_limit_excess": soft_limit_excess.astype(np.float32),
            "net_joint_torque": net_torque.astype(np.float32),
            "applied_joint_torque": active_torque.astype(np.float32),
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
            joint_activation=np.asarray(actuation["joint_activation"], dtype=np.float32),
            applied_joint_torque=np.asarray(actuation["applied_joint_torque"], dtype=np.float32),
        )
        wrench = self._compute_external_wrench(predicted_state, include_debug=False)
        dynamics = self.fish_preset.dynamics
        return {
            "desired_joint_activation": np.asarray(actuation["desired_joint_activation"], dtype=np.float32),
            "joint_activation": np.asarray(actuation["joint_activation"], dtype=np.float32),
            "active_joint_torque": np.asarray(actuation["active_joint_torque"], dtype=np.float32),
            "passive_joint_torque": np.asarray(actuation["passive_joint_torque"], dtype=np.float32),
            "soft_limit_joint_torque": np.asarray(actuation["soft_limit_joint_torque"], dtype=np.float32),
            "soft_limit_excess": np.asarray(actuation["soft_limit_excess"], dtype=np.float32),
            "net_joint_torque": np.asarray(actuation["net_joint_torque"], dtype=np.float32),
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
        action = np.clip(_as_float32_array(action, shape=(self.num_joints,)), -1.0, 1.0)
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
            joint_activation=actuation["joint_activation"].astype(np.float32).copy(),
            applied_joint_torque=actuation["applied_joint_torque"].astype(np.float32).copy(),
        )
        wrench = self._compute_external_wrench(predicted_state, include_debug=True)
        dynamics = self.fish_preset.dynamics
        total_force = wrench["total_force"].astype(np.float32)
        total_torque = float(wrench["total_torque"])
        root_acceleration = total_force / float(dynamics.mass)
        root_angular_acceleration = float(total_torque / float(dynamics.inertia))
        return {
            "desired_joint_activation": actuation["desired_joint_activation"].astype(np.float32),
            "joint_activation": actuation["joint_activation"].astype(np.float32),
            "active_joint_torque": actuation["active_joint_torque"].astype(np.float32),
            "passive_joint_torque": actuation["passive_joint_torque"].astype(np.float32),
            "soft_limit_joint_torque": actuation["soft_limit_joint_torque"].astype(np.float32),
            "soft_limit_excess": actuation["soft_limit_excess"].astype(np.float32),
            "net_joint_torque": actuation["net_joint_torque"].astype(np.float32),
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
            "body_linear_drag_force": wrench["body_linear_drag_force"].astype(np.float32),
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
            prev_action=np.clip(_as_float32_array(action, shape=(self.num_joints,)), -1.0, 1.0).astype(np.float32),
            prev_message_token=int(prev_message_token),
            joint_activation=np.asarray(dynamics_step["joint_activation"], dtype=np.float32).copy(),
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
            prev_action=np.clip(_as_float32_array(motion_action, shape=(self.num_joints,)), -1.0, 1.0).astype(np.float32),
            prev_message_token=int(message_token),
            joint_activation=self.fish_states[agent_id].joint_activation.astype(np.float32).copy(),
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
        terminated: bool = False,
        episode_end_reason: str = "none",
    ) -> Dict[str, float | bool | int | str]:
        metrics = self._current_motion_metrics(agent_id)
        pellet_reward_total = float(self.pellet_reward * food_eaten_this_step) if self.reward_mode == "forage" else 0.0
        locomotion_debug_terms = self._locomotion_debug_reward(agent_id, metrics)
        locomotion_assist_raw = self._swim_assist_reward(metrics)
        if self.reward_mode == "forage":
            forage_reward = float(pellet_reward_total - self.step_cost)
            locomotion_assist_applied = float(self.swim_assist_weight * locomotion_assist_raw)
            total_reward = float(forage_reward + locomotion_assist_applied)
        else:
            forage_reward = 0.0
            locomotion_assist_raw = float(locomotion_debug_terms["locomotion_debug_reward"])
            locomotion_assist_applied = float(locomotion_assist_raw)
            total_reward = float(locomotion_assist_raw)
        return {
            "agent_id": agent_id,
            "agent_team": self.get_agent_team_name(agent_id),
            "reward_mode": self.reward_mode,
            "training_phase": self.training_phase,
            "swim_assist_state": str(self.swim_assist_state),
            "swim_assist_weight": float(self.swim_assist_weight),
            "food_eaten_this_step": int(food_eaten_this_step),
            "pellet_reward_total": pellet_reward_total,
            "step_cost": float(self.step_cost if self.reward_mode == "forage" else 0.0),
            "forage_reward": float(forage_reward),
            "forward_velocity": float(metrics["forward_velocity"]),
            "lateral_velocity": float(metrics["lateral_velocity"]),
            "angular_velocity": float(metrics["angular_velocity"]),
            "forward_velocity_norm": float(metrics["forward_velocity_norm"]),
            "lateral_velocity_norm": float(metrics["lateral_velocity_norm"]),
            "angular_velocity_norm": float(metrics["angular_velocity_norm"]),
            "mean_abs_desired_activation": float(metrics["mean_abs_desired_activation"]),
            "mean_abs_activation": float(metrics["mean_abs_activation"]),
            "mean_abs_applied_torque": float(metrics["mean_abs_applied_torque"]),
            "mean_abs_applied_torque_norm": float(metrics["mean_abs_applied_torque_norm"]),
            "mean_joint_limit_ratio": float(metrics["mean_joint_limit_ratio"]),
            "mean_joint_limit_ratio_sq": float(metrics["mean_joint_limit_ratio_sq"]),
            "near_limit_penalty": float(metrics["near_limit_penalty"]),
            "fraction_near_limit_joints": float(metrics["fraction_near_limit_joints"]),
            "mean_joint_limit_excess": float(metrics["mean_joint_limit_excess"]),
            "joint_limit_high": bool(metrics["joint_limit_high"]),
            "joints_quiet": bool(metrics["joints_quiet"]),
            "negative_forward_velocity": bool(metrics["negative_forward_velocity"]),
            "locomotion_assist_raw": float(locomotion_assist_raw),
            "locomotion_assist_applied": float(locomotion_assist_applied),
            "locomotion_forward_progress": float(locomotion_debug_terms["forward_progress"]),
            "locomotion_backward_progress": float(locomotion_debug_terms["backward_progress"]),
            "locomotion_positive_forward_speed": float(locomotion_debug_terms["positive_forward_speed"]),
            "locomotion_lateral_penalty": float(locomotion_debug_terms["lateral_penalty"]),
            "locomotion_angular_penalty": float(locomotion_debug_terms["angular_penalty"]),
            "locomotion_joint_limit_penalty": float(locomotion_debug_terms["joint_limit_penalty"]),
            "locomotion_near_limit_penalty": float(locomotion_debug_terms["near_limit_penalty"]),
            "locomotion_fraction_near_limit_joints": float(locomotion_debug_terms["fraction_near_limit_joints"]),
            "locomotion_mean_joint_limit_excess": float(locomotion_debug_terms["mean_joint_limit_excess"]),
            "locomotion_saturation_penalty": float(locomotion_debug_terms["saturation_penalty"]),
            "locomotion_torque_penalty": float(locomotion_debug_terms["torque_penalty"]),
            "locomotion_teacher_imitation_reward": float(locomotion_debug_terms["teacher_imitation_reward"]),
            "locomotion_oscillation_bonus": float(locomotion_debug_terms["oscillation_bonus"]),
            "locomotion_progress_frame": str(locomotion_debug_terms["progress_frame"]),
            "locomotion_phase_sin": float(locomotion_debug_terms["phase_features"][0]),
            "locomotion_phase_cos": float(locomotion_debug_terms["phase_features"][1]),
            "locomotion_target_activation_0": float(
                locomotion_debug_terms["target_activation"][0]
                if locomotion_debug_terms["target_activation"].size > 0
                else 0.0
            ),
            "locomotion_target_activation_1": float(
                locomotion_debug_terms["target_activation"][1]
                if locomotion_debug_terms["target_activation"].size > 1
                else 0.0
            ),
            "total_reward": total_reward,
            "steps_since_last_food": int(self.steps_since_last_food),
            "forage_idle_timeout_steps": int(self.forage_idle_timeout_steps),
            "remaining_idle_budget_fraction": float(self._remaining_idle_budget_fraction()),
            "remaining_food_total": int(self._remaining_food_counts()["total"]),
            "remaining_food_red": int(self._remaining_food_counts()["red"]),
            "remaining_food_blue": int(self._remaining_food_counts()["blue"]),
            "food_respawn_mode": str(self.food_respawn_mode),
            "forage_timeout_mode": str(self.forage_timeout_mode),
            "episode_end_reason": str(episode_end_reason),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    def _current_motion_metrics(self, agent_id: str) -> dict[str, float]:
        state = self.fish_states[agent_id]
        dynamics = self.fish_preset.dynamics
        actuation = self.fish_preset.actuation
        body_velocity = _body_frame(state.root_velocity.astype(np.float32), float(state.root_theta))
        joint_limit_ratio = np.abs(state.joint_angles.astype(np.float32)) / max(float(actuation.joint_limit), 1e-6)
        limit_efficiency = self._joint_limit_efficiency_terms(state)
        return {
            "forward_velocity": float(body_velocity[0]),
            "lateral_velocity": float(body_velocity[1]),
            "angular_velocity": float(state.root_omega),
            "forward_velocity_norm": float(np.clip(body_velocity[0] / dynamics.max_speed, -1.0, 1.0)),
            "lateral_velocity_norm": float(np.clip(body_velocity[1] / dynamics.max_speed, -1.0, 1.0)),
            "angular_velocity_norm": float(np.clip(state.root_omega / dynamics.max_angular_speed, -1.0, 1.0)),
            "mean_abs_desired_activation": float(np.mean(np.abs(self.last_motion_commands[agent_id]))),
            "mean_abs_activation": float(np.mean(np.abs(state.joint_activation))),
            "mean_abs_applied_torque": float(np.mean(np.abs(state.applied_joint_torque))),
            "mean_abs_applied_torque_norm": float(
                np.mean(np.abs(state.applied_joint_torque)) / max(float(actuation.joint_torque_limit), 1e-6)
            ),
            "mean_joint_limit_ratio": float(np.mean(joint_limit_ratio)),
            "mean_joint_limit_ratio_sq": float(np.mean(np.square(joint_limit_ratio))),
            "near_limit_penalty": float(limit_efficiency["near_limit_penalty"]),
            "fraction_near_limit_joints": float(limit_efficiency["fraction_near_limit_joints"]),
            "mean_joint_limit_excess": float(limit_efficiency["mean_joint_limit_excess"]),
            "joint_limit_high": bool(float(np.mean(joint_limit_ratio)) > 0.75),
            "joints_quiet": bool(np.max(np.abs(state.joint_velocities.astype(np.float32))) < 0.1),
            "negative_forward_velocity": bool(body_velocity[0] < 0.0),
        }

    def _swim_assist_reward(self, metrics: dict[str, float]) -> float:
        forward_term = 0.70 * max(float(metrics["forward_velocity_norm"]), 0.0)
        lateral_term = -0.15 * abs(float(metrics["lateral_velocity_norm"]))
        angular_term = -0.15 * abs(float(metrics["angular_velocity_norm"]))
        torque_term = -0.01 * float(metrics["mean_abs_applied_torque_norm"])
        joint_limit_term = -0.02 * float(metrics["mean_joint_limit_ratio_sq"])
        return float(forward_term + lateral_term + angular_term + torque_term + joint_limit_term)

    def _locomotion_debug_reward(self, agent_id: str, metrics: dict[str, float]) -> dict[str, Any]:
        forward_progress = self._debug_forward_progress(metrics)
        lateral_penalty = self._debug_lateral_penalty(metrics)
        angular_penalty = self._debug_angular_penalty(metrics)
        joint_limit_penalty = float(metrics["mean_joint_limit_ratio_sq"])
        near_limit_penalty = float(metrics["near_limit_penalty"])
        fraction_near_limit_joints = float(metrics["fraction_near_limit_joints"])
        mean_joint_limit_excess = float(metrics["mean_joint_limit_excess"])
        phase_features = self._teacher_phase_features(self.timestep - 1)
        backward_progress = 0.0
        positive_forward_speed = float(forward_progress)
        saturation_penalty = self._propulsion_saturation_penalty(agent_id)
        torque_penalty = float(np.clip(metrics["mean_abs_applied_torque_norm"], 0.0, 1.0))
        progress_frame = "body_forward"
        if self.training_phase == "locomotion_teacher":
            teacher_imitation_reward, target_activation = self._teacher_imitation_reward(agent_id)
            oscillation_bonus = 0.0
            locomotion_debug_reward = float(
                (0.70 * teacher_imitation_reward)
                + (0.25 * forward_progress)
                - (0.025 * lateral_penalty)
                - (0.025 * angular_penalty)
                - (0.02 * joint_limit_penalty)
            )
        elif self.training_phase == "locomotion_self":
            teacher_imitation_reward = 0.0
            target_activation = self._zero_joint_vector()
            oscillation_bonus = self._oscillation_bonus(agent_id)
            locomotion_debug_reward = float(
                (0.80 * forward_progress)
                + (0.20 * oscillation_bonus)
                - (0.05 * lateral_penalty)
                - (0.05 * angular_penalty)
                - (0.02 * joint_limit_penalty)
            )
        else:
            phase_features = np.zeros(2, dtype=np.float32)
            propulsion_terms = self._propulsion_progress_terms(agent_id, metrics)
            forward_progress = float(propulsion_terms["positive_window_progress"])
            backward_progress = float(propulsion_terms["backward_window_progress"])
            positive_forward_speed = float(propulsion_terms["positive_forward_speed"])
            progress_frame = str(propulsion_terms["progress_frame"])
            teacher_imitation_reward = 0.0
            target_activation = self._zero_joint_vector()
            oscillation_bonus = 0.0
            locomotion_debug_reward = float(
                (0.55 * forward_progress)
                + (0.20 * positive_forward_speed)
                - (0.70 * backward_progress)
                - (0.08 * lateral_penalty)
                - (0.08 * angular_penalty)
                + (self.propulsion_near_limit_weight * near_limit_penalty)
                + (self.propulsion_saturation_weight * saturation_penalty)
                + (self.propulsion_torque_weight * torque_penalty)
            )
        return {
            "locomotion_debug_reward": float(locomotion_debug_reward),
            "forward_progress": float(forward_progress),
            "backward_progress": float(backward_progress),
            "positive_forward_speed": float(positive_forward_speed),
            "lateral_penalty": float(lateral_penalty),
            "angular_penalty": float(angular_penalty),
            "joint_limit_penalty": float(joint_limit_penalty),
            "near_limit_penalty": float(near_limit_penalty),
            "fraction_near_limit_joints": float(fraction_near_limit_joints),
            "mean_joint_limit_excess": float(mean_joint_limit_excess),
            "saturation_penalty": float(saturation_penalty),
            "torque_penalty": float(torque_penalty),
            "teacher_imitation_reward": float(teacher_imitation_reward),
            "oscillation_bonus": float(oscillation_bonus),
            "target_activation": np.asarray(target_activation, dtype=np.float32),
            "phase_features": np.asarray(phase_features, dtype=np.float32),
            "progress_frame": progress_frame,
        }

    def _spawn_school_states(self) -> None:
        angle_offset = float(self.np_random.uniform(-math.pi, math.pi))
        sector_angle = math.pi / float(max(self.num_fish, 2))
        min_required_radius = float(self.min_spawn_separation / max(2.0 * math.sin(sector_angle), 1e-6))
        spawn_radius = float(max(self.spawn_ring_radius, min_required_radius))
        zeroed_locomotion_reset = self.training_phase in {
            "locomotion_teacher",
            "locomotion_self",
            "locomotion_propulsion_easy",
            "locomotion_propulsion_robust",
        }
        self.fish_states = {}
        self.last_message_tokens = {agent_id: 0 for agent_id in self.possible_agents}
        self.last_motion_commands = {agent_id: self._zero_joint_vector() for agent_id in self.possible_agents}
        self.position_progress_history = {}
        self.body_forward_velocity_history = {}
        for idx, agent_id in enumerate(self.possible_agents):
            angle = angle_offset + ((2.0 * math.pi * idx) / float(self.num_fish))
            position = np.array(
                [spawn_radius * math.cos(angle), spawn_radius * math.sin(angle)],
                dtype=np.float32,
            )
            theta = float(self.np_random.uniform(-math.pi, math.pi))
            if zeroed_locomotion_reset:
                position = np.zeros(2, dtype=np.float32)
                if self.training_phase == "locomotion_propulsion_robust":
                    theta = float(self.np_random.uniform(-math.pi, math.pi))
                else:
                    theta = 0.0
                joint_angles = self._zero_joint_vector()
                joint_velocities = self._zero_joint_vector()
                joint_activation = self._zero_joint_vector()
            elif self.reward_mode == "locomotion_debug":
                joint_angles = self.np_random.uniform(-0.18, 0.18, size=self.num_joints).astype(np.float32)
                joint_velocities = self.np_random.uniform(-0.65, 0.65, size=self.num_joints).astype(np.float32)
                joint_activation = self.np_random.uniform(-0.25, 0.25, size=self.num_joints).astype(np.float32)
            else:
                joint_angles = self._zero_joint_vector()
                joint_velocities = self._zero_joint_vector()
                joint_activation = self._zero_joint_vector()
            self.fish_states[agent_id] = ArticulatedFishState(
                root_position=position,
                root_velocity=np.zeros(2, dtype=np.float32),
                root_theta=theta,
                root_omega=0.0,
                joint_angles=joint_angles,
                joint_velocities=joint_velocities,
                prev_action=self._zero_joint_vector(),
                prev_message_token=0,
                joint_activation=joint_activation,
                applied_joint_torque=self._zero_joint_vector(),
            )
            self._initialize_control_history(agent_id)
            self._reset_propulsion_history(agent_id)
            self.last_capture_distance[agent_id] = float("nan")
            self.last_joint_limit_occupancy[agent_id] = 0.0
            self.last_joint_zero_crossings[agent_id] = 0
            self.last_joint_limit_high[agent_id] = False
            self.last_joints_quiet[agent_id] = True
            self.last_negative_forward_velocity[agent_id] = False
            self.last_activation_sign_changes_step[agent_id] = self._zero_joint_int_vector()
            self.activation_sign_changes_episode[agent_id] = self._zero_joint_int_vector()

    def _compute_agent_obs(self, agent_id: str) -> np.ndarray:
        return self._compute_observation_bundle([agent_id])[agent_id]

    def _get_obs_dict(self) -> dict[str, np.ndarray]:
        return self._compute_observation_bundle(self.possible_agents)

    def _build_info_dict(
        self,
        *,
        truncated: bool,
        terminated: bool,
        food_eaten_by_agent: dict[str, int] | None = None,
    ) -> dict[str, dict[str, Any]]:
        food_eaten_by_agent = food_eaten_by_agent or {agent_id: 0 for agent_id in self.possible_agents}
        infos: dict[str, dict[str, Any]] = {}
        remaining_food = self._remaining_food_counts()
        for agent_id in self.possible_agents:
            visible_counts = self.last_visible_counts[agent_id]
            metrics = self._current_motion_metrics(agent_id)
            infos[agent_id] = {
                "agent_id": agent_id,
                "agent_team": self.get_agent_team_name(agent_id),
                "reward_mode": self.reward_mode,
                "swim_assist_state": str(self.swim_assist_state),
                "swim_assist_weight": float(self.swim_assist_weight),
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
                "steps_since_last_food": int(self.steps_since_last_food),
                "forage_idle_timeout_steps": int(self.forage_idle_timeout_steps),
                "remaining_idle_budget_fraction": float(self._remaining_idle_budget_fraction()),
                "remaining_food_total": int(remaining_food["total"]),
                "remaining_food_red": int(remaining_food["red"]),
                "remaining_food_blue": int(remaining_food["blue"]),
                "food_respawn_mode": str(self.food_respawn_mode),
                "forage_timeout_mode": str(self.forage_timeout_mode),
                "episode_end_reason": str(self.episode_end_reason),
                "forward_velocity": float(metrics["forward_velocity"]),
                "lateral_velocity": float(metrics["lateral_velocity"]),
                "angular_velocity": float(metrics["angular_velocity"]),
                "forward_velocity_norm": float(metrics["forward_velocity_norm"]),
                "lateral_velocity_norm": float(metrics["lateral_velocity_norm"]),
                "angular_velocity_norm": float(metrics["angular_velocity_norm"]),
                "motion_command": self.last_motion_commands[agent_id].astype(np.float32).copy(),
                "joint_activation_vector": self.fish_states[agent_id].joint_activation.astype(np.float32).copy(),
                "mean_abs_desired_activation": float(metrics["mean_abs_desired_activation"]),
                "mean_abs_activation": float(metrics["mean_abs_activation"]),
                "mean_abs_applied_torque": float(metrics["mean_abs_applied_torque"]),
                "mean_abs_applied_torque_norm": float(metrics["mean_abs_applied_torque_norm"]),
                "mean_joint_limit_ratio": float(metrics["mean_joint_limit_ratio"]),
                "near_limit_penalty": float(metrics["near_limit_penalty"]),
                "fraction_near_limit_joints": float(metrics["fraction_near_limit_joints"]),
                "mean_joint_limit_excess": float(metrics["mean_joint_limit_excess"]),
                "joint_limit_high": bool(metrics["joint_limit_high"]),
                "joints_quiet": bool(metrics["joints_quiet"]),
                "negative_forward_velocity": bool(metrics["negative_forward_velocity"]),
                "locomotion_saturation_penalty": float(self.last_reward_breakdown[agent_id].get("locomotion_saturation_penalty", 0.0)),
                "locomotion_torque_penalty": float(self.last_reward_breakdown[agent_id].get("locomotion_torque_penalty", 0.0)),
                "joint_velocity_zero_crossings": int(self.last_joint_zero_crossings[agent_id]),
                "activation_sign_changes_this_step": int(np.sum(self.last_activation_sign_changes_step[agent_id])),
                "activation_sign_changes_episode": int(np.sum(self.activation_sign_changes_episode[agent_id])),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "fish_preset": self.fish_preset.name,
            }
        return infos

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        self.agents = list(self.possible_agents)
        self.timestep = 0
        self.set_swim_assist_weight(self.swim_assist_weight if self.reward_mode == "forage" else 0.0)
        self.agent_food_eaten_episode = {agent_id: 0 for agent_id in self.possible_agents}
        self.team_food_eaten_episode = {RED_TEAM: 0, BLUE_TEAM: 0}
        self._spawn_school_states()
        self._spawn_food_field()
        self.last_rewards = {agent_id: 0.0 for agent_id in self.possible_agents}
        self.last_motion_commands = {agent_id: self._zero_joint_vector() for agent_id in self.possible_agents}
        self.last_reward_breakdown = {
            agent_id: self.get_reward_breakdown(agent_id=agent_id, food_eaten_this_step=0, truncated=False, terminated=False)
            for agent_id in self.possible_agents
        }
        self.last_dynamics_debug = {agent_id: self._empty_dynamics_debug() for agent_id in self.possible_agents}
        obs = self._get_obs_dict()
        infos = self._build_info_dict(truncated=False, terminated=False)
        return obs, infos

    def _normalize_action(self, action: Any) -> dict[str, Any]:
        motion = self._zero_joint_vector()
        message = 0
        if isinstance(action, dict):
            motion = np.clip(self._coerce_joint_vector(action.get("motion", self._zero_joint_vector())), -1.0, 1.0)
            message = _normalize_message_token(action.get("message", 0), num_tokens=self.num_message_tokens)
        else:
            array = np.asarray(action, dtype=np.float32).reshape(-1)
            if array.size >= self.num_joints:
                motion = np.clip(array[: self.num_joints], -1.0, 1.0).astype(np.float32)
            if array.size >= self.num_joints + 1:
                message = _normalize_message_token(array[self.num_joints], num_tokens=self.num_message_tokens)
        if self.motion_epsilon > 0.0:
            if bool(self.np_random.random() < self.motion_epsilon):
                motion = self.np_random.uniform(-1.0, 1.0, size=self.num_joints).astype(np.float32)
        if self.message_epsilon > 0.0:
            if bool(self.np_random.random() < self.message_epsilon):
                message = int(self.np_random.integers(0, self.num_message_tokens))
        return {"motion": motion.astype(np.float32), "message": int(message)}

    def _resolve_step_actions(self, action_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
        resolved: dict[str, dict[str, Any]] = {}
        for agent_id in self.possible_agents:
            resolved[agent_id] = self._normalize_action(
                action_dict.get(agent_id, {"motion": self._zero_joint_vector(), "message": 0})
            )
        return resolved

    def _food_capture_assignments(self) -> tuple[dict[str, int], np.ndarray, dict[str, float]]:
        active_food_indices = np.flatnonzero(self.food_active_mask)
        if self.food_count == 0 or active_food_indices.size == 0:
            return {agent_id: 0 for agent_id in self.possible_agents}, np.zeros(0, dtype=bool), {
                agent_id: float("nan") for agent_id in self.possible_agents
            }
        fish_positions = self._all_mouth_positions()
        active_food_positions = self.food_positions[active_food_indices]
        active_food_team_indices = self.food_team_indices[active_food_indices]
        distances = np.linalg.norm(
            fish_positions[:, None, :] - active_food_positions[None, :, :],
            axis=2,
        )
        captured_mask = np.zeros(self.food_count, dtype=bool)
        food_eaten_by_agent = {agent_id: 0 for agent_id in self.possible_agents}
        capture_distances: dict[str, list[float]] = {agent_id: [] for agent_id in self.possible_agents}
        for active_idx, food_index in enumerate(active_food_indices):
            food_team = int(active_food_team_indices[active_idx])
            capturers = [
                idx
                for idx, agent_id in enumerate(self.possible_agents)
                if self.get_agent_team_index(agent_id) == food_team and distances[idx, active_idx] <= self.food_capture_radius
            ]
            if not capturers:
                continue
            best_idx = int(min(capturers, key=lambda idx: float(distances[idx, active_idx])))
            captured_mask[int(food_index)] = True
            best_agent_id = self.possible_agents[best_idx]
            food_eaten_by_agent[best_agent_id] += 1
            capture_distances[best_agent_id].append(float(distances[best_idx, active_idx]))
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
        previous_joint_activations = {
            agent_id: self.fish_states[agent_id].joint_activation.astype(np.float32).copy()
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
        self.last_motion_commands = {
            agent_id: np.asarray(action["motion"], dtype=np.float32).copy()
            for agent_id, action in actions.items()
        }
        self.last_message_tokens = {agent_id: int(action["message"]) for agent_id, action in actions.items()}
        self.timestep += 1
        actuation = self.fish_preset.actuation
        for agent_id in self.possible_agents:
            self.last_joint_zero_crossings[agent_id] = self._joint_zero_crossings(
                previous_joint_velocities[agent_id],
                self.fish_states[agent_id].joint_velocities,
            )
            activation_sign_changes = self._sign_change_counts(
                previous_joint_activations[agent_id],
                self.fish_states[agent_id].joint_activation,
            )
            self.last_activation_sign_changes_step[agent_id] = activation_sign_changes.astype(np.int32)
            self.activation_sign_changes_episode[agent_id] = (
                self.activation_sign_changes_episode[agent_id] + activation_sign_changes.astype(np.int32)
            )
            self.last_joint_limit_occupancy[agent_id] = float(
                np.mean(np.abs(self.fish_states[agent_id].joint_angles) / max(float(actuation.joint_limit), 1e-6))
            )
            self.last_joint_limit_high[agent_id] = bool(self.last_joint_limit_occupancy[agent_id] > 0.75)
            self.last_joints_quiet[agent_id] = bool(
                np.max(np.abs(self.fish_states[agent_id].joint_velocities.astype(np.float32))) < 0.1
            )
            self.last_negative_forward_velocity[agent_id] = bool(self._current_motion_metrics(agent_id)["forward_velocity"] < 0.0)
            self._append_control_history(agent_id)
            self._append_propulsion_history(agent_id)

        food_eaten_by_agent, eaten_mask, capture_distances = self._food_capture_assignments()
        self.last_capture_distance = capture_distances
        if np.any(eaten_mask):
            for agent_id, eaten_count in food_eaten_by_agent.items():
                self.agent_food_eaten_episode[agent_id] += int(eaten_count)
                self.team_food_eaten_episode[self.get_agent_team_index(agent_id)] += int(eaten_count)
            if self.reward_mode == "forage" and self.food_respawn_mode == "deplete":
                self.food_active_mask[eaten_mask] = False
            else:
                self._respawn_food_indices(eaten_mask)
            self.steps_since_last_food = 0
        elif self._forage_uses_idle_timeout():
            self.steps_since_last_food += 1

        terminated = self._all_food_depleted()
        if terminated:
            truncated = False
            self.episode_end_reason = "food_depleted"
        elif self._forage_uses_idle_timeout():
            truncated = bool(self.steps_since_last_food >= self.forage_idle_timeout_steps)
            self.episode_end_reason = "idle_timeout" if truncated else "none"
        else:
            truncated = bool(self.timestep >= self.time_limit)
            self.episode_end_reason = "time_limit" if truncated else "none"
        self.last_reward_breakdown = {
            agent_id: self.get_reward_breakdown(
                agent_id=agent_id,
                food_eaten_this_step=int(food_eaten_by_agent[agent_id]),
                truncated=truncated,
                terminated=terminated,
                episode_end_reason=self.episode_end_reason,
            )
            for agent_id in self.possible_agents
        }
        self.last_rewards = {
            agent_id: float(self.last_reward_breakdown[agent_id]["total_reward"])
            for agent_id in self.possible_agents
        }

        obs = self._get_obs_dict()
        infos = self._build_info_dict(
            truncated=truncated,
            terminated=terminated,
            food_eaten_by_agent=food_eaten_by_agent,
        )
        rewards = {agent_id: float(self.last_rewards[agent_id]) for agent_id in self.possible_agents}
        terminateds = {agent_id: terminated for agent_id in self.possible_agents}
        terminateds["__all__"] = terminated
        truncateds = {agent_id: truncated for agent_id in self.possible_agents}
        truncateds["__all__"] = truncated
        return obs, rewards, terminateds, truncateds, infos

    def set_debug_state(
        self,
        *,
        agent_states: dict[str, dict[str, Any]],
        food_positions,
        food_team_indices=None,
        food_active_mask=None,
        timestep: int = 0,
        steps_since_last_food: int = 0,
        episode_end_reason: str = "none",
        focus_agent_id: str | None = None,
        last_message_tokens: dict[str, int] | None = None,
        food_eaten_episode_by_agent: dict[str, int] | None = None,
    ) -> None:
        self.agents = list(self.possible_agents)
        self.timestep = int(timestep)
        self.steps_since_last_food = int(max(steps_since_last_food, 0))
        self.episode_end_reason = str(episode_end_reason)
        self.agent_food_eaten_episode = {agent_id: 0 for agent_id in self.possible_agents}
        self.team_food_eaten_episode = {RED_TEAM: 0, BLUE_TEAM: 0}
        self.last_motion_commands = {agent_id: self._zero_joint_vector() for agent_id in self.possible_agents}
        self.fish_states = {}
        self.position_progress_history = {}
        self.body_forward_velocity_history = {}
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
                joint_angles=self._coerce_joint_vector(data.get("joint_angles", None)),
                joint_velocities=self._coerce_joint_vector(data.get("joint_velocities", None)),
                prev_action=self._coerce_joint_vector(data.get("prev_action", None)),
                prev_message_token=prev_message_token,
                joint_activation=self._coerce_joint_vector(data.get("joint_activation", None)),
                applied_joint_torque=self._coerce_joint_vector(data.get("applied_joint_torque", None)),
            )
            if food_eaten_episode_by_agent and agent_id in food_eaten_episode_by_agent:
                self.agent_food_eaten_episode[agent_id] = int(food_eaten_episode_by_agent[agent_id])
            team_index = self.get_agent_team_index(agent_id)
            self.team_food_eaten_episode[team_index] += int(self.agent_food_eaten_episode[agent_id])
            self.last_motion_commands[agent_id] = self.fish_states[agent_id].prev_action.astype(np.float32).copy()
            raw_history = np.asarray(data.get("control_history", []), dtype=np.float32).reshape(
                -1,
                self._control_history_feature_count(),
            )
            if raw_history.shape == (self.history_length, self._control_history_feature_count()):
                self.control_histories[agent_id] = raw_history.copy()
            else:
                self._initialize_control_history(agent_id)
            self._reset_propulsion_history(agent_id)
            self.last_joint_limit_occupancy[agent_id] = float(
                np.mean(np.abs(self.fish_states[agent_id].joint_angles) / max(float(self.fish_preset.actuation.joint_limit), 1e-6))
            )
            self.last_joint_zero_crossings[agent_id] = 0
            self.last_capture_distance[agent_id] = float("nan")
            self.last_joint_limit_high[agent_id] = bool(self.last_joint_limit_occupancy[agent_id] > 0.75)
            self.last_joints_quiet[agent_id] = bool(np.max(np.abs(self.fish_states[agent_id].joint_velocities)) < 0.1)
            self.last_negative_forward_velocity[agent_id] = bool(self._current_motion_metrics(agent_id)["forward_velocity"] < 0.0)
            self.last_activation_sign_changes_step[agent_id] = self._zero_joint_int_vector()
            self.activation_sign_changes_episode[agent_id] = self._zero_joint_int_vector()
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
        if food_active_mask is None:
            self.food_active_mask = np.ones(self.food_count, dtype=bool)
        else:
            active_mask = np.asarray(food_active_mask, dtype=bool).reshape(-1)
            if active_mask.shape[0] != self.food_count:
                raise ValueError(f"Expected {self.food_count} food active-mask entries, got {active_mask.shape[0]}.")
            self.food_active_mask = active_mask.copy()
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
            agent_id: self.get_reward_breakdown(
                agent_id=agent_id,
                food_eaten_this_step=0,
                truncated=False,
                terminated=False,
                episode_end_reason=self.episode_end_reason,
            )
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
            "motion_command": self.last_motion_commands[agent_id].astype(np.float32).copy(),
            "joint_activation": state.joint_activation.astype(np.float32).copy(),
            "applied_joint_torque": state.applied_joint_torque.astype(np.float32).copy(),
            "mouth_position": geometry["mouth_position"].astype(np.float32).copy(),
            "food_positions": self.food_positions.astype(np.float32).copy(),
            "food_team_indices": self.food_team_indices.astype(np.int64).copy(),
            "food_active_mask": self.food_active_mask.astype(bool).copy(),
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
            "steps_since_last_food": int(self.steps_since_last_food),
            "forage_idle_timeout_steps": int(self.forage_idle_timeout_steps),
            "remaining_idle_budget_fraction": float(self._remaining_idle_budget_fraction()),
            "remaining_food_total": int(self._remaining_food_counts()["total"]),
            "remaining_food_red": int(self._remaining_food_counts()["red"]),
            "remaining_food_blue": int(self._remaining_food_counts()["blue"]),
            "food_respawn_mode": str(self.food_respawn_mode),
            "forage_timeout_mode": str(self.forage_timeout_mode),
            "episode_end_reason": str(self.episode_end_reason),
            "timestep": int(self.timestep),
            "observation": obs.copy(),
            "control_history": self.control_histories[agent_id].astype(np.float32).copy(),
            "joint_velocity_zero_crossings": int(self.last_joint_zero_crossings[agent_id]),
            "activation_sign_changes_step": self.last_activation_sign_changes_step[agent_id].astype(np.int32).copy(),
            "activation_sign_changes_episode": self.activation_sign_changes_episode[agent_id].astype(np.int32).copy(),
            "mean_joint_limit_ratio": float(self.last_joint_limit_occupancy[agent_id]),
            "joint_limit_high": bool(self.last_joint_limit_high[agent_id]),
            "joints_quiet": bool(self.last_joints_quiet[agent_id]),
            "negative_forward_velocity": bool(self.last_negative_forward_velocity[agent_id]),
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
        plt, patch_types = _get_render_matplotlib()
        Polygon = patch_types["Polygon"]
        Wedge = patch_types["Wedge"]
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.fig.patch.set_facecolor("#08131c")
        self.ax.set_facecolor("#08131c")
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
        _, patch_types = _get_render_matplotlib()
        Rectangle = patch_types["Rectangle"]
        FancyArrowPatch = patch_types["FancyArrowPatch"]
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

    def _perform_safe_redraw(self, *, force_full_redraw: bool = False) -> None:
        if self.fig is None:
            return
        self.render_backend_name = self._resolve_render_backend_name()
        if force_full_redraw or self.render_profile == "full":
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
            self._clear_render_background()
            self.render_force_full_redraw = False
            self._render_background_size = tuple(self.fig.canvas.get_width_height())
            return
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

        active_mask = self.food_active_mask.astype(bool)
        red_positions = self.food_positions[active_mask & (self.food_team_indices == RED_TEAM)]
        blue_positions = self.food_positions[active_mask & (self.food_team_indices == BLUE_TEAM)]
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
            remaining_food = self._remaining_food_counts()
            if self._forage_uses_idle_timeout():
                timer_text = (
                    f"idle={self.steps_since_last_food}/{self.forage_idle_timeout_steps} "
                    f"budget={self._remaining_idle_budget_fraction():.2f}"
                )
            else:
                timer_text = f"step={self.timestep}/{self.time_limit}"
            self.status_text.set_text(
                f"V9 | {timer_text} | red={self.team_food_eaten_episode[RED_TEAM]} "
                f"blue={self.team_food_eaten_episode[BLUE_TEAM]} | focus={focus_agent_id}({focus_team}) "
                f"remaining_food={remaining_food['total']} "
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
        if self.render_profile == "full":
            self._perform_safe_redraw(force_full_redraw=True)
            return
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
            plt, _ = _get_render_matplotlib()
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
