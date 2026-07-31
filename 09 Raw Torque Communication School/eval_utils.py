"""Shared deterministic evaluation utilities for V9 muscle-activation schooling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import torch
from ray.rllib.core.columns import Columns


DEFAULT_NUM_RED_FISH = 10
DEFAULT_NUM_BLUE_FISH = 10
DEFAULT_NUM_RED_PELLETS = 48
DEFAULT_NUM_BLUE_PELLETS = 48
DEFAULT_TIME_LIMIT = 300
DEFAULT_PELLET_REWARD = 1.0
DEFAULT_STEP_COST = 0.002
DEFAULT_SECTOR_RADIUS = 5.0
DEFAULT_SECTOR_NUM = 6
DEFAULT_NUM_MESSAGE_TOKENS = 4
DEFAULT_SATURATED_COMMAND_THRESHOLD = 0.75
SHARED_POLICY_ID = "shared_fish_policy"


@dataclass
class ColorCommEvalResult:
    episodes: int
    mean_total_reward: float
    mean_reward_red: float
    mean_reward_blue: float
    mean_steps: float
    mean_pellets_red_eaten_by_red: float
    mean_pellets_blue_eaten_by_blue: float
    mean_pellets_per_fish: float
    mean_visible_food_count: float
    mean_visible_teammate_count: float
    mean_visible_opponent_count: float
    mean_forward_velocity: float
    mean_lateral_velocity: float
    mean_abs_angular_velocity: float
    num_motion_joints: int
    mean_motion_command_abs: float
    mean_motion_command_std_mean: float
    fraction_saturated_motion_commands: float
    mean_abs_desired_activation: float
    mean_abs_activation: float
    mean_abs_applied_torque: float
    mean_joint_limit_occupancy: float
    mean_near_limit_penalty: float
    fraction_near_limit_joints: float
    mean_joint_limit_excess: float
    mean_saturation_penalty: float
    mean_torque_penalty: float
    fraction_joint_limit_high_steps: float
    fraction_joints_quiet_steps: float
    fraction_negative_forward_velocity_steps: float
    mean_joint_velocity_zero_crossings_per_fish: float
    mean_activation_sign_changes_per_fish: float
    mean_nearest_food_distance: float
    mean_capture_distance: float
    mean_message_entropy: float
    token_freq_0: float
    token_freq_1: float
    token_freq_2: float
    token_freq_3: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def flatten_result(result: ColorCommEvalResult, *, prefix: str = "") -> dict[str, float | int]:
    return {f"{prefix}{key}": value for key, value in asdict(result).items()}


def compare_results(
    candidate: ColorCommEvalResult,
    incumbent: ColorCommEvalResult | None,
    *,
    candidate_comm_gain: float | None = None,
    incumbent_comm_gain: float | None = None,
) -> int:
    if incumbent is None:
        return 1
    if candidate_comm_gain is None or incumbent_comm_gain is None:
        candidate_key = (
            float(candidate.mean_pellets_per_fish),
            float(candidate.mean_total_reward),
            float(candidate.mean_pellets_red_eaten_by_red + candidate.mean_pellets_blue_eaten_by_blue),
        )
        incumbent_key = (
            float(incumbent.mean_pellets_per_fish),
            float(incumbent.mean_total_reward),
            float(incumbent.mean_pellets_red_eaten_by_red + incumbent.mean_pellets_blue_eaten_by_blue),
        )
    else:
        candidate_key = (
            float(candidate.mean_pellets_per_fish),
            float(candidate.mean_total_reward),
            float(candidate_comm_gain),
            float(candidate.mean_pellets_red_eaten_by_red + candidate.mean_pellets_blue_eaten_by_blue),
        )
        incumbent_key = (
            float(incumbent.mean_pellets_per_fish),
            float(incumbent.mean_total_reward),
            float(incumbent_comm_gain),
            float(incumbent.mean_pellets_red_eaten_by_red + incumbent.mean_pellets_blue_eaten_by_blue),
        )
    if candidate_key > incumbent_key:
        return 1
    if candidate_key < incumbent_key:
        return -1
    return 0


def find_latest_checkpoint(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")
    candidates = [path for path in root.rglob("checkpoint_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint directories found under: {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def uri_to_local_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Unsupported URI scheme for local checkpoint: {uri}")
    path = unquote(parsed.path)
    if len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return Path(path)


def _motion_dim_from_action_space(action_space: Any) -> int:
    motion_space = getattr(action_space, "spaces", {}).get("motion")
    if motion_space is None or not hasattr(motion_space, "shape") or len(tuple(motion_space.shape)) != 1:
        raise ValueError("Expected Dict action space with 1-D 'motion' branch.")
    motion_dim = int(motion_space.shape[0])
    if motion_dim <= 0:
        raise ValueError("Motion dimension must be > 0.")
    return motion_dim


def _policy_motion_dim(algo, policy_id: str = SHARED_POLICY_ID) -> int:
    action_space = None
    get_module = getattr(algo, "get_module", None)
    if callable(get_module):
        try:
            module = get_module(policy_id)
            action_space = getattr(module, "action_space", None)
        except Exception:
            action_space = None
    if action_space is None:
        policy = algo.get_policy(policy_id)
        action_space = getattr(policy, "action_space", None)
    return _motion_dim_from_action_space(action_space)


def _normalize_action(action: Any, *, motion_dim: int) -> dict[str, Any]:
    if isinstance(action, tuple):
        action = action[0]
    if isinstance(action, dict):
        motion = np.asarray(action.get("motion", np.zeros(motion_dim, dtype=np.float32)), dtype=np.float32).reshape(motion_dim)
        message = int(np.asarray(action.get("message", 0)).reshape(-1)[0])
    else:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        motion = np.zeros(motion_dim, dtype=np.float32)
        message = 0
        if values.size >= motion_dim:
            motion = values[:motion_dim].astype(np.float32)
        if values.size >= motion_dim + 1:
            message = int(values[motion_dim])
    return {
        "motion": np.clip(motion, -1.0, 1.0).astype(np.float32),
        "message": int(np.clip(message, 0, DEFAULT_NUM_MESSAGE_TOKENS - 1)),
    }


def _to_numpy_action_dict(action: Any, *, motion_dim: int) -> dict[str, Any]:
    if isinstance(action, dict):
        motion = action.get("motion", np.zeros((1, motion_dim), dtype=np.float32))
        message = action.get("message", np.zeros((1,), dtype=np.int64))
        motion_np = np.asarray(motion, dtype=np.float32).reshape(-1)
        message_np = np.asarray(message).reshape(-1)
        return {
            "motion": motion_np[:motion_dim].astype(np.float32) if motion_np.size >= motion_dim else np.zeros(motion_dim, dtype=np.float32),
            "message": int(message_np[0]) if message_np.size else 0,
        }
    return _normalize_action(action, motion_dim=motion_dim)


def _compute_action_new_stack(algo, obs: np.ndarray, *, stochastic: bool = False, policy_id: str = SHARED_POLICY_ID) -> dict[str, Any]:
    module = algo.get_module(policy_id)
    obs_batch = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        if stochastic:
            out = module.forward_exploration({"obs": obs_batch})
            dist_cls = module.get_exploration_action_dist_cls()
        else:
            out = module.forward_inference({"obs": obs_batch})
            dist_cls = module.get_inference_action_dist_cls()
    actions = out.get(Columns.ACTIONS)
    if actions is None:
        if Columns.ACTION_DIST_INPUTS not in out:
            raise RuntimeError("RLModule inference did not return action dist inputs for mixed action space.")
        action_dist = dist_cls.from_logits(out[Columns.ACTION_DIST_INPUTS])
        if not stochastic:
            action_dist = action_dist.to_deterministic()
        actions = action_dist.sample()
    motion_dim = _policy_motion_dim(algo, policy_id=policy_id)
    return _to_numpy_action_dict(actions, motion_dim=motion_dim)


def _compute_action_batch_new_stack(algo, obs_batch: np.ndarray, *, stochastic: bool = False, policy_id: str = SHARED_POLICY_ID):
    module = algo.get_module(policy_id)
    obs_tensor = torch.tensor(obs_batch, dtype=torch.float32)
    with torch.no_grad():
        if stochastic:
            out = module.forward_exploration({"obs": obs_tensor})
            dist_cls = module.get_exploration_action_dist_cls()
        else:
            out = module.forward_inference({"obs": obs_tensor})
            dist_cls = module.get_inference_action_dist_cls()
    actions = out.get(Columns.ACTIONS)
    if actions is None:
        if Columns.ACTION_DIST_INPUTS not in out:
            raise RuntimeError("RLModule inference did not return batched dist inputs for mixed action space.")
        action_dist = dist_cls.from_logits(out[Columns.ACTION_DIST_INPUTS])
        if not stochastic:
            action_dist = action_dist.to_deterministic()
        actions = action_dist.sample()
    return actions


def compute_deterministic_action(
    algo,
    obs: np.ndarray,
    *,
    stack_mode: str = "old",
    policy_id: str = SHARED_POLICY_ID,
) -> dict[str, Any]:
    if stack_mode == "new":
        motion_dim = _policy_motion_dim(algo, policy_id=policy_id)
        return _normalize_action(_compute_action_new_stack(algo, obs, stochastic=False, policy_id=policy_id), motion_dim=motion_dim)
    action = algo.compute_single_action(obs, policy_id=policy_id, explore=False)
    motion_dim = _policy_motion_dim(algo, policy_id=policy_id)
    return _normalize_action(action, motion_dim=motion_dim)


def compute_stochastic_action(
    algo,
    obs: np.ndarray,
    *,
    stack_mode: str = "old",
    policy_id: str = SHARED_POLICY_ID,
) -> dict[str, Any]:
    if stack_mode == "new":
        motion_dim = _policy_motion_dim(algo, policy_id=policy_id)
        return _normalize_action(_compute_action_new_stack(algo, obs, stochastic=True, policy_id=policy_id), motion_dim=motion_dim)
    action = algo.compute_single_action(obs, policy_id=policy_id, explore=True)
    motion_dim = _policy_motion_dim(algo, policy_id=policy_id)
    return _normalize_action(action, motion_dim=motion_dim)


def _normalize_batched_actions(action_batch: Any, batch_size: int, *, motion_dim: int) -> list[dict[str, Any]]:
    if isinstance(action_batch, dict):
        motion = np.asarray(action_batch.get("motion", np.zeros((batch_size, motion_dim), dtype=np.float32)), dtype=np.float32)
        message = np.asarray(action_batch.get("message", np.zeros((batch_size,), dtype=np.int64)))
        if motion.ndim == 1:
            motion = np.broadcast_to(motion.reshape(1, -1), (batch_size, motion.shape[0]))
        if motion.shape[0] != batch_size:
            motion = motion.reshape(batch_size, -1)
        message = message.reshape(-1)
        if message.size == 1 and batch_size > 1:
            message = np.broadcast_to(message, (batch_size,))
        return [
            {
                "motion": np.clip(motion[idx, :motion_dim], -1.0, 1.0).astype(np.float32),
                "message": int(np.clip(message[idx], 0, DEFAULT_NUM_MESSAGE_TOKENS - 1)),
            }
            for idx in range(batch_size)
        ]
    values = np.asarray(action_batch)
    if values.ndim == 1:
        values = np.broadcast_to(values.reshape(1, -1), (batch_size, values.shape[0]))
    return [_normalize_action(values[idx], motion_dim=motion_dim) for idx in range(batch_size)]


def compute_batched_deterministic_actions(
    algo,
    obs_dict: dict[str, np.ndarray],
    *,
    stack_mode: str = "old",
    policy_id: str = SHARED_POLICY_ID,
) -> dict[str, dict[str, Any]]:
    agent_ids = list(obs_dict.keys())
    if not agent_ids:
        return {}
    obs_batch = np.stack([np.asarray(obs_dict[agent_id], dtype=np.float32) for agent_id in agent_ids], axis=0)
    motion_dim = _policy_motion_dim(algo, policy_id=policy_id)
    if stack_mode == "new":
        action_batch = _compute_action_batch_new_stack(algo, obs_batch, stochastic=False, policy_id=policy_id)
    else:
        policy = algo.get_policy(policy_id)
        action_batch, _, _ = policy.compute_actions(obs_batch, explore=False)
    normalized = _normalize_batched_actions(action_batch, len(agent_ids), motion_dim=motion_dim)
    return {agent_id: normalized[idx] for idx, agent_id in enumerate(agent_ids)}


def compute_batched_stochastic_actions(
    algo,
    obs_dict: dict[str, np.ndarray],
    *,
    stack_mode: str = "old",
    policy_id: str = SHARED_POLICY_ID,
) -> dict[str, dict[str, Any]]:
    agent_ids = list(obs_dict.keys())
    if not agent_ids:
        return {}
    obs_batch = np.stack([np.asarray(obs_dict[agent_id], dtype=np.float32) for agent_id in agent_ids], axis=0)
    motion_dim = _policy_motion_dim(algo, policy_id=policy_id)
    if stack_mode == "new":
        action_batch = _compute_action_batch_new_stack(algo, obs_batch, stochastic=True, policy_id=policy_id)
    else:
        policy = algo.get_policy(policy_id)
        action_batch, _, _ = policy.compute_actions(obs_batch, explore=True)
    normalized = _normalize_batched_actions(action_batch, len(agent_ids), motion_dim=motion_dim)
    return {agent_id: normalized[idx] for idx, agent_id in enumerate(agent_ids)}


def sample_random_action(rng: np.random.Generator, *, motion_dim: int) -> dict[str, Any]:
    return {
        "motion": rng.uniform(-1.0, 1.0, size=motion_dim).astype(np.float32),
        "message": int(rng.integers(0, DEFAULT_NUM_MESSAGE_TOKENS)),
    }


def _message_entropy(histogram: np.ndarray) -> float:
    total = float(np.sum(histogram))
    if total <= 0.0:
        return 0.0
    probs = histogram / total
    probs = probs[probs > 0.0]
    return float(-np.sum(probs * np.log2(probs)))


def _mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    finite = [float(value) for value in values if np.isfinite(float(value))]
    if not finite:
        return float("nan")
    return float(np.mean(np.asarray(finite, dtype=np.float32)))


def _evaluate_multi_agent_rollouts(
    *,
    env_factory,
    num_episodes: int,
    base_seed: int,
    action_fn,
) -> ColorCommEvalResult:
    episode_total_rewards: list[float] = []
    episode_red_rewards: list[float] = []
    episode_blue_rewards: list[float] = []
    episode_steps: list[int] = []
    episode_red_food: list[int] = []
    episode_blue_food: list[int] = []
    episode_pellets_per_fish: list[float] = []
    episode_visible_food: list[float] = []
    episode_visible_teammate: list[float] = []
    episode_visible_opponent: list[float] = []
    episode_forward_velocity: list[float] = []
    episode_lateral_velocity: list[float] = []
    episode_abs_angular_velocity: list[float] = []
    episode_motion_command_abs: list[float] = []
    episode_motion_command_std_mean: list[float] = []
    episode_saturated_motion_fraction: list[float] = []
    episode_abs_desired_activation: list[float] = []
    episode_abs_activation: list[float] = []
    episode_abs_applied_torque: list[float] = []
    episode_joint_limit_occupancy: list[float] = []
    episode_near_limit_penalty: list[float] = []
    episode_fraction_near_limit_joints: list[float] = []
    episode_joint_limit_excess: list[float] = []
    episode_saturation_penalty: list[float] = []
    episode_torque_penalty: list[float] = []
    episode_joint_limit_high_fraction: list[float] = []
    episode_joints_quiet_fraction: list[float] = []
    episode_negative_forward_fraction: list[float] = []
    episode_zero_crossings_per_fish: list[float] = []
    episode_activation_sign_changes_per_fish: list[float] = []
    episode_nearest_food_distance: list[float] = []
    episode_capture_distance: list[float] = []
    episode_entropy: list[float] = []
    token_hist_total = np.zeros(DEFAULT_NUM_MESSAGE_TOKENS, dtype=np.float64)

    env = env_factory()
    try:
        motion_dim = _motion_dim_from_action_space(env.action_space)
        for episode_idx in range(num_episodes):
            obs_dict, _ = env.reset(seed=base_seed + episode_idx)
            total_reward = 0.0
            red_reward = 0.0
            blue_reward = 0.0
            red_food = 0
            blue_food = 0
            visible_food_accum = 0.0
            visible_teammate_accum = 0.0
            visible_opponent_accum = 0.0
            forward_velocity_accum = 0.0
            lateral_velocity_accum = 0.0
            abs_angular_velocity_accum = 0.0
            motion_command_sum = np.zeros(motion_dim, dtype=np.float64)
            motion_command_sq_sum = np.zeros(motion_dim, dtype=np.float64)
            motion_command_abs_accum = 0.0
            saturated_motion_count = 0.0
            desired_activation_abs_accum = 0.0
            abs_activation_accum = 0.0
            abs_applied_torque_accum = 0.0
            joint_limit_occupancy_accum = 0.0
            near_limit_penalty_accum = 0.0
            fraction_near_limit_joints_accum = 0.0
            joint_limit_excess_accum = 0.0
            saturation_penalty_accum = 0.0
            torque_penalty_accum = 0.0
            joint_limit_high_accum = 0.0
            joints_quiet_accum = 0.0
            negative_forward_accum = 0.0
            zero_crossings_total = 0.0
            activation_sign_changes_total = 0.0
            nearest_food_samples: list[float] = []
            capture_distance_samples: list[float] = []
            steps = 0
            token_hist_episode = np.zeros(DEFAULT_NUM_MESSAGE_TOKENS, dtype=np.float64)

            while True:
                action_dict = action_fn(
                    env=env,
                    obs_dict=obs_dict,
                    episode_seed=base_seed + episode_idx,
                    step_idx=steps,
                    motion_dim=motion_dim,
                )
                for action in action_dict.values():
                    motion = np.asarray(action.get("motion", np.zeros(motion_dim, dtype=np.float32)), dtype=np.float32).reshape(motion_dim)
                    motion_command_sum += motion.astype(np.float64)
                    motion_command_sq_sum += np.square(motion.astype(np.float64))
                    motion_command_abs_accum += float(np.mean(np.abs(motion)))
                    saturated_motion_count += float(np.count_nonzero(np.abs(motion) > DEFAULT_SATURATED_COMMAND_THRESHOLD))
                    desired_activation_abs_accum += float(np.mean(np.abs(motion)))
                obs_dict, rewards, terminateds, truncateds, infos = env.step(action_dict)
                steps += 1

                for agent_id, reward in rewards.items():
                    total_reward += float(reward)
                    if infos[agent_id]["agent_team"] == "red":
                        red_reward += float(reward)
                        red_food += int(infos[agent_id]["food_eaten_this_step"])
                    else:
                        blue_reward += float(reward)
                        blue_food += int(infos[agent_id]["food_eaten_this_step"])
                    visible_food_accum += float(infos[agent_id].get("visible_food_count", 0))
                    visible_teammate_accum += float(infos[agent_id].get("visible_teammate_count", 0))
                    visible_opponent_accum += float(infos[agent_id].get("visible_opponent_count", 0))
                    forward_velocity_accum += float(infos[agent_id].get("forward_velocity", 0.0))
                    lateral_velocity_accum += float(infos[agent_id].get("lateral_velocity", 0.0))
                    abs_angular_velocity_accum += abs(float(infos[agent_id].get("angular_velocity", 0.0)))
                    abs_activation_accum += abs(float(infos[agent_id].get("mean_abs_activation", 0.0)))
                    abs_applied_torque_accum += abs(float(infos[agent_id].get("mean_abs_applied_torque", 0.0)))
                    joint_limit_occupancy_accum += float(infos[agent_id].get("mean_joint_limit_ratio", 0.0))
                    near_limit_penalty_accum += float(infos[agent_id].get("near_limit_penalty", 0.0))
                    fraction_near_limit_joints_accum += float(infos[agent_id].get("fraction_near_limit_joints", 0.0))
                    joint_limit_excess_accum += float(infos[agent_id].get("mean_joint_limit_excess", 0.0))
                    saturation_penalty_accum += float(infos[agent_id].get("locomotion_saturation_penalty", 0.0))
                    torque_penalty_accum += float(infos[agent_id].get("locomotion_torque_penalty", 0.0))
                    joint_limit_high_accum += float(bool(infos[agent_id].get("joint_limit_high", False)))
                    joints_quiet_accum += float(bool(infos[agent_id].get("joints_quiet", False)))
                    negative_forward_accum += float(bool(infos[agent_id].get("negative_forward_velocity", False)))
                    zero_crossings_total += float(infos[agent_id].get("joint_velocity_zero_crossings", 0))
                    activation_sign_changes_total += float(infos[agent_id].get("activation_sign_changes_this_step", 0))
                    nearest_food_distance = float(infos[agent_id].get("nearest_food_distance", float("nan")))
                    if np.isfinite(nearest_food_distance):
                        nearest_food_samples.append(nearest_food_distance)
                    capture_distance = float(infos[agent_id].get("capture_distance_this_step", float("nan")))
                    if np.isfinite(capture_distance):
                        capture_distance_samples.append(capture_distance)
                    token = int(infos[agent_id].get("emitted_message_token", 0))
                    token_hist_episode[token] += 1.0

                if terminateds["__all__"] or truncateds["__all__"]:
                    break

            num_agents = max(len(env.get_agent_ids()), 1)
            denom = float(max(steps * num_agents, 1))
            token_hist_total += token_hist_episode
            episode_total_rewards.append(total_reward)
            episode_red_rewards.append(red_reward)
            episode_blue_rewards.append(blue_reward)
            episode_steps.append(steps)
            episode_red_food.append(red_food)
            episode_blue_food.append(blue_food)
            episode_pellets_per_fish.append(float(red_food + blue_food) / float(num_agents))
            episode_visible_food.append(visible_food_accum / denom)
            episode_visible_teammate.append(visible_teammate_accum / denom)
            episode_visible_opponent.append(visible_opponent_accum / denom)
            episode_forward_velocity.append(forward_velocity_accum / denom)
            episode_lateral_velocity.append(lateral_velocity_accum / denom)
            episode_abs_angular_velocity.append(abs_angular_velocity_accum / denom)
            motion_command_mean = motion_command_sum / denom
            motion_command_var = np.maximum((motion_command_sq_sum / denom) - np.square(motion_command_mean), 0.0)
            episode_motion_command_abs.append(motion_command_abs_accum / denom)
            episode_motion_command_std_mean.append(float(np.mean(np.sqrt(motion_command_var.astype(np.float64)))))
            episode_saturated_motion_fraction.append(saturated_motion_count / float(max(steps * num_agents * motion_dim, 1)))
            episode_abs_desired_activation.append(desired_activation_abs_accum / denom)
            episode_abs_activation.append(abs_activation_accum / denom)
            episode_abs_applied_torque.append(abs_applied_torque_accum / denom)
            episode_joint_limit_occupancy.append(joint_limit_occupancy_accum / denom)
            episode_near_limit_penalty.append(near_limit_penalty_accum / denom)
            episode_fraction_near_limit_joints.append(fraction_near_limit_joints_accum / denom)
            episode_joint_limit_excess.append(joint_limit_excess_accum / denom)
            episode_saturation_penalty.append(saturation_penalty_accum / denom)
            episode_torque_penalty.append(torque_penalty_accum / denom)
            episode_joint_limit_high_fraction.append(joint_limit_high_accum / denom)
            episode_joints_quiet_fraction.append(joints_quiet_accum / denom)
            episode_negative_forward_fraction.append(negative_forward_accum / denom)
            episode_zero_crossings_per_fish.append(zero_crossings_total / float(num_agents))
            episode_activation_sign_changes_per_fish.append(activation_sign_changes_total / float(num_agents))
            episode_nearest_food_distance.append(_mean_or_nan(nearest_food_samples))
            episode_capture_distance.append(_mean_or_nan(capture_distance_samples))
            episode_entropy.append(_message_entropy(token_hist_episode))
    finally:
        env.close()

    token_total = float(np.sum(token_hist_total))
    token_freqs = token_hist_total / token_total if token_total > 0.0 else np.zeros_like(token_hist_total)
    mean_steps = float(np.mean(episode_steps)) if episode_steps else float("nan")
    return ColorCommEvalResult(
        episodes=int(num_episodes),
        mean_total_reward=float(np.mean(episode_total_rewards)) if episode_total_rewards else float("nan"),
        mean_reward_red=float(np.mean(episode_red_rewards)) if episode_red_rewards else float("nan"),
        mean_reward_blue=float(np.mean(episode_blue_rewards)) if episode_blue_rewards else float("nan"),
        mean_steps=mean_steps,
        mean_pellets_red_eaten_by_red=float(np.mean(episode_red_food)) if episode_red_food else float("nan"),
        mean_pellets_blue_eaten_by_blue=float(np.mean(episode_blue_food)) if episode_blue_food else float("nan"),
        mean_pellets_per_fish=float(np.mean(episode_pellets_per_fish)) if episode_pellets_per_fish else float("nan"),
        mean_visible_food_count=float(np.mean(episode_visible_food)) if episode_visible_food else float("nan"),
        mean_visible_teammate_count=float(np.mean(episode_visible_teammate)) if episode_visible_teammate else float("nan"),
        mean_visible_opponent_count=float(np.mean(episode_visible_opponent)) if episode_visible_opponent else float("nan"),
        mean_forward_velocity=float(np.mean(episode_forward_velocity)) if episode_forward_velocity else float("nan"),
        mean_lateral_velocity=float(np.mean(episode_lateral_velocity)) if episode_lateral_velocity else float("nan"),
        mean_abs_angular_velocity=float(np.mean(episode_abs_angular_velocity)) if episode_abs_angular_velocity else float("nan"),
        num_motion_joints=int(motion_dim),
        mean_motion_command_abs=float(np.mean(episode_motion_command_abs)) if episode_motion_command_abs else float("nan"),
        mean_motion_command_std_mean=float(np.mean(episode_motion_command_std_mean)) if episode_motion_command_std_mean else float("nan"),
        fraction_saturated_motion_commands=(
            float(np.mean(episode_saturated_motion_fraction)) if episode_saturated_motion_fraction else float("nan")
        ),
        mean_abs_desired_activation=float(np.mean(episode_abs_desired_activation)) if episode_abs_desired_activation else float("nan"),
        mean_abs_activation=float(np.mean(episode_abs_activation)) if episode_abs_activation else float("nan"),
        mean_abs_applied_torque=float(np.mean(episode_abs_applied_torque)) if episode_abs_applied_torque else float("nan"),
        mean_joint_limit_occupancy=float(np.mean(episode_joint_limit_occupancy)) if episode_joint_limit_occupancy else float("nan"),
        mean_near_limit_penalty=float(np.mean(episode_near_limit_penalty)) if episode_near_limit_penalty else float("nan"),
        fraction_near_limit_joints=(
            float(np.mean(episode_fraction_near_limit_joints)) if episode_fraction_near_limit_joints else float("nan")
        ),
        mean_joint_limit_excess=float(np.mean(episode_joint_limit_excess)) if episode_joint_limit_excess else float("nan"),
        mean_saturation_penalty=float(np.mean(episode_saturation_penalty)) if episode_saturation_penalty else float("nan"),
        mean_torque_penalty=float(np.mean(episode_torque_penalty)) if episode_torque_penalty else float("nan"),
        fraction_joint_limit_high_steps=(
            float(np.mean(episode_joint_limit_high_fraction)) if episode_joint_limit_high_fraction else float("nan")
        ),
        fraction_joints_quiet_steps=(
            float(np.mean(episode_joints_quiet_fraction)) if episode_joints_quiet_fraction else float("nan")
        ),
        fraction_negative_forward_velocity_steps=(
            float(np.mean(episode_negative_forward_fraction)) if episode_negative_forward_fraction else float("nan")
        ),
        mean_joint_velocity_zero_crossings_per_fish=float(np.mean(episode_zero_crossings_per_fish)) if episode_zero_crossings_per_fish else float("nan"),
        mean_activation_sign_changes_per_fish=(
            float(np.mean(episode_activation_sign_changes_per_fish))
            if episode_activation_sign_changes_per_fish
            else float("nan")
        ),
        mean_nearest_food_distance=_mean_or_nan(episode_nearest_food_distance),
        mean_capture_distance=_mean_or_nan(episode_capture_distance),
        mean_message_entropy=float(np.mean(episode_entropy)) if episode_entropy else float("nan"),
        token_freq_0=float(token_freqs[0]),
        token_freq_1=float(token_freqs[1]),
        token_freq_2=float(token_freqs[2]),
        token_freq_3=float(token_freqs[3]),
    )


def evaluate_multi_agent_rollouts(
    *,
    algo,
    env_factory,
    num_episodes: int,
    base_seed: int,
    action_selection: str = "deterministic",
    stack_mode: str = "old",
    policy_id: str = SHARED_POLICY_ID,
) -> ColorCommEvalResult:
    if action_selection == "stochastic":
        action_fn = lambda *, env, obs_dict, episode_seed, step_idx, motion_dim: compute_batched_stochastic_actions(
            algo,
            obs_dict,
            stack_mode=stack_mode,
            policy_id=policy_id,
        )
    else:
        action_fn = lambda *, env, obs_dict, episode_seed, step_idx, motion_dim: compute_batched_deterministic_actions(
            algo,
            obs_dict,
            stack_mode=stack_mode,
            policy_id=policy_id,
        )
    return _evaluate_multi_agent_rollouts(
        env_factory=env_factory,
        num_episodes=num_episodes,
        base_seed=base_seed,
        action_fn=action_fn,
    )


def evaluate_multi_agent_random_rollouts(
    *,
    env_factory,
    num_episodes: int,
    base_seed: int,
) -> ColorCommEvalResult:
    episode_rngs: dict[int, np.random.Generator] = {}

    def action_fn(*, env, obs_dict, episode_seed, step_idx, motion_dim):
        rng = episode_rngs.setdefault(episode_seed, np.random.default_rng(episode_seed))
        return {agent_id: sample_random_action(rng, motion_dim=motion_dim) for agent_id in obs_dict.keys()}

    return _evaluate_multi_agent_rollouts(
        env_factory=env_factory,
        num_episodes=num_episodes,
        base_seed=base_seed,
        action_fn=action_fn,
    )
