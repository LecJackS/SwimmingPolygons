"""Shared deterministic evaluation utilities for V9 raw-torque schooling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import torch


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
    mean_abs_applied_torque: float
    mean_joint_limit_occupancy: float
    mean_joint_velocity_zero_crossings_per_fish: float
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


def _normalize_action(action: Any) -> dict[str, Any]:
    if isinstance(action, tuple):
        action = action[0]
    if isinstance(action, dict):
        motion = np.asarray(action.get("motion", [0.0, 0.0]), dtype=np.float32).reshape(2)
        message = int(np.asarray(action.get("message", 0)).reshape(-1)[0])
    else:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        motion = np.zeros(2, dtype=np.float32)
        message = 0
        if values.size >= 2:
            motion = values[:2].astype(np.float32)
        if values.size >= 3:
            message = int(values[2])
    return {
        "motion": np.clip(motion, -1.0, 1.0).astype(np.float32),
        "message": int(np.clip(message, 0, DEFAULT_NUM_MESSAGE_TOKENS - 1)),
    }


def _to_numpy_action_dict(action: Any) -> dict[str, Any]:
    if isinstance(action, dict):
        motion = action.get("motion", np.zeros((1, 2), dtype=np.float32))
        message = action.get("message", np.zeros((1,), dtype=np.int64))
        motion_np = np.asarray(motion, dtype=np.float32).reshape(-1)
        message_np = np.asarray(message).reshape(-1)
        return {
            "motion": motion_np[:2].astype(np.float32) if motion_np.size >= 2 else np.zeros(2, dtype=np.float32),
            "message": int(message_np[0]) if message_np.size else 0,
        }
    return _normalize_action(action)


def _compute_action_new_stack(algo, obs: np.ndarray) -> dict[str, Any]:
    module = algo.get_module()
    obs_batch = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = module.forward_inference({"obs": obs_batch})
    if "actions" not in out:
        raise RuntimeError("RLModule inference did not return actions for mixed action space.")
    return _to_numpy_action_dict(out["actions"])


def _compute_action_batch_new_stack(algo, obs_batch: np.ndarray):
    module = algo.get_module()
    obs_tensor = torch.tensor(obs_batch, dtype=torch.float32)
    with torch.no_grad():
        out = module.forward_inference({"obs": obs_tensor})
    if "actions" not in out:
        raise RuntimeError("RLModule inference did not return batched actions for mixed action space.")
    return out["actions"]


def compute_deterministic_action(
    algo,
    obs: np.ndarray,
    *,
    stack_mode: str = "old",
    policy_id: str = SHARED_POLICY_ID,
) -> dict[str, Any]:
    if stack_mode == "new":
        return _normalize_action(_compute_action_new_stack(algo, obs))
    action = algo.compute_single_action(obs, policy_id=policy_id, explore=False)
    return _normalize_action(action)


def _normalize_batched_actions(action_batch: Any, batch_size: int) -> list[dict[str, Any]]:
    if isinstance(action_batch, dict):
        motion = np.asarray(action_batch.get("motion", np.zeros((batch_size, 2), dtype=np.float32)), dtype=np.float32)
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
                "motion": np.clip(motion[idx, :2], -1.0, 1.0).astype(np.float32),
                "message": int(np.clip(message[idx], 0, DEFAULT_NUM_MESSAGE_TOKENS - 1)),
            }
            for idx in range(batch_size)
        ]
    values = np.asarray(action_batch)
    if values.ndim == 1:
        values = np.broadcast_to(values.reshape(1, -1), (batch_size, values.shape[0]))
    return [_normalize_action(values[idx]) for idx in range(batch_size)]


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
    if stack_mode == "new":
        action_batch = _compute_action_batch_new_stack(algo, obs_batch)
    else:
        policy = algo.get_policy(policy_id)
        action_batch, _, _ = policy.compute_actions(obs_batch, explore=False)
    normalized = _normalize_batched_actions(action_batch, len(agent_ids))
    return {agent_id: normalized[idx] for idx, agent_id in enumerate(agent_ids)}


def sample_random_action(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "motion": rng.uniform(-1.0, 1.0, size=2).astype(np.float32),
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
    episode_abs_applied_torque: list[float] = []
    episode_joint_limit_occupancy: list[float] = []
    episode_zero_crossings_per_fish: list[float] = []
    episode_nearest_food_distance: list[float] = []
    episode_capture_distance: list[float] = []
    episode_entropy: list[float] = []
    token_hist_total = np.zeros(DEFAULT_NUM_MESSAGE_TOKENS, dtype=np.float64)

    env = env_factory()
    try:
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
            abs_applied_torque_accum = 0.0
            joint_limit_occupancy_accum = 0.0
            zero_crossings_total = 0.0
            nearest_food_samples: list[float] = []
            capture_distance_samples: list[float] = []
            steps = 0
            token_hist_episode = np.zeros(DEFAULT_NUM_MESSAGE_TOKENS, dtype=np.float64)

            while True:
                action_dict = action_fn(env=env, obs_dict=obs_dict, episode_seed=base_seed + episode_idx, step_idx=steps)
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
                    abs_applied_torque_accum += abs(float(infos[agent_id].get("mean_abs_applied_torque", 0.0)))
                    joint_limit_occupancy_accum += float(infos[agent_id].get("mean_joint_limit_ratio", 0.0))
                    zero_crossings_total += float(infos[agent_id].get("joint_velocity_zero_crossings", 0))
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
            episode_abs_applied_torque.append(abs_applied_torque_accum / denom)
            episode_joint_limit_occupancy.append(joint_limit_occupancy_accum / denom)
            episode_zero_crossings_per_fish.append(zero_crossings_total / float(num_agents))
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
        mean_abs_applied_torque=float(np.mean(episode_abs_applied_torque)) if episode_abs_applied_torque else float("nan"),
        mean_joint_limit_occupancy=float(np.mean(episode_joint_limit_occupancy)) if episode_joint_limit_occupancy else float("nan"),
        mean_joint_velocity_zero_crossings_per_fish=float(np.mean(episode_zero_crossings_per_fish)) if episode_zero_crossings_per_fish else float("nan"),
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
    stack_mode: str = "old",
    policy_id: str = SHARED_POLICY_ID,
) -> ColorCommEvalResult:
    return _evaluate_multi_agent_rollouts(
        env_factory=env_factory,
        num_episodes=num_episodes,
        base_seed=base_seed,
        action_fn=lambda *, env, obs_dict, episode_seed, step_idx: compute_batched_deterministic_actions(
            algo,
            obs_dict,
            stack_mode=stack_mode,
            policy_id=policy_id,
        ),
    )


def evaluate_multi_agent_random_rollouts(
    *,
    env_factory,
    num_episodes: int,
    base_seed: int,
) -> ColorCommEvalResult:
    episode_rngs: dict[int, np.random.Generator] = {}

    def action_fn(*, env, obs_dict, episode_seed, step_idx):
        rng = episode_rngs.setdefault(episode_seed, np.random.default_rng(episode_seed))
        return {agent_id: sample_random_action(rng) for agent_id in obs_dict.keys()}

    return _evaluate_multi_agent_rollouts(
        env_factory=env_factory,
        num_episodes=num_episodes,
        base_seed=base_seed,
        action_fn=action_fn,
    )
