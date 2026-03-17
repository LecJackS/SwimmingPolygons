"""Shared deterministic evaluation utilities for V5 training and scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import torch


DEFAULT_CURRICULUM_STAGES = [0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.7, 8.0, 10.0]


@dataclass
class DistanceEvalResult:
    distance: float
    time_limit: int
    success_rate: float
    mean_steps: float
    mean_reward: float
    successes: int
    episodes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_stage_distances(raw: str) -> list[float]:
    parts = [chunk.strip() for chunk in raw.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("Distance list must be a non-empty comma-separated list.")

    distances = [float(part) for part in parts]
    if any(value <= 0 for value in distances):
        raise ValueError("Distance list values must be > 0.")
    if any(distances[idx] >= distances[idx + 1] for idx in range(len(distances) - 1)):
        raise ValueError("Distance list must be strictly increasing.")
    return distances


def compute_stage_time_limit(
    distance: float,
    *,
    min_stage_distance: float,
    max_stage_distance: float,
    base_limit: int,
    max_limit: int,
) -> int:
    if max_stage_distance <= min_stage_distance:
        return int(base_limit)
    ratio = (float(distance) - min_stage_distance) / (max_stage_distance - min_stage_distance)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    late_stage_boost = ratio + 0.15 * ratio * (1.0 - ratio)
    return int(round(base_limit + late_stage_boost * (max_limit - base_limit)))


def distance_label(distance: float) -> str:
    raw = f"{float(distance):g}"
    return f"d{raw.replace('-', 'm').replace('.', '_')}"


def weighted_success_score(distance_results: list[DistanceEvalResult]) -> float:
    if not distance_results:
        return float("nan")
    weights = np.asarray([max(result.distance, 1e-6) for result in distance_results], dtype=np.float64)
    scores = np.asarray([result.success_rate for result in distance_results], dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        return float("nan")
    return float(np.average(scores, weights=weights))


def flatten_distance_results(distance_results: list[DistanceEvalResult]) -> dict[str, float | int]:
    flat: dict[str, float | int] = {}
    for result in distance_results:
        prefix = distance_label(result.distance)
        flat[f"{prefix}_distance"] = float(result.distance)
        flat[f"{prefix}_time_limit"] = int(result.time_limit)
        flat[f"{prefix}_success_rate"] = float(result.success_rate)
        flat[f"{prefix}_mean_steps"] = float(result.mean_steps)
        flat[f"{prefix}_mean_reward"] = float(result.mean_reward)
        flat[f"{prefix}_successes"] = int(result.successes)
        flat[f"{prefix}_episodes"] = int(result.episodes)
    return flat


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


def _compute_action_new_stack(algo, obs: np.ndarray) -> np.ndarray:
    module = algo.get_module()
    obs_batch = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = module.forward_inference({"obs": obs_batch})
    logits = out["action_dist_inputs"].detach().cpu().numpy().reshape(2, 3)
    return np.argmax(logits, axis=1).astype(np.int64)


def compute_deterministic_action(algo, obs: np.ndarray, *, stack_mode: str = "old") -> np.ndarray:
    if stack_mode == "new":
        action = _compute_action_new_stack(algo, obs)
    else:
        action = algo.compute_single_action(obs, explore=False)
        if isinstance(action, tuple):
            action = action[0]
    action = np.asarray(action, dtype=np.int64).reshape(-1)
    if action.size != 2:
        return np.array([1, 1], dtype=np.int64)
    return action


def evaluate_env_rollouts(
    *,
    algo,
    env_factory,
    num_episodes: int,
    base_seed: int,
    stack_mode: str = "old",
) -> dict[str, float | int]:
    successes = 0
    episode_steps: list[int] = []
    episode_rewards: list[float] = []

    env = env_factory()
    try:
        for episode_idx in range(num_episodes):
            obs, _ = env.reset(seed=base_seed + episode_idx)
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = compute_deterministic_action(algo, obs, stack_mode=stack_mode)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                steps += 1

            if terminated:
                successes += 1
            episode_steps.append(steps)
            episode_rewards.append(total_reward)
    finally:
        env.close()

    return {
        "success_rate": successes / float(num_episodes),
        "mean_steps": float(np.mean(episode_steps)) if episode_steps else float("nan"),
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else float("nan"),
        "successes": successes,
        "episodes": num_episodes,
    }
