"""Shared deterministic evaluation utilities for V6 continuous foraging."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import torch


DEFAULT_FOOD_COUNT = 48
DEFAULT_TIME_LIMIT = 600
DEFAULT_PELLET_REWARD = 1.0
DEFAULT_STEP_COST = 0.002
DEFAULT_SENSOR_RADIUS = 4.5
DEFAULT_SENSOR_RING_EDGES = [1.5, 3.0, 4.5]
DEFAULT_SENSOR_NUM_SECTORS = 12


@dataclass
class ForagingEvalResult:
    episodes: int
    mean_food_eaten: float
    mean_reward: float
    mean_steps: float
    food_per_100_steps: float
    mean_visible_food_count: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_float_list(raw: str) -> list[float]:
    parts = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not parts:
        raise ValueError("Expected a non-empty comma-separated float list.")
    return [float(part) for part in parts]


def parse_ring_edges(raw: str) -> list[float]:
    values = parse_float_list(raw)
    if any(value <= 0.0 for value in values):
        raise ValueError("Sensor ring edges must be > 0.")
    if any(values[idx] >= values[idx + 1] for idx in range(len(values) - 1)):
        raise ValueError("Sensor ring edges must be strictly increasing.")
    return values


def flatten_foraging_result(result: ForagingEvalResult) -> dict[str, float | int]:
    return {
        "episodes": int(result.episodes),
        "mean_food_eaten": float(result.mean_food_eaten),
        "mean_reward": float(result.mean_reward),
        "mean_steps": float(result.mean_steps),
        "food_per_100_steps": float(result.food_per_100_steps),
        "mean_visible_food_count": float(result.mean_visible_food_count),
    }


def compare_foraging_results(candidate: ForagingEvalResult, incumbent: ForagingEvalResult | None) -> int:
    if incumbent is None:
        return 1
    candidate_key = (
        float(candidate.mean_food_eaten),
        float(candidate.food_per_100_steps),
        float(candidate.mean_reward),
    )
    incumbent_key = (
        float(incumbent.mean_food_eaten),
        float(incumbent.food_per_100_steps),
        float(incumbent.mean_reward),
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


def _compute_action_new_stack(algo, obs: np.ndarray) -> np.ndarray:
    module = algo.get_module()
    obs_batch = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = module.forward_inference({"obs": obs_batch})
    if "actions" in out:
        return out["actions"].detach().cpu().numpy().reshape(-1).astype(np.float32)
    action_inputs = out["action_dist_inputs"].detach().cpu().numpy().reshape(-1)
    if action_inputs.size % 2 == 0 and action_inputs.size > 2:
        half = action_inputs.size // 2
        return action_inputs[:half].astype(np.float32)
    return action_inputs.astype(np.float32)


def compute_deterministic_action(algo, obs: np.ndarray, *, stack_mode: str = "old") -> np.ndarray:
    if stack_mode == "new":
        action = _compute_action_new_stack(algo, obs)
    else:
        action = algo.compute_single_action(obs, explore=False)
        if isinstance(action, tuple):
            action = action[0]
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size != 2:
        return np.zeros(2, dtype=np.float32)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def evaluate_env_rollouts(
    *,
    algo,
    env_factory,
    num_episodes: int,
    base_seed: int,
    stack_mode: str = "old",
) -> ForagingEvalResult:
    episode_rewards: list[float] = []
    episode_steps: list[int] = []
    episode_food_eaten: list[int] = []
    episode_visible_food_count: list[float] = []

    env = env_factory()
    try:
        for episode_idx in range(num_episodes):
            obs, _ = env.reset(seed=base_seed + episode_idx)
            total_reward = 0.0
            steps = 0
            food_eaten = 0
            visible_food_accum = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = compute_deterministic_action(algo, obs, stack_mode=stack_mode)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                food_eaten += int(info.get("food_eaten_this_step", 0))
                visible_food_accum += float(info.get("visible_food_count", 0.0))

            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            episode_food_eaten.append(food_eaten)
            denom = float(max(steps, 1))
            episode_visible_food_count.append(visible_food_accum / denom)
    finally:
        env.close()

    mean_steps = float(np.mean(episode_steps)) if episode_steps else float("nan")
    mean_food_eaten = float(np.mean(episode_food_eaten)) if episode_food_eaten else float("nan")
    return ForagingEvalResult(
        episodes=int(num_episodes),
        mean_food_eaten=mean_food_eaten,
        mean_reward=float(np.mean(episode_rewards)) if episode_rewards else float("nan"),
        mean_steps=mean_steps,
        food_per_100_steps=(100.0 * mean_food_eaten / mean_steps) if mean_steps > 0.0 else float("nan"),
        mean_visible_food_count=float(np.mean(episode_visible_food_count))
        if episode_visible_food_count
        else float("nan"),
    )
