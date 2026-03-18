"""Shared deterministic evaluation utilities for V7 shared-policy schooling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import torch


DEFAULT_NUM_FISH = 8
DEFAULT_FOOD_COUNT = 48
DEFAULT_TIME_LIMIT = 600
DEFAULT_PELLET_REWARD = 1.0
DEFAULT_STEP_COST = 0.002
DEFAULT_SENSOR_RADIUS = 4.5
DEFAULT_SENSOR_RING_EDGES = [1.5, 3.0, 4.5]
DEFAULT_SENSOR_NUM_SECTORS = 12
SHARED_POLICY_ID = "shared_fish_policy"


@dataclass
class TeamForagingEvalResult:
    episodes: int
    mean_team_food_eaten: float
    mean_team_reward: float
    mean_steps: float
    team_food_per_100_steps: float
    mean_food_eaten_per_fish: float
    mean_visible_food_count_per_fish: float

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


def flatten_team_result(result: TeamForagingEvalResult) -> dict[str, float | int]:
    return {
        "episodes": int(result.episodes),
        "mean_team_food_eaten": float(result.mean_team_food_eaten),
        "mean_team_reward": float(result.mean_team_reward),
        "mean_steps": float(result.mean_steps),
        "team_food_per_100_steps": float(result.team_food_per_100_steps),
        "mean_food_eaten_per_fish": float(result.mean_food_eaten_per_fish),
        "mean_visible_food_count_per_fish": float(result.mean_visible_food_count_per_fish),
    }


def compare_team_results(candidate: TeamForagingEvalResult, incumbent: TeamForagingEvalResult | None) -> int:
    if incumbent is None:
        return 1
    candidate_key = (
        float(candidate.mean_team_food_eaten),
        float(candidate.team_food_per_100_steps),
        float(candidate.mean_team_reward),
    )
    incumbent_key = (
        float(incumbent.mean_team_food_eaten),
        float(incumbent.team_food_per_100_steps),
        float(incumbent.mean_team_reward),
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


def compute_deterministic_action(
    algo,
    obs: np.ndarray,
    *,
    stack_mode: str = "old",
    policy_id: str = SHARED_POLICY_ID,
) -> np.ndarray:
    if stack_mode == "new":
        action = _compute_action_new_stack(algo, obs)
    else:
        action = algo.compute_single_action(obs, policy_id=policy_id, explore=False)
        if isinstance(action, tuple):
            action = action[0]
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size != 2:
        return np.zeros(2, dtype=np.float32)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _evaluate_multi_agent_rollouts(
    *,
    env_factory,
    num_episodes: int,
    base_seed: int,
    action_fn,
) -> TeamForagingEvalResult:
    episode_team_rewards: list[float] = []
    episode_steps: list[int] = []
    episode_team_food: list[int] = []
    episode_food_per_fish: list[float] = []
    episode_visible_food_per_fish: list[float] = []

    env = env_factory()
    try:
        for episode_idx in range(num_episodes):
            obs_dict, _ = env.reset(seed=base_seed + episode_idx)
            total_team_reward = 0.0
            steps = 0
            team_food = 0
            visible_food_accum = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action_dict = action_fn(env=env, obs_dict=obs_dict)
                obs_dict, rewards, terminateds, truncateds, infos = env.step(action_dict)
                first_agent_id = next(iter(env.get_agent_ids()))
                total_team_reward += float(rewards[first_agent_id])
                team_food += int(infos[first_agent_id].get("team_food_eaten_this_step", 0))
                visible_food_accum += float(
                    np.mean([info.get("visible_food_count", 0.0) for info in infos.values()])
                )
                steps += 1
                terminated = bool(terminateds["__all__"])
                truncated = bool(truncateds["__all__"])

            num_agents = max(len(env.get_agent_ids()), 1)
            episode_team_rewards.append(total_team_reward)
            episode_steps.append(steps)
            episode_team_food.append(team_food)
            episode_food_per_fish.append(float(team_food) / float(num_agents))
            episode_visible_food_per_fish.append(visible_food_accum / float(max(steps, 1)))
    finally:
        env.close()

    mean_steps = float(np.mean(episode_steps)) if episode_steps else float("nan")
    mean_team_food_eaten = float(np.mean(episode_team_food)) if episode_team_food else float("nan")
    return TeamForagingEvalResult(
        episodes=int(num_episodes),
        mean_team_food_eaten=mean_team_food_eaten,
        mean_team_reward=float(np.mean(episode_team_rewards)) if episode_team_rewards else float("nan"),
        mean_steps=mean_steps,
        team_food_per_100_steps=(100.0 * mean_team_food_eaten / mean_steps) if mean_steps > 0.0 else float("nan"),
        mean_food_eaten_per_fish=float(np.mean(episode_food_per_fish)) if episode_food_per_fish else float("nan"),
        mean_visible_food_count_per_fish=float(np.mean(episode_visible_food_per_fish))
        if episode_visible_food_per_fish
        else float("nan"),
    )


def evaluate_multi_agent_rollouts(
    *,
    algo,
    env_factory,
    num_episodes: int,
    base_seed: int,
    stack_mode: str = "old",
    policy_id: str = SHARED_POLICY_ID,
) -> TeamForagingEvalResult:
    return _evaluate_multi_agent_rollouts(
        env_factory=env_factory,
        num_episodes=num_episodes,
        base_seed=base_seed,
        action_fn=lambda *, env, obs_dict: {
            agent_id: compute_deterministic_action(
                algo,
                obs,
                stack_mode=stack_mode,
                policy_id=policy_id,
            )
            for agent_id, obs in obs_dict.items()
        },
    )


def evaluate_multi_agent_random_rollouts(
    *,
    env_factory,
    num_episodes: int,
    base_seed: int,
) -> TeamForagingEvalResult:
    return _evaluate_multi_agent_rollouts(
        env_factory=env_factory,
        num_episodes=num_episodes,
        base_seed=base_seed,
        action_fn=lambda *, env, obs_dict: {
            agent_id: env.action_space.sample().astype(np.float32)
            for agent_id in obs_dict.keys()
        },
    )
