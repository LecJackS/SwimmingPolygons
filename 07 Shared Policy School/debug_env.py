"""Environment-only validation harness for V7 shared-policy schooling."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from triangles import EEL_3SEG_PRESET, SchoolingFishEnv


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V7 shared-school environment probes.")
    parser.add_argument("--probe-set", type=str, default="baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def default_output_dir(probe_set: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SCRIPT_DIR / "media" / "debug_reports" / f"{probe_set}_{stamp}"


def build_env(*, seed: int, render_mode: str | None = None, **kwargs) -> SchoolingFishEnv:
    env = SchoolingFishEnv(
        render_mode=render_mode,
        fish_preset=EEL_3SEG_PRESET,
        epsilon=0.0,
        time_limit=kwargs.pop("time_limit", 200),
        food_count=kwargs.pop("food_count", 48),
        food_capture_radius=kwargs.pop("food_capture_radius", 0.45),
        pellet_reward=kwargs.pop("pellet_reward", 1.0),
        step_cost=kwargs.pop("step_cost", 0.002),
        sensor_radius=kwargs.pop("sensor_radius", 4.5),
        sensor_ring_edges=kwargs.pop("sensor_ring_edges", (1.5, 3.0, 4.5)),
        sensor_num_sectors=kwargs.pop("sensor_num_sectors", 12),
        show_sensor_overlay=kwargs.pop("show_sensor_overlay", False),
        num_fish=kwargs.pop("num_fish", 8),
        focus_agent_id=kwargs.pop("focus_agent_id", "fish_0"),
    )
    env.reset(seed=seed)
    return env


def far_food_positions(env: SchoolingFishEnv) -> np.ndarray:
    positions = []
    radius = env.playable_half_extent
    for idx in range(env.food_count):
        angle = (2.0 * np.pi * idx) / float(max(env.food_count, 1))
        positions.append(
            np.array(
                [radius * np.cos(angle), radius * np.sin(angle)],
                dtype=np.float32,
            )
        )
    return np.asarray(positions, dtype=np.float32)


def expected_bin(env: SchoolingFishEnv, *, ring_index: int, sector_index: int) -> int:
    return (ring_index * env.sensor_num_sectors) + sector_index


def probe_food_field_contract(seed: int) -> dict[str, Any]:
    env = build_env(seed=seed)
    try:
        env.reset(seed=seed)
        count_ok = True
        bounds_ok = True
        for _ in range(80):
            action_dict = {agent_id: np.zeros(2, dtype=np.float32) for agent_id in env.get_agent_ids()}
            _, _, _, _, infos = env.step(action_dict)
            count_ok = count_ok and env.food_positions.shape == (env.food_count, 2)
            bounds_ok = bounds_ok and bool(np.all(np.abs(env.food_positions) <= env.playable_half_extent + 1e-6))
            if infos["fish_0"]["team_food_eaten_this_step"] < 0:
                return {"pass": False, "reason": "negative food count"}
        return {
            "pass": bool(count_ok and bounds_ok),
            "food_count": int(env.food_positions.shape[0]),
            "bounds_ok": bool(bounds_ok),
        }
    finally:
        env.close()


def probe_spawn_separation(seed: int) -> dict[str, Any]:
    env = build_env(seed=seed)
    try:
        env.reset(seed=seed)
        positions = np.asarray([env.fish_states[agent_id].root_position for agent_id in env.get_agent_ids()], dtype=np.float32)
        min_distance = float("inf")
        for idx in range(len(positions)):
            for jdx in range(idx + 1, len(positions)):
                min_distance = min(min_distance, float(np.linalg.norm(positions[idx] - positions[jdx])))
        return {
            "pass": bool(min_distance >= env.min_spawn_separation),
            "min_spawn_distance": float(min_distance),
            "required_min_distance": float(env.min_spawn_separation),
        }
    finally:
        env.close()


def probe_peer_sensor_contract(seed: int) -> dict[str, Any]:
    env = build_env(seed=seed)
    try:
        food_positions = far_food_positions(env)
        states = {
            "fish_0": {"position": [0.0, 0.0], "theta": 0.0},
            "fish_1": {"position": [1.0, 0.0], "theta": 0.0},
            "fish_2": {"position": [0.0, 1.0], "theta": 0.0},
        }
        for idx in range(3, env.num_fish):
            states[f"fish_{idx}"] = {"position": [6.0 + (0.5 * (idx - 3)), 6.0], "theta": 0.0}
        env.set_debug_state(agent_states=states, food_positions=food_positions, focus_agent_id="fish_0")
        snapshot = env.get_debug_snapshot("fish_0")
        peer_bins = snapshot["peer_sensor_bins"]
        front_bin = expected_bin(env, ring_index=0, sector_index=0)
        left_bin = expected_bin(env, ring_index=0, sector_index=3)
        first_pass = peer_bins[front_bin] > 0.0 and peer_bins[left_bin] > 0.0

        states["fish_0"]["theta"] = float(np.pi / 2.0)
        env.set_debug_state(agent_states=states, food_positions=food_positions, focus_agent_id="fish_0")
        rotated = env.get_debug_snapshot("fish_0")["peer_sensor_bins"]
        right_bin = expected_bin(env, ring_index=0, sector_index=9)
        rotated_pass = rotated[right_bin] > 0.0 and rotated[front_bin] > 0.0

        obs = env.get_debug_snapshot("fish_0")["observation"]
        obs_low = env.observation_space.low
        obs_high = env.observation_space.high
        return {
            "pass": bool(
                first_pass
                and rotated_pass
                and np.all(np.isfinite(obs))
                and np.all(obs >= (obs_low - 1e-6))
                and np.all(obs <= (obs_high + 1e-6))
            ),
            "front_bin": int(front_bin),
            "left_bin": int(left_bin),
            "right_bin_after_rotation": int(right_bin),
            "active_bins_initial": snapshot["peer_sensor_active_bins"],
        }
    finally:
        env.close()


def probe_shared_reward_contract(seed: int) -> dict[str, Any]:
    env = build_env(seed=seed)
    try:
        food_positions = np.full((env.food_count, 2), -env.playable_half_extent, dtype=np.float32)
        states = {}
        for idx in range(env.num_fish):
            states[f"fish_{idx}"] = {"position": [6.0 + (0.5 * idx), 6.0], "theta": 0.0}
        states["fish_0"]["position"] = [0.0, 0.0]
        states["fish_1"]["position"] = [2.0, 0.0]
        food_positions[0] = np.array([0.0, 0.0], dtype=np.float32)
        food_positions[1] = np.array([2.0, 0.0], dtype=np.float32)
        env.set_debug_state(agent_states=states, food_positions=food_positions)
        actions = {agent_id: np.zeros(2, dtype=np.float32) for agent_id in env.get_agent_ids()}
        _, rewards, _, truncateds, infos = env.step(actions)
        reward_values = list(rewards.values())
        expected_reward = float((2.0 * env.pellet_reward) - env.step_cost)
        identical = bool(all(abs(value - reward_values[0]) < 1e-6 for value in reward_values))
        assigned = infos["fish_0"]["food_eaten_this_step"] == 1 and infos["fish_1"]["food_eaten_this_step"] == 1
        return {
            "pass": bool(
                identical
                and assigned
                and abs(reward_values[0] - expected_reward) < 1e-6
                and not truncateds["__all__"]
            ),
            "reward": float(reward_values[0]),
            "expected_reward": expected_reward,
            "team_food_eaten_this_step": int(infos["fish_0"]["team_food_eaten_this_step"]),
        }
    finally:
        env.close()


def probe_simultaneous_step_consistency(seed: int) -> dict[str, Any]:
    env_a = build_env(seed=seed)
    env_b = build_env(seed=seed)
    try:
        env_a.reset(seed=seed)
        env_b.reset(seed=seed)
        action_dict = {
            agent_id: np.array([0.4 if idx % 2 == 0 else -0.2, 0.3 if idx % 3 == 0 else -0.1], dtype=np.float32)
            for idx, agent_id in enumerate(env_a.get_agent_ids())
        }
        for _ in range(40):
            env_a.step(action_dict)
            env_b.step(action_dict)
        positions_a = np.asarray([env_a.fish_states[agent_id].root_position for agent_id in env_a.get_agent_ids()], dtype=np.float32)
        positions_b = np.asarray([env_b.fish_states[agent_id].root_position for agent_id in env_b.get_agent_ids()], dtype=np.float32)
        foods_match = bool(np.allclose(env_a.food_positions, env_b.food_positions, atol=1e-6))
        states_match = bool(np.allclose(positions_a, positions_b, atol=1e-6))
        return {
            "pass": bool(foods_match and states_match),
            "foods_match": bool(foods_match),
            "states_match": bool(states_match),
        }
    finally:
        env_a.close()
        env_b.close()


PROBES = {
    "food_field_contract": probe_food_field_contract,
    "spawn_separation": probe_spawn_separation,
    "peer_sensor_contract": probe_peer_sensor_contract,
    "shared_reward_contract": probe_shared_reward_contract,
    "simultaneous_step_consistency": probe_simultaneous_step_consistency,
}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.probe_set)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.probe_set == "baseline":
        results = {name: probe(args.seed) for name, probe in PROBES.items()}
    else:
        if args.probe_set not in PROBES:
            raise ValueError(f"Unknown probe set: {args.probe_set}")
        results = {args.probe_set: PROBES[args.probe_set](args.seed)}

    overall_pass = bool(all(result.get("pass", False) for result in results.values()))
    summary = {
        "probe_set": args.probe_set,
        "seed": int(args.seed),
        "overall_pass": overall_pass,
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
