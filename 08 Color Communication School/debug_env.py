"""Environment-only validation harness for V8 color communication schooling."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from eval_utils import (
    DEFAULT_NUM_BLUE_FISH,
    DEFAULT_NUM_BLUE_PELLETS,
    DEFAULT_NUM_RED_FISH,
    DEFAULT_NUM_RED_PELLETS,
)
from triangles import BLUE_TEAM, EEL_3SEG_PRESET, RED_TEAM, CommunicatingSchoolEnv


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V8 color-school environment probes.")
    parser.add_argument("--probe-set", type=str, default="baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def default_output_dir(probe_set: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SCRIPT_DIR / "media" / "debug_reports" / f"{probe_set}_{stamp}"


def build_env(*, seed: int, render_mode: str | None = None, **kwargs) -> CommunicatingSchoolEnv:
    env = CommunicatingSchoolEnv(
        render_mode=render_mode,
        fish_preset=EEL_3SEG_PRESET,
        epsilon=0.0,
        time_limit=kwargs.pop("time_limit", 120),
        num_red_fish=kwargs.pop("num_red_fish", DEFAULT_NUM_RED_FISH),
        num_blue_fish=kwargs.pop("num_blue_fish", DEFAULT_NUM_BLUE_FISH),
        num_red_pellets=kwargs.pop("num_red_pellets", DEFAULT_NUM_RED_PELLETS),
        num_blue_pellets=kwargs.pop("num_blue_pellets", DEFAULT_NUM_BLUE_PELLETS),
        food_capture_radius=kwargs.pop("food_capture_radius", 0.45),
        pellet_reward=kwargs.pop("pellet_reward", 1.0),
        step_cost=kwargs.pop("step_cost", 0.002),
        sector_radius=kwargs.pop("sector_radius", 5.0),
        sector_num=kwargs.pop("sector_num", 6),
        communication_radius=kwargs.pop("communication_radius", 5.0),
        show_sensor_overlay=kwargs.pop("show_sensor_overlay", False),
        focus_agent_id=kwargs.pop("focus_agent_id", "fish_0"),
        mute_received_messages=kwargs.pop("mute_received_messages", False),
    )
    env.reset(seed=seed)
    return env


def probe_color_food_field_contract(seed: int) -> dict[str, Any]:
    env = build_env(seed=seed)
    try:
        env.reset(seed=seed)
        bounds_ok = True
        color_counts_ok = True
        for _ in range(40):
            actions = {agent_id: {"motion": np.zeros(2, dtype=np.float32), "message": 0} for agent_id in env.get_agent_ids()}
            env.step(actions)
            bounds_ok = bounds_ok and bool(np.all(np.abs(env.food_positions) <= env.playable_half_extent + 1e-6))
            color_counts_ok = color_counts_ok and int(np.sum(env.food_team_indices == RED_TEAM)) == env.num_red_pellets
            color_counts_ok = color_counts_ok and int(np.sum(env.food_team_indices == BLUE_TEAM)) == env.num_blue_pellets
        return {
            "pass": bool(bounds_ok and color_counts_ok),
            "bounds_ok": bool(bounds_ok),
            "red_pellets": int(np.sum(env.food_team_indices == RED_TEAM)),
            "blue_pellets": int(np.sum(env.food_team_indices == BLUE_TEAM)),
        }
    finally:
        env.close()


def probe_food_spawn_distribution_contract(seed: int) -> dict[str, Any]:
    env = build_env(seed=seed)
    try:
        num_samples = 24
        quadrant_hits = {
            "red": np.zeros(4, dtype=np.int32),
            "blue": np.zeros(4, dtype=np.int32),
        }
        red_centroids: list[np.ndarray] = []
        blue_centroids: list[np.ndarray] = []
        layout_signatures: list[float] = []
        bounds_ok = True
        spacing_ok = True
        fish_clearance_ok = True

        for sample_idx in range(num_samples):
            env.reset(seed=seed + sample_idx)
            fish_positions = np.asarray(
                [env.fish_states[agent_id].root_position for agent_id in env.get_agent_ids()],
                dtype=np.float32,
            )
            positions = env.food_positions.astype(np.float32)
            teams = env.food_team_indices.astype(np.int64)
            bounds_ok = bounds_ok and bool(np.all(np.abs(positions) <= env.playable_half_extent + 1e-6))

            if len(positions) > 1:
                deltas = positions[:, None, :] - positions[None, :, :]
                distances = np.linalg.norm(deltas, axis=-1)
                upper = distances[np.triu_indices(len(positions), k=1)]
                if upper.size:
                    spacing_ok = spacing_ok and bool(np.min(upper) >= env.food_min_spacing - 1e-6)

            if fish_positions.size:
                pellet_to_fish = np.linalg.norm(positions[:, None, :] - fish_positions[None, :, :], axis=-1)
                fish_clearance_ok = fish_clearance_ok and bool(
                    float(np.min(pellet_to_fish)) >= env.food_min_spawn_distance - 1e-6
                )

            for team_name, team_index in (("red", RED_TEAM), ("blue", BLUE_TEAM)):
                team_positions = positions[teams == team_index]
                if team_positions.size == 0:
                    continue
                centroid = np.mean(team_positions, axis=0)
                if team_name == "red":
                    red_centroids.append(centroid)
                else:
                    blue_centroids.append(centroid)
                for pellet in team_positions:
                    quadrant = (1 if pellet[0] >= 0.0 else 0) + (2 if pellet[1] >= 0.0 else 0)
                    quadrant_hits[team_name][quadrant] += 1
                layout_signatures.append(float(np.mean(team_positions[:, 0]) + (0.5 * np.mean(team_positions[:, 1]))))

        red_centroids_arr = np.asarray(red_centroids, dtype=np.float32)
        blue_centroids_arr = np.asarray(blue_centroids, dtype=np.float32)
        centroid_deltas = np.linalg.norm(red_centroids_arr - blue_centroids_arr, axis=1)
        centroid_separation_mean = float(np.mean(centroid_deltas)) if centroid_deltas.size else float("inf")
        centroid_separation_max = float(np.max(centroid_deltas)) if centroid_deltas.size else float("inf")
        red_centroid_std = float(np.linalg.norm(np.std(red_centroids_arr, axis=0))) if red_centroids_arr.size else 0.0
        blue_centroid_std = float(np.linalg.norm(np.std(blue_centroids_arr, axis=0))) if blue_centroids_arr.size else 0.0
        layout_signature_std = float(np.std(np.asarray(layout_signatures, dtype=np.float32))) if layout_signatures else 0.0

        quadrant_coverage_ok = bool(
            np.all(quadrant_hits["red"] > 0)
            and np.all(quadrant_hits["blue"] > 0)
        )
        centroid_mix_ok = bool(
            centroid_separation_mean <= (0.35 * env.playable_half_extent)
            and centroid_separation_max <= (0.60 * env.playable_half_extent)
        )
        variation_ok = bool(
            red_centroid_std >= 0.25
            and blue_centroid_std >= 0.25
            and layout_signature_std >= 0.25
        )

        return {
            "pass": bool(
                quadrant_coverage_ok
                and centroid_mix_ok
                and variation_ok
                and bounds_ok
                and spacing_ok
                and fish_clearance_ok
            ),
            "num_samples": int(num_samples),
            "quadrant_hits_red": quadrant_hits["red"].astype(int).tolist(),
            "quadrant_hits_blue": quadrant_hits["blue"].astype(int).tolist(),
            "quadrant_coverage_ok": quadrant_coverage_ok,
            "centroid_separation_mean": centroid_separation_mean,
            "centroid_separation_max": centroid_separation_max,
            "centroid_mix_ok": centroid_mix_ok,
            "red_centroid_std": red_centroid_std,
            "blue_centroid_std": blue_centroid_std,
            "layout_signature_std": layout_signature_std,
            "variation_ok": variation_ok,
            "bounds_ok": bounds_ok,
            "spacing_ok": spacing_ok,
            "fish_clearance_ok": fish_clearance_ok,
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
            "pass": bool(min_distance >= (env.min_spawn_separation - 1e-6)),
            "min_spawn_distance": float(min_distance),
            "required_min_distance": float(env.min_spawn_separation),
        }
    finally:
        env.close()


def probe_color_matched_consumption_contract(seed: int) -> dict[str, Any]:
    env = build_env(seed=seed, num_red_pellets=2, num_blue_pellets=2)
    try:
        red_focus = "fish_0"
        blue_focus = f"fish_{env.num_red_fish}"
        far = np.full((env.food_count, 2), 9.0, dtype=np.float32)
        states = {agent_id: {"position": [8.0, 8.0], "theta": 0.0} for agent_id in env.get_agent_ids()}
        states[red_focus] = {"position": [0.0, 0.0], "theta": 0.0}
        states[blue_focus] = {"position": [2.0, 0.0], "theta": 0.0}
        far[0] = np.array([0.0, 0.0], dtype=np.float32)   # red pellet
        far[1] = np.array([0.0, 0.2], dtype=np.float32)   # red pellet
        far[2] = np.array([2.0, 0.0], dtype=np.float32)   # blue pellet
        far[3] = np.array([0.0, 0.0], dtype=np.float32)   # blue pellet near red fish, should not be eaten
        food_teams = [RED_TEAM, RED_TEAM, BLUE_TEAM, BLUE_TEAM]
        env.set_debug_state(agent_states=states, food_positions=far, food_team_indices=food_teams)
        actions = {agent_id: {"motion": np.zeros(2, dtype=np.float32), "message": 0} for agent_id in env.get_agent_ids()}
        _, rewards, _, _, infos = env.step(actions)
        return {
            "pass": bool(
                infos[red_focus]["food_eaten_this_step"] == 2
                and infos[blue_focus]["food_eaten_this_step"] == 1
                and abs(rewards[red_focus] - ((2.0 * env.pellet_reward) - env.step_cost)) < 1e-6
                and abs(rewards[blue_focus] - (env.pellet_reward - env.step_cost)) < 1e-6
            ),
            "reward_red_focus": float(rewards[red_focus]),
            "reward_blue_focus": float(rewards[blue_focus]),
            "red_food_step": int(infos[red_focus]["food_eaten_this_step"]),
            "blue_food_step": int(infos[blue_focus]["food_eaten_this_step"]),
        }
    finally:
        env.close()


def probe_communication_locality_contract(seed: int) -> dict[str, Any]:
    env = build_env(seed=seed)
    try:
        red_focus = "fish_0"
        red_teammate = "fish_1"
        blue_opponent = f"fish_{env.num_red_fish}"
        blue_far = f"fish_{env.num_red_fish + 1}"
        states = {agent_id: {"position": [8.0, 8.0], "theta": 0.0} for agent_id in env.get_agent_ids()}
        states[red_focus] = {"position": [0.0, 0.0], "theta": 0.0}
        states[red_teammate] = {"position": [2.0, 0.0], "theta": 0.0}   # red teammate
        states[blue_opponent] = {"position": [0.0, 2.0], "theta": 0.0}  # blue opponent
        states[blue_far] = {"position": [7.0, 0.0], "theta": 0.0}       # outside radius
        food_positions = np.full((env.food_count, 2), 9.0, dtype=np.float32)
        env.set_debug_state(agent_states=states, food_positions=food_positions)
        before = env.get_debug_snapshot(red_focus)
        actions = {
            agent_id: {"motion": np.zeros(2, dtype=np.float32), "message": 0}
            for agent_id in env.get_agent_ids()
        }
        actions[red_teammate]["message"] = 3
        actions[blue_opponent]["message"] = 2
        actions[blue_far]["message"] = 3
        obs, _, _, _, _ = env.step(actions)
        after = obs[red_focus]
        teammate_msg = after[24:30]
        opponent_msg = after[30:36]
        return {
            "pass": bool(
                np.allclose(before["observation"][24:36], 0.0)
                and float(np.max(teammate_msg)) > 0.9
                and float(np.max(opponent_msg)) > 0.6
                and int(np.count_nonzero(opponent_msg > 0.0)) == 1
            ),
            "teammate_message_bins": teammate_msg.tolist(),
            "opponent_message_bins": opponent_msg.tolist(),
        }
    finally:
        env.close()


def probe_observation_contract(seed: int) -> dict[str, Any]:
    env = build_env(seed=seed)
    try:
        red_focus = "fish_0"
        red_teammate = "fish_1"
        blue_opponent = f"fish_{env.num_red_fish}"
        states = {agent_id: {"position": [8.0, 8.0], "theta": 0.0} for agent_id in env.get_agent_ids()}
        states[red_focus] = {"position": [0.0, 0.0], "theta": 0.0}
        states[red_teammate] = {"position": [2.0, 0.0], "theta": 0.0}
        states[blue_opponent] = {"position": [0.0, -2.0], "theta": 0.0}
        food_positions = np.full((env.food_count, 2), 9.0, dtype=np.float32)
        food_positions[0] = np.array([1.0, 0.0], dtype=np.float32)   # edible for red focus
        food_positions[-1] = np.array([0.0, -1.0], dtype=np.float32) # non-edible for red focus
        env.set_debug_state(agent_states=states, food_positions=food_positions)
        snapshot = env.get_debug_snapshot(red_focus)
        obs = snapshot["observation"]
        return {
            "pass": bool(
                obs.shape == (48,)
                and np.all(np.isfinite(obs))
                and np.all(obs >= env.observation_space.low - 1e-6)
                and np.all(obs <= env.observation_space.high + 1e-6)
                and obs[0] > 0.0
                and obs[6:12].sum() > 0.0
                and obs[12:18].sum() > 0.0
                and obs[18:24].sum() > 0.0
            ),
            "observation_shape": list(obs.shape),
            "channel_sums": {
                "edible_food": float(np.sum(obs[0:6])),
                "non_edible_food": float(np.sum(obs[6:12])),
                "teammate": float(np.sum(obs[12:18])),
                "opponent": float(np.sum(obs[18:24])),
            },
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
            agent_id: {
                "motion": np.array([0.4 if idx % 2 == 0 else -0.2, 0.3 if idx % 3 == 0 else -0.1], dtype=np.float32),
                "message": int(idx % 4),
            }
            for idx, agent_id in enumerate(env_a.get_agent_ids())
        }
        for _ in range(30):
            env_a.step(action_dict)
            env_b.step(action_dict)
        positions_a = np.asarray([env_a.fish_states[agent_id].root_position for agent_id in env_a.get_agent_ids()], dtype=np.float32)
        positions_b = np.asarray([env_b.fish_states[agent_id].root_position for agent_id in env_b.get_agent_ids()], dtype=np.float32)
        return {
            "pass": bool(np.allclose(positions_a, positions_b, atol=1e-6) and np.allclose(env_a.food_positions, env_b.food_positions, atol=1e-6)),
            "foods_match": bool(np.allclose(env_a.food_positions, env_b.food_positions, atol=1e-6)),
            "states_match": bool(np.allclose(positions_a, positions_b, atol=1e-6)),
        }
    finally:
        env_a.close()
        env_b.close()


PROBES = {
    "color_food_field_contract": probe_color_food_field_contract,
    "food_spawn_distribution_contract": probe_food_spawn_distribution_contract,
    "spawn_separation": probe_spawn_separation,
    "color_matched_consumption_contract": probe_color_matched_consumption_contract,
    "communication_locality_contract": probe_communication_locality_contract,
    "observation_contract": probe_observation_contract,
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
