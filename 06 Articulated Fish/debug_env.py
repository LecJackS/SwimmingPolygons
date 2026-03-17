"""Environment-only validation harness for V6 continuous foraging."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from triangles import EEL_3SEG_PRESET, OctopusEnv


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V6 continuous-foraging environment probes.")
    parser.add_argument("--probe-set", type=str, default="baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def default_output_dir(probe_set: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SCRIPT_DIR / "media" / "debug_reports" / f"{probe_set}_{stamp}"


def build_env(*, seed: int, render_mode: str | None = None, dt: float | None = None, **kwargs) -> OctopusEnv:
    preset = EEL_3SEG_PRESET
    if dt is not None:
        preset = replace(
            preset,
            dynamics=replace(preset.dynamics, dt=float(dt)),
        )
    env = OctopusEnv(
        render_mode=render_mode,
        fish_preset=preset,
        epsilon=0.0,
        time_limit=kwargs.pop("time_limit", 600),
        food_count=kwargs.pop("food_count", 48),
        food_capture_radius=kwargs.pop("food_capture_radius", 0.45),
        pellet_reward=kwargs.pop("pellet_reward", 1.0),
        step_cost=kwargs.pop("step_cost", 0.002),
        sensor_radius=kwargs.pop("sensor_radius", 4.5),
        sensor_ring_edges=kwargs.pop("sensor_ring_edges", (1.5, 3.0, 4.5)),
        sensor_num_sectors=kwargs.pop("sensor_num_sectors", 12),
        show_sensor_overlay=kwargs.pop("show_sensor_overlay", False),
    )
    env.reset(seed=seed)
    return env


def pad_food_positions(env: OctopusEnv, lead_positions: list[list[float]]) -> np.ndarray:
    lead = [np.asarray(position, dtype=np.float32) for position in lead_positions]
    positions: list[np.ndarray] = lead[:]
    radius = max(env.sensor_radius + 1.0, 8.0)
    angle_count = max(env.food_count, 16)
    for idx in range(angle_count * 4):
        if len(positions) >= env.food_count:
            break
        angle = (2.0 * np.pi * idx) / float(angle_count)
        candidate = np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32)
        candidate = np.clip(candidate, -env.playable_half_extent, env.playable_half_extent).astype(np.float32)
        too_close = any(float(np.linalg.norm(candidate - existing)) < 0.6 for existing in positions)
        if not too_close:
            positions.append(candidate)
    while len(positions) < env.food_count:
        positions.append(np.array([env.playable_half_extent, env.playable_half_extent], dtype=np.float32))
    return np.asarray(positions[: env.food_count], dtype=np.float32)


def set_canonical_state(env: OctopusEnv, *, theta: float = 0.0, food_positions: np.ndarray | None = None) -> None:
    if food_positions is None:
        food_positions = pad_food_positions(env, [[8.0, 0.0]])
    env.set_debug_state(
        position=[0.0, 0.0],
        velocity=[0.0, 0.0],
        theta=theta,
        omega=0.0,
        food_positions=food_positions,
        timestep=0,
        joint_angles=[0.0, 0.0],
        joint_velocities=[0.0, 0.0],
        prev_action=[0.0, 0.0],
        swim_phase=0.0,
    )


def rollout_metrics(env: OctopusEnv, action: np.ndarray, *, steps: int) -> dict[str, Any]:
    total_reward = 0.0
    total_food_eaten = 0
    terminated = False
    truncated = False
    last_info: dict[str, Any] = {}
    for _ in range(steps):
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_food_eaten += int(info.get("food_eaten_this_step", 0))
        last_info = info
        if terminated or truncated:
            break
    snapshot = env.get_debug_snapshot()
    return {
        "final_x": float(snapshot["root_position"][0]),
        "final_y": float(snapshot["root_position"][1]),
        "heading": float(snapshot["root_theta"]),
        "final_speed": float(np.linalg.norm(snapshot["root_velocity"])),
        "reward_sum": float(total_reward),
        "food_eaten": int(total_food_eaten),
        "visible_food_count": int(last_info.get("visible_food_count", 0)),
        "steps_executed": int(env.timestep),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def probe_joint_rest_decay(seed: int) -> dict[str, Any]:
    env = build_env(seed=seed)
    try:
        initial_root_speed = float(np.linalg.norm(np.asarray([2.0, -0.5], dtype=np.float32)))
        env.set_debug_state(
            position=[0.0, 0.0],
            velocity=[2.0, -0.5],
            theta=0.0,
            omega=1.2,
            food_positions=pad_food_positions(env, [[8.0, 0.0]]),
            timestep=0,
            joint_angles=[0.6, -0.4],
            joint_velocities=[2.0, -1.5],
            prev_action=[-1.0, 0.0],
            swim_phase=0.0,
        )
        root_speeds = []
        joint_speeds = []
        for _ in range(120):
            env.step(np.array([-1.0, 0.0], dtype=np.float32))
            snapshot = env.get_debug_snapshot()
            root_speeds.append(float(np.linalg.norm(snapshot["root_velocity"])))
            joint_speeds.append(float(np.linalg.norm(snapshot["joint_velocities"])))
        return {
            "pass": bool(
                root_speeds[-1] < 0.1
                and joint_speeds[-1] < 0.1
                and max(root_speeds) <= initial_root_speed + 1e-6
            ),
            "final_root_speed": float(root_speeds[-1]),
            "final_joint_speed_norm": float(joint_speeds[-1]),
            "max_root_speed": float(max(root_speeds)),
            "max_joint_speed_norm": float(max(joint_speeds)),
            "initial_root_speed": float(initial_root_speed),
        }
    finally:
        env.close()


def probe_propulsion_grid(seed: int) -> dict[str, Any]:
    env = build_env(seed=seed)
    try:
        trials = []
        for drive, steer in [(-1.0, 0.0), (0.25, 0.0), (0.6, 0.0), (1.0, 0.0), (0.6, -0.4), (0.6, 0.4)]:
            set_canonical_state(env)
            metrics = rollout_metrics(env, np.array([drive, steer], dtype=np.float32), steps=120)
            metrics["drive"] = float(drive)
            metrics["steer"] = float(steer)
            trials.append(metrics)
        idle_forward = next(trial["final_x"] for trial in trials if trial["drive"] == -1.0)
        best_forward = max(trial["final_x"] for trial in trials)
        return {
            "pass": bool(abs(idle_forward) < 0.25 and best_forward > 3.0),
            "idle_forward_progress": float(idle_forward),
            "best_forward_progress": float(best_forward),
            "trials": trials,
        }
    finally:
        env.close()


def probe_mirror_turn(seed: int) -> dict[str, Any]:
    env = build_env(seed=seed)
    try:
        set_canonical_state(env)
        left = rollout_metrics(env, np.array([0.6, 0.6], dtype=np.float32), steps=120)
        set_canonical_state(env)
        right = rollout_metrics(env, np.array([0.6, -0.6], dtype=np.float32), steps=120)
        pass_flag = (
            abs(left["final_x"] - right["final_x"]) < 0.5
            and abs(left["final_y"] + right["final_y"]) < 0.6
            and abs(left["heading"] + right["heading"]) < 0.25
            and abs(left["heading"]) >= 0.6
            and abs(right["heading"]) >= 0.6
        )
        return {"pass": bool(pass_flag), "left": left, "right": right}
    finally:
        env.close()


def probe_steering_authority(seed: int) -> dict[str, Any]:
    env = build_env(seed=seed)
    try:
        set_canonical_state(env)
        left = rollout_metrics(env, np.array([0.6, 1.0], dtype=np.float32), steps=120)
        set_canonical_state(env)
        right = rollout_metrics(env, np.array([0.6, -1.0], dtype=np.float32), steps=120)
        pass_flag = (
            abs(left["heading"]) >= 0.6
            and abs(right["heading"]) >= 0.6
            and abs(left["heading"] + right["heading"]) < 0.25
            and abs(left["final_y"] + right["final_y"]) < 0.8
        )
        return {"pass": bool(pass_flag), "left": left, "right": right}
    finally:
        env.close()


def probe_drag_anisotropy(seed: int) -> dict[str, Any]:
    env = build_env(seed=seed)
    try:
        set_canonical_state(env)
        env.set_debug_state(
            position=[0.0, 0.0],
            velocity=[1.0, 0.0],
            theta=0.0,
            omega=0.0,
            food_positions=pad_food_positions(env, [[8.0, 0.0]]),
            timestep=0,
        )
        aligned = env.get_dynamics_breakdown(np.array([-1.0, 0.0], dtype=np.float32))
        env.set_debug_state(
            position=[0.0, 0.0],
            velocity=[0.0, 1.0],
            theta=0.0,
            omega=0.0,
            food_positions=pad_food_positions(env, [[8.0, 0.0]]),
            timestep=0,
        )
        lateral = env.get_dynamics_breakdown(np.array([-1.0, 0.0], dtype=np.float32))
        aligned_force = float(np.linalg.norm(aligned["total_force"]))
        lateral_force = float(np.linalg.norm(lateral["total_force"]))
        ratio = lateral_force / max(aligned_force, 1e-6)
        return {"pass": bool(ratio > 2.0), "aligned_force": aligned_force, "lateral_force": lateral_force, "ratio": ratio}
    finally:
        env.close()


def probe_dt_sensitivity(seed: int) -> dict[str, Any]:
    total_horizon_seconds = 4.0
    results = {}
    reference_position = None
    reference_heading = None
    for dt in (0.025, 0.05, 0.1):
        steps = int(round(total_horizon_seconds / dt))
        env = build_env(seed=seed, dt=dt)
        try:
            set_canonical_state(env)
            action = np.array([0.6, 0.2], dtype=np.float32)
            for _ in range(steps):
                env.step(action)
            snapshot = env.get_debug_snapshot()
            results[f"{dt:g}"] = {
                "position": np.asarray(snapshot["root_position"], dtype=np.float32).tolist(),
                "heading": float(snapshot["root_theta"]),
            }
            if dt == 0.05:
                reference_position = np.asarray(snapshot["root_position"], dtype=np.float32)
                reference_heading = float(snapshot["root_theta"])
        finally:
            env.close()
    assert reference_position is not None and reference_heading is not None
    errors = []
    for key, value in results.items():
        position = np.asarray(value["position"], dtype=np.float32)
        heading = float(value["heading"])
        position_error = float(np.linalg.norm(position - reference_position) / max(np.linalg.norm(reference_position), 1e-6))
        heading_error = abs(heading - reference_heading) / max(abs(reference_heading), 1e-6)
        errors.append(max(position_error, heading_error))
    return {"pass": bool(max(errors) < 0.12), "dt_max_relative_error": float(max(errors)), "traces": results}


def probe_food_field_contract(seed: int) -> dict[str, Any]:
    env = build_env(seed=seed)
    try:
        env.reset(seed=seed)
        initial_count = int(env.food_positions.shape[0])
        in_bounds = True
        no_nans = True
        count_constant = True
        for _ in range(50):
            env.step(env.action_space.sample())
            count_constant = count_constant and (env.food_positions.shape[0] == env.food_count == initial_count)
            in_bounds = in_bounds and bool(np.all(np.abs(env.food_positions) <= env.playable_half_extent + 1e-6))
            no_nans = no_nans and bool(np.isfinite(env.food_positions).all())
        set_canonical_state(env, food_positions=pad_food_positions(env, [[0.1, 0.0], [0.2, 0.1], [0.3, -0.1]]))
        _, _, _, _, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))
        respawned = bool(info["food_eaten_this_step"] == 3 and env.food_positions.shape[0] == env.food_count)
        return {
            "pass": bool(count_constant and in_bounds and no_nans and respawned),
            "count_constant": bool(count_constant),
            "in_bounds": bool(in_bounds),
            "no_nans": bool(no_nans),
            "respawned": bool(respawned),
        }
    finally:
        env.close()


def expected_bin(env: OctopusEnv, *, ring_index: int, sector_index: int) -> int:
    return ring_index * env.sensor_num_sectors + sector_index


def probe_polar_sensor_contract(seed: int) -> dict[str, Any]:
    env = build_env(seed=seed)
    try:
        lead_positions = [
            [1.0, 0.0],
            [0.0, 2.0],
            [-4.0, 0.0],
        ]
        set_canonical_state(env, theta=0.0, food_positions=pad_food_positions(env, lead_positions))
        snapshot_a = env.get_debug_snapshot()
        active_a = set(snapshot_a["sensor_active_bins"])
        expected_a = {
            expected_bin(env, ring_index=0, sector_index=0),
            expected_bin(env, ring_index=1, sector_index=3),
            expected_bin(env, ring_index=2, sector_index=6),
        }

        set_canonical_state(env, theta=np.pi / 2.0, food_positions=pad_food_positions(env, lead_positions))
        snapshot_b = env.get_debug_snapshot()
        active_b = set(snapshot_b["sensor_active_bins"])
        expected_b = {
            expected_bin(env, ring_index=0, sector_index=9),
            expected_bin(env, ring_index=1, sector_index=0),
            expected_bin(env, ring_index=2, sector_index=3),
        }
        bounds_ok = bool(np.all(snapshot_a["observation"] >= env.observation_space.low - 1e-6)) and bool(
            np.all(snapshot_a["observation"] <= env.observation_space.high + 1e-6)
        )
        finite_ok = bool(np.isfinite(snapshot_a["observation"]).all() and np.isfinite(snapshot_b["observation"]).all())
        return {
            "pass": bool(expected_a.issubset(active_a) and expected_b.issubset(active_b) and bounds_ok and finite_ok),
            "active_bins_theta0": sorted(active_a),
            "expected_theta0": sorted(expected_a),
            "active_bins_theta90": sorted(active_b),
            "expected_theta90": sorted(expected_b),
            "bounds_ok": bool(bounds_ok),
            "finite_ok": bool(finite_ok),
        }
    finally:
        env.close()


def probe_foraging_reward_contract(seed: int) -> dict[str, Any]:
    env = build_env(seed=seed)
    try:
        set_canonical_state(
            env,
            food_positions=pad_food_positions(env, [[0.1, 0.0], [0.15, 0.1], [0.2, -0.1], [0.55, 0.0]]),
        )
        _, reward, terminated, truncated, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))
        breakdown = info["reward_breakdown"]
        expected_reward = 3.0 - 0.002
        first_pass = (
            int(info["food_eaten_this_step"]) == 3
            and abs(float(reward) - expected_reward) < 1e-6
            and abs(float(breakdown["total_reward"]) - expected_reward) < 1e-6
            and not terminated
            and not truncated
        )
        _, reward_2, _, _, info_2 = env.step(np.array([-1.0, 0.0], dtype=np.float32))
        if int(info_2["food_eaten_this_step"]) == 0:
            second_pass = abs(float(reward_2) + 0.002) < 1e-6
        else:
            second_pass = True
        return {
            "pass": bool(first_pass and second_pass),
            "first_reward": float(reward),
            "first_food_eaten": int(info["food_eaten_this_step"]),
            "second_reward": float(reward_2),
            "second_food_eaten": int(info_2["food_eaten_this_step"]),
        }
    finally:
        env.close()


PROBES = {
    "joint_rest_decay": probe_joint_rest_decay,
    "propulsion_grid": probe_propulsion_grid,
    "mirror_turn": probe_mirror_turn,
    "steering_authority": probe_steering_authority,
    "drag_anisotropy": probe_drag_anisotropy,
    "dt_sensitivity": probe_dt_sensitivity,
    "food_field_contract": probe_food_field_contract,
    "polar_sensor_contract": probe_polar_sensor_contract,
    "foraging_reward_contract": probe_foraging_reward_contract,
}


BASELINE_PROBES = [
    "joint_rest_decay",
    "propulsion_grid",
    "mirror_turn",
    "steering_authority",
    "drag_anisotropy",
    "dt_sensitivity",
    "food_field_contract",
    "polar_sensor_contract",
    "foraging_reward_contract",
]


def run_probe_set(probe_set: str, seed: int) -> dict[str, Any]:
    probe_names = BASELINE_PROBES if probe_set == "baseline" else [probe_set]
    results = {}
    for index, probe_name in enumerate(probe_names):
        if probe_name not in PROBES:
            raise ValueError(f"Unknown probe set: {probe_name}")
        results[probe_name] = PROBES[probe_name](seed + (index * 10_000))
    results["overall_pass"] = bool(all(bool(result["pass"]) for result in results.values() if isinstance(result, dict)))
    return results


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.probe_set)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_probe_set(args.probe_set, args.seed)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("V6 environment validation complete")
    print(f"Probe set: {args.probe_set}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Overall pass: {summary['overall_pass']}")
    for probe_name, result in summary.items():
        if probe_name == "overall_pass":
            continue
        print(f"{probe_name}: pass={result['pass']}")


if __name__ == "__main__":
    main()
