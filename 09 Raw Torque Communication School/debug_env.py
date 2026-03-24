"""Environment-only validation harness for V9 raw-torque schooling."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from eval_utils import (
    DEFAULT_NUM_BLUE_FISH,
    DEFAULT_NUM_BLUE_PELLETS,
    DEFAULT_NUM_RED_FISH,
    DEFAULT_NUM_RED_PELLETS,
)
from triangles import EEL_3SEG_PRESET, CommunicatingSchoolEnv


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V9 raw-torque environment probes.")
    parser.add_argument("--probe-set", type=str, default="baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def default_output_dir(probe_set: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SCRIPT_DIR / "media" / "debug_reports" / f"{probe_set}_{stamp}"


def build_env(*, seed: int, reward_mode: str = "forage", render_mode: str | None = None, **kwargs) -> CommunicatingSchoolEnv:
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
        reward_mode=reward_mode,
        history_length=kwargs.pop("history_length", 8),
        actuator_time_constant=kwargs.pop("actuator_time_constant", 0.10),
        show_sensor_overlay=kwargs.pop("show_sensor_overlay", False),
        focus_agent_id=kwargs.pop("focus_agent_id", "fish_0"),
        mute_received_messages=kwargs.pop("mute_received_messages", False),
    )
    env.reset(seed=seed)
    return env


def build_single_fish_debug_env(seed: int) -> CommunicatingSchoolEnv:
    return build_env(
        seed=seed,
        reward_mode="locomotion_debug",
        time_limit=180,
        num_red_fish=1,
        num_blue_fish=0,
        num_red_pellets=0,
        num_blue_pellets=0,
    )


def zero_action_dict(env: CommunicatingSchoolEnv) -> dict[str, dict[str, Any]]:
    return {
        agent_id: {"motion": np.zeros(2, dtype=np.float32), "message": 0}
        for agent_id in env.get_agent_ids()
    }


def scripted_wave_action(step_idx: int, *, amplitude: float = 0.9, phase_delta: float = math.pi / 2.0, bias: float = 0.0) -> np.ndarray:
    phase = 0.32 * float(step_idx)
    torque_0 = amplitude * math.sin(phase) + bias
    torque_1 = amplitude * math.sin(phase - phase_delta) - bias
    return np.clip(np.array([torque_0, torque_1], dtype=np.float32), -1.0, 1.0)


def action_dict_for_motion(env: CommunicatingSchoolEnv, motion: np.ndarray, *, message: int = 0) -> dict[str, dict[str, Any]]:
    return {
        agent_id: {"motion": np.asarray(motion, dtype=np.float32).reshape(2), "message": int(message)}
        for agent_id in env.get_agent_ids()
    }


def probe_torque_rest_decay(seed: int) -> dict[str, Any]:
    env = build_single_fish_debug_env(seed)
    try:
        env.set_debug_state(
            agent_states={
                "fish_0": {
                    "position": [0.0, 0.0],
                    "theta": 0.0,
                    "joint_angles": [0.25, -0.20],
                    "joint_velocities": [1.25, -1.10],
                    "applied_joint_torque": [0.0, 0.0],
                }
            },
            food_positions=np.zeros((0, 2), dtype=np.float32),
            food_team_indices=np.zeros(0, dtype=np.int64),
        )
        for _ in range(60):
            env.step(zero_action_dict(env))
        snapshot = env.get_debug_snapshot("fish_0")
        final_joint_speed = float(np.max(np.abs(snapshot["joint_velocities"])))
        final_speed = float(np.linalg.norm(snapshot["root_velocity"]))
        final_torque = float(np.max(np.abs(snapshot["applied_joint_torque"])))
        return {
            "pass": bool(final_joint_speed < 0.12 and final_speed < 0.12 and final_torque < 1e-5),
            "final_joint_speed": final_joint_speed,
            "final_root_speed": final_speed,
            "final_abs_applied_torque": final_torque,
        }
    finally:
        env.close()


def probe_no_hidden_drive_assist(seed: int) -> dict[str, Any]:
    env = build_single_fish_debug_env(seed)
    try:
        env.set_debug_state(
            agent_states={"fish_0": {"position": [0.0, 0.0], "theta": 0.0}},
            food_positions=np.zeros((0, 2), dtype=np.float32),
            food_team_indices=np.zeros(0, dtype=np.int64),
        )
        start_position = env.fish_states["fish_0"].root_position.copy()
        for _ in range(100):
            env.step(zero_action_dict(env))
        state = env.fish_states["fish_0"]
        displacement = state.root_position - start_position
        body_velocity = env.get_debug_snapshot("fish_0")["motion_metrics"]["forward_velocity"]
        return {
            "pass": bool(float(np.linalg.norm(displacement)) < 1e-4 and abs(float(body_velocity)) < 1e-5),
            "displacement_norm": float(np.linalg.norm(displacement)),
            "forward_velocity": float(body_velocity),
        }
    finally:
        env.close()


def probe_scripted_wave_propulsion(seed: int) -> dict[str, Any]:
    env = build_single_fish_debug_env(seed)
    try:
        env.set_debug_state(
            agent_states={"fish_0": {"position": [0.0, 0.0], "theta": 0.0}},
            food_positions=np.zeros((0, 2), dtype=np.float32),
            food_team_indices=np.zeros(0, dtype=np.int64),
        )
        start_position = env.fish_states["fish_0"].root_position.copy()
        forward_velocity_samples: list[float] = []
        joint_limit_samples: list[float] = []
        for step_idx in range(140):
            motion = scripted_wave_action(step_idx, amplitude=0.95)
            _, _, _, _, infos = env.step(action_dict_for_motion(env, motion))
            info = infos["fish_0"]
            forward_velocity_samples.append(float(info.get("forward_velocity", 0.0)))
            joint_limit_samples.append(float(info.get("mean_joint_limit_ratio", 0.0)))
        end_position = env.fish_states["fish_0"].root_position.copy()
        displacement = end_position - start_position
        mean_forward_velocity = float(np.mean(np.asarray(forward_velocity_samples, dtype=np.float32)))
        mean_joint_limit = float(np.mean(np.asarray(joint_limit_samples, dtype=np.float32)))
        return {
            "pass": bool(float(displacement[0]) > 0.25 and mean_forward_velocity > 0.03 and mean_joint_limit < 0.98),
            "forward_displacement": float(displacement[0]),
            "lateral_displacement": float(displacement[1]),
            "mean_forward_velocity": mean_forward_velocity,
            "mean_joint_limit_ratio": mean_joint_limit,
        }
    finally:
        env.close()


def probe_mirror_torque_turn(seed: int) -> dict[str, Any]:
    left_env = build_single_fish_debug_env(seed)
    right_env = build_single_fish_debug_env(seed)
    try:
        initial_state = {"fish_0": {"position": [0.0, 0.0], "theta": 0.0}}
        empty_food = np.zeros((0, 2), dtype=np.float32)
        empty_food_teams = np.zeros(0, dtype=np.int64)
        left_env.set_debug_state(agent_states=initial_state, food_positions=empty_food, food_team_indices=empty_food_teams)
        right_env.set_debug_state(agent_states=initial_state, food_positions=empty_food, food_team_indices=empty_food_teams)
        for step_idx in range(120):
            left_motion = scripted_wave_action(step_idx, amplitude=0.80, bias=0.22)
            right_motion = scripted_wave_action(step_idx, amplitude=0.80, bias=-0.22)
            left_env.step(action_dict_for_motion(left_env, left_motion))
            right_env.step(action_dict_for_motion(right_env, right_motion))
        left_theta = float(left_env.fish_states["fish_0"].root_theta)
        right_theta = float(right_env.fish_states["fish_0"].root_theta)
        left_y = float(left_env.fish_states["fish_0"].root_position[1])
        right_y = float(right_env.fish_states["fish_0"].root_position[1])
        return {
            "pass": bool(np.sign(left_theta) != np.sign(right_theta) and abs(left_theta) > 0.1 and abs(right_theta) > 0.1),
            "left_theta": left_theta,
            "right_theta": right_theta,
            "left_y": left_y,
            "right_y": right_y,
        }
    finally:
        left_env.close()
        right_env.close()


def probe_history_contract(seed: int) -> dict[str, Any]:
    env = build_single_fish_debug_env(seed)
    try:
        obs, _ = env.reset(seed=seed)
        fish_obs = obs["fish_0"]
        initial_history = fish_obs[39:].reshape(env.history_length, 9)
        initial_history_equal = bool(np.allclose(initial_history, initial_history[0:1], atol=1e-6))
        action = action_dict_for_motion(env, np.array([0.5, -0.5], dtype=np.float32))
        obs_after, _, _, _, _ = env.step(action)
        history_after = obs_after["fish_0"][39:].reshape(env.history_length, 9)
        expected_latest = env.control_histories["fish_0"][-1]
        return {
            "pass": bool(
                fish_obs.shape == (111,)
                and env.action_space["motion"].shape == (2,)
                and int(env.action_space["message"].n) == 4
                and initial_history_equal
                and np.allclose(history_after[-1], expected_latest, atol=1e-6)
                and np.allclose(history_after[:-1], initial_history[1:], atol=1e-6)
            ),
            "observation_shape": list(fish_obs.shape),
            "history_length": int(env.history_length),
            "history_initial_repeated": initial_history_equal,
            "history_shift_ok": bool(np.allclose(history_after[:-1], initial_history[1:], atol=1e-6)),
            "history_latest_ok": bool(np.allclose(history_after[-1], expected_latest, atol=1e-6)),
        }
    finally:
        env.close()


PROBES = {
    "torque_rest_decay": probe_torque_rest_decay,
    "scripted_wave_propulsion": probe_scripted_wave_propulsion,
    "mirror_torque_turn": probe_mirror_torque_turn,
    "no_hidden_drive_assist": probe_no_hidden_drive_assist,
    "history_contract": probe_history_contract,
}


PROBE_SETS = {
    "baseline": [
        "torque_rest_decay",
        "scripted_wave_propulsion",
        "mirror_torque_turn",
        "no_hidden_drive_assist",
        "history_contract",
    ],
}


def main() -> None:
    args = parse_args()
    if args.probe_set not in PROBES and args.probe_set not in PROBE_SETS:
        raise ValueError(f"Unknown probe or probe set: {args.probe_set}")

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.probe_set)
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_names = PROBE_SETS.get(args.probe_set, [args.probe_set])
    results: dict[str, Any] = {}
    overall_pass = True

    for index, probe_name in enumerate(probe_names):
        result = PROBES[probe_name](args.seed + (index * 1000))
        results[probe_name] = result
        overall_pass = overall_pass and bool(result.get("pass", False))
        print(f"{probe_name}: {'PASS' if result.get('pass') else 'FAIL'}")

    summary = {
        "probe_set": args.probe_set,
        "seed": int(args.seed),
        "pass": bool(overall_pass),
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"summary_json={summary_path.resolve()}")


if __name__ == "__main__":
    main()
