"""Environment-only validation harness for V9 muscle-activation schooling."""

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
from triangles import RED_TEAM, CommunicatingSchoolEnv


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V9 muscle-activation environment probes.")
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
        activation_time_constant=kwargs.pop("activation_time_constant", 0.12),
        joint_passive_stiffness=kwargs.pop("joint_passive_stiffness", 10.0),
        body_linear_drag=kwargs.pop("body_linear_drag", 1.0),
        show_sensor_overlay=kwargs.pop("show_sensor_overlay", False),
        focus_agent_id=kwargs.pop("focus_agent_id", "fish_0"),
        mute_received_messages=kwargs.pop("mute_received_messages", False),
    )
    env.reset(seed=seed)
    return env


def zero_motion(env: CommunicatingSchoolEnv) -> np.ndarray:
    return np.zeros(env.num_joints, dtype=np.float32)


def with_motion_dim(env: CommunicatingSchoolEnv, values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == env.num_joints:
        return array.astype(np.float32, copy=True)
    if array.size == 2 and env.num_joints > 0:
        motion = np.zeros(env.num_joints, dtype=np.float32)
        motion[: min(2, env.num_joints)] = array[: min(2, env.num_joints)]
        return motion
    raise ValueError(f"Expected {env.num_joints} motion values, got {array.size}.")


def build_single_fish_debug_env(seed: int) -> CommunicatingSchoolEnv:
    return build_env(
        seed=seed,
        reward_mode="locomotion_debug",
        time_limit=240,
        num_red_fish=1,
        num_blue_fish=0,
        num_red_pellets=0,
        num_blue_pellets=0,
    )


def build_single_fish_food_env(seed: int) -> CommunicatingSchoolEnv:
    return build_env(
        seed=seed,
        reward_mode="locomotion_debug",
        time_limit=20,
        num_red_fish=1,
        num_blue_fish=0,
        num_red_pellets=1,
        num_blue_pellets=0,
    )


def zero_action_dict(env: CommunicatingSchoolEnv) -> dict[str, dict[str, Any]]:
    return {agent_id: {"motion": zero_motion(env), "message": 0} for agent_id in env.get_agent_ids()}


def action_dict_for_motion(env: CommunicatingSchoolEnv, motion: np.ndarray, *, message: int = 0) -> dict[str, dict[str, Any]]:
    motion = with_motion_dim(env, motion)
    return {agent_id: {"motion": motion.copy(), "message": int(message)} for agent_id in env.get_agent_ids()}


def scripted_wave_activation(
    env: CommunicatingSchoolEnv,
    step_idx: int,
    *,
    amplitude: float = 0.95,
    phase_rate: float = 0.34,
    phase_delta: float = math.pi / 2.0,
    bias: float = 0.0,
) -> np.ndarray:
    phase = phase_rate * float(step_idx)
    phase_offsets = np.arange(env.num_joints, dtype=np.float32) * float(phase_delta)
    activations = (amplitude * np.sin(phase - phase_offsets)).astype(np.float32)
    if activations.size:
        activations[0] = float(np.clip(activations[0] + bias, -1.0, 1.0))
    if activations.size > 1:
        activations[1] = float(np.clip(activations[1] - bias, -1.0, 1.0))
    return np.clip(activations, -1.0, 1.0)


def run_single_fish_rollout(
    env: CommunicatingSchoolEnv,
    *,
    steps: int,
    motion_fn,
) -> dict[str, Any]:
    start_position = env.fish_states["fish_0"].root_position.astype(np.float32).copy()
    forward_velocity_samples: list[float] = []
    joint_limit_samples: list[float] = []
    activation_samples: list[float] = []
    zero_crossings_total = 0
    activation_sign_changes_total = 0
    for step_idx in range(steps):
        motion = with_motion_dim(env, motion_fn(step_idx))
        _, _, _, _, infos = env.step(action_dict_for_motion(env, motion))
        info = infos["fish_0"]
        forward_velocity_samples.append(float(info.get("forward_velocity", 0.0)))
        joint_limit_samples.append(float(info.get("mean_joint_limit_ratio", 0.0)))
        activation_samples.append(float(info.get("mean_abs_activation", 0.0)))
        zero_crossings_total += int(info.get("joint_velocity_zero_crossings", 0))
        activation_sign_changes_total += int(info.get("activation_sign_changes_this_step", 0))
    end_position = env.fish_states["fish_0"].root_position.astype(np.float32).copy()
    return {
        "forward_displacement": float(end_position[0] - start_position[0]),
        "lateral_displacement": float(end_position[1] - start_position[1]),
        "mean_forward_velocity": float(np.mean(np.asarray(forward_velocity_samples, dtype=np.float32))),
        "mean_joint_limit_ratio": float(np.mean(np.asarray(joint_limit_samples, dtype=np.float32))),
        "mean_abs_activation": float(np.mean(np.asarray(activation_samples, dtype=np.float32))),
        "late_mean_forward_velocity": float(np.mean(np.asarray(forward_velocity_samples[-40:], dtype=np.float32))),
        "joint_zero_crossings_total": int(zero_crossings_total),
        "activation_sign_changes_total": int(activation_sign_changes_total),
    }


def probe_activation_rest_decay(seed: int) -> dict[str, Any]:
    env = build_single_fish_debug_env(seed)
    try:
        seeded_joint_angles = with_motion_dim(env, [0.42, -0.33])
        seeded_joint_velocities = with_motion_dim(env, [1.40, -1.10])
        seeded_joint_activation = with_motion_dim(env, [0.75, -0.55])
        env.set_debug_state(
            agent_states={
                "fish_0": {
                    "position": [0.0, 0.0],
                    "theta": 0.0,
                    "velocity": [0.45, -0.20],
                    "omega": 0.65,
                    "joint_angles": seeded_joint_angles,
                    "joint_velocities": seeded_joint_velocities,
                    "joint_activation": seeded_joint_activation,
                    "applied_joint_torque": zero_motion(env),
                }
            },
            food_positions=np.zeros((0, 2), dtype=np.float32),
            food_team_indices=np.zeros(0, dtype=np.int64),
        )
        for _ in range(120):
            env.step(zero_action_dict(env))
        snapshot = env.get_debug_snapshot("fish_0")
        final_joint_speed = float(np.max(np.abs(snapshot["joint_velocities"])))
        final_root_speed = float(np.linalg.norm(snapshot["root_velocity"]))
        final_activation = float(np.max(np.abs(snapshot["joint_activation"])))
        final_omega = abs(float(snapshot["root_omega"]))
        return {
            "pass": bool(
                final_joint_speed < 0.08
                and final_root_speed < 0.08
                and final_activation < 0.05
                and final_omega < 0.08
            ),
            "final_joint_speed": final_joint_speed,
            "final_root_speed": final_root_speed,
            "final_abs_activation": final_activation,
            "final_abs_angular_velocity": final_omega,
        }
    finally:
        env.close()


def probe_constant_activation_no_cruise(seed: int) -> dict[str, Any]:
    env = build_single_fish_debug_env(seed)
    try:
        env.set_debug_state(
            agent_states={"fish_0": {"position": [0.0, 0.0], "theta": 0.0}},
            food_positions=np.zeros((0, 2), dtype=np.float32),
            food_team_indices=np.zeros(0, dtype=np.int64),
        )
        result = run_single_fish_rollout(
            env,
            steps=220,
            motion_fn=lambda step_idx: [0.90, -0.10],
        )
        return {
            "pass": bool(
                abs(result["late_mean_forward_velocity"]) < 0.11
                and result["forward_displacement"] < 1.25
                and result["mean_joint_limit_ratio"] < 0.98
            ),
            **result,
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
        result = run_single_fish_rollout(
            env,
            steps=180,
            motion_fn=lambda step_idx: scripted_wave_activation(env, step_idx, amplitude=0.95),
        )
        return {
            "pass": bool(
                result["forward_displacement"] > 0.60
                and result["mean_forward_velocity"] > 0.06
                and result["activation_sign_changes_total"] >= 30
            ),
            **result,
        }
    finally:
        env.close()


def probe_wave_beats_static_activation(seed: int) -> dict[str, Any]:
    wave_env = build_single_fish_debug_env(seed)
    static_env = build_single_fish_debug_env(seed)
    try:
        initial_state = {"fish_0": {"position": [0.0, 0.0], "theta": 0.0}}
        empty_food = np.zeros((0, 2), dtype=np.float32)
        empty_teams = np.zeros(0, dtype=np.int64)
        wave_env.set_debug_state(agent_states=initial_state, food_positions=empty_food, food_team_indices=empty_teams)
        static_env.set_debug_state(agent_states=initial_state, food_positions=empty_food, food_team_indices=empty_teams)
        wave_result = run_single_fish_rollout(
            wave_env,
            steps=180,
            motion_fn=lambda step_idx: scripted_wave_activation(wave_env, step_idx, amplitude=0.95),
        )
        static_result = run_single_fish_rollout(
            static_env,
            steps=180,
            motion_fn=lambda step_idx: [0.90, -0.10],
        )
        return {
            "pass": bool(
                wave_result["forward_displacement"] > (static_result["forward_displacement"] + 0.35)
                and wave_result["mean_forward_velocity"] > (static_result["mean_forward_velocity"] + 0.04)
            ),
            "wave_forward_displacement": wave_result["forward_displacement"],
            "static_forward_displacement": static_result["forward_displacement"],
            "wave_mean_forward_velocity": wave_result["mean_forward_velocity"],
            "static_mean_forward_velocity": static_result["mean_forward_velocity"],
            "wave_activation_sign_changes_total": wave_result["activation_sign_changes_total"],
            "static_activation_sign_changes_total": static_result["activation_sign_changes_total"],
        }
    finally:
        wave_env.close()
        static_env.close()


def probe_mouth_capture_contract(seed: int) -> dict[str, Any]:
    env = build_single_fish_food_env(seed)
    try:
        env.set_debug_state(
            agent_states={"fish_0": {"position": [0.0, 0.0], "theta": 0.0}},
            food_positions=np.asarray([[0.56, 0.0]], dtype=np.float32),
            food_team_indices=np.asarray([RED_TEAM], dtype=np.int64),
        )
        snapshot = env.get_debug_snapshot("fish_0")
        root_position = np.asarray(snapshot["root_position"], dtype=np.float32)
        mouth_position = np.asarray(snapshot["mouth_position"], dtype=np.float32)
        food_position = np.asarray(snapshot["food_positions"][0], dtype=np.float32)
        root_distance = float(np.linalg.norm(food_position - root_position))
        mouth_distance = float(np.linalg.norm(food_position - mouth_position))
        _, _, _, _, infos = env.step(zero_action_dict(env))
        ate_food = int(infos["fish_0"].get("food_eaten_this_step", 0))
        return {
            "pass": bool(
                ate_food == 1
                and mouth_distance < env.food_capture_radius
                and root_distance > env.food_capture_radius
            ),
            "food_eaten_this_step": ate_food,
            "mouth_distance": mouth_distance,
            "root_distance": root_distance,
            "capture_radius": float(env.food_capture_radius),
        }
    finally:
        env.close()


def probe_history_contract(seed: int) -> dict[str, Any]:
    env = build_single_fish_debug_env(seed)
    try:
        obs, _ = env.reset(seed=seed)
        fish_obs = obs["fish_0"]
        history_feature_count = env._control_history_feature_count()
        initial_history = fish_obs[39:].reshape(env.history_length, history_feature_count)
        initial_history_equal = bool(np.allclose(initial_history, initial_history[0:1], atol=1e-6))
        action = action_dict_for_motion(env, [0.55, -0.45])
        obs_after, _, _, _, _ = env.step(action)
        history_after = obs_after["fish_0"][39:].reshape(env.history_length, history_feature_count)
        expected_latest = env.control_histories["fish_0"][-1]
        latest_activation = env.fish_states["fish_0"].joint_activation.astype(np.float32).copy()
        return {
            "pass": bool(
                fish_obs.shape == env.observation_space.shape
                and env.action_space["motion"].shape == (env.num_joints,)
                and int(env.action_space["message"].n) == 4
                and initial_history_equal
                and np.allclose(history_after[:-1], initial_history[1:], atol=1e-6)
                and np.allclose(history_after[-1], expected_latest, atol=1e-6)
                and np.allclose(history_after[-1][-env.num_joints :], latest_activation, atol=1e-6)
            ),
            "observation_shape": list(fish_obs.shape),
            "history_length": int(env.history_length),
            "history_feature_count": int(history_feature_count),
            "history_initial_repeated": initial_history_equal,
            "history_shift_ok": bool(np.allclose(history_after[:-1], initial_history[1:], atol=1e-6)),
            "history_latest_ok": bool(np.allclose(history_after[-1], expected_latest, atol=1e-6)),
            "history_tracks_activation": bool(np.allclose(history_after[-1][-env.num_joints :], latest_activation, atol=1e-6)),
        }
    finally:
        env.close()


PROBES = {
    "activation_rest_decay": probe_activation_rest_decay,
    "constant_activation_no_cruise": probe_constant_activation_no_cruise,
    "scripted_wave_propulsion": probe_scripted_wave_propulsion,
    "wave_beats_static_activation": probe_wave_beats_static_activation,
    "mouth_capture_contract": probe_mouth_capture_contract,
    "history_contract": probe_history_contract,
}


PROBE_SETS = {
    "baseline": [
        "activation_rest_decay",
        "constant_activation_no_cruise",
        "scripted_wave_propulsion",
        "wave_beats_static_activation",
        "mouth_capture_contract",
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
