"""Env-only validation harness for V5 simulator choices."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from triangles import FishConfig, OctopusEnv


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FOOD_POSITION = np.array([10.0, 0.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic V5 environment validation probes.")
    parser.add_argument(
        "--probe-set",
        type=str,
        default="baseline",
        choices=("baseline", "physics_audit", "control_contract"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps-short", type=int, default=10)
    parser.add_argument("--steps-long", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SCRIPT_DIR / "media" / "debug_reports" / timestamp


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_serializable(value) for key, value in row.items()})


def signed_angle_delta(current: float, initial: float) -> float:
    delta = float(current - initial)
    return float(math.atan2(math.sin(delta), math.cos(delta)))


def monotonic_non_increasing(values: list[float], tol: float = 1e-6) -> tuple[bool, list[int]]:
    violations: list[int] = []
    for idx in range(1, len(values)):
        if values[idx] > values[idx - 1] + tol:
            violations.append(idx)
    return len(violations) == 0, violations


def build_env(seed: int, *, dt: float = 0.05, time_limit: int = 100) -> OctopusEnv:
    fish_config = replace(FishConfig(), dt=float(dt))
    env = OctopusEnv(
        epsilon=0.0,
        render_mode=None,
        fish_config=fish_config,
        enable_curriculum=False,
        fixed_food_distance=10.0,
        time_limit=time_limit,
    )
    env.reset(seed=seed)
    return env


def prepare_env(seed: int) -> OctopusEnv:
    return build_env(seed, dt=0.05, time_limit=100)


def heading_vector(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)


def canonical_debug_state() -> dict[str, Any]:
    return {
        "position": np.zeros(2, dtype=np.float32),
        "velocity": np.zeros(2, dtype=np.float32),
        "theta": 0.0,
        "omega": 0.0,
        "food_position": DEFAULT_FOOD_POSITION.copy(),
        "timestep": 0,
        "prev_turn_action": 1,
        "prev_push_action": 1,
    }


def set_state_from_dict(env: OctopusEnv, state: dict[str, Any]) -> None:
    env.set_debug_state(
        position=state["position"],
        velocity=state["velocity"],
        theta=state["theta"],
        omega=state["omega"],
        food_position=state["food_position"],
        timestep=state.get("timestep", 0),
        prev_turn_action=state.get("prev_turn_action", 1),
        prev_push_action=state.get("prev_push_action", 1),
    )


def run_repeated_action_probe(
    env: OctopusEnv,
    *,
    action: tuple[int, int],
    steps: int,
    initial_state: dict[str, Any],
) -> dict[str, Any]:
    set_state_from_dict(env, initial_state)
    reward_sum = 0.0
    terminated = False
    truncated = False
    for _ in range(steps):
        _, reward, terminated, truncated, _ = env.step(np.array(action, dtype=np.int64))
        reward_sum += float(reward)
        if terminated or truncated:
            break

    snapshot = env.get_debug_snapshot()
    final_position = np.asarray(snapshot["position"], dtype=np.float32)
    final_velocity = np.asarray(snapshot["velocity"], dtype=np.float32)
    heading_change = signed_angle_delta(float(snapshot["theta"]), float(initial_state["theta"]))
    return {
        "turn_idx": int(action[0]),
        "push_idx": int(action[1]),
        "steps_requested": int(steps),
        "steps_executed": int(snapshot["timestep"]),
        "final_x": float(final_position[0]),
        "final_y": float(final_position[1]),
        "forward_progress": float(final_position[0] - initial_state["position"][0]),
        "lateral_drift": float(final_position[1] - initial_state["position"][1]),
        "final_speed": float(np.linalg.norm(final_velocity)),
        "heading_change_rad": heading_change,
        "heading_change_deg": math.degrees(heading_change),
        "angular_speed": float(snapshot["omega"]),
        "reward_sum": float(reward_sum),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def plot_action_heatmaps(
    rows: list[dict[str, Any]],
    *,
    horizons: list[int],
    output_path: Path,
) -> None:
    metric_specs = [
        ("forward_progress", "Forward progress"),
        ("lateral_drift", "Lateral drift"),
        ("heading_change_deg", "Heading change (deg)"),
        ("reward_sum", "Reward sum"),
    ]
    fig, axes = plt.subplots(len(horizons), len(metric_specs), figsize=(16, 6), squeeze=False)
    for row_idx, horizon in enumerate(horizons):
        subset = [row for row in rows if row["steps_requested"] == horizon]
        for col_idx, (metric_key, title) in enumerate(metric_specs):
            matrix = np.zeros((3, 3), dtype=np.float32)
            for row in subset:
                matrix[int(row["turn_idx"]), int(row["push_idx"])] = float(row[metric_key])
            ax = axes[row_idx][col_idx]
            image = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
            ax.set_xticks([0, 1, 2])
            ax.set_yticks([0, 1, 2])
            ax.set_xlabel("Push idx")
            ax.set_ylabel("Turn idx")
            ax.set_title(f"{title} | {horizon} steps")
            for turn_idx in range(3):
                for push_idx in range(3):
                    ax.text(
                        push_idx,
                        turn_idx,
                        f"{matrix[turn_idx, push_idx]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def action_response_probe(
    env: OctopusEnv,
    *,
    steps_short: int,
    steps_long: int,
    output_dir: Path,
    no_plots: bool,
) -> dict[str, Any]:
    initial_state = canonical_debug_state()
    rows: list[dict[str, Any]] = []
    horizons = [steps_short, steps_long]
    for horizon in horizons:
        for turn_idx in range(3):
            for push_idx in range(3):
                rows.append(
                    run_repeated_action_probe(
                        env,
                        action=(turn_idx, push_idx),
                        steps=horizon,
                        initial_state=initial_state,
                    )
                )

    write_csv(output_dir / "action_metrics.csv", rows)
    if not no_plots:
        plot_action_heatmaps(rows, horizons=horizons, output_path=output_dir / "action_heatmaps.png")

    rows_by_key = {
        (row["steps_requested"], row["turn_idx"], row["push_idx"]): row
        for row in rows
    }
    neutral_motion_tol = 1e-4
    mirror_failures: list[dict[str, Any]] = []
    neutral_stationary = True
    thrust_forward = True

    for horizon in horizons:
        neutral_row = rows_by_key[(horizon, 1, 1)]
        neutral_motion = math.hypot(float(neutral_row["final_x"]), float(neutral_row["final_y"]))
        if neutral_motion > neutral_motion_tol or float(neutral_row["final_speed"]) > neutral_motion_tol:
            neutral_stationary = False

        thrust_row = rows_by_key[(horizon, 1, 2)]
        if float(thrust_row["forward_progress"]) <= 0.0:
            thrust_forward = False

        for push_idx in range(3):
            left = rows_by_key[(horizon, 0, push_idx)]
            right = rows_by_key[(horizon, 2, push_idx)]
            reference = max(
                abs(float(left["forward_progress"])),
                abs(float(right["forward_progress"])),
                abs(float(left["lateral_drift"])),
                abs(float(right["lateral_drift"])),
                abs(float(left["heading_change_rad"])),
                abs(float(right["heading_change_rad"])),
                abs(float(left["final_speed"])),
                abs(float(right["final_speed"])),
                1e-6,
            )
            abs_tol = 0.01
            mirror_ok = (
                np.isclose(float(left["forward_progress"]), float(right["forward_progress"]), rtol=0.05, atol=abs_tol)
                and np.isclose(float(left["final_speed"]), float(right["final_speed"]), rtol=0.05, atol=abs_tol)
                and abs(float(left["lateral_drift"]) + float(right["lateral_drift"])) <= max(reference * 0.05, abs_tol)
                and abs(float(left["heading_change_rad"]) + float(right["heading_change_rad"]))
                <= max(reference * 0.05, abs_tol)
            )
            if not mirror_ok:
                mirror_failures.append(
                    {
                        "steps_requested": horizon,
                        "push_idx": push_idx,
                        "left_forward_progress": float(left["forward_progress"]),
                        "right_forward_progress": float(right["forward_progress"]),
                        "left_lateral_drift": float(left["lateral_drift"]),
                        "right_lateral_drift": float(right["lateral_drift"]),
                        "left_heading_change_rad": float(left["heading_change_rad"]),
                        "right_heading_change_rad": float(right["heading_change_rad"]),
                    }
                )

    return {
        "pass": bool(neutral_stationary and thrust_forward and not mirror_failures),
        "neutral_stationary": bool(neutral_stationary),
        "thrust_forward": bool(thrust_forward),
        "mirror_pass": len(mirror_failures) == 0,
        "mirror_failures": mirror_failures,
        "rows": rows,
    }


def plot_decay_trace(rows: list[dict[str, Any]], output_path: Path) -> None:
    steps = [int(row["step"]) for row in rows]
    speeds = [float(row["speed"]) for row in rows]
    omegas = [float(row["abs_omega"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(steps, speeds, marker="o")
    axes[0].set_title("Passive speed decay")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Speed")

    axes[1].plot(steps, omegas, marker="o", color="tab:orange")
    axes[1].set_title("Passive angular decay")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("|omega|")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def passive_decay_probe(env: OctopusEnv, *, output_dir: Path, no_plots: bool) -> dict[str, Any]:
    set_state_from_dict(
        env,
        {
            "position": np.zeros(2, dtype=np.float32),
            "velocity": np.array([3.0, 0.0], dtype=np.float32),
            "theta": 0.0,
            "omega": 2.0,
            "food_position": DEFAULT_FOOD_POSITION.copy(),
            "timestep": 0,
            "prev_turn_action": 1,
            "prev_push_action": 1,
        },
    )

    rows: list[dict[str, Any]] = []
    for step in range(1, 21):
        _, reward, terminated, truncated, _ = env.step(np.array([1, 1], dtype=np.int64))
        snapshot = env.get_debug_snapshot()
        rows.append(
            {
                "step": step,
                "speed": float(np.linalg.norm(snapshot["velocity"])),
                "abs_omega": abs(float(snapshot["omega"])),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

    write_csv(output_dir / "decay_metrics.csv", rows)
    if not no_plots:
        plot_decay_trace(rows, output_dir / "drag_decay.png")

    speeds = [float(row["speed"]) for row in rows]
    omegas = [float(row["abs_omega"]) for row in rows]
    speed_monotonic, speed_violations = monotonic_non_increasing(speeds)
    omega_monotonic, omega_violations = monotonic_non_increasing(omegas)
    return {
        "pass": bool(speed_monotonic and omega_monotonic),
        "speed_monotonic": bool(speed_monotonic),
        "speed_violations": speed_violations,
        "omega_monotonic": bool(omega_monotonic),
        "omega_violations": omega_violations,
        "rows": rows,
    }


def rollout_observation_scenario(
    env: OctopusEnv,
    *,
    label: str,
    food_position: np.ndarray,
    actions: list[np.ndarray],
) -> dict[str, Any]:
    set_state_from_dict(
        env,
        {
            "position": np.zeros(2, dtype=np.float32),
            "velocity": np.zeros(2, dtype=np.float32),
            "theta": 0.0,
            "omega": 0.0,
            "food_position": food_position,
            "timestep": 0,
            "prev_turn_action": 1,
            "prev_push_action": 1,
        },
    )
    prev_snapshot = env.get_debug_snapshot()
    records: list[dict[str, Any]] = []
    for step_idx, action in enumerate(actions, start=1):
        _, reward, terminated, truncated, _ = env.step(action)
        snapshot = env.get_debug_snapshot()
        movement = np.asarray(snapshot["position"]) - np.asarray(prev_snapshot["position"])
        prev_target = np.asarray(prev_snapshot["target_relative"])
        movement_toward = float(np.dot(movement, prev_target))
        records.append(
            {
                "scenario": label,
                "step": step_idx,
                "action_turn": int(action[0]),
                "action_push": int(action[1]),
                "x": float(snapshot["position"][0]),
                "y": float(snapshot["position"][1]),
                "theta": float(snapshot["theta"]),
                "distance_to_food": float(snapshot["distance_to_food"]),
                "movement_toward": movement_toward,
                "reward": float(reward),
                "obs_rel_dist": float(snapshot["observation"][2]),
                "obs_cos_theta": float(snapshot["observation"][3]),
                "obs_sin_theta": float(snapshot["observation"][4]),
                "obs_prev_turn": float(snapshot["observation"][8]),
                "obs_prev_push": float(snapshot["observation"][9]),
                "obs_progress": float(snapshot["observation"][10]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        prev_snapshot = snapshot
        if terminated or truncated:
            break
    return {"label": label, "food_position": food_position.copy(), "records": records}

def plot_observation_traces(scenarios: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(12, 4 * len(scenarios)), squeeze=False)
    rel_scale = 10.0
    for row_idx, scenario in enumerate(scenarios):
        records = scenario["records"]
        xs = [0.0] + [float(record["x"]) for record in records]
        ys = [0.0] + [float(record["y"]) for record in records]
        food = np.asarray(scenario["food_position"], dtype=np.float32)

        ax_traj = axes[row_idx][0]
        ax_traj.plot(xs, ys, marker="o")
        ax_traj.scatter([food[0]], [food[1]], color="red", marker="x", s=80)
        ax_traj.set_title(f"{scenario['label']} trajectory")
        ax_traj.set_xlabel("x")
        ax_traj.set_ylabel("y")
        ax_traj.set_aspect("equal", adjustable="box")

        ax_obs = axes[row_idx][1]
        steps = [int(record["step"]) for record in records]
        raw_dist = [float(record["distance_to_food"]) for record in records]
        decoded_dist = [float(record["obs_rel_dist"]) * rel_scale for record in records]
        progress = [float(record["obs_progress"]) for record in records]
        ax_obs.plot(steps, raw_dist, marker="o", label="raw dist")
        ax_obs.plot(steps, decoded_dist, marker="x", label="obs rel_dist * 10")
        ax_obs.set_title(f"{scenario['label']} observation traces")
        ax_obs.set_xlabel("step")
        ax_obs.set_ylabel("distance")
        ax_progress = ax_obs.twinx()
        ax_progress.plot(steps, progress, linestyle="--", color="tab:green", label="progress")
        ax_progress.set_ylabel("progress")
        handles, labels = ax_obs.get_legend_handles_labels()
        handles2, labels2 = ax_progress.get_legend_handles_labels()
        ax_obs.legend(handles + handles2, labels + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def observation_sanity_probe(
    env: OctopusEnv,
    *,
    steps_long: int,
    output_dir: Path,
    no_plots: bool,
) -> dict[str, Any]:
    left_turn_steps = min(max(steps_long // 5, 4), steps_long)
    behind_turn_steps = min(max((steps_long * 2) // 5, 8), steps_long)
    scenario_specs = [
        {
            "label": "target_ahead",
            "food_position": np.array([10.0, 0.0], dtype=np.float32),
            "actions": [np.array([1, 2], dtype=np.int64) for _ in range(steps_long)],
        },
        {
            "label": "target_left",
            "food_position": np.array([0.0, 10.0], dtype=np.float32),
            "actions": [np.array([2, 1], dtype=np.int64) for _ in range(left_turn_steps)]
            + [np.array([1, 2], dtype=np.int64) for _ in range(max(steps_long - left_turn_steps, 0))],
        },
        {
            "label": "target_behind",
            "food_position": np.array([-10.0, 0.0], dtype=np.float32),
            "actions": [np.array([2, 1], dtype=np.int64) for _ in range(behind_turn_steps)]
            + [np.array([1, 2], dtype=np.int64) for _ in range(max(steps_long - behind_turn_steps, 0))],
        },
    ]
    scenarios = [
        rollout_observation_scenario(
            env,
            label=spec["label"],
            food_position=spec["food_position"],
            actions=spec["actions"],
        )
        for spec in scenario_specs
    ]

    low = env.obs_low
    high = env.obs_high
    scenario_results: dict[str, Any] = {}
    overall_finite = True
    overall_bounds = True
    overall_cos_sin = True
    overall_prev_actions = True
    overall_progress = True
    overall_toward_motion = True

    for scenario in scenarios:
        label = scenario["label"]
        records = scenario["records"]
        finite = True
        in_bounds = True
        cos_sin_consistent = True
        prev_actions_consistent = True
        progress_monotonic = True
        toward_motion_checks = 0
        toward_motion_violations = 0
        last_progress = -float("inf")

        for idx, record in enumerate(records):
            snapshot_bounds = np.array(
                [
                    record["obs_rel_dist"],
                    record["obs_cos_theta"],
                    record["obs_sin_theta"],
                    record["obs_prev_turn"],
                    record["obs_prev_push"],
                    record["obs_progress"],
                ],
                dtype=np.float32,
            )
            if not np.all(np.isfinite(snapshot_bounds)):
                finite = False
            if (
                record["obs_rel_dist"] < low[2] - 1e-6
                or record["obs_rel_dist"] > high[2] + 1e-6
                or record["obs_cos_theta"] < low[3] - 1e-6
                or record["obs_cos_theta"] > high[3] + 1e-6
                or record["obs_sin_theta"] < low[4] - 1e-6
                or record["obs_sin_theta"] > high[4] + 1e-6
                or record["obs_prev_turn"] < low[8] - 1e-6
                or record["obs_prev_turn"] > high[8] + 1e-6
                or record["obs_prev_push"] < low[9] - 1e-6
                or record["obs_prev_push"] > high[9] + 1e-6
                or record["obs_progress"] < low[10] - 1e-6
                or record["obs_progress"] > high[10] + 1e-6
            ):
                in_bounds = False
            if not np.isclose(record["obs_cos_theta"], math.cos(record["theta"]), atol=1e-6):
                cos_sin_consistent = False
            if not np.isclose(record["obs_sin_theta"], math.sin(record["theta"]), atol=1e-6):
                cos_sin_consistent = False
            if not np.isclose(record["obs_prev_turn"], record["action_turn"] - 1, atol=1e-6):
                prev_actions_consistent = False
            if not np.isclose(record["obs_prev_push"], record["action_push"] - 1, atol=1e-6):
                prev_actions_consistent = False
            if record["obs_progress"] + 1e-6 < last_progress:
                progress_monotonic = False
            last_progress = record["obs_progress"]
            if record["movement_toward"] > 1e-5:
                toward_motion_checks += 1
                if idx == 0:
                    prev_dist = float(np.linalg.norm(np.asarray(scenario["food_position"], dtype=np.float32)))
                else:
                    prev_dist = float(records[idx - 1]["distance_to_food"])
                if record["distance_to_food"] > prev_dist + 1e-5:
                    toward_motion_violations += 1

        scenario_pass = (
            finite
            and in_bounds
            and cos_sin_consistent
            and prev_actions_consistent
            and progress_monotonic
            and toward_motion_checks > 0
            and toward_motion_violations == 0
        )
        scenario_results[label] = {
            "pass": bool(scenario_pass),
            "finite": bool(finite),
            "within_bounds": bool(in_bounds),
            "cos_sin_consistent": bool(cos_sin_consistent),
            "prev_actions_consistent": bool(prev_actions_consistent),
            "progress_monotonic": bool(progress_monotonic),
            "toward_motion_checks": int(toward_motion_checks),
            "toward_motion_violations": int(toward_motion_violations),
            "steps_recorded": int(len(records)),
        }
        overall_finite &= finite
        overall_bounds &= in_bounds
        overall_cos_sin &= cos_sin_consistent
        overall_prev_actions &= prev_actions_consistent
        overall_progress &= progress_monotonic
        overall_toward_motion &= toward_motion_checks > 0 and toward_motion_violations == 0

    if not no_plots:
        plot_observation_traces(scenarios, output_dir / "observation_traces.png")

    result = {
        "pass": bool(
            overall_finite
            and overall_bounds
            and overall_cos_sin
            and overall_prev_actions
            and overall_progress
            and overall_toward_motion
        ),
        "all_finite": bool(overall_finite),
        "within_bounds": bool(overall_bounds),
        "cos_sin_consistent": bool(overall_cos_sin),
        "prev_actions_consistent": bool(overall_prev_actions),
        "progress_monotonic": bool(overall_progress),
        "toward_motion_consistent": bool(overall_toward_motion),
        "scenario_results": scenario_results,
    }
    write_json(output_dir / "observation_checks.json", result)
    return result

def run_reward_trajectory(
    env: OctopusEnv,
    *,
    label: str,
    initial_state: dict[str, Any],
    actions: list[np.ndarray],
) -> list[dict[str, Any]]:
    set_state_from_dict(env, initial_state)
    rows: list[dict[str, Any]] = []
    cumulative_reward = 0.0
    for step_idx, action in enumerate(actions, start=1):
        _, reward, terminated, truncated, _ = env.step(action)
        breakdown = env.get_debug_snapshot()["reward_breakdown"]
        cumulative_reward += float(reward)
        rows.append(
            {
                "label": label,
                "step": step_idx,
                "action_turn": int(action[0]),
                "action_push": int(action[1]),
                "base_step_penalty": float(breakdown["base_step_penalty"]),
                "progress_delta": float(breakdown["progress_delta"]),
                "progress_reward": float(breakdown["progress_reward"]),
                "success_bonus": float(breakdown["success_bonus"]),
                "total_reward": float(breakdown["total_reward"]),
                "cumulative_reward": float(cumulative_reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        if terminated or truncated:
            break
    return rows


def plot_reward_trace(
    toward_rows: list[dict[str, Any]],
    away_rows: list[dict[str, Any]],
    success_reward: float,
    timeout_reward: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for rows, label, color in (
        (toward_rows, "toward", "tab:blue"),
        (away_rows, "away", "tab:orange"),
    ):
        steps = [int(row["step"]) for row in rows]
        total = [float(row["total_reward"]) for row in rows]
        progress = [float(row["progress_reward"]) for row in rows]
        axes[0].plot(steps, total, marker="o", label=f"{label} total", color=color)
        axes[0].plot(steps, progress, linestyle="--", label=f"{label} progress", color=color, alpha=0.7)
    axes[0].set_title("Reward trace")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("reward")
    axes[0].legend(loc="best")

    axes[1].bar(
        ["toward total", "away total", "success step", "timeout step"],
        [
            sum(float(row["total_reward"]) for row in toward_rows),
            sum(float(row["total_reward"]) for row in away_rows),
            success_reward,
            timeout_reward,
        ],
        color=["tab:blue", "tab:orange", "tab:green", "tab:red"],
    )
    axes[1].set_title("Reward comparison")
    axes[1].set_ylabel("reward")
    axes[1].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def reward_decomposition_probe(
    env: OctopusEnv,
    *,
    steps_long: int,
    output_dir: Path,
    no_plots: bool,
) -> dict[str, Any]:
    toward_rows = run_reward_trajectory(
        env,
        label="toward",
        initial_state={
            "position": np.zeros(2, dtype=np.float32),
            "velocity": np.zeros(2, dtype=np.float32),
            "theta": 0.0,
            "omega": 0.0,
            "food_position": DEFAULT_FOOD_POSITION.copy(),
            "timestep": 0,
            "prev_turn_action": 1,
            "prev_push_action": 1,
        },
        actions=[np.array([1, 2], dtype=np.int64) for _ in range(steps_long)],
    )
    away_rows = run_reward_trajectory(
        env,
        label="away",
        initial_state={
            "position": np.zeros(2, dtype=np.float32),
            "velocity": np.zeros(2, dtype=np.float32),
            "theta": math.pi,
            "omega": 0.0,
            "food_position": DEFAULT_FOOD_POSITION.copy(),
            "timestep": 0,
            "prev_turn_action": 1,
            "prev_push_action": 1,
        },
        actions=[np.array([1, 2], dtype=np.int64) for _ in range(steps_long)],
    )
    success_rows = run_reward_trajectory(
        env,
        label="success",
        initial_state={
            "position": np.array([0.45, 0.0], dtype=np.float32),
            "velocity": np.zeros(2, dtype=np.float32),
            "theta": 0.0,
            "omega": 0.0,
            "food_position": np.zeros(2, dtype=np.float32),
            "timestep": 0,
            "prev_turn_action": 1,
            "prev_push_action": 1,
        },
        actions=[np.array([1, 1], dtype=np.int64)],
    )
    timeout_rows = run_reward_trajectory(
        env,
        label="timeout_near_target",
        initial_state={
            "position": np.array([0.55, 0.0], dtype=np.float32),
            "velocity": np.zeros(2, dtype=np.float32),
            "theta": 0.0,
            "omega": 0.0,
            "food_position": np.zeros(2, dtype=np.float32),
            "timestep": env.time_limit - 1,
            "prev_turn_action": 1,
            "prev_push_action": 1,
        },
        actions=[np.array([1, 1], dtype=np.int64)],
    )

    toward_total = float(sum(float(row["total_reward"]) for row in toward_rows))
    away_total = float(sum(float(row["total_reward"]) for row in away_rows))
    success_reward = float(success_rows[0]["total_reward"])
    timeout_reward = float(timeout_rows[0]["total_reward"])
    result = {
        "pass": bool(toward_total > away_total and success_reward > timeout_reward),
        "toward_beats_away": bool(toward_total > away_total),
        "success_beats_timeout": bool(success_reward > timeout_reward),
        "toward_total_reward": toward_total,
        "away_total_reward": away_total,
        "success_reward": success_reward,
        "timeout_reward": timeout_reward,
        "toward_rows": toward_rows,
        "away_rows": away_rows,
        "success_rows": success_rows,
        "timeout_rows": timeout_rows,
    }
    write_json(output_dir / "reward_checks.json", result)
    if not no_plots:
        plot_reward_trace(toward_rows, away_rows, success_reward, timeout_reward, output_dir / "reward_trace.png")
    return result


def rotate_vec90(vector: np.ndarray) -> np.ndarray:
    return np.array([-vector[1], vector[0]], dtype=np.float32)


def wrapped_target_error(theta: float, target_relative: np.ndarray) -> float:
    target_angle = math.atan2(float(target_relative[1]), float(target_relative[0]))
    return abs(signed_angle_delta(target_angle, theta))


def plot_physics_audit(
    dt_rows: list[dict[str, Any]],
    decay_rows: list[dict[str, Any]],
    thrust_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax_dt = axes[0][0]
    for label, color in (("forward", "tab:blue"), ("turn_left", "tab:orange")):
        subset = [row for row in dt_rows if row["action_label"] == label]
        ax_dt.plot(
            [row["dt"] for row in subset],
            [row["forward_progress"] for row in subset],
            marker="o",
            color=color,
            label=label,
        )
    ax_dt.set_title("DT sensitivity")
    ax_dt.set_xlabel("dt")
    ax_dt.set_ylabel("forward progress")
    ax_dt.legend(loc="best")

    ax_decay = axes[0][1]
    for heading in (0.0, math.pi / 2.0, math.pi):
        subset = [row for row in decay_rows if math.isclose(row["heading_rad"], heading)]
        ax_decay.plot(
            [row["step"] for row in subset],
            [row["speed"] for row in subset],
            marker="o",
            label=f"theta={heading:.2f}",
        )
    ax_decay.set_title("Decay from multiple headings")
    ax_decay.set_xlabel("step")
    ax_decay.set_ylabel("speed")
    ax_decay.legend(loc="best")

    ax_thrust = axes[1][0]
    ax_thrust.bar(
        [f"{row['heading_rad']:.2f}" for row in thrust_rows],
        [row["progress_along_heading"] for row in thrust_rows],
        color="tab:green",
    )
    ax_thrust.set_title("Thrust response by heading")
    ax_thrust.set_xlabel("heading")
    ax_thrust.set_ylabel("progress along heading")

    ax_lateral = axes[1][1]
    ax_lateral.bar(
        [f"{row['heading_rad']:.2f}" for row in thrust_rows],
        [row["lateral_error"] for row in thrust_rows],
        color="tab:red",
    )
    ax_lateral.set_title("Lateral error by heading")
    ax_lateral.set_xlabel("heading")
    ax_lateral.set_ylabel("lateral error")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def physics_audit_probe(
    *,
    seed: int,
    steps_short: int,
    steps_long: int,
    output_dir: Path,
    no_plots: bool,
) -> dict[str, Any]:
    reference_dt = 0.05
    dt_values = [0.025, 0.05, 0.1]
    sim_time = steps_long * reference_dt
    dt_rows: list[dict[str, Any]] = []
    for dt in dt_values:
        env = build_env(seed, dt=dt, time_limit=100)
        try:
            steps = max(1, int(round(sim_time / dt)))
            for label, action in (("forward", (1, 2)), ("turn_left", (2, 2))):
                row = run_repeated_action_probe(
                    env,
                    action=action,
                    steps=steps,
                    initial_state=canonical_debug_state(),
                )
                row["dt"] = float(dt)
                row["sim_time"] = float(steps * dt)
                row["action_label"] = label
                dt_rows.append(row)
        finally:
            env.close()
    write_csv(output_dir / "physics_dt_sensitivity.csv", dt_rows)

    dt_reference = {
        row["action_label"]: row
        for row in dt_rows
        if math.isclose(float(row["dt"]), reference_dt)
    }
    dt_max_relative_error = 0.0
    for row in dt_rows:
        reference = dt_reference[row["action_label"]]
        for metric in ("forward_progress", "final_speed", "heading_change_rad"):
            baseline = float(reference[metric])
            current = float(row[metric])
            denom = max(abs(baseline), 1e-6)
            dt_max_relative_error = max(dt_max_relative_error, abs(current - baseline) / denom)
    dt_sensitivity_pass = bool(dt_max_relative_error <= 0.10)

    decay_rows: list[dict[str, Any]] = []
    decay_checks: dict[str, Any] = {}
    env = prepare_env(seed)
    try:
        for heading in (0.0, math.pi / 2.0, math.pi):
            set_state_from_dict(
                env,
                {
                    "position": np.zeros(2, dtype=np.float32),
                    "velocity": heading_vector(heading) * 3.0,
                    "theta": heading,
                    "omega": 2.0,
                    "food_position": DEFAULT_FOOD_POSITION.copy(),
                    "timestep": 0,
                    "prev_turn_action": 1,
                    "prev_push_action": 1,
                },
            )
            speeds: list[float] = []
            omegas: list[float] = []
            for step in range(1, 21):
                _, _, _, _, _ = env.step(np.array([1, 1], dtype=np.int64))
                snapshot = env.get_debug_snapshot()
                speed = float(np.linalg.norm(snapshot["velocity"]))
                abs_omega = abs(float(snapshot["omega"]))
                speeds.append(speed)
                omegas.append(abs_omega)
                decay_rows.append(
                    {
                        "heading_rad": float(heading),
                        "step": int(step),
                        "speed": speed,
                        "abs_omega": abs_omega,
                    }
                )
            speed_ok, speed_violations = monotonic_non_increasing(speeds)
            omega_ok, omega_violations = monotonic_non_increasing(omegas)
            decay_checks[f"{heading:.3f}"] = {
                "speed_monotonic": bool(speed_ok),
                "speed_violations": speed_violations,
                "omega_monotonic": bool(omega_ok),
                "omega_violations": omega_violations,
            }
    finally:
        env.close()
    write_csv(output_dir / "physics_heading_decay.csv", decay_rows)
    decay_pass = all(
        check["speed_monotonic"] and check["omega_monotonic"]
        for check in decay_checks.values()
    )

    thrust_rows: list[dict[str, Any]] = []
    env = prepare_env(seed)
    try:
        for heading in (0.0, math.pi / 2.0, math.pi):
            initial_state = {
                "position": np.zeros(2, dtype=np.float32),
                "velocity": np.zeros(2, dtype=np.float32),
                "theta": heading,
                "omega": 0.0,
                "food_position": DEFAULT_FOOD_POSITION.copy(),
                "timestep": 0,
                "prev_turn_action": 1,
                "prev_push_action": 1,
            }
            row = run_repeated_action_probe(
                env,
                action=(1, 2),
                steps=steps_short,
                initial_state=initial_state,
            )
            displacement = np.array([row["final_x"], row["final_y"]], dtype=np.float32)
            forward_axis = heading_vector(heading)
            progress_along_heading = float(np.dot(displacement, forward_axis))
            lateral_error = float(abs(displacement[0] * forward_axis[1] - displacement[1] * forward_axis[0]))
            row.update(
                {
                    "heading_rad": float(heading),
                    "progress_along_heading": progress_along_heading,
                    "lateral_error": lateral_error,
                }
            )
            thrust_rows.append(row)
    finally:
        env.close()
    write_csv(output_dir / "physics_thrust_response.csv", thrust_rows)
    thrust_response_pass = all(
        row["progress_along_heading"] > 0.0
        and row["lateral_error"] <= max(abs(row["progress_along_heading"]) * 0.25, 0.05)
        for row in thrust_rows
    )

    env = prepare_env(seed)
    try:
        mirrored_a = run_repeated_action_probe(
            env,
            action=(2, 2),
            steps=steps_short,
            initial_state={
                "position": np.zeros(2, dtype=np.float32),
                "velocity": np.array([0.2, -0.1], dtype=np.float32),
                "theta": 0.3,
                "omega": 0.0,
                "food_position": np.array([5.0, 2.0], dtype=np.float32),
                "timestep": 0,
                "prev_turn_action": 1,
                "prev_push_action": 1,
            },
        )
        mirrored_b = run_repeated_action_probe(
            env,
            action=(0, 2),
            steps=steps_short,
            initial_state={
                "position": np.zeros(2, dtype=np.float32),
                "velocity": np.array([0.2, 0.1], dtype=np.float32),
                "theta": -0.3,
                "omega": 0.0,
                "food_position": np.array([5.0, -2.0], dtype=np.float32),
                "timestep": 0,
                "prev_turn_action": 1,
                "prev_push_action": 1,
            },
        )
    finally:
        env.close()
    mirror_invariance_pass = (
        np.isclose(mirrored_a["final_x"], mirrored_b["final_x"], rtol=0.05, atol=0.02)
        and abs(mirrored_a["final_y"] + mirrored_b["final_y"]) <= 0.05
        and abs(mirrored_a["heading_change_rad"] + mirrored_b["heading_change_rad"]) <= 0.05
    )

    if not no_plots:
        plot_physics_audit(dt_rows, decay_rows, thrust_rows, output_dir / "physics_audit.png")

    result = {
        "pass": bool(dt_sensitivity_pass and decay_pass and thrust_response_pass and mirror_invariance_pass),
        "dt_sensitivity_pass": bool(dt_sensitivity_pass),
        "dt_max_relative_error": float(dt_max_relative_error),
        "decay_pass": bool(decay_pass),
        "decay_checks": decay_checks,
        "thrust_response_pass": bool(thrust_response_pass),
        "thrust_rows": thrust_rows,
        "mirror_invariance_pass": bool(mirror_invariance_pass),
        "mirror_rows": {"a": mirrored_a, "b": mirrored_b},
    }
    write_json(output_dir / "physics_audit.json", result)
    return result


def plot_control_contract(
    acquisition_rows: list[dict[str, Any]],
    rotation_errors: dict[str, float],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in acquisition_rows:
        by_label.setdefault(row["label"], []).append(row)
    for label, rows in by_label.items():
        axes[0].plot(
            [row["step"] for row in rows],
            [row["target_error_rad"] for row in rows],
            marker="o",
            label=label,
        )
    axes[0].set_title("Heading acquisition")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("|target bearing error|")
    axes[0].legend(loc="best")

    axes[1].bar(list(rotation_errors.keys()), list(rotation_errors.values()), color="tab:purple")
    axes[1].set_title("Observation rotation consistency error")
    axes[1].set_ylabel("absolute error")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def control_contract_probe(
    *,
    seed: int,
    steps_short: int,
    steps_long: int,
    output_dir: Path,
    no_plots: bool,
) -> dict[str, Any]:
    env = prepare_env(seed)
    try:
        neutral = run_repeated_action_probe(env, action=(1, 1), steps=steps_short, initial_state=canonical_debug_state())
        forward = run_repeated_action_probe(env, action=(1, 2), steps=steps_short, initial_state=canonical_debug_state())
        turn_left = run_repeated_action_probe(env, action=(2, 2), steps=steps_short, initial_state=canonical_debug_state())
        turn_right = run_repeated_action_probe(env, action=(0, 2), steps=steps_short, initial_state=canonical_debug_state())
    finally:
        env.close()
    controllability_pass = (
        math.hypot(neutral["final_x"], neutral["final_y"]) <= 1e-4
        and forward["forward_progress"] > 0.0
        and turn_left["heading_change_rad"] > 0.0
        and turn_right["heading_change_rad"] < 0.0
    )

    scenario_specs = [
        {
            "label": "ahead",
            "food_position": np.array([10.0, 0.0], dtype=np.float32),
            "actions": [np.array([1, 2], dtype=np.int64) for _ in range(steps_long)],
        },
        {
            "label": "left",
            "food_position": np.array([0.0, 10.0], dtype=np.float32),
            "actions": [np.array([2, 1], dtype=np.int64) for _ in range(max(steps_long // 5, 4))]
            + [np.array([1, 2], dtype=np.int64) for _ in range(max(steps_long - max(steps_long // 5, 4), 0))],
        },
        {
            "label": "behind",
            "food_position": np.array([-10.0, 0.0], dtype=np.float32),
            "actions": [np.array([2, 1], dtype=np.int64) for _ in range(max((steps_long * 2) // 5, 8))]
            + [np.array([1, 2], dtype=np.int64) for _ in range(max(steps_long - max((steps_long * 2) // 5, 8), 0))],
        },
    ]
    acquisition_rows: list[dict[str, Any]] = []
    acquisition_pass = True
    env = prepare_env(seed)
    try:
        for spec in scenario_specs:
            set_state_from_dict(
                env,
                {
                    "position": np.zeros(2, dtype=np.float32),
                    "velocity": np.zeros(2, dtype=np.float32),
                    "theta": 0.0,
                    "omega": 0.0,
                    "food_position": spec["food_position"],
                    "timestep": 0,
                    "prev_turn_action": 1,
                    "prev_push_action": 1,
                },
            )
            start_error = wrapped_target_error(env.fish_state.theta, env._target_relative_vector())
            for step_idx, action in enumerate(spec["actions"], start=1):
                _, _, terminated, truncated, _ = env.step(action)
                target_error = wrapped_target_error(env.fish_state.theta, env._target_relative_vector())
                acquisition_rows.append(
                    {
                        "label": spec["label"],
                        "step": int(step_idx),
                        "target_error_rad": float(target_error),
                    }
                )
                if terminated or truncated:
                    break
            end_error = acquisition_rows[-1]["target_error_rad"]
            if end_error > start_error + 1e-6:
                acquisition_pass = False
    finally:
        env.close()

    env = prepare_env(seed)
    try:
        base_state = {
            "position": np.array([0.0, 0.0], dtype=np.float32),
            "velocity": np.array([1.0, 0.2], dtype=np.float32),
            "theta": 0.4,
            "omega": 0.7,
            "food_position": np.array([4.0, 1.0], dtype=np.float32),
            "timestep": 7,
            "prev_turn_action": 2,
            "prev_push_action": 0,
        }
        set_state_from_dict(env, base_state)
        obs_a = env.get_debug_snapshot()["observation"]
        rotated_food = rotate_vec90(base_state["food_position"])
        rotated_velocity = rotate_vec90(base_state["velocity"])
        rotated_theta = base_state["theta"] + math.pi / 2.0
        set_state_from_dict(
            env,
            {
                **base_state,
                "velocity": rotated_velocity,
                "theta": rotated_theta,
                "food_position": rotated_food,
            },
        )
        obs_b = env.get_debug_snapshot()["observation"]
        rotation_errors = {
            "rel_x": abs(float(obs_b[0]) + float(obs_a[1])),
            "rel_y": abs(float(obs_b[1]) - float(obs_a[0])),
            "rel_dist": abs(float(obs_b[2]) - float(obs_a[2])),
            "cos": abs(float(obs_b[3]) + float(obs_a[4])),
            "sin": abs(float(obs_b[4]) - float(obs_a[3])),
            "vx": abs(float(obs_b[5]) + float(obs_a[6])),
            "vy": abs(float(obs_b[6]) - float(obs_a[5])),
            "omega": abs(float(obs_b[7]) - float(obs_a[7])),
        }
        set_state_from_dict(env, base_state)
        obs_target_a = env.get_debug_snapshot()["observation"]
        moved_food = base_state["food_position"] + np.array([1.0, 0.0], dtype=np.float32)
        set_state_from_dict(env, {**base_state, "food_position": moved_food})
        obs_target_b = env.get_debug_snapshot()["observation"]
    finally:
        env.close()

    rotation_consistency_pass = all(error <= 1e-6 for error in rotation_errors.values())
    target_move_consistency_pass = (
        not np.isclose(float(obs_target_a[0]), float(obs_target_b[0]), atol=1e-6)
        and not np.isclose(float(obs_target_a[2]), float(obs_target_b[2]), atol=1e-6)
        and np.allclose(obs_target_a[3:], obs_target_b[3:], atol=1e-6)
    )

    if not no_plots:
        plot_control_contract(acquisition_rows, rotation_errors, output_dir / "control_contract.png")

    result = {
        "pass": bool(controllability_pass and acquisition_pass and rotation_consistency_pass and target_move_consistency_pass),
        "controllability_pass": bool(controllability_pass),
        "neutral_row": neutral,
        "forward_row": forward,
        "turn_left_row": turn_left,
        "turn_right_row": turn_right,
        "heading_acquisition_pass": bool(acquisition_pass),
        "acquisition_rows": acquisition_rows,
        "rotation_consistency_pass": bool(rotation_consistency_pass),
        "rotation_errors": rotation_errors,
        "target_move_consistency_pass": bool(target_move_consistency_pass),
    }
    write_json(output_dir / "control_contract.json", result)
    return result


def main() -> None:
    args = parse_args()
    if args.steps_short <= 0:
        raise ValueError("--steps-short must be > 0.")
    if args.steps_long <= 0:
        raise ValueError("--steps-long must be > 0.")
    if args.steps_long < args.steps_short:
        raise ValueError("--steps-long must be >= --steps-short.")

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    ensure_dir(output_dir)

    if args.probe_set == "baseline":
        env = prepare_env(args.seed)
        try:
            action_result = action_response_probe(
                env,
                steps_short=args.steps_short,
                steps_long=args.steps_long,
                output_dir=output_dir,
                no_plots=args.no_plots,
            )
            decay_result = passive_decay_probe(env, output_dir=output_dir, no_plots=args.no_plots)
            observation_result = observation_sanity_probe(
                env,
                steps_long=args.steps_long,
                output_dir=output_dir,
                no_plots=args.no_plots,
            )
            reward_result = reward_decomposition_probe(
                env,
                steps_long=args.steps_long,
                output_dir=output_dir,
                no_plots=args.no_plots,
            )
        finally:
            env.close()

        summary = {
            "probe_set": args.probe_set,
            "seed": int(args.seed),
            "steps_short": int(args.steps_short),
            "steps_long": int(args.steps_long),
            "overall_pass": bool(
                action_result["pass"]
                and decay_result["pass"]
                and observation_result["pass"]
                and reward_result["pass"]
            ),
            "probes": {
                "action_response": {
                    "pass": bool(action_result["pass"]),
                    "neutral_stationary": bool(action_result["neutral_stationary"]),
                    "thrust_forward": bool(action_result["thrust_forward"]),
                    "mirror_pass": bool(action_result["mirror_pass"]),
                    "mirror_failures": action_result["mirror_failures"],
                },
                "passive_decay": {
                    "pass": bool(decay_result["pass"]),
                    "speed_monotonic": bool(decay_result["speed_monotonic"]),
                    "speed_violations": decay_result["speed_violations"],
                    "omega_monotonic": bool(decay_result["omega_monotonic"]),
                    "omega_violations": decay_result["omega_violations"],
                },
                "observation_sanity": {
                    "pass": bool(observation_result["pass"]),
                    "all_finite": bool(observation_result["all_finite"]),
                    "within_bounds": bool(observation_result["within_bounds"]),
                    "cos_sin_consistent": bool(observation_result["cos_sin_consistent"]),
                    "prev_actions_consistent": bool(observation_result["prev_actions_consistent"]),
                    "progress_monotonic": bool(observation_result["progress_monotonic"]),
                    "toward_motion_consistent": bool(observation_result["toward_motion_consistent"]),
                    "scenario_results": observation_result["scenario_results"],
                },
                "reward_decomposition": {
                    "pass": bool(reward_result["pass"]),
                    "toward_beats_away": bool(reward_result["toward_beats_away"]),
                    "success_beats_timeout": bool(reward_result["success_beats_timeout"]),
                    "toward_total_reward": float(reward_result["toward_total_reward"]),
                    "away_total_reward": float(reward_result["away_total_reward"]),
                    "success_reward": float(reward_result["success_reward"]),
                    "timeout_reward": float(reward_result["timeout_reward"]),
                },
            },
        }
    elif args.probe_set == "physics_audit":
        audit_result = physics_audit_probe(
            seed=args.seed,
            steps_short=args.steps_short,
            steps_long=args.steps_long,
            output_dir=output_dir,
            no_plots=args.no_plots,
        )
        summary = {
            "probe_set": args.probe_set,
            "seed": int(args.seed),
            "steps_short": int(args.steps_short),
            "steps_long": int(args.steps_long),
            "overall_pass": bool(audit_result["pass"]),
            "probes": {
                "physics_audit": {
                    "pass": bool(audit_result["pass"]),
                    "dt_sensitivity_pass": bool(audit_result["dt_sensitivity_pass"]),
                    "dt_max_relative_error": float(audit_result["dt_max_relative_error"]),
                    "decay_pass": bool(audit_result["decay_pass"]),
                    "thrust_response_pass": bool(audit_result["thrust_response_pass"]),
                    "mirror_invariance_pass": bool(audit_result["mirror_invariance_pass"]),
                }
            },
        }
    else:
        contract_result = control_contract_probe(
            seed=args.seed,
            steps_short=args.steps_short,
            steps_long=args.steps_long,
            output_dir=output_dir,
            no_plots=args.no_plots,
        )
        summary = {
            "probe_set": args.probe_set,
            "seed": int(args.seed),
            "steps_short": int(args.steps_short),
            "steps_long": int(args.steps_long),
            "overall_pass": bool(contract_result["pass"]),
            "probes": {
                "control_contract": {
                    "pass": bool(contract_result["pass"]),
                    "controllability_pass": bool(contract_result["controllability_pass"]),
                    "heading_acquisition_pass": bool(contract_result["heading_acquisition_pass"]),
                    "rotation_consistency_pass": bool(contract_result["rotation_consistency_pass"]),
                    "target_move_consistency_pass": bool(contract_result["target_move_consistency_pass"]),
                }
            },
        }
    write_json(output_dir / "summary.json", summary)

    print("V5 environment validation complete")
    print(f"Probe set: {args.probe_set}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Overall pass: {summary['overall_pass']}")
    for probe_name, probe_summary in summary["probes"].items():
        print(f"{probe_name}: pass={probe_summary['pass']}")


if __name__ == "__main__":
    main()
