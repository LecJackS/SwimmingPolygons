"""Env-only validation harness for V4 simulator choices."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from triangles import OctopusEnv


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FOOD_POSITION = np.array([10.0, 0.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic V4 environment validation probes.")
    parser.add_argument("--probe-set", type=str, default="baseline")
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


def prepare_env(seed: int) -> OctopusEnv:
    env = OctopusEnv(
        epsilon=0.0,
        render_mode=None,
        enable_curriculum=False,
        fixed_food_distance=10.0,
        time_limit=100,
    )
    env.reset(seed=seed)
    return env


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


def main() -> None:
    args = parse_args()
    if args.probe_set != "baseline":
        raise ValueError("Only --probe-set baseline is currently supported.")
    if args.steps_short <= 0:
        raise ValueError("--steps-short must be > 0.")
    if args.steps_long <= 0:
        raise ValueError("--steps-long must be > 0.")
    if args.steps_long < args.steps_short:
        raise ValueError("--steps-long must be >= --steps-short.")

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    ensure_dir(output_dir)

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
    write_json(output_dir / "summary.json", summary)

    print("V4 environment validation complete")
    print(f"Probe set: {args.probe_set}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Overall pass: {summary['overall_pass']}")
    for probe_name, probe_summary in summary["probes"].items():
        print(f"{probe_name}: pass={probe_summary['pass']}")


if __name__ == "__main__":
    main()
