"""V6 RLlib PPO training entrypoint for continuous foraging."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import re
from typing import Any
import warnings

import numpy as np
import pyarrow.fs as pafs
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import NoopLogger
from ray.tune.registry import register_env
import torch

from eval_utils import (
    DEFAULT_FOOD_COUNT,
    DEFAULT_PELLET_REWARD,
    DEFAULT_SENSOR_NUM_SECTORS,
    DEFAULT_SENSOR_RADIUS,
    DEFAULT_SENSOR_RING_EDGES,
    DEFAULT_STEP_COST,
    DEFAULT_TIME_LIMIT,
    ForagingEvalResult,
    compare_foraging_results,
    evaluate_env_rollouts,
    flatten_foraging_result,
    parse_ring_edges,
)
from triangles import OctopusEnv


ENV_ID = "v6_articulated_fish_foraging_env"

warnings.filterwarnings(
    "ignore",
    message=r".*multi_gpu_train_one_step.*deprecated.*",
    category=DeprecationWarning,
)
logging.getLogger("ray.rllib.execution.train_ops").setLevel(logging.ERROR)
logging.getLogger("ray.rllib.utils.sgd").setLevel(logging.ERROR)
logging.getLogger("ray._common.deprecation").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V6 articulated fish policy on continuous foraging.")
    parser.add_argument("--train-iterations", type=int, default=100)
    parser.add_argument("--num-env-runners", type=int, default=8)
    parser.add_argument("--num-envs-per-runner", type=int, default=2)
    parser.add_argument("--checkpoint-every-iterations", type=int, default=5)
    parser.add_argument("--checkpoint-root", type=str, default="./rllib_checkpoints_baseline_v6_foraging")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda. Defaults to SB3_DEVICE or auto.")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--food-count", type=int, default=DEFAULT_FOOD_COUNT)
    parser.add_argument("--time-limit", type=int, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--pellet-reward", type=float, default=DEFAULT_PELLET_REWARD)
    parser.add_argument("--step-cost", type=float, default=DEFAULT_STEP_COST)
    parser.add_argument("--sensor-radius", type=float, default=DEFAULT_SENSOR_RADIUS)
    parser.add_argument(
        "--sensor-ring-edges",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_SENSOR_RING_EDGES),
        help="Comma-separated radial edges for the local polar food sensor.",
    )
    parser.add_argument("--sensor-sectors", type=int, default=DEFAULT_SENSOR_NUM_SECTORS)
    parser.add_argument("--eval-report-episodes", type=int, default=10)
    parser.add_argument("--eval-report-seed", type=int, default=20_240)
    return parser.parse_args()


def resolve_device(cli_device: str | None) -> str:
    if cli_device:
        return cli_device
    env_device = os.getenv("SB3_DEVICE")
    if env_device:
        return env_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_args(args: argparse.Namespace, sensor_ring_edges: list[float]) -> None:
    if args.train_iterations <= 0:
        raise ValueError("--train-iterations must be > 0.")
    if args.num_env_runners <= 0:
        raise ValueError("--num-env-runners must be > 0.")
    if args.num_envs_per_runner <= 0:
        raise ValueError("--num-envs-per-runner must be > 0.")
    if args.checkpoint_every_iterations <= 0:
        raise ValueError("--checkpoint-every-iterations must be > 0.")
    if args.food_count <= 0:
        raise ValueError("--food-count must be > 0.")
    if args.time_limit <= 0:
        raise ValueError("--time-limit must be > 0.")
    if args.pellet_reward <= 0.0:
        raise ValueError("--pellet-reward must be > 0.")
    if args.step_cost < 0.0:
        raise ValueError("--step-cost must be >= 0.")
    if args.sensor_radius <= 0.0:
        raise ValueError("--sensor-radius must be > 0.")
    if args.sensor_sectors <= 0:
        raise ValueError("--sensor-sectors must be > 0.")
    if not sensor_ring_edges:
        raise ValueError("--sensor-ring-edges must be non-empty.")
    if sensor_ring_edges[-1] > args.sensor_radius + 1e-6:
        raise ValueError("--sensor-ring-edges must lie within --sensor-radius.")
    if args.eval_report_episodes <= 0:
        raise ValueError("--eval-report-episodes must be > 0.")


def make_env(env_config: dict) -> OctopusEnv:
    return OctopusEnv(
        epsilon=float(env_config.get("epsilon", 0.0)),
        render_mode=env_config.get("render_mode"),
        fish_preset=env_config.get("fish_preset"),
        time_limit=int(env_config.get("time_limit", DEFAULT_TIME_LIMIT)),
        food_count=int(env_config.get("food_count", DEFAULT_FOOD_COUNT)),
        food_capture_radius=float(env_config.get("food_capture_radius", 0.45)),
        pellet_reward=float(env_config.get("pellet_reward", DEFAULT_PELLET_REWARD)),
        step_cost=float(env_config.get("step_cost", DEFAULT_STEP_COST)),
        sensor_radius=float(env_config.get("sensor_radius", DEFAULT_SENSOR_RADIUS)),
        sensor_ring_edges=env_config.get("sensor_ring_edges", DEFAULT_SENSOR_RING_EDGES),
        sensor_num_sectors=int(env_config.get("sensor_num_sectors", DEFAULT_SENSOR_NUM_SECTORS)),
        show_sensor_overlay=bool(env_config.get("show_sensor_overlay", False)),
    )


def build_env_config(args: argparse.Namespace, *, show_sensor_overlay: bool = False) -> dict[str, Any]:
    return {
        "epsilon": float(args.epsilon),
        "render_mode": None,
        "time_limit": int(args.time_limit),
        "food_count": int(args.food_count),
        "food_capture_radius": 0.45,
        "pellet_reward": float(args.pellet_reward),
        "step_cost": float(args.step_cost),
        "sensor_radius": float(args.sensor_radius),
        "sensor_ring_edges": parse_ring_edges(args.sensor_ring_edges),
        "sensor_num_sectors": int(args.sensor_sectors),
        "show_sensor_overlay": bool(show_sensor_overlay),
    }


def build_algo(args: argparse.Namespace, *, num_gpus: int, env_config: dict[str, Any]):
    config = (
        PPOConfig()
        .environment(env=ENV_ID, env_config=env_config)
        .framework("torch")
        .resources(num_gpus=num_gpus)
        .env_runners(num_env_runners=args.num_env_runners, num_envs_per_env_runner=args.num_envs_per_runner)
        .training(gamma=0.9, lr=1e-3, entropy_coeff=0.01, num_epochs=10)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .fault_tolerance(
            restart_failed_env_runners=False,
            max_num_env_runner_restarts=0,
        )
        .debugging(seed=args.seed, logger_creator=lambda cfg: NoopLogger(cfg, "."))
    )
    return config.build_algo()


def format_checkpoint_path(saved_obj: Any) -> str:
    if hasattr(saved_obj, "path"):
        return str(saved_obj.path)
    checkpoint = getattr(saved_obj, "checkpoint", None)
    if checkpoint is not None and hasattr(checkpoint, "path"):
        return str(checkpoint.path)
    match = re.search(r"path=([^,\)]+)", str(saved_obj))
    if match:
        return match.group(1)
    return str(saved_obj)


def save_algorithm_checkpoint(algo, checkpoint_path: Path, filesystem: pafs.LocalFileSystem) -> str:
    abs_path = checkpoint_path.resolve()
    try:
        saved = algo.save_to_path(path=abs_path, filesystem=filesystem)
        return format_checkpoint_path(saved)
    except RuntimeError as exc:
        if "not supported on the old API stack" not in str(exc):
            raise
        saved = algo.save(checkpoint_dir=str(abs_path))
        return format_checkpoint_path(saved)


def _first_finite_float(candidates: list[Any]) -> float:
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return float("nan")


def _first_non_negative_int(candidates: list[Any]) -> int:
    for candidate in candidates:
        try:
            value = int(candidate)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            return value
    return -1


def format_metric(value: float, precision: int = 3) -> str:
    if np.isfinite(value):
        return f"{value:.{precision}f}"
    return "nan"


def extract_reward_mean(result: dict[str, Any]) -> float:
    env_runners = result.get("env_runners", {})
    sampler_results = result.get("sampler_results", {})
    return _first_finite_float(
        [
            result.get("episode_reward_mean"),
            result.get("episode_return_mean"),
            env_runners.get("episode_return_mean"),
            env_runners.get("episode_reward_mean"),
            sampler_results.get("episode_reward_mean"),
        ]
    )


def extract_timesteps_total(result: dict[str, Any]) -> int:
    counters = result.get("counters", {})
    return _first_non_negative_int(
        [
            result.get("timesteps_total"),
            result.get("num_env_steps_sampled_lifetime"),
            counters.get("num_env_steps_sampled"),
        ]
    )


def make_eval_env_factory(args: argparse.Namespace):
    env_config = build_env_config(args, show_sensor_overlay=False)
    return lambda: OctopusEnv(**env_config)


def run_foraging_eval(algo, *, args: argparse.Namespace, base_seed: int) -> ForagingEvalResult:
    return evaluate_env_rollouts(
        algo=algo,
        env_factory=make_eval_env_factory(args),
        num_episodes=args.eval_report_episodes,
        base_seed=base_seed,
        stack_mode="old",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_flat_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_run_summary(
    *,
    best_checkpoint_record: dict[str, Any] | None,
    best_eval_result: ForagingEvalResult | None,
    num_checkpoint_evaluations: int,
    time_to_first_positive_return: dict[str, Any] | None,
    time_to_food_rate_threshold: dict[str, Any] | None,
    final_checkpoint_path: str | None,
) -> dict[str, Any]:
    return {
        "best_checkpoint": best_checkpoint_record,
        "best_mean_food_eaten": float(best_eval_result.mean_food_eaten) if best_eval_result is not None else None,
        "best_food_per_100_steps": float(best_eval_result.food_per_100_steps) if best_eval_result is not None else None,
        "best_mean_reward": float(best_eval_result.mean_reward) if best_eval_result is not None else None,
        "num_checkpoint_evaluations": int(num_checkpoint_evaluations),
        "time_to_first_positive_return": time_to_first_positive_return,
        "time_to_food_rate_threshold": time_to_food_rate_threshold,
        "final_checkpoint_path": final_checkpoint_path,
    }


def main() -> None:
    args = parse_args()
    sensor_ring_edges = parse_ring_edges(args.sensor_ring_edges)
    validate_args(args, sensor_ring_edges)

    device = resolve_device(args.device)
    num_gpus = 1 if device == "cuda" else 0
    checkpoint_root = Path(args.checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    eval_report_jsonl_path = checkpoint_root / "eval_reports.jsonl"
    eval_report_csv_path = checkpoint_root / "eval_reports.csv"
    run_summary_path = checkpoint_root / "run_summary.json"
    local_fs = pafs.LocalFileSystem()

    register_env(ENV_ID, make_env)
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=True)

    algo = build_algo(
        args,
        num_gpus=num_gpus,
        env_config=build_env_config(args, show_sensor_overlay=False),
    )

    print("V6 - Articulated Fish RLlib training (continuous foraging)")
    print(
        "Config: "
        f"iterations={args.train_iterations}, "
        f"env_runners={args.num_env_runners}, "
        f"envs_per_runner={args.num_envs_per_runner}, "
        f"checkpoint_every={args.checkpoint_every_iterations}, "
        f"device={device}, num_gpus={num_gpus}, "
        f"checkpoint_root={checkpoint_root.resolve()}, "
        f"food_count={args.food_count}, "
        f"time_limit={args.time_limit}, "
        f"pellet_reward={args.pellet_reward}, "
        f"step_cost={args.step_cost}, "
        f"sensor_radius={args.sensor_radius}, "
        f"sensor_ring_edges={sensor_ring_edges}, "
        f"sensor_sectors={args.sensor_sectors}, "
        f"eval_report_episodes={args.eval_report_episodes}, "
        f"eval_report_seed={args.eval_report_seed}"
    )

    latest_checkpoint = None
    report_rows_nested: list[dict[str, Any]] = []
    report_rows_flat: list[dict[str, Any]] = []
    best_checkpoint_record: dict[str, Any] | None = None
    best_eval_result: ForagingEvalResult | None = None
    time_to_first_positive_return: dict[str, Any] | None = None
    time_to_food_rate_threshold: dict[str, Any] | None = None

    try:
        for iteration in range(1, args.train_iterations + 1):
            result = algo.train()
            reward_mean = extract_reward_mean(result)
            timesteps_total = extract_timesteps_total(result)
            print(
                f"iter={iteration:03d} "
                f"timesteps_total={timesteps_total} "
                f"episode_reward_mean={format_metric(reward_mean)}"
            )

            if iteration % args.checkpoint_every_iterations != 0:
                continue

            checkpoint_path = checkpoint_root / f"checkpoint_{iteration:05d}"
            latest_checkpoint = save_algorithm_checkpoint(algo, checkpoint_path, local_fs)
            print(f"checkpoint_saved: {latest_checkpoint}")

            eval_result = run_foraging_eval(
                algo,
                args=args,
                base_seed=args.eval_report_seed + (iteration * 1_000_000),
            )
            report_nested = {
                "iteration": int(iteration),
                "timesteps_total": int(timesteps_total),
                "checkpoint_path": str(latest_checkpoint),
                "training_reward_mean": float(reward_mean),
                "eval_result": eval_result.to_dict(),
            }
            report_flat = {
                "iteration": int(iteration),
                "timesteps_total": int(timesteps_total),
                "checkpoint_path": str(latest_checkpoint),
                "training_reward_mean": float(reward_mean),
                **flatten_foraging_result(eval_result),
            }
            report_rows_nested.append(report_nested)
            report_rows_flat.append(report_flat)
            write_jsonl(eval_report_jsonl_path, report_rows_nested)
            write_flat_csv(eval_report_csv_path, report_rows_flat)

            if compare_foraging_results(eval_result, best_eval_result) >= 0:
                best_eval_result = eval_result
                best_checkpoint_record = {
                    "iteration": int(iteration),
                    "timesteps_total": int(timesteps_total),
                    "checkpoint_path": str(latest_checkpoint),
                    **flatten_foraging_result(eval_result),
                }

            if time_to_first_positive_return is None and np.isfinite(eval_result.mean_reward) and eval_result.mean_reward > 0.0:
                time_to_first_positive_return = {
                    "iteration": int(iteration),
                    "timesteps_total": int(timesteps_total),
                    "mean_reward": float(eval_result.mean_reward),
                }

            summary = build_run_summary(
                best_checkpoint_record=best_checkpoint_record,
                best_eval_result=best_eval_result,
                num_checkpoint_evaluations=len(report_rows_nested),
                time_to_first_positive_return=time_to_first_positive_return,
                time_to_food_rate_threshold=time_to_food_rate_threshold,
                final_checkpoint_path=None,
            )
            run_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

            print(
                "eval_report: "
                f"iter={iteration:03d} "
                f"mean_food_eaten={format_metric(eval_result.mean_food_eaten)} "
                f"food_per_100_steps={format_metric(eval_result.food_per_100_steps)} "
                f"mean_reward={format_metric(eval_result.mean_reward)} "
                f"mean_visible_food_count={format_metric(eval_result.mean_visible_food_count)} "
                f"jsonl={eval_report_jsonl_path.name}"
            )
    finally:
        final_checkpoint_path = checkpoint_root / "checkpoint_final"
        try:
            latest_checkpoint = save_algorithm_checkpoint(algo, final_checkpoint_path, local_fs)
            print(f"final_checkpoint_saved: {latest_checkpoint}")
        finally:
            summary = build_run_summary(
                best_checkpoint_record=best_checkpoint_record,
                best_eval_result=best_eval_result,
                num_checkpoint_evaluations=len(report_rows_nested),
                time_to_first_positive_return=time_to_first_positive_return,
                time_to_food_rate_threshold=time_to_food_rate_threshold,
                final_checkpoint_path=str(final_checkpoint_path.resolve()),
            )
            run_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            print("training_status: reached_iteration_budget")
            algo.stop()
            ray.shutdown()


if __name__ == "__main__":
    main()
