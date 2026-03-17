"""V5 RLlib PPO training entrypoint with staged curriculum."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
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
    DEFAULT_CURRICULUM_STAGES,
    DistanceEvalResult,
    compute_stage_time_limit,
    evaluate_env_rollouts,
    flatten_distance_results,
    parse_stage_distances,
    weighted_success_score,
)
from triangles import OctopusEnv


ENV_ID = "v5_octopus_env"

# Reduce noisy upstream deprecation logs from RLlib internals that have no
# direct user-land replacement in Ray 2.54.
warnings.filterwarnings(
    "ignore",
    message=r".*multi_gpu_train_one_step.*deprecated.*",
    category=DeprecationWarning,
)
logging.getLogger("ray.rllib.execution.train_ops").setLevel(logging.ERROR)
logging.getLogger("ray.rllib.utils.sgd").setLevel(logging.ERROR)
logging.getLogger("ray._common.deprecation").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V5 fish policy with RLlib PPO.")
    parser.add_argument("--train-iterations", type=int, default=100)
    parser.add_argument("--num-env-runners", type=int, default=8)
    parser.add_argument("--num-envs-per-runner", type=int, default=2)
    parser.add_argument("--checkpoint-every-iterations", type=int, default=5)
    parser.add_argument("--checkpoint-root", type=str, default="./rllib_checkpoints")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda. Defaults to SB3_DEVICE or auto.")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--curriculum-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable global fixed-distance stage curriculum.",
    )
    parser.add_argument(
        "--curriculum-stages",
        type=str,
        default=",".join(str(v) for v in DEFAULT_CURRICULUM_STAGES),
        help="Comma-separated fixed distances for staged curriculum.",
    )
    parser.add_argument("--curriculum-success-rate", type=float, default=0.8)
    parser.add_argument("--curriculum-eval-episodes", type=int, default=20)
    parser.add_argument("--curriculum-consecutive-evals", type=int, default=2)
    parser.add_argument("--curriculum-time-limit-base", type=int, default=100)
    parser.add_argument("--curriculum-time-limit-max", type=int, default=180)
    parser.add_argument(
        "--eval-report-distances",
        type=str,
        default=",".join(str(v) for v in DEFAULT_CURRICULUM_STAGES),
        help="Comma-separated fixed distances to score at each checkpoint.",
    )
    parser.add_argument("--eval-report-episodes", type=int, default=5)
    parser.add_argument("--eval-report-seed", type=int, default=20_240)

    parser.add_argument(
        "--early-stop-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable final early stopping based on fixed-distance evaluation.",
    )
    parser.add_argument("--early-stop-distance", type=float, default=10.0)
    parser.add_argument("--early-stop-success-rate", type=float, default=0.9)
    parser.add_argument("--early-stop-eval-episodes", type=int, default=20)
    parser.add_argument("--early-stop-consecutive-evals", type=int, default=3)
    parser.add_argument(
        "--early-stop-reward-floor",
        type=float,
        default=None,
        help="Optional training reward floor required alongside early-stop eval pass.",
    )
    parser.add_argument(
        "--early-stop-eval-seed",
        type=int,
        default=12345,
        help="Base seed for deterministic fixed-distance evaluations.",
    )
    return parser.parse_args()


def resolve_device(cli_device: str | None) -> str:
    if cli_device:
        return cli_device
    env_device = os.getenv("SB3_DEVICE")
    if env_device:
        return env_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_args(
    args: argparse.Namespace,
    stage_distances: list[float],
    eval_report_distances: list[float],
) -> None:
    if args.train_iterations <= 0:
        raise ValueError("--train-iterations must be > 0.")
    if args.num_env_runners <= 0:
        raise ValueError("--num-env-runners must be > 0.")
    if args.num_envs_per_runner <= 0:
        raise ValueError("--num-envs-per-runner must be > 0.")
    if args.checkpoint_every_iterations <= 0:
        raise ValueError("--checkpoint-every-iterations must be > 0.")

    if not stage_distances:
        raise ValueError("--curriculum-stages must contain at least one distance.")
    if not (0.0 <= args.curriculum_success_rate <= 1.0):
        raise ValueError("--curriculum-success-rate must be in [0, 1].")
    if args.curriculum_eval_episodes <= 0:
        raise ValueError("--curriculum-eval-episodes must be > 0.")
    if args.curriculum_consecutive_evals <= 0:
        raise ValueError("--curriculum-consecutive-evals must be > 0.")
    if args.curriculum_time_limit_base <= 0:
        raise ValueError("--curriculum-time-limit-base must be > 0.")
    if args.curriculum_time_limit_max < args.curriculum_time_limit_base:
        raise ValueError("--curriculum-time-limit-max must be >= --curriculum-time-limit-base.")
    if not eval_report_distances:
        raise ValueError("--eval-report-distances must contain at least one distance.")
    if args.eval_report_episodes <= 0:
        raise ValueError("--eval-report-episodes must be > 0.")

    if args.early_stop_distance <= 0:
        raise ValueError("--early-stop-distance must be > 0.")
    if not (0.0 <= args.early_stop_success_rate <= 1.0):
        raise ValueError("--early-stop-success-rate must be in [0, 1].")
    if args.early_stop_eval_episodes <= 0:
        raise ValueError("--early-stop-eval-episodes must be > 0.")
    if args.early_stop_consecutive_evals <= 0:
        raise ValueError("--early-stop-consecutive-evals must be > 0.")


def make_env(config: dict[str, Any]) -> OctopusEnv:
    return OctopusEnv(
        epsilon=float(config.get("epsilon", 0.0)),
        render_mode=config.get("render_mode"),
        enable_curriculum=bool(config.get("enable_curriculum", False)),
        fixed_food_distance=config.get("fixed_food_distance"),
        time_limit=int(config.get("time_limit", 100)),
    )


def build_env_config(args: argparse.Namespace, *, distance: float, time_limit: int) -> dict[str, Any]:
    return {
        "epsilon": args.epsilon,
        "render_mode": None,
        "enable_curriculum": False,
        "fixed_food_distance": float(distance),
        "time_limit": int(time_limit),
    }


def build_algo(
    args: argparse.Namespace,
    *,
    num_gpus: int,
    env_config: dict[str, Any],
):
    config = (
        PPOConfig()
        .environment(env=ENV_ID, env_config=env_config)
        .framework("torch")
        .resources(num_gpus=num_gpus)
        .env_runners(
            num_env_runners=args.num_env_runners,
            num_envs_per_env_runner=args.num_envs_per_runner,
            rollout_fragment_length="auto",
        )
        .training(
            gamma=0.9,
            lr=1e-3,
            entropy_coeff=0.01,
            num_epochs=10,
        )
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


def compute_stage_success_threshold(
    distance: float,
    *,
    min_stage_distance: float,
    max_stage_distance: float,
    base_threshold: float,
) -> float:
    threshold = float(base_threshold)
    if max_stage_distance > min_stage_distance:
        ratio = (float(distance) - min_stage_distance) / (max_stage_distance - min_stage_distance)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        if ratio >= 0.5:
            threshold -= 0.05
        if ratio >= 0.75:
            threshold -= 0.05
    return float(np.clip(threshold, 0.0, 1.0))


def format_checkpoint_path(saved_obj: Any) -> str:
    if hasattr(saved_obj, "path"):
        return str(saved_obj.path)
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
        if hasattr(saved, "checkpoint") and hasattr(saved.checkpoint, "path"):
            return str(saved.checkpoint.path)
        return str(abs_path)


def _try_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _first_finite_float(candidates: list[Any]) -> float:
    for candidate in candidates:
        value = _try_float(candidate)
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


def make_eval_env_factory(*, distance: float, time_limit: int):
    return lambda: OctopusEnv(
        epsilon=0.0,
        render_mode=None,
        enable_curriculum=False,
        fixed_food_distance=distance,
        time_limit=time_limit,
    )


def run_fixed_distance_eval(
    algo,
    *,
    distance: float,
    time_limit: int,
    num_episodes: int,
    base_seed: int,
) -> dict[str, float]:
    result = evaluate_env_rollouts(
        algo=algo,
        env_factory=make_eval_env_factory(distance=distance, time_limit=time_limit),
        num_episodes=num_episodes,
        base_seed=base_seed,
        stack_mode="old",
    )
    return {
        "success_rate": float(result["success_rate"]),
        "mean_steps": float(result["mean_steps"]),
        "mean_reward": float(result["mean_reward"]),
    }


def evaluate_distance_suite(
    algo,
    *,
    distances: list[float],
    num_episodes: int,
    base_seed: int,
    min_stage_distance: float,
    max_stage_distance: float,
    base_limit: int,
    max_limit: int,
) -> list[DistanceEvalResult]:
    results: list[DistanceEvalResult] = []
    for distance_index, distance in enumerate(distances):
        time_limit = compute_stage_time_limit(
            distance,
            min_stage_distance=min_stage_distance,
            max_stage_distance=max_stage_distance,
            base_limit=base_limit,
            max_limit=max_limit,
        )
        metrics = evaluate_env_rollouts(
            algo=algo,
            env_factory=make_eval_env_factory(distance=distance, time_limit=time_limit),
            num_episodes=num_episodes,
            base_seed=base_seed + (distance_index * 10_000),
            stack_mode="old",
        )
        results.append(
            DistanceEvalResult(
                distance=float(distance),
                time_limit=int(time_limit),
                success_rate=float(metrics["success_rate"]),
                mean_steps=float(metrics["mean_steps"]),
                mean_reward=float(metrics["mean_reward"]),
                successes=int(metrics["successes"]),
                episodes=int(metrics["episodes"]),
            )
        )
    return results


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_flat_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_run_summary(
    *,
    run_reports: list[dict[str, Any]],
    best_checkpoint_record: dict[str, Any] | None,
    highest_stage_reached: int,
    first_promotion: dict[str, Any] | None,
    stage_six_reached: dict[str, Any] | None,
    early_stop_event: dict[str, Any] | None,
) -> dict[str, Any]:
    best_per_distance: dict[str, float] = {}
    if run_reports:
        distance_keys = run_reports[0]["distance_results"].keys()
        for distance_key in distance_keys:
            best_per_distance[distance_key] = max(
                float(report["distance_results"][distance_key]["success_rate"]) for report in run_reports
            )

    return {
        "num_checkpoint_evaluations": len(run_reports),
        "highest_stage_reached": int(highest_stage_reached),
        "best_checkpoint": best_checkpoint_record,
        "best_per_distance_success_rate": best_per_distance,
        "time_to_first_promotion": first_promotion,
        "time_to_stage_6": stage_six_reached,
        "distance_10_early_stop": early_stop_event,
    }


def main() -> None:
    args = parse_args()
    stage_distances = parse_stage_distances(args.curriculum_stages)
    eval_report_distances = parse_stage_distances(args.eval_report_distances)
    validate_args(args, stage_distances, eval_report_distances)

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

    min_stage_distance = float(stage_distances[0])
    max_stage_distance = float(stage_distances[-1])

    if args.curriculum_enabled:
        stage_index = 0
        current_distance = float(stage_distances[stage_index])
    else:
        stage_index = len(stage_distances) - 1
        current_distance = float(args.early_stop_distance)

    current_time_limit = compute_stage_time_limit(
        current_distance,
        min_stage_distance=min_stage_distance,
        max_stage_distance=max_stage_distance,
        base_limit=args.curriculum_time_limit_base,
        max_limit=args.curriculum_time_limit_max,
    )

    algo = build_algo(
        args,
        num_gpus=num_gpus,
        env_config=build_env_config(args, distance=current_distance, time_limit=current_time_limit),
    )

    print("V5 - Validated Physics RLlib training")
    print(
        "Config: "
        f"iterations={args.train_iterations}, "
        f"env_runners={args.num_env_runners}, "
        f"envs_per_runner={args.num_envs_per_runner}, "
        f"checkpoint_every={args.checkpoint_every_iterations}, "
        f"device={device}, num_gpus={num_gpus}, "
        f"checkpoint_root={checkpoint_root.resolve()}, "
        f"curriculum_enabled={args.curriculum_enabled}, "
        f"curriculum_stages={stage_distances}, "
        f"curriculum_success_rate={args.curriculum_success_rate}, "
        f"curriculum_eval_episodes={args.curriculum_eval_episodes}, "
        f"curriculum_consecutive_evals={args.curriculum_consecutive_evals}, "
        f"curriculum_time_limit_base={args.curriculum_time_limit_base}, "
        f"curriculum_time_limit_max={args.curriculum_time_limit_max}, "
        f"eval_report_distances={eval_report_distances}, "
        f"eval_report_episodes={args.eval_report_episodes}, "
        f"eval_report_seed={args.eval_report_seed}, "
        f"early_stop_enabled={args.early_stop_enabled}, "
        f"early_stop_distance={args.early_stop_distance}, "
        f"early_stop_success_rate={args.early_stop_success_rate}, "
        f"early_stop_eval_episodes={args.early_stop_eval_episodes}, "
        f"early_stop_consecutive_evals={args.early_stop_consecutive_evals}, "
        f"early_stop_reward_floor={args.early_stop_reward_floor}, "
        f"early_stop_eval_seed={args.early_stop_eval_seed}"
    )
    print(
        "Active stage: "
        f"index={stage_index}, distance={current_distance:.3f}, time_limit={current_time_limit}"
    )

    latest_checkpoint = None
    stage_pass_streak = 0
    early_stop_pass_streak = 0
    early_stop_triggered = False
    highest_stage_reached = int(stage_index)
    first_promotion: dict[str, Any] | None = None
    stage_six_reached: dict[str, Any] | None = None
    early_stop_event: dict[str, Any] | None = None
    best_checkpoint_record: dict[str, Any] | None = None
    best_checkpoint_score = float("-inf")
    report_rows_nested: list[dict[str, Any]] = []
    report_rows_flat: list[dict[str, Any]] = []

    try:
        for i in range(1, args.train_iterations + 1):
            result = algo.train()
            reward_mean = extract_reward_mean(result)
            timesteps_total = extract_timesteps_total(result)
            print(
                f"iter={i:03d} "
                f"timesteps_total={timesteps_total} "
                f"episode_reward_mean={format_metric(reward_mean)}"
            )

            if i % args.checkpoint_every_iterations != 0:
                continue

            checkpoint_path = checkpoint_root / f"checkpoint_{i:05d}"
            latest_checkpoint = save_algorithm_checkpoint(algo, checkpoint_path, local_fs)
            print(f"checkpoint_saved: {latest_checkpoint}")

            report_results = evaluate_distance_suite(
                algo,
                distances=eval_report_distances,
                num_episodes=args.eval_report_episodes,
                base_seed=args.eval_report_seed + (i * 1_000_000),
                min_stage_distance=min_stage_distance,
                max_stage_distance=max_stage_distance,
                base_limit=args.curriculum_time_limit_base,
                max_limit=args.curriculum_time_limit_max,
            )
            report_score = weighted_success_score(report_results)
            report_nested = {
                "iteration": int(i),
                "timesteps_total": int(timesteps_total),
                "checkpoint_path": str(latest_checkpoint),
                "current_stage_index": int(stage_index),
                "current_stage_distance": float(current_distance),
                "current_stage_time_limit": int(current_time_limit),
                "training_reward_mean": float(reward_mean),
                "weighted_success_score": float(report_score),
                "distance_results": {
                    f"{result.distance:g}": result.to_dict() for result in report_results
                },
            }
            report_flat = {
                "iteration": int(i),
                "timesteps_total": int(timesteps_total),
                "checkpoint_path": str(latest_checkpoint),
                "current_stage_index": int(stage_index),
                "current_stage_distance": float(current_distance),
                "current_stage_time_limit": int(current_time_limit),
                "training_reward_mean": float(reward_mean),
                "weighted_success_score": float(report_score),
                **flatten_distance_results(report_results),
            }
            report_rows_nested.append(report_nested)
            report_rows_flat.append(report_flat)
            write_jsonl(eval_report_jsonl_path, report_rows_nested)
            write_flat_csv(eval_report_csv_path, report_rows_flat)

            if np.isfinite(report_score) and report_score >= best_checkpoint_score:
                best_checkpoint_score = float(report_score)
                best_checkpoint_record = {
                    "iteration": int(i),
                    "timesteps_total": int(timesteps_total),
                    "checkpoint_path": str(latest_checkpoint),
                    "weighted_success_score": float(report_score),
                }

            distance_summary = " ".join(
                f"{result.distance:g}:{format_metric(result.success_rate)}"
                for result in report_results
            )
            print(
                "eval_report: "
                f"iter={i:03d} "
                f"score={format_metric(report_score)} "
                f"success_by_distance=[{distance_summary}] "
                f"jsonl={eval_report_jsonl_path.name}"
            )

            if args.early_stop_enabled:
                early_stop_time_limit = compute_stage_time_limit(
                    args.early_stop_distance,
                    min_stage_distance=min_stage_distance,
                    max_stage_distance=max_stage_distance,
                    base_limit=args.curriculum_time_limit_base,
                    max_limit=args.curriculum_time_limit_max,
                )
                eval_stats = run_fixed_distance_eval(
                    algo,
                    distance=args.early_stop_distance,
                    time_limit=early_stop_time_limit,
                    num_episodes=args.early_stop_eval_episodes,
                    base_seed=args.early_stop_eval_seed,
                )
                eval_success_rate = eval_stats["success_rate"]
                eval_mean_steps = eval_stats["mean_steps"]
                eval_mean_reward = eval_stats["mean_reward"]

                metrics_valid = (
                    np.isfinite(eval_success_rate)
                    and np.isfinite(eval_mean_steps)
                    and np.isfinite(eval_mean_reward)
                )
                eval_gate_pass = metrics_valid and (eval_success_rate >= args.early_stop_success_rate)

                reward_gate_enabled = args.early_stop_reward_floor is not None
                reward_gate_pass = True
                if reward_gate_enabled:
                    reward_gate_pass = np.isfinite(reward_mean) and (
                        reward_mean >= float(args.early_stop_reward_floor)
                    )

                early_stop_pass = eval_gate_pass and reward_gate_pass
                early_stop_pass_streak = early_stop_pass_streak + 1 if early_stop_pass else 0

                print(
                    "early_stop_eval: "
                    f"iter={i:03d} "
                    f"distance={args.early_stop_distance:.3f} "
                    f"time_limit={early_stop_time_limit} "
                    f"success_rate={format_metric(eval_success_rate)} "
                    f"mean_steps={format_metric(eval_mean_steps, precision=2)} "
                    f"mean_reward={format_metric(eval_mean_reward)} "
                    f"train_reward={format_metric(reward_mean)} "
                    f"eval_gate_pass={eval_gate_pass} "
                    f"reward_gate_enabled={reward_gate_enabled} "
                    f"reward_gate_pass={reward_gate_pass} "
                    f"streak={early_stop_pass_streak}/{args.early_stop_consecutive_evals}"
                )

                if early_stop_pass_streak >= args.early_stop_consecutive_evals:
                    early_stop_path = checkpoint_root / f"checkpoint_early_stop_iter_{i:05d}"
                    latest_checkpoint = save_algorithm_checkpoint(algo, early_stop_path, local_fs)
                    early_stop_triggered = True
                    early_stop_event = {
                        "iteration": int(i),
                        "timesteps_total": int(timesteps_total),
                        "checkpoint_path": str(latest_checkpoint),
                        "success_rate": float(eval_success_rate),
                        "required_success_rate": float(args.early_stop_success_rate),
                    }
                    print(
                        "early_stop_triggered: "
                        f"iter={i:03d} "
                        f"distance={args.early_stop_distance:.3f} "
                        f"success_rate={format_metric(eval_success_rate)} "
                        f"streak={early_stop_pass_streak}/{args.early_stop_consecutive_evals} "
                        f"checkpoint={latest_checkpoint}"
                    )
                    break

            if not args.curriculum_enabled:
                summary = build_run_summary(
                    run_reports=report_rows_nested,
                    best_checkpoint_record=best_checkpoint_record,
                    highest_stage_reached=highest_stage_reached,
                    first_promotion=first_promotion,
                    stage_six_reached=stage_six_reached,
                    early_stop_event=early_stop_event,
                )
                run_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
                continue

            curriculum_stats = run_fixed_distance_eval(
                algo,
                distance=current_distance,
                time_limit=current_time_limit,
                num_episodes=args.curriculum_eval_episodes,
                base_seed=args.early_stop_eval_seed + 100_000,
            )
            curriculum_success_rate = curriculum_stats["success_rate"]
            curriculum_mean_steps = curriculum_stats["mean_steps"]
            curriculum_mean_reward = curriculum_stats["mean_reward"]
            current_stage_threshold = compute_stage_success_threshold(
                current_distance,
                min_stage_distance=min_stage_distance,
                max_stage_distance=max_stage_distance,
                base_threshold=args.curriculum_success_rate,
            )

            curriculum_metrics_valid = (
                np.isfinite(curriculum_success_rate)
                and np.isfinite(curriculum_mean_steps)
                and np.isfinite(curriculum_mean_reward)
            )
            curriculum_gate_pass = curriculum_metrics_valid and (
                curriculum_success_rate >= current_stage_threshold
            )
            stage_pass_streak = stage_pass_streak + 1 if curriculum_gate_pass else 0

            print(
                "curriculum_eval: "
                f"iter={i:03d} "
                f"stage_idx={stage_index} "
                f"distance={current_distance:.3f} "
                f"time_limit={current_time_limit} "
                f"success_rate={format_metric(curriculum_success_rate)} "
                f"mean_steps={format_metric(curriculum_mean_steps, precision=2)} "
                f"mean_reward={format_metric(curriculum_mean_reward)} "
                f"gate_threshold={format_metric(current_stage_threshold)} "
                f"gate_pass={curriculum_gate_pass} "
                f"streak={stage_pass_streak}/{args.curriculum_consecutive_evals}"
            )

            if stage_pass_streak < args.curriculum_consecutive_evals:
                summary = build_run_summary(
                    run_reports=report_rows_nested,
                    best_checkpoint_record=best_checkpoint_record,
                    highest_stage_reached=highest_stage_reached,
                    first_promotion=first_promotion,
                    stage_six_reached=stage_six_reached,
                    early_stop_event=early_stop_event,
                )
                run_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
                continue
            if stage_index >= len(stage_distances) - 1:
                print("curriculum_status: final_stage_reached_no_further_promotion")
                summary = build_run_summary(
                    run_reports=report_rows_nested,
                    best_checkpoint_record=best_checkpoint_record,
                    highest_stage_reached=highest_stage_reached,
                    first_promotion=first_promotion,
                    stage_six_reached=stage_six_reached,
                    early_stop_event=early_stop_event,
                )
                run_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
                continue

            stage_checkpoint_path = checkpoint_root / f"checkpoint_stage_{stage_index:02d}_iter_{i:05d}"
            latest_checkpoint = save_algorithm_checkpoint(algo, stage_checkpoint_path, local_fs)

            previous_stage = stage_index
            stage_index += 1
            next_distance = float(stage_distances[stage_index])
            next_time_limit = compute_stage_time_limit(
                next_distance,
                min_stage_distance=min_stage_distance,
                max_stage_distance=max_stage_distance,
                base_limit=args.curriculum_time_limit_base,
                max_limit=args.curriculum_time_limit_max,
            )

            print(
                "curriculum_promoted: "
                f"iter={i:03d} "
                f"from_stage={previous_stage} to_stage={stage_index} "
                f"from_distance={float(stage_distances[previous_stage]):.3f} "
                f"to_distance={next_distance:.3f} "
                f"checkpoint={latest_checkpoint}"
            )

            highest_stage_reached = max(highest_stage_reached, int(stage_index))
            promotion_event = {
                "iteration": int(i),
                "timesteps_total": int(timesteps_total),
                "stage_index": int(stage_index),
                "distance": float(next_distance),
            }
            if first_promotion is None:
                first_promotion = promotion_event
            if stage_index >= 6 and stage_six_reached is None:
                stage_six_reached = promotion_event

            restore_checkpoint = latest_checkpoint
            algo.stop()
            algo = build_algo(
                args,
                num_gpus=num_gpus,
                env_config=build_env_config(args, distance=next_distance, time_limit=next_time_limit),
            )
            algo.restore(restore_checkpoint)

            current_distance = next_distance
            current_time_limit = next_time_limit
            stage_pass_streak = 0
            print(
                "curriculum_stage_active: "
                f"stage_idx={stage_index} "
                f"distance={current_distance:.3f} "
                f"time_limit={current_time_limit}"
            )
            summary = build_run_summary(
                run_reports=report_rows_nested,
                best_checkpoint_record=best_checkpoint_record,
                highest_stage_reached=highest_stage_reached,
                first_promotion=first_promotion,
                stage_six_reached=stage_six_reached,
                early_stop_event=early_stop_event,
            )
            run_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    finally:
        if algo is not None:
            final_path = checkpoint_root / "checkpoint_final"
            latest_checkpoint = save_algorithm_checkpoint(algo, final_path, local_fs)
            print(f"final_checkpoint_saved: {latest_checkpoint}")
            if early_stop_triggered:
                print("training_status: stopped_early_on_distance_rule")
            else:
                print("training_status: reached_iteration_budget_or_stopped_without_distance_rule")
            summary = build_run_summary(
                run_reports=report_rows_nested,
                best_checkpoint_record=best_checkpoint_record,
                highest_stage_reached=highest_stage_reached,
                first_promotion=first_promotion,
                stage_six_reached=stage_six_reached,
                early_stop_event=early_stop_event,
            )
            summary["final_checkpoint_path"] = str(latest_checkpoint)
            summary["early_stop_triggered"] = bool(early_stop_triggered)
            run_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
