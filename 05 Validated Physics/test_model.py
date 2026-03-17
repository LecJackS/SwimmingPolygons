"""V5 RLlib PPO checkpoint evaluation entrypoint."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import NoopLogger
from ray.tune.registry import register_env
import torch

from eval_utils import (
    DEFAULT_CURRICULUM_STAGES,
    DistanceEvalResult,
    compute_deterministic_action,
    compute_stage_time_limit,
    evaluate_env_rollouts,
    find_latest_checkpoint,
    flatten_distance_results,
    parse_stage_distances,
    uri_to_local_path,
    weighted_success_score,
)
from triangles import OctopusEnv


ENV_ID = "v5_octopus_env_eval"

logging.getLogger("ray._common.deprecation").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained V5 RLlib PPO checkpoint.")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=str, default="./rllib_checkpoints")
    parser.add_argument("--max-frames", type=int, default=10_000)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda. Defaults to SB3_DEVICE or auto.")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per distance in batch scoring mode.")
    parser.add_argument("--time-limit", type=int, default=None)
    parser.add_argument(
        "--curriculum-stages",
        type=str,
        default=",".join(str(v) for v in DEFAULT_CURRICULUM_STAGES),
        help="Comma-separated stage distances used to derive fixed-distance eval time limits.",
    )
    parser.add_argument("--curriculum-time-limit-base", type=int, default=100)
    parser.add_argument("--curriculum-time-limit-max", type=int, default=180)
    parser.add_argument(
        "--fixed-food-distance",
        type=float,
        default=None,
        help="Use a fixed target spawn distance (disables curriculum during evaluation).",
    )
    parser.add_argument(
        "--distances",
        type=str,
        default=None,
        help="Comma-separated distances to score in batch mode. Defaults to fixed distance or curriculum stages.",
    )
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument("--summary-csv", type=str, default=None)
    return parser.parse_args()


def resolve_device(cli_device: str | None) -> str:
    if cli_device:
        return cli_device
    env_device = os.getenv("SB3_DEVICE")
    if env_device:
        return env_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_eval_algo(
    *,
    env_id: str,
    env_config: dict,
    num_gpus: int,
    seed: int,
    use_old_stack: bool,
):
    config = (
        PPOConfig()
        .environment(env=env_id, env_config=env_config)
        .framework("torch")
        .resources(num_gpus=num_gpus)
        .env_runners(num_env_runners=0)
        .debugging(seed=seed, logger_creator=lambda cfg: NoopLogger(cfg, "."))
    )
    if use_old_stack:
        config = config.api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        ).fault_tolerance(
            restart_failed_env_runners=False,
            max_num_env_runner_restarts=0,
        )
    return config.build_algo()


def resolve_eval_distances(args: argparse.Namespace, stage_distances: list[float]) -> list[float]:
    explicit_batch_mode = bool(args.distances or args.summary_json or args.summary_csv or args.episodes > 1)
    if not explicit_batch_mode:
        return []
    if args.distances:
        return parse_stage_distances(args.distances)
    if args.fixed_food_distance is not None:
        return [float(args.fixed_food_distance)]
    return stage_distances


def write_summary_csv(path: Path, row: dict[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def load_checkpoint_path(args: argparse.Namespace, checkpoint_root: Path) -> tuple[Path, str]:
    if args.checkpoint_path:
        if args.checkpoint_path.startswith("file://"):
            checkpoint_path = uri_to_local_path(args.checkpoint_path)
            restore_target = str(checkpoint_path.resolve())
        else:
            checkpoint_path = Path(args.checkpoint_path)
            restore_target = str(checkpoint_path.resolve())
    else:
        checkpoint_path = find_latest_checkpoint(checkpoint_root)
        restore_target = str(checkpoint_path.resolve())

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path, restore_target


def main() -> None:
    args = parse_args()
    if args.fixed_food_distance is not None and args.fixed_food_distance <= 0:
        raise ValueError("--fixed-food-distance must be > 0.")
    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0.")
    if args.time_limit is not None and args.time_limit <= 0:
        raise ValueError("--time-limit must be > 0.")
    if args.curriculum_time_limit_base <= 0:
        raise ValueError("--curriculum-time-limit-base must be > 0.")
    if args.curriculum_time_limit_max < args.curriculum_time_limit_base:
        raise ValueError("--curriculum-time-limit-max must be >= --curriculum-time-limit-base.")

    device = resolve_device(args.device)
    num_gpus = 1 if device == "cuda" else 0
    checkpoint_root = Path(args.checkpoint_root)
    stage_distances = parse_stage_distances(args.curriculum_stages)
    min_stage_distance = float(stage_distances[0])
    max_stage_distance = float(stage_distances[-1])
    batch_distances = resolve_eval_distances(args, stage_distances)
    batch_mode = bool(batch_distances)

    checkpoint_path, restore_target = load_checkpoint_path(args, checkpoint_root)

    register_env(
        ENV_ID,
        lambda config: OctopusEnv(
            epsilon=float(config.get("epsilon", 0.0)),
            render_mode=config.get("render_mode"),
            enable_curriculum=bool(config.get("enable_curriculum", True)),
            fixed_food_distance=config.get("fixed_food_distance"),
            time_limit=int(config.get("time_limit", args.curriculum_time_limit_base)),
        ),
    )
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=True)

    eval_env_config = {
        "epsilon": args.epsilon,
        "render_mode": None,
        "enable_curriculum": args.fixed_food_distance is None,
        "fixed_food_distance": args.fixed_food_distance,
        "time_limit": int(args.time_limit or args.curriculum_time_limit_base),
    }
    algo = build_eval_algo(
        env_id=ENV_ID,
        env_config=eval_env_config,
        num_gpus=num_gpus,
        seed=args.seed,
        use_old_stack=True,
    )
    stack_mode = "old"
    try:
        algo.restore(restore_target)
    except Exception:
        algo.stop()
        algo = build_eval_algo(
            env_id=ENV_ID,
            env_config=eval_env_config,
            num_gpus=num_gpus,
            seed=args.seed,
            use_old_stack=False,
        )
        algo.restore(restore_target)
        stack_mode = "new"

    print("V5 - Validated Physics RLlib evaluation")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Device: {device}")
    print(f"Stack mode: {stack_mode}")

    try:
        if batch_mode:
            results: list[DistanceEvalResult] = []
            for distance_index, distance in enumerate(batch_distances):
                time_limit = int(args.time_limit) if args.time_limit is not None else compute_stage_time_limit(
                    distance,
                    min_stage_distance=min_stage_distance,
                    max_stage_distance=max_stage_distance,
                    base_limit=args.curriculum_time_limit_base,
                    max_limit=args.curriculum_time_limit_max,
                )
                metrics = evaluate_env_rollouts(
                    algo=algo,
                    env_factory=lambda d=distance, t=time_limit: OctopusEnv(
                        epsilon=args.epsilon,
                        render_mode=None,
                        enable_curriculum=False,
                        fixed_food_distance=d,
                        time_limit=t,
                    ),
                    num_episodes=args.episodes,
                    base_seed=args.seed + (distance_index * 10_000),
                    stack_mode=stack_mode,
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

            weighted_score = weighted_success_score(results)
            summary = {
                "checkpoint_path": str(checkpoint_path.resolve()),
                "device": device,
                "stack_mode": stack_mode,
                "episodes_per_distance": int(args.episodes),
                "weighted_success_score": float(weighted_score),
                "distance_results": {f"{result.distance:g}": result.to_dict() for result in results},
            }
            flat_summary = {
                "checkpoint_path": str(checkpoint_path.resolve()),
                "device": device,
                "stack_mode": stack_mode,
                "episodes_per_distance": int(args.episodes),
                "weighted_success_score": float(weighted_score),
                **flatten_distance_results(results),
            }

            print("Mode: batch deterministic scoring")
            print(f"Render: False")
            print(f"Episodes per distance: {args.episodes}")
            print(f"Weighted success score: {weighted_score:.3f}")
            for result in results:
                print(
                    f"distance={result.distance:.3f} time_limit={result.time_limit} "
                    f"success_rate={result.success_rate:.3f} "
                    f"mean_steps={result.mean_steps:.2f} "
                    f"mean_reward={result.mean_reward:.3f}"
                )

            if args.summary_json:
                summary_json_path = Path(args.summary_json)
                summary_json_path.parent.mkdir(parents=True, exist_ok=True)
                summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
                print(f"summary_json_saved: {summary_json_path.resolve()}")
            if args.summary_csv:
                summary_csv_path = Path(args.summary_csv)
                summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
                write_summary_csv(summary_csv_path, flat_summary)
                print(f"summary_csv_saved: {summary_csv_path.resolve()}")
            return

        use_fixed_distance = args.fixed_food_distance is not None
        if args.time_limit is not None:
            eval_time_limit = int(args.time_limit)
        elif use_fixed_distance:
            eval_time_limit = compute_stage_time_limit(
                args.fixed_food_distance,
                min_stage_distance=min_stage_distance,
                max_stage_distance=max_stage_distance,
                base_limit=args.curriculum_time_limit_base,
                max_limit=args.curriculum_time_limit_max,
            )
        else:
            eval_time_limit = args.curriculum_time_limit_base

        render_mode = None if args.no_render else "human"
        env = OctopusEnv(
            epsilon=args.epsilon,
            render_mode=render_mode,
            enable_curriculum=not use_fixed_distance,
            fixed_food_distance=args.fixed_food_distance,
            time_limit=eval_time_limit,
        )

        print("Mode: single rollout")
        print(f"Render: {not args.no_render}")
        print(f"Time limit: {eval_time_limit}")
        if use_fixed_distance:
            print(f"Food distance: fixed at {args.fixed_food_distance:.3f}")
        else:
            print("Food distance: curriculum-enabled env defaults")

        try:
            obs, info = env.reset(seed=args.seed)
            for frame_idx in range(args.max_frames):
                action = compute_deterministic_action(algo, obs, stack_mode=stack_mode)
                obs, reward, terminated, truncated, info = env.step(action)
                if not args.no_render:
                    env.render()

                if args.log_every > 0 and frame_idx % args.log_every == 0:
                    print(
                        f"frame={frame_idx:05d} action={action} reward={reward:.2f} "
                        f"dist={info.get('distance_to_food', float('nan')):.3f}"
                    )

                if terminated or truncated:
                    print(
                        f"episode_end frame={frame_idx:05d} terminated={terminated} "
                        f"truncated={truncated} reward={reward:.2f}"
                    )
                    obs, info = env.reset()
        finally:
            env.close()
    finally:
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
