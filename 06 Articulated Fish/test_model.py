"""V6 RLlib PPO checkpoint evaluation entrypoint for continuous foraging."""

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
    DEFAULT_FOOD_COUNT,
    DEFAULT_PELLET_REWARD,
    DEFAULT_SENSOR_NUM_SECTORS,
    DEFAULT_SENSOR_RADIUS,
    DEFAULT_SENSOR_RING_EDGES,
    DEFAULT_STEP_COST,
    DEFAULT_TIME_LIMIT,
    ForagingEvalResult,
    compute_deterministic_action,
    evaluate_env_rollouts,
    find_latest_checkpoint,
    flatten_foraging_result,
    parse_ring_edges,
    uri_to_local_path,
)
from triangles import OctopusEnv


ENV_ID = "v6_articulated_fish_foraging_env_eval"

logging.getLogger("ray._common.deprecation").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained V6 RLlib PPO checkpoint.")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=str, default="./rllib_checkpoints_baseline_v6_foraging")
    parser.add_argument("--max-frames", type=int, default=10_000)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda. Defaults to SB3_DEVICE or auto.")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--hide-sensor-overlay", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1, help="Episodes in batch scoring mode.")
    parser.add_argument("--time-limit", type=int, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--food-count", type=int, default=DEFAULT_FOOD_COUNT)
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


def validate_args(args: argparse.Namespace, sensor_ring_edges: list[float]) -> None:
    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0.")
    if args.time_limit <= 0:
        raise ValueError("--time-limit must be > 0.")
    if args.food_count <= 0:
        raise ValueError("--food-count must be > 0.")
    if args.pellet_reward <= 0.0:
        raise ValueError("--pellet-reward must be > 0.")
    if args.step_cost < 0.0:
        raise ValueError("--step-cost must be >= 0.")
    if args.sensor_radius <= 0.0:
        raise ValueError("--sensor-radius must be > 0.")
    if args.sensor_sectors <= 0:
        raise ValueError("--sensor-sectors must be > 0.")
    if sensor_ring_edges[-1] > args.sensor_radius + 1e-6:
        raise ValueError("--sensor-ring-edges must lie within --sensor-radius.")


def build_env_config(args: argparse.Namespace, *, render_mode: str | None, show_sensor_overlay: bool) -> dict:
    return {
        "epsilon": float(args.epsilon),
        "render_mode": render_mode,
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


def main() -> None:
    args = parse_args()
    sensor_ring_edges = parse_ring_edges(args.sensor_ring_edges)
    validate_args(args, sensor_ring_edges)

    device = resolve_device(args.device)
    num_gpus = 1 if device == "cuda" else 0
    checkpoint_root = Path(args.checkpoint_root)
    checkpoint_path, restore_target = load_checkpoint_path(args, checkpoint_root)
    explicit_batch_mode = bool(args.summary_json or args.summary_csv or args.episodes > 1 or args.no_render)

    register_env(
        ENV_ID,
        lambda config: OctopusEnv(
            epsilon=float(config.get("epsilon", 0.0)),
            render_mode=config.get("render_mode"),
            time_limit=int(config.get("time_limit", DEFAULT_TIME_LIMIT)),
            food_count=int(config.get("food_count", DEFAULT_FOOD_COUNT)),
            food_capture_radius=float(config.get("food_capture_radius", 0.45)),
            pellet_reward=float(config.get("pellet_reward", DEFAULT_PELLET_REWARD)),
            step_cost=float(config.get("step_cost", DEFAULT_STEP_COST)),
            sensor_radius=float(config.get("sensor_radius", DEFAULT_SENSOR_RADIUS)),
            sensor_ring_edges=config.get("sensor_ring_edges", DEFAULT_SENSOR_RING_EDGES),
            sensor_num_sectors=int(config.get("sensor_num_sectors", DEFAULT_SENSOR_NUM_SECTORS)),
            show_sensor_overlay=bool(config.get("show_sensor_overlay", True)),
        ),
    )
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=True)

    eval_env_config = build_env_config(
        args,
        render_mode=None,
        show_sensor_overlay=not args.hide_sensor_overlay,
    )
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

    print("V6 - Articulated Fish RLlib evaluation (continuous foraging)")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Device: {device}")
    print(f"Stack mode: {stack_mode}")

    try:
        if explicit_batch_mode:
            result = evaluate_env_rollouts(
                algo=algo,
                env_factory=lambda: OctopusEnv(**build_env_config(args, render_mode=None, show_sensor_overlay=False)),
                num_episodes=args.episodes,
                base_seed=args.seed,
                stack_mode=stack_mode,
            )
            summary = {
                "checkpoint_path": str(checkpoint_path.resolve()),
                "device": device,
                "stack_mode": stack_mode,
                **result.to_dict(),
            }
            flat_summary = {
                "checkpoint_path": str(checkpoint_path.resolve()),
                "device": device,
                "stack_mode": stack_mode,
                **flatten_foraging_result(result),
            }

            print("Mode: batch deterministic scoring")
            print("Render: False")
            print(f"Episodes: {args.episodes}")
            print(f"mean_food_eaten={result.mean_food_eaten:.3f}")
            print(f"food_per_100_steps={result.food_per_100_steps:.3f}")
            print(f"mean_reward={result.mean_reward:.3f}")
            print(f"mean_visible_food_count={result.mean_visible_food_count:.3f}")

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

        render_mode = None if args.no_render else "human"
        env = OctopusEnv(
            **build_env_config(
                args,
                render_mode=render_mode,
                show_sensor_overlay=not args.hide_sensor_overlay,
            )
        )
        print(f"Render: {not args.no_render}")
        print(f"Sensor overlay: {not args.hide_sensor_overlay}")
        print(f"Food count: {args.food_count}")

        try:
            obs, _ = env.reset(seed=args.seed)
            for frame_idx in range(1, args.max_frames + 1):
                action = compute_deterministic_action(algo, obs, stack_mode=stack_mode)
                obs, reward, terminated, truncated, info = env.step(action)
                if not args.no_render:
                    env.render()
                if frame_idx % args.log_every == 0 or info.get("food_eaten_this_step", 0):
                    print(
                        f"frame={frame_idx:05d} reward={reward:.3f} "
                        f"food_step={info.get('food_eaten_this_step', 0)} "
                        f"food_episode={info.get('food_eaten_episode', 0)} "
                        f"nearest={info.get('nearest_food_distance', float('nan')):.3f} "
                        f"visible={info.get('visible_food_count', 0)}"
                    )
                if terminated or truncated:
                    print(
                        f"episode_end frame={frame_idx:05d} terminated={terminated} truncated={truncated} "
                        f"food_episode={info.get('food_eaten_episode', 0)} reward={info.get('last_reward', float('nan')):.3f}"
                    )
                    obs, _ = env.reset(seed=args.seed + frame_idx)
        finally:
            env.close()
    finally:
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
