"""V7 RLlib PPO checkpoint evaluation entrypoint for shared-policy schooling."""

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
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import NoopLogger
from ray.tune.registry import register_env
import torch

from eval_utils import (
    DEFAULT_FOOD_COUNT,
    DEFAULT_NUM_FISH,
    DEFAULT_PELLET_REWARD,
    DEFAULT_SENSOR_NUM_SECTORS,
    DEFAULT_SENSOR_RADIUS,
    DEFAULT_SENSOR_RING_EDGES,
    DEFAULT_STEP_COST,
    DEFAULT_TIME_LIMIT,
    SHARED_POLICY_ID,
    compute_deterministic_action,
    evaluate_multi_agent_rollouts,
    evaluate_multi_agent_random_rollouts,
    find_latest_checkpoint,
    flatten_team_result,
    parse_ring_edges,
    uri_to_local_path,
)
from triangles import SchoolingFishEnv


ENV_ID = "v7_shared_policy_school_env_eval"

logging.getLogger("ray._common.deprecation").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained V7 shared-policy checkpoint.")
    parser.add_argument("--policy-mode", type=str, choices=["trained", "random"], default="trained")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=str, default="./rllib_checkpoints_baseline_v7_school")
    parser.add_argument("--max-frames", type=int, default=10_000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--hide-sensor-overlay", action="store_true")
    parser.add_argument("--focus-agent-id", type=str, default="fish_0")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--num-fish", type=int, default=DEFAULT_NUM_FISH)
    parser.add_argument("--time-limit", type=int, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--food-count", type=int, default=DEFAULT_FOOD_COUNT)
    parser.add_argument("--pellet-reward", type=float, default=DEFAULT_PELLET_REWARD)
    parser.add_argument("--step-cost", type=float, default=DEFAULT_STEP_COST)
    parser.add_argument("--sensor-radius", type=float, default=DEFAULT_SENSOR_RADIUS)
    parser.add_argument(
        "--sensor-ring-edges",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_SENSOR_RING_EDGES),
        help="Comma-separated radial edges for local food/peer sensors.",
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
    if args.num_fish <= 0:
        raise ValueError("--num-fish must be > 0.")
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
        "num_fish": int(args.num_fish),
        "focus_agent_id": str(args.focus_agent_id),
    }


def build_eval_algo(*, env_id: str, env_config: dict, num_gpus: int, seed: int, use_old_stack: bool):
    sample_env = SchoolingFishEnv(**env_config)
    try:
        obs_space = sample_env.observation_space
        action_space = sample_env.action_space
    finally:
        sample_env.close()
    config = (
        PPOConfig()
        .environment(env=env_id, env_config=env_config)
        .framework("torch")
        .resources(num_gpus=num_gpus)
        .env_runners(num_env_runners=0)
        .multi_agent(
            policies={
                SHARED_POLICY_ID: PolicySpec(
                    observation_space=obs_space,
                    action_space=action_space,
                    config={},
                )
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: SHARED_POLICY_ID,
            policies_to_train=[SHARED_POLICY_ID],
        )
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
    checkpoint_path = None
    restore_target = None
    if args.policy_mode == "trained":
        checkpoint_path, restore_target = load_checkpoint_path(args, checkpoint_root)
    explicit_batch_mode = bool(args.summary_json or args.summary_csv or args.episodes > 1 or args.no_render or args.policy_mode == "random")

    register_env(ENV_ID, lambda config: SchoolingFishEnv(**config))
    eval_env_config = build_env_config(args, render_mode=None, show_sensor_overlay=not args.hide_sensor_overlay)
    algo = None
    stack_mode = None

    if args.policy_mode == "trained":
        os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=True)
        algo = build_eval_algo(env_id=ENV_ID, env_config=eval_env_config, num_gpus=num_gpus, seed=args.seed, use_old_stack=True)
        stack_mode = "old"
        try:
            algo.restore(restore_target)
        except Exception:
            algo.stop()
            algo = build_eval_algo(env_id=ENV_ID, env_config=eval_env_config, num_gpus=num_gpus, seed=args.seed, use_old_stack=False)
            algo.restore(restore_target)
            stack_mode = "new"

    checkpoint_text = str(checkpoint_path.resolve()) if checkpoint_path is not None else None
    print("V7 - Shared Policy School RLlib evaluation")
    print(f"Policy mode: {args.policy_mode}")
    print(f"Checkpoint: {checkpoint_text or 'random_policy'}")
    print(f"Device: {device if args.policy_mode == 'trained' else 'n/a'}")
    if stack_mode is not None:
        print(f"Stack mode: {stack_mode}")

    try:
        if explicit_batch_mode:
            if args.policy_mode == "trained":
                result = evaluate_multi_agent_rollouts(
                    algo=algo,
                    env_factory=lambda: SchoolingFishEnv(**build_env_config(args, render_mode=None, show_sensor_overlay=False)),
                    num_episodes=args.episodes,
                    base_seed=args.seed,
                    stack_mode=stack_mode,
                    policy_id=SHARED_POLICY_ID,
                )
            else:
                result = evaluate_multi_agent_random_rollouts(
                    env_factory=lambda: SchoolingFishEnv(**build_env_config(args, render_mode=None, show_sensor_overlay=False)),
                    num_episodes=args.episodes,
                    base_seed=args.seed,
                )
            summary = {
                "checkpoint_path": checkpoint_text,
                "policy_mode": args.policy_mode,
                "device": device if args.policy_mode == "trained" else None,
                "stack_mode": stack_mode,
                **result.to_dict(),
            }
            flat_summary = {
                "checkpoint_path": checkpoint_text,
                "policy_mode": args.policy_mode,
                "device": device if args.policy_mode == "trained" else None,
                "stack_mode": stack_mode,
                **flatten_team_result(result),
            }
            print("Mode: batch deterministic scoring" if args.policy_mode == "trained" else "Mode: batch random-policy scoring")
            print(f"Episodes: {args.episodes}")
            print(f"mean_team_food_eaten={result.mean_team_food_eaten:.3f}")
            print(f"team_food_per_100_steps={result.team_food_per_100_steps:.3f}")
            print(f"mean_team_reward={result.mean_team_reward:.3f}")
            print(f"mean_food_eaten_per_fish={result.mean_food_eaten_per_fish:.3f}")
            if args.summary_json:
                summary_json_path = Path(args.summary_json)
                summary_json_path.parent.mkdir(parents=True, exist_ok=True)
                summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            if args.summary_csv:
                summary_csv_path = Path(args.summary_csv)
                summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
                write_summary_csv(summary_csv_path, flat_summary)
            return

        render_mode = None if args.no_render else "human"
        env = SchoolingFishEnv(**build_env_config(args, render_mode=render_mode, show_sensor_overlay=not args.hide_sensor_overlay))
        print(f"Render: {not args.no_render}")
        print(f"Sensor overlay: {not args.hide_sensor_overlay}")
        print(f"Focus agent: {args.focus_agent_id}")
        try:
            obs_dict, _ = env.reset(seed=args.seed)
            for frame_idx in range(1, args.max_frames + 1):
                if args.policy_mode == "trained":
                    action_dict = {
                        agent_id: compute_deterministic_action(
                            algo,
                            obs,
                            stack_mode=stack_mode,
                            policy_id=SHARED_POLICY_ID,
                        )
                        for agent_id, obs in obs_dict.items()
                    }
                else:
                    action_dict = {
                        agent_id: env.action_space.sample().astype(np.float32)
                        for agent_id in obs_dict.keys()
                    }
                obs_dict, rewards, terminateds, truncateds, infos = env.step(action_dict)
                if not args.no_render:
                    env.render()
                focus_info = infos[args.focus_agent_id]
                if frame_idx % args.log_every == 0 or focus_info.get("team_food_eaten_this_step", 0):
                    print(
                        f"frame={frame_idx:05d} team_reward={rewards[args.focus_agent_id]:.3f} "
                        f"team_food_step={focus_info.get('team_food_eaten_this_step', 0)} "
                        f"team_food_episode={focus_info.get('team_food_eaten_episode', 0)} "
                        f"focus_visible_food={focus_info.get('visible_food_count', 0)} "
                        f"focus_visible_peers={focus_info.get('visible_peer_count', 0)} "
                        f"focus_nearest={focus_info.get('nearest_food_distance', float('nan')):.3f}"
                    )
                if terminateds["__all__"] or truncateds["__all__"]:
                    print(
                        f"episode_end frame={frame_idx:05d} team_food_episode={focus_info.get('team_food_eaten_episode', 0)} "
                        f"team_reward={rewards[args.focus_agent_id]:.3f}"
                    )
                    obs_dict, _ = env.reset(seed=args.seed + frame_idx)
        finally:
            env.close()
    finally:
        if algo is not None:
            algo.stop()
            ray.shutdown()


if __name__ == "__main__":
    main()
