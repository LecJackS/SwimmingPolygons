"""V4 RLlib PPO checkpoint evaluation entrypoint."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import os
from urllib.parse import urlparse, unquote

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import NoopLogger
import numpy as np
import torch

from triangles import OctopusEnv


ENV_ID = "v4_octopus_env_eval"
DEFAULT_CURRICULUM_STAGES = [0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.7, 8.0, 10.0]

logging.getLogger("ray._common.deprecation").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained V4 RLlib PPO checkpoint.")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=str, default="./rllib_checkpoints")
    parser.add_argument("--max-frames", type=int, default=10_000)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda. Defaults to SB3_DEVICE or auto.")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
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
    return parser.parse_args()


def resolve_device(cli_device: str | None) -> str:
    if cli_device:
        return cli_device
    env_device = os.getenv("SB3_DEVICE")
    if env_device:
        return env_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_stage_distances(raw: str) -> list[float]:
    parts = [chunk.strip() for chunk in raw.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("--curriculum-stages must be a non-empty comma-separated list.")

    distances = [float(part) for part in parts]
    if any(value <= 0 for value in distances):
        raise ValueError("--curriculum-stages values must be > 0.")
    if any(distances[idx] >= distances[idx + 1] for idx in range(len(distances) - 1)):
        raise ValueError("--curriculum-stages must be strictly increasing.")
    return distances


def compute_stage_time_limit(
    distance: float,
    *,
    min_stage_distance: float,
    max_stage_distance: float,
    base_limit: int,
    max_limit: int,
) -> int:
    if max_stage_distance <= min_stage_distance:
        return int(base_limit)
    ratio = (float(distance) - min_stage_distance) / (max_stage_distance - min_stage_distance)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    late_stage_boost = ratio + 0.15 * ratio * (1.0 - ratio)
    return int(round(base_limit + late_stage_boost * (max_limit - base_limit)))


def find_latest_checkpoint(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")
    candidates = [p for p in root.rglob("checkpoint_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint directories found under: {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def uri_to_local_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Unsupported URI scheme for local checkpoint: {uri}")
    path = unquote(parsed.path)
    # On Windows, file:///C:/... becomes /C:/...
    if len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return Path(path)


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
    if args.fixed_food_distance is not None and args.fixed_food_distance <= 0:
        raise ValueError("--fixed-food-distance must be > 0.")
    if args.time_limit is not None and args.time_limit <= 0:
        raise ValueError("--time-limit must be > 0.")
    if args.curriculum_time_limit_base <= 0:
        raise ValueError("--curriculum-time-limit-base must be > 0.")
    if args.curriculum_time_limit_max < args.curriculum_time_limit_base:
        raise ValueError("--curriculum-time-limit-max must be >= --curriculum-time-limit-base.")

    device = resolve_device(args.device)
    num_gpus = 1 if device == "cuda" else 0
    checkpoint_root = Path(args.checkpoint_root)
    use_fixed_distance = args.fixed_food_distance is not None
    stage_distances = parse_stage_distances(args.curriculum_stages)
    min_stage_distance = float(stage_distances[0])
    max_stage_distance = float(stage_distances[-1])

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

    render_mode = None if args.no_render else "human"
    env = OctopusEnv(
        epsilon=args.epsilon,
        render_mode=render_mode,
        enable_curriculum=not use_fixed_distance,
        fixed_food_distance=args.fixed_food_distance,
        time_limit=eval_time_limit,
    )

    register_env(
        ENV_ID,
        lambda config: OctopusEnv(
            epsilon=float(config.get("epsilon", 0.0)),
            render_mode=config.get("render_mode"),
            enable_curriculum=bool(config.get("enable_curriculum", True)),
            fixed_food_distance=config.get("fixed_food_distance"),
            time_limit=int(config.get("time_limit", eval_time_limit)),
        ),
    )
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=True)

    eval_env_config = {
        "epsilon": args.epsilon,
        "render_mode": None,
        "enable_curriculum": not use_fixed_distance,
        "fixed_food_distance": args.fixed_food_distance,
        "time_limit": eval_time_limit,
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

    print("V4 - Accelerated Physics RLlib evaluation")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Device: {device}")
    print(f"Stack mode: {stack_mode}")
    print(f"Render: {not args.no_render}")
    print(f"Time limit: {eval_time_limit}")
    if use_fixed_distance:
        print(f"Food distance: fixed at {args.fixed_food_distance:.3f}")
    else:
        print("Food distance: curriculum-enabled env defaults")

    try:
        obs, info = env.reset()
        module = algo.get_module() if stack_mode == "new" else None
        for i in range(args.max_frames):
            if stack_mode == "new":
                obs_batch = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    out = module.forward_inference({"obs": obs_batch})
                logits = out["action_dist_inputs"].detach().cpu().numpy().reshape(2, 3)
                action = np.argmax(logits, axis=1).astype(np.int64)
            else:
                action = algo.compute_single_action(obs, explore=False)
                if isinstance(action, tuple):
                    action = action[0]
            action = np.asarray(action, dtype=np.int64).reshape(-1)
            if action.size != 2:
                action = np.array([1, 1], dtype=np.int64)
            obs, reward, terminated, truncated, info = env.step(action)
            if not args.no_render:
                env.render()

            if args.log_every > 0 and i % args.log_every == 0:
                print(
                    f"frame={i:05d} action={action} reward={reward:.2f} "
                    f"dist={info.get('distance_to_food', float('nan')):.3f}"
                )

            if terminated or truncated:
                print(
                    f"episode_end frame={i:05d} terminated={terminated} "
                    f"truncated={truncated} reward={reward:.2f}"
                )
                obs, info = env.reset()
    finally:
        env.close()
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
