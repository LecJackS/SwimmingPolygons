"""Performance benchmark harness for V8 color communication schooling."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
import ray
from ray.tune.registry import register_env

from eval_utils import SHARED_POLICY_ID, compute_batched_deterministic_actions, sample_random_action
from test_model import ENV_ID, build_env_config, build_eval_algo, load_checkpoint_path, resolve_device
from triangles import CommunicatingSchoolEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark V8 environment, rendering, and inference throughput.")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=str, default="./rllib_checkpoints_smoke_v8_color_comm")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps-no-render", type=int, default=300)
    parser.add_argument("--steps-render", type=int, default=120)
    parser.add_argument("--summary-json", type=str, default="./perf_benchmark.json")
    parser.add_argument("--summary-csv", type=str, default="./perf_benchmark.csv")
    parser.add_argument("--focus-agent-id", type=str, default="fish_0")
    parser.add_argument("--render-engine", type=str, choices=["auto", "blit", "safe"], default="auto")
    parser.add_argument("--time-limit", type=int, default=600)
    parser.add_argument("--num-red-fish", type=int, default=5)
    parser.add_argument("--num-blue-fish", type=int, default=5)
    parser.add_argument("--num-red-pellets", type=int, default=24)
    parser.add_argument("--num-blue-pellets", type=int, default=24)
    parser.add_argument("--pellet-reward", type=float, default=1.0)
    parser.add_argument("--step-cost", type=float, default=0.002)
    parser.add_argument("--sector-radius", type=float, default=5.0)
    parser.add_argument("--sector-num", type=int, default=6)
    parser.add_argument("--epsilon", type=float, default=0.0)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def make_env_args(args: argparse.Namespace, *, render_profile: str, render_mode: str | None) -> argparse.Namespace:
    return argparse.Namespace(
        epsilon=float(args.epsilon),
        render_profile=str(render_profile),
        render_engine=str(args.render_engine),
        time_limit=int(args.time_limit),
        num_red_fish=int(args.num_red_fish),
        num_blue_fish=int(args.num_blue_fish),
        num_red_pellets=int(args.num_red_pellets),
        num_blue_pellets=int(args.num_blue_pellets),
        pellet_reward=float(args.pellet_reward),
        step_cost=float(args.step_cost),
        sector_radius=float(args.sector_radius),
        sector_num=int(args.sector_num),
        focus_agent_id=str(args.focus_agent_id),
        hide_sensor_overlay=(render_profile != "full"),
        mute_messages=False,
    )


def benchmark_case(
    *,
    name: str,
    steps: int,
    env: CommunicatingSchoolEnv,
    action_fn,
    render_enabled: bool,
    base_seed: int,
) -> dict[str, object]:
    obs_dict, _ = env.reset(seed=base_seed)
    action_time = 0.0
    step_time = 0.0
    render_time = 0.0
    for frame_idx in range(steps):
        t0 = time.perf_counter()
        action_dict = action_fn(obs_dict)
        action_time += time.perf_counter() - t0

        t1 = time.perf_counter()
        obs_dict, _, terminateds, truncateds, _ = env.step(action_dict)
        step_time += time.perf_counter() - t1

        if render_enabled:
            t2 = time.perf_counter()
            env.render()
            render_time += time.perf_counter() - t2

        if terminateds["__all__"] or truncateds["__all__"]:
            obs_dict, _ = env.reset(seed=base_seed + frame_idx + 1)

    total_time = action_time + step_time + render_time
    return {
        "case": name,
        "steps": int(steps),
        "fps": float(steps / max(total_time, 1e-9)),
        "ms_per_frame": float((total_time / max(steps, 1)) * 1000.0),
        "mean_action_ms": float((action_time / max(steps, 1)) * 1000.0),
        "mean_step_ms": float((step_time / max(steps, 1)) * 1000.0),
        "mean_render_ms": float((render_time / max(steps, 1)) * 1000.0),
        "render_enabled": bool(render_enabled),
        "render_profile": str(env.render_profile),
        "render_engine": str(env.render_engine),
        "backend": str(env.fig.canvas.__class__.__name__) if env.fig is not None else None,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_root = Path(args.checkpoint_root)
    checkpoint_path = None
    restore_target = None
    if Path(args.checkpoint_root).exists() or args.checkpoint_path:
        checkpoint_path, restore_target = load_checkpoint_path(args, checkpoint_root)

    results: list[dict[str, object]] = []
    rng = np.random.default_rng(args.seed)

    env_args_fast = make_env_args(args, render_profile="fast", render_mode="human")
    env_args_full = make_env_args(args, render_profile="full", render_mode="human")
    env_args_headless = make_env_args(args, render_profile="fast", render_mode=None)

    env_only = CommunicatingSchoolEnv(**build_env_config(env_args_headless, render_mode=None, show_sensor_overlay=False, mute_received_messages=False))
    try:
        results.append(
            benchmark_case(
                name="env_only_random",
                steps=args.steps_no_render,
                env=env_only,
                action_fn=lambda obs_dict: {agent_id: sample_random_action(rng) for agent_id in obs_dict.keys()},
                render_enabled=False,
                base_seed=args.seed,
            )
        )
    finally:
        env_only.close()

    random_fast = CommunicatingSchoolEnv(**build_env_config(env_args_fast, render_mode="human", show_sensor_overlay=False, mute_received_messages=False))
    try:
        results.append(
            benchmark_case(
                name="random_fast_render",
                steps=args.steps_render,
                env=random_fast,
                action_fn=lambda obs_dict: {agent_id: sample_random_action(rng) for agent_id in obs_dict.keys()},
                render_enabled=True,
                base_seed=args.seed + 1_000,
            )
        )
    finally:
        random_fast.close()

    random_full = CommunicatingSchoolEnv(**build_env_config(env_args_full, render_mode="human", show_sensor_overlay=True, mute_received_messages=False))
    try:
        results.append(
            benchmark_case(
                name="random_full_render",
                steps=args.steps_render,
                env=random_full,
                action_fn=lambda obs_dict: {agent_id: sample_random_action(rng) for agent_id in obs_dict.keys()},
                render_enabled=True,
                base_seed=args.seed + 2_000,
            )
        )
    finally:
        random_full.close()

    if checkpoint_path is not None:
        register_env(ENV_ID, lambda config: CommunicatingSchoolEnv(**config))
        os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
        algo = None
        stack_mode = "old"
        try:
            eval_env_config = build_env_config(env_args_fast, render_mode=None, show_sensor_overlay=False, mute_received_messages=False)
            algo = build_eval_algo(env_id=ENV_ID, env_config=eval_env_config, num_gpus=1 if device == "cuda" else 0, seed=args.seed, use_old_stack=True)
            try:
                algo.restore(restore_target)
            except Exception:
                algo.stop()
                algo = build_eval_algo(env_id=ENV_ID, env_config=eval_env_config, num_gpus=1 if device == "cuda" else 0, seed=args.seed, use_old_stack=False)
                algo.restore(restore_target)
                stack_mode = "new"

            trained_headless = CommunicatingSchoolEnv(**build_env_config(env_args_headless, render_mode=None, show_sensor_overlay=False, mute_received_messages=False))
            try:
                results.append(
                    benchmark_case(
                        name="trained_no_render",
                        steps=args.steps_no_render,
                        env=trained_headless,
                        action_fn=lambda obs_dict: compute_batched_deterministic_actions(
                            algo,
                            obs_dict,
                            stack_mode=stack_mode,
                            policy_id=SHARED_POLICY_ID,
                        ),
                        render_enabled=False,
                        base_seed=args.seed + 3_000,
                    )
                )
            finally:
                trained_headless.close()

            trained_fast = CommunicatingSchoolEnv(**build_env_config(env_args_fast, render_mode="human", show_sensor_overlay=False, mute_received_messages=False))
            try:
                results.append(
                    benchmark_case(
                        name="trained_fast_render",
                        steps=args.steps_render,
                        env=trained_fast,
                        action_fn=lambda obs_dict: compute_batched_deterministic_actions(
                            algo,
                            obs_dict,
                            stack_mode=stack_mode,
                            policy_id=SHARED_POLICY_ID,
                        ),
                        render_enabled=True,
                        base_seed=args.seed + 4_000,
                    )
                )
            finally:
                trained_fast.close()

            trained_full = CommunicatingSchoolEnv(**build_env_config(env_args_full, render_mode="human", show_sensor_overlay=True, mute_received_messages=False))
            try:
                results.append(
                    benchmark_case(
                        name="trained_full_render",
                        steps=args.steps_render,
                        env=trained_full,
                        action_fn=lambda obs_dict: compute_batched_deterministic_actions(
                            algo,
                            obs_dict,
                            stack_mode=stack_mode,
                            policy_id=SHARED_POLICY_ID,
                        ),
                        render_enabled=True,
                        base_seed=args.seed + 5_000,
                    )
                )
            finally:
                trained_full.close()
        finally:
            if algo is not None:
                algo.stop()
            ray.shutdown()

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path is not None else None,
        "device": str(device),
        "results": results,
    }

    summary_json_path = Path(args.summary_json)
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    summary_csv_path = Path(args.summary_csv)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(summary_csv_path, results)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
