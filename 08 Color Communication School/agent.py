"""V8 RLlib PPO training entrypoint for color communication schooling."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable
import warnings

import numpy as np
import pyarrow.fs as pafs
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import NoopLogger
from ray.tune.registry import register_env
import torch

from eval_utils import (
    DEFAULT_NUM_BLUE_FISH,
    DEFAULT_NUM_BLUE_PELLETS,
    DEFAULT_NUM_RED_FISH,
    DEFAULT_NUM_RED_PELLETS,
    DEFAULT_PELLET_REWARD,
    DEFAULT_SECTOR_NUM,
    DEFAULT_SECTOR_RADIUS,
    DEFAULT_STEP_COST,
    DEFAULT_TIME_LIMIT,
    SHARED_POLICY_ID,
    ColorCommEvalResult,
    compare_results,
    evaluate_multi_agent_rollouts,
    flatten_result,
    uri_to_local_path,
)
from triangles import CommunicatingSchoolEnv

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ENV_ID = "v8_color_communication_school_env"

warnings.filterwarnings(
    "ignore",
    message=r".*multi_gpu_train_one_step.*deprecated.*",
    category=DeprecationWarning,
)
logging.getLogger("ray.rllib.execution.train_ops").setLevel(logging.ERROR)
logging.getLogger("ray.rllib.utils.sgd").setLevel(logging.ERROR)
logging.getLogger("ray._common.deprecation").setLevel(logging.ERROR)


DEFAULT_TRAIN_BATCH_SIZE = 8000
DEFAULT_MINIBATCH_SIZE = 1024
DEFAULT_NUM_EPOCHS = 6
DEFAULT_LIGHT_EVAL_EPISODES = 2
DEFAULT_ROLLOUT_FRAGMENT_LENGTH = 250
SAMPLE_TIMEOUT_S = 180.0
COUNT_STEPS_BY = "agent_steps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V8 color communication schooling fish.")
    parser.add_argument("--train-iterations", type=int, default=200)
    parser.add_argument("--num-env-runners", type=int, default=8)
    parser.add_argument("--num-envs-per-runner", type=int, default=2)
    parser.add_argument("--checkpoint-every-iterations", type=int, default=20)
    parser.add_argument("--checkpoint-root", type=str, default="./rllib_checkpoints_baseline_v8_color_comm")
    parser.add_argument("--restore-from-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--num-red-fish", type=int, default=DEFAULT_NUM_RED_FISH)
    parser.add_argument("--num-blue-fish", type=int, default=DEFAULT_NUM_BLUE_FISH)
    parser.add_argument("--num-red-pellets", type=int, default=DEFAULT_NUM_RED_PELLETS)
    parser.add_argument("--num-blue-pellets", type=int, default=DEFAULT_NUM_BLUE_PELLETS)
    parser.add_argument("--time-limit", type=int, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--pellet-reward", type=float, default=DEFAULT_PELLET_REWARD)
    parser.add_argument("--step-cost", type=float, default=DEFAULT_STEP_COST)
    parser.add_argument("--sector-radius", type=float, default=DEFAULT_SECTOR_RADIUS)
    parser.add_argument("--sector-num", type=int, default=DEFAULT_SECTOR_NUM)
    parser.add_argument("--light-eval-episodes", type=int, default=DEFAULT_LIGHT_EVAL_EPISODES)
    parser.add_argument("--eval-report-episodes", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--eval-report-seed", type=int, default=20_240)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--minibatch-size", type=int, default=DEFAULT_MINIBATCH_SIZE)
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--rollout-fragment-length", type=int, default=DEFAULT_ROLLOUT_FRAGMENT_LENGTH)
    parser.add_argument("--fcnet-hiddens", type=str, default="256,256")
    parser.add_argument("--fcnet-activation", type=str, default="tanh")
    return parser.parse_args()


def resolve_device(cli_device: str | None) -> str:
    if cli_device:
        return cli_device
    env_device = os.getenv("SB3_DEVICE")
    if env_device:
        return env_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_args(args: argparse.Namespace) -> None:
    if args.train_iterations <= 0:
        raise ValueError("--train-iterations must be > 0.")
    if args.num_env_runners <= 0:
        raise ValueError("--num-env-runners must be > 0.")
    if args.num_envs_per_runner <= 0:
        raise ValueError("--num-envs-per-runner must be > 0.")
    if args.checkpoint_every_iterations <= 0:
        raise ValueError("--checkpoint-every-iterations must be > 0.")
    if args.num_red_fish <= 0 or args.num_blue_fish <= 0:
        raise ValueError("--num-red-fish and --num-blue-fish must be > 0.")
    if args.num_red_pellets <= 0 or args.num_blue_pellets <= 0:
        raise ValueError("--num-red-pellets and --num-blue-pellets must be > 0.")
    if args.time_limit <= 0:
        raise ValueError("--time-limit must be > 0.")
    if args.pellet_reward <= 0.0:
        raise ValueError("--pellet-reward must be > 0.")
    if args.step_cost < 0.0:
        raise ValueError("--step-cost must be >= 0.")
    if args.sector_radius <= 0.0:
        raise ValueError("--sector-radius must be > 0.")
    if args.sector_num != DEFAULT_SECTOR_NUM:
        raise ValueError(f"--sector-num must remain {DEFAULT_SECTOR_NUM} in V8.")
    if args.light_eval_episodes <= 0:
        raise ValueError("--light-eval-episodes must be > 0.")
    if not (0.0 < args.gae_lambda <= 1.0):
        raise ValueError("--gae-lambda must be in (0, 1].")
    if not (0.0 < args.gamma <= 1.0):
        raise ValueError("--gamma must be in (0, 1].")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be > 0.")
    if args.entropy_coeff < 0.0:
        raise ValueError("--entropy-coeff must be >= 0.")
    if args.train_batch_size <= 0:
        raise ValueError("--train-batch-size must be > 0.")
    if args.minibatch_size <= 0:
        raise ValueError("--minibatch-size must be > 0.")
    if args.num_epochs <= 0:
        raise ValueError("--num-epochs must be > 0.")
    if args.rollout_fragment_length <= 0:
        raise ValueError("--rollout-fragment-length must be > 0.")
    if args.minibatch_size > args.train_batch_size:
        raise ValueError("--minibatch-size must be <= --train-batch-size.")
    if not isinstance(args.fcnet_hiddens, list) or not args.fcnet_hiddens:
        raise ValueError("--fcnet-hiddens must define at least one hidden layer.")
    if any(int(size) <= 0 for size in args.fcnet_hiddens):
        raise ValueError("--fcnet-hiddens values must be > 0.")
    if not str(args.fcnet_activation).strip():
        raise ValueError("--fcnet-activation must be non-empty.")


def normalize_args(args: argparse.Namespace) -> None:
    if args.eval_report_episodes is not None:
        print("warning: --eval-report-episodes is deprecated; use --light-eval-episodes.")
        args.light_eval_episodes = int(args.eval_report_episodes)
    args.fcnet_hiddens = parse_csv_ints(args.fcnet_hiddens)


def parse_csv_ints(raw: str | list[int]) -> list[int]:
    if isinstance(raw, list):
        return [int(value) for value in raw]
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not parts:
        raise ValueError("Expected at least one integer value.")
    return [int(part) for part in parts]


def configure_cpu_threading() -> None:
    torch.set_num_threads(1)
    set_interop = getattr(torch, "set_num_interop_threads", None)
    if callable(set_interop):
        try:
            set_interop(1)
        except RuntimeError:
            pass


def make_env(env_config: dict) -> CommunicatingSchoolEnv:
    return CommunicatingSchoolEnv(
        epsilon=float(env_config.get("epsilon", 0.0)),
        render_mode=env_config.get("render_mode"),
        fish_preset=env_config.get("fish_preset"),
        time_limit=int(env_config.get("time_limit", DEFAULT_TIME_LIMIT)),
        num_red_fish=int(env_config.get("num_red_fish", DEFAULT_NUM_RED_FISH)),
        num_blue_fish=int(env_config.get("num_blue_fish", DEFAULT_NUM_BLUE_FISH)),
        num_red_pellets=int(env_config.get("num_red_pellets", DEFAULT_NUM_RED_PELLETS)),
        num_blue_pellets=int(env_config.get("num_blue_pellets", DEFAULT_NUM_BLUE_PELLETS)),
        food_capture_radius=float(env_config.get("food_capture_radius", 0.45)),
        pellet_reward=float(env_config.get("pellet_reward", DEFAULT_PELLET_REWARD)),
        step_cost=float(env_config.get("step_cost", DEFAULT_STEP_COST)),
        sector_radius=float(env_config.get("sector_radius", DEFAULT_SECTOR_RADIUS)),
        sector_num=int(env_config.get("sector_num", DEFAULT_SECTOR_NUM)),
        communication_radius=float(env_config.get("communication_radius", DEFAULT_SECTOR_RADIUS)),
        num_message_tokens=4,
        show_sensor_overlay=bool(env_config.get("show_sensor_overlay", False)),
        focus_agent_id=str(env_config.get("focus_agent_id", "fish_0")),
        mute_received_messages=bool(env_config.get("mute_received_messages", False)),
    )


def build_env_config(args: argparse.Namespace, *, show_sensor_overlay: bool = False, mute_received_messages: bool = False) -> dict[str, Any]:
    return {
        "epsilon": float(args.epsilon),
        "render_mode": None,
        "time_limit": int(args.time_limit),
        "num_red_fish": int(args.num_red_fish),
        "num_blue_fish": int(args.num_blue_fish),
        "num_red_pellets": int(args.num_red_pellets),
        "num_blue_pellets": int(args.num_blue_pellets),
        "food_capture_radius": 0.45,
        "pellet_reward": float(args.pellet_reward),
        "step_cost": float(args.step_cost),
        "sector_radius": float(args.sector_radius),
        "sector_num": int(args.sector_num),
        "communication_radius": float(args.sector_radius),
        "show_sensor_overlay": bool(show_sensor_overlay),
        "focus_agent_id": "fish_0",
        "mute_received_messages": bool(mute_received_messages),
    }


def build_algo(args: argparse.Namespace, *, num_gpus: int, env_config: dict[str, Any]):
    sample_env = make_env(env_config)
    try:
        obs_space = sample_env.observation_space
        action_space = sample_env.action_space
    finally:
        sample_env.close()

    config = (
        PPOConfig()
        .environment(env=ENV_ID, env_config=env_config)
        .framework("torch")
        .resources(num_gpus=num_gpus)
        .env_runners(
            num_env_runners=args.num_env_runners,
            num_envs_per_env_runner=args.num_envs_per_runner,
            rollout_fragment_length=args.rollout_fragment_length,
            sample_timeout_s=SAMPLE_TIMEOUT_S,
        )
        .training(
            train_batch_size=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            lambda_=args.gae_lambda,
            gamma=args.gamma,
            lr=args.learning_rate,
            entropy_coeff=args.entropy_coeff,
            num_epochs=args.num_epochs,
            model={
                "fcnet_hiddens": list(args.fcnet_hiddens),
                "fcnet_activation": str(args.fcnet_activation),
            },
        )
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
            count_steps_by=COUNT_STEPS_BY,
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
    return config, config.build_algo()


def resolve_restore_target(raw_path: str | None) -> str | None:
    if not raw_path:
        return None
    if raw_path.startswith("file://"):
        checkpoint_path = uri_to_local_path(raw_path)
    else:
        checkpoint_path = Path(raw_path)
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Restore checkpoint not found: {checkpoint_path}")
    return str(checkpoint_path)


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


def _nested_get(mapping: Any, path: Iterable[str]) -> Any:
    current = mapping
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _first_path_float(mapping: dict[str, Any], candidates: list[tuple[str, ...]]) -> float:
    values = [_nested_get(mapping, path) for path in candidates]
    return _first_finite_float(values)


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


def extract_agent_steps_total(result: dict[str, Any]) -> int:
    counters = result.get("counters", {})
    return _first_non_negative_int(
        [
            result.get("num_agent_steps_sampled_lifetime"),
            counters.get("num_agent_steps_sampled"),
        ]
    )


def extract_sample_time_ms(result: dict[str, Any]) -> float:
    return _first_path_float(
        result,
        [
            ("timers", "sample_time_ms"),
            ("timers", "env_runner_sampling_timer"),
            ("timers", "sample_timer"),
        ],
    )


def extract_learner_time_ms(result: dict[str, Any]) -> float:
    return _first_path_float(
        result,
        [
            ("timers", "learn_time_ms"),
            ("timers", "learner_update_timer"),
            ("timers", "learn_on_batch_time_ms"),
        ],
    )


def resolve_ray_session_dir() -> str | None:
    try:
        global_node = getattr(ray._private.worker, "_global_node", None)
        if global_node is not None:
            get_session_dir_path = getattr(global_node, "get_session_dir_path", None)
            if callable(get_session_dir_path):
                return str(Path(get_session_dir_path()).resolve())
            address_info = getattr(global_node, "address_info", None)
            if isinstance(address_info, dict) and address_info.get("session_dir"):
                return str(Path(address_info["session_dir"]).resolve())
        worker_node = getattr(getattr(ray._private.worker, "global_worker", None), "node", None)
        if worker_node is not None:
            address_info = getattr(worker_node, "address_info", None)
            if isinstance(address_info, dict) and address_info.get("session_dir"):
                return str(Path(address_info["session_dir"]).resolve())
    except Exception:
        return None
    return None


def make_eval_env_factory(args: argparse.Namespace, *, mute_received_messages: bool = False):
    env_config = build_env_config(args, show_sensor_overlay=False, mute_received_messages=mute_received_messages)
    return lambda: CommunicatingSchoolEnv(**env_config)


def run_light_eval(algo, *, args: argparse.Namespace, base_seed: int) -> ColorCommEvalResult:
    return evaluate_multi_agent_rollouts(
        algo=algo,
        env_factory=make_eval_env_factory(args, mute_received_messages=False),
        num_episodes=args.light_eval_episodes,
        base_seed=base_seed,
        stack_mode="old",
        policy_id=SHARED_POLICY_ID,
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
    best_eval_result: ColorCommEvalResult | None,
    num_checkpoint_evaluations: int,
    time_to_first_positive_total_reward: dict[str, Any] | None,
    final_checkpoint_path: str | None,
    checkpoint_eval_mode: str,
    light_eval_episodes: int,
) -> dict[str, Any]:
    return {
        "best_checkpoint": best_checkpoint_record,
        "best_mean_total_reward": float(best_eval_result.mean_total_reward) if best_eval_result is not None else None,
        "best_mean_pellets_per_fish": float(best_eval_result.mean_pellets_per_fish) if best_eval_result is not None else None,
        "checkpoint_eval_mode": checkpoint_eval_mode,
        "light_eval_episodes": int(light_eval_episodes),
        "num_checkpoint_evaluations": int(num_checkpoint_evaluations),
        "time_to_first_positive_total_reward": time_to_first_positive_total_reward,
        "final_checkpoint_path": final_checkpoint_path,
    }


def write_training_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    device: str,
    num_gpus: int,
    env_config: dict[str, Any],
    restore_target: str | None,
) -> None:
    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(device),
        "num_gpus": int(num_gpus),
        "restore_from_checkpoint": restore_target,
        "env_config": env_config,
        "model_config": {
            "fcnet_hiddens": list(args.fcnet_hiddens),
            "fcnet_activation": str(args.fcnet_activation),
        },
        "algo_config": {
            "count_steps_by": COUNT_STEPS_BY,
            "rollout_fragment_length": int(args.rollout_fragment_length),
            "sample_timeout_s": SAMPLE_TIMEOUT_S,
            "train_batch_size": int(args.train_batch_size),
            "minibatch_size": int(args.minibatch_size),
            "num_epochs": int(args.num_epochs),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "learning_rate": float(args.learning_rate),
            "entropy_coeff": float(args.entropy_coeff),
            "checkpoint_every_iterations": int(args.checkpoint_every_iterations),
            "light_eval_episodes": int(args.light_eval_episodes),
        },
        "seed": int(args.seed),
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    normalize_args(args)
    validate_args(args)
    configure_cpu_threading()

    device = resolve_device(args.device)
    num_gpus = 1 if device == "cuda" else 0
    checkpoint_root = Path(args.checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    restore_target = resolve_restore_target(args.restore_from_checkpoint)
    eval_report_jsonl_path = checkpoint_root / "eval_reports.jsonl"
    eval_report_csv_path = checkpoint_root / "eval_reports.csv"
    run_summary_path = checkpoint_root / "run_summary.json"
    training_metadata_path = checkpoint_root / "training_metadata.json"
    local_fs = pafs.LocalFileSystem()

    register_env(ENV_ID, make_env)
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=True)

    env_config = build_env_config(args, show_sensor_overlay=False, mute_received_messages=False)
    algo_config, algo = build_algo(
        args,
        num_gpus=num_gpus,
        env_config=env_config,
    )
    session_dir = resolve_ray_session_dir()
    if restore_target is not None:
        algo.restore(restore_target)
    write_training_metadata(
        training_metadata_path,
        args=args,
        device=device,
        num_gpus=num_gpus,
        env_config=env_config,
        restore_target=restore_target,
    )

    print("V8 - Color Communication School RLlib training")
    print(
        "Config: "
        f"iterations={args.train_iterations}, "
        f"env_runners={args.num_env_runners}, "
        f"envs_per_runner={args.num_envs_per_runner}, "
        f"checkpoint_every={args.checkpoint_every_iterations}, "
        f"device={device}, num_gpus={num_gpus}, "
        f"checkpoint_root={checkpoint_root.resolve()}, "
        f"restore_from_checkpoint={restore_target or 'none'}, "
        f"num_red_fish={args.num_red_fish}, "
        f"num_blue_fish={args.num_blue_fish}, "
        f"num_red_pellets={args.num_red_pellets}, "
        f"num_blue_pellets={args.num_blue_pellets}, "
        f"time_limit={args.time_limit}, "
        f"pellet_reward={args.pellet_reward}, "
        f"step_cost={args.step_cost}, "
        f"sector_radius={args.sector_radius}, "
        f"sector_num={args.sector_num}, "
        f"learning_rate={args.learning_rate}, "
        f"entropy_coeff={args.entropy_coeff}, "
        f"light_eval_episodes={args.light_eval_episodes}, "
        f"eval_report_seed={args.eval_report_seed}, "
        f"gamma={args.gamma}, "
        f"gae_lambda={args.gae_lambda}, "
        f"fcnet_hiddens={','.join(str(size) for size in args.fcnet_hiddens)}, "
        f"fcnet_activation={args.fcnet_activation}, "
        f"count_steps_by={algo_config.count_steps_by}, "
        f"rollout_fragment_length={algo_config.get_rollout_fragment_length()}, "
        f"sample_timeout_s={algo_config.sample_timeout_s}, "
        f"train_batch_size={algo_config.total_train_batch_size}, "
        f"minibatch_size={args.minibatch_size}, "
        f"num_epochs={args.num_epochs}"
    )
    if session_dir:
        print(f"ray_session_dir={session_dir}")

    latest_checkpoint = None
    report_rows_nested: list[dict[str, Any]] = []
    report_rows_flat: list[dict[str, Any]] = []
    best_checkpoint_record: dict[str, Any] | None = None
    best_eval_result: ColorCommEvalResult | None = None
    time_to_first_positive_total_reward: dict[str, Any] | None = None

    try:
        for iteration in range(1, args.train_iterations + 1):
            train_wall_start = time.perf_counter()
            result = algo.train()
            train_loop_wall_ms = (time.perf_counter() - train_wall_start) * 1000.0
            reward_mean = extract_reward_mean(result)
            env_steps_total = extract_timesteps_total(result)
            agent_steps_total = extract_agent_steps_total(result)
            sample_time_ms = extract_sample_time_ms(result)
            learner_time_ms = extract_learner_time_ms(result)
            print(
                f"iter={iteration:03d} "
                f"env_steps_total={env_steps_total} "
                f"agent_steps_total={agent_steps_total if agent_steps_total >= 0 else 'na'} "
                f"episode_reward_mean={format_metric(reward_mean)} "
                f"sample_time_ms={format_metric(sample_time_ms, precision=1)} "
                f"learner_time_ms={format_metric(learner_time_ms, precision=1)} "
                f"train_loop_wall_ms={format_metric(train_loop_wall_ms, precision=1)}"
            )

            if iteration % args.checkpoint_every_iterations != 0:
                continue

            checkpoint_path = checkpoint_root / f"checkpoint_{iteration:05d}"
            latest_checkpoint = save_algorithm_checkpoint(algo, checkpoint_path, local_fs)
            print(f"checkpoint_saved: {latest_checkpoint}")

            light_eval_start = time.perf_counter()
            light_result = run_light_eval(
                algo,
                args=args,
                base_seed=args.eval_report_seed + (iteration * 1_000_000),
            )
            light_eval_wall_ms = (time.perf_counter() - light_eval_start) * 1000.0

            report_nested = {
                "iteration": int(iteration),
                "timesteps_total": int(env_steps_total),
                "checkpoint_path": str(latest_checkpoint),
                "training_reward_mean": float(reward_mean),
                "eval_mode": "light",
                "light_eval_episodes": int(args.light_eval_episodes),
                "eval_result": light_result.to_dict(),
            }
            report_flat = {
                "iteration": int(iteration),
                "timesteps_total": int(env_steps_total),
                "checkpoint_path": str(latest_checkpoint),
                "training_reward_mean": float(reward_mean),
                "eval_mode": "light",
                "light_eval_episodes": int(args.light_eval_episodes),
                **flatten_result(light_result),
            }
            report_rows_nested.append(report_nested)
            report_rows_flat.append(report_flat)
            write_jsonl(eval_report_jsonl_path, report_rows_nested)
            write_flat_csv(eval_report_csv_path, report_rows_flat)

            if compare_results(light_result, best_eval_result) >= 0:
                best_eval_result = light_result
                best_checkpoint_record = {
                    "iteration": int(iteration),
                    "timesteps_total": int(env_steps_total),
                    "checkpoint_path": str(latest_checkpoint),
                    "eval_mode": "light",
                    "light_eval_episodes": int(args.light_eval_episodes),
                    **flatten_result(light_result),
                }

            if (
                time_to_first_positive_total_reward is None
                and np.isfinite(light_result.mean_total_reward)
                and light_result.mean_total_reward > 0.0
            ):
                time_to_first_positive_total_reward = {
                    "iteration": int(iteration),
                    "timesteps_total": int(env_steps_total),
                    "mean_total_reward": float(light_result.mean_total_reward),
                }

            summary = build_run_summary(
                best_checkpoint_record=best_checkpoint_record,
                best_eval_result=best_eval_result,
                num_checkpoint_evaluations=len(report_rows_nested),
                time_to_first_positive_total_reward=time_to_first_positive_total_reward,
                final_checkpoint_path=None,
                checkpoint_eval_mode="light",
                light_eval_episodes=args.light_eval_episodes,
            )
            run_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

            print(
                "light_eval_report: "
                f"iter={iteration:03d} "
                f"mean_total_reward={format_metric(light_result.mean_total_reward)} "
                f"mean_pellets_per_fish={format_metric(light_result.mean_pellets_per_fish)} "
                f"red_food={format_metric(light_result.mean_pellets_red_eaten_by_red)} "
                f"blue_food={format_metric(light_result.mean_pellets_blue_eaten_by_blue)} "
                f"light_eval_wall_ms={format_metric(light_eval_wall_ms, precision=1)} "
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
                time_to_first_positive_total_reward=time_to_first_positive_total_reward,
                final_checkpoint_path=str(final_checkpoint_path.resolve()),
                checkpoint_eval_mode="light",
                light_eval_episodes=args.light_eval_episodes,
            )
            run_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            print("training_status: reached_iteration_budget")
            algo.stop()
            ray.shutdown()


if __name__ == "__main__":
    main()
