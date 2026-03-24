"""V9 RLlib PPO checkpoint evaluation entrypoint for raw-torque schooling."""

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
    compute_batched_deterministic_actions,
    evaluate_multi_agent_random_rollouts,
    evaluate_multi_agent_rollouts,
    find_latest_checkpoint,
    flatten_result,
    sample_random_action,
    uri_to_local_path,
)
from triangles import CommunicatingSchoolEnv

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ENV_ID = "v9_raw_torque_communication_school_env_eval"

logging.getLogger("ray._common.deprecation").setLevel(logging.ERROR)

DEFAULT_MODEL_CONFIG = {
    "fcnet_hiddens": [512, 512, 256],
    "fcnet_activation": "tanh",
}


def configure_cpu_threading() -> None:
    torch.set_num_threads(1)
    set_interop = getattr(torch, "set_num_interop_threads", None)
    if callable(set_interop):
        try:
            set_interop(1)
        except RuntimeError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained V9 shared-policy checkpoint.")
    parser.add_argument("--policy-mode", type=str, choices=["trained", "random"], default="trained")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint-list-file", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=str, default="./rllib_checkpoints_baseline_v9_raw_torque_comm")
    parser.add_argument("--max-frames", type=int, default=10_000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--hide-sensor-overlay", action="store_true")
    parser.add_argument("--render-profile", type=str, choices=["fast", "full"], default="fast")
    parser.add_argument("--render-engine", type=str, choices=["auto", "blit", "safe"], default="auto")
    parser.add_argument("--focus-agent-id", type=str, default="fish_0")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--num-red-fish", type=int, default=DEFAULT_NUM_RED_FISH)
    parser.add_argument("--num-blue-fish", type=int, default=DEFAULT_NUM_BLUE_FISH)
    parser.add_argument("--num-red-pellets", type=int, default=DEFAULT_NUM_RED_PELLETS)
    parser.add_argument("--num-blue-pellets", type=int, default=DEFAULT_NUM_BLUE_PELLETS)
    parser.add_argument("--time-limit", type=int, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--pellet-reward", type=float, default=DEFAULT_PELLET_REWARD)
    parser.add_argument("--step-cost", type=float, default=DEFAULT_STEP_COST)
    parser.add_argument("--sector-radius", type=float, default=DEFAULT_SECTOR_RADIUS)
    parser.add_argument("--sector-num", type=int, default=DEFAULT_SECTOR_NUM)
    parser.add_argument("--reward-mode", type=str, choices=["forage", "locomotion_debug"], default="forage")
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--actuator-time-constant", type=float, default=0.10)
    parser.add_argument("--mute-mode", type=str, choices=["normal", "both"], default="normal")
    parser.add_argument("--mute-messages", action="store_true")
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


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def write_summary_json(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 1:
        path.write_text(json.dumps(rows[0], indent=2, sort_keys=True), encoding="utf-8")
        return
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_checkpoint_path_from_text(path_text: str) -> tuple[Path, str]:
    if path_text.startswith("file://"):
        checkpoint_path = uri_to_local_path(path_text)
        restore_target = str(checkpoint_path.resolve())
    else:
        checkpoint_path = Path(path_text)
        restore_target = str(checkpoint_path.resolve())
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path, restore_target


def load_checkpoint_path(args: argparse.Namespace, checkpoint_root: Path) -> tuple[Path, str]:
    if args.checkpoint_path:
        return load_checkpoint_path_from_text(args.checkpoint_path)
    checkpoint_path = find_latest_checkpoint(checkpoint_root)
    return checkpoint_path, str(checkpoint_path.resolve())


def load_checkpoint_targets(args: argparse.Namespace, checkpoint_root: Path) -> list[tuple[Path, str]]:
    if args.policy_mode != "trained":
        return []
    if args.checkpoint_list_file:
        list_path = Path(args.checkpoint_list_file).resolve()
        if not list_path.exists():
            raise FileNotFoundError(f"Checkpoint list file not found: {list_path}")
        targets: list[tuple[Path, str]] = []
        for raw_line in list_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("file://"):
                targets.append(load_checkpoint_path_from_text(line))
            else:
                candidate_path = Path(line)
                if not candidate_path.is_absolute():
                    relative_to_list = (list_path.parent / candidate_path).resolve()
                    if relative_to_list.exists():
                        candidate_path = relative_to_list
                    else:
                        candidate_path = candidate_path.resolve()
                targets.append(load_checkpoint_path_from_text(str(candidate_path)))
        if not targets:
            raise ValueError(f"No checkpoint paths found in list file: {list_path}")
        return targets
    return [load_checkpoint_path(args, checkpoint_root)]


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0.")
    if args.num_red_fish <= 0:
        raise ValueError("--num-red-fish must be > 0.")
    if args.reward_mode == "forage":
        if args.num_blue_fish <= 0:
            raise ValueError("--num-blue-fish must be > 0 in forage mode.")
        if args.num_red_pellets <= 0 or args.num_blue_pellets <= 0:
            raise ValueError("--num-red-pellets and --num-blue-pellets must be > 0 in forage mode.")
    else:
        if args.num_blue_fish < 0:
            raise ValueError("--num-blue-fish must be >= 0 in locomotion_debug mode.")
        if args.num_red_pellets < 0 or args.num_blue_pellets < 0:
            raise ValueError("--num-red-pellets and --num-blue-pellets must be >= 0 in locomotion_debug mode.")
    if args.time_limit <= 0:
        raise ValueError("--time-limit must be > 0.")
    if args.pellet_reward <= 0.0:
        raise ValueError("--pellet-reward must be > 0.")
    if args.step_cost < 0.0:
        raise ValueError("--step-cost must be >= 0.")
    if args.sector_radius <= 0.0:
        raise ValueError("--sector-radius must be > 0.")
    if args.sector_num != DEFAULT_SECTOR_NUM:
        raise ValueError(f"--sector-num must remain {DEFAULT_SECTOR_NUM} in V9.")
    if args.history_length <= 0:
        raise ValueError("--history-length must be > 0.")
    if args.actuator_time_constant < 0.0:
        raise ValueError("--actuator-time-constant must be >= 0.")
    if args.checkpoint_list_file and args.checkpoint_path:
        raise ValueError("--checkpoint-path and --checkpoint-list-file are mutually exclusive.")


def locate_training_metadata(checkpoint_path: Path) -> Path | None:
    current = checkpoint_path.resolve()
    for candidate_dir in [current, *current.parents]:
        metadata_path = candidate_dir / "training_metadata.json"
        if metadata_path.exists():
            return metadata_path
    return None


def load_training_metadata(checkpoint_path: Path) -> dict[str, object]:
    metadata_path = locate_training_metadata(checkpoint_path)
    if metadata_path is None:
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def build_env_config(
    args: argparse.Namespace,
    *,
    render_mode: str | None,
    show_sensor_overlay: bool,
    mute_received_messages: bool,
    base_env_config: dict[str, object] | None = None,
) -> dict:
    source = dict(base_env_config or {})
    return {
        "epsilon": float(source.get("epsilon", args.epsilon)),
        "render_mode": render_mode,
        "render_profile": str(args.render_profile),
        "render_engine": str(args.render_engine),
        "time_limit": int(source.get("time_limit", args.time_limit)),
        "num_red_fish": int(source.get("num_red_fish", args.num_red_fish)),
        "num_blue_fish": int(source.get("num_blue_fish", args.num_blue_fish)),
        "num_red_pellets": int(source.get("num_red_pellets", args.num_red_pellets)),
        "num_blue_pellets": int(source.get("num_blue_pellets", args.num_blue_pellets)),
        "food_capture_radius": float(source.get("food_capture_radius", 0.45)),
        "pellet_reward": float(source.get("pellet_reward", args.pellet_reward)),
        "step_cost": float(source.get("step_cost", args.step_cost)),
        "sector_radius": float(source.get("sector_radius", args.sector_radius)),
        "sector_num": int(source.get("sector_num", args.sector_num)),
        "communication_radius": float(source.get("communication_radius", args.sector_radius)),
        "reward_mode": str(source.get("reward_mode", args.reward_mode)),
        "history_length": int(source.get("history_length", args.history_length)),
        "actuator_time_constant": float(source.get("actuator_time_constant", args.actuator_time_constant)),
        "show_sensor_overlay": bool(show_sensor_overlay),
        "focus_agent_id": str(args.focus_agent_id),
        "mute_received_messages": bool(mute_received_messages),
    }


def build_eval_algo(
    *,
    env_id: str,
    env_config: dict,
    num_gpus: int,
    seed: int,
    use_old_stack: bool,
    model_config: dict[str, object],
):
    sample_env = CommunicatingSchoolEnv(**env_config)
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
        .training(model=model_config)
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


def evaluate_pair(
    *,
    args: argparse.Namespace,
    algo,
    stack_mode: str | None,
    normal_env_config: dict,
    muted_env_config: dict,
) -> tuple[dict[str, object], dict[str, object] | None, float | None]:
    if args.policy_mode == "trained":
        normal_result = evaluate_multi_agent_rollouts(
            algo=algo,
            env_factory=lambda: CommunicatingSchoolEnv(**normal_env_config),
            num_episodes=args.episodes,
            base_seed=args.seed,
            stack_mode=stack_mode,
            policy_id=SHARED_POLICY_ID,
        )
        muted_result = None
        if args.mute_mode == "both":
            muted_result = evaluate_multi_agent_rollouts(
                algo=algo,
                env_factory=lambda: CommunicatingSchoolEnv(**muted_env_config),
                num_episodes=args.episodes,
                base_seed=args.seed,
                stack_mode=stack_mode,
                policy_id=SHARED_POLICY_ID,
            )
    else:
        normal_result = evaluate_multi_agent_random_rollouts(
            env_factory=lambda: CommunicatingSchoolEnv(**normal_env_config),
            num_episodes=args.episodes,
            base_seed=args.seed,
        )
        muted_result = None
        if args.mute_mode == "both":
            muted_result = evaluate_multi_agent_random_rollouts(
                env_factory=lambda: CommunicatingSchoolEnv(**muted_env_config),
                num_episodes=args.episodes,
                base_seed=args.seed,
            )

    comm_gain = None
    if muted_result is not None:
        comm_gain = float(normal_result.mean_total_reward - muted_result.mean_total_reward)
    return normal_result.to_dict(), muted_result.to_dict() if muted_result is not None else None, comm_gain


def summarize_row(
    *,
    checkpoint_path: str | None,
    policy_mode: str,
    device: str | None,
    stack_mode: str | None,
    mute_mode: str,
    normal_summary: dict[str, object],
    muted_summary: dict[str, object] | None,
    comm_gain: float | None,
) -> tuple[dict[str, object], dict[str, object]]:
    nested = {
        "checkpoint_path": checkpoint_path,
        "policy_mode": policy_mode,
        "device": device,
        "stack_mode": stack_mode,
        "mute_mode": mute_mode,
        "eval_result": normal_summary,
    }
    flat = {
        "checkpoint_path": checkpoint_path,
        "policy_mode": policy_mode,
        "device": device,
        "stack_mode": stack_mode,
        "mute_mode": mute_mode,
        **flatten_result(ColorCommEvalResult(**normal_summary)),
    }
    if muted_summary is not None:
        nested["message_muted_eval"] = muted_summary
        nested["comm_gain_total_reward"] = comm_gain
        flat.update(flatten_result(ColorCommEvalResult(**muted_summary), prefix="muted_"))
        flat["comm_gain_total_reward"] = comm_gain
    return nested, flat


def evaluate_trained_checkpoint(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    restore_target: str,
    device: str,
) -> tuple[dict[str, object], dict[str, object]]:
    metadata = load_training_metadata(checkpoint_path)
    env_template = metadata.get("env_config", {}) if isinstance(metadata.get("env_config"), dict) else {}
    model_config = metadata.get("model_config", {}) if isinstance(metadata.get("model_config"), dict) else {}
    if not model_config:
        model_config = dict(DEFAULT_MODEL_CONFIG)
    num_gpus = 1 if device == "cuda" else 0

    register_env(ENV_ID, lambda config: CommunicatingSchoolEnv(**config))
    normal_env_config = build_env_config(
        args,
        render_mode=None,
        show_sensor_overlay=not args.hide_sensor_overlay,
        mute_received_messages=False,
        base_env_config=env_template,
    )
    muted_env_config = build_env_config(
        args,
        render_mode=None,
        show_sensor_overlay=False,
        mute_received_messages=True,
        base_env_config=env_template,
    )

    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=True)
    algo = build_eval_algo(
        env_id=ENV_ID,
        env_config=normal_env_config,
        num_gpus=num_gpus,
        seed=args.seed,
        use_old_stack=True,
        model_config=model_config,
    )
    stack_mode = "old"
    try:
        try:
            algo.restore(restore_target)
        except Exception:
            algo.stop()
            algo = build_eval_algo(
                env_id=ENV_ID,
                env_config=normal_env_config,
                num_gpus=num_gpus,
                seed=args.seed,
                use_old_stack=False,
                model_config=model_config,
            )
            algo.restore(restore_target)
            stack_mode = "new"

        normal_summary, muted_summary, comm_gain = evaluate_pair(
            args=args,
            algo=algo,
            stack_mode=stack_mode,
            normal_env_config=normal_env_config,
            muted_env_config=muted_env_config,
        )
    finally:
        algo.stop()
        ray.shutdown()

    return summarize_row(
        checkpoint_path=str(checkpoint_path.resolve()),
        policy_mode="trained",
        device=device,
        stack_mode=stack_mode,
        mute_mode=args.mute_mode,
        normal_summary=normal_summary,
        muted_summary=muted_summary,
        comm_gain=comm_gain,
    )


def evaluate_random_policy(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    normal_env_config = build_env_config(
        args,
        render_mode=None,
        show_sensor_overlay=False,
        mute_received_messages=False,
    )
    muted_env_config = build_env_config(
        args,
        render_mode=None,
        show_sensor_overlay=False,
        mute_received_messages=True,
    )
    normal_summary, muted_summary, comm_gain = evaluate_pair(
        args=args,
        algo=None,
        stack_mode=None,
        normal_env_config=normal_env_config,
        muted_env_config=muted_env_config,
    )
    return summarize_row(
        checkpoint_path=None,
        policy_mode="random",
        device=None,
        stack_mode=None,
        mute_mode=args.mute_mode,
        normal_summary=normal_summary,
        muted_summary=muted_summary,
        comm_gain=comm_gain,
    )


def batch_mode_requested(args: argparse.Namespace) -> bool:
    return bool(
        args.summary_json
        or args.summary_csv
        or args.episodes > 1
        or args.no_render
        or args.policy_mode == "random"
        or args.checkpoint_list_file
    )


def run_visual_rollout(args: argparse.Namespace, *, checkpoint_path: Path, restore_target: str, device: str) -> None:
    metadata = load_training_metadata(checkpoint_path)
    env_template = metadata.get("env_config", {}) if isinstance(metadata.get("env_config"), dict) else {}
    model_config = metadata.get("model_config", {}) if isinstance(metadata.get("model_config"), dict) else {}
    if not model_config:
        model_config = dict(DEFAULT_MODEL_CONFIG)

    register_env(ENV_ID, lambda config: CommunicatingSchoolEnv(**config))
    eval_env_config = build_env_config(
        args,
        render_mode=None,
        show_sensor_overlay=not args.hide_sensor_overlay,
        mute_received_messages=args.mute_messages,
        base_env_config=env_template,
    )
    num_gpus = 1 if device == "cuda" else 0
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=True)
    algo = build_eval_algo(
        env_id=ENV_ID,
        env_config=eval_env_config,
        num_gpus=num_gpus,
        seed=args.seed,
        use_old_stack=True,
        model_config=model_config,
    )
    stack_mode = "old"
    try:
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
                model_config=model_config,
            )
            algo.restore(restore_target)
            stack_mode = "new"

        print("V9 - Raw Torque Communication School RLlib evaluation")
        print(f"Policy mode: trained")
        print(f"Checkpoint: {checkpoint_path.resolve()}")
        print(f"Device: {device}")
        print(f"Stack mode: {stack_mode}")

        render_mode = None if args.no_render else "human"
        env = CommunicatingSchoolEnv(
            **build_env_config(
                args,
                render_mode=render_mode,
                show_sensor_overlay=not args.hide_sensor_overlay,
                mute_received_messages=args.mute_messages,
                base_env_config=env_template,
            )
        )
        print(f"Render: {not args.no_render}")
        print(f"Sensor overlay: {not args.hide_sensor_overlay}")
        print(f"Render profile: {args.render_profile}")
        print(f"Render engine: {args.render_engine}")
        print(f"Focus agent: {args.focus_agent_id}")
        print(f"Reward mode: {eval_env_config.get('reward_mode', 'forage')}")
        print(f"History length: {eval_env_config.get('history_length', args.history_length)}")
        try:
            obs_dict, _ = env.reset(seed=args.seed)
            for frame_idx in range(1, args.max_frames + 1):
                action_dict = compute_batched_deterministic_actions(
                    algo,
                    obs_dict,
                    stack_mode=stack_mode,
                    policy_id=SHARED_POLICY_ID,
                )
                obs_dict, rewards, terminateds, truncateds, infos = env.step(action_dict)
                if not args.no_render:
                    env.render()
                focus_info = infos[args.focus_agent_id]
                if frame_idx % args.log_every == 0 or focus_info.get("food_eaten_this_step", 0):
                    print(
                        f"frame={frame_idx:05d} reward={rewards[args.focus_agent_id]:.3f} "
                        f"food_step={focus_info.get('food_eaten_this_step', 0)} "
                        f"food_episode={focus_info.get('food_eaten_episode', 0)} "
                        f"red_food_episode={focus_info.get('food_eaten_episode_red', 0)} "
                        f"blue_food_episode={focus_info.get('food_eaten_episode_blue', 0)} "
                        f"visible_food={focus_info.get('visible_food_count', 0)} "
                        f"msg={focus_info.get('emitted_message_token', 0)} "
                        f"fwd={focus_info.get('forward_velocity', 0.0):.3f} "
                        f"lat={focus_info.get('lateral_velocity', 0.0):.3f} "
                        f"ang={focus_info.get('angular_velocity', 0.0):.3f} "
                        f"torque={focus_info.get('mean_abs_applied_torque', 0.0):.3f} "
                        f"joint_limit={focus_info.get('mean_joint_limit_ratio', 0.0):.3f}"
                    )
                if terminateds["__all__"] or truncateds["__all__"]:
                    print(
                        f"episode_end frame={frame_idx:05d} food_episode={focus_info.get('food_eaten_episode', 0)} "
                        f"reward={rewards[args.focus_agent_id]:.3f} "
                        f"zero_crossings={focus_info.get('joint_velocity_zero_crossings', 0)}"
                    )
                    obs_dict, _ = env.reset(seed=args.seed + frame_idx)
        finally:
            env.close()
    finally:
        algo.stop()
        ray.shutdown()


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_cpu_threading()

    device = resolve_device(args.device)
    checkpoint_root = Path(args.checkpoint_root)
    explicit_batch_mode = batch_mode_requested(args)

    if args.policy_mode == "trained":
        targets = load_checkpoint_targets(args, checkpoint_root)
    else:
        targets = []

    if explicit_batch_mode:
        nested_rows: list[dict[str, object]] = []
        flat_rows: list[dict[str, object]] = []
        if args.policy_mode == "trained":
            for checkpoint_path, restore_target in targets:
                nested, flat = evaluate_trained_checkpoint(
                    args=args,
                    checkpoint_path=checkpoint_path,
                    restore_target=restore_target,
                    device=device,
                )
                nested_rows.append(nested)
                flat_rows.append(flat)
        else:
            nested, flat = evaluate_random_policy(args)
            nested_rows.append(nested)
            flat_rows.append(flat)

        print(
            "Mode: batch checkpoint scoring"
            if args.policy_mode == "trained"
            else "Mode: batch random-policy scoring"
        )
        print(f"Checkpoint count: {len(nested_rows)}")
        best_row = max(
            flat_rows,
            key=lambda row: float(row.get("mean_pellets_per_fish", float("-inf"))),
        )
        print(f"best_checkpoint={best_row.get('checkpoint_path', 'random_policy')}")
        print(f"best_mean_pellets_per_fish={float(best_row.get('mean_pellets_per_fish', float('nan'))):.3f}")
        if args.summary_json:
            write_summary_json(Path(args.summary_json), nested_rows)
        if args.summary_csv:
            summary_csv_path = Path(args.summary_csv)
            summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
            write_summary_csv(summary_csv_path, flat_rows)
        return

    checkpoint_path, restore_target = targets[0]
    run_visual_rollout(args, checkpoint_path=checkpoint_path, restore_target=restore_target, device=device)


if __name__ == "__main__":
    main()
