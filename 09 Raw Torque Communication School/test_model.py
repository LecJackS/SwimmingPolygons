"""V9 RLlib PPO checkpoint evaluation entrypoint for muscle-activation schooling."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
_existing_pythonpath = os.environ.get("PYTHONPATH", "")
if str(SCRIPT_DIR) not in _existing_pythonpath.split(os.pathsep):
    os.environ["PYTHONPATH"] = str(SCRIPT_DIR) if not _existing_pythonpath else str(SCRIPT_DIR) + os.pathsep + _existing_pythonpath
_HEADLESS_TESTMODEL_ARGS = {"--no-render", "--summary-json", "--summary-csv"}
if any(flag in sys.argv for flag in _HEADLESS_TESTMODEL_ARGS):
    os.environ.setdefault("MPLBACKEND", "Agg")

import imageio.v2 as imageio
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import NoopLogger
from ray.tune.registry import register_env
import torch

from newstack_policy import build_v9_newstack_multi_module_spec
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
    compute_batched_stochastic_actions,
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

ENV_ID = "v9_muscle_activation_communication_school_env_eval"

logging.getLogger("ray._common.deprecation").setLevel(logging.ERROR)

DEFAULT_MODEL_CONFIG = {
    "fcnet_hiddens": [512, 512, 256],
    "fcnet_activation": "tanh",
}

_PYPLOT = None


def get_pyplot(*, force_agg: bool = False):
    global _PYPLOT
    if _PYPLOT is None:
        import matplotlib

        if force_agg:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        _PYPLOT = plt
    elif force_agg:
        backend = str(_PYPLOT.get_backend()).lower()
        if "agg" not in backend:
            _PYPLOT.switch_backend("Agg")
    return _PYPLOT


def configure_cpu_threading() -> None:
    torch.set_num_threads(1)
    set_interop = getattr(torch, "set_num_interop_threads", None)
    if callable(set_interop):
        try:
            set_interop(1)
        except RuntimeError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained V9 shared-policy muscle-activation checkpoint.")
    parser.add_argument("--policy-mode", type=str, choices=["trained", "random"], default="trained")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint-list-file", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=str, default="./rllib_checkpoints_baseline_v9_muscle_activation_comm")
    parser.add_argument("--max-frames", type=int, default=10_000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--hide-sensor-overlay", action="store_true")
    parser.add_argument("--render-profile", type=str, choices=["fast", "full"], default="fast")
    parser.add_argument("--render-engine", type=str, choices=["auto", "blit", "safe"], default="auto")
    parser.add_argument("--focus-agent-id", type=str, default="fish_0")
    parser.add_argument("--save-gif", type=str, default=None)
    parser.add_argument("--gif-seconds", type=float, default=6.0)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--num-red-fish", type=int, default=DEFAULT_NUM_RED_FISH)
    parser.add_argument("--num-blue-fish", type=int, default=DEFAULT_NUM_BLUE_FISH)
    parser.add_argument("--num-red-pellets", type=int, default=DEFAULT_NUM_RED_PELLETS)
    parser.add_argument("--num-blue-pellets", type=int, default=DEFAULT_NUM_BLUE_PELLETS)
    parser.add_argument("--num-body-segments", type=int, default=5)
    parser.add_argument("--time-limit", type=int, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--pellet-reward", type=float, default=DEFAULT_PELLET_REWARD)
    parser.add_argument("--step-cost", type=float, default=DEFAULT_STEP_COST)
    parser.add_argument("--food-respawn-mode", type=str, choices=["respawn", "deplete"], default="respawn")
    parser.add_argument(
        "--forage-timeout-mode",
        type=str,
        choices=["fixed_time_limit", "reset_on_food"],
        default="fixed_time_limit",
    )
    parser.add_argument("--forage-idle-timeout-steps", type=int, default=500)
    parser.add_argument(
        "--forage-time-context-mode",
        type=str,
        choices=["episode_progress", "idle_budget_remaining"],
        default="episode_progress",
    )
    parser.add_argument("--sector-radius", type=float, default=DEFAULT_SECTOR_RADIUS)
    parser.add_argument("--sector-num", type=int, default=DEFAULT_SECTOR_NUM)
    parser.add_argument("--reward-mode", type=str, choices=["forage", "locomotion_debug"], default="forage")
    parser.add_argument("--observation-profile", type=str, choices=["full_v9", "minimal_wave"], default="full_v9")
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--activation-time-constant", type=float, default=0.12)
    parser.add_argument("--joint-passive-stiffness", type=float, default=10.0)
    parser.add_argument("--joint-soft-limit-start-ratio", type=float, default=0.70)
    parser.add_argument("--joint-soft-limit-stiffness", type=float, default=18.0)
    parser.add_argument("--joint-soft-limit-damping", type=float, default=2.0)
    parser.add_argument("--body-linear-drag", type=float, default=1.0)
    parser.add_argument("--propulsion-near-limit-weight", type=float, default=-0.22)
    parser.add_argument("--propulsion-saturation-weight", type=float, default=-0.10)
    parser.add_argument("--propulsion-torque-weight", type=float, default=-0.05)
    parser.add_argument("--mute-mode", type=str, choices=["normal", "both"], default="normal")
    parser.add_argument("--action-selection", type=str, choices=["deterministic", "stochastic", "both"], default="deterministic")
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


def capture_frame_rgb(env: CommunicatingSchoolEnv) -> np.ndarray:
    if env.fig is None:
        raise RuntimeError("Render figure is not initialized; cannot capture frame.")
    frame = np.asarray(env.fig.canvas.buffer_rgba(), dtype=np.uint8)
    return frame[:, :, :3].copy()


def pump_live_window(seconds: float = 0.01) -> None:
    plt = get_pyplot()
    backend = plt.get_backend().lower()
    if "agg" in backend:
        return
    plt.pause(seconds)


def resolve_live_render_profile(requested_profile: str, *, save_gif: bool, no_render: bool) -> str:
    if requested_profile == "full" and not save_gif and not no_render:
        print("warning: live full mode is unreliable on Windows; using fast for live view.")
        print("warning: use --render-profile full together with --save-gif for diagnostic capture.")
        return "fast"
    return requested_profile


def write_gif(path: Path, frames: list[np.ndarray], *, fps: int) -> None:
    if not frames:
        raise ValueError("Cannot write GIF with zero frames.")
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, format="GIF", duration=(1.0 / float(fps)))


def live_action_selections(requested: str) -> list[str]:
    return ["deterministic", "stochastic"] if requested == "both" else [requested]


def resolve_gif_output_path(path: Path | None, *, action_selection: str, multi_mode: bool) -> Path | None:
    if path is None or not multi_mode:
        return path
    return path.with_name(f"{path.stem}_{action_selection}{path.suffix}")


def compute_batched_actions_for_selection(
    algo,
    obs_dict: dict[str, np.ndarray],
    *,
    stack_mode: str,
    action_selection: str,
) -> dict[str, dict[str, object]]:
    if action_selection == "stochastic":
        return compute_batched_stochastic_actions(
            algo,
            obs_dict,
            stack_mode=stack_mode,
            policy_id=SHARED_POLICY_ID,
        )
    return compute_batched_deterministic_actions(
        algo,
        obs_dict,
        stack_mode=stack_mode,
        policy_id=SHARED_POLICY_ID,
    )


def score_row_mean_pellets_per_fish(row: dict[str, object]) -> float:
    if "mean_pellets_per_fish" in row:
        return float(row.get("mean_pellets_per_fish", float("-inf")))
    if "deterministic_mean_pellets_per_fish" in row:
        return float(row.get("deterministic_mean_pellets_per_fish", float("-inf")))
    return float("-inf")


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
    if args.num_body_segments < 2:
        raise ValueError("--num-body-segments must be >= 2.")
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
    if args.forage_idle_timeout_steps <= 0:
        raise ValueError("--forage-idle-timeout-steps must be > 0.")
    if args.sector_radius <= 0.0:
        raise ValueError("--sector-radius must be > 0.")
    if args.sector_num != DEFAULT_SECTOR_NUM:
        raise ValueError(f"--sector-num must remain {DEFAULT_SECTOR_NUM} in V9.")
    if args.history_length <= 0:
        raise ValueError("--history-length must be > 0.")
    if args.activation_time_constant < 0.0:
        raise ValueError("--activation-time-constant must be >= 0.")
    if args.joint_passive_stiffness < 0.0:
        raise ValueError("--joint-passive-stiffness must be >= 0.")
    if not (0.0 <= args.joint_soft_limit_start_ratio < 1.0):
        raise ValueError("--joint-soft-limit-start-ratio must be in [0, 1).")
    if args.joint_soft_limit_stiffness < 0.0:
        raise ValueError("--joint-soft-limit-stiffness must be >= 0.")
    if args.joint_soft_limit_damping < 0.0:
        raise ValueError("--joint-soft-limit-damping must be >= 0.")
    if args.body_linear_drag < 0.0:
        raise ValueError("--body-linear-drag must be >= 0.")
    for field_name in (
        "propulsion_near_limit_weight",
        "propulsion_saturation_weight",
        "propulsion_torque_weight",
    ):
        if not np.isfinite(float(getattr(args, field_name))):
            raise ValueError(f"--{field_name.replace('_', '-')} must be finite.")
    if args.checkpoint_list_file and args.checkpoint_path:
        raise ValueError("--checkpoint-path and --checkpoint-list-file are mutually exclusive.")
    if args.gif_seconds <= 0.0:
        raise ValueError("--gif-seconds must be > 0.")
    if args.gif_fps <= 0:
        raise ValueError("--gif-fps must be > 0.")
    if args.epsilon != 0.0:
        print("warning: --epsilon is ignored in V9 evaluation; eval and live viewing always use zero epsilon.")


def locate_training_metadata(checkpoint_path: Path) -> Path | None:
    current = checkpoint_path.resolve()
    for candidate_dir in [current, *current.parents]:
        metadata_path = candidate_dir / "training_metadata.json"
        if metadata_path.exists():
            return metadata_path
    return None


def infer_num_body_segments_from_checkpoint(checkpoint_path: Path) -> int | None:
    checkpoint_dir = checkpoint_path.resolve()
    ctor_args_path = checkpoint_dir / "class_and_ctor_args.pkl"
    if not ctor_args_path.exists():
        return None
    try:
        payload = pickle.loads(ctor_args_path.read_bytes())
        ctor_args_and_kwargs = payload.get("ctor_args_and_kwargs")
        if not isinstance(ctor_args_and_kwargs, tuple) or not ctor_args_and_kwargs:
            return None
        ctor_args = ctor_args_and_kwargs[0]
        if not isinstance(ctor_args, dict):
            return None
        policies = ctor_args.get("policies")
        if not isinstance(policies, dict):
            return None
        policy_spec = policies.get(SHARED_POLICY_ID)
        if not isinstance(policy_spec, tuple) or len(policy_spec) < 2:
            return None
        action_space = policy_spec[1]
        motion_space = getattr(action_space, "spaces", {}).get("motion")
        motion_shape = getattr(motion_space, "shape", None)
        if not motion_shape:
            return None
        motion_dim = int(motion_shape[0])
        if motion_dim <= 0:
            return None
        return motion_dim + 1
    except Exception:
        return None


def load_training_metadata(checkpoint_path: Path) -> dict[str, object]:
    metadata_path = locate_training_metadata(checkpoint_path)
    if metadata_path is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path.resolve()} is unsupported because training_metadata.json is missing."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    env_config = metadata.get("env_config", {})
    if not isinstance(env_config, dict):
        raise ValueError(
            f"Checkpoint {checkpoint_path.resolve()} is unsupported because env_config is not a dictionary."
        )
    if "num_body_segments" not in env_config:
        inferred_num_body_segments = infer_num_body_segments_from_checkpoint(checkpoint_path)
        if inferred_num_body_segments is not None:
            env_config["num_body_segments"] = inferred_num_body_segments
        else:
            env_config["num_body_segments"] = 5
    if "num_body_segments" not in env_config:
        raise ValueError(
            f"Checkpoint {checkpoint_path.resolve()} is unsupported because num_body_segments is missing from training metadata."
        )
    return metadata


def build_env_config(
    args: argparse.Namespace,
    *,
    render_mode: str | None,
    show_sensor_overlay: bool,
    mute_received_messages: bool,
    base_env_config: dict[str, object] | None = None,
    render_profile_override: str | None = None,
) -> dict:
    source = dict(base_env_config or {})
    return {
        "epsilon": 0.0,
        "motion_epsilon": 0.0,
        "message_epsilon": 0.0,
        "render_mode": render_mode,
        "render_profile": str(render_profile_override or args.render_profile),
        "render_engine": str(args.render_engine),
        "num_body_segments": int(source.get("num_body_segments", args.num_body_segments)),
        "time_limit": int(source.get("time_limit", args.time_limit)),
        "num_red_fish": int(source.get("num_red_fish", args.num_red_fish)),
        "num_blue_fish": int(source.get("num_blue_fish", args.num_blue_fish)),
        "num_red_pellets": int(source.get("num_red_pellets", args.num_red_pellets)),
        "num_blue_pellets": int(source.get("num_blue_pellets", args.num_blue_pellets)),
        "food_capture_radius": float(source.get("food_capture_radius", 0.45)),
        "pellet_reward": float(source.get("pellet_reward", args.pellet_reward)),
        "step_cost": float(source.get("step_cost", args.step_cost)),
        "food_respawn_mode": str(source.get("food_respawn_mode", args.food_respawn_mode)),
        "forage_timeout_mode": str(source.get("forage_timeout_mode", args.forage_timeout_mode)),
        "forage_idle_timeout_steps": int(source.get("forage_idle_timeout_steps", args.forage_idle_timeout_steps)),
        "forage_time_context_mode": str(source.get("forage_time_context_mode", args.forage_time_context_mode)),
        "sector_radius": float(source.get("sector_radius", args.sector_radius)),
        "sector_num": int(source.get("sector_num", args.sector_num)),
        "communication_radius": float(source.get("communication_radius", args.sector_radius)),
        "reward_mode": str(source.get("reward_mode", args.reward_mode)),
        "training_phase": str(source.get("training_phase", "forage_full")),
        "observation_profile": str(source.get("observation_profile", args.observation_profile)),
        "history_length": int(source.get("history_length", args.history_length)),
        "activation_time_constant": float(
            source.get("activation_time_constant", source.get("actuator_time_constant", args.activation_time_constant))
        ),
        "joint_passive_stiffness": float(source.get("joint_passive_stiffness", args.joint_passive_stiffness)),
        "joint_soft_limit_start_ratio": float(source.get("joint_soft_limit_start_ratio", args.joint_soft_limit_start_ratio)),
        "joint_soft_limit_stiffness": float(source.get("joint_soft_limit_stiffness", args.joint_soft_limit_stiffness)),
        "joint_soft_limit_damping": float(source.get("joint_soft_limit_damping", args.joint_soft_limit_damping)),
        "body_linear_drag": float(source.get("body_linear_drag", args.body_linear_drag)),
        "propulsion_near_limit_weight": float(source.get("propulsion_near_limit_weight", args.propulsion_near_limit_weight)),
        "propulsion_saturation_weight": float(source.get("propulsion_saturation_weight", args.propulsion_saturation_weight)),
        "propulsion_torque_weight": float(source.get("propulsion_torque_weight", args.propulsion_torque_weight)),
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
        .multi_agent(
            policies={
                SHARED_POLICY_ID: PolicySpec(
                    observation_space=obs_space,
                    action_space=action_space,
                    config={},
                )
            },
            policy_mapping_fn=lambda agent_id, episode, worker=None, **kwargs: SHARED_POLICY_ID,
            policies_to_train=[SHARED_POLICY_ID],
        )
        .fault_tolerance(
            restart_failed_env_runners=False,
            max_num_env_runner_restarts=0,
        )
        .debugging(seed=seed, logger_creator=lambda cfg: NoopLogger(cfg, "."))
    )
    if use_old_stack:
        config = config.training(model=model_config).api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    else:
        module_spec = build_v9_newstack_multi_module_spec(
            observation_space=obs_space,
            action_space=action_space,
            model_config=dict(model_config),
            inference_only=False,
        )
        config = config.rl_module(rl_module_spec=module_spec).api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
    return config.build_algo()


def evaluate_single_action_selection(
    *,
    args: argparse.Namespace,
    algo,
    stack_mode: str | None,
    normal_env_config: dict,
    muted_env_config: dict,
    action_selection: str,
) -> tuple[dict[str, object], dict[str, object] | None, float | None]:
    if args.policy_mode == "trained":
        normal_result = evaluate_multi_agent_rollouts(
            algo=algo,
            env_factory=lambda: CommunicatingSchoolEnv(**normal_env_config),
            num_episodes=args.episodes,
            base_seed=args.seed,
            action_selection=action_selection,
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
                action_selection=action_selection,
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


def diff_result_summaries(
    candidate: dict[str, object],
    baseline: dict[str, object],
) -> dict[str, float]:
    delta: dict[str, float] = {}
    for key, candidate_value in candidate.items():
        baseline_value = baseline.get(key)
        if isinstance(candidate_value, (int, float)) and isinstance(baseline_value, (int, float)):
            delta[key] = float(candidate_value) - float(baseline_value)
    return delta


def summarize_single_mode(
    *,
    checkpoint_path: str | None,
    policy_mode: str,
    device: str | None,
    stack_mode: str | None,
    mute_mode: str,
    action_selection: str,
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
        "action_selection": action_selection,
        "eval_result": normal_summary,
    }
    flat = {
        "checkpoint_path": checkpoint_path,
        "policy_mode": policy_mode,
        "device": device,
        "stack_mode": stack_mode,
        "mute_mode": mute_mode,
        "action_selection": action_selection,
        **flatten_result(ColorCommEvalResult(**normal_summary)),
    }
    if muted_summary is not None:
        nested["message_muted_eval"] = muted_summary
        nested["comm_gain_total_reward"] = comm_gain
        flat.update(flatten_result(ColorCommEvalResult(**muted_summary), prefix="muted_"))
        flat["comm_gain_total_reward"] = comm_gain
    return nested, flat


def summarize_both_modes(
    *,
    checkpoint_path: str | None,
    policy_mode: str,
    device: str | None,
    stack_mode: str | None,
    mute_mode: str,
    deterministic_summary: dict[str, object],
    deterministic_muted_summary: dict[str, object] | None,
    deterministic_comm_gain: float | None,
    stochastic_summary: dict[str, object],
    stochastic_muted_summary: dict[str, object] | None,
    stochastic_comm_gain: float | None,
) -> tuple[dict[str, object], dict[str, object]]:
    delta_summary = diff_result_summaries(stochastic_summary, deterministic_summary)
    nested = {
        "checkpoint_path": checkpoint_path,
        "policy_mode": policy_mode,
        "device": device,
        "stack_mode": stack_mode,
        "mute_mode": mute_mode,
        "action_selection": "both",
        "deterministic_eval": deterministic_summary,
        "stochastic_eval": stochastic_summary,
        "delta_eval": delta_summary,
    }
    flat = {
        "checkpoint_path": checkpoint_path,
        "policy_mode": policy_mode,
        "device": device,
        "stack_mode": stack_mode,
        "mute_mode": mute_mode,
        "action_selection": "both",
        **flatten_result(ColorCommEvalResult(**deterministic_summary), prefix="deterministic_"),
        **flatten_result(ColorCommEvalResult(**stochastic_summary), prefix="stochastic_"),
        **{f"delta_{key}": value for key, value in delta_summary.items()},
    }
    if deterministic_muted_summary is not None:
        nested["deterministic_message_muted_eval"] = deterministic_muted_summary
        flat.update(flatten_result(ColorCommEvalResult(**deterministic_muted_summary), prefix="deterministic_muted_"))
        flat["deterministic_comm_gain_total_reward"] = deterministic_comm_gain
    if stochastic_muted_summary is not None:
        nested["stochastic_message_muted_eval"] = stochastic_muted_summary
        flat.update(flatten_result(ColorCommEvalResult(**stochastic_muted_summary), prefix="stochastic_muted_"))
        flat["stochastic_comm_gain_total_reward"] = stochastic_comm_gain
    if deterministic_comm_gain is not None:
        nested["deterministic_comm_gain_total_reward"] = deterministic_comm_gain
    if stochastic_comm_gain is not None:
        nested["stochastic_comm_gain_total_reward"] = stochastic_comm_gain
    if deterministic_comm_gain is not None and stochastic_comm_gain is not None:
        nested["delta_comm_gain_total_reward"] = float(stochastic_comm_gain - deterministic_comm_gain)
        flat["delta_comm_gain_total_reward"] = float(stochastic_comm_gain - deterministic_comm_gain)
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
    preferred_stack = str(metadata.get("policy_stack", "")).strip().lower()

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
    if preferred_stack == "new":
        algo = build_eval_algo(
            env_id=ENV_ID,
            env_config=normal_env_config,
            num_gpus=num_gpus,
            seed=args.seed,
            use_old_stack=False,
            model_config=model_config,
        )
        stack_mode = "new"
    else:
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
            if preferred_stack == "new":
                raise
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

        if args.action_selection == "both":
            deterministic_summary, deterministic_muted_summary, deterministic_comm_gain = evaluate_single_action_selection(
                args=args,
                algo=algo,
                stack_mode=stack_mode,
                normal_env_config=normal_env_config,
                muted_env_config=muted_env_config,
                action_selection="deterministic",
            )
            stochastic_summary, stochastic_muted_summary, stochastic_comm_gain = evaluate_single_action_selection(
                args=args,
                algo=algo,
                stack_mode=stack_mode,
                normal_env_config=normal_env_config,
                muted_env_config=muted_env_config,
                action_selection="stochastic",
            )
        else:
            normal_summary, muted_summary, comm_gain = evaluate_single_action_selection(
                args=args,
                algo=algo,
                stack_mode=stack_mode,
                normal_env_config=normal_env_config,
                muted_env_config=muted_env_config,
                action_selection=args.action_selection,
            )
    finally:
        algo.stop()
        ray.shutdown()

    if args.action_selection == "both":
        return summarize_both_modes(
            checkpoint_path=str(checkpoint_path.resolve()),
            policy_mode="trained",
            device=device,
            stack_mode=stack_mode,
            mute_mode=args.mute_mode,
            deterministic_summary=deterministic_summary,
            deterministic_muted_summary=deterministic_muted_summary,
            deterministic_comm_gain=deterministic_comm_gain,
            stochastic_summary=stochastic_summary,
            stochastic_muted_summary=stochastic_muted_summary,
            stochastic_comm_gain=stochastic_comm_gain,
        )
    return summarize_single_mode(
        checkpoint_path=str(checkpoint_path.resolve()),
        policy_mode="trained",
        device=device,
        stack_mode=stack_mode,
        mute_mode=args.mute_mode,
        action_selection=args.action_selection,
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
    normal_summary, muted_summary, comm_gain = evaluate_single_action_selection(
        args=args,
        algo=None,
        stack_mode=None,
        normal_env_config=normal_env_config,
        muted_env_config=muted_env_config,
        action_selection="deterministic",
    )
    return summarize_single_mode(
        checkpoint_path=None,
        policy_mode="random",
        device=None,
        stack_mode=None,
        mute_mode=args.mute_mode,
        action_selection="deterministic",
        normal_summary=normal_summary,
        muted_summary=muted_summary,
        comm_gain=comm_gain,
    )


def batch_mode_requested(args: argparse.Namespace) -> bool:
    return bool(
        args.summary_json
        or args.summary_csv
        or args.episodes > 1
        or (args.no_render and not args.save_gif)
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
    preferred_stack = str(metadata.get("policy_stack", "")).strip().lower()
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=True)
    if preferred_stack == "new":
        algo = build_eval_algo(
            env_id=ENV_ID,
            env_config=eval_env_config,
            num_gpus=num_gpus,
            seed=args.seed,
            use_old_stack=False,
            model_config=model_config,
        )
        stack_mode = "new"
    else:
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
            if preferred_stack == "new":
                raise
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

        print("V9 - Muscle Activation Communication School RLlib evaluation")
        print(f"Policy mode: trained")
        print(f"Checkpoint: {checkpoint_path.resolve()}")
        print(f"Device: {device}")
        print(f"Stack mode: {stack_mode}")
        base_gif_output_path = Path(args.save_gif).resolve() if args.save_gif else None
        if base_gif_output_path is not None:
            get_pyplot(force_agg=True)
        selections = live_action_selections(args.action_selection)
        multi_mode = len(selections) > 1
        for selection_idx, action_selection in enumerate(selections):
            if selection_idx > 0:
                print("-" * 80)
            gif_output_path = resolve_gif_output_path(
                base_gif_output_path,
                action_selection=action_selection,
                multi_mode=multi_mode,
            )
            effective_render_profile = resolve_live_render_profile(
                args.render_profile,
                save_gif=gif_output_path is not None,
                no_render=args.no_render,
            )
            render_mode = "human" if (gif_output_path is not None or not args.no_render) else None
            env = CommunicatingSchoolEnv(
                **build_env_config(
                    args,
                    render_mode=render_mode,
                    show_sensor_overlay=not args.hide_sensor_overlay,
                    mute_received_messages=args.mute_messages,
                    base_env_config=env_template,
                    render_profile_override=effective_render_profile,
                )
            )
            print(f"Action selection: {action_selection}")
            print(f"Render: {not args.no_render}")
            print(f"Sensor overlay: {not args.hide_sensor_overlay}")
            print(f"Render profile: {effective_render_profile}")
            print(f"Render engine: {args.render_engine}")
            print(f"Focus agent: {args.focus_agent_id}")
            print(f"Reward mode: {eval_env_config.get('reward_mode', 'forage')}")
            print(f"Observation profile: {eval_env_config.get('observation_profile', args.observation_profile)}")
            print(f"History length: {eval_env_config.get('history_length', args.history_length)}")
            print(f"Body segments: {eval_env_config.get('num_body_segments', args.num_body_segments)}")
            print(f"Food respawn mode: {eval_env_config.get('food_respawn_mode', args.food_respawn_mode)}")
            print(f"Forage timeout mode: {eval_env_config.get('forage_timeout_mode', args.forage_timeout_mode)}")
            print(
                "Forage idle timeout steps: "
                f"{eval_env_config.get('forage_idle_timeout_steps', args.forage_idle_timeout_steps)}"
            )
            print(
                "Forage time context mode: "
                f"{eval_env_config.get('forage_time_context_mode', args.forage_time_context_mode)}"
            )
            if gif_output_path is not None:
                print(f"GIF export: {gif_output_path}")
            try:
                obs_dict, _ = env.reset(seed=args.seed)
                gif_frames: list[np.ndarray] = []
                gif_frame_limit = int(round(args.gif_seconds * float(args.gif_fps))) if gif_output_path is not None else 0
                if render_mode == "human":
                    env.render()
                    if gif_output_path is not None and len(gif_frames) < gif_frame_limit:
                        gif_frames.append(capture_frame_rgb(env))
                    elif gif_output_path is None:
                        pump_live_window()
                for frame_idx in range(1, args.max_frames + 1):
                    action_dict = compute_batched_actions_for_selection(
                        algo,
                        obs_dict,
                        stack_mode=stack_mode,
                        action_selection=action_selection,
                    )
                    obs_dict, rewards, terminateds, truncateds, infos = env.step(action_dict)
                    if render_mode == "human":
                        env.render()
                        if gif_output_path is not None and len(gif_frames) < gif_frame_limit:
                            gif_frames.append(capture_frame_rgb(env))
                        elif gif_output_path is None:
                            pump_live_window()
                    focus_info = infos[args.focus_agent_id]
                    focus_action = np.asarray(action_dict[args.focus_agent_id]["motion"], dtype=np.float32).reshape(-1)
                    focus_activation = np.asarray(
                        focus_info.get(
                            "joint_activation_vector",
                            np.zeros_like(focus_action, dtype=np.float32),
                        ),
                        dtype=np.float32,
                    ).reshape(-1)
                    focus_action_preview = ", ".join(
                        f"{value:.3f}" for value in focus_action[: min(4, focus_action.size)]
                    )
                    focus_activation_preview = ", ".join(
                        f"{value:.3f}" for value in focus_activation[: min(4, focus_activation.size)]
                    )
                    if frame_idx % args.log_every == 0 or focus_info.get("food_eaten_this_step", 0):
                        print(
                            f"frame={frame_idx:05d} reward={rewards[args.focus_agent_id]:.3f} "
                            f"food_step={focus_info.get('food_eaten_this_step', 0)} "
                            f"food_episode={focus_info.get('food_eaten_episode', 0)} "
                            f"red_food_episode={focus_info.get('food_eaten_episode_red', 0)} "
                            f"blue_food_episode={focus_info.get('food_eaten_episode_blue', 0)} "
                            f"visible_food={focus_info.get('visible_food_count', 0)} "
                            f"msg={focus_info.get('emitted_message_token', 0)} "
                            f"cmd=[{focus_action_preview}] "
                            f"act=[{focus_activation_preview}] "
                            f"fwd={focus_info.get('forward_velocity', 0.0):.3f} "
                            f"lat={focus_info.get('lateral_velocity', 0.0):.3f} "
                            f"ang={focus_info.get('angular_velocity', 0.0):.3f} "
                            f"desired={focus_info.get('mean_abs_desired_activation', 0.0):.3f} "
                            f"activation={focus_info.get('mean_abs_activation', 0.0):.3f} "
                            f"torque={focus_info.get('mean_abs_applied_torque', 0.0):.3f} "
                            f"joint_limit={focus_info.get('mean_joint_limit_ratio', 0.0):.3f} "
                            f"quiet={int(bool(focus_info.get('joints_quiet', False)))} "
                            f"neg_fwd={int(bool(focus_info.get('negative_forward_velocity', False)))}"
                        )
                    if terminateds["__all__"] or truncateds["__all__"]:
                        print(
                            f"episode_end frame={frame_idx:05d} food_episode={focus_info.get('food_eaten_episode', 0)} "
                            f"reward={rewards[args.focus_agent_id]:.3f} "
                            f"zero_crossings={focus_info.get('joint_velocity_zero_crossings', 0)} "
                            f"activation_sign_changes={focus_info.get('activation_sign_changes_episode', 0)}"
                        )
                        obs_dict, _ = env.reset(seed=args.seed + frame_idx)
                    if gif_output_path is not None and len(gif_frames) >= gif_frame_limit:
                        break
                if gif_output_path is not None:
                    write_gif(gif_output_path, gif_frames, fps=args.gif_fps)
                    print(f"saved_gif={gif_output_path}")
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
            key=score_row_mean_pellets_per_fish,
        )
        print(f"best_checkpoint={best_row.get('checkpoint_path', 'random_policy')}")
        print(f"best_mean_pellets_per_fish={score_row_mean_pellets_per_fish(best_row):.3f}")
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
