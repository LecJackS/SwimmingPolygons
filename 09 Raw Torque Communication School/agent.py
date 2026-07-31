"""V9 RLlib PPO training entrypoint for muscle-activation communication schooling."""

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
os.environ.setdefault("MPLBACKEND", "Agg")
import re
import time
import traceback
from typing import Any, Iterable
import warnings

import numpy as np
import pyarrow.fs as pafs
import ray
from ray.rllib.algorithms.algorithm import Algorithm
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
    compare_results,
    evaluate_multi_agent_rollouts,
    flatten_result,
    uri_to_local_path,
)
from triangles import CommunicatingSchoolEnv

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ENV_ID = "v9_muscle_activation_communication_school_env"

warnings.filterwarnings(
    "ignore",
    message=r".*multi_gpu_train_one_step.*deprecated.*",
    category=DeprecationWarning,
)
logging.getLogger("ray.rllib.execution.train_ops").setLevel(logging.ERROR)
logging.getLogger("ray.rllib.utils.sgd").setLevel(logging.ERROR)
logging.getLogger("ray._common.deprecation").setLevel(logging.ERROR)


DEFAULT_TRAIN_BATCH_SIZE = 16000
DEFAULT_MINIBATCH_SIZE = 2048
DEFAULT_NUM_EPOCHS = 6
DEFAULT_LIGHT_EVAL_EPISODES = 2
DEFAULT_ROLLOUT_FRAGMENT_LENGTH = 500
SAMPLE_TIMEOUT_S = 180.0
COUNT_STEPS_BY = "agent_steps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V9 muscle-activation communication schooling fish.")
    parser.add_argument("--train-iterations", type=int, default=200)
    parser.add_argument("--num-env-runners", type=int, default=8)
    parser.add_argument("--num-envs-per-runner", type=int, default=2)
    parser.add_argument("--checkpoint-every-iterations", type=int, default=20)
    parser.add_argument("--checkpoint-root", type=str, default="./rllib_checkpoints_baseline_v9_muscle_activation_comm")
    parser.add_argument("--restore-from-checkpoint", type=str, default=None)
    parser.add_argument("--policy-stack", type=str, choices=["old", "new"], default="old")
    parser.add_argument(
        "--training-phase",
        type=str,
        choices=[
            "locomotion_only",
            "locomotion_teacher",
            "locomotion_self",
            "locomotion_propulsion_easy",
            "locomotion_propulsion_robust",
            "forage_full",
        ],
        default="forage_full",
    )
    parser.add_argument("--warmstart-motion-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epsilon", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--motion-epsilon-start", type=float, default=None)
    parser.add_argument("--motion-epsilon-end", type=float, default=None)
    parser.add_argument("--motion-epsilon-decay-iterations", type=int, default=None)
    parser.add_argument("--message-epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)

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
    parser.add_argument("--swim-assist-start-weight", type=float, default=0.35)
    parser.add_argument("--swim-assist-min-iterations", type=int, default=40)
    parser.add_argument("--swim-assist-disable-forward-velocity", type=float, default=0.03)
    parser.add_argument("--swim-assist-disable-joint-limit-occupancy", type=float, default=0.35)
    parser.add_argument("--swim-assist-disable-negative-forward-frac", type=float, default=0.45)
    parser.add_argument("--swim-assist-disable-consecutive-evals", type=int, default=2)
    parser.add_argument("--swim-assist-fade-evals", type=int, default=2)
    parser.add_argument("--light-eval-episodes", type=int, default=DEFAULT_LIGHT_EVAL_EPISODES)
    parser.add_argument("--eval-report-episodes", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--eval-report-seed", type=int, default=20_240)
    parser.add_argument("--gae-lambda", type=float, default=0.97)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--minibatch-size", type=int, default=DEFAULT_MINIBATCH_SIZE)
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--rollout-fragment-length", type=int, default=DEFAULT_ROLLOUT_FRAGMENT_LENGTH)
    parser.add_argument("--fcnet-hiddens", type=str, default="512,512,256")
    parser.add_argument("--fcnet-activation", type=str, default="tanh")
    return parser.parse_args()


def canonical_training_phase(raw_phase: str) -> str:
    phase = str(raw_phase).strip().lower()
    if phase == "locomotion_only":
        return "locomotion_self"
    return phase


def is_locomotion_training_phase(raw_phase: str) -> bool:
    return canonical_training_phase(raw_phase) in {
        "locomotion_teacher",
        "locomotion_self",
        "locomotion_propulsion_easy",
        "locomotion_propulsion_robust",
    }


def uses_teacher_phase_signal(raw_phase: str) -> bool:
    return canonical_training_phase(raw_phase) == "locomotion_teacher"


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
    if args.restore_from_checkpoint and args.warmstart_motion_checkpoint:
        raise ValueError("--restore-from-checkpoint and --warmstart-motion-checkpoint are mutually exclusive.")
    if args.warmstart_motion_checkpoint and args.policy_stack != "new":
        raise ValueError("--warmstart-motion-checkpoint requires --policy-stack new.")
    if args.warmstart_motion_checkpoint and canonical_training_phase(args.training_phase) not in {
        "locomotion_self",
        "locomotion_propulsion_easy",
        "locomotion_propulsion_robust",
        "forage_full",
    }:
        raise ValueError(
            "--warmstart-motion-checkpoint requires --training-phase locomotion_self, locomotion_propulsion_easy, locomotion_propulsion_robust, or forage_full."
        )
    if args.num_red_fish <= 0:
        raise ValueError("--num-red-fish must be > 0.")
    if args.num_body_segments < 2:
        raise ValueError("--num-body-segments must be >= 2.")
    effective_reward = effective_reward_mode(args)
    if effective_reward == "forage":
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
    if not (0.0 <= args.motion_epsilon_start <= 1.0):
        raise ValueError("--motion-epsilon-start must be in [0, 1].")
    if not (0.0 <= args.motion_epsilon_end <= 1.0):
        raise ValueError("--motion-epsilon-end must be in [0, 1].")
    if args.motion_epsilon_decay_iterations <= 0:
        raise ValueError("--motion-epsilon-decay-iterations must be > 0.")
    if not (0.0 <= args.message_epsilon <= 1.0):
        raise ValueError("--message-epsilon must be in [0, 1].")
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
    if args.swim_assist_start_weight < 0.0:
        raise ValueError("--swim-assist-start-weight must be >= 0.")
    if args.swim_assist_min_iterations < 0:
        raise ValueError("--swim-assist-min-iterations must be >= 0.")
    if args.swim_assist_disable_forward_velocity < 0.0:
        raise ValueError("--swim-assist-disable-forward-velocity must be >= 0.")
    if args.swim_assist_disable_joint_limit_occupancy < 0.0:
        raise ValueError("--swim-assist-disable-joint-limit-occupancy must be >= 0.")
    if args.swim_assist_disable_negative_forward_frac < 0.0:
        raise ValueError("--swim-assist-disable-negative-forward-frac must be >= 0.")
    if args.swim_assist_disable_consecutive_evals <= 0:
        raise ValueError("--swim-assist-disable-consecutive-evals must be > 0.")
    if args.swim_assist_fade_evals <= 0:
        raise ValueError("--swim-assist-fade-evals must be > 0.")
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
    args.training_phase = canonical_training_phase(args.training_phase)
    args.fcnet_hiddens = parse_csv_ints(args.fcnet_hiddens)
    motion_schedule_explicit = any(
        value is not None
        for value in (
            args.motion_epsilon_start,
            args.motion_epsilon_end,
            args.motion_epsilon_decay_iterations,
        )
    )
    if str(args.training_phase) == "forage_full":
        default_motion_epsilon_start = 0.25
        default_motion_epsilon_end = 0.0
        default_decay_iterations = 60
    else:
        default_motion_epsilon_start = 0.0
        default_motion_epsilon_end = 0.0
        default_decay_iterations = 1
    if motion_schedule_explicit:
        if args.epsilon is not None:
            print("warning: --epsilon is deprecated and ignored because motion-epsilon schedule args were provided.")
        if args.motion_epsilon_start is None:
            args.motion_epsilon_start = default_motion_epsilon_start
        if args.motion_epsilon_end is None:
            args.motion_epsilon_end = default_motion_epsilon_end
        if args.motion_epsilon_decay_iterations is None:
            args.motion_epsilon_decay_iterations = default_decay_iterations
        return
    if args.epsilon is not None:
        print("warning: --epsilon is deprecated; treating it as a constant motion epsilon for training only.")
        constant_epsilon = float(args.epsilon)
        args.motion_epsilon_start = constant_epsilon
        args.motion_epsilon_end = constant_epsilon
        args.motion_epsilon_decay_iterations = 1
        return
    args.motion_epsilon_start = default_motion_epsilon_start
    args.motion_epsilon_end = default_motion_epsilon_end
    args.motion_epsilon_decay_iterations = default_decay_iterations


def parse_csv_ints(raw: str | list[int]) -> list[int]:
    if isinstance(raw, list):
        return [int(value) for value in raw]
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not parts:
        raise ValueError("Expected at least one integer value.")
    return [int(part) for part in parts]


def effective_reward_mode(args: argparse.Namespace) -> str:
    return "locomotion_debug" if is_locomotion_training_phase(args.training_phase) else str(args.reward_mode)


def effective_message_head_mode(args: argparse.Namespace) -> str:
    return "fixed_zero" if is_locomotion_training_phase(args.training_phase) else "trainable"


def stack_mode_for_args(args: argparse.Namespace) -> str:
    return "new" if str(args.policy_stack) == "new" else "old"


def build_newstack_model_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "fcnet_hiddens": list(args.fcnet_hiddens),
        "fcnet_activation": str(args.fcnet_activation),
        "training_phase": str(args.training_phase),
        "message_head_mode": effective_message_head_mode(args),
        "motion_std_min": 0.15,
        "motion_std_max": 1.0,
        "motion_std_init": 0.35,
        "phase_signal_dim": 2 if uses_teacher_phase_signal(args.training_phase) else 0,
    }


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
        epsilon=env_config.get("epsilon"),
        motion_epsilon=float(env_config.get("motion_epsilon", env_config.get("epsilon", 0.0))),
        message_epsilon=float(env_config.get("message_epsilon", 0.0)),
        render_mode=env_config.get("render_mode"),
        num_body_segments=int(env_config.get("num_body_segments", 5)),
        fish_preset=env_config.get("fish_preset"),
        time_limit=int(env_config.get("time_limit", DEFAULT_TIME_LIMIT)),
        num_red_fish=int(env_config.get("num_red_fish", DEFAULT_NUM_RED_FISH)),
        num_blue_fish=int(env_config.get("num_blue_fish", DEFAULT_NUM_BLUE_FISH)),
        num_red_pellets=int(env_config.get("num_red_pellets", DEFAULT_NUM_RED_PELLETS)),
        num_blue_pellets=int(env_config.get("num_blue_pellets", DEFAULT_NUM_BLUE_PELLETS)),
        food_capture_radius=float(env_config.get("food_capture_radius", 0.45)),
        pellet_reward=float(env_config.get("pellet_reward", DEFAULT_PELLET_REWARD)),
        step_cost=float(env_config.get("step_cost", DEFAULT_STEP_COST)),
        food_respawn_mode=str(env_config.get("food_respawn_mode", "respawn")),
        forage_timeout_mode=str(env_config.get("forage_timeout_mode", "fixed_time_limit")),
        forage_idle_timeout_steps=int(env_config.get("forage_idle_timeout_steps", 500)),
        forage_time_context_mode=str(env_config.get("forage_time_context_mode", "episode_progress")),
        sector_radius=float(env_config.get("sector_radius", DEFAULT_SECTOR_RADIUS)),
        sector_num=int(env_config.get("sector_num", DEFAULT_SECTOR_NUM)),
        communication_radius=float(env_config.get("communication_radius", DEFAULT_SECTOR_RADIUS)),
        num_message_tokens=4,
        reward_mode=str(env_config.get("reward_mode", "forage")),
        training_phase=str(env_config.get("training_phase", "forage_full")),
        observation_profile=str(env_config.get("observation_profile", "full_v9")),
        history_length=int(env_config.get("history_length", 8)),
        activation_time_constant=float(env_config.get("activation_time_constant", 0.12)),
        joint_passive_stiffness=float(env_config.get("joint_passive_stiffness", 10.0)),
        joint_soft_limit_start_ratio=float(env_config.get("joint_soft_limit_start_ratio", 0.70)),
        joint_soft_limit_stiffness=float(env_config.get("joint_soft_limit_stiffness", 18.0)),
        joint_soft_limit_damping=float(env_config.get("joint_soft_limit_damping", 2.0)),
        body_linear_drag=float(env_config.get("body_linear_drag", 1.0)),
        propulsion_near_limit_weight=float(env_config.get("propulsion_near_limit_weight", -0.22)),
        propulsion_saturation_weight=float(env_config.get("propulsion_saturation_weight", -0.10)),
        propulsion_torque_weight=float(env_config.get("propulsion_torque_weight", -0.05)),
        swim_assist_start_weight=float(env_config.get("swim_assist_start_weight", 0.0)),
        show_sensor_overlay=bool(env_config.get("show_sensor_overlay", False)),
        focus_agent_id=str(env_config.get("focus_agent_id", "fish_0")),
        mute_received_messages=bool(env_config.get("mute_received_messages", False)),
    )


def build_env_config(
    args: argparse.Namespace,
    *,
    show_sensor_overlay: bool = False,
    mute_received_messages: bool = False,
    swim_assist_start_weight_override: float | None = None,
    motion_epsilon_override: float | None = None,
    message_epsilon_override: float | None = None,
) -> dict[str, Any]:
    training_phase = canonical_training_phase(args.training_phase)
    locomotion_phase = is_locomotion_training_phase(training_phase)
    reward_mode = effective_reward_mode(args)
    swim_assist_weight = (
        0.0
        if reward_mode != "forage"
        else float(
            args.swim_assist_start_weight
            if swim_assist_start_weight_override is None
            else swim_assist_start_weight_override
        )
    )
    return {
        "training_phase": str(training_phase),
        "motion_epsilon": float(
            args.motion_epsilon_start if motion_epsilon_override is None else motion_epsilon_override
        ),
        "message_epsilon": float(args.message_epsilon if message_epsilon_override is None else message_epsilon_override),
        "render_mode": None,
        "num_body_segments": int(args.num_body_segments),
        "time_limit": int(args.time_limit),
        "num_red_fish": 1 if locomotion_phase else int(args.num_red_fish),
        "num_blue_fish": 0 if locomotion_phase else int(args.num_blue_fish),
        "num_red_pellets": 0 if locomotion_phase else int(args.num_red_pellets),
        "num_blue_pellets": 0 if locomotion_phase else int(args.num_blue_pellets),
        "food_capture_radius": 0.45,
        "pellet_reward": float(args.pellet_reward),
        "step_cost": float(args.step_cost),
        "food_respawn_mode": str(args.food_respawn_mode),
        "forage_timeout_mode": str(args.forage_timeout_mode),
        "forage_idle_timeout_steps": int(args.forage_idle_timeout_steps),
        "forage_time_context_mode": str(args.forage_time_context_mode),
        "sector_radius": float(args.sector_radius),
        "sector_num": int(args.sector_num),
        "communication_radius": float(args.sector_radius),
        "reward_mode": reward_mode,
        "observation_profile": str(args.observation_profile),
        "history_length": int(args.history_length),
        "activation_time_constant": float(args.activation_time_constant),
        "joint_passive_stiffness": float(args.joint_passive_stiffness),
        "joint_soft_limit_start_ratio": float(args.joint_soft_limit_start_ratio),
        "joint_soft_limit_stiffness": float(args.joint_soft_limit_stiffness),
        "joint_soft_limit_damping": float(args.joint_soft_limit_damping),
        "body_linear_drag": float(args.body_linear_drag),
        "propulsion_near_limit_weight": float(args.propulsion_near_limit_weight),
        "propulsion_saturation_weight": float(args.propulsion_saturation_weight),
        "propulsion_torque_weight": float(args.propulsion_torque_weight),
        "swim_assist_start_weight": float(swim_assist_weight),
        "show_sensor_overlay": bool(show_sensor_overlay),
        "focus_agent_id": "fish_0",
        "mute_received_messages": bool(mute_received_messages or locomotion_phase),
    }


def build_old_stack_algo(args: argparse.Namespace, *, num_gpus: int, env_config: dict[str, Any]):
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
            policy_mapping_fn=lambda agent_id, episode, worker=None, **kwargs: SHARED_POLICY_ID,
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


def build_new_stack_algo(args: argparse.Namespace, *, num_gpus: int, env_config: dict[str, Any]):
    sample_env = make_env(env_config)
    try:
        obs_space = sample_env.observation_space
        action_space = sample_env.action_space
    finally:
        sample_env.close()

    module_spec = build_v9_newstack_multi_module_spec(
        observation_space=obs_space,
        action_space=action_space,
        model_config=build_newstack_model_config(args),
        inference_only=False,
    )
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
        )
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
            count_steps_by=COUNT_STEPS_BY,
        )
        .rl_module(rl_module_spec=module_spec)
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .fault_tolerance(
            restart_failed_env_runners=False,
            max_num_env_runner_restarts=0,
        )
        .debugging(seed=args.seed, logger_creator=lambda cfg: NoopLogger(cfg, "."))
    )
    return config, config.build_algo()


def build_algo(args: argparse.Namespace, *, num_gpus: int, env_config: dict[str, Any]):
    if str(args.policy_stack) == "new":
        return build_new_stack_algo(args, num_gpus=num_gpus, env_config=env_config)
    return build_old_stack_algo(args, num_gpus=num_gpus, env_config=env_config)


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


def locate_training_metadata(checkpoint_target: str | Path) -> Path | None:
    current = Path(checkpoint_target).resolve()
    for candidate_dir in [current, *current.parents]:
        metadata_path = candidate_dir / "training_metadata.json"
        if metadata_path.exists():
            return metadata_path
    return None


def infer_num_body_segments_from_checkpoint(checkpoint_target: str | Path) -> int | None:
    checkpoint_dir = Path(checkpoint_target).resolve()
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


def load_checkpoint_num_body_segments(checkpoint_target: str | Path) -> int:
    metadata_path = locate_training_metadata(checkpoint_target)
    if metadata_path is None:
        raise ValueError(
            f"Checkpoint {Path(checkpoint_target).resolve()} is unsupported because training_metadata.json is missing."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    env_config = metadata.get("env_config", {})
    if not isinstance(env_config, dict):
        raise ValueError(
            f"Checkpoint {Path(checkpoint_target).resolve()} is unsupported because env_config is not a dictionary."
        )
    if "num_body_segments" not in env_config:
        inferred_num_body_segments = infer_num_body_segments_from_checkpoint(checkpoint_target)
        if inferred_num_body_segments is not None:
            env_config["num_body_segments"] = inferred_num_body_segments
        else:
            env_config["num_body_segments"] = 5
    if "num_body_segments" not in env_config:
        raise ValueError(
            f"Checkpoint {Path(checkpoint_target).resolve()} is unsupported because num_body_segments is missing from training metadata."
        )
    return int(env_config["num_body_segments"])


def assert_checkpoint_segment_count(checkpoint_target: str, *, expected_num_body_segments: int, purpose: str) -> None:
    checkpoint_segments = load_checkpoint_num_body_segments(checkpoint_target)
    if checkpoint_segments != int(expected_num_body_segments):
        raise ValueError(
            f"{purpose} checkpoint segment-count mismatch: requested {expected_num_body_segments}, checkpoint has {checkpoint_segments}."
        )


def apply_motion_warmstart(algo, warmstart_target: str, *, expected_num_body_segments: int) -> list[str]:
    assert_checkpoint_segment_count(
        warmstart_target,
        expected_num_body_segments=expected_num_body_segments,
        purpose="Warmstart",
    )
    donor_algo = Algorithm.from_checkpoint(warmstart_target)
    try:
        donor_weights = donor_algo.learner_group.get_weights([SHARED_POLICY_ID])
    finally:
        donor_algo.stop()
    recipient_weights = algo.learner_group.get_weights([SHARED_POLICY_ID])
    donor_module_weights = donor_weights.get(SHARED_POLICY_ID, {})
    recipient_module_weights = recipient_weights.get(SHARED_POLICY_ID, {})
    loaded_submodules: list[str] = []
    for key, value in donor_module_weights.items():
        if key == "motion_log_std" or key.startswith("encoder.") or key.startswith("motion_mean_head."):
            recipient_module_weights[key] = value.detach().clone() if hasattr(value, "detach") else value
            loaded_submodules.append(key)
    recipient_weights[SHARED_POLICY_ID] = recipient_module_weights
    algo.learner_group.set_weights(recipient_weights)
    algo.env_runner_group.sync_weights(
        from_worker_or_learner_group=algo.learner_group,
        policies=[SHARED_POLICY_ID],
        inference_only=True,
    )
    return loaded_submodules


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
    env_config = build_env_config(
        args,
        show_sensor_overlay=False,
        mute_received_messages=mute_received_messages,
        swim_assist_start_weight_override=0.0,
        motion_epsilon_override=0.0,
        message_epsilon_override=0.0,
    )
    return lambda: CommunicatingSchoolEnv(**env_config)


def motion_epsilon_for_iteration(iteration: int, args: argparse.Namespace) -> float:
    start = float(args.motion_epsilon_start)
    end = float(args.motion_epsilon_end)
    decay_iterations = max(int(args.motion_epsilon_decay_iterations), 1)
    if decay_iterations == 1 or iteration >= decay_iterations:
        return float(end)
    if iteration <= 1:
        return float(start)
    progress = float(iteration - 1) / float(decay_iterations - 1)
    return float(start + ((end - start) * progress))


def run_light_eval(algo, *, args: argparse.Namespace, base_seed: int) -> ColorCommEvalResult:
    return evaluate_multi_agent_rollouts(
        algo=algo,
        env_factory=make_eval_env_factory(args, mute_received_messages=False),
        num_episodes=args.light_eval_episodes,
        base_seed=base_seed,
        stack_mode=stack_mode_for_args(args),
        policy_id=SHARED_POLICY_ID,
    )


def initial_swim_assist_status(args: argparse.Namespace) -> dict[str, Any]:
    start_weight = float(args.swim_assist_start_weight) if effective_reward_mode(args) == "forage" else 0.0
    return {
        "enabled": bool(start_weight > 0.0),
        "start_weight": float(start_weight),
        "weight": float(start_weight),
        "state": "on" if start_weight > 0.0 else "off",
        "mastery_streak": 0,
        "gate_passed": False,
        "fade_step": 0,
        "disabled_iteration": None,
        "disabled_timestep": None,
    }


def serialize_swim_assist_status(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "swim_assist_enabled": bool(status["enabled"]),
        "swim_assist_state": str(status["state"]),
        "swim_assist_weight": float(status["weight"]),
        "swim_assist_start_weight": float(status["start_weight"]),
        "swim_mastery_streak": int(status["mastery_streak"]),
        "swim_mastery_gate_passed": bool(status["gate_passed"]),
        "swim_assist_disabled_iteration": status["disabled_iteration"],
        "swim_assist_disabled_timestep": status["disabled_timestep"],
    }


def swim_assist_gate_passed(light_result: ColorCommEvalResult, args: argparse.Namespace) -> bool:
    return bool(
        np.isfinite(light_result.mean_forward_velocity)
        and np.isfinite(light_result.mean_joint_limit_occupancy)
        and np.isfinite(light_result.fraction_negative_forward_velocity_steps)
        and light_result.mean_forward_velocity >= float(args.swim_assist_disable_forward_velocity)
        and light_result.mean_joint_limit_occupancy <= float(args.swim_assist_disable_joint_limit_occupancy)
        and light_result.fraction_negative_forward_velocity_steps <= float(args.swim_assist_disable_negative_forward_frac)
    )


def fade_swim_assist_weight(*, start_weight: float, fade_step: int, fade_evals: int) -> float:
    ratio = max(0.0, 1.0 - (float(fade_step) / float(max(fade_evals, 1))))
    return float(start_weight * ratio)


def advance_swim_assist_status(
    status: dict[str, Any],
    *,
    light_result: ColorCommEvalResult,
    args: argparse.Namespace,
    iteration: int,
    timesteps_total: int,
) -> bool:
    status["gate_passed"] = False
    if not status["enabled"] or status["state"] == "off":
        return False
    if status["state"] == "fading":
        status["fade_step"] = int(status["fade_step"]) + 1
        next_weight = fade_swim_assist_weight(
            start_weight=float(status["start_weight"]),
            fade_step=int(status["fade_step"]),
            fade_evals=int(args.swim_assist_fade_evals),
        )
        status["weight"] = float(next_weight)
        if next_weight <= 0.0:
            status["state"] = "off"
            status["disabled_iteration"] = int(iteration)
            status["disabled_timestep"] = int(timesteps_total)
        return True
    if iteration < int(args.swim_assist_min_iterations):
        status["mastery_streak"] = 0
        return False
    gate_passed = swim_assist_gate_passed(light_result, args)
    status["gate_passed"] = bool(gate_passed)
    status["mastery_streak"] = int(status["mastery_streak"]) + 1 if gate_passed else 0
    if int(status["mastery_streak"]) < int(args.swim_assist_disable_consecutive_evals):
        return False
    status["state"] = "fading"
    status["fade_step"] = 1
    status["weight"] = fade_swim_assist_weight(
        start_weight=float(status["start_weight"]),
        fade_step=int(status["fade_step"]),
        fade_evals=int(args.swim_assist_fade_evals),
    )
    if float(status["weight"]) <= 0.0:
        status["state"] = "off"
        status["disabled_iteration"] = int(iteration)
        status["disabled_timestep"] = int(timesteps_total)
    return True


def apply_swim_assist_weight(algo, weight: float) -> None:
    env_runner_group = getattr(algo, "env_runner_group", None)
    if env_runner_group is None:
        return

    def _update_worker(worker) -> None:
        foreach_env = getattr(worker, "foreach_env", None)
        if callable(foreach_env):
            foreach_env(lambda env: getattr(env, "set_swim_assist_weight", lambda _weight: None)(weight))

    foreach_worker = getattr(env_runner_group, "foreach_worker", None)
    if not callable(foreach_worker):
        return
    try:
        foreach_worker(_update_worker, local_worker=True)
    except TypeError:
        foreach_worker(_update_worker)


def apply_action_epsilons(algo, *, motion_epsilon: float, message_epsilon: float) -> None:
    env_runner_group = getattr(algo, "env_runner_group", None)
    if env_runner_group is None:
        return

    def _update_worker(worker) -> None:
        foreach_env = getattr(worker, "foreach_env", None)
        if not callable(foreach_env):
            return

        def _update_env(env) -> None:
            set_action_epsilons = getattr(env, "set_action_epsilons", None)
            if callable(set_action_epsilons):
                set_action_epsilons(
                    motion_epsilon=float(motion_epsilon),
                    message_epsilon=float(message_epsilon),
                )
                return
            getattr(env, "set_motion_epsilon", lambda _value: None)(float(motion_epsilon))
            getattr(env, "set_message_epsilon", lambda _value: None)(float(message_epsilon))

        foreach_env(_update_env)

    foreach_worker = getattr(env_runner_group, "foreach_worker", None)
    if not callable(foreach_worker):
        return
    try:
        foreach_worker(_update_worker, local_worker=True)
    except TypeError:
        foreach_worker(_update_worker)


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
    swim_assist_status: dict[str, Any],
    train_motion_epsilon: float,
    train_message_epsilon: float,
    training_status: str,
    failed_iteration: int | None,
    failure_message: str | None,
    failure_traceback: str | None,
    args: argparse.Namespace,
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
        "training_status": training_status,
        "failed_iteration": int(failed_iteration) if failed_iteration is not None else None,
        "failure_message": failure_message,
        "failure_traceback": failure_traceback,
        "train_motion_epsilon": float(train_motion_epsilon),
        "train_message_epsilon": float(train_message_epsilon),
        "eval_motion_epsilon": 0.0,
        "eval_message_epsilon": 0.0,
        "motion_epsilon_start": float(args.motion_epsilon_start),
        "motion_epsilon_end": float(args.motion_epsilon_end),
        "motion_epsilon_decay_iterations": int(args.motion_epsilon_decay_iterations),
        "policy_stack": str(args.policy_stack),
        "training_phase": str(args.training_phase),
        "message_head_mode": effective_message_head_mode(args),
        **serialize_swim_assist_status(swim_assist_status),
    }

def write_training_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    device: str,
    num_gpus: int,
    env_config: dict[str, Any],
    restore_target: str | None,
    warmstart_target: str | None,
    warmstart_loaded_submodules: list[str],
) -> None:
    model_config = (
        build_newstack_model_config(args)
        if str(args.policy_stack) == "new"
        else {
            "fcnet_hiddens": list(args.fcnet_hiddens),
            "fcnet_activation": str(args.fcnet_activation),
        }
    )
    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(device),
        "num_gpus": int(num_gpus),
        "policy_stack": str(args.policy_stack),
        "training_phase": str(args.training_phase),
        "message_head_mode": effective_message_head_mode(args),
        "restore_from_checkpoint": restore_target,
        "warmstart_parent_checkpoint": warmstart_target,
        "warmstart_loaded_submodules": list(warmstart_loaded_submodules),
        "env_config": env_config,
        "model_config": model_config,
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
            "motion_epsilon_start": float(args.motion_epsilon_start),
            "motion_epsilon_end": float(args.motion_epsilon_end),
            "motion_epsilon_decay_iterations": int(args.motion_epsilon_decay_iterations),
            "message_epsilon": float(args.message_epsilon),
            "swim_assist_start_weight": float(args.swim_assist_start_weight),
            "swim_assist_min_iterations": int(args.swim_assist_min_iterations),
            "swim_assist_disable_forward_velocity": float(args.swim_assist_disable_forward_velocity),
            "swim_assist_disable_joint_limit_occupancy": float(args.swim_assist_disable_joint_limit_occupancy),
            "swim_assist_disable_negative_forward_frac": float(args.swim_assist_disable_negative_forward_frac),
            "swim_assist_disable_consecutive_evals": int(args.swim_assist_disable_consecutive_evals),
            "swim_assist_fade_evals": int(args.swim_assist_fade_evals),
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
    warmstart_target = resolve_restore_target(args.warmstart_motion_checkpoint)
    if restore_target is not None:
        assert_checkpoint_segment_count(
            restore_target,
            expected_num_body_segments=int(args.num_body_segments),
            purpose="Restore",
        )
    if warmstart_target is not None:
        assert_checkpoint_segment_count(
            warmstart_target,
            expected_num_body_segments=int(args.num_body_segments),
            purpose="Warmstart",
        )
    eval_report_jsonl_path = checkpoint_root / "eval_reports.jsonl"
    eval_report_csv_path = checkpoint_root / "eval_reports.csv"
    run_summary_path = checkpoint_root / "run_summary.json"
    training_metadata_path = checkpoint_root / "training_metadata.json"
    local_fs = pafs.LocalFileSystem()

    register_env(ENV_ID, make_env)
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=True)

    initial_train_motion_epsilon = motion_epsilon_for_iteration(1, args)
    initial_train_message_epsilon = float(args.message_epsilon)
    env_config = build_env_config(
        args,
        show_sensor_overlay=False,
        mute_received_messages=False,
        motion_epsilon_override=initial_train_motion_epsilon,
        message_epsilon_override=initial_train_message_epsilon,
    )
    algo_config, algo = build_algo(
        args,
        num_gpus=num_gpus,
        env_config=env_config,
    )
    swim_assist_status = initial_swim_assist_status(args)
    current_train_motion_epsilon = float(initial_train_motion_epsilon)
    current_train_message_epsilon = float(initial_train_message_epsilon)
    session_dir = resolve_ray_session_dir()
    warmstart_loaded_submodules: list[str] = []
    if restore_target is not None:
        algo.restore(restore_target)
    elif warmstart_target is not None:
        warmstart_loaded_submodules = apply_motion_warmstart(
            algo,
            warmstart_target,
            expected_num_body_segments=int(args.num_body_segments),
        )
    apply_swim_assist_weight(algo, float(swim_assist_status["weight"]))
    apply_action_epsilons(
        algo,
        motion_epsilon=float(current_train_motion_epsilon),
        message_epsilon=float(current_train_message_epsilon),
    )
    write_training_metadata(
        training_metadata_path,
        args=args,
        device=device,
        num_gpus=num_gpus,
        env_config=env_config,
        restore_target=restore_target,
        warmstart_target=warmstart_target,
        warmstart_loaded_submodules=warmstart_loaded_submodules,
    )

    print("V9 - Muscle Activation Communication School RLlib training")
    print(
        "Config: "
        f"iterations={args.train_iterations}, "
        f"env_runners={args.num_env_runners}, "
        f"envs_per_runner={args.num_envs_per_runner}, "
        f"checkpoint_every={args.checkpoint_every_iterations}, "
        f"device={device}, num_gpus={num_gpus}, "
        f"checkpoint_root={checkpoint_root.resolve()}, "
        f"policy_stack={args.policy_stack}, "
        f"training_phase={args.training_phase}, "
        f"restore_from_checkpoint={restore_target or 'none'}, "
        f"warmstart_motion_checkpoint={warmstart_target or 'none'}, "
        f"num_red_fish={args.num_red_fish}, "
        f"num_blue_fish={args.num_blue_fish}, "
        f"num_red_pellets={args.num_red_pellets}, "
        f"num_blue_pellets={args.num_blue_pellets}, "
        f"time_limit={args.time_limit}, "
        f"reward_mode={effective_reward_mode(args)}, "
        f"observation_profile={args.observation_profile}, "
        f"history_length={args.history_length}, "
        f"activation_time_constant={args.activation_time_constant}, "
        f"joint_passive_stiffness={args.joint_passive_stiffness}, "
        f"joint_soft_limit_start_ratio={args.joint_soft_limit_start_ratio}, "
        f"joint_soft_limit_stiffness={args.joint_soft_limit_stiffness}, "
        f"joint_soft_limit_damping={args.joint_soft_limit_damping}, "
        f"body_linear_drag={args.body_linear_drag}, "
        f"propulsion_near_limit_weight={args.propulsion_near_limit_weight}, "
        f"propulsion_saturation_weight={args.propulsion_saturation_weight}, "
        f"propulsion_torque_weight={args.propulsion_torque_weight}, "
        f"motion_epsilon_start={args.motion_epsilon_start}, "
        f"motion_epsilon_end={args.motion_epsilon_end}, "
        f"motion_epsilon_decay_iterations={args.motion_epsilon_decay_iterations}, "
        f"message_epsilon={args.message_epsilon}, "
        f"swim_assist_start_weight={args.swim_assist_start_weight}, "
        f"swim_assist_min_iterations={args.swim_assist_min_iterations}, "
        f"swim_assist_disable_forward_velocity={args.swim_assist_disable_forward_velocity}, "
        f"swim_assist_disable_joint_limit_occupancy={args.swim_assist_disable_joint_limit_occupancy}, "
        f"swim_assist_disable_negative_forward_frac={args.swim_assist_disable_negative_forward_frac}, "
        f"swim_assist_disable_consecutive_evals={args.swim_assist_disable_consecutive_evals}, "
        f"swim_assist_fade_evals={args.swim_assist_fade_evals}, "
        f"pellet_reward={args.pellet_reward}, "
        f"step_cost={args.step_cost}, "
        f"food_respawn_mode={args.food_respawn_mode}, "
        f"forage_timeout_mode={args.forage_timeout_mode}, "
        f"forage_idle_timeout_steps={args.forage_idle_timeout_steps}, "
        f"forage_time_context_mode={args.forage_time_context_mode}, "
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
    training_status = "running"
    failed_iteration: int | None = None
    failure_message: str | None = None
    failure_traceback: str | None = None
    failure_exc: Exception | None = None

    def write_current_summary(*, final_checkpoint_path: str | None) -> None:
        summary = build_run_summary(
            best_checkpoint_record=best_checkpoint_record,
            best_eval_result=best_eval_result,
            num_checkpoint_evaluations=len(report_rows_nested),
            time_to_first_positive_total_reward=time_to_first_positive_total_reward,
            final_checkpoint_path=final_checkpoint_path,
            checkpoint_eval_mode="light_pure",
            light_eval_episodes=args.light_eval_episodes,
            swim_assist_status=swim_assist_status,
            train_motion_epsilon=float(current_train_motion_epsilon),
            train_message_epsilon=float(current_train_message_epsilon),
            training_status=training_status,
            failed_iteration=failed_iteration if training_status == "failed_exception" else None,
            failure_message=failure_message,
            failure_traceback=failure_traceback,
            args=args,
        )
        run_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    try:
        for iteration in range(1, args.train_iterations + 1):
            failed_iteration = int(iteration)
            current_train_motion_epsilon = motion_epsilon_for_iteration(iteration, args)
            current_train_message_epsilon = float(args.message_epsilon)
            apply_action_epsilons(
                algo,
                motion_epsilon=float(current_train_motion_epsilon),
                message_epsilon=float(current_train_message_epsilon),
            )
            print(
                f"iter={iteration:03d} "
                f"train_call_start "
                f"train_motion_epsilon={format_metric(current_train_motion_epsilon)} "
                f"train_message_epsilon={format_metric(current_train_message_epsilon)}"
            )
            train_wall_start = time.perf_counter()
            result = algo.train()
            train_loop_wall_ms = (time.perf_counter() - train_wall_start) * 1000.0
            print(
                f"iter={iteration:03d} "
                f"train_call_done "
                f"train_loop_wall_ms={format_metric(train_loop_wall_ms, precision=1)}"
            )
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
                f"train_motion_epsilon={format_metric(current_train_motion_epsilon)} "
                f"train_message_epsilon={format_metric(current_train_message_epsilon)} "
                f"train_assist_state={swim_assist_status['state']} "
                f"train_assist_weight={format_metric(float(swim_assist_status['weight']))} "
                f"sample_time_ms={format_metric(sample_time_ms, precision=1)} "
                f"learner_time_ms={format_metric(learner_time_ms, precision=1)} "
                f"train_loop_wall_ms={format_metric(train_loop_wall_ms, precision=1)}"
            )

            if iteration % args.checkpoint_every_iterations != 0:
                continue

            checkpoint_path = checkpoint_root / f"checkpoint_{iteration:05d}"
            latest_checkpoint = save_algorithm_checkpoint(algo, checkpoint_path, local_fs)
            print(f"checkpoint_saved: {latest_checkpoint}")

            train_swim_assist_state = str(swim_assist_status["state"])
            train_swim_assist_weight = float(swim_assist_status["weight"])

            light_eval_start = time.perf_counter()
            light_result = run_light_eval(
                algo,
                args=args,
                base_seed=args.eval_report_seed + (iteration * 1_000_000),
            )
            light_eval_wall_ms = (time.perf_counter() - light_eval_start) * 1000.0

            swim_state_changed = advance_swim_assist_status(
                swim_assist_status,
                light_result=light_result,
                args=args,
                iteration=iteration,
                timesteps_total=env_steps_total,
            )
            if swim_state_changed:
                apply_swim_assist_weight(algo, float(swim_assist_status["weight"]))
                print(
                    "swim_assist_update: "
                    f"iter={iteration:03d} "
                    f"next_state={swim_assist_status['state']} "
                    f"next_weight={format_metric(float(swim_assist_status['weight']))} "
                    f"mastery_streak={int(swim_assist_status['mastery_streak'])}"
                )

            swim_status_flat = serialize_swim_assist_status(swim_assist_status)
            report_nested = {
                "iteration": int(iteration),
                "timesteps_total": int(env_steps_total),
                "checkpoint_path": str(latest_checkpoint),
                "training_reward_mean": float(reward_mean),
                "policy_stack": str(args.policy_stack),
                "training_phase": str(args.training_phase),
                "message_head_mode": effective_message_head_mode(args),
                "training_reward_mode": "forage_plus_swim_assist" if train_swim_assist_weight > 0.0 else effective_reward_mode(args),
                "train_motion_epsilon": float(current_train_motion_epsilon),
                "train_message_epsilon": float(current_train_message_epsilon),
                "eval_motion_epsilon": 0.0,
                "eval_message_epsilon": 0.0,
                "train_swim_assist_state": train_swim_assist_state,
                "train_swim_assist_weight": float(train_swim_assist_weight),
                "eval_mode": "light_pure",
                "light_eval_episodes": int(args.light_eval_episodes),
                "swim_assist": dict(swim_status_flat),
                "eval_result": light_result.to_dict(),
            }
            report_flat = {
                "iteration": int(iteration),
                "timesteps_total": int(env_steps_total),
                "checkpoint_path": str(latest_checkpoint),
                "training_reward_mean": float(reward_mean),
                "policy_stack": str(args.policy_stack),
                "training_phase": str(args.training_phase),
                "message_head_mode": effective_message_head_mode(args),
                "training_reward_mode": "forage_plus_swim_assist" if train_swim_assist_weight > 0.0 else effective_reward_mode(args),
                "train_motion_epsilon": float(current_train_motion_epsilon),
                "train_message_epsilon": float(current_train_message_epsilon),
                "eval_motion_epsilon": 0.0,
                "eval_message_epsilon": 0.0,
                "train_swim_assist_state": train_swim_assist_state,
                "train_swim_assist_weight": float(train_swim_assist_weight),
                "eval_mode": "light_pure",
                "light_eval_episodes": int(args.light_eval_episodes),
                **swim_status_flat,
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
                    "eval_mode": "light_pure",
                    "light_eval_episodes": int(args.light_eval_episodes),
                    "policy_stack": str(args.policy_stack),
                    "training_phase": str(args.training_phase),
                    "message_head_mode": effective_message_head_mode(args),
                    "train_motion_epsilon": float(current_train_motion_epsilon),
                    "train_message_epsilon": float(current_train_message_epsilon),
                    "eval_motion_epsilon": 0.0,
                    "eval_message_epsilon": 0.0,
                    **swim_status_flat,
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

            write_current_summary(final_checkpoint_path=None)

            print(
                "light_eval_report: "
                f"iter={iteration:03d} "
                f"train_motion_epsilon={format_metric(current_train_motion_epsilon)} "
                f"train_message_epsilon={format_metric(current_train_message_epsilon)} "
                f"train_assist_weight={format_metric(train_swim_assist_weight)} "
                f"next_assist_state={swim_assist_status['state']} "
                f"next_assist_weight={format_metric(float(swim_assist_status['weight']))} "
                f"mean_total_reward={format_metric(light_result.mean_total_reward)} "
                f"mean_pellets_per_fish={format_metric(light_result.mean_pellets_per_fish)} "
                f"red_food={format_metric(light_result.mean_pellets_red_eaten_by_red)} "
                f"blue_food={format_metric(light_result.mean_pellets_blue_eaten_by_blue)} "
                f"mean_forward_velocity={format_metric(light_result.mean_forward_velocity)} "
                f"mean_abs_activation={format_metric(light_result.mean_abs_activation)} "
                f"limit_occ={format_metric(light_result.mean_joint_limit_occupancy)} "
                f"near_limit_pen={format_metric(light_result.mean_near_limit_penalty)} "
                f"sat_frac={format_metric(light_result.fraction_saturated_motion_commands)} "
                f"joint_limit_high_frac={format_metric(light_result.fraction_joint_limit_high_steps)} "
                f"joints_quiet_frac={format_metric(light_result.fraction_joints_quiet_steps)} "
                f"neg_fwd_frac={format_metric(light_result.fraction_negative_forward_velocity_steps)} "
                f"mastery_streak={int(swim_assist_status['mastery_streak'])} "
                f"gate_passed={bool(swim_assist_status['gate_passed'])} "
                f"activation_sign_changes={format_metric(light_result.mean_activation_sign_changes_per_fish)} "
                f"light_eval_wall_ms={format_metric(light_eval_wall_ms, precision=1)} "
                f"jsonl={eval_report_jsonl_path.name}"
            )
        training_status = "reached_iteration_budget"
    except Exception as exc:
        training_status = "failed_exception"
        failure_exc = exc
        failure_message = f"{type(exc).__name__}: {exc}"
        failure_traceback = traceback.format_exc()
        print(
            "training_exception: "
            f"iter={failed_iteration if failed_iteration is not None else 'na'} "
            f"message={failure_message}",
            file=sys.stderr,
        )
        try:
            write_current_summary(final_checkpoint_path=None)
        except Exception as summary_exc:
            print(
                "run_summary_write_failed_during_exception: "
                f"{type(summary_exc).__name__}: {summary_exc}",
                file=sys.stderr,
            )
    finally:
        final_checkpoint_path = checkpoint_root / "checkpoint_final"
        final_checkpoint_path_str: str | None = None
        try:
            latest_checkpoint = save_algorithm_checkpoint(algo, final_checkpoint_path, local_fs)
            final_checkpoint_path_str = str(final_checkpoint_path.resolve())
            print(f"final_checkpoint_saved: {latest_checkpoint}")
        except Exception as checkpoint_exc:
            if failure_exc is None:
                training_status = "failed_exception"
                failure_exc = checkpoint_exc
                failure_message = f"{type(checkpoint_exc).__name__}: {checkpoint_exc}"
                failure_traceback = traceback.format_exc()
                print(
                    "final_checkpoint_exception: "
                    f"iter={failed_iteration if failed_iteration is not None else 'na'} "
                    f"message={failure_message}",
                    file=sys.stderr,
                )
        finally:
            try:
                write_current_summary(final_checkpoint_path=final_checkpoint_path_str)
            except Exception as summary_exc:
                print(
                    "run_summary_write_failed: "
                    f"{type(summary_exc).__name__}: {summary_exc}",
                    file=sys.stderr,
                )
            print(f"training_status: {training_status}")
            try:
                algo.stop()
            except Exception as stop_exc:
                print(f"algo_stop_failed: {type(stop_exc).__name__}: {stop_exc}", file=sys.stderr)
            try:
                ray.shutdown()
            except Exception as shutdown_exc:
                print(f"ray_shutdown_failed: {type(shutdown_exc).__name__}: {shutdown_exc}", file=sys.stderr)

    if failure_exc is not None:
        raise failure_exc


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


