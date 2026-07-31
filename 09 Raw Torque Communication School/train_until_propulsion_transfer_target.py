"""Run a warm-start V9 thrust-retention transfer campaign."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import math
import os
import traceback
from typing import Any

import train_until_propulsion_target as base


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DONOR_CHECKPOINT = (
    SCRIPT_DIR
    / "rllib_checkpoints_v9_day_campaign_limitpressure"
    / "a1_initial_lp_early_soft"
    / "checkpoint_00060"
)
DEFAULT_MANIFEST_PATH = SCRIPT_DIR / "propulsion_transfer_campaign_manifest.json"
DEFAULT_TARGET_ROOT = SCRIPT_DIR / "rllib_checkpoints_v9_transfer_campaign"
DEFAULT_CONTROLLER_POLL_SECONDS = 30
R1_CONFIRM_EPISODES = 10
R2_CONFIRM_EPISODES = 10
FORAGE_CONFIRM_EPISODES = 10
RANDOM_BASELINE_EPISODES = 20
TERMINAL_STATUSES = base.TERMINAL_STATUSES
PHASE_COMPLETE = base.PHASE_COMPLETE
PHASE_CONFIRM_FAILED = base.PHASE_CONFIRM_FAILED
TERMINAL_PHASE_STATUSES = base.TERMINAL_PHASE_STATUSES


@dataclass(frozen=True)
class R1Branch:
    family: base.FamilySpec
    train_iterations: int


@dataclass(frozen=True)
class PhaseBudget:
    default_hours: float
    buffer_hours: float


@dataclass(frozen=True)
class ForageRunSpec:
    checkpoint_root: Path
    restore_checkpoint: str | None
    warmstart_checkpoint: str | None
    train_iterations: int
    learning_rate: float
    entropy_coeff: float


def default_python_executable() -> str:
    return base.default_python_executable()


def now_local() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def now_iso() -> str:
    return now_local().isoformat(timespec="seconds")


def log(message: str) -> None:
    base.log(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the unattended V9 warm-start thrust-retention campaign.")
    parser.add_argument("--python-executable", type=str, default=default_python_executable())
    parser.add_argument("--donor-checkpoint", type=str, default=str(DEFAULT_DONOR_CHECKPOINT))
    parser.add_argument("--manifest-path", type=str, default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--target-root", type=str, default=str(DEFAULT_TARGET_ROOT))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-device", type=str, default="cuda")
    parser.add_argument("--max-wall-clock-hours", type=float, default=8.0)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_CONTROLLER_POLL_SECONDS)
    return parser.parse_args()


def init_manifest(args: argparse.Namespace, *, manifest_path: Path, target_root: Path) -> dict[str, Any]:
    started_at = now_local()
    return {
        "status": "running",
        "stop_reason": None,
        "campaign_kind": "propulsion_transfer",
        "manifest_version": 1,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": None,
        "updated_at": started_at.isoformat(timespec="seconds"),
        "deadline_at": (started_at + timedelta(hours=float(args.max_wall_clock_hours))).isoformat(timespec="seconds"),
        "manifest_path": str(manifest_path.resolve()),
        "target_root": str(target_root.resolve()),
        "donor_checkpoint": str(Path(args.donor_checkpoint).resolve()),
        "smoke": bool(args.smoke),
        "controller_pid": int(os.getpid()),
        "controller_heartbeat_at": started_at.isoformat(timespec="seconds"),
        "active_phase_id": None,
        "last_resume_recovery_at": None,
        "next_phase_index": 1,
        "phases": [],
        "artifacts": {},
    }


def ensure_resume_allowed(args: argparse.Namespace, manifest_path: Path, target_root: Path) -> dict[str, Any]:
    existing_manifest = base.try_load_json(manifest_path)
    if existing_manifest is not None:
        if not args.resume_existing:
            raise FileExistsError(
                f"Manifest already exists at {manifest_path}. Use --resume-existing or choose a different target root."
            )
        return existing_manifest
    if args.resume_existing and target_root.exists():
        raise FileNotFoundError(f"--resume-existing requested but manifest not found: {manifest_path}")
    target_root.mkdir(parents=True, exist_ok=True)
    manifest = init_manifest(args, manifest_path=manifest_path, target_root=target_root)
    base.save_manifest(manifest_path, manifest)
    return manifest


def r1_branch_specs() -> list[R1Branch]:
    return [
        R1Branch(base.FamilySpec("hold20", 0.65, 18.0, 2.5, 0.12, -0.22, -0.10, -0.05), 20),
        R1Branch(base.FamilySpec("limit_light", 0.65, 20.0, 2.75, 0.12, -0.23, -0.11, -0.055), 40),
        R1Branch(base.FamilySpec("limit_medium", 0.65, 22.0, 3.0, 0.12, -0.24, -0.12, -0.06), 40),
        R1Branch(base.FamilySpec("start_ratio_early", 0.63, 22.0, 3.0, 0.12, -0.24, -0.11, -0.055), 40),
        R1Branch(base.FamilySpec("torque_focus", 0.65, 18.0, 2.5, 0.12, -0.22, -0.10, -0.07), 40),
        R1Branch(base.FamilySpec("smooth_medium", 0.65, 20.0, 2.75, 0.14, -0.23, -0.12, -0.06), 40),
    ]


def smoke_branch() -> R1Branch:
    return r1_branch_specs()[0]

def phase_budget_for_kind(phase_kind: str) -> PhaseBudget:
    if phase_kind == "donor_confirm":
        return PhaseBudget(default_hours=0.20, buffer_hours=0.10)
    if phase_kind == "baseline_random":
        return PhaseBudget(default_hours=0.30, buffer_hours=0.10)
    if phase_kind.startswith("r1"):
        return PhaseBudget(default_hours=0.75, buffer_hours=0.25)
    if phase_kind.startswith("r2"):
        return PhaseBudget(default_hours=0.75, buffer_hours=0.25)
    if phase_kind.startswith("forage"):
        return PhaseBudget(default_hours=1.50, buffer_hours=0.25)
    return PhaseBudget(default_hours=0.50, buffer_hours=0.25)


def _phase_duration_hours(phase: dict[str, Any]) -> float | None:
    started_at = phase.get("started_at")
    finished_at = phase.get("finished_at")
    if not started_at or not finished_at:
        return None
    try:
        start_dt = datetime.fromisoformat(str(started_at))
        end_dt = datetime.fromisoformat(str(finished_at))
    except ValueError:
        return None
    hours = (end_dt - start_dt).total_seconds() / 3600.0
    if hours <= 0.0:
        return None
    return hours


def estimated_phase_hours(manifest: dict[str, Any], phase_kind: str) -> float:
    budget = phase_budget_for_kind(phase_kind)
    observed: list[float] = []
    for phase in manifest.get("phases", []):
        if str(phase.get("phase_kind")) != str(phase_kind):
            continue
        duration = _phase_duration_hours(phase)
        if duration is not None:
            observed.append(duration)
    if not observed:
        return budget.default_hours
    observed.sort()
    return max(budget.default_hours, observed[len(observed) // 2])


def can_start_phase(manifest: dict[str, Any], phase_kind: str) -> bool:
    needed_hours = estimated_phase_hours(manifest, phase_kind) + phase_budget_for_kind(phase_kind).buffer_hours
    return base.remaining_hours(manifest) >= needed_hours


def phase_root(target_root: Path, phase_kind: str, family_id: str) -> Path:
    return target_root / f"{phase_kind}_{family_id}"


def metric_float(mapping: dict[str, Any] | None, key: str, default: float) -> float:
    if not mapping:
        return float(default)
    value = mapping.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return number


def training_phase_complete_and_unconfirmed(phase: dict[str, Any]) -> bool:
    if str(phase.get("terminal_phase_status")) != PHASE_COMPLETE:
        return False
    if phase.get("promotion_decision") is not None:
        return False
    return str(phase.get("phase_kind")) in {
        "r1_branch",
        "r2_initial",
        "r2_continue",
        "smoke_r1",
        "smoke_r2",
        "forage_initial",
        "forage_continue",
    }


def donor_confirm_pending(phase: dict[str, Any]) -> bool:
    return str(phase.get("phase_kind")) == "donor_confirm" and phase.get("terminal_phase_status") not in TERMINAL_PHASE_STATUSES


def baseline_pending(phase: dict[str, Any]) -> bool:
    return str(phase.get("phase_kind")) == "baseline_random" and phase.get("terminal_phase_status") not in TERMINAL_PHASE_STATUSES


def donor_metrics(manifest: dict[str, Any]) -> dict[str, Any] | None:
    payload = manifest.get("artifacts", {}).get("donor_confirm_primary_metrics")
    if not isinstance(payload, dict):
        return None
    return base.normalize_metric_dict(payload)


def compute_transfer_metrics(*, donor: dict[str, Any], branch: dict[str, Any]) -> dict[str, Any]:
    donor_forward = metric_float(donor, "mean_forward_velocity", float("nan"))
    branch_forward = metric_float(branch, "mean_forward_velocity", float("nan"))
    if math.isfinite(donor_forward) and donor_forward > 1e-9 and math.isfinite(branch_forward):
        forward_retention = branch_forward / donor_forward
    else:
        forward_retention = float("nan")
    donor_limit = metric_float(donor, "mean_joint_limit_occupancy", float("nan"))
    branch_limit = metric_float(branch, "mean_joint_limit_occupancy", float("nan"))
    donor_sat = metric_float(donor, "fraction_saturated_motion_commands", float("nan"))
    branch_sat = metric_float(branch, "fraction_saturated_motion_commands", float("nan"))
    limit_improvement = donor_limit - branch_limit if math.isfinite(donor_limit) and math.isfinite(branch_limit) else float("nan")
    saturation_improvement = donor_sat - branch_sat if math.isfinite(donor_sat) and math.isfinite(branch_sat) else float("nan")
    return {
        "forward_retention": forward_retention,
        "limit_improvement": limit_improvement,
        "saturation_improvement": saturation_improvement,
    }


def r1_gate_class(metrics: dict[str, Any], transfer: dict[str, Any]) -> str:
    joint_high = metric_float(metrics, "fraction_joint_limit_high_steps", 1.0)
    if (
        metric_float(transfer, "forward_retention", -1.0) >= 0.70
        and metric_float(metrics, "mean_forward_velocity", 0.0) >= 0.10
        and metric_float(metrics, "mean_abs_activation", 0.0) >= 0.10
        and metric_float(transfer, "limit_improvement", -1.0) >= 0.03
        and metric_float(transfer, "saturation_improvement", -1.0) >= 0.05
        and metric_float(metrics, "fraction_negative_forward_velocity_steps", 1.0) < 0.20
        and joint_high <= 0.0
    ):
        return "full_pass"
    if (
        metric_float(transfer, "forward_retention", -1.0) >= 0.60
        and metric_float(metrics, "mean_forward_velocity", 0.0) >= 0.08
        and metric_float(metrics, "mean_abs_activation", 0.0) >= 0.08
        and metric_float(transfer, "limit_improvement", -1.0) >= 0.01
        and metric_float(transfer, "saturation_improvement", -1.0) >= 0.02
        and metric_float(metrics, "fraction_negative_forward_velocity_steps", 1.0) < 0.25
        and joint_high <= 0.0
    ):
        return "near_pass"
    return "fail"


def r2_gate_class(metrics: dict[str, Any]) -> str:
    joint_high = metric_float(metrics, "fraction_joint_limit_high_steps", 1.0)
    if (
        metric_float(metrics, "mean_forward_velocity", 0.0) >= 0.06
        and metric_float(metrics, "mean_abs_activation", 0.0) >= 0.08
        and metric_float(metrics, "fraction_joints_quiet_steps", 1.0) < 0.20
        and metric_float(metrics, "fraction_negative_forward_velocity_steps", 1.0) < 0.35
        and metric_float(metrics, "fraction_saturated_motion_commands", 1.0) < 0.55
        and metric_float(metrics, "mean_joint_limit_occupancy", 1.0) < 0.38
        and joint_high <= 0.0
    ):
        return "full_pass"
    if (
        metric_float(metrics, "mean_forward_velocity", 0.0) >= 0.05
        and metric_float(metrics, "mean_abs_activation", 0.0) >= 0.08
        and metric_float(metrics, "fraction_joints_quiet_steps", 1.0) < 0.25
        and metric_float(metrics, "fraction_negative_forward_velocity_steps", 1.0) < 0.40
        and metric_float(metrics, "fraction_saturated_motion_commands", 1.0) < 0.58
        and metric_float(metrics, "mean_joint_limit_occupancy", 1.0) < 0.40
        and joint_high <= 0.0
    ):
        return "near_pass"
    return "fail"

def r1_rank_key(record: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = base.normalize_metric_dict(record.get("confirm_primary_metrics"))
    transfer = base.normalize_metric_dict(record.get("transfer_metrics"))
    return (
        metric_float(transfer, "limit_improvement", float("-inf")),
        metric_float(transfer, "saturation_improvement", float("-inf")),
        -metric_float(metrics, "fraction_negative_forward_velocity_steps", float("inf")),
        metric_float(transfer, "forward_retention", float("-inf")),
    )


def r2_rank_key(record: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = base.normalize_metric_dict(record.get("confirm_primary_metrics"))
    return (
        -metric_float(metrics, "mean_joint_limit_occupancy", float("inf")),
        -metric_float(metrics, "fraction_saturated_motion_commands", float("inf")),
        -metric_float(metrics, "fraction_negative_forward_velocity_steps", float("inf")),
        metric_float(metrics, "mean_forward_velocity", float("-inf")),
    )


def choose_best_r1(records: list[dict[str, Any]]) -> dict[str, Any]:
    return max(records, key=r1_rank_key)


def choose_best_r2(records: list[dict[str, Any]]) -> dict[str, Any]:
    return max(records, key=r2_rank_key)


def build_common_training_params(args: argparse.Namespace, family: base.FamilySpec, checkpoint_root: Path) -> dict[str, Any]:
    return {
        "policy_stack": "new",
        "device": str(args.device),
        "checkpoint_root": str(checkpoint_root.resolve()),
        "checkpoint_every_iterations": 10,
        "num_env_runners": 8,
        "num_envs_per_runner": 2,
        "light_eval_episodes": 4,
        "train_batch_size": 16000,
        "minibatch_size": 2048,
        "num_epochs": 6,
        "rollout_fragment_length": 500,
        "gamma": 0.97,
        "gae_lambda": 0.97,
        "fcnet_hiddens": "512,512,256",
        "fcnet_activation": "tanh",
        "reward_mode": "locomotion_debug",
        "observation_profile": "full_v9",
        "history_length": 8,
        "activation_time_constant": float(family.activation_time_constant),
        "joint_passive_stiffness": 10.0,
        "joint_soft_limit_start_ratio": float(family.joint_soft_limit_start_ratio),
        "joint_soft_limit_stiffness": float(family.joint_soft_limit_stiffness),
        "joint_soft_limit_damping": float(family.joint_soft_limit_damping),
        "body_linear_drag": 1.0,
        "propulsion_near_limit_weight": float(family.propulsion_near_limit_weight),
        "propulsion_saturation_weight": float(family.propulsion_saturation_weight),
        "propulsion_torque_weight": float(family.propulsion_torque_weight),
        "motion_epsilon_start": 0.0,
        "motion_epsilon_end": 0.0,
        "motion_epsilon_decay_iterations": 1,
        "message_epsilon": 0.0,
        "seed": 0,
        "num_red_fish": 1,
        "num_blue_fish": 0,
        "num_red_pellets": 0,
        "num_blue_pellets": 0,
    }


def build_r1_training_params(args: argparse.Namespace, family: base.FamilySpec, checkpoint_root: Path, *, warmstart_checkpoint: str, train_iterations: int) -> dict[str, Any]:
    params = build_common_training_params(args, family, checkpoint_root)
    params.update(
        {
            "training_phase": "locomotion_propulsion_easy",
            "train_iterations": int(train_iterations),
            "learning_rate": 1e-4,
            "entropy_coeff": 0.005,
            "time_limit": 150,
            "warmstart_motion_checkpoint": str(warmstart_checkpoint),
        }
    )
    return params


def build_r2_training_params(args: argparse.Namespace, family: base.FamilySpec, checkpoint_root: Path, *, warmstart_checkpoint: str) -> dict[str, Any]:
    params = build_common_training_params(args, family, checkpoint_root)
    params.update(
        {
            "training_phase": "locomotion_propulsion_robust",
            "train_iterations": 40,
            "learning_rate": 1.5e-4,
            "entropy_coeff": 0.01,
            "time_limit": 300,
            "warmstart_motion_checkpoint": str(warmstart_checkpoint),
        }
    )
    return params


def build_r2_continuation_params(args: argparse.Namespace, family: base.FamilySpec, checkpoint_root: Path, *, restore_checkpoint: str) -> dict[str, Any]:
    params = build_common_training_params(args, family, checkpoint_root)
    params.update(
        {
            "training_phase": "locomotion_propulsion_robust",
            "train_iterations": 20,
            "learning_rate": 7.5e-5,
            "entropy_coeff": 0.003,
            "time_limit": 300,
            "restore_from_checkpoint": str(restore_checkpoint),
        }
    )
    return params


def build_forage_training_params(args: argparse.Namespace, family: base.FamilySpec, spec: ForageRunSpec) -> dict[str, Any]:
    params = {
        "policy_stack": "new",
        "training_phase": "forage_full",
        "device": str(args.device),
        "checkpoint_root": str(spec.checkpoint_root.resolve()),
        "train_iterations": int(spec.train_iterations),
        "checkpoint_every_iterations": 20,
        "num_env_runners": 8,
        "num_envs_per_runner": 2,
        "light_eval_episodes": 4,
        "train_batch_size": 16000,
        "minibatch_size": 2048,
        "num_epochs": 6,
        "rollout_fragment_length": 500,
        "learning_rate": float(spec.learning_rate),
        "entropy_coeff": float(spec.entropy_coeff),
        "gamma": 0.97,
        "gae_lambda": 0.97,
        "fcnet_hiddens": "512,512,256",
        "fcnet_activation": "tanh",
        "time_limit": 300,
        "reward_mode": "forage",
        "observation_profile": "full_v9",
        "history_length": 8,
        "activation_time_constant": float(family.activation_time_constant),
        "joint_passive_stiffness": 10.0,
        "joint_soft_limit_start_ratio": float(family.joint_soft_limit_start_ratio),
        "joint_soft_limit_stiffness": float(family.joint_soft_limit_stiffness),
        "joint_soft_limit_damping": float(family.joint_soft_limit_damping),
        "body_linear_drag": 1.0,
        "propulsion_near_limit_weight": float(family.propulsion_near_limit_weight),
        "propulsion_saturation_weight": float(family.propulsion_saturation_weight),
        "propulsion_torque_weight": float(family.propulsion_torque_weight),
        "motion_epsilon_start": 0.0,
        "motion_epsilon_end": 0.0,
        "motion_epsilon_decay_iterations": 1,
        "message_epsilon": 0.0,
        "seed": 0,
        "num_red_fish": 10,
        "num_blue_fish": 10,
        "num_red_pellets": 48,
        "num_blue_pellets": 48,
    }
    if spec.restore_checkpoint is not None:
        params["restore_from_checkpoint"] = str(spec.restore_checkpoint)
    elif spec.warmstart_checkpoint is not None:
        params["warmstart_motion_checkpoint"] = str(spec.warmstart_checkpoint)
    return params

def build_smoke_r1_params(args: argparse.Namespace, family: base.FamilySpec, checkpoint_root: Path, *, donor_checkpoint: str) -> dict[str, Any]:
    params = build_r1_training_params(args, family, checkpoint_root, warmstart_checkpoint=donor_checkpoint, train_iterations=1)
    params.update(
        {
            "device": "cpu",
            "checkpoint_every_iterations": 1,
            "num_env_runners": 1,
            "num_envs_per_runner": 1,
            "light_eval_episodes": 1,
            "train_batch_size": 250,
            "minibatch_size": 125,
            "num_epochs": 1,
            "rollout_fragment_length": 50,
            "time_limit": 60,
        }
    )
    return params


def build_smoke_r2_params(args: argparse.Namespace, family: base.FamilySpec, checkpoint_root: Path, *, warmstart_checkpoint: str) -> dict[str, Any]:
    params = build_r2_training_params(args, family, checkpoint_root, warmstart_checkpoint=warmstart_checkpoint)
    params.update(
        {
            "device": "cpu",
            "train_iterations": 1,
            "checkpoint_every_iterations": 1,
            "num_env_runners": 1,
            "num_envs_per_runner": 1,
            "light_eval_episodes": 1,
            "train_batch_size": 250,
            "minibatch_size": 125,
            "num_epochs": 1,
            "rollout_fragment_length": 50,
            "time_limit": 60,
        }
    )
    return params


def best_checkpoint_from_phase(phase: dict[str, Any]) -> str:
    checkpoint_path = phase.get("best_checkpoint_path") or phase.get("final_checkpoint_path")
    if not checkpoint_path:
        raise RuntimeError(f"Phase {phase['phase_id']} is missing best_checkpoint_path")
    return str(checkpoint_path)


def donor_phase_record(manifest: dict[str, Any]) -> dict[str, Any] | None:
    records = [phase for phase in manifest.get("phases", []) if str(phase.get("phase_kind")) == "donor_confirm"]
    return records[-1] if records else None


def pending_postprocess_phase(manifest: dict[str, Any]) -> dict[str, Any] | None:
    for phase in reversed(list(manifest.get("phases", []))):
        if training_phase_complete_and_unconfirmed(phase):
            return phase
        if donor_confirm_pending(phase):
            return phase
        if baseline_pending(phase):
            return phase
    return None


def finish_donor_confirm(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any], phase: dict[str, Any]) -> None:
    checkpoint_path = str(manifest["donor_checkpoint"])
    output_root = Path(phase["checkpoint_root"])
    phase["phase_status"] = "running"
    phase["started_at"] = phase.get("started_at") or now_iso()
    base.save_manifest(manifest_path, manifest)
    try:
        confirm = base.run_summary_confirm_eval(
            args,
            checkpoint_path=checkpoint_path,
            output_root=output_root,
            stem="donor_confirm_eval",
            episodes=R1_CONFIRM_EPISODES,
            action_selection="deterministic",
        )
        phase["confirm_eval"] = confirm["summary"]
        phase["confirm_primary_metrics"] = confirm["primary_metrics"]
        phase["donor_checkpoint"] = checkpoint_path
        phase["promotion_decision"] = "donor_confirmed"
        phase["phase_status"] = PHASE_COMPLETE
        phase["terminal_phase_status"] = PHASE_COMPLETE
        manifest.setdefault("artifacts", {})["donor_confirm_summary"] = confirm["summary"]
        manifest["artifacts"]["donor_confirm_primary_metrics"] = confirm["primary_metrics"]
        manifest["artifacts"]["donor_confirm_checkpoint"] = checkpoint_path
    except Exception as exc:
        phase["phase_status"] = PHASE_CONFIRM_FAILED
        phase["terminal_phase_status"] = PHASE_CONFIRM_FAILED
        phase["promotion_decision"] = PHASE_CONFIRM_FAILED
        phase["failure_message"] = f"donor_confirm_failed: {exc}"
        base.stop_campaign(manifest, status="failed", stop_reason="donor_confirm_failed")
    phase["finished_at"] = now_iso()
    base.save_manifest(manifest_path, manifest)


def finish_r1_phase(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any], phase: dict[str, Any]) -> None:
    donor = donor_metrics(manifest)
    if donor is None:
        raise RuntimeError("Donor metrics missing before R1 confirm.")
    checkpoint_root = Path(phase["checkpoint_root"])
    confirm = base.run_summary_confirm_eval(
        args,
        checkpoint_path=best_checkpoint_from_phase(phase),
        output_root=checkpoint_root,
        stem="confirm_eval",
        episodes=R1_CONFIRM_EPISODES if str(phase.get("phase_kind")) == "r1_branch" else 1,
        action_selection="deterministic",
    )
    transfer = compute_transfer_metrics(donor=donor, branch=confirm["primary_metrics"])
    phase["confirm_eval"] = confirm["summary"]
    phase["confirm_primary_metrics"] = confirm["primary_metrics"]
    phase["transfer_metrics"] = transfer
    phase["donor_checkpoint"] = str(manifest["donor_checkpoint"])
    phase["promotion_decision"] = r1_gate_class(confirm["primary_metrics"], transfer)
    phase["phase_status"] = PHASE_COMPLETE
    phase["terminal_phase_status"] = PHASE_COMPLETE
    base.save_manifest(manifest_path, manifest)


def finish_r2_phase(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any], phase: dict[str, Any]) -> None:
    checkpoint_root = Path(phase["checkpoint_root"])
    confirm = base.run_summary_confirm_eval(
        args,
        checkpoint_path=best_checkpoint_from_phase(phase),
        output_root=checkpoint_root,
        stem="confirm_eval",
        episodes=R2_CONFIRM_EPISODES if str(phase.get("phase_kind")) != "smoke_r2" else 1,
        action_selection="deterministic",
    )
    phase["confirm_eval"] = confirm["summary"]
    phase["confirm_primary_metrics"] = confirm["primary_metrics"]
    phase["promotion_decision"] = r2_gate_class(confirm["primary_metrics"])
    phase["phase_status"] = PHASE_COMPLETE
    phase["terminal_phase_status"] = PHASE_COMPLETE
    base.save_manifest(manifest_path, manifest)

def finish_forage_phase(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any], phase: dict[str, Any]) -> None:
    checkpoint_root = Path(phase["checkpoint_root"])
    best_confirm = base.run_summary_confirm_eval(
        args,
        checkpoint_path=best_checkpoint_from_phase(phase),
        output_root=checkpoint_root,
        stem="confirm_eval_best_both",
        episodes=FORAGE_CONFIRM_EPISODES,
        action_selection="both",
    )
    final_checkpoint_path = str(phase.get("final_checkpoint_path") or best_checkpoint_from_phase(phase))
    final_confirm = base.run_summary_confirm_eval(
        args,
        checkpoint_path=final_checkpoint_path,
        output_root=checkpoint_root,
        stem="confirm_eval_final_both",
        episodes=FORAGE_CONFIRM_EPISODES,
        action_selection="both",
    )
    phase["confirm_eval_best"] = best_confirm["summary"]
    phase["confirm_eval_final"] = final_confirm["summary"]
    phase["confirm_primary_metrics_best"] = base.normalize_metric_dict(best_confirm["summary"].get("deterministic_eval"))
    phase["confirm_primary_metrics_final"] = base.normalize_metric_dict(final_confirm["summary"].get("deterministic_eval"))
    better = max(
        [
            {"label": "best", "checkpoint_path": best_checkpoint_from_phase(phase), "summary": best_confirm["summary"]},
            {"label": "final", "checkpoint_path": final_checkpoint_path, "summary": final_confirm["summary"]},
        ],
        key=lambda record: (
            metric_float(base.normalize_metric_dict(record["summary"].get("deterministic_eval")), "mean_pellets_per_fish", float("-inf")),
            metric_float(base.normalize_metric_dict(record["summary"].get("deterministic_eval")), "mean_forward_velocity", float("-inf")),
            metric_float(base.normalize_metric_dict(record["summary"].get("deterministic_eval")), "mean_abs_activation", float("-inf")),
        ),
    )
    better_metrics = base.normalize_metric_dict(better["summary"].get("deterministic_eval"))
    phase["selected_confirm_checkpoint_label"] = better["label"]
    phase["selected_confirm_checkpoint_path"] = better["checkpoint_path"]
    phase["selected_confirm_primary_metrics"] = better_metrics
    phase["motion_alive_gate_passed"] = base.forage_motion_alive(better_metrics)
    phase["forage_signal_gate_passed"] = base.forage_signal_alive(better_metrics)
    phase["promotion_decision"] = "forage_signal" if phase["forage_signal_gate_passed"] else ("motion_alive_only" if phase["motion_alive_gate_passed"] else "no_signal")
    phase["phase_status"] = PHASE_COMPLETE
    phase["terminal_phase_status"] = PHASE_COMPLETE
    base.save_manifest(manifest_path, manifest)


def recover_if_needed(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any]) -> bool:
    phase = base.current_training_phase(manifest)
    if phase is not None:
        base.recover_training_subprocess(args, manifest_path, manifest, phase)
        return True
    phase = pending_postprocess_phase(manifest)
    if phase is None:
        return False
    manifest["last_resume_recovery_at"] = now_iso()
    phase_kind = str(phase.get("phase_kind"))
    try:
        if phase_kind == "donor_confirm":
            finish_donor_confirm(args, manifest_path, manifest, phase)
        elif phase_kind in {"r1_branch", "smoke_r1"}:
            finish_r1_phase(args, manifest_path, manifest, phase)
        elif phase_kind in {"r2_initial", "r2_continue", "smoke_r2"}:
            finish_r2_phase(args, manifest_path, manifest, phase)
        elif phase_kind in {"forage_initial", "forage_continue"}:
            finish_forage_phase(args, manifest_path, manifest, phase)
        elif phase_kind == "baseline_random":
            family = base.get_family_from_phase(phase)
            baseline = base.run_random_baseline(args, output_root=Path(manifest["target_root"]), family=family)
            phase["random_baseline"] = baseline["summary"]
            phase["random_baseline_json_path"] = baseline["summary_json_path"]
            phase["random_baseline_csv_path"] = baseline["summary_csv_path"]
            phase["phase_status"] = PHASE_COMPLETE
            phase["terminal_phase_status"] = PHASE_COMPLETE
            phase["finished_at"] = phase.get("finished_at") or now_iso()
            base.save_manifest(manifest_path, manifest)
        else:
            return False
    except Exception as exc:
        phase["phase_status"] = PHASE_CONFIRM_FAILED
        phase["terminal_phase_status"] = PHASE_CONFIRM_FAILED
        phase["promotion_decision"] = PHASE_CONFIRM_FAILED
        phase["failure_message"] = f"resume_postprocess_failed: {exc}"
        phase["finished_at"] = phase.get("finished_at") or now_iso()
        base.save_manifest(manifest_path, manifest)
    return True


def run_donor_confirm_phase(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    checkpoint_root = Path(manifest["target_root"]) / "donor_confirm"
    phase = base.new_phase_record(
        manifest,
        phase_kind="donor_confirm",
        family_spec=None,
        training_phase=None,
        checkpoint_root=checkpoint_root,
        restore_from_checkpoint=str(manifest["donor_checkpoint"]),
        params={"episodes": R1_CONFIRM_EPISODES},
    )
    base.save_manifest(manifest_path, manifest)
    finish_donor_confirm(args, manifest_path, manifest, phase)
    return phase


def run_r1_phase(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any], branch: R1Branch, *, phase_kind: str = "r1_branch", params_override: dict[str, Any] | None = None) -> dict[str, Any]:
    checkpoint_root = phase_root(Path(manifest["target_root"]), phase_kind, branch.family.family_id)
    if params_override is None:
        params = build_r1_training_params(args, branch.family, checkpoint_root, warmstart_checkpoint=str(manifest["donor_checkpoint"]), train_iterations=int(branch.train_iterations))
    else:
        params = dict(params_override)
    phase = base.new_phase_record(
        manifest,
        phase_kind=phase_kind,
        family_spec=branch.family,
        training_phase="locomotion_propulsion_easy",
        checkpoint_root=checkpoint_root,
        restore_from_checkpoint=params.get("warmstart_motion_checkpoint") or params.get("restore_from_checkpoint"),
        params=params,
    )
    base.save_manifest(manifest_path, manifest)
    base.run_training_subprocess(args, manifest_path, manifest, phase, base.build_agent_command(args, params))
    if str(phase.get("terminal_phase_status")) != PHASE_COMPLETE:
        return phase
    try:
        finish_r1_phase(args, manifest_path, manifest, phase)
    except Exception as exc:
        phase["phase_status"] = PHASE_CONFIRM_FAILED
        phase["terminal_phase_status"] = PHASE_CONFIRM_FAILED
        phase["promotion_decision"] = PHASE_CONFIRM_FAILED
        phase["failure_message"] = f"confirm_eval_failed: {exc}"
        base.save_manifest(manifest_path, manifest)
    return phase

def run_r2_phase(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any], *, source_phase: dict[str, Any], continuation_checkpoint: str | None = None, params_override: dict[str, Any] | None = None, phase_kind_override: str | None = None) -> dict[str, Any]:
    family = base.get_family_from_phase(source_phase)
    phase_kind = phase_kind_override or ("r2_continue" if continuation_checkpoint is not None else "r2_initial")
    checkpoint_root = phase_root(Path(manifest["target_root"]), phase_kind, family.family_id)
    if params_override is not None:
        params = dict(params_override)
    elif continuation_checkpoint is not None:
        params = build_r2_continuation_params(args, family, checkpoint_root, restore_checkpoint=continuation_checkpoint)
    else:
        params = build_r2_training_params(args, family, checkpoint_root, warmstart_checkpoint=best_checkpoint_from_phase(source_phase))
    phase = base.new_phase_record(
        manifest,
        phase_kind=phase_kind,
        family_spec=family,
        training_phase="locomotion_propulsion_robust",
        checkpoint_root=checkpoint_root,
        restore_from_checkpoint=params.get("restore_from_checkpoint") or params.get("warmstart_motion_checkpoint"),
        params=params,
    )
    base.save_manifest(manifest_path, manifest)
    base.run_training_subprocess(args, manifest_path, manifest, phase, base.build_agent_command(args, params))
    if str(phase.get("terminal_phase_status")) != PHASE_COMPLETE:
        return phase
    try:
        finish_r2_phase(args, manifest_path, manifest, phase)
    except Exception as exc:
        phase["phase_status"] = PHASE_CONFIRM_FAILED
        phase["terminal_phase_status"] = PHASE_CONFIRM_FAILED
        phase["promotion_decision"] = PHASE_CONFIRM_FAILED
        phase["failure_message"] = f"confirm_eval_failed: {exc}"
        base.save_manifest(manifest_path, manifest)
    return phase


def run_forage_phase(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any], *, source_phase: dict[str, Any], restore_checkpoint: str | None = None, train_iterations: int = 80, learning_rate: float = 2e-4, entropy_coeff: float = 0.01) -> dict[str, Any]:
    family = base.get_family_from_phase(source_phase)
    phase_kind = "forage_continue" if restore_checkpoint is not None else "forage_initial"
    checkpoint_root = phase_root(Path(manifest["target_root"]), phase_kind, family.family_id)
    spec = ForageRunSpec(
        checkpoint_root=checkpoint_root,
        restore_checkpoint=restore_checkpoint,
        warmstart_checkpoint=None if restore_checkpoint is not None else best_checkpoint_from_phase(source_phase),
        train_iterations=train_iterations,
        learning_rate=learning_rate,
        entropy_coeff=entropy_coeff,
    )
    params = build_forage_training_params(args, family, spec)
    phase = base.new_phase_record(
        manifest,
        phase_kind=phase_kind,
        family_spec=family,
        training_phase="forage_full",
        checkpoint_root=checkpoint_root,
        restore_from_checkpoint=params.get("restore_from_checkpoint") or params.get("warmstart_motion_checkpoint"),
        params=params,
    )
    base.save_manifest(manifest_path, manifest)
    base.run_training_subprocess(args, manifest_path, manifest, phase, base.build_agent_command(args, params))
    if str(phase.get("terminal_phase_status")) != PHASE_COMPLETE:
        return phase
    try:
        finish_forage_phase(args, manifest_path, manifest, phase)
    except Exception as exc:
        phase["phase_status"] = PHASE_CONFIRM_FAILED
        phase["terminal_phase_status"] = PHASE_CONFIRM_FAILED
        phase["promotion_decision"] = PHASE_CONFIRM_FAILED
        phase["failure_message"] = f"confirm_eval_failed: {exc}"
        base.save_manifest(manifest_path, manifest)
    return phase


def run_random_baseline_phase(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any], family: base.FamilySpec) -> dict[str, Any]:
    checkpoint_root = Path(manifest["target_root"])
    phase = base.new_phase_record(
        manifest,
        phase_kind="baseline_random",
        family_spec=family,
        training_phase=None,
        checkpoint_root=checkpoint_root,
        restore_from_checkpoint=None,
        params={"episodes": RANDOM_BASELINE_EPISODES},
    )
    phase["phase_status"] = "running"
    phase["started_at"] = now_iso()
    base.save_manifest(manifest_path, manifest)
    try:
        baseline = base.run_random_baseline(args, output_root=checkpoint_root, family=family)
        phase["random_baseline"] = baseline["summary"]
        phase["random_baseline_json_path"] = baseline["summary_json_path"]
        phase["random_baseline_csv_path"] = baseline["summary_csv_path"]
        phase["phase_status"] = PHASE_COMPLETE
        phase["terminal_phase_status"] = PHASE_COMPLETE
    except Exception as exc:
        phase["phase_status"] = PHASE_CONFIRM_FAILED
        phase["terminal_phase_status"] = PHASE_CONFIRM_FAILED
        phase["promotion_decision"] = PHASE_CONFIRM_FAILED
        phase["failure_message"] = f"baseline_generation_failed: {exc}"
    phase["finished_at"] = now_iso()
    base.save_manifest(manifest_path, manifest)
    return phase


def maybe_run_smoke(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any], target_root: Path) -> bool:
    if not args.smoke:
        return False
    if base.terminal_status(manifest):
        return True
    if manifest.get("phases"):
        base.stop_campaign(manifest, status="success", stop_reason="smoke_complete")
        base.save_manifest(manifest_path, manifest)
        return True
    donor_phase = run_donor_confirm_phase(args, manifest_path, manifest)
    if str(donor_phase.get("terminal_phase_status")) != PHASE_COMPLETE:
        return True
    branch = smoke_branch()
    r1_params = build_smoke_r1_params(args, branch.family, target_root / "smoke_r1_hold20", donor_checkpoint=str(manifest["donor_checkpoint"]))
    r1_phase = run_r1_phase(args, manifest_path, manifest, branch, phase_kind="smoke_r1", params_override=r1_params)
    if str(r1_phase.get("terminal_phase_status")) != PHASE_COMPLETE or r1_phase.get("confirm_primary_metrics") is None:
        base.stop_campaign(manifest, status="failed", stop_reason="smoke_r1_failed")
        base.save_manifest(manifest_path, manifest)
        return True
    r2_params = build_smoke_r2_params(args, branch.family, target_root / "smoke_r2_hold20", warmstart_checkpoint=best_checkpoint_from_phase(r1_phase))
    r2_phase = run_r2_phase(args, manifest_path, manifest, source_phase=r1_phase, params_override=r2_params, phase_kind_override="smoke_r2")
    if str(r2_phase.get("terminal_phase_status")) != PHASE_COMPLETE or r2_phase.get("confirm_primary_metrics") is None:
        base.stop_campaign(manifest, status="failed", stop_reason="smoke_r2_failed")
        base.save_manifest(manifest_path, manifest)
        return True
    base.stop_campaign(manifest, status="success", stop_reason="smoke_complete")
    base.save_manifest(manifest_path, manifest)
    return True

def selected_r1_candidates(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = base.completed_phase_records(manifest, "r1_branch")
    full = [record for record in records if str(record.get("promotion_decision")) == "full_pass"]
    if full:
        return sorted(full, key=r1_rank_key, reverse=True)[:2]
    near = [record for record in records if str(record.get("promotion_decision")) == "near_pass"]
    if near:
        return [choose_best_r1(near)]
    return []


def next_pending_r1_branch(manifest: dict[str, Any], branches: list[R1Branch]) -> R1Branch | None:
    for branch in branches:
        rows = [phase for phase in manifest.get("phases", []) if str(phase.get("phase_kind")) == "r1_branch" and str(phase.get("family_id")) == branch.family.family_id]
        if not rows:
            return branch
        if str(rows[-1].get("terminal_phase_status")) not in TERMINAL_PHASE_STATUSES:
            return None
    return None


def maybe_start_r1_branch(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any], branches: list[R1Branch]) -> bool:
    branch = next_pending_r1_branch(manifest, branches)
    if branch is None:
        return False
    if not can_start_phase(manifest, "r1_branch"):
        base.stop_campaign(manifest, status="deadline_reached", stop_reason="insufficient_time_for_r1_branch")
        base.save_manifest(manifest_path, manifest)
        return True
    run_r1_phase(args, manifest_path, manifest, branch)
    return True


def maybe_start_r2(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any], selected_r1: list[dict[str, Any]]) -> bool:
    selected_ids = {record["family_id"] for record in selected_r1}
    existing = {record["family_id"] for record in base.terminal_phase_records(manifest, "r2_initial")}
    missing = [record for record in selected_r1 if record["family_id"] not in existing]
    if not missing:
        return False
    if not can_start_phase(manifest, "r2_initial"):
        base.stop_campaign(manifest, status="deadline_reached", stop_reason="insufficient_time_for_r2_initial")
        base.save_manifest(manifest_path, manifest)
        return True
    run_r2_phase(args, manifest_path, manifest, source_phase=missing[0])
    return True


def maybe_start_r2_continue(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any], selected_r1: list[dict[str, Any]]) -> bool:
    selected_ids = {record["family_id"] for record in selected_r1}
    promoted = [record for record in base.completed_phase_records(manifest, "r2_initial", "r2_continue") if record["family_id"] in selected_ids]
    if any(str(record.get("promotion_decision")) == "full_pass" for record in promoted):
        return False
    near = [record for record in promoted if str(record.get("promotion_decision")) == "near_pass"]
    if near and not base.terminal_phase_records(manifest, "r2_continue") and base.remaining_hours(manifest) > 1.5 and can_start_phase(manifest, "r2_continue"):
        best_near = choose_best_r2(near)
        run_r2_phase(args, manifest_path, manifest, source_phase=best_near, continuation_checkpoint=best_checkpoint_from_phase(best_near))
        return True
    return False


def best_r2_phase(manifest: dict[str, Any], selected_r1: list[dict[str, Any]]) -> dict[str, Any] | None:
    selected_ids = {record["family_id"] for record in selected_r1}
    promoted = [record for record in base.completed_phase_records(manifest, "r2_initial", "r2_continue") if record["family_id"] in selected_ids]
    full = [record for record in promoted if str(record.get("promotion_decision")) == "full_pass"]
    if not full:
        return None
    return choose_best_r2(full)


def maybe_start_baseline(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any], source_phase: dict[str, Any]) -> bool:
    if base.terminal_phase_records(manifest, "baseline_random"):
        return False
    if not can_start_phase(manifest, "baseline_random"):
        base.stop_campaign(manifest, status="deadline_reached", stop_reason="insufficient_time_for_random_baseline")
        base.save_manifest(manifest_path, manifest)
        return True
    run_random_baseline_phase(args, manifest_path, manifest, base.get_family_from_phase(source_phase))
    return True


def maybe_start_forage(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any], source_phase: dict[str, Any]) -> bool:
    if base.terminal_phase_records(manifest, "forage_initial"):
        return False
    if not can_start_phase(manifest, "forage_initial"):
        base.stop_campaign(manifest, status="deadline_reached", stop_reason="insufficient_time_for_forage_initial")
        base.save_manifest(manifest_path, manifest)
        return True
    run_forage_phase(args, manifest_path, manifest, source_phase=source_phase)
    return True


def maybe_start_forage_continue(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any], best_forage_phase: dict[str, Any], source_phase: dict[str, Any]) -> bool:
    if not bool(best_forage_phase.get("forage_signal_gate_passed")):
        return False
    if base.terminal_phase_records(manifest, "forage_continue"):
        return False
    if base.remaining_hours(manifest) <= 2.0 or not can_start_phase(manifest, "forage_continue"):
        return False
    restore_checkpoint = str(best_forage_phase["selected_confirm_checkpoint_path"])
    run_forage_phase(args, manifest_path, manifest, source_phase=source_phase, restore_checkpoint=restore_checkpoint, train_iterations=100, learning_rate=2e-4, entropy_coeff=0.01)
    return True


def finalize_campaign_from_forage(manifest_path: Path, manifest: dict[str, Any]) -> None:
    final_forage_records = base.completed_phase_records(manifest, "forage_initial", "forage_continue")
    if not final_forage_records:
        base.stop_campaign(manifest, status="failed", stop_reason="no_successful_forage_phase")
        base.save_manifest(manifest_path, manifest)
        return
    successful = [record for record in final_forage_records if bool(record.get("forage_signal_gate_passed"))]
    if successful:
        best_success = max(
            successful,
            key=lambda record: (
                metric_float(base.normalize_metric_dict(record.get("selected_confirm_primary_metrics")), "mean_pellets_per_fish", float("-inf")),
                metric_float(base.normalize_metric_dict(record.get("selected_confirm_primary_metrics")), "mean_forward_velocity", float("-inf")),
                metric_float(base.normalize_metric_dict(record.get("selected_confirm_primary_metrics")), "mean_abs_activation", float("-inf")),
            ),
        )
        manifest.setdefault("artifacts", {})["best_forage_phase_id"] = best_success["phase_id"]
        manifest["artifacts"]["best_forage_checkpoint_path"] = best_success["selected_confirm_checkpoint_path"]
        base.stop_campaign(manifest, status="success", stop_reason="forage_signal_confirmed")
        base.save_manifest(manifest_path, manifest)
        return
    base.stop_campaign(manifest, status="failed", stop_reason="no_forage_signal")
    base.save_manifest(manifest_path, manifest)

def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path).resolve()
    target_root = Path(args.target_root).resolve()
    manifest = ensure_resume_allowed(args, manifest_path, target_root)
    manifest["donor_checkpoint"] = str(Path(args.donor_checkpoint).resolve())
    base.save_manifest(manifest_path, manifest)
    log(f"controller_started manifest={manifest_path} target_root={target_root} donor={manifest['donor_checkpoint']} smoke={args.smoke}")

    while True:
        base.save_manifest(manifest_path, manifest)
        if base.terminal_status(manifest):
            base.save_manifest(manifest_path, manifest)
            log(f"controller_stopping status={manifest['status']} stop_reason={manifest.get('stop_reason')}")
            return

        if base.remaining_hours(manifest) <= 0.0:
            base.stop_campaign(manifest, status="deadline_reached", stop_reason="max_wall_clock_exhausted")
            base.save_manifest(manifest_path, manifest)
            continue

        if recover_if_needed(args, manifest_path, manifest):
            continue

        if maybe_run_smoke(args, manifest_path, manifest, target_root):
            continue

        donor_phase = donor_phase_record(manifest)
        if donor_phase is None:
            if not can_start_phase(manifest, "donor_confirm"):
                base.stop_campaign(manifest, status="deadline_reached", stop_reason="insufficient_time_for_donor_confirm")
                base.save_manifest(manifest_path, manifest)
                continue
            run_donor_confirm_phase(args, manifest_path, manifest)
            continue
        if str(donor_phase.get("terminal_phase_status")) != PHASE_COMPLETE:
            if str(donor_phase.get("terminal_phase_status")) == PHASE_CONFIRM_FAILED:
                base.stop_campaign(manifest, status="failed", stop_reason="donor_confirm_failed")
                base.save_manifest(manifest_path, manifest)
                continue
            finish_donor_confirm(args, manifest_path, manifest, donor_phase)
            continue

        branches = r1_branch_specs()
        if maybe_start_r1_branch(args, manifest_path, manifest, branches):
            continue

        selected_r1 = selected_r1_candidates(manifest)
        if not selected_r1:
            base.stop_campaign(manifest, status="blocked", stop_reason="blocked_no_retained_swimmer")
            base.save_manifest(manifest_path, manifest)
            continue
        manifest.setdefault("artifacts", {})["selected_r1_family_ids"] = [record["family_id"] for record in selected_r1]
        base.save_manifest(manifest_path, manifest)

        if maybe_start_r2(args, manifest_path, manifest, selected_r1):
            continue
        if maybe_start_r2_continue(args, manifest_path, manifest, selected_r1):
            continue

        best_r2 = best_r2_phase(manifest, selected_r1)
        if best_r2 is None:
            base.stop_campaign(manifest, status="blocked", stop_reason="blocked_no_robust_swimmer")
            base.save_manifest(manifest_path, manifest)
            continue
        manifest.setdefault("artifacts", {})["best_r2_phase_id"] = best_r2["phase_id"]
        manifest["artifacts"]["best_r2_checkpoint_path"] = best_checkpoint_from_phase(best_r2)
        base.save_manifest(manifest_path, manifest)

        if maybe_start_baseline(args, manifest_path, manifest, best_r2):
            continue
        if maybe_start_forage(args, manifest_path, manifest, best_r2):
            continue

        forage_records = base.completed_phase_records(manifest, "forage_initial", "forage_continue")
        if not forage_records:
            base.stop_campaign(manifest, status="failed", stop_reason="no_successful_forage_phase")
            base.save_manifest(manifest_path, manifest)
            continue
        best_forage = max(
            forage_records,
            key=lambda record: (
                metric_float(base.normalize_metric_dict(record.get("selected_confirm_primary_metrics")), "mean_pellets_per_fish", float("-inf")),
                metric_float(base.normalize_metric_dict(record.get("selected_confirm_primary_metrics")), "mean_forward_velocity", float("-inf")),
                metric_float(base.normalize_metric_dict(record.get("selected_confirm_primary_metrics")), "mean_abs_activation", float("-inf")),
            ),
        )
        manifest.setdefault("artifacts", {})["best_forage_phase_id"] = best_forage["phase_id"]
        manifest["artifacts"]["best_forage_checkpoint_path"] = best_forage.get("selected_confirm_checkpoint_path")
        base.save_manifest(manifest_path, manifest)
        if maybe_start_forage_continue(args, manifest_path, manifest, best_forage, best_r2):
            continue
        finalize_campaign_from_forage(manifest_path, manifest)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        args = parse_args()
        manifest_path = Path(args.manifest_path).resolve()
        target_root = Path(args.target_root).resolve()
        manifest = base.try_load_json(manifest_path)
        if manifest is None:
            manifest = init_manifest(args, manifest_path=manifest_path, target_root=target_root)
        manifest["controller_exception"] = str(exc)
        manifest["controller_traceback"] = traceback.format_exc()
        base.stop_campaign(manifest, status="failed", stop_reason="controller_exception")
        base.save_manifest(manifest_path, manifest)
        log(f"controller_exception: {exc}")
        raise
