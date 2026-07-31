"""Run an unattended V9 anti-limit propulsion campaign with gating and promotion."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST_PATH = SCRIPT_DIR / "propulsion_limitpressure_campaign_manifest.json"
DEFAULT_TARGET_ROOT = SCRIPT_DIR / "rllib_checkpoints_v9_day_campaign_limitpressure"
DEFAULT_CONTROLLER_POLL_SECONDS = 30
HEARTBEAT_WRITE_SECONDS = 30
A1_CONFIRM_EPISODES = 10
A2_CONFIRM_EPISODES = 10
FORAGE_CONFIRM_EPISODES = 10
RANDOM_BASELINE_EPISODES = 20
TERMINAL_STATUSES = {"success", "failed", "blocked", "deadline_reached"}
PHASE_COMPLETE = "complete"
PHASE_INFRA_FAILED = "infra_failed"
PHASE_TRAINING_FAILED = "training_failed"
PHASE_CONFIRM_FAILED = "confirm_failed"
TERMINAL_PHASE_STATUSES = {
    PHASE_COMPLETE,
    PHASE_INFRA_FAILED,
    PHASE_TRAINING_FAILED,
    PHASE_CONFIRM_FAILED,
}
A1_INITIAL_PHASE_KINDS = ("a1_initial", "a1_initial_retry")
A1_ALL_PHASE_KINDS = ("a1_initial", "a1_initial_retry", "a1_refine")
A2_PHASE_KINDS = ("a2_initial", "a2_continue")
FORAGE_PHASE_KINDS = ("forage_initial", "forage_continue")


def default_python_executable() -> str:
    venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python.resolve())
    return sys.executable


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    joint_soft_limit_start_ratio: float
    joint_soft_limit_stiffness: float
    joint_soft_limit_damping: float
    activation_time_constant: float
    propulsion_near_limit_weight: float
    propulsion_saturation_weight: float
    propulsion_torque_weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "joint_soft_limit_start_ratio": float(self.joint_soft_limit_start_ratio),
            "joint_soft_limit_stiffness": float(self.joint_soft_limit_stiffness),
            "joint_soft_limit_damping": float(self.joint_soft_limit_damping),
            "activation_time_constant": float(self.activation_time_constant),
            "propulsion_near_limit_weight": float(self.propulsion_near_limit_weight),
            "propulsion_saturation_weight": float(self.propulsion_saturation_weight),
            "propulsion_torque_weight": float(self.propulsion_torque_weight),
        }


def now_local() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def now_iso() -> str:
    return now_local().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the unattended V9 anti-limit propulsion campaign.")
    parser.add_argument("--python-executable", type=str, default=default_python_executable())
    parser.add_argument("--manifest-path", type=str, default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--target-root", type=str, default=str(DEFAULT_TARGET_ROOT))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-device", type=str, default="cuda")
    parser.add_argument("--max-wall-clock-hours", type=float, default=22.0)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_CONTROLLER_POLL_SECONDS)
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def try_load_json(path: Path) -> dict[str, Any] | None:
    try:
        return load_json(path)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def tail_text(path: Path, *, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def maybe_number(value: Any) -> Any:
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    text = str(value).strip()
    if text == "":
        return text
    lowered = text.lower()
    if lowered == "nan":
        return float("nan")
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def normalize_metric_dict(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not raw:
        return {}
    return {str(key): maybe_number(value) for key, value in raw.items()}


def is_process_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def format_command(command: list[str]) -> str:
    parts: list[str] = []
    for part in command:
        text = str(part)
        if " " in text or "\t" in text:
            parts.append(f'"{text}"')
        else:
            parts.append(text)
    return " ".join(parts)


def headless_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    return env


def checkpoint_root_for_phase(
    target_root: Path,
    *,
    phase_kind: str,
    family_id: str,
    attempt_index: int = 1,
) -> Path:
    stem = f"{phase_kind}_{family_id}"
    if attempt_index > 1:
        stem = f"{stem}_retry{attempt_index - 1}"
    return target_root / stem


def classify_phase_failure(
    *,
    training_status: str | None,
    failure_message: str | None,
    failure_traceback: str | None,
    stderr_tail: str,
    stdout_tail: str,
) -> tuple[str, str]:
    haystack = "\n".join(
        part
        for part in (
            str(training_status or ""),
            str(failure_message or ""),
            str(failure_traceback or ""),
            str(stderr_tail or ""),
            str(stdout_tail or ""),
        )
        if part
    ).lower()
    infra_markers = (
        "window-close event",
        "forrtl: error (200)",
        "actordiederror",
        "worker died unexpectedly",
        "system_error",
        "connection error code 10054",
        "raylet",
        "failed_exception",
        "run_summary_missing_after_training",
    )
    if any(marker in haystack for marker in infra_markers):
        reason = str(failure_message or training_status or "infra_failed").strip()
        return PHASE_INFRA_FAILED, reason
    reason = str(failure_message or training_status or "training_failed").strip()
    return PHASE_TRAINING_FAILED, reason


def initial_families() -> list[FamilySpec]:
    return [
        FamilySpec("lp_ref", 0.70, 18.0, 2.0, 0.12, -0.22, -0.10, -0.05),
        FamilySpec("lp_early_soft", 0.65, 18.0, 2.5, 0.12, -0.22, -0.10, -0.05),
        FamilySpec("lp_firmer", 0.70, 26.0, 3.0, 0.12, -0.22, -0.10, -0.05),
        FamilySpec("lp_smooth", 0.70, 18.0, 2.0, 0.16, -0.24, -0.12, -0.07),
        FamilySpec("lp_combo_a", 0.65, 24.0, 3.0, 0.16, -0.26, -0.14, -0.08),
        FamilySpec("lp_combo_b", 0.60, 28.0, 3.5, 0.18, -0.28, -0.16, -0.08),
    ]


def refinement_families(base: FamilySpec) -> list[FamilySpec]:
    base_id = base.family_id
    tighter_start = max(0.60, float(base.joint_soft_limit_start_ratio) - 0.05)
    tighter_stiffness = float(base.joint_soft_limit_stiffness) + 4.0
    tighter_damping = float(base.joint_soft_limit_damping) + 0.5
    smoother_tau = float(base.activation_time_constant) + 0.02
    smoother_sat = float(base.propulsion_saturation_weight) - 0.04
    smoother_torque = float(base.propulsion_torque_weight) - 0.02
    stronger_near = float(base.propulsion_near_limit_weight) - 0.04
    return [
        FamilySpec(f"{base_id}_refine_tighter", tighter_start, tighter_stiffness, tighter_damping, base.activation_time_constant, base.propulsion_near_limit_weight, base.propulsion_saturation_weight, base.propulsion_torque_weight),
        FamilySpec(f"{base_id}_refine_smoother", base.joint_soft_limit_start_ratio, base.joint_soft_limit_stiffness, base.joint_soft_limit_damping, smoother_tau, base.propulsion_near_limit_weight, smoother_sat, smoother_torque),
        FamilySpec(f"{base_id}_refine_efficiency", base.joint_soft_limit_start_ratio, base.joint_soft_limit_stiffness, base.joint_soft_limit_damping, base.activation_time_constant, stronger_near, smoother_sat, smoother_torque),
        FamilySpec(f"{base_id}_refine_balanced", tighter_start, tighter_stiffness, tighter_damping, smoother_tau, stronger_near, smoother_sat, smoother_torque),
    ]


def init_manifest(args: argparse.Namespace, *, manifest_path: Path, target_root: Path) -> dict[str, Any]:
    started_at = now_local()
    return {
        "created_at": started_at.isoformat(timespec="seconds"),
        "updated_at": started_at.isoformat(timespec="seconds"),
        "status": "running",
        "stop_reason": None,
        "manifest_version": 1,
        "python_executable": str(Path(args.python_executable).resolve()),
        "device": str(args.device),
        "eval_device": str(args.eval_device),
        "max_wall_clock_hours": float(args.max_wall_clock_hours),
        "deadline_at": (started_at + timedelta(hours=float(args.max_wall_clock_hours))).isoformat(timespec="seconds"),
        "manifest_path": str(manifest_path.resolve()),
        "target_root": str(target_root.resolve()),
        "smoke": bool(args.smoke),
        "controller_pid": int(os.getpid()),
        "controller_heartbeat_at": started_at.isoformat(timespec="seconds"),
        "active_phase_id": None,
        "last_resume_recovery_at": None,
        "next_phase_index": 1,
        "phases": [],
        "artifacts": {},
    }


def active_phase_id_from_manifest(manifest: dict[str, Any]) -> str | None:
    for phase in reversed(list(manifest.get("phases", []))):
        if str(phase.get("phase_status")) == "training":
            return str(phase.get("phase_id"))
    return None


def save_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    now = now_iso()
    manifest["updated_at"] = now
    manifest["controller_pid"] = int(os.getpid())
    manifest["controller_heartbeat_at"] = now
    manifest["active_phase_id"] = active_phase_id_from_manifest(manifest)
    save_json(manifest_path, manifest)


def remaining_hours(manifest: dict[str, Any]) -> float:
    deadline_text = str(manifest.get("deadline_at"))
    deadline = datetime.fromisoformat(deadline_text)
    delta = deadline - now_local()
    return delta.total_seconds() / 3600.0


def terminal_status(manifest: dict[str, Any]) -> bool:
    return str(manifest.get("status")) in TERMINAL_STATUSES


def stop_campaign(manifest: dict[str, Any], *, status: str, stop_reason: str) -> None:
    manifest["status"] = status
    manifest["stop_reason"] = stop_reason
    manifest["finished_at"] = now_iso()


def new_phase_record(
    manifest: dict[str, Any],
    *,
    phase_kind: str,
    family_spec: FamilySpec | None,
    training_phase: str | None,
    checkpoint_root: Path,
    restore_from_checkpoint: str | None,
    params: dict[str, Any],
    attempt_index: int = 1,
    retry_of_phase_id: str | None = None,
) -> dict[str, Any]:
    phase_index = int(manifest.get("next_phase_index", 1))
    manifest["next_phase_index"] = phase_index + 1
    family_id = family_spec.family_id if family_spec is not None else phase_kind
    phase_id = f"phase_{phase_index:03d}_{phase_kind}_{family_id}"
    stdout_log = checkpoint_root / "phase_stdout.log"
    stderr_log = checkpoint_root / "phase_stderr.log"
    record = {
        "phase_id": phase_id,
        "phase_kind": phase_kind,
        "family_id": family_id,
        "family_config": family_spec.to_dict() if family_spec is not None else None,
        "training_phase": training_phase,
        "seed": int(params.get("seed", 0)),
        "checkpoint_root": str(checkpoint_root.resolve()),
        "restore_from_checkpoint": restore_from_checkpoint,
        "params": params,
        "stdout_log_path": str(stdout_log.resolve()),
        "stderr_log_path": str(stderr_log.resolve()),
        "phase_status": "scheduled",
        "terminal_phase_status": None,
        "promotion_decision": None,
        "attempt_index": int(attempt_index),
        "retry_of_phase_id": retry_of_phase_id,
        "infra_failure_reason": None,
        "started_at": None,
        "finished_at": None,
    }
    manifest.setdefault("phases", []).append(record)
    return record


def current_training_phase(manifest: dict[str, Any]) -> dict[str, Any] | None:
    for phase in reversed(list(manifest.get("phases", []))):
        if str(phase.get("phase_status")) == "training":
            return phase
    return None


def completed_phase_records(manifest: dict[str, Any], *phase_kinds: str) -> list[dict[str, Any]]:
    wanted = set(phase_kinds)
    rows: list[dict[str, Any]] = []
    for phase in manifest.get("phases", []):
        if wanted and str(phase.get("phase_kind")) not in wanted:
            continue
        if str(phase.get("terminal_phase_status")) != PHASE_COMPLETE:
            continue
        rows.append(phase)
    return rows


def terminal_phase_records(manifest: dict[str, Any], *phase_kinds: str) -> list[dict[str, Any]]:
    wanted = set(phase_kinds)
    rows: list[dict[str, Any]] = []
    for phase in manifest.get("phases", []):
        if wanted and str(phase.get("phase_kind")) not in wanted:
            continue
        if str(phase.get("terminal_phase_status")) not in TERMINAL_PHASE_STATUSES:
            continue
        rows.append(phase)
    return rows


def family_phase_records(manifest: dict[str, Any], family_id: str, *phase_kinds: str) -> list[dict[str, Any]]:
    rows = terminal_phase_records(manifest, *phase_kinds)
    return [phase for phase in rows if str(phase.get("family_id")) == str(family_id)]


def latest_family_phase(manifest: dict[str, Any], family_id: str, *phase_kinds: str) -> dict[str, Any] | None:
    rows = family_phase_records(manifest, family_id, *phase_kinds)
    if not rows:
        return None
    return rows[-1]


def is_retry_config_misfire(phase: dict[str, Any] | None) -> bool:
    if not phase:
        return False
    haystack = "\n".join(
        str(part or "")
        for part in (
            phase.get("failure_message"),
            phase.get("failure_traceback"),
            phase.get("stderr_tail"),
            phase.get("stdout_tail"),
        )
    ).lower()
    return (
        "validate_train_batch_size_vs_rollout_fragment_length" in haystack
        or "try setting `rollout_fragment_length` to 'auto' or to a value of 1333" in haystack
        or "rollout_fragment_length=500" in haystack and "num_env_runners=6" in haystack
    )


def next_pending_a1_initial_family(manifest: dict[str, Any], families: list[FamilySpec]) -> tuple[FamilySpec, dict[str, Any] | None, int, int | None] | None:
    for family in families:
        latest = latest_family_phase(manifest, family.family_id, *A1_INITIAL_PHASE_KINDS)
        if latest is None:
            return family, None, 1, None
        terminal = str(latest.get("terminal_phase_status"))
        attempt_index = int(latest.get("attempt_index", 1))
        if terminal == PHASE_COMPLETE:
            continue
        if terminal == PHASE_INFRA_FAILED and attempt_index < 2:
            return family, latest, 2, 6
        if terminal == PHASE_INFRA_FAILED and attempt_index == 2 and is_retry_config_misfire(latest):
            return family, latest, 3, 6
        continue
    return None


def initial_a1_family_resolved(manifest: dict[str, Any], family_id: str) -> bool:
    latest = latest_family_phase(manifest, family_id, *A1_INITIAL_PHASE_KINDS)
    if latest is None:
        return False
    terminal = str(latest.get("terminal_phase_status"))
    attempt_index = int(latest.get("attempt_index", 1))
    if terminal == PHASE_COMPLETE:
        return True
    if terminal == PHASE_INFRA_FAILED and attempt_index < 2:
        return False
    if terminal == PHASE_INFRA_FAILED and attempt_index == 2 and is_retry_config_misfire(latest):
        return False
    return terminal in TERMINAL_PHASE_STATUSES


def build_smoke_training_params(checkpoint_root: Path) -> dict[str, Any]:
    return {
        "policy_stack": "new",
        "training_phase": "locomotion_propulsion_easy",
        "device": "cpu",
        "checkpoint_root": str(checkpoint_root.resolve()),
        "train_iterations": 1,
        "checkpoint_every_iterations": 1,
        "num_env_runners": 1,
        "num_envs_per_runner": 1,
        "light_eval_episodes": 1,
        "train_batch_size": 250,
        "minibatch_size": 125,
        "num_epochs": 1,
        "rollout_fragment_length": 50,
        "learning_rate": 3e-4,
        "entropy_coeff": 0.02,
        "gamma": 0.97,
        "gae_lambda": 0.97,
        "fcnet_hiddens": "64,64",
        "fcnet_activation": "tanh",
        "time_limit": 60,
        "reward_mode": "locomotion_debug",
        "observation_profile": "full_v9",
        "history_length": 8,
        "joint_passive_stiffness": 10.0,
        "body_linear_drag": 1.0,
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


def build_a1_training_params(
    args: argparse.Namespace,
    family: FamilySpec,
    checkpoint_root: Path,
    *,
    train_iterations: int,
    num_env_runners_override: int | None = None,
) -> dict[str, Any]:
    num_env_runners = int(num_env_runners_override if num_env_runners_override is not None else 8)
    rollout_fragment_length = 1333 if num_env_runners == 6 else 500
    return {
        "policy_stack": "new",
        "training_phase": "locomotion_propulsion_easy",
        "device": str(args.device),
        "checkpoint_root": str(checkpoint_root.resolve()),
        "train_iterations": int(train_iterations),
        "checkpoint_every_iterations": 10,
        "num_env_runners": num_env_runners,
        "num_envs_per_runner": 2,
        "light_eval_episodes": 4,
        "train_batch_size": 16000,
        "minibatch_size": 2048,
        "num_epochs": 6,
        "rollout_fragment_length": rollout_fragment_length,
        "learning_rate": 3e-4,
        "entropy_coeff": 0.02,
        "gamma": 0.97,
        "gae_lambda": 0.97,
        "fcnet_hiddens": "512,512,256",
        "fcnet_activation": "tanh",
        "time_limit": 150,
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


def build_a2_training_params(args: argparse.Namespace, family: FamilySpec, checkpoint_root: Path, *, warmstart_checkpoint: str) -> dict[str, Any]:
    params = build_a1_training_params(args, family, checkpoint_root, train_iterations=60)
    params["training_phase"] = "locomotion_propulsion_robust"
    params["entropy_coeff"] = 0.01
    params["time_limit"] = 300
    params["warmstart_motion_checkpoint"] = str(warmstart_checkpoint)
    return params


def build_a2_continuation_params(args: argparse.Namespace, family: FamilySpec, checkpoint_root: Path, *, restore_checkpoint: str) -> dict[str, Any]:
    params = build_a1_training_params(args, family, checkpoint_root, train_iterations=40)
    params["training_phase"] = "locomotion_propulsion_robust"
    params["entropy_coeff"] = 0.01
    params["time_limit"] = 300
    params["restore_from_checkpoint"] = str(restore_checkpoint)
    return params


def build_forage_training_params(args: argparse.Namespace, family: FamilySpec, checkpoint_root: Path, *, warmstart_checkpoint: str, train_iterations: int) -> dict[str, Any]:
    return {
        "policy_stack": "new",
        "training_phase": "forage_full",
        "device": str(args.device),
        "checkpoint_root": str(checkpoint_root.resolve()),
        "train_iterations": int(train_iterations),
        "checkpoint_every_iterations": 20,
        "num_env_runners": 8,
        "num_envs_per_runner": 2,
        "light_eval_episodes": 4,
        "train_batch_size": 16000,
        "minibatch_size": 2048,
        "num_epochs": 6,
        "rollout_fragment_length": 500,
        "learning_rate": 3e-4,
        "entropy_coeff": 0.02,
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
        "warmstart_motion_checkpoint": str(warmstart_checkpoint),
    }


def build_forage_continuation_params(args: argparse.Namespace, family: FamilySpec, checkpoint_root: Path, *, restore_checkpoint: str) -> dict[str, Any]:
    params = build_forage_training_params(args, family, checkpoint_root, warmstart_checkpoint=restore_checkpoint, train_iterations=200)
    params.pop("warmstart_motion_checkpoint", None)
    params["restore_from_checkpoint"] = str(restore_checkpoint)
    return params


def build_agent_command(args: argparse.Namespace, params: dict[str, Any]) -> list[str]:
    command = [str(Path(args.python_executable).resolve()), "-u", str((SCRIPT_DIR / "agent.py").resolve())]

    def add(flag: str, value: Any) -> None:
        if value is None:
            return
        command.extend([flag, str(value)])

    add("--policy-stack", params.get("policy_stack"))
    add("--training-phase", params.get("training_phase"))
    add("--device", params.get("device"))
    add("--checkpoint-root", params.get("checkpoint_root"))
    add("--restore-from-checkpoint", params.get("restore_from_checkpoint"))
    add("--warmstart-motion-checkpoint", params.get("warmstart_motion_checkpoint"))
    add("--train-iterations", params.get("train_iterations"))
    add("--checkpoint-every-iterations", params.get("checkpoint_every_iterations"))
    add("--num-env-runners", params.get("num_env_runners"))
    add("--num-envs-per-runner", params.get("num_envs_per_runner"))
    add("--light-eval-episodes", params.get("light_eval_episodes"))
    add("--train-batch-size", params.get("train_batch_size"))
    add("--minibatch-size", params.get("minibatch_size"))
    add("--num-epochs", params.get("num_epochs"))
    add("--rollout-fragment-length", params.get("rollout_fragment_length"))
    add("--learning-rate", params.get("learning_rate"))
    add("--entropy-coeff", params.get("entropy_coeff"))
    add("--gamma", params.get("gamma"))
    add("--gae-lambda", params.get("gae_lambda"))
    add("--fcnet-hiddens", params.get("fcnet_hiddens"))
    add("--fcnet-activation", params.get("fcnet_activation"))
    add("--time-limit", params.get("time_limit"))
    add("--reward-mode", params.get("reward_mode"))
    add("--observation-profile", params.get("observation_profile"))
    add("--history-length", params.get("history_length"))
    add("--food-respawn-mode", params.get("food_respawn_mode"))
    add("--forage-timeout-mode", params.get("forage_timeout_mode"))
    add("--forage-idle-timeout-steps", params.get("forage_idle_timeout_steps"))
    add("--forage-time-context-mode", params.get("forage_time_context_mode"))
    add("--activation-time-constant", params.get("activation_time_constant"))
    add("--joint-passive-stiffness", params.get("joint_passive_stiffness"))
    add("--joint-soft-limit-start-ratio", params.get("joint_soft_limit_start_ratio"))
    add("--joint-soft-limit-stiffness", params.get("joint_soft_limit_stiffness"))
    add("--joint-soft-limit-damping", params.get("joint_soft_limit_damping"))
    add("--body-linear-drag", params.get("body_linear_drag"))
    add("--propulsion-near-limit-weight", params.get("propulsion_near_limit_weight"))
    add("--propulsion-saturation-weight", params.get("propulsion_saturation_weight"))
    add("--propulsion-torque-weight", params.get("propulsion_torque_weight"))
    add("--motion-epsilon-start", params.get("motion_epsilon_start"))
    add("--motion-epsilon-end", params.get("motion_epsilon_end"))
    add("--motion-epsilon-decay-iterations", params.get("motion_epsilon_decay_iterations"))
    add("--message-epsilon", params.get("message_epsilon"))
    add("--seed", params.get("seed"))
    add("--num-red-fish", params.get("num_red_fish"))
    add("--num-blue-fish", params.get("num_blue_fish"))
    add("--num-red-pellets", params.get("num_red_pellets"))
    add("--num-blue-pellets", params.get("num_blue_pellets"))
    return command


def extract_primary_eval(nested_summary: dict[str, Any], *, action_selection: str) -> dict[str, Any]:
    if action_selection == "both":
        return normalize_metric_dict(nested_summary.get("deterministic_eval"))
    return normalize_metric_dict(nested_summary.get("eval_result"))


def build_test_model_command(
    args: argparse.Namespace,
    *,
    checkpoint_path: str | None,
    summary_json_path: Path,
    summary_csv_path: Path,
    episodes: int,
    action_selection: str,
    policy_mode: str = "trained",
    env_overrides: dict[str, Any] | None = None,
) -> list[str]:
    command = [str(Path(args.python_executable).resolve()), "-u", str((SCRIPT_DIR / "test_model.py").resolve())]
    command.extend(["--policy-mode", str(policy_mode)])
    if checkpoint_path is not None:
        command.extend(["--checkpoint-path", str(checkpoint_path)])
    command.extend([
        "--episodes", str(int(episodes)),
        "--device", str(args.eval_device),
        "--no-render",
        "--summary-json", str(summary_json_path.resolve()),
        "--summary-csv", str(summary_csv_path.resolve()),
        "--action-selection", str(action_selection),
    ])
    if env_overrides:
        for key, value in env_overrides.items():
            if value is None:
                continue
            command.extend([f"--{key.replace('_', '-')}", str(value)])
    return command


def a1_gate_class(metrics: dict[str, Any]) -> str:
    if (
        float(metrics.get("mean_forward_velocity", 0.0)) >= 0.08
        and float(metrics.get("mean_abs_activation", 0.0)) >= 0.08
        and float(metrics.get("fraction_joints_quiet_steps", 1.0)) < 0.30
        and float(metrics.get("fraction_negative_forward_velocity_steps", 1.0)) < 0.25
        and float(metrics.get("fraction_saturated_motion_commands", 1.0)) < 0.40
        and float(metrics.get("mean_joint_limit_occupancy", 1.0)) < 0.30
        and float(metrics.get("fraction_joint_limit_high_steps", 1.0)) < 0.05
    ):
        return "full_pass"
    if (
        float(metrics.get("mean_forward_velocity", 0.0)) >= 0.08
        and float(metrics.get("mean_abs_activation", 0.0)) >= 0.08
        and float(metrics.get("fraction_joints_quiet_steps", 1.0)) < 0.30
        and float(metrics.get("fraction_negative_forward_velocity_steps", 1.0)) < 0.30
        and float(metrics.get("fraction_saturated_motion_commands", 1.0)) < 0.55
        and float(metrics.get("mean_joint_limit_occupancy", 1.0)) < 0.40
        and float(metrics.get("fraction_joint_limit_high_steps", 1.0)) < 0.10
    ):
        return "near_pass"
    return "fail"


def a2_gate_class(metrics: dict[str, Any]) -> str:
    if (
        float(metrics.get("mean_forward_velocity", 0.0)) >= 0.05
        and float(metrics.get("mean_abs_activation", 0.0)) >= 0.08
        and float(metrics.get("fraction_joints_quiet_steps", 1.0)) < 0.30
        and float(metrics.get("fraction_negative_forward_velocity_steps", 1.0)) < 0.35
        and float(metrics.get("fraction_saturated_motion_commands", 1.0)) < 0.35
        and float(metrics.get("mean_joint_limit_occupancy", 1.0)) < 0.28
        and float(metrics.get("fraction_joint_limit_high_steps", 1.0)) < 0.05
    ):
        return "full_pass"
    if (
        float(metrics.get("mean_forward_velocity", 0.0)) >= 0.04
        and float(metrics.get("mean_abs_activation", 0.0)) >= 0.08
        and float(metrics.get("fraction_joints_quiet_steps", 1.0)) < 0.35
        and float(metrics.get("fraction_negative_forward_velocity_steps", 1.0)) < 0.40
        and float(metrics.get("fraction_saturated_motion_commands", 1.0)) < 0.45
        and float(metrics.get("mean_joint_limit_occupancy", 1.0)) < 0.34
        and float(metrics.get("fraction_joint_limit_high_steps", 1.0)) < 0.10
    ):
        return "near_pass"
    return "fail"


def candidate_rank_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics.get("mean_joint_limit_occupancy", float("inf"))),
        float(metrics.get("fraction_saturated_motion_commands", float("inf"))),
        float(metrics.get("fraction_negative_forward_velocity_steps", float("inf"))),
        -float(metrics.get("mean_forward_velocity", float("-inf"))),
    )


def choose_best_candidate(records: list[dict[str, Any]], *, metrics_field: str = "confirm_primary_metrics") -> dict[str, Any]:
    return min(records, key=lambda record: candidate_rank_key(normalize_metric_dict(record.get(metrics_field))))


def forage_candidate_rank_key(nested_summary: dict[str, Any]) -> tuple[float, float, float]:
    deterministic = normalize_metric_dict(nested_summary.get("deterministic_eval"))
    return (
        -float(deterministic.get("mean_pellets_per_fish", float("-inf"))),
        -float(deterministic.get("mean_forward_velocity", float("-inf"))),
        -float(deterministic.get("mean_abs_activation", float("-inf"))),
    )


def forage_motion_alive(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics.get("mean_abs_activation", 0.0)) >= 0.05
        and float(metrics.get("fraction_joints_quiet_steps", 1.0)) < 0.80
    )


def forage_signal_alive(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics.get("mean_pellets_per_fish", 0.0)) > 0.0
        and float(metrics.get("mean_forward_velocity", 0.0)) > 0.0
    )


def write_phase_log_header(path: Path, *, phase_id: str, command: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_iso()}] {phase_id} start\n")
        handle.write(f"[{now_iso()}] command: {format_command(command)}\n")


def finalize_training_phase(manifest: dict[str, Any], phase: dict[str, Any]) -> None:
    checkpoint_root = Path(phase["checkpoint_root"])
    run_summary_path = checkpoint_root / "run_summary.json"
    eval_csv_path = checkpoint_root / "eval_reports.csv"
    run_summary = try_load_json(run_summary_path)
    eval_rows = load_csv_rows(eval_csv_path)
    stderr_tail = tail_text(Path(phase["stderr_log_path"]))
    stdout_tail = tail_text(Path(phase["stdout_log_path"]))
    phase["run_summary_path"] = str(run_summary_path.resolve())
    phase["eval_reports_csv_path"] = str(eval_csv_path.resolve())
    phase["eval_report_row_count"] = len(eval_rows)
    phase["stderr_tail"] = stderr_tail
    phase["stdout_tail"] = stdout_tail
    if eval_rows:
        phase["latest_light_eval_row"] = normalize_metric_dict(eval_rows[-1])
    if run_summary is None:
        phase["phase_status"] = PHASE_INFRA_FAILED
        phase["terminal_phase_status"] = PHASE_INFRA_FAILED
        phase["failure_message"] = "run_summary_missing_after_training"
        phase["infra_failure_reason"] = "run_summary_missing_after_training"
        phase["promotion_decision"] = PHASE_INFRA_FAILED
        return

    phase["training_status"] = run_summary.get("training_status")
    phase["failed_iteration"] = run_summary.get("failed_iteration")
    phase["failure_message"] = run_summary.get("failure_message")
    if run_summary.get("failure_traceback"):
        phase["failure_traceback"] = run_summary.get("failure_traceback")
    best_record = normalize_metric_dict(run_summary.get("best_checkpoint"))
    phase["light_eval_best_checkpoint"] = best_record
    phase["best_checkpoint_path"] = best_record.get("checkpoint_path")
    phase["final_checkpoint_path"] = run_summary.get("final_checkpoint_path")
    phase["time_to_first_positive_total_reward"] = run_summary.get("time_to_first_positive_total_reward")
    if str(run_summary.get("training_status")) != "reached_iteration_budget":
        terminal_status, reason = classify_phase_failure(
            training_status=str(run_summary.get("training_status")),
            failure_message=str(run_summary.get("failure_message") or ""),
            failure_traceback=str(run_summary.get("failure_traceback") or ""),
            stderr_tail=stderr_tail,
            stdout_tail=stdout_tail,
        )
        phase["phase_status"] = terminal_status
        phase["terminal_phase_status"] = terminal_status
        phase["promotion_decision"] = terminal_status
        if terminal_status == PHASE_INFRA_FAILED:
            phase["infra_failure_reason"] = reason
        return
    phase["phase_status"] = PHASE_COMPLETE
    phase["terminal_phase_status"] = PHASE_COMPLETE


def run_training_subprocess(
    args: argparse.Namespace,
    manifest_path: Path,
    manifest: dict[str, Any],
    phase: dict[str, Any],
    command: list[str],
) -> None:
    stdout_log_path = Path(phase["stdout_log_path"])
    stderr_log_path = Path(phase["stderr_log_path"])
    write_phase_log_header(stdout_log_path, phase_id=str(phase["phase_id"]), command=command)
    write_phase_log_header(stderr_log_path, phase_id=str(phase["phase_id"]), command=command)

    stdout_handle = stdout_log_path.open("a", encoding="utf-8")
    stderr_handle = stderr_log_path.open("a", encoding="utf-8")
    last_heartbeat_time = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=SCRIPT_DIR,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            env=headless_subprocess_env(),
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()

    phase["phase_status"] = "training"
    phase["started_at"] = now_iso()
    phase["training_pid"] = int(process.pid)
    phase["command"] = command
    save_manifest(manifest_path, manifest)
    log(f"phase_started phase_id={phase['phase_id']} pid={process.pid}")

    while True:
        exit_code = process.poll()
        if exit_code is not None:
            break
        time.sleep(max(int(args.poll_seconds), 1))
        if time.monotonic() - last_heartbeat_time >= HEARTBEAT_WRITE_SECONDS:
            save_manifest(manifest_path, manifest)
            last_heartbeat_time = time.monotonic()

    phase["training_exit_code"] = int(exit_code)
    phase["finished_at"] = now_iso()
    if int(exit_code) != 0:
        phase["failure_message"] = f"training_subprocess_exit_{exit_code}"

    finalize_training_phase(manifest, phase)
    save_manifest(manifest_path, manifest)


def recover_training_subprocess(
    args: argparse.Namespace,
    manifest_path: Path,
    manifest: dict[str, Any],
    phase: dict[str, Any],
) -> None:
    pid = phase.get("training_pid")
    if is_process_alive(int(pid) if pid is not None else None):
        log(f"waiting_for_running_phase phase_id={phase['phase_id']} pid={pid}")
        last_heartbeat_time = time.monotonic()
        while is_process_alive(int(pid)):
            time.sleep(max(int(args.poll_seconds), 1))
            if time.monotonic() - last_heartbeat_time >= HEARTBEAT_WRITE_SECONDS:
                save_manifest(manifest_path, manifest)
                last_heartbeat_time = time.monotonic()
    phase["finished_at"] = phase.get("finished_at") or now_iso()
    manifest["last_resume_recovery_at"] = now_iso()
    finalize_training_phase(manifest, phase)
    save_manifest(manifest_path, manifest)


def run_summary_confirm_eval(
    args: argparse.Namespace,
    *,
    checkpoint_path: str,
    output_root: Path,
    stem: str,
    episodes: int,
    action_selection: str,
    env_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary_json_path = output_root / f"{stem}.json"
    summary_csv_path = output_root / f"{stem}.csv"
    command = build_test_model_command(
        args,
        checkpoint_path=checkpoint_path,
        summary_json_path=summary_json_path,
        summary_csv_path=summary_csv_path,
        episodes=episodes,
        action_selection=action_selection,
        policy_mode="trained",
        env_overrides=env_overrides,
    )
    log(f"confirm_eval_start checkpoint={checkpoint_path} action_selection={action_selection}")
    result = subprocess.run(
        command,
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        check=False,
        env=headless_subprocess_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Confirm eval failed for checkpoint={checkpoint_path} exit_code={result.returncode}\n{result.stderr}"
        )
    nested_summary = load_json(summary_json_path)
    return {
        "summary_json_path": str(summary_json_path.resolve()),
        "summary_csv_path": str(summary_csv_path.resolve()),
        "summary": nested_summary,
        "primary_metrics": extract_primary_eval(nested_summary, action_selection=action_selection),
        "stdout_tail": result.stdout.strip().splitlines()[-20:],
    }


def run_random_baseline(
    args: argparse.Namespace,
    *,
    output_root: Path,
    family: FamilySpec,
    env_overrides: dict[str, Any] | None = None,
    summary_stem: str = "random_policy_baseline_limitpressure",
) -> dict[str, Any]:
    summary_json_path = output_root / f"{summary_stem}.json"
    summary_csv_path = output_root / f"{summary_stem}.csv"
    default_env_overrides = {
        "reward_mode": "forage",
        "num_red_fish": 10,
        "num_blue_fish": 10,
        "num_red_pellets": 48,
        "num_blue_pellets": 48,
        "time_limit": 300,
        "observation_profile": "full_v9",
        "history_length": 8,
        "activation_time_constant": family.activation_time_constant,
        "joint_passive_stiffness": 10.0,
        "joint_soft_limit_start_ratio": family.joint_soft_limit_start_ratio,
        "joint_soft_limit_stiffness": family.joint_soft_limit_stiffness,
        "joint_soft_limit_damping": family.joint_soft_limit_damping,
        "body_linear_drag": 1.0,
        "propulsion_near_limit_weight": family.propulsion_near_limit_weight,
        "propulsion_saturation_weight": family.propulsion_saturation_weight,
        "propulsion_torque_weight": family.propulsion_torque_weight,
    }
    merged_env_overrides = dict(default_env_overrides)
    if env_overrides:
        merged_env_overrides.update(env_overrides)
    command = build_test_model_command(
        args,
        checkpoint_path=None,
        summary_json_path=summary_json_path,
        summary_csv_path=summary_csv_path,
        episodes=RANDOM_BASELINE_EPISODES,
        action_selection="deterministic",
        policy_mode="random",
        env_overrides=merged_env_overrides,
    )
    log("random_baseline_start")
    result = subprocess.run(
        command,
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        check=False,
        env=headless_subprocess_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Random baseline generation failed with exit code {result.returncode}\n{result.stderr}")
    nested_summary = load_json(summary_json_path)
    return {
        "summary_json_path": str(summary_json_path.resolve()),
        "summary_csv_path": str(summary_csv_path.resolve()),
        "summary": nested_summary,
        "stdout_tail": result.stdout.strip().splitlines()[-20:],
    }


def maybe_run_smoke(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any], target_root: Path) -> bool:
    if not args.smoke:
        return False
    if manifest.get("phases"):
        stop_campaign(manifest, status="success", stop_reason="smoke_complete")
        save_manifest(manifest_path, manifest)
        return True
    family = initial_families()[0]
    checkpoint_root = target_root / "smoke_lp_ref"
    params = build_smoke_training_params(checkpoint_root)
    params.update(
        {
            "activation_time_constant": family.activation_time_constant,
            "joint_soft_limit_start_ratio": family.joint_soft_limit_start_ratio,
            "joint_soft_limit_stiffness": family.joint_soft_limit_stiffness,
            "joint_soft_limit_damping": family.joint_soft_limit_damping,
            "propulsion_near_limit_weight": family.propulsion_near_limit_weight,
            "propulsion_saturation_weight": family.propulsion_saturation_weight,
            "propulsion_torque_weight": family.propulsion_torque_weight,
        }
    )
    phase = new_phase_record(
        manifest,
        phase_kind="smoke_a1",
        family_spec=family,
        training_phase="locomotion_propulsion_easy",
        checkpoint_root=checkpoint_root,
        restore_from_checkpoint=None,
        params=params,
    )
    save_manifest(manifest_path, manifest)
    run_training_subprocess(args, manifest_path, manifest, phase, build_agent_command(args, params))
    if str(phase.get("terminal_phase_status")) != PHASE_COMPLETE:
        stop_campaign(manifest, status="failed", stop_reason="smoke_training_failed")
        save_manifest(manifest_path, manifest)
        return True
    try:
        confirm = run_summary_confirm_eval(
            args,
            checkpoint_path=str(phase["best_checkpoint_path"]),
            output_root=checkpoint_root,
            stem="confirm_eval",
            episodes=1,
            action_selection="deterministic",
        )
        phase["confirm_eval"] = confirm["summary"]
        phase["confirm_primary_metrics"] = confirm["primary_metrics"]
        phase["promotion_decision"] = "smoke_complete"
        phase["phase_status"] = PHASE_COMPLETE
        phase["terminal_phase_status"] = PHASE_COMPLETE
    except Exception as exc:
        phase["phase_status"] = PHASE_CONFIRM_FAILED
        phase["terminal_phase_status"] = PHASE_CONFIRM_FAILED
        phase["promotion_decision"] = PHASE_CONFIRM_FAILED
        phase["failure_message"] = f"confirm_eval_failed: {exc}"
        stop_campaign(manifest, status="failed", stop_reason="smoke_confirm_failed")
        save_manifest(manifest_path, manifest)
        return True
    save_manifest(manifest_path, manifest)
    stop_campaign(manifest, status="success", stop_reason="smoke_complete")
    save_manifest(manifest_path, manifest)
    return True


def best_checkpoint_from_phase(phase: dict[str, Any]) -> str:
    checkpoint_path = phase.get("best_checkpoint_path")
    if not checkpoint_path:
        raise RuntimeError(f"Phase {phase['phase_id']} is missing best_checkpoint_path")
    return str(checkpoint_path)


def get_family_from_phase(phase: dict[str, Any]) -> FamilySpec:
    payload = dict(phase.get("family_config") or {})
    return FamilySpec(
        family_id=str(payload["family_id"]),
        joint_soft_limit_start_ratio=float(payload["joint_soft_limit_start_ratio"]),
        joint_soft_limit_stiffness=float(payload["joint_soft_limit_stiffness"]),
        joint_soft_limit_damping=float(payload["joint_soft_limit_damping"]),
        activation_time_constant=float(payload["activation_time_constant"]),
        propulsion_near_limit_weight=float(payload["propulsion_near_limit_weight"]),
        propulsion_saturation_weight=float(payload["propulsion_saturation_weight"]),
        propulsion_torque_weight=float(payload["propulsion_torque_weight"]),
    )


def run_a1_phase(
    args: argparse.Namespace,
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    family: FamilySpec,
    phase_kind: str,
    train_iterations: int,
    attempt_index: int = 1,
    retry_of_phase: dict[str, Any] | None = None,
    num_env_runners_override: int | None = None,
) -> dict[str, Any]:
    checkpoint_root = checkpoint_root_for_phase(
        Path(manifest["target_root"]),
        phase_kind=phase_kind,
        family_id=family.family_id,
        attempt_index=attempt_index,
    )
    params = build_a1_training_params(
        args,
        family,
        checkpoint_root,
        train_iterations=train_iterations,
        num_env_runners_override=num_env_runners_override,
    )
    phase = new_phase_record(
        manifest,
        phase_kind=phase_kind,
        family_spec=family,
        training_phase="locomotion_propulsion_easy",
        checkpoint_root=checkpoint_root,
        restore_from_checkpoint=None,
        params=params,
        attempt_index=attempt_index,
        retry_of_phase_id=None if retry_of_phase is None else str(retry_of_phase["phase_id"]),
    )
    save_manifest(manifest_path, manifest)
    run_training_subprocess(args, manifest_path, manifest, phase, build_agent_command(args, params))
    if str(phase.get("terminal_phase_status")) != PHASE_COMPLETE:
        save_manifest(manifest_path, manifest)
        return phase
    try:
        confirm = run_summary_confirm_eval(
            args,
            checkpoint_path=best_checkpoint_from_phase(phase),
            output_root=checkpoint_root,
            stem="confirm_eval",
            episodes=A1_CONFIRM_EPISODES,
            action_selection="deterministic",
        )
        phase["confirm_eval"] = confirm["summary"]
        phase["confirm_primary_metrics"] = confirm["primary_metrics"]
        phase["promotion_decision"] = a1_gate_class(confirm["primary_metrics"])
        phase["phase_status"] = PHASE_COMPLETE
        phase["terminal_phase_status"] = PHASE_COMPLETE
    except Exception as exc:
        phase["phase_status"] = PHASE_CONFIRM_FAILED
        phase["terminal_phase_status"] = PHASE_CONFIRM_FAILED
        phase["promotion_decision"] = PHASE_CONFIRM_FAILED
        phase["failure_message"] = f"confirm_eval_failed: {exc}"
    save_manifest(manifest_path, manifest)
    return phase


def run_a2_phase(
    args: argparse.Namespace,
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    source_phase: dict[str, Any],
    continuation_checkpoint: str | None = None,
    continuation_iterations: int | None = None,
) -> dict[str, Any]:
    family = get_family_from_phase(source_phase)
    phase_kind = "a2_continue" if continuation_checkpoint is not None else "a2_initial"
    checkpoint_root = Path(manifest["target_root"]) / f"{phase_kind}_{family.family_id}"
    if continuation_checkpoint is not None:
        params = build_a2_continuation_params(args, family, checkpoint_root, restore_checkpoint=continuation_checkpoint)
    else:
        params = build_a2_training_params(args, family, checkpoint_root, warmstart_checkpoint=best_checkpoint_from_phase(source_phase))
    if continuation_iterations is not None:
        params["train_iterations"] = int(continuation_iterations)
    phase = new_phase_record(
        manifest,
        phase_kind=phase_kind,
        family_spec=family,
        training_phase="locomotion_propulsion_robust",
        checkpoint_root=checkpoint_root,
        restore_from_checkpoint=params.get("restore_from_checkpoint") or params.get("warmstart_motion_checkpoint"),
        params=params,
    )
    save_manifest(manifest_path, manifest)
    run_training_subprocess(args, manifest_path, manifest, phase, build_agent_command(args, params))
    if str(phase.get("terminal_phase_status")) != PHASE_COMPLETE:
        save_manifest(manifest_path, manifest)
        return phase
    try:
        confirm = run_summary_confirm_eval(
            args,
            checkpoint_path=best_checkpoint_from_phase(phase),
            output_root=checkpoint_root,
            stem="confirm_eval",
            episodes=A2_CONFIRM_EPISODES,
            action_selection="deterministic",
        )
        phase["confirm_eval"] = confirm["summary"]
        phase["confirm_primary_metrics"] = confirm["primary_metrics"]
        phase["promotion_decision"] = a2_gate_class(confirm["primary_metrics"])
        phase["phase_status"] = PHASE_COMPLETE
        phase["terminal_phase_status"] = PHASE_COMPLETE
    except Exception as exc:
        phase["phase_status"] = PHASE_CONFIRM_FAILED
        phase["terminal_phase_status"] = PHASE_CONFIRM_FAILED
        phase["promotion_decision"] = PHASE_CONFIRM_FAILED
        phase["failure_message"] = f"confirm_eval_failed: {exc}"
    save_manifest(manifest_path, manifest)
    return phase


def run_forage_phase(
    args: argparse.Namespace,
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    source_phase: dict[str, Any],
    restore_checkpoint: str | None = None,
) -> dict[str, Any]:
    family = get_family_from_phase(source_phase)
    phase_kind = "forage_continue" if restore_checkpoint is not None else "forage_initial"
    checkpoint_root = Path(manifest["target_root"]) / f"{phase_kind}_{family.family_id}"
    if restore_checkpoint is not None:
        params = build_forage_continuation_params(args, family, checkpoint_root, restore_checkpoint=restore_checkpoint)
    else:
        params = build_forage_training_params(args, family, checkpoint_root, warmstart_checkpoint=best_checkpoint_from_phase(source_phase), train_iterations=100)
    phase = new_phase_record(
        manifest,
        phase_kind=phase_kind,
        family_spec=family,
        training_phase="forage_full",
        checkpoint_root=checkpoint_root,
        restore_from_checkpoint=params.get("restore_from_checkpoint") or params.get("warmstart_motion_checkpoint"),
        params=params,
    )
    save_manifest(manifest_path, manifest)
    run_training_subprocess(args, manifest_path, manifest, phase, build_agent_command(args, params))
    if str(phase.get("terminal_phase_status")) != PHASE_COMPLETE:
        save_manifest(manifest_path, manifest)
        return phase
    try:
        best_confirm = run_summary_confirm_eval(
            args,
            checkpoint_path=best_checkpoint_from_phase(phase),
            output_root=checkpoint_root,
            stem="confirm_eval_best_both",
            episodes=FORAGE_CONFIRM_EPISODES,
            action_selection="both",
        )
        final_checkpoint_path = str(phase.get("final_checkpoint_path") or best_checkpoint_from_phase(phase))
        final_confirm = run_summary_confirm_eval(
            args,
            checkpoint_path=final_checkpoint_path,
            output_root=checkpoint_root,
            stem="confirm_eval_final_both",
            episodes=FORAGE_CONFIRM_EPISODES,
            action_selection="both",
        )
        phase["confirm_eval_best"] = best_confirm["summary"]
        phase["confirm_eval_final"] = final_confirm["summary"]
        phase["confirm_primary_metrics_best"] = normalize_metric_dict(best_confirm["summary"].get("deterministic_eval"))
        phase["confirm_primary_metrics_final"] = normalize_metric_dict(final_confirm["summary"].get("deterministic_eval"))
        better = min(
            [
                {"label": "best", "checkpoint_path": best_checkpoint_from_phase(phase), "summary": best_confirm["summary"]},
                {"label": "final", "checkpoint_path": final_checkpoint_path, "summary": final_confirm["summary"]},
            ],
            key=lambda record: forage_candidate_rank_key(record["summary"]),
        )
        better_metrics = normalize_metric_dict(better["summary"].get("deterministic_eval"))
        phase["selected_confirm_checkpoint_label"] = better["label"]
        phase["selected_confirm_checkpoint_path"] = better["checkpoint_path"]
        phase["selected_confirm_primary_metrics"] = better_metrics
        phase["motion_alive_gate_passed"] = forage_motion_alive(better_metrics)
        phase["forage_signal_gate_passed"] = forage_signal_alive(better_metrics)
        phase["promotion_decision"] = (
            "forage_signal"
            if phase["forage_signal_gate_passed"]
            else ("motion_alive_only" if phase["motion_alive_gate_passed"] else "no_signal")
        )
        phase["phase_status"] = PHASE_COMPLETE
        phase["terminal_phase_status"] = PHASE_COMPLETE
    except Exception as exc:
        phase["phase_status"] = PHASE_CONFIRM_FAILED
        phase["terminal_phase_status"] = PHASE_CONFIRM_FAILED
        phase["promotion_decision"] = PHASE_CONFIRM_FAILED
        phase["failure_message"] = f"confirm_eval_failed: {exc}"
    save_manifest(manifest_path, manifest)
    return phase


def recover_if_needed(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any]) -> bool:
    phase = current_training_phase(manifest)
    if phase is None:
        return False
    recover_training_subprocess(args, manifest_path, manifest, phase)
    return True


def ensure_resume_allowed(args: argparse.Namespace, manifest_path: Path, target_root: Path) -> dict[str, Any]:
    existing_manifest = try_load_json(manifest_path)
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
    save_manifest(manifest_path, manifest)
    return manifest


def run_baseline_phase(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any], family: FamilySpec) -> dict[str, Any]:
    checkpoint_root = Path(manifest["target_root"])
    phase = new_phase_record(
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
    save_manifest(manifest_path, manifest)
    try:
        baseline = run_random_baseline(args, output_root=checkpoint_root, family=family)
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
    save_manifest(manifest_path, manifest)
    return phase


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path).resolve()
    target_root = Path(args.target_root).resolve()
    manifest = ensure_resume_allowed(args, manifest_path, target_root)

    log(f"controller_started manifest={manifest_path} target_root={target_root} smoke={args.smoke}")

    while True:
        save_manifest(manifest_path, manifest)
        if terminal_status(manifest):
            save_manifest(manifest_path, manifest)
            log(f"controller_stopping status={manifest['status']} stop_reason={manifest.get('stop_reason')}")
            return

        if remaining_hours(manifest) <= 0.0:
            stop_campaign(manifest, status="deadline_reached", stop_reason="max_wall_clock_exhausted")
            save_manifest(manifest_path, manifest)
            continue

        if recover_if_needed(args, manifest_path, manifest):
            continue

        if maybe_run_smoke(args, manifest_path, manifest, target_root):
            continue

        initial_specs = initial_families()
        next_initial = next_pending_a1_initial_family(manifest, initial_specs)
        if next_initial is not None:
            next_family, retry_source, attempt_index, env_runner_override = next_initial
            run_a1_phase(
                args,
                manifest_path,
                manifest,
                family=next_family,
                phase_kind="a1_initial" if retry_source is None else "a1_initial_retry",
                train_iterations=60,
                attempt_index=attempt_index,
                retry_of_phase=retry_source,
                num_env_runners_override=env_runner_override,
            )
            continue

        all_a1_records = completed_phase_records(manifest, *A1_ALL_PHASE_KINDS)
        a1_full = [record for record in all_a1_records if str(record.get("promotion_decision")) == "full_pass"]
        selected_for_a2: list[dict[str, Any]]
        if a1_full:
            selected_for_a2 = sorted(a1_full, key=lambda record: candidate_rank_key(record["confirm_primary_metrics"]))[:2]
        else:
            refine_done = terminal_phase_records(manifest, "a1_refine")
            if not refine_done:
                near_initial = [
                    record
                    for record in completed_phase_records(manifest, *A1_INITIAL_PHASE_KINDS)
                    if str(record.get("promotion_decision")) == "near_pass"
                ]
                if near_initial:
                    best_near_initial = choose_best_candidate(near_initial)
                    derived_specs = refinement_families(get_family_from_phase(best_near_initial))
                    if len(refine_done) < len(derived_specs):
                        next_family = derived_specs[len(refine_done)]
                        run_a1_phase(args, manifest_path, manifest, family=next_family, phase_kind="a1_refine", train_iterations=40)
                        continue
                else:
                    stop_campaign(manifest, status="blocked", stop_reason="blocked_no_viable_a1")
                    save_manifest(manifest_path, manifest)
                    continue

            all_a1_records = completed_phase_records(manifest, *A1_ALL_PHASE_KINDS)
            a1_full = [record for record in all_a1_records if str(record.get("promotion_decision")) == "full_pass"]
            a1_near = [record for record in all_a1_records if str(record.get("promotion_decision")) == "near_pass"]
            if a1_full:
                selected_for_a2 = sorted(a1_full, key=lambda record: candidate_rank_key(record["confirm_primary_metrics"]))[:2]
            elif a1_near:
                selected_for_a2 = [choose_best_candidate(a1_near)]
            else:
                stop_campaign(manifest, status="blocked", stop_reason="blocked_no_viable_a1")
                save_manifest(manifest_path, manifest)
                continue

        manifest.setdefault("artifacts", {})["selected_a1_family_ids"] = [record["family_id"] for record in selected_for_a2]
        save_manifest(manifest_path, manifest)

        a2_records = completed_phase_records(manifest, *A2_PHASE_KINDS)
        selected_ids = {record["family_id"] for record in selected_for_a2}
        initial_a2_by_family = {record["family_id"]: record for record in terminal_phase_records(manifest, "a2_initial")}
        missing_initial_a2 = [record for record in selected_for_a2 if record["family_id"] not in initial_a2_by_family]
        if missing_initial_a2:
            run_a2_phase(args, manifest_path, manifest, source_phase=missing_initial_a2[0])
            continue

        promoted_a2_records = [record for record in a2_records if record["family_id"] in selected_ids]
        a2_full = [record for record in promoted_a2_records if str(record.get("promotion_decision")) == "full_pass"]
        if a2_full:
            best_a2 = choose_best_candidate(a2_full)
            manifest.setdefault("artifacts", {})["best_a2_phase_id"] = best_a2["phase_id"]
            manifest["artifacts"]["best_a2_checkpoint_path"] = best_checkpoint_from_phase(best_a2)
            save_manifest(manifest_path, manifest)
        else:
            a2_near = [record for record in promoted_a2_records if str(record.get("promotion_decision")) == "near_pass"]
            continuation_done = terminal_phase_records(manifest, "a2_continue")
            if a2_near and not continuation_done and remaining_hours(manifest) > 6.0:
                best_near_a2 = choose_best_candidate(a2_near)
                run_a2_phase(
                    args,
                    manifest_path,
                    manifest,
                    source_phase=best_near_a2,
                    continuation_checkpoint=best_checkpoint_from_phase(best_near_a2),
                    continuation_iterations=40,
                )
                continue
            promoted_a2_records = [record for record in completed_phase_records(manifest, *A2_PHASE_KINDS) if record["family_id"] in selected_ids]
            a2_full = [record for record in promoted_a2_records if str(record.get("promotion_decision")) == "full_pass"]
            if not a2_full:
                stop_campaign(manifest, status="blocked", stop_reason="blocked_no_robust_swimmer")
                save_manifest(manifest_path, manifest)
                continue
            best_a2 = choose_best_candidate(a2_full)
            manifest.setdefault("artifacts", {})["best_a2_phase_id"] = best_a2["phase_id"]
            manifest["artifacts"]["best_a2_checkpoint_path"] = best_checkpoint_from_phase(best_a2)
            save_manifest(manifest_path, manifest)

        best_a2_phase_id = manifest["artifacts"]["best_a2_phase_id"]
        best_a2_phase = next(record for record in completed_phase_records(manifest, *A2_PHASE_KINDS) if record["phase_id"] == best_a2_phase_id)

        if not terminal_phase_records(manifest, "baseline_random"):
            run_baseline_phase(args, manifest_path, manifest, get_family_from_phase(best_a2_phase))
            continue

        forage_initial_records = terminal_phase_records(manifest, "forage_initial")
        if not forage_initial_records:
            run_forage_phase(args, manifest_path, manifest, source_phase=best_a2_phase)
            continue

        all_forage_records = completed_phase_records(manifest, *FORAGE_PHASE_KINDS)
        if not all_forage_records:
            stop_campaign(manifest, status="failed", stop_reason="no_successful_forage_phase")
            save_manifest(manifest_path, manifest)
            continue
        best_forage_phase = min(
            all_forage_records,
            key=lambda record: forage_candidate_rank_key(record.get("confirm_eval_best") if record.get("selected_confirm_checkpoint_label") == "best" else record.get("confirm_eval_final")),
        )
        manifest.setdefault("artifacts", {})["best_forage_phase_id"] = best_forage_phase["phase_id"]
        manifest["artifacts"]["best_forage_checkpoint_path"] = best_forage_phase["selected_confirm_checkpoint_path"]
        save_manifest(manifest_path, manifest)

        if bool(best_forage_phase.get("forage_signal_gate_passed")) and remaining_hours(manifest) > 4.0 and not terminal_phase_records(manifest, "forage_continue"):
            run_forage_phase(
                args,
                manifest_path,
                manifest,
                source_phase=best_a2_phase,
                restore_checkpoint=str(best_forage_phase["selected_confirm_checkpoint_path"]),
            )
            continue

        final_forage_records = completed_phase_records(manifest, *FORAGE_PHASE_KINDS)
        successful_forage = [record for record in final_forage_records if bool(record.get("forage_signal_gate_passed"))]
        if successful_forage:
            best_success = min(
                successful_forage,
                key=lambda record: forage_candidate_rank_key(record.get("confirm_eval_best") if record.get("selected_confirm_checkpoint_label") == "best" else record.get("confirm_eval_final")),
            )
            manifest.setdefault("artifacts", {})["best_forage_phase_id"] = best_success["phase_id"]
            manifest["artifacts"]["best_forage_checkpoint_path"] = best_success["selected_confirm_checkpoint_path"]
            stop_campaign(manifest, status="success", stop_reason="forage_signal_confirmed")
            save_manifest(manifest_path, manifest)
            continue

        stop_campaign(manifest, status="failed", stop_reason="no_forage_signal")
        save_manifest(manifest_path, manifest)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        args = parse_args()
        manifest_path = Path(args.manifest_path).resolve()
        target_root = Path(args.target_root).resolve()
        manifest = try_load_json(manifest_path)
        if manifest is None:
            manifest = init_manifest(args, manifest_path=manifest_path, target_root=target_root)
        manifest["controller_exception"] = str(exc)
        manifest["controller_traceback"] = traceback.format_exc()
        stop_campaign(manifest, status="failed", stop_reason="controller_exception")
        save_manifest(manifest_path, manifest)
        log(f"controller_exception: {exc}")
        raise
