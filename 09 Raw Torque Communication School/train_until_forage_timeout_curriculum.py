from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import traceback
from typing import Any

import train_until_propulsion_target as base


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DONOR_CHECKPOINT = (
    SCRIPT_DIR
    / "rllib_checkpoints_v9_transfer_campaign"
    / "r2_initial_limit_medium"
    / "checkpoint_00040"
)
DEFAULT_MANIFEST_PATH = SCRIPT_DIR / "forage_timeout_curriculum_manifest.json"
DEFAULT_TARGET_ROOT = SCRIPT_DIR / "rllib_checkpoints_v9_forage_timeout_curriculum"
DEFAULT_CONTROLLER_POLL_SECONDS = 30
DEFAULT_MAX_WALL_CLOCK_HOURS = 12.0
DONOR_CONFIRM_EPISODES = 10
FORAGE_CONFIRM_EPISODES = 10
RANDOM_BASELINE_EPISODES = 20
PHASE_COMPLETE = base.PHASE_COMPLETE
PHASE_CONFIRM_FAILED = base.PHASE_CONFIRM_FAILED
TERMINAL_PHASE_STATUSES = base.TERMINAL_PHASE_STATUSES
TERMINAL_STATUSES = {
    "success",
    "failed",
    "deadline_reached",
    "blocked_stage1_no_food",
    "blocked_stage2_no_food",
    "blocked_stage3_no_food",
    "blocked_stage4_no_food",
}


@dataclass(frozen=True)
class StageSpec:
    stage_index: int
    stage_name: str
    idle_timeout_steps: int
    step_cost: float
    train_iterations: int
    food_gate_threshold: float

    @property
    def initial_phase_kind(self) -> str:
        return self.stage_name

    @property
    def continuation_phase_kind(self) -> str:
        return f"{self.stage_name}_continue"


@dataclass(frozen=True)
class PhaseBudget:
    default_hours: float
    buffer_hours: float


def default_python_executable() -> str:
    return base.default_python_executable()


def now_local() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def now_iso() -> str:
    return now_local().isoformat(timespec="seconds")


def log(message: str) -> None:
    base.log(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the unattended V9 forage timeout curriculum.")
    parser.add_argument("--python-executable", type=str, default=default_python_executable())
    parser.add_argument("--donor-checkpoint", type=str, default=str(DEFAULT_DONOR_CHECKPOINT))
    parser.add_argument("--manifest-path", type=str, default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--target-root", type=str, default=str(DEFAULT_TARGET_ROOT))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-device", type=str, default="cuda")
    parser.add_argument("--max-wall-clock-hours", type=float, default=DEFAULT_MAX_WALL_CLOCK_HOURS)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_CONTROLLER_POLL_SECONDS)
    return parser.parse_args()


def stage_specs() -> list[StageSpec]:
    return [
        StageSpec(1, "forage_stage1_timeout500", 500, 0.0005, 80, 0.0),
        StageSpec(2, "forage_stage2_timeout1000", 1000, 0.0010, 80, 0.03),
        StageSpec(3, "forage_stage3_timeout2000", 2000, 0.0015, 100, 0.07),
        StageSpec(4, "forage_stage4_timeout5000", 5000, 0.0020, 120, 0.10),
    ]


def stage_by_index(stage_index: int) -> StageSpec:
    for stage in stage_specs():
        if int(stage.stage_index) == int(stage_index):
            return stage
    raise KeyError(f"Unknown stage index: {stage_index}")


def init_manifest(args: argparse.Namespace, *, manifest_path: Path, target_root: Path) -> dict[str, Any]:
    started_at = now_local()
    donor_checkpoint = str(Path(args.donor_checkpoint).resolve())
    return {
        "status": "running",
        "stop_reason": None,
        "campaign_kind": "forage_timeout_curriculum",
        "manifest_version": 1,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": None,
        "updated_at": started_at.isoformat(timespec="seconds"),
        "deadline_at": (started_at + timedelta(hours=float(args.max_wall_clock_hours))).isoformat(timespec="seconds"),
        "manifest_path": str(manifest_path.resolve()),
        "target_root": str(target_root.resolve()),
        "donor_checkpoint": donor_checkpoint,
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


def terminal_status(manifest: dict[str, Any]) -> bool:
    return str(manifest.get("status")) in TERMINAL_STATUSES


def stop_campaign(manifest: dict[str, Any], *, status: str, stop_reason: str) -> None:
    manifest["status"] = status
    manifest["stop_reason"] = stop_reason
    manifest["finished_at"] = now_iso()


def blocked_status(stage: StageSpec) -> str:
    return f"blocked_stage{int(stage.stage_index)}_no_food"


def phase_budget_for_kind(phase_kind: str) -> PhaseBudget:
    if phase_kind == "donor_confirm":
        return PhaseBudget(default_hours=0.25, buffer_hours=0.10)
    if phase_kind == "baseline_random":
        return PhaseBudget(default_hours=0.40, buffer_hours=0.10)
    if phase_kind == "forage_stage1_timeout500":
        return PhaseBudget(default_hours=1.25, buffer_hours=0.25)
    if phase_kind == "forage_stage1_timeout500_continue":
        return PhaseBudget(default_hours=1.00, buffer_hours=0.25)
    if phase_kind == "forage_stage2_timeout1000":
        return PhaseBudget(default_hours=1.50, buffer_hours=0.25)
    if phase_kind == "forage_stage2_timeout1000_continue":
        return PhaseBudget(default_hours=1.00, buffer_hours=0.25)
    if phase_kind == "forage_stage3_timeout2000":
        return PhaseBudget(default_hours=2.00, buffer_hours=0.25)
    if phase_kind == "forage_stage3_timeout2000_continue":
        return PhaseBudget(default_hours=1.00, buffer_hours=0.25)
    if phase_kind == "forage_stage4_timeout5000":
        return PhaseBudget(default_hours=2.50, buffer_hours=0.25)
    if phase_kind == "forage_stage4_timeout5000_continue":
        return PhaseBudget(default_hours=1.25, buffer_hours=0.25)
    return PhaseBudget(default_hours=1.00, buffer_hours=0.25)


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


def phase_root(target_root: Path, phase_kind: str) -> Path:
    return target_root / phase_kind


def metric_float(mapping: dict[str, Any] | None, key: str, default: float) -> float:
    if not mapping:
        return float(default)
    value = mapping.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not number == number:
        return float(default)
    return number


def locate_training_metadata(checkpoint_path: Path) -> Path | None:
    current = checkpoint_path.resolve()
    for candidate_dir in [current, *current.parents]:
        metadata_path = candidate_dir / "training_metadata.json"
        if metadata_path.exists():
            return metadata_path
    return None


def donor_family_from_checkpoint(checkpoint_path: Path) -> base.FamilySpec:
    metadata_path = locate_training_metadata(checkpoint_path)
    env_config: dict[str, Any] = {}
    if metadata_path is not None:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(payload.get("env_config"), dict):
                env_config = payload["env_config"]
        except (OSError, json.JSONDecodeError):
            env_config = {}
    parent_name = checkpoint_path.parent.name
    family_id = parent_name.split("_", 2)[-1] if parent_name.count("_") >= 2 else parent_name
    return base.FamilySpec(
        family_id=family_id,
        joint_soft_limit_start_ratio=float(env_config.get("joint_soft_limit_start_ratio", 0.65)),
        joint_soft_limit_stiffness=float(env_config.get("joint_soft_limit_stiffness", 22.0)),
        joint_soft_limit_damping=float(env_config.get("joint_soft_limit_damping", 3.0)),
        activation_time_constant=float(env_config.get("activation_time_constant", 0.12)),
        propulsion_near_limit_weight=float(env_config.get("propulsion_near_limit_weight", -0.24)),
        propulsion_saturation_weight=float(env_config.get("propulsion_saturation_weight", -0.12)),
        propulsion_torque_weight=float(env_config.get("propulsion_torque_weight", -0.06)),
    )


def ensure_family_artifact(manifest: dict[str, Any]) -> base.FamilySpec:
    family_payload = manifest.setdefault("artifacts", {}).get("donor_family")
    if isinstance(family_payload, dict):
        return base.FamilySpec(
            family_id=str(family_payload.get("family_id", "forage_timeout_curriculum")),
            joint_soft_limit_start_ratio=float(family_payload.get("joint_soft_limit_start_ratio", 0.65)),
            joint_soft_limit_stiffness=float(family_payload.get("joint_soft_limit_stiffness", 22.0)),
            joint_soft_limit_damping=float(family_payload.get("joint_soft_limit_damping", 3.0)),
            activation_time_constant=float(family_payload.get("activation_time_constant", 0.12)),
            propulsion_near_limit_weight=float(family_payload.get("propulsion_near_limit_weight", -0.24)),
            propulsion_saturation_weight=float(family_payload.get("propulsion_saturation_weight", -0.12)),
            propulsion_torque_weight=float(family_payload.get("propulsion_torque_weight", -0.06)),
        )
    family = donor_family_from_checkpoint(Path(manifest["donor_checkpoint"]))
    manifest["artifacts"]["donor_family"] = family.to_dict()
    return family


def donor_confirm_record(manifest: dict[str, Any]) -> dict[str, Any] | None:
    records = [phase for phase in manifest.get("phases", []) if str(phase.get("phase_kind")) == "donor_confirm"]
    return records[-1] if records else None


def baseline_record(manifest: dict[str, Any]) -> dict[str, Any] | None:
    records = [phase for phase in manifest.get("phases", []) if str(phase.get("phase_kind")) == "baseline_random"]
    return records[-1] if records else None


def stage_initial_record(manifest: dict[str, Any], stage: StageSpec) -> dict[str, Any] | None:
    records = [phase for phase in manifest.get("phases", []) if str(phase.get("phase_kind")) == stage.initial_phase_kind]
    return records[-1] if records else None


def stage_continuation_record(manifest: dict[str, Any], stage: StageSpec) -> dict[str, Any] | None:
    records = [phase for phase in manifest.get("phases", []) if str(phase.get("phase_kind")) == stage.continuation_phase_kind]
    return records[-1] if records else None


def stage_terminal_record(manifest: dict[str, Any], stage: StageSpec) -> dict[str, Any] | None:
    continuation = stage_continuation_record(manifest, stage)
    if continuation is not None:
        return continuation
    return stage_initial_record(manifest, stage)


def stage_promoted_record(manifest: dict[str, Any], stage: StageSpec) -> dict[str, Any] | None:
    record = stage_terminal_record(manifest, stage)
    if record is None:
        return None
    if str(record.get("promotion_decision")) == "food_gate_pass":
        return record
    return None


def pending_postprocess_phase(manifest: dict[str, Any]) -> dict[str, Any] | None:
    for phase in reversed(list(manifest.get("phases", []))):
        phase_kind = str(phase.get("phase_kind"))
        terminal_phase_status = str(phase.get("terminal_phase_status")) if phase.get("terminal_phase_status") is not None else None
        if phase_kind in {"donor_confirm", "baseline_random"} and terminal_phase_status not in TERMINAL_PHASE_STATUSES:
            return phase
        if phase_kind.startswith("forage_stage") and terminal_phase_status == PHASE_COMPLETE and phase.get("promotion_decision") is None:
            return phase
    return None


def stage_env_overrides(stage: StageSpec) -> dict[str, Any]:
    return {
        "reward_mode": "forage",
        "num_red_fish": 10,
        "num_blue_fish": 10,
        "num_red_pellets": 48,
        "num_blue_pellets": 48,
        "time_limit": 300,
        "observation_profile": "full_v9",
        "history_length": 8,
        "food_respawn_mode": "deplete",
        "forage_timeout_mode": "reset_on_food",
        "forage_idle_timeout_steps": int(stage.idle_timeout_steps),
        "forage_time_context_mode": "idle_budget_remaining",
        "step_cost": float(stage.step_cost),
    }


def build_stage_training_params(
    args: argparse.Namespace,
    family: base.FamilySpec,
    stage: StageSpec,
    checkpoint_root: Path,
    *,
    restore_checkpoint: str | None,
    warmstart_motion_checkpoint: str | None,
    train_iterations: int,
    learning_rate: float,
    entropy_coeff: float,
    checkpoint_every_iterations: int = 20,
    num_env_runners: int = 8,
    num_envs_per_runner: int = 2,
    light_eval_episodes: int = 4,
    train_batch_size: int = 16000,
    minibatch_size: int = 2048,
    num_epochs: int = 6,
    rollout_fragment_length: int = 500,
) -> dict[str, Any]:
    params = {
        "policy_stack": "new",
        "training_phase": "forage_full",
        "device": str(args.device),
        "checkpoint_root": str(checkpoint_root.resolve()),
        "train_iterations": int(train_iterations),
        "checkpoint_every_iterations": int(checkpoint_every_iterations),
        "num_env_runners": int(num_env_runners),
        "num_envs_per_runner": int(num_envs_per_runner),
        "light_eval_episodes": int(light_eval_episodes),
        "train_batch_size": int(train_batch_size),
        "minibatch_size": int(minibatch_size),
        "num_epochs": int(num_epochs),
        "rollout_fragment_length": int(rollout_fragment_length),
        "learning_rate": float(learning_rate),
        "entropy_coeff": float(entropy_coeff),
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
        "food_respawn_mode": "deplete",
        "forage_timeout_mode": "reset_on_food",
        "forage_idle_timeout_steps": int(stage.idle_timeout_steps),
        "forage_time_context_mode": "idle_budget_remaining",
        "step_cost": float(stage.step_cost),
    }
    if restore_checkpoint is not None:
        params["restore_from_checkpoint"] = str(restore_checkpoint)
    elif warmstart_motion_checkpoint is not None:
        params["warmstart_motion_checkpoint"] = str(warmstart_motion_checkpoint)
    return params


def deterministic_metrics(nested_summary: dict[str, Any]) -> dict[str, Any]:
    return base.normalize_metric_dict(nested_summary.get("deterministic_eval"))


def checkpoint_rank_key(nested_summary: dict[str, Any]) -> tuple[float, float, float]:
    metrics = deterministic_metrics(nested_summary)
    return (
        metric_float(metrics, "mean_pellets_per_fish", float("-inf")),
        metric_float(metrics, "mean_forward_velocity", float("-inf")),
        metric_float(metrics, "mean_abs_activation", float("-inf")),
    )


def motion_alive_gate(metrics: dict[str, Any]) -> bool:
    return (
        metric_float(metrics, "mean_abs_activation", 0.0) >= 0.05
        and metric_float(metrics, "fraction_joints_quiet_steps", 1.0) < 0.80
    )


def food_gate_passed(stage: StageSpec, metrics: dict[str, Any]) -> bool:
    pellets = metric_float(metrics, "mean_pellets_per_fish", 0.0)
    if stage.stage_index == 4:
        return pellets >= float(stage.food_gate_threshold) and metric_float(metrics, "mean_forward_velocity", 0.0) > 0.0
    if stage.stage_index == 1:
        return pellets > 0.0
    return pellets >= float(stage.food_gate_threshold)


def finish_donor_confirm(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any], phase: dict[str, Any]) -> None:
    checkpoint_path = str(manifest["donor_checkpoint"])
    try:
        confirm = base.run_summary_confirm_eval(
            args,
            checkpoint_path=checkpoint_path,
            output_root=Path(phase["checkpoint_root"]),
            stem="donor_confirm_eval",
            episodes=int(phase.get("confirm_episodes", DONOR_CONFIRM_EPISODES)),
            action_selection="deterministic",
        )
        phase["confirm_eval"] = confirm["summary"]
        phase["confirm_primary_metrics"] = confirm["primary_metrics"]
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
        stop_campaign(manifest, status="failed", stop_reason="donor_confirm_failed")
    phase["finished_at"] = now_iso()
    base.save_manifest(manifest_path, manifest)


def finish_stage_phase(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any], phase: dict[str, Any]) -> None:
    checkpoint_root = Path(phase["checkpoint_root"])
    confirm_episodes = int(phase.get("confirm_episodes", FORAGE_CONFIRM_EPISODES))
    stage = stage_by_index(int(phase["stage_index"]))
    best_confirm = base.run_summary_confirm_eval(
        args,
        checkpoint_path=base.best_checkpoint_from_phase(phase),
        output_root=checkpoint_root,
        stem="confirm_eval_best_both",
        episodes=confirm_episodes,
        action_selection="both",
    )
    final_checkpoint_path = str(phase.get("final_checkpoint_path") or base.best_checkpoint_from_phase(phase))
    final_confirm = base.run_summary_confirm_eval(
        args,
        checkpoint_path=final_checkpoint_path,
        output_root=checkpoint_root,
        stem="confirm_eval_final_both",
        episodes=confirm_episodes,
        action_selection="both",
    )
    phase["confirm_eval_best"] = best_confirm["summary"]
    phase["confirm_eval_final"] = final_confirm["summary"]
    phase["confirm_primary_metrics_best"] = deterministic_metrics(best_confirm["summary"])
    phase["confirm_primary_metrics_final"] = deterministic_metrics(final_confirm["summary"])
    better = max(
        [
            {"label": "best", "checkpoint_path": base.best_checkpoint_from_phase(phase), "summary": best_confirm["summary"]},
            {"label": "final", "checkpoint_path": final_checkpoint_path, "summary": final_confirm["summary"]},
        ],
        key=lambda record: checkpoint_rank_key(record["summary"]),
    )
    better_metrics = deterministic_metrics(better["summary"])
    phase["selected_confirm_checkpoint_label"] = better["label"]
    phase["selected_confirm_checkpoint_path"] = better["checkpoint_path"]
    phase["selected_confirm_primary_metrics"] = better_metrics
    phase["motion_alive_gate_passed"] = motion_alive_gate(better_metrics)
    phase["food_gate_threshold"] = float(stage.food_gate_threshold)
    phase["food_gate_passed"] = food_gate_passed(stage, better_metrics)
    if bool(phase["food_gate_passed"]):
        phase["promotion_decision"] = "food_gate_pass"
    elif bool(phase["motion_alive_gate_passed"]):
        phase["promotion_decision"] = "motion_alive_only"
    else:
        phase["promotion_decision"] = "no_motion_or_food"
    phase["phase_status"] = PHASE_COMPLETE
    phase["terminal_phase_status"] = PHASE_COMPLETE
    phase["finished_at"] = phase.get("finished_at") or now_iso()
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
        elif phase_kind == "baseline_random":
            family = ensure_family_artifact(manifest)
            stage1 = stage_specs()[0]
            baseline = base.run_random_baseline(
                args,
                output_root=Path(manifest["target_root"]),
                family=family,
                env_overrides=stage_env_overrides(stage1),
                summary_stem="random_policy_baseline_timeout_curriculum",
            )
            phase["random_baseline"] = baseline["summary"]
            phase["random_baseline_json_path"] = baseline["summary_json_path"]
            phase["random_baseline_csv_path"] = baseline["summary_csv_path"]
            phase["phase_status"] = PHASE_COMPLETE
            phase["terminal_phase_status"] = PHASE_COMPLETE
            phase["promotion_decision"] = "baseline_complete"
            phase["finished_at"] = phase.get("finished_at") or now_iso()
            base.save_manifest(manifest_path, manifest)
        elif phase_kind.startswith("forage_stage"):
            finish_stage_phase(args, manifest_path, manifest, phase)
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
        family_spec=ensure_family_artifact(manifest),
        training_phase=None,
        checkpoint_root=checkpoint_root,
        restore_from_checkpoint=str(manifest["donor_checkpoint"]),
        params={"episodes": DONOR_CONFIRM_EPISODES},
    )
    phase["confirm_episodes"] = DONOR_CONFIRM_EPISODES
    base.save_manifest(manifest_path, manifest)
    finish_donor_confirm(args, manifest_path, manifest, phase)
    return phase


def run_random_baseline_phase(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    stage1 = stage_specs()[0]
    family = ensure_family_artifact(manifest)
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
        baseline = base.run_random_baseline(
            args,
            output_root=checkpoint_root,
            family=family,
            env_overrides=stage_env_overrides(stage1),
            summary_stem="random_policy_baseline_timeout_curriculum",
        )
        phase["random_baseline"] = baseline["summary"]
        phase["random_baseline_json_path"] = baseline["summary_json_path"]
        phase["random_baseline_csv_path"] = baseline["summary_csv_path"]
        phase["promotion_decision"] = "baseline_complete"
        phase["phase_status"] = PHASE_COMPLETE
        phase["terminal_phase_status"] = PHASE_COMPLETE
        manifest.setdefault("artifacts", {})["random_baseline_summary"] = baseline["summary"]
    except Exception as exc:
        phase["phase_status"] = PHASE_CONFIRM_FAILED
        phase["terminal_phase_status"] = PHASE_CONFIRM_FAILED
        phase["promotion_decision"] = PHASE_CONFIRM_FAILED
        phase["failure_message"] = f"baseline_generation_failed: {exc}"
        stop_campaign(manifest, status="failed", stop_reason="baseline_generation_failed")
    phase["finished_at"] = now_iso()
    base.save_manifest(manifest_path, manifest)
    return phase


def run_stage_phase(
    args: argparse.Namespace,
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    stage: StageSpec,
    restore_checkpoint: str | None,
    warmstart_motion_checkpoint: str | None,
    continuation: bool,
    train_iterations: int,
    learning_rate: float,
    entropy_coeff: float,
    params_override: dict[str, Any] | None = None,
    confirm_episodes: int = FORAGE_CONFIRM_EPISODES,
) -> dict[str, Any]:
    family = ensure_family_artifact(manifest)
    phase_kind = stage.continuation_phase_kind if continuation else stage.initial_phase_kind
    default_checkpoint_root = phase_root(Path(manifest["target_root"]), phase_kind)
    if params_override is None:
        params = build_stage_training_params(
            args,
            family,
            stage,
            default_checkpoint_root,
            restore_checkpoint=restore_checkpoint,
            warmstart_motion_checkpoint=warmstart_motion_checkpoint,
            train_iterations=train_iterations,
            learning_rate=learning_rate,
            entropy_coeff=entropy_coeff,
        )
    else:
        params = dict(params_override)
    checkpoint_root = Path(params.get("checkpoint_root", default_checkpoint_root)).resolve()
    phase = base.new_phase_record(
        manifest,
        phase_kind=phase_kind,
        family_spec=family,
        training_phase="forage_full",
        checkpoint_root=checkpoint_root,
        restore_from_checkpoint=params.get("restore_from_checkpoint") or params.get("warmstart_motion_checkpoint"),
        params=params,
    )
    phase["stage_index"] = int(stage.stage_index)
    phase["stage_name"] = stage.stage_name
    phase["idle_timeout_steps"] = int(stage.idle_timeout_steps)
    phase["stage_step_cost"] = float(stage.step_cost)
    phase["confirm_episodes"] = int(confirm_episodes)
    base.save_manifest(manifest_path, manifest)
    base.run_training_subprocess(args, manifest_path, manifest, phase, base.build_agent_command(args, params))
    if str(phase.get("terminal_phase_status")) != PHASE_COMPLETE:
        return phase
    try:
        finish_stage_phase(args, manifest_path, manifest, phase)
    except Exception as exc:
        phase["phase_status"] = PHASE_CONFIRM_FAILED
        phase["terminal_phase_status"] = PHASE_CONFIRM_FAILED
        phase["promotion_decision"] = PHASE_CONFIRM_FAILED
        phase["failure_message"] = f"confirm_eval_failed: {exc}"
        phase["finished_at"] = phase.get("finished_at") or now_iso()
        base.save_manifest(manifest_path, manifest)
    return phase


def maybe_run_smoke(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any], target_root: Path) -> bool:
    if not args.smoke:
        return False
    if manifest.get("phases"):
        stop_campaign(manifest, status="success", stop_reason="smoke_complete")
        base.save_manifest(manifest_path, manifest)
        return True
    donor_phase = run_donor_confirm_phase(args, manifest_path, manifest)
    if str(donor_phase.get("terminal_phase_status")) != PHASE_COMPLETE:
        return True
    stage = stage_specs()[0]
    smoke_params = build_stage_training_params(
        args,
        ensure_family_artifact(manifest),
        stage,
        target_root / "smoke_forage_stage1_timeout500",
        restore_checkpoint=None,
        warmstart_motion_checkpoint=str(manifest["donor_checkpoint"]),
        train_iterations=1,
        learning_rate=2e-4,
        entropy_coeff=0.01,
        checkpoint_every_iterations=1,
        num_env_runners=1,
        num_envs_per_runner=1,
        light_eval_episodes=1,
        train_batch_size=500,
        minibatch_size=250,
        num_epochs=1,
        rollout_fragment_length=50,
    )
    phase = run_stage_phase(
        args,
        manifest_path,
        manifest,
        stage=stage,
        restore_checkpoint=None,
        warmstart_motion_checkpoint=str(manifest["donor_checkpoint"]),
        continuation=False,
        train_iterations=1,
        learning_rate=2e-4,
        entropy_coeff=0.01,
        params_override=smoke_params,
        confirm_episodes=1,
    )
    if str(phase.get("terminal_phase_status")) != PHASE_COMPLETE:
        stop_campaign(manifest, status="failed", stop_reason="smoke_stage1_failed")
    else:
        stop_campaign(manifest, status="success", stop_reason="smoke_complete")
    base.save_manifest(manifest_path, manifest)
    return True


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path).resolve()
    target_root = Path(args.target_root).resolve()
    manifest = ensure_resume_allowed(args, manifest_path, target_root)
    ensure_family_artifact(manifest)
    base.save_manifest(manifest_path, manifest)
    log(f"controller_started manifest={manifest_path} target_root={target_root} smoke={args.smoke}")

    while True:
        base.save_manifest(manifest_path, manifest)
        if terminal_status(manifest):
            base.save_manifest(manifest_path, manifest)
            log(f"controller_stopping status={manifest['status']} stop_reason={manifest.get('stop_reason')}")
            return
        if base.remaining_hours(manifest) <= 0.0:
            stop_campaign(manifest, status="deadline_reached", stop_reason="max_wall_clock_exhausted")
            base.save_manifest(manifest_path, manifest)
            continue
        if recover_if_needed(args, manifest_path, manifest):
            continue
        if maybe_run_smoke(args, manifest_path, manifest, target_root):
            continue

        donor_phase = donor_confirm_record(manifest)
        if donor_phase is None:
            if not can_start_phase(manifest, "donor_confirm"):
                stop_campaign(manifest, status="deadline_reached", stop_reason="insufficient_time_for_donor_confirm")
                base.save_manifest(manifest_path, manifest)
                continue
            run_donor_confirm_phase(args, manifest_path, manifest)
            continue
        if str(donor_phase.get("terminal_phase_status")) != PHASE_COMPLETE:
            stop_campaign(manifest, status="failed", stop_reason="donor_confirm_incomplete")
            base.save_manifest(manifest_path, manifest)
            continue

        baseline_phase = baseline_record(manifest)
        if baseline_phase is None:
            if not can_start_phase(manifest, "baseline_random"):
                stop_campaign(manifest, status="deadline_reached", stop_reason="insufficient_time_for_random_baseline")
                base.save_manifest(manifest_path, manifest)
                continue
            run_random_baseline_phase(args, manifest_path, manifest)
            continue
        if str(baseline_phase.get("terminal_phase_status")) != PHASE_COMPLETE:
            stop_campaign(manifest, status="failed", stop_reason="baseline_generation_incomplete")
            base.save_manifest(manifest_path, manifest)
            continue

        progressed = False
        stages = stage_specs()
        for stage in stages:
            initial_phase = stage_initial_record(manifest, stage)
            continuation_phase = stage_continuation_record(manifest, stage)

            if initial_phase is not None and str(initial_phase.get("terminal_phase_status")) != PHASE_COMPLETE:
                stop_campaign(manifest, status="failed", stop_reason=f"{stage.initial_phase_kind}_phase_failed")
                base.save_manifest(manifest_path, manifest)
                progressed = True
                break
            if continuation_phase is not None and str(continuation_phase.get("terminal_phase_status")) != PHASE_COMPLETE:
                stop_campaign(manifest, status="failed", stop_reason=f"{stage.continuation_phase_kind}_phase_failed")
                base.save_manifest(manifest_path, manifest)
                progressed = True
                break

            if initial_phase is None:
                if stage.stage_index == 1:
                    restore_checkpoint = None
                    warmstart_motion_checkpoint = str(manifest["donor_checkpoint"])
                else:
                    previous_stage = stage_promoted_record(manifest, stage_by_index(stage.stage_index - 1))
                    if previous_stage is None:
                        break
                    restore_checkpoint = str(previous_stage["selected_confirm_checkpoint_path"])
                    warmstart_motion_checkpoint = None
                if not can_start_phase(manifest, stage.initial_phase_kind):
                    stop_campaign(manifest, status="deadline_reached", stop_reason=f"insufficient_time_for_{stage.initial_phase_kind}")
                    base.save_manifest(manifest_path, manifest)
                    progressed = True
                    break
                run_stage_phase(
                    args,
                    manifest_path,
                    manifest,
                    stage=stage,
                    restore_checkpoint=restore_checkpoint,
                    warmstart_motion_checkpoint=warmstart_motion_checkpoint,
                    continuation=False,
                    train_iterations=stage.train_iterations,
                    learning_rate=2e-4,
                    entropy_coeff=0.01,
                )
                progressed = True
                break

            active_phase = continuation_phase or initial_phase
            if bool(active_phase.get("food_gate_passed")):
                if stage.stage_index == stages[-1].stage_index:
                    stop_campaign(manifest, status="success", stop_reason="stage4_food_gate_passed")
                    manifest.setdefault("artifacts", {})["best_forage_checkpoint_path"] = str(active_phase["selected_confirm_checkpoint_path"])
                    base.save_manifest(manifest_path, manifest)
                    progressed = True
                    break
                continue

            if continuation_phase is not None:
                stop_campaign(manifest, status=blocked_status(stage), stop_reason="stage_food_gate_failed_after_continuation")
                manifest.setdefault("artifacts", {})["best_forage_checkpoint_path"] = str(active_phase.get("selected_confirm_checkpoint_path", ""))
                base.save_manifest(manifest_path, manifest)
                progressed = True
                break

            if not bool(initial_phase.get("motion_alive_gate_passed")):
                stop_campaign(manifest, status=blocked_status(stage), stop_reason="stage_motion_alive_gate_failed")
                manifest.setdefault("artifacts", {})["best_forage_checkpoint_path"] = str(initial_phase.get("selected_confirm_checkpoint_path", ""))
                base.save_manifest(manifest_path, manifest)
                progressed = True
                break

            if base.remaining_hours(manifest) < 2.0:
                stop_campaign(manifest, status="deadline_reached", stop_reason=f"insufficient_time_for_{stage.continuation_phase_kind}")
                base.save_manifest(manifest_path, manifest)
                progressed = True
                break
            if not can_start_phase(manifest, stage.continuation_phase_kind):
                stop_campaign(manifest, status="deadline_reached", stop_reason=f"insufficient_time_for_{stage.continuation_phase_kind}")
                base.save_manifest(manifest_path, manifest)
                progressed = True
                break

            run_stage_phase(
                args,
                manifest_path,
                manifest,
                stage=stage,
                restore_checkpoint=str(initial_phase["selected_confirm_checkpoint_path"]),
                warmstart_motion_checkpoint=None,
                continuation=True,
                train_iterations=40,
                learning_rate=1e-4,
                entropy_coeff=0.01,
            )
            progressed = True
            break

        if progressed:
            continue

        final_stage = stage_promoted_record(manifest, stages[-1])
        if final_stage is not None:
            stop_campaign(manifest, status="success", stop_reason="stage4_food_gate_passed")
            manifest.setdefault("artifacts", {})["best_forage_checkpoint_path"] = str(final_stage["selected_confirm_checkpoint_path"])
            base.save_manifest(manifest_path, manifest)
            continue

        stop_campaign(manifest, status="failed", stop_reason="controller_reached_unexpected_state")
        base.save_manifest(manifest_path, manifest)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        manifest_path = DEFAULT_MANIFEST_PATH.resolve()
        target_root = DEFAULT_TARGET_ROOT.resolve()
        manifest = base.try_load_json(manifest_path)
        if manifest is None:
            class Args:
                donor_checkpoint = str(DEFAULT_DONOR_CHECKPOINT)
                max_wall_clock_hours = DEFAULT_MAX_WALL_CLOCK_HOURS
                smoke = False

            manifest = init_manifest(Args(), manifest_path=manifest_path, target_root=target_root)
        manifest["controller_exception"] = str(exc)
        manifest["controller_traceback"] = traceback.format_exc()
        stop_campaign(manifest, status="failed", stop_reason="controller_exception")
        base.save_manifest(manifest_path, manifest)
        raise
