"""Run an open-ended V8 training campaign until confirmation eval reaches the target multiple."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE_JSON = SCRIPT_DIR / "random_policy_baseline.json"
DEFAULT_MANIFEST_PATH = SCRIPT_DIR / "target_training_manifest.json"
DEFAULT_SMOKE_MANIFEST_PATH = SCRIPT_DIR / "target_training_manifest_smoke.json"
DEFAULT_TARGET_ROOT = SCRIPT_DIR / "rllib_checkpoints_target_v8_4x"
DEFAULT_RESUME_SOURCE = SCRIPT_DIR / "rllib_checkpoints_target_v8_attempt_01" / "checkpoint_final"
PROMOTED_COMBO_COUNT = 2
DEFAULT_CANDIDATE_SPACING = 40
BLOCKED_SCORE_THRESHOLD = 3.2
CONTINUATION_INCREMENT = 400
PHASE_STATUS_SCHEDULED = "scheduled"
PHASE_STATUS_TRAINING = "training"
PHASE_STATUS_CANDIDATE_EVAL = "candidate_eval"
PHASE_STATUS_CONFIRM_EVAL = "confirm_eval"
PHASE_STATUS_COMPLETE = "complete"
PHASE_STATUS_FAILED_RECOVERABLE = "failed_recoverable"
PHASE_STATUS_FAILED_TERMINAL = "failed_terminal"
TERMINAL_CAMPAIGN_STATUSES = {"success", "failed", "blocked"}


class RecoverablePhaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class FamilyTemplate:
    family_id: str
    initial_train_iterations: int
    continuation_train_iterations: int
    num_env_runners: int
    num_envs_per_runner: int
    rollout_fragment_length: int
    checkpoint_every_iterations: int
    light_eval_episodes: int
    time_limit: int
    num_red_fish: int
    num_blue_fish: int
    num_red_pellets: int
    num_blue_pellets: int
    train_batch_size: int
    minibatch_size: int
    num_epochs: int
    learning_rate: float
    entropy_coeff: float
    gamma: float
    gae_lambda: float
    fcnet_hiddens: str
    fcnet_activation: str
    max_seeds: int
    max_total_phases: int | None
    initial_restore_from: str | None = None
    continuation_requires_score: float | None = None
    continuation_requires_late_best: bool = False


@dataclass
class ComboState:
    family_id: str
    seed: int
    phase_index: int
    restore_from_checkpoint: str | None
    best_candidate_score: float | None = None
    best_candidate_checkpoint: str | None = None
    best_candidate_iteration: int | None = None
    best_phase_index: int | None = None
    insufficient_improvement_streak: int = 0
    last_phase_best_iteration: int | None = None
    last_phase_iterations: int | None = None
    retired: bool = False
    retired_reason: str | None = None
    total_added_iterations: int = 0

    @property
    def combo_id(self) -> str:
        return f"{self.family_id}_seed{self.seed}"

    @property
    def phase_id(self) -> str:
        return f"{self.combo_id}_phase{self.phase_index:02d}"


@dataclass
class ControllerState:
    family_next_seed: dict[str, int]
    current_combos: dict[str, ComboState]
    scheduled_round: list[ComboState]
    tested_seed_count: int
    incumbent_best_candidate_score: float | None
    incumbent_best_candidate_checkpoint: str | None


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def try_load_json(path: Path) -> dict[str, Any] | None:
    try:
        return load_json(path)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def file_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def ensure_clean_path(path: Path, *, force_clean: bool) -> None:
    if not path.exists():
        return
    if not force_clean:
        raise FileExistsError(f"Path already exists: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the V8 training campaign until confirmation eval reaches the target.")
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-device", type=str, default="cuda")
    parser.add_argument("--baseline-json", type=str, default=str(DEFAULT_BASELINE_JSON))
    parser.add_argument("--manifest-path", type=str, default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--target-root", type=str, default=str(DEFAULT_TARGET_ROOT))
    parser.add_argument("--resume-source", type=str, default=str(DEFAULT_RESUME_SOURCE))
    parser.add_argument("--target-multiple", type=float, default=4.0)
    parser.add_argument("--candidate-count", type=int, default=6)
    parser.add_argument("--candidate-eval-episodes", type=int, default=20)
    parser.add_argument("--confirm-eval-episodes", type=int, default=50)
    parser.add_argument("--max-seeds-per-family", type=int, default=5)
    parser.add_argument("--plateau-delta", type=float, default=0.10)
    parser.add_argument("--promotion-threshold", type=float, default=3.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    return parser.parse_args()


def maybe_override_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    if args.candidate_count == 6:
        args.candidate_count = 2
    if args.candidate_eval_episodes == 20:
        args.candidate_eval_episodes = 2
    if args.confirm_eval_episodes == 50:
        args.confirm_eval_episodes = 4


def build_family_templates(args: argparse.Namespace) -> dict[str, FamilyTemplate]:
    max_seeds = int(args.max_seeds_per_family)
    if args.smoke:
        return {
            "resume_a": FamilyTemplate("resume_a", 20, 20, 2, 1, 250, 20, 2, 300, 10, 10, 48, 48, 2000, 256, 2, 1e-3, 0.01, 0.9, 0.95, "256,256", "tanh", 1, 2, str(Path(args.resume_source).resolve()), float(args.promotion_threshold), True),
            "fresh_b": FamilyTemplate("fresh_b", 20, 20, 2, 1, 250, 20, 2, 300, 10, 10, 48, 48, 2000, 256, 2, 3e-4, 0.01, 0.95, 0.97, "512,512", "tanh", 1, 2),
        }
    return {
        "resume_a": FamilyTemplate("resume_a", 400, CONTINUATION_INCREMENT, 8, 2, 250, 20, 2, 300, 10, 10, 48, 48, 8000, 1024, 6, 1e-3, 0.01, 0.9, 0.95, "256,256", "tanh", 1, 4, str(Path(args.resume_source).resolve()), float(args.promotion_threshold), True),
        "fresh_b": FamilyTemplate("fresh_b", 800, CONTINUATION_INCREMENT, 8, 2, 250, 20, 2, 300, 10, 10, 48, 48, 12000, 2048, 6, 3e-4, 0.01, 0.95, 0.97, "512,512", "tanh", max_seeds, None),
        "fresh_c": FamilyTemplate("fresh_c", 1000, CONTINUATION_INCREMENT, 10, 2, 800, 20, 2, 300, 10, 10, 48, 48, 16000, 2048, 6, 1e-4, 0.005, 0.97, 0.98, "512,512", "tanh", max_seeds, None),
        "fresh_d": FamilyTemplate("fresh_d", 800, CONTINUATION_INCREMENT, 10, 2, 600, 20, 2, 300, 10, 10, 48, 48, 12000, 2048, 4, 3e-4, 0.02, 0.95, 0.97, "512,512", "tanh", max_seeds, None),
    }


def serialize_combo(combo: ComboState) -> dict[str, Any]:
    return {
        "family_id": combo.family_id,
        "seed": combo.seed,
        "phase_index": combo.phase_index,
        "restore_from_checkpoint": combo.restore_from_checkpoint,
        "best_candidate_score": combo.best_candidate_score,
        "best_candidate_checkpoint": combo.best_candidate_checkpoint,
        "best_candidate_iteration": combo.best_candidate_iteration,
        "best_phase_index": combo.best_phase_index,
        "insufficient_improvement_streak": combo.insufficient_improvement_streak,
        "last_phase_best_iteration": combo.last_phase_best_iteration,
        "last_phase_iterations": combo.last_phase_iterations,
        "retired": combo.retired,
        "retired_reason": combo.retired_reason,
        "total_added_iterations": combo.total_added_iterations,
    }


def deserialize_combo(payload: dict[str, Any]) -> ComboState:
    return ComboState(
        family_id=str(payload["family_id"]),
        seed=int(payload["seed"]),
        phase_index=int(payload["phase_index"]),
        restore_from_checkpoint=payload.get("restore_from_checkpoint"),
        best_candidate_score=None if payload.get("best_candidate_score") is None else float(payload["best_candidate_score"]),
        best_candidate_checkpoint=payload.get("best_candidate_checkpoint"),
        best_candidate_iteration=parse_int(payload.get("best_candidate_iteration")),
        best_phase_index=parse_int(payload.get("best_phase_index")),
        insufficient_improvement_streak=int(payload.get("insufficient_improvement_streak", 0)),
        last_phase_best_iteration=parse_int(payload.get("last_phase_best_iteration")),
        last_phase_iterations=parse_int(payload.get("last_phase_iterations")),
        retired=bool(payload.get("retired", False)),
        retired_reason=payload.get("retired_reason"),
        total_added_iterations=int(payload.get("total_added_iterations", 0)),
    )


def default_controller_state(templates: dict[str, FamilyTemplate]) -> ControllerState:
    current_combos = {family_id: ComboState(family_id, 0, 1, template.initial_restore_from) for family_id, template in templates.items()}
    return ControllerState(
        family_next_seed={family_id: 1 for family_id in templates},
        current_combos=current_combos,
        scheduled_round=[current_combos[family_id] for family_id in templates],
        tested_seed_count=len(current_combos),
        incumbent_best_candidate_score=None,
        incumbent_best_candidate_checkpoint=None,
    )


def save_manifest_state(path: Path, manifest: dict[str, Any], controller_state: ControllerState) -> None:
    existing = try_load_json(path)
    if existing is not None and "watchdog_state" in existing and "watchdog_state" not in manifest:
        manifest["watchdog_state"] = existing["watchdog_state"]
    manifest["controller_state"] = {
        "family_next_seed": controller_state.family_next_seed,
        "current_combos": {key: serialize_combo(value) for key, value in controller_state.current_combos.items()},
        "scheduled_round": [serialize_combo(combo) for combo in controller_state.scheduled_round],
        "tested_seed_count": controller_state.tested_seed_count,
        "incumbent_best_candidate_score": controller_state.incumbent_best_candidate_score,
        "incumbent_best_candidate_checkpoint": controller_state.incumbent_best_candidate_checkpoint,
    }
    manifest["tested_seed_count"] = controller_state.tested_seed_count
    manifest["incumbent_best_candidate_mean_pellets_per_fish"] = controller_state.incumbent_best_candidate_score
    manifest["incumbent_best_candidate_checkpoint"] = controller_state.incumbent_best_candidate_checkpoint
    save_json(path, manifest)

def load_controller_state(manifest: dict[str, Any], templates: dict[str, FamilyTemplate]) -> ControllerState:
    payload = manifest.get("controller_state")
    if payload:
        current_combos = {key: deserialize_combo(value) for key, value in payload.get("current_combos", {}).items()}
        for family_id, template in templates.items():
            current_combos.setdefault(family_id, ComboState(family_id, 0, 1, template.initial_restore_from))
        family_next_seed = {key: int(value) for key, value in payload.get("family_next_seed", {}).items()}
        for family_id in templates:
            family_next_seed.setdefault(family_id, 1)
        return ControllerState(
            family_next_seed=family_next_seed,
            current_combos=current_combos,
            scheduled_round=[deserialize_combo(item) for item in payload.get("scheduled_round", [])],
            tested_seed_count=int(payload.get("tested_seed_count", manifest.get("tested_seed_count", 0))),
            incumbent_best_candidate_score=None if payload.get("incumbent_best_candidate_score") is None else float(payload["incumbent_best_candidate_score"]),
            incumbent_best_candidate_checkpoint=payload.get("incumbent_best_candidate_checkpoint"),
        )
    state = default_controller_state(templates)
    state.incumbent_best_candidate_score = None if manifest.get("incumbent_best_candidate_mean_pellets_per_fish") is None else float(manifest.get("incumbent_best_candidate_mean_pellets_per_fish"))
    state.incumbent_best_candidate_checkpoint = manifest.get("incumbent_best_candidate_checkpoint")
    return state


def run_subprocess(command: list[str], *, cwd: Path, stdout_path: Path, stderr_path: Path) -> tuple[int, float]:
    started = time.perf_counter()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        result = subprocess.run(command, cwd=cwd, stdout=stdout_handle, stderr=stderr_handle, text=True, check=False)
    return int(result.returncode), float(time.perf_counter() - started)


def artifacts_complete(checkpoint_root: Path) -> bool:
    return file_exists(checkpoint_root / "run_summary.json") and file_exists(checkpoint_root / "checkpoint_final")


def load_light_rows(eval_reports_csv_path: Path) -> list[dict[str, Any]]:
    if not eval_reports_csv_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with eval_reports_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({
                "iteration": parse_int(row.get("iteration")),
                "checkpoint_path": row.get("checkpoint_path"),
                "mean_pellets_per_fish": parse_float(row.get("mean_pellets_per_fish")),
                "mean_total_reward": parse_float(row.get("mean_total_reward")),
            })
    return rows


def build_shortlist(*, checkpoint_root: Path, candidate_count: int, min_spacing: int) -> list[dict[str, Any]]:
    run_summary_path = checkpoint_root / "run_summary.json"
    if not run_summary_path.exists():
        return []
    run_summary = load_json(run_summary_path)
    ranked_rows = sorted(
        [row for row in load_light_rows(checkpoint_root / "eval_reports.csv") if row.get("checkpoint_path")],
        key=lambda row: (float(row.get("mean_pellets_per_fish", float("-inf"))), float(row.get("mean_total_reward", float("-inf")))),
        reverse=True,
    )
    shortlist: list[dict[str, Any]] = []
    for row in ranked_rows:
        iteration = row.get("iteration")
        if iteration is None:
            continue
        if any(existing.get("iteration") is not None and abs(int(iteration) - int(existing["iteration"])) < min_spacing for existing in shortlist):
            continue
        shortlist.append({
            "checkpoint_path": str(Path(str(row["checkpoint_path"])).resolve()),
            "iteration": int(iteration),
            "source": "light_eval",
            "light_mean_pellets_per_fish": float(row["mean_pellets_per_fish"]),
            "light_mean_total_reward": float(row["mean_total_reward"]),
        })
        if len(shortlist) >= candidate_count:
            break
    final_checkpoint_path = run_summary.get("final_checkpoint_path")
    if final_checkpoint_path:
        final_resolved = str(Path(str(final_checkpoint_path)).resolve())
        if final_resolved not in {entry["checkpoint_path"] for entry in shortlist}:
            best_checkpoint = run_summary.get("best_checkpoint")
            fallback_iteration = best_checkpoint.get("iteration") if isinstance(best_checkpoint, dict) else None
            shortlist.append({
                "checkpoint_path": final_resolved,
                "iteration": parse_int(fallback_iteration),
                "source": "final_checkpoint",
                "light_mean_pellets_per_fish": None,
                "light_mean_total_reward": None,
            })
    return shortlist


def write_checkpoint_list(path: Path, shortlisted: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in shortlisted:
            handle.write(f"{item['checkpoint_path']}\n")


def load_checkpoint_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_candidate_eval_csv(path: Path) -> list[dict[str, Any]] | None:
    try:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append({
                    "checkpoint_path": row.get("checkpoint_path"),
                    "mean_pellets_per_fish": parse_float(row.get("mean_pellets_per_fish")),
                    "mean_total_reward": parse_float(row.get("mean_total_reward")),
                })
        return rows
    except (OSError, csv.Error):
        return None


def load_json_records(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not content:
        return None
    try:
        if "\n" not in content and content.startswith("{"):
            payload = json.loads(content)
            return [payload] if isinstance(payload, dict) else None
        records = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    records.append(payload)
        return records if records else None
    except json.JSONDecodeError:
        return None


def highest_scoring_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: (float(row.get("mean_pellets_per_fish", float("-inf"))), float(row.get("mean_total_reward", float("-inf")))))


def build_training_command(python_executable: str, template: FamilyTemplate, combo: ComboState, checkpoint_root: Path, device: str, train_iterations: int) -> list[str]:
    command = [
        python_executable, "-u", "agent.py", "--device", device, "--seed", str(combo.seed),
        "--train-iterations", str(train_iterations),
        "--num-env-runners", str(template.num_env_runners),
        "--num-envs-per-runner", str(template.num_envs_per_runner),
        "--rollout-fragment-length", str(template.rollout_fragment_length),
        "--checkpoint-every-iterations", str(template.checkpoint_every_iterations),
        "--checkpoint-root", str(checkpoint_root.resolve()),
        "--num-red-fish", str(template.num_red_fish), "--num-blue-fish", str(template.num_blue_fish),
        "--num-red-pellets", str(template.num_red_pellets), "--num-blue-pellets", str(template.num_blue_pellets),
        "--time-limit", str(template.time_limit), "--light-eval-episodes", str(template.light_eval_episodes),
        "--train-batch-size", str(template.train_batch_size), "--minibatch-size", str(template.minibatch_size),
        "--num-epochs", str(template.num_epochs), "--learning-rate", str(template.learning_rate),
        "--entropy-coeff", str(template.entropy_coeff), "--gamma", str(template.gamma),
        "--gae-lambda", str(template.gae_lambda), "--fcnet-hiddens", template.fcnet_hiddens,
        "--fcnet-activation", template.fcnet_activation,
    ]
    if combo.restore_from_checkpoint:
        command.extend(["--restore-from-checkpoint", str(combo.restore_from_checkpoint)])
    return command


def run_candidate_eval(*, python_executable: str, shortlist_file: Path, summary_json_path: Path, summary_csv_path: Path, eval_device: str, episodes: int, seed: int) -> tuple[int, float]:
    command = [
        python_executable, "test_model.py", "--checkpoint-list-file", str(shortlist_file.resolve()),
        "--episodes", str(episodes), "--no-render", "--device", str(eval_device),
        "--mute-mode", "normal", "--seed", str(seed),
        "--summary-json", str(summary_json_path.resolve()), "--summary-csv", str(summary_csv_path.resolve()),
    ]
    return run_subprocess(command, cwd=SCRIPT_DIR, stdout_path=summary_json_path.with_suffix(".stdout.log"), stderr_path=summary_json_path.with_suffix(".stderr.log"))


def run_confirm_eval(*, python_executable: str, checkpoint_path: str, summary_json_path: Path, summary_csv_path: Path, eval_device: str, episodes: int, seed: int) -> tuple[int, float]:
    command = [
        python_executable, "test_model.py", "--checkpoint-path", str(Path(checkpoint_path).resolve()),
        "--episodes", str(episodes), "--no-render", "--device", str(eval_device),
        "--mute-mode", "both", "--seed", str(seed),
        "--summary-json", str(summary_json_path.resolve()), "--summary-csv", str(summary_csv_path.resolve()),
    ]
    return run_subprocess(command, cwd=SCRIPT_DIR, stdout_path=summary_json_path.with_suffix(".stdout.log"), stderr_path=summary_json_path.with_suffix(".stderr.log"))


def make_phase_root(base_root: Path, combo: ComboState) -> Path:
    return base_root / combo.family_id / f"seed{combo.seed}" / f"phase{combo.phase_index:02d}"

def initialize_manifest(*, manifest_path: Path, args: argparse.Namespace, baseline_path: Path, baseline_mean: float, target_mean: float, target_root: Path) -> dict[str, Any]:
    manifest = {
        "campaign_started_at": now_iso(),
        "campaign_mode": "smoke" if args.smoke else "full",
        "python_executable": str(Path(args.python_executable).resolve()),
        "device": str(args.device),
        "eval_device": str(args.eval_device),
        "baseline_json": str(baseline_path.resolve()),
        "target_root": str(target_root.resolve()),
        "resume_source": str(Path(args.resume_source).resolve()),
        "random_mean_pellets_per_fish": float(baseline_mean),
        "target_multiple": float(args.target_multiple),
        "target_mean_pellets_per_fish": float(target_mean),
        "candidate_count": int(args.candidate_count),
        "candidate_eval_episodes": int(args.candidate_eval_episodes),
        "confirm_eval_episodes": int(args.confirm_eval_episodes),
        "plateau_delta": float(args.plateau_delta),
        "promotion_threshold": float(args.promotion_threshold),
        "max_seeds_per_family": int(args.max_seeds_per_family),
        "phases": [],
        "status": "running",
        "stop_reason": None,
        "tested_seed_count": 0,
        "incumbent_best_candidate_mean_pellets_per_fish": None,
        "incumbent_best_candidate_checkpoint": None,
        "incumbent_best_confirmed_mean_pellets_per_fish": None,
        "success_phase_id": None,
        "success_checkpoint_path": None,
        "success_confirmed_mean_pellets_per_fish": None,
    }
    save_json(manifest_path, manifest)
    return manifest


def find_phase_record(manifest: dict[str, Any], phase_id: str) -> dict[str, Any] | None:
    for phase in manifest.get("phases", []):
        if phase.get("phase_id") == phase_id:
            return phase
    return None


def upsert_phase_record(manifest: dict[str, Any], phase_record: dict[str, Any]) -> None:
    existing = find_phase_record(manifest, str(phase_record["phase_id"]))
    if existing is None:
        manifest.setdefault("phases", []).append(phase_record)
    else:
        if existing is phase_record:
            return
        existing.clear()
        existing.update(phase_record)


def set_phase_status(phase_record: dict[str, Any], status: str, *, error: str | None = None) -> None:
    phase_record["phase_status"] = status
    phase_record["phase_error"] = error
    phase_record["updated_at"] = now_iso()


def phase_output_paths(checkpoint_root: Path, winner_checkpoint: str | None = None) -> dict[str, Path | None]:
    paths: dict[str, Path | None] = {
        "train_stdout": checkpoint_root / "train_stdout.log",
        "train_stderr": checkpoint_root / "train_stderr.log",
        "run_summary": checkpoint_root / "run_summary.json",
        "candidate_list": checkpoint_root / "candidate_checkpoints.txt",
        "candidate_json": checkpoint_root / "candidate_eval.jsonl",
        "candidate_csv": checkpoint_root / "candidate_eval.csv",
        "confirm_json": None,
        "confirm_csv": None,
    }
    if winner_checkpoint:
        winner_name = Path(winner_checkpoint).name
        paths["confirm_json"] = checkpoint_root / f"confirm_eval_{winner_name}.json"
        paths["confirm_csv"] = checkpoint_root / f"confirm_eval_{winner_name}.csv"
    return paths


def build_phase_record(combo: ComboState, template: FamilyTemplate, checkpoint_root: Path, phase_iterations: int) -> dict[str, Any]:
    paths = phase_output_paths(checkpoint_root)
    return {
        "phase_id": combo.phase_id,
        "family": combo.family_id,
        "seed": int(combo.seed),
        "phase_index": int(combo.phase_index),
        "checkpoint_root": str(checkpoint_root.resolve()),
        "restore_from_checkpoint": combo.restore_from_checkpoint,
        "additional_train_iterations": int(phase_iterations),
        "config": asdict(template),
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "phase_status": PHASE_STATUS_SCHEDULED,
        "phase_error": None,
        "prior_best_candidate_score": combo.best_candidate_score,
        "combo_state_applied": False,
        "promoted": False,
        "retired": False,
        "retired_reason": None,
        "training_stdout_log": str(paths["train_stdout"].resolve()),
        "training_stderr_log": str(paths["train_stderr"].resolve()),
        "run_summary_path": str(paths["run_summary"].resolve()),
        "artifacts_complete": False,
        "train_return_code": None,
        "train_elapsed_seconds": None,
        "shortlisted_checkpoints": [],
        "candidate_list_file": str(paths["candidate_list"].resolve()),
        "candidate_eval_triggered": False,
        "candidate_eval_return_code": None,
        "candidate_eval_elapsed_seconds": None,
        "candidate_eval_json_path": str(paths["candidate_json"].resolve()),
        "candidate_eval_csv_path": str(paths["candidate_csv"].resolve()),
        "candidate_eval_winner_checkpoint": None,
        "candidate_eval_winner_iteration": None,
        "candidate_eval_winner_mean_pellets_per_fish": None,
        "candidate_eval_winner_mean_total_reward": None,
        "confirm_eval_triggered": False,
        "confirm_eval_return_code": None,
        "confirm_eval_elapsed_seconds": None,
        "confirm_eval_json_path": None,
        "confirm_eval_csv_path": None,
        "confirm_eval_result": None,
        "confirm_eval_success": False,
        "insufficient_improvement_streak": int(combo.insufficient_improvement_streak),
        "incumbent_best_candidate_mean_pellets_per_fish": combo.best_candidate_score,
        "incumbent_best_candidate_checkpoint": combo.best_candidate_checkpoint,
    }


def normalize_existing_phase_record(phase_record: dict[str, Any]) -> None:
    if phase_record.get("phase_status"):
        return
    checkpoint_root = Path(str(phase_record["checkpoint_root"]))
    if phase_record.get("confirm_eval_triggered") and not phase_record.get("confirm_eval_result"):
        set_phase_status(phase_record, PHASE_STATUS_CONFIRM_EVAL)
    elif phase_record.get("candidate_eval_triggered"):
        set_phase_status(phase_record, PHASE_STATUS_COMPLETE)
    elif artifacts_complete(checkpoint_root):
        set_phase_status(phase_record, PHASE_STATUS_CANDIDATE_EVAL)
    elif checkpoint_root.exists():
        set_phase_status(phase_record, PHASE_STATUS_TRAINING)
    else:
        set_phase_status(phase_record, PHASE_STATUS_SCHEDULED)


def synthesize_initial_round_records(manifest: dict[str, Any], controller_state: ControllerState, templates: dict[str, FamilyTemplate], target_root: Path) -> None:
    for combo in controller_state.scheduled_round:
        template = templates[combo.family_id]
        phase_iterations = template.initial_train_iterations if combo.phase_index == 1 else template.continuation_train_iterations
        checkpoint_root = make_phase_root(target_root, combo)
        if find_phase_record(manifest, combo.phase_id):
            continue
        phase_record = build_phase_record(combo, template, checkpoint_root, phase_iterations)
        if artifacts_complete(checkpoint_root):
            phase_record["artifacts_complete"] = True
            set_phase_status(phase_record, PHASE_STATUS_CANDIDATE_EVAL)
        elif checkpoint_root.exists():
            set_phase_status(phase_record, PHASE_STATUS_TRAINING)
        else:
            set_phase_status(phase_record, PHASE_STATUS_SCHEDULED)
        upsert_phase_record(manifest, phase_record)


def ensure_phase_records_for_round(manifest: dict[str, Any], controller_state: ControllerState, templates: dict[str, FamilyTemplate], target_root: Path) -> dict[str, dict[str, Any]]:
    if not manifest.get("phases"):
        synthesize_initial_round_records(manifest, controller_state, templates, target_root)
    phase_entries: dict[str, dict[str, Any]] = {}
    for combo in controller_state.scheduled_round:
        template = templates[combo.family_id]
        checkpoint_root = make_phase_root(target_root, combo)
        phase_iterations = template.initial_train_iterations if combo.phase_index == 1 else template.continuation_train_iterations
        phase_record = find_phase_record(manifest, combo.phase_id)
        if phase_record is None:
            phase_record = build_phase_record(combo, template, checkpoint_root, phase_iterations)
            if artifacts_complete(checkpoint_root):
                phase_record["artifacts_complete"] = True
                set_phase_status(phase_record, PHASE_STATUS_CANDIDATE_EVAL)
            elif checkpoint_root.exists():
                set_phase_status(phase_record, PHASE_STATUS_TRAINING)
            else:
                set_phase_status(phase_record, PHASE_STATUS_SCHEDULED)
            upsert_phase_record(manifest, phase_record)
        else:
            normalize_existing_phase_record(phase_record)
        phase_entries[combo.combo_id] = phase_record
    return phase_entries


def candidate_eval_is_complete(phase_record: dict[str, Any]) -> bool:
    shortlist = load_checkpoint_list(Path(str(phase_record["candidate_list_file"])))
    if not shortlist:
        return False
    rows = load_candidate_eval_csv(Path(str(phase_record["candidate_eval_csv_path"])))
    json_rows = load_json_records(Path(str(phase_record["candidate_eval_json_path"])))
    return rows is not None and json_rows is not None and len(rows) == len(shortlist) and len(json_rows) == len(shortlist)


def confirm_eval_is_complete(phase_record: dict[str, Any]) -> bool:
    json_path = phase_record.get("confirm_eval_json_path")
    csv_path = phase_record.get("confirm_eval_csv_path")
    if not json_path or not csv_path:
        return False
    payload = try_load_json(Path(str(json_path)))
    rows = load_candidate_eval_csv(Path(str(csv_path)))
    return payload is not None and isinstance(payload.get("eval_result"), dict) and rows is not None and len(rows) >= 1


def refresh_training_artifact_fields(phase_record: dict[str, Any]) -> None:
    checkpoint_root = Path(str(phase_record["checkpoint_root"]))
    phase_record["artifacts_complete"] = bool(artifacts_complete(checkpoint_root))
    phase_record["run_summary_path"] = str((checkpoint_root / "run_summary.json").resolve())
    phase_record["updated_at"] = now_iso()


def reset_phase_root_for_training(checkpoint_root: Path) -> None:
    if checkpoint_root.exists():
        shutil.rmtree(checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)


def mark_recoverable_failure(phase_record: dict[str, Any], *, message: str) -> None:
    set_phase_status(phase_record, PHASE_STATUS_FAILED_RECOVERABLE, error=message)


def mark_terminal_failure(phase_record: dict[str, Any], *, message: str) -> None:
    set_phase_status(phase_record, PHASE_STATUS_FAILED_TERMINAL, error=message)

def apply_candidate_results(
    combo: ComboState,
    phase_record: dict[str, Any],
    *,
    winner_checkpoint: str,
    winner_iteration: int | None,
    winner_score: float,
    winner_reward: float,
    plateau_delta: float,
    incumbent_best_candidate_score: float | None,
    incumbent_best_candidate_checkpoint: str | None,
) -> tuple[float | None, str | None]:
    if not phase_record.get("combo_state_applied", False):
        combo.last_phase_best_iteration = int(winner_iteration) if winner_iteration is not None else None
        combo.last_phase_iterations = int(phase_record["additional_train_iterations"])
        combo.total_added_iterations += int(phase_record["additional_train_iterations"])
        prior_best_score = phase_record.get("prior_best_candidate_score")
        if combo.best_candidate_score is None or winner_score > combo.best_candidate_score:
            combo.best_candidate_score = winner_score
            combo.best_candidate_checkpoint = winner_checkpoint
            combo.best_candidate_iteration = int(winner_iteration) if winner_iteration is not None else None
            combo.best_phase_index = combo.phase_index
        if prior_best_score is None:
            combo.insufficient_improvement_streak = 0
        else:
            improvement = float(combo.best_candidate_score or float("-inf")) - float(prior_best_score)
            combo.insufficient_improvement_streak = 0 if improvement >= plateau_delta else combo.insufficient_improvement_streak + 1
        phase_record["combo_state_applied"] = True
    phase_record["candidate_eval_winner_checkpoint"] = winner_checkpoint
    phase_record["candidate_eval_winner_iteration"] = int(winner_iteration) if winner_iteration is not None else None
    phase_record["candidate_eval_winner_mean_pellets_per_fish"] = float(winner_score)
    phase_record["candidate_eval_winner_mean_total_reward"] = float(winner_reward)
    phase_record["insufficient_improvement_streak"] = int(combo.insufficient_improvement_streak)
    if incumbent_best_candidate_score is None or winner_score > incumbent_best_candidate_score:
        incumbent_best_candidate_score = float(winner_score)
        incumbent_best_candidate_checkpoint = winner_checkpoint
    phase_record["incumbent_best_candidate_mean_pellets_per_fish"] = incumbent_best_candidate_score
    phase_record["incumbent_best_candidate_checkpoint"] = incumbent_best_candidate_checkpoint
    return incumbent_best_candidate_score, incumbent_best_candidate_checkpoint


def process_phase(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    manifest_path: Path,
    controller_state: ControllerState,
    combo: ComboState,
    template: FamilyTemplate,
    phase_record: dict[str, Any],
    target_mean: float,
) -> bool:
    checkpoint_root = Path(str(phase_record["checkpoint_root"]))
    phase_iterations = int(phase_record["additional_train_iterations"])
    if str(phase_record.get("phase_status")) == PHASE_STATUS_COMPLETE:
        return False

    if str(phase_record.get("phase_status")) in {PHASE_STATUS_SCHEDULED, PHASE_STATUS_TRAINING, PHASE_STATUS_FAILED_RECOVERABLE}:
        refresh_training_artifact_fields(phase_record)
        if not phase_record["artifacts_complete"]:
            set_phase_status(phase_record, PHASE_STATUS_TRAINING)
            save_manifest_state(manifest_path, manifest, controller_state)
            reset_phase_root_for_training(checkpoint_root)
            return_code, elapsed_seconds = run_subprocess(
                build_training_command(args.python_executable, template, combo, checkpoint_root, args.device, phase_iterations),
                cwd=SCRIPT_DIR,
                stdout_path=Path(str(phase_record["training_stdout_log"])),
                stderr_path=Path(str(phase_record["training_stderr_log"])),
            )
            phase_record["train_return_code"] = int(return_code)
            phase_record["train_elapsed_seconds"] = float(elapsed_seconds)
            refresh_training_artifact_fields(phase_record)
            if not phase_record["artifacts_complete"]:
                mark_recoverable_failure(phase_record, message="training_artifacts_incomplete")
                save_manifest_state(manifest_path, manifest, controller_state)
                raise RecoverablePhaseError(f"Training artifacts incomplete for {phase_record['phase_id']}")
        set_phase_status(phase_record, PHASE_STATUS_CANDIDATE_EVAL)
        save_manifest_state(manifest_path, manifest, controller_state)

    if str(phase_record.get("phase_status")) == PHASE_STATUS_CANDIDATE_EVAL:
        shortlist = build_shortlist(checkpoint_root=checkpoint_root, candidate_count=args.candidate_count, min_spacing=DEFAULT_CANDIDATE_SPACING)
        if not shortlist:
            mark_terminal_failure(phase_record, message="candidate_shortlist_empty")
            save_manifest_state(manifest_path, manifest, controller_state)
            raise RuntimeError(f"Candidate shortlist empty for {phase_record['phase_id']}")
        phase_record["shortlisted_checkpoints"] = shortlist
        shortlist_file = Path(str(phase_record["candidate_list_file"]))
        write_checkpoint_list(shortlist_file, shortlist)
        save_manifest_state(manifest_path, manifest, controller_state)
        if not candidate_eval_is_complete(phase_record):
            return_code, elapsed_seconds = run_candidate_eval(
                python_executable=args.python_executable,
                shortlist_file=shortlist_file,
                summary_json_path=Path(str(phase_record["candidate_eval_json_path"])),
                summary_csv_path=Path(str(phase_record["candidate_eval_csv_path"])),
                eval_device=args.eval_device,
                episodes=args.candidate_eval_episodes,
                seed=combo.seed,
            )
            phase_record["candidate_eval_return_code"] = int(return_code)
            phase_record["candidate_eval_elapsed_seconds"] = float(elapsed_seconds)
            save_manifest_state(manifest_path, manifest, controller_state)
            if not candidate_eval_is_complete(phase_record):
                mark_recoverable_failure(phase_record, message="candidate_eval_incomplete")
                save_manifest_state(manifest_path, manifest, controller_state)
                raise RecoverablePhaseError(f"Candidate eval incomplete for {phase_record['phase_id']}")
        candidate_rows = load_candidate_eval_csv(Path(str(phase_record["candidate_eval_csv_path"])))
        winner = highest_scoring_row(candidate_rows or [])
        if winner is None:
            mark_terminal_failure(phase_record, message="candidate_eval_produced_no_rows")
            save_manifest_state(manifest_path, manifest, controller_state)
            raise RuntimeError(f"Candidate evaluation produced no rows for {phase_record['phase_id']}")
        candidate_to_iteration = {entry["checkpoint_path"]: entry.get("iteration") for entry in shortlist}
        winner_checkpoint = str(winner["checkpoint_path"])
        winner_score = float(winner["mean_pellets_per_fish"])
        winner_reward = float(winner["mean_total_reward"])
        winner_iteration = candidate_to_iteration.get(winner_checkpoint)
        if winner_iteration is None and Path(winner_checkpoint).name == "checkpoint_final":
            winner_iteration = phase_iterations
        controller_state.incumbent_best_candidate_score, controller_state.incumbent_best_candidate_checkpoint = apply_candidate_results(
            combo,
            phase_record,
            winner_checkpoint=winner_checkpoint,
            winner_iteration=winner_iteration,
            winner_score=winner_score,
            winner_reward=winner_reward,
            plateau_delta=args.plateau_delta,
            incumbent_best_candidate_score=controller_state.incumbent_best_candidate_score,
            incumbent_best_candidate_checkpoint=controller_state.incumbent_best_candidate_checkpoint,
        )
        phase_record["candidate_eval_triggered"] = True
        controller_state.current_combos[combo.family_id] = combo
        if winner_score >= target_mean:
            phase_record["confirm_eval_triggered"] = True
            set_phase_status(phase_record, PHASE_STATUS_CONFIRM_EVAL)
        else:
            set_phase_status(phase_record, PHASE_STATUS_COMPLETE)
        save_manifest_state(manifest_path, manifest, controller_state)

    if str(phase_record.get("phase_status")) == PHASE_STATUS_CONFIRM_EVAL:
        winner_checkpoint = str(phase_record["candidate_eval_winner_checkpoint"])
        confirm_paths = phase_output_paths(checkpoint_root, winner_checkpoint)
        phase_record["confirm_eval_json_path"] = str(confirm_paths["confirm_json"].resolve()) if confirm_paths["confirm_json"] else None
        phase_record["confirm_eval_csv_path"] = str(confirm_paths["confirm_csv"].resolve()) if confirm_paths["confirm_csv"] else None
        save_manifest_state(manifest_path, manifest, controller_state)
        if not confirm_eval_is_complete(phase_record):
            return_code, elapsed_seconds = run_confirm_eval(
                python_executable=args.python_executable,
                checkpoint_path=winner_checkpoint,
                summary_json_path=Path(str(phase_record["confirm_eval_json_path"])),
                summary_csv_path=Path(str(phase_record["confirm_eval_csv_path"])),
                eval_device=args.eval_device,
                episodes=args.confirm_eval_episodes,
                seed=combo.seed + 10000,
            )
            phase_record["confirm_eval_return_code"] = int(return_code)
            phase_record["confirm_eval_elapsed_seconds"] = float(elapsed_seconds)
            save_manifest_state(manifest_path, manifest, controller_state)
            if not confirm_eval_is_complete(phase_record):
                mark_recoverable_failure(phase_record, message="confirm_eval_incomplete")
                save_manifest_state(manifest_path, manifest, controller_state)
                raise RecoverablePhaseError(f"Confirm eval incomplete for {phase_record['phase_id']}")
        payload = try_load_json(Path(str(phase_record["confirm_eval_json_path"])))
        if payload is None or not isinstance(payload.get("eval_result"), dict):
            mark_recoverable_failure(phase_record, message="confirm_eval_json_corrupt")
            save_manifest_state(manifest_path, manifest, controller_state)
            raise RecoverablePhaseError(f"Confirm eval corrupt for {phase_record['phase_id']}")
        phase_record["confirm_eval_result"] = payload
        phase_record["confirm_eval_success"] = bool(float(payload["eval_result"]["mean_pellets_per_fish"]) >= target_mean)
        set_phase_status(phase_record, PHASE_STATUS_COMPLETE)
        save_manifest_state(manifest_path, manifest, controller_state)
        if phase_record["confirm_eval_success"]:
            manifest["status"] = "success"
            manifest["stop_reason"] = "target_reached"
            manifest["success_phase_id"] = combo.phase_id
            manifest["success_checkpoint_path"] = winner_checkpoint
            manifest["success_confirmed_mean_pellets_per_fish"] = float(payload["eval_result"]["mean_pellets_per_fish"])
            manifest["incumbent_best_confirmed_mean_pellets_per_fish"] = float(payload["eval_result"]["mean_pellets_per_fish"])
            save_manifest_state(manifest_path, manifest, controller_state)
            return True

    return False


def should_continue_combo(combo: ComboState, *, template: FamilyTemplate, promotion_threshold: float) -> bool:
    if combo.retired or combo.best_candidate_checkpoint is None or combo.best_candidate_score is None:
        return False
    if template.max_total_phases is not None and combo.phase_index >= template.max_total_phases:
        return False
    if template.continuation_requires_score is not None and combo.best_candidate_score < template.continuation_requires_score:
        return False
    if combo.insufficient_improvement_streak >= 2:
        return False
    if template.continuation_requires_late_best:
        if combo.last_phase_iterations is None or combo.last_phase_best_iteration is None:
            return False
        if combo.last_phase_best_iteration < int(combo.last_phase_iterations * 0.75):
            return False
    if combo.family_id == "resume_a" and combo.best_candidate_score < promotion_threshold:
        return False
    return True


def finalize_round(*, manifest: dict[str, Any], manifest_path: Path, controller_state: ControllerState, templates: dict[str, FamilyTemplate], phase_entries: dict[str, dict[str, Any]], args: argparse.Namespace) -> list[ComboState]:
    save_manifest_state(manifest_path, manifest, controller_state)
    incumbent_score = controller_state.incumbent_best_candidate_score
    if controller_state.tested_seed_count >= 5 and (incumbent_score is None or incumbent_score < BLOCKED_SCORE_THRESHOLD):
        manifest["status"] = "blocked"
        manifest["stop_reason"] = "training_only_4x_likely_blocked_before_3p2"
        save_manifest_state(manifest_path, manifest, controller_state)
        return []
    live_combos = [combo for combo in controller_state.current_combos.values() if combo and not combo.retired and combo.best_candidate_score is not None]
    ranked_live = sorted(live_combos, key=lambda combo: float(combo.best_candidate_score or float("-inf")), reverse=True)
    promoted = ranked_live[:PROMOTED_COMBO_COUNT]
    next_round: list[ComboState] = []
    for combo in promoted:
        template = templates[combo.family_id]
        phase_record = phase_entries.get(combo.combo_id)
        if should_continue_combo(combo, template=template, promotion_threshold=args.promotion_threshold):
            next_combo = ComboState(combo.family_id, combo.seed, combo.phase_index + 1, combo.best_candidate_checkpoint, combo.best_candidate_score, combo.best_candidate_checkpoint, combo.best_candidate_iteration, combo.best_phase_index, combo.insufficient_improvement_streak, combo.last_phase_best_iteration, combo.last_phase_iterations, False, None, combo.total_added_iterations)
            next_round.append(next_combo)
            controller_state.current_combos[combo.family_id] = next_combo
            if phase_record is not None:
                phase_record["promoted"] = True
                phase_record["retired"] = False
                phase_record["retired_reason"] = None
        else:
            next_seed = controller_state.family_next_seed.get(combo.family_id, 1)
            if next_seed < template.max_seeds:
                next_combo = ComboState(combo.family_id, next_seed, 1, template.initial_restore_from)
                controller_state.family_next_seed[combo.family_id] = next_seed + 1
                controller_state.current_combos[combo.family_id] = next_combo
                controller_state.tested_seed_count += 1
                next_round.append(next_combo)
                if phase_record is not None:
                    phase_record["promoted"] = True
                    phase_record["retired"] = True
                    phase_record["retired_reason"] = "launched_next_seed"
            else:
                combo.retired = True
                combo.retired_reason = "family_exhausted"
                controller_state.current_combos[combo.family_id] = combo
                if phase_record is not None:
                    phase_record["retired"] = True
                    phase_record["retired_reason"] = combo.retired_reason
    for combo in ranked_live[PROMOTED_COMBO_COUNT:]:
        combo.retired = True
        combo.retired_reason = "not_promoted"
        controller_state.current_combos[combo.family_id] = combo
        phase_record = phase_entries.get(combo.combo_id)
        if phase_record is not None:
            phase_record["retired"] = True
            phase_record["retired_reason"] = combo.retired_reason
    controller_state.scheduled_round = next_round
    save_manifest_state(manifest_path, manifest, controller_state)
    if not next_round:
        manifest["status"] = "failed"
        manifest["stop_reason"] = "all_live_combos_retired_without_reaching_target"
        save_manifest_state(manifest_path, manifest, controller_state)
    return next_round

def prepare_manifest_and_state(args: argparse.Namespace, templates: dict[str, FamilyTemplate], baseline_path: Path, baseline_mean: float, target_mean: float, manifest_path: Path, target_root: Path) -> tuple[dict[str, Any], ControllerState]:
    if args.resume_existing:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found for --resume-existing: {manifest_path}")
        manifest = load_json(manifest_path)
        controller_state = load_controller_state(manifest, templates)
        if not manifest.get("phases"):
            synthesize_initial_round_records(manifest, controller_state, templates, target_root)
            save_manifest_state(manifest_path, manifest, controller_state)
        return manifest, controller_state

    ensure_clean_path(manifest_path, force_clean=args.force_clean)
    ensure_clean_path(target_root, force_clean=args.force_clean)
    manifest = initialize_manifest(
        manifest_path=manifest_path,
        args=args,
        baseline_path=baseline_path,
        baseline_mean=baseline_mean,
        target_mean=target_mean,
        target_root=target_root,
    )
    controller_state = default_controller_state(templates)
    save_manifest_state(manifest_path, manifest, controller_state)
    return manifest, controller_state


def main() -> None:
    args = parse_args()
    maybe_override_smoke_defaults(args)
    if args.force_clean and args.resume_existing:
        raise ValueError("--force-clean and --resume-existing are mutually exclusive.")

    baseline_path = Path(args.baseline_json).resolve()
    if not file_exists(baseline_path):
        raise FileNotFoundError(f"Random baseline not found: {baseline_path}")
    resume_source = Path(args.resume_source).resolve()
    if not resume_source.exists():
        raise FileNotFoundError(f"Resume source checkpoint not found: {resume_source}")

    manifest_path = Path(args.manifest_path).resolve()
    if args.smoke and str(manifest_path) == str(DEFAULT_MANIFEST_PATH.resolve()):
        manifest_path = DEFAULT_SMOKE_MANIFEST_PATH.resolve()
    target_root = Path(args.target_root).resolve()

    baseline = load_json(baseline_path)
    baseline_mean = float(baseline["eval_result"]["mean_pellets_per_fish"])
    target_mean = float(args.target_multiple) * baseline_mean
    templates = build_family_templates(args)

    manifest, controller_state = prepare_manifest_and_state(args, templates, baseline_path, baseline_mean, target_mean, manifest_path, target_root)
    if manifest.get("status") in TERMINAL_CAMPAIGN_STATUSES:
        return

    while controller_state.scheduled_round:
        phase_entries = ensure_phase_records_for_round(manifest, controller_state, templates, target_root)
        save_manifest_state(manifest_path, manifest, controller_state)
        for combo in list(controller_state.scheduled_round):
            phase_record = phase_entries[combo.combo_id]
            upsert_phase_record(manifest, phase_record)
            if str(phase_record.get("phase_status")) == PHASE_STATUS_COMPLETE:
                continue
            try:
                success = process_phase(
                    args=args,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    controller_state=controller_state,
                    combo=combo,
                    template=templates[combo.family_id],
                    phase_record=phase_record,
                    target_mean=target_mean,
                )
            except RecoverablePhaseError as exc:
                upsert_phase_record(manifest, phase_record)
                save_manifest_state(manifest_path, manifest, controller_state)
                print(f"recoverable_phase_exit phase={phase_record['phase_id']} error={exc}", flush=True)
                return
            upsert_phase_record(manifest, phase_record)
            save_manifest_state(manifest_path, manifest, controller_state)
            if success:
                return
        next_round = finalize_round(
            manifest=manifest,
            manifest_path=manifest_path,
            controller_state=controller_state,
            templates=templates,
            phase_entries=phase_entries,
            args=args,
        )
        if manifest.get("status") in TERMINAL_CAMPAIGN_STATUSES:
            return
        controller_state.scheduled_round = next_round

    manifest["status"] = "failed"
    manifest["stop_reason"] = "campaign_exhausted_without_reaching_target"
    save_manifest_state(manifest_path, manifest, controller_state)


if __name__ == "__main__":
    main()
