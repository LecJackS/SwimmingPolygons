"""Delayed finish-check watcher for V8 color communication schooling runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a V8 color communication training run and auto-evaluate when it finishes.")
    parser.add_argument("--pid-file", type=str, default="./baseline_v8_color_comm.pid")
    parser.add_argument("--checkpoint-root", type=str, default="./rllib_checkpoints_baseline_v8_color_comm")
    parser.add_argument("--log-path", type=str, default="./baseline_v8_color_comm.out.log")
    parser.add_argument("--check-offset-minutes", type=str, default="90,120", help="Comma-separated offsets in minutes from watcher start.")
    parser.add_argument("--primary-eval-episodes", type=int, default=10)
    parser.add_argument("--secondary-eval-episodes", type=int, default=10)
    parser.add_argument("--visual-max-frames", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--eval-device", type=str, default="cpu")
    parser.add_argument("--auto-eval-dir-name", type=str, default="auto_eval")
    parser.add_argument("--no-visual-launch", action="store_true")
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value.")
    parsed = [float(value) for value in values]
    if any(value < 0.0 for value in parsed):
        raise ValueError("Check offsets must be >= 0.")
    return parsed


def slugify_minutes(minutes: float) -> str:
    if abs(minutes - round(minutes)) < 1e-9:
        return f"{int(round(minutes))}m"
    text = f"{minutes:.3f}".rstrip("0").rstrip(".")
    return f"{text.replace('.', 'p')}m"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_pid(pid_file: Path) -> int | None:
    if not pid_file.exists():
        return None
    text = pid_file.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def is_process_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_log_state(log_text: str) -> dict[str, Any]:
    latest_iteration = None
    training_status = None
    final_checkpoint_logged = None

    iter_matches = list(re.finditer(r"iter=(\d{3})", log_text))
    if iter_matches:
        latest_iteration = int(iter_matches[-1].group(1))

    training_status_matches = list(re.finditer(r"training_status:\s+([^\r\n]+)", log_text))
    if training_status_matches:
        training_status = training_status_matches[-1].group(1).strip()

    final_checkpoint_matches = list(re.finditer(r"final_checkpoint_saved:\s+([^\r\n]+)", log_text))
    if final_checkpoint_matches:
        final_checkpoint_logged = final_checkpoint_matches[-1].group(1).strip()

    return {
        "latest_iteration": latest_iteration,
        "training_status": training_status,
        "final_checkpoint_logged": final_checkpoint_logged,
    }


def load_run_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def normalize_checkpoint_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    raw = str(path_text).strip()
    direct = Path(raw)
    if direct.exists():
        return direct
    match = re.search(r"path=([^,\)]+)", raw)
    if match:
        candidate = Path(match.group(1).strip())
        if candidate.exists():
            return candidate
    return None


def checkpoint_exists(path_text: str | None) -> bool:
    normalized = normalize_checkpoint_path(path_text)
    return normalized is not None and normalized.exists()


def select_primary_checkpoint(run_summary: dict[str, Any], checkpoint_root: Path) -> Path | None:
    best_checkpoint = run_summary.get("best_checkpoint")
    if isinstance(best_checkpoint, dict):
        checkpoint_path = normalize_checkpoint_path(best_checkpoint.get("checkpoint_path"))
        if checkpoint_path is not None:
            return checkpoint_path
    final_checkpoint = run_summary.get("final_checkpoint_path")
    final_path = normalize_checkpoint_path(final_checkpoint)
    if final_path is not None:
        return final_path
    fallback = checkpoint_root / "checkpoint_final"
    if fallback.exists():
        return fallback
    return None


def select_secondary_checkpoint(run_summary: dict[str, Any], primary_checkpoint: Path | None) -> Path | None:
    final_path = normalize_checkpoint_path(run_summary.get("final_checkpoint_path"))
    if final_path is None:
        return None
    if primary_checkpoint is not None and final_path.resolve() == primary_checkpoint.resolve():
        return None
    return final_path


def is_training_finished(*, process_alive: bool, log_state: dict[str, Any], final_checkpoint_exists: bool) -> bool:
    if log_state.get("training_status"):
        return True
    if log_state.get("final_checkpoint_logged"):
        return True
    if final_checkpoint_exists and not process_alive:
        return True
    return False


def write_watcher_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_iso()}] {message}\n")


def write_snapshot(snapshot_path: Path, payload: dict[str, Any]) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_batch_evaluation(
    *,
    python_executable: str,
    eval_device: str,
    checkpoint_path: Path,
    script_dir: Path,
    episodes: int,
    seed: int,
    summary_json_path: Path,
    summary_csv_path: Path,
    watcher_log_path: Path,
    ) -> float:
    command = [
        python_executable,
        "test_model.py",
        "--checkpoint-path",
        str(checkpoint_path.resolve()),
        "--mute-mode",
        "both",
        "--episodes",
        str(episodes),
        "--no-render",
        "--device",
        str(eval_device),
        "--seed",
        str(seed),
        "--summary-json",
        str(summary_json_path.resolve()),
        "--summary-csv",
        str(summary_csv_path.resolve()),
    ]
    write_watcher_log(watcher_log_path, f"running batch evaluation: {' '.join(command)}")
    started = time.perf_counter()
    result = subprocess.run(command, cwd=script_dir, text=True, capture_output=True, check=False)
    wall_ms = (time.perf_counter() - started) * 1000.0
    if result.stdout:
        write_watcher_log(watcher_log_path, result.stdout.strip())
    if result.stderr:
        write_watcher_log(watcher_log_path, f"stderr: {result.stderr.strip()}")
    if result.returncode != 0:
        raise RuntimeError(f"Batch evaluation failed for {checkpoint_path} with exit code {result.returncode}.")
    write_watcher_log(watcher_log_path, f"batch evaluation wall_ms={wall_ms:.1f} checkpoint={checkpoint_path.resolve()}")
    return wall_ms


def build_visual_command(*, python_executable: str, checkpoint_path: Path, max_frames: int) -> list[str]:
    return [
        python_executable,
        "test_model.py",
        "--checkpoint-path",
        str(checkpoint_path.resolve()),
        "--render-profile",
        "fast",
        "--render-engine",
        "auto",
        "--max-frames",
        str(max_frames),
    ]


def launch_visual_evaluation(*, command: list[str], script_dir: Path, watcher_log_path: Path) -> None:
    write_watcher_log(watcher_log_path, f"launching visual evaluation: {' '.join(command)}")
    creationflags = 0
    if hasattr(subprocess, "CREATE_NEW_CONSOLE"):
        creationflags = subprocess.CREATE_NEW_CONSOLE
    subprocess.Popen(command, cwd=script_dir, creationflags=creationflags)


def evaluate_checkpoint_pair(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    episodes: int,
    seed: int,
    auto_eval_dir: Path,
    watcher_log_path: Path,
) -> dict[str, str]:
    summary_base = auto_eval_dir / f"{checkpoint_path.name}_summary"
    wall_ms = run_batch_evaluation(
        python_executable=args.python_executable,
        eval_device=args.eval_device,
        checkpoint_path=checkpoint_path,
        script_dir=SCRIPT_DIR,
        episodes=episodes,
        seed=seed,
        summary_json_path=summary_base.with_suffix(".json"),
        summary_csv_path=summary_base.with_suffix(".csv"),
        watcher_log_path=watcher_log_path,
    )
    return {
        "summary_json": str(summary_base.with_suffix(".json").resolve()),
        "summary_csv": str(summary_base.with_suffix(".csv").resolve()),
        "eval_wall_ms": float(wall_ms),
    }


def run_auto_evaluation(
    *,
    args: argparse.Namespace,
    checkpoint_root: Path,
    auto_eval_dir: Path,
    watcher_log_path: Path,
    run_summary: dict[str, Any],
) -> dict[str, Any]:
    primary_checkpoint = select_primary_checkpoint(run_summary, checkpoint_root)
    if primary_checkpoint is None:
        raise FileNotFoundError("Unable to resolve a primary checkpoint for auto evaluation.")
    secondary_checkpoint = select_secondary_checkpoint(run_summary, primary_checkpoint)

    auto_eval_dir.mkdir(parents=True, exist_ok=True)
    primary_outputs = evaluate_checkpoint_pair(
        args=args,
        checkpoint_path=primary_checkpoint,
        episodes=args.primary_eval_episodes,
        seed=args.seed,
        auto_eval_dir=auto_eval_dir,
        watcher_log_path=watcher_log_path,
    )
    secondary_outputs = None
    if secondary_checkpoint is not None:
        secondary_outputs = evaluate_checkpoint_pair(
            args=args,
            checkpoint_path=secondary_checkpoint,
            episodes=args.secondary_eval_episodes,
            seed=args.seed + 10_000,
            auto_eval_dir=auto_eval_dir,
            watcher_log_path=watcher_log_path,
        )

    visual_command = build_visual_command(
        python_executable=args.python_executable,
        checkpoint_path=primary_checkpoint,
        max_frames=args.visual_max_frames,
    )
    visual_command_path = auto_eval_dir / "visual_eval_command.txt"
    visual_command_path.write_text(" ".join(visual_command), encoding="utf-8")
    write_watcher_log(watcher_log_path, f"saved visual eval command: {visual_command_path.resolve()}")

    launched_visual = False
    if not args.no_visual_launch:
        launch_visual_evaluation(command=visual_command, script_dir=SCRIPT_DIR, watcher_log_path=watcher_log_path)
        launched_visual = True
    else:
        write_watcher_log(watcher_log_path, "visual launch disabled by --no-visual-launch")

    manifest = {
        "timestamp": now_iso(),
        "eval_device": str(args.eval_device),
        "primary_checkpoint": str(primary_checkpoint.resolve()),
        "secondary_checkpoint": str(secondary_checkpoint.resolve()) if secondary_checkpoint is not None else None,
        "primary_outputs": primary_outputs,
        "secondary_outputs": secondary_outputs,
        "visual_command_path": str(visual_command_path.resolve()),
        "visual_launched": bool(launched_visual),
    }
    manifest_path = auto_eval_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_watcher_log(watcher_log_path, f"auto evaluation complete: {manifest_path.resolve()}")
    return manifest


def collect_status(*, pid_file: Path, log_path: Path, checkpoint_root: Path) -> dict[str, Any]:
    pid = read_pid(pid_file)
    process_alive = is_process_alive(pid)
    log_text = read_text_if_exists(log_path)
    log_state = parse_log_state(log_text)
    final_checkpoint_path = checkpoint_root / "checkpoint_final"
    final_checkpoint_exists = final_checkpoint_path.exists() or checkpoint_exists(log_state.get("final_checkpoint_logged"))
    run_summary = load_run_summary(checkpoint_root / "run_summary.json")

    return {
        "pid_file_exists": pid_file.exists(),
        "pid": pid,
        "process_alive": process_alive,
        "log_path_exists": log_path.exists(),
        "latest_iteration": log_state["latest_iteration"],
        "training_status": log_state["training_status"],
        "final_checkpoint_logged": log_state["final_checkpoint_logged"],
        "final_checkpoint_exists": final_checkpoint_exists,
        "run_summary": run_summary,
    }


def main() -> None:
    args = parse_args()
    check_offsets = sorted(parse_float_list(args.check_offset_minutes))
    if args.primary_eval_episodes <= 0 or args.secondary_eval_episodes <= 0:
        raise ValueError("Evaluation episodes must be > 0.")
    if args.visual_max_frames <= 0:
        raise ValueError("--visual-max-frames must be > 0.")

    pid_file = (SCRIPT_DIR / args.pid_file).resolve()
    checkpoint_root = (SCRIPT_DIR / args.checkpoint_root).resolve()
    log_path = (SCRIPT_DIR / args.log_path).resolve()
    auto_eval_dir = checkpoint_root / args.auto_eval_dir_name
    watcher_log_path = auto_eval_dir / "watcher.log"
    start_time = datetime.now()

    write_watcher_log(
        watcher_log_path,
        f"watcher_started pid_file={pid_file} checkpoint_root={checkpoint_root} log_path={log_path} eval_device={args.eval_device}",
    )

    initial_status = collect_status(pid_file=pid_file, log_path=log_path, checkpoint_root=checkpoint_root)
    initial_finished = is_training_finished(
        process_alive=bool(initial_status["process_alive"]),
        log_state=initial_status,
        final_checkpoint_exists=bool(initial_status["final_checkpoint_exists"]),
    )
    if initial_finished:
        snapshot_name = f"check_{slugify_minutes(check_offsets[0])}.json"
        snapshot_payload = {
            "timestamp": now_iso(),
            "scheduled_offset_minutes": float(check_offsets[0]),
            "elapsed_minutes": 0.0,
            "observed_before_first_wait": True,
            "evaluation_triggered": True,
            **{key: value for key, value in initial_status.items() if key != "run_summary"},
        }
        manifest = run_auto_evaluation(
            args=args,
            checkpoint_root=checkpoint_root,
            auto_eval_dir=auto_eval_dir,
            watcher_log_path=watcher_log_path,
            run_summary=initial_status["run_summary"],
        )
        snapshot_payload["evaluation_manifest"] = manifest
        write_snapshot(auto_eval_dir / snapshot_name, snapshot_payload)
        return

    evaluation_done = False
    for offset_minutes in check_offsets:
        scheduled_time = start_time + timedelta(minutes=float(offset_minutes))
        sleep_seconds = max(0.0, (scheduled_time - datetime.now()).total_seconds())
        if sleep_seconds > 0.0:
            write_watcher_log(
                watcher_log_path,
                f"sleeping_until_check offset_minutes={offset_minutes} sleep_seconds={sleep_seconds:.1f}",
            )
            time.sleep(sleep_seconds)

        status = collect_status(pid_file=pid_file, log_path=log_path, checkpoint_root=checkpoint_root)
        finished = is_training_finished(
            process_alive=bool(status["process_alive"]),
            log_state=status,
            final_checkpoint_exists=bool(status["final_checkpoint_exists"]),
        )
        snapshot_payload = {
            "timestamp": now_iso(),
            "scheduled_offset_minutes": float(offset_minutes),
            "elapsed_minutes": round((datetime.now() - start_time).total_seconds() / 60.0, 3),
            "observed_before_first_wait": False,
            "evaluation_triggered": False,
            **{key: value for key, value in status.items() if key != "run_summary"},
        }
        if finished and not evaluation_done:
            manifest = run_auto_evaluation(
                args=args,
                checkpoint_root=checkpoint_root,
                auto_eval_dir=auto_eval_dir,
                watcher_log_path=watcher_log_path,
                run_summary=status["run_summary"],
            )
            snapshot_payload["evaluation_triggered"] = True
            snapshot_payload["evaluation_manifest"] = manifest
            evaluation_done = True

        snapshot_path = auto_eval_dir / f"check_{slugify_minutes(offset_minutes)}.json"
        write_snapshot(snapshot_path, snapshot_payload)
        write_watcher_log(
            watcher_log_path,
            "check_complete "
            f"offset_minutes={offset_minutes} "
            f"process_alive={status['process_alive']} "
            f"latest_iteration={status['latest_iteration']} "
            f"final_checkpoint_exists={status['final_checkpoint_exists']} "
            f"evaluation_triggered={snapshot_payload['evaluation_triggered']}",
        )
        if evaluation_done:
            return

    write_watcher_log(watcher_log_path, "watcher_finished_without_auto_evaluation")


if __name__ == "__main__":
    main()
