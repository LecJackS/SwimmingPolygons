"""Watch and relaunch the V9 forage timeout curriculum controller."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DONOR_CHECKPOINT = (
    SCRIPT_DIR
    / "rllib_checkpoints_v9_transfer_campaign"
    / "r2_initial_limit_medium"
    / "checkpoint_00040"
)
DEFAULT_MANIFEST_PATH = SCRIPT_DIR / "forage_timeout_curriculum_manifest.json"
DEFAULT_TARGET_ROOT = SCRIPT_DIR / "rllib_checkpoints_v9_forage_timeout_curriculum"
CONTROLLER_STDOUT = SCRIPT_DIR / "forage_timeout_curriculum_controller.out.log"
CONTROLLER_STDERR = SCRIPT_DIR / "forage_timeout_curriculum_controller.err.log"
BACKOFF_SCHEDULE = [60, 120, 300, 600]
TERMINAL_STATUSES = {
    "success",
    "failed",
    "deadline_reached",
    "blocked_stage1_no_food",
    "blocked_stage2_no_food",
    "blocked_stage3_no_food",
    "blocked_stage4_no_food",
}


def default_python_executable() -> str:
    venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python.resolve())
    return sys.executable


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch and relaunch the V9 forage timeout curriculum controller.")
    parser.add_argument("--python-executable", type=str, default=default_python_executable())
    parser.add_argument("--donor-checkpoint", type=str, default=str(DEFAULT_DONOR_CHECKPOINT))
    parser.add_argument("--manifest-path", type=str, default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--target-root", type=str, default=str(DEFAULT_TARGET_ROOT))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-device", type=str, default="cuda")
    parser.add_argument("--max-wall-clock-hours", type=float, default=12.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--poll-interval-seconds", type=int, default=120)
    parser.add_argument("--launch-grace-seconds", type=int, default=15)
    parser.add_argument("--heartbeat-stale-seconds", type=int, default=300)
    return parser.parse_args()


def try_load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def ps_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def manifest_fingerprint(manifest: dict[str, Any] | None) -> str:
    if not manifest:
        return "manifest-missing"
    phases = manifest.get("phases") or []
    last_phase = phases[-1] if phases else {}
    return "|".join(
        [
            str(manifest.get("status")),
            str(len(phases)),
            str(last_phase.get("phase_id")),
            str(last_phase.get("phase_status")),
            str(manifest.get("controller_heartbeat_at")),
        ]
    )


def is_process_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def heartbeat_age_seconds(manifest: dict[str, Any] | None) -> float | None:
    if not manifest:
        return None
    heartbeat_text = manifest.get("controller_heartbeat_at")
    if not heartbeat_text:
        return None
    try:
        heartbeat = datetime.fromisoformat(str(heartbeat_text))
    except ValueError:
        return None
    return max((datetime.now(timezone.utc).astimezone() - heartbeat).total_seconds(), 0.0)


def find_controller_processes(manifest_path: Path) -> list[dict[str, Any]]:
    command = (
        "$manifest = "
        + ps_quote(str(manifest_path.resolve()))
        + "; $rx = [regex]::Escape($manifest); "
        + "$rows = Get-CimInstance Win32_Process | Where-Object { "
        + " $_.Name -match 'python' -and $_.CommandLine -match 'train_until_forage_timeout_curriculum.py' -and $_.CommandLine -match $rx "
        + "} | Select-Object ProcessId, Name, CommandLine; "
        + "$rows | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-Command", command],
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        log(f"process_scan_failed returncode={result.returncode}")
        return []
    raw = (result.stdout or "").strip()
    if not raw:
        return []
    payload = json.loads(raw)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    return []


def launch_controller(args: argparse.Namespace, *, resume_existing: bool) -> subprocess.Popen[str]:
    command = [
        str(Path(args.python_executable).resolve()),
        str((SCRIPT_DIR / "train_until_forage_timeout_curriculum.py").resolve()),
        "--donor-checkpoint",
        str(Path(args.donor_checkpoint).resolve()),
        "--manifest-path",
        str(Path(args.manifest_path).resolve()),
        "--target-root",
        str(Path(args.target_root).resolve()),
        "--device",
        str(args.device),
        "--eval-device",
        str(args.eval_device),
        "--max-wall-clock-hours",
        str(args.max_wall_clock_hours),
    ]
    if args.smoke:
        command.append("--smoke")
    if resume_existing:
        command.append("--resume-existing")
    stdout_handle = CONTROLLER_STDOUT.open("a", encoding="utf-8")
    stderr_handle = CONTROLLER_STDERR.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(command, cwd=SCRIPT_DIR, stdout=stdout_handle, stderr=stderr_handle, text=True)
    finally:
        stdout_handle.close()
        stderr_handle.close()
    return process


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path).resolve()
    target_root = Path(args.target_root).resolve()
    retry_count = 0
    last_failure_fingerprint: str | None = None
    log(
        f"watchdog_started manifest={manifest_path} target_root={target_root} poll_interval={args.poll_interval_seconds}"
    )

    while True:
        manifest = try_load_json(manifest_path)
        if manifest is not None and str(manifest.get("status")) in TERMINAL_STATUSES:
            log(
                f"watchdog_stopping terminal_status={manifest.get('status')} stop_reason={manifest.get('stop_reason')}"
            )
            return

        processes = find_controller_processes(manifest_path)
        manifest_controller_pid = None if manifest is None else manifest.get("controller_pid")
        heartbeat_age = heartbeat_age_seconds(manifest)
        heartbeat_stale = heartbeat_age is not None and heartbeat_age > float(args.heartbeat_stale_seconds)
        process_scan_alive = bool(processes)
        manifest_pid_alive = is_process_alive(int(manifest_controller_pid)) if manifest_controller_pid else False
        controller_alive = process_scan_alive or manifest_pid_alive

        if controller_alive:
            if heartbeat_stale:
                log(
                    f"heartbeat_stale controller_pid={manifest_controller_pid} heartbeat_age_seconds={heartbeat_age:.1f}"
                )
            retry_count = 0
            last_failure_fingerprint = None
            time.sleep(args.poll_interval_seconds)
            continue

        if not manifest_path.exists() and target_root.exists():
            log("watchdog_waiting manifest_missing_target_root_exists")
            time.sleep(BACKOFF_SCHEDULE[-1])
            continue

        if heartbeat_stale:
            log(
                f"controller_missing_after_stale_heartbeat controller_pid={manifest_controller_pid} heartbeat_age_seconds={heartbeat_age:.1f}"
            )

        fingerprint = manifest_fingerprint(manifest)
        resume_existing = manifest_path.exists()
        process = launch_controller(args, resume_existing=resume_existing)
        log(f"controller_launched pid={process.pid} resume_existing={resume_existing}")
        time.sleep(args.launch_grace_seconds)

        exit_code = process.poll()
        if exit_code is None:
            retry_count = 0
            last_failure_fingerprint = None
            time.sleep(args.poll_interval_seconds)
            continue

        if fingerprint == last_failure_fingerprint:
            retry_count += 1
        else:
            retry_count = 1
            last_failure_fingerprint = fingerprint
        delay = BACKOFF_SCHEDULE[min(retry_count - 1, len(BACKOFF_SCHEDULE) - 1)]
        log(
            f"controller_exited_quickly pid={process.pid} exit_code={exit_code} retry_count={retry_count} delay_seconds={delay}"
        )
        time.sleep(delay)


if __name__ == "__main__":
    main()
