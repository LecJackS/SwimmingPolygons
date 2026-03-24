"""Watchdog for unattended V8 4x training campaigns."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = SCRIPT_DIR / "target_training_manifest_4x_rerun.json"
DEFAULT_TARGET_ROOT = SCRIPT_DIR / "rllib_checkpoints_target_v8_4x_rerun"
CONTROLLER_STDOUT = SCRIPT_DIR / "target_training_controller.out.log"
CONTROLLER_STDERR = SCRIPT_DIR / "target_training_controller.err.log"
BACKOFF_SCHEDULE = [60, 120, 300, 600]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch and relaunch the V8 target training controller.")
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--manifest-path", type=str, default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--target-root", type=str, default=str(DEFAULT_TARGET_ROOT))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-device", type=str, default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--poll-interval-seconds", type=int, default=120)
    parser.add_argument("--launch-grace-seconds", type=int, default=15)
    return parser.parse_args()


def try_load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def ps_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def manifest_fingerprint(manifest: dict[str, Any] | None) -> str:
    if not manifest:
        return "manifest-missing"
    phases = manifest.get("phases") or []
    last_phase = phases[-1] if phases else {}
    return "|".join([
        str(manifest.get("status")),
        str(len(phases)),
        str(last_phase.get("phase_id")),
        str(last_phase.get("phase_status")),
    ])


def update_watchdog_state(manifest_path: Path, **updates: Any) -> None:
    manifest = try_load_json(manifest_path)
    if manifest is None:
        return
    watchdog_state = dict(manifest.get("watchdog_state") or {})
    watchdog_state.update(updates)
    watchdog_state["updated_at"] = now_iso()
    manifest["watchdog_state"] = watchdog_state
    save_json(manifest_path, manifest)


def find_controller_processes(manifest_path: Path) -> list[dict[str, Any]]:
    command = (
        "$manifest = " + ps_quote(str(manifest_path.resolve())) + "; "
        "$rx = [regex]::Escape($manifest); "
        "$rows = Get-CimInstance Win32_Process | Where-Object { "
        " $_.Name -match 'python' -and $_.CommandLine -match 'train_until_target.py' -and $_.CommandLine -match $rx "
        "} | Select-Object ProcessId, Name, CommandLine; "
        "$rows | ConvertTo-Json -Compress"
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
        str((SCRIPT_DIR / "train_until_target.py").resolve()),
        "--manifest-path", str(Path(args.manifest_path).resolve()),
        "--target-root", str(Path(args.target_root).resolve()),
        "--device", str(args.device),
        "--eval-device", str(args.eval_device),
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
    log(f"watchdog_started manifest={manifest_path} target_root={target_root} poll_interval={args.poll_interval_seconds}")

    while True:
        manifest = try_load_json(manifest_path)
        if manifest is not None and manifest.get("status") in {"success", "failed", "blocked"}:
            log(f"watchdog_stopping terminal_status={manifest.get('status')} stop_reason={manifest.get('stop_reason')}")
            update_watchdog_state(manifest_path, last_terminal_status=manifest.get("status"), last_terminal_reason=manifest.get("stop_reason"), retry_count=retry_count)
            return

        processes = find_controller_processes(manifest_path)
        if processes:
            retry_count = 0
            last_failure_fingerprint = None
            active_pid = int(processes[0]["ProcessId"]) if processes and processes[0].get("ProcessId") is not None else None
            update_watchdog_state(
                manifest_path,
                active_controller_pid=active_pid,
                active_controller_process_count=len(processes),
                retry_count=retry_count,
                last_seen_alive_at=now_iso(),
            )
            time.sleep(args.poll_interval_seconds)
            continue

        if not manifest_path.exists() and target_root.exists():
            log("watchdog_waiting manifest_missing_target_root_exists")
            time.sleep(BACKOFF_SCHEDULE[-1])
            continue

        fingerprint = manifest_fingerprint(manifest)
        resume_existing = manifest_path.exists()
        process = launch_controller(args, resume_existing=resume_existing)
        log(f"controller_launched pid={process.pid} resume_existing={resume_existing}")
        update_watchdog_state(manifest_path, active_controller_pid=process.pid, retry_count=retry_count, last_launch_at=now_iso())
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
        log(f"controller_exited_quickly pid={process.pid} exit_code={exit_code} retry_count={retry_count} delay_seconds={delay}")
        update_watchdog_state(
            manifest_path,
            active_controller_pid=None,
            retry_count=retry_count,
            last_error=f"controller_exit_{exit_code}",
            last_failure_fingerprint=fingerprint,
            last_retry_delay_seconds=delay,
        )
        time.sleep(delay)


if __name__ == "__main__":
    main()
