#!/usr/bin/env python3
"""Wait for a long GPT-2 FT run handoff, then launch the next command."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


WAIT_STATUSES = ("waiting", "waiting_for_process")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--pid-file", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint directory that must contain --checkpoint-ready-file.",
    )
    parser.add_argument("--checkpoint-ready-file", default="model.safetensors")
    parser.add_argument(
        "--status-card",
        type=Path,
        default=None,
        help="Optional JSON status card to wait on before launch.",
    )
    parser.add_argument(
        "--status-card-wait-status",
        action="append",
        default=[],
        help=(
            "Status value that means the card is still waiting. Defaults to "
            "waiting,waiting_for_process."
        ),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--jsonl-out",
        type=Path,
        default=None,
        help="Append every wait/launch state to this JSONL history file.",
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--checkpoint-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--status-card-timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=None,
        help=(
            "Minimum free disk GiB to preserve before launching. When omitted, "
            "the wrapped command's --min-free-disk-gb is used if present."
        ),
    )
    parser.add_argument(
        "--launched-pid-file",
        type=Path,
        default=None,
        help="Write the launched command PID here before waiting for it.",
    )
    parser.add_argument(
        "--launched-log-file",
        type=Path,
        default=None,
        help="Redirect launched command stdout/stderr to this log file.",
    )
    parser.add_argument(
        "--launched-log-mode",
        choices=("append", "write"),
        default="append",
        help="Append to or replace --launched-log-file. Defaults to append.",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help=(
            "Return after spawning the launched command instead of waiting for it. "
            "Useful for multi-hour chained FT runs that have their own monitors."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0.0:
        parser.error("--poll-seconds must be positive")
    if args.checkpoint_timeout_seconds < 0.0:
        parser.error("--checkpoint-timeout-seconds must be non-negative")
    if args.status_card_timeout_seconds < 0.0:
        parser.error("--status-card-timeout-seconds must be non-negative")
    if args.min_free_disk_gb is not None and args.min_free_disk_gb < 0.0:
        parser.error("--min-free-disk-gb must be non-negative")
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command to launch is required after --")
    return args


def _read_pid(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _resolved_pid(args: argparse.Namespace) -> int | None:
    return args.pid if args.pid is not None else _read_pid(args.pid_file)


def _process_alive(pid: int | None) -> bool | None:
    if pid is None:
        return None
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "unreadable"}
    return dict(payload) if isinstance(payload, Mapping) else {"status": "invalid"}


def _status_card_status(args: argparse.Namespace) -> object:
    payload = _read_json(args.status_card)
    if payload is None:
        return None
    return payload.get("status")


def _status_card_ready(args: argparse.Namespace) -> bool:
    if args.status_card is None:
        return True
    status = _status_card_status(args)
    wait_statuses = set(args.status_card_wait_status or WAIT_STATUSES)
    return status is not None and status not in wait_statuses


def _checkpoint_ready(args: argparse.Namespace) -> bool | None:
    if args.checkpoint is None:
        return None
    return (args.checkpoint / str(args.checkpoint_ready_file)).is_file()


def _command_flag_value(command: Sequence[object], flag: str) -> str | None:
    values = [str(item) for item in command]
    for index, item in enumerate(values):
        if item == flag and index + 1 < len(values):
            return values[index + 1]
        if item.startswith(f"{flag}="):
            return item.split("=", 1)[1]
    return None


def _positive_int_flag(command: Sequence[object], flag: str) -> int | None:
    value = _command_flag_value(command, flag)
    if value is None:
        return None
    try:
        parsed = int(float(value))
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _nonnegative_float_flag(command: Sequence[object], flag: str) -> float | None:
    value = _command_flag_value(command, flag)
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if parsed >= 0.0 else None


def _path_size_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        try:
            return int(path.stat().st_size)
        except OSError:
            return None
    total = 0
    try:
        for child in path.rglob("*"):
            if not child.is_file():
                continue
            try:
                total += int(child.stat().st_size)
            except OSError:
                continue
    except OSError:
        return None
    return total


def _nearest_existing_parent(path: Path) -> Path | None:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current if current.exists() else None


def _disk_free_bytes(path: Path) -> int | None:
    parent = _nearest_existing_parent(path)
    if parent is None:
        return None
    try:
        return int(shutil.disk_usage(parent).free)
    except OSError:
        return None


def _launch_disk_guard(args: argparse.Namespace) -> dict[str, Any]:
    gib = 1024.0**3
    command = [str(item) for item in args.command]
    output_dir_value = _command_flag_value(command, "--output-dir")
    resume_value = _command_flag_value(command, "--resume-from-checkpoint")
    output_dir = Path(output_dir_value) if output_dir_value else None
    resume_checkpoint = (
        Path(resume_value)
        if resume_value is not None
        else args.checkpoint
        if args.checkpoint is not None
        else None
    )
    min_free_gb = (
        args.min_free_disk_gb
        if args.min_free_disk_gb is not None
        else _nonnegative_float_flag(command, "--min-free-disk-gb")
    )
    save_total_limit = _positive_int_flag(command, "--save-total-limit") or 1
    checkpoint_bytes = (
        None if resume_checkpoint is None else _path_size_bytes(resume_checkpoint)
    )
    estimated_peak_checkpoint_count = max(save_total_limit + 1, 1)
    estimated_peak_checkpoint_bytes = (
        None
        if checkpoint_bytes is None
        else int(checkpoint_bytes) * estimated_peak_checkpoint_count
    )
    disk_anchor = output_dir or resume_checkpoint
    free_bytes = None if disk_anchor is None else _disk_free_bytes(disk_anchor)
    free_after_estimated_peak_bytes = (
        None
        if free_bytes is None or estimated_peak_checkpoint_bytes is None
        else int(free_bytes) - int(estimated_peak_checkpoint_bytes)
    )
    checked_free_gb = (
        None
        if free_after_estimated_peak_bytes is None
        else float(free_after_estimated_peak_bytes) / gib
    )
    if min_free_gb is None:
        status = "unchecked"
        meets_min_free = None
    elif checked_free_gb is not None:
        meets_min_free = checked_free_gb >= float(min_free_gb)
        status = "ok" if meets_min_free else "blocked"
    elif free_bytes is not None:
        free_gb = float(free_bytes) / gib
        meets_min_free = free_gb >= float(min_free_gb)
        status = "ok" if meets_min_free else "blocked"
    else:
        status = "unknown"
        meets_min_free = None
    return {
        "row_type": "hf_gpt2_ft_wait_launch_disk_guard",
        "status": status,
        "output_dir": None if output_dir is None else str(output_dir),
        "resume_from_checkpoint": (
            None if resume_checkpoint is None else str(resume_checkpoint)
        ),
        "min_free_gb": min_free_gb,
        "meets_min_free": meets_min_free,
        "save_total_limit": save_total_limit,
        "resume_checkpoint_bytes": checkpoint_bytes,
        "resume_checkpoint_gb": (
            None if checkpoint_bytes is None else float(checkpoint_bytes) / gib
        ),
        "estimated_peak_checkpoint_count": estimated_peak_checkpoint_count,
        "estimated_peak_checkpoint_bytes": estimated_peak_checkpoint_bytes,
        "estimated_peak_checkpoint_gb": (
            None
            if estimated_peak_checkpoint_bytes is None
            else float(estimated_peak_checkpoint_bytes) / gib
        ),
        "free_bytes": free_bytes,
        "free_gb": None if free_bytes is None else float(free_bytes) / gib,
        "free_after_estimated_peak_bytes": free_after_estimated_peak_bytes,
        "free_after_estimated_peak_gb": checked_free_gb,
    }


def _write_manifest(
    args: argparse.Namespace,
    status: str,
    *,
    returncode: int | None = None,
    launch_error: str | None = None,
    launched_pid: int | None = None,
) -> dict[str, Any]:
    pid = _resolved_pid(args)
    launched_pid = (
        launched_pid if launched_pid is not None else _read_pid(args.launched_pid_file)
    )
    payload: dict[str, Any] = {
        "row_type": "hf_gpt2_finetune_wait_launch",
        "status": status,
        "time_unix_s": time.time(),
        "pid": pid,
        "pid_file": None if args.pid_file is None else str(args.pid_file),
        "process_alive": _process_alive(pid),
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "checkpoint_ready_file": args.checkpoint_ready_file,
        "checkpoint_ready": _checkpoint_ready(args),
        "status_card": None if args.status_card is None else str(args.status_card),
        "status_card_status": _status_card_status(args),
        "launched_pid_file": (
            None if args.launched_pid_file is None else str(args.launched_pid_file)
        ),
        "launched_pid": launched_pid,
        "launched_log_file": (
            None if args.launched_log_file is None else str(args.launched_log_file)
        ),
        "launched_log_mode": args.launched_log_mode,
        "command": [str(item) for item in args.command],
        "launch_disk_guard": _launch_disk_guard(args),
        "detach": bool(args.detach),
        "dry_run": bool(args.dry_run),
        "returncode": returncode,
        "launch_error": launch_error,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.jsonl_out is not None:
        args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl_out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        "hf_gpt2_ft_wait_launch "
        f"status={status} "
        f"process_alive={payload['process_alive']} "
        f"checkpoint_ready={payload['checkpoint_ready']} "
        f"status_card_status={payload['status_card_status']} "
        f"launch_disk_status={payload['launch_disk_guard'].get('status')} "
        f"launch_disk_free_after_gb={payload['launch_disk_guard'].get('free_after_estimated_peak_gb')} "
        f"launched_pid={payload['launched_pid']} "
        f"returncode={returncode}",
        flush=True,
    )
    return payload


def _wait_until(
    args: argparse.Namespace,
    *,
    status: str,
    ready,
    timeout_seconds: float,
) -> bool:
    start = time.time()
    while not ready():
        _write_manifest(args, status)
        if timeout_seconds == 0.0 or time.time() - start >= timeout_seconds:
            return False
        elapsed = time.time() - start
        remaining = max(0.0, timeout_seconds - elapsed)
        time.sleep(min(float(args.poll_seconds), remaining))
    return True


def _run_command(args: argparse.Namespace) -> tuple[int, str | None, int | None]:
    command = [str(item) for item in args.command]
    log_handle = None
    try:
        stdout = None
        stderr = None
        if args.launched_log_file is not None:
            args.launched_log_file.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if args.launched_log_mode == "append" else "w"
            log_handle = args.launched_log_file.open(mode, encoding="utf-8")
            stdout = log_handle
            stderr = subprocess.STDOUT
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        if args.launched_pid_file is not None:
            args.launched_pid_file.parent.mkdir(parents=True, exist_ok=True)
            args.launched_pid_file.write_text(
                f"{int(process.pid)}\n",
                encoding="utf-8",
            )
        launched_pid = int(process.pid)
        if args.detach:
            return 0, None, launched_pid
        return int(process.wait()), None, launched_pid
    except OSError as exc:
        return 2, f"{exc.__class__.__name__}: {exc}", None
    finally:
        if log_handle is not None:
            log_handle.close()


def run_wait_launch(args: argparse.Namespace) -> dict[str, Any]:
    pid = _resolved_pid(args)
    if pid is not None:
        while _process_alive(pid):
            _write_manifest(args, "waiting_for_process")
            time.sleep(float(args.poll_seconds))
            pid = _resolved_pid(args)

    if args.checkpoint is not None and not _wait_until(
        args,
        status="waiting_for_checkpoint",
        ready=lambda: _checkpoint_ready(args) is True,
        timeout_seconds=float(args.checkpoint_timeout_seconds),
    ):
        return _write_manifest(args, "blocked_missing_checkpoint", returncode=2)

    if args.status_card is not None and not _wait_until(
        args,
        status="waiting_for_status_card",
        ready=lambda: _status_card_ready(args),
        timeout_seconds=float(args.status_card_timeout_seconds),
    ):
        return _write_manifest(args, "blocked_waiting_status_card", returncode=2)

    if args.dry_run:
        return _write_manifest(args, "dry_run", returncode=0)

    disk_guard = _launch_disk_guard(args)
    if disk_guard.get("status") == "blocked":
        return _write_manifest(args, "blocked_prelaunch_disk", returncode=2)

    _write_manifest(args, "launching")
    returncode, launch_error, launched_pid = _run_command(args)
    if args.detach and launch_error is None:
        return _write_manifest(
            args,
            "launched",
            returncode=returncode,
            launched_pid=launched_pid,
        )
    return _write_manifest(
        args,
        "finished" if launch_error is None else "launch_error",
        returncode=returncode,
        launch_error=launch_error,
        launched_pid=launched_pid,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_wait_launch(args)
    return int(payload.get("returncode") or 0)


if __name__ == "__main__":
    raise SystemExit(main())
