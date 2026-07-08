#!/usr/bin/env python3
"""Summarize a live SpiralTorch GPT-2 fine-tuning run directory."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import spiraltorch as st

DEFAULT_TRACE_NAME = "spiraltorch-hf-gpt2-ft-trainer-trace.jsonl"
DEFAULT_RUN_CARD_NAME = "spiraltorch-hf-gpt2-ft-run-card.json"
DEFAULT_LOG_NAME = "ft.log"
PROGRESS_RE = re.compile(
    r"\b(?P<step>\d+)/(?:\s*)?(?P<max_steps>\d+)\s*\["
    r"(?P<elapsed>[^<,\]]+)(?:<(?P<remaining>[^,\]]+))?"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--trace-jsonl", type=Path, default=None)
    parser.add_argument("--run-card", type=Path, default=None)
    parser.add_argument("--pid-file", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--log-tail-bytes", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-card", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--min-free-disk-gb", type=float, default=None)
    parser.add_argument("--final-checkpoint", default=None)
    parser.add_argument("--tail-evals", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--lines-out", type=Path, default=None)
    parser.add_argument("--jsonl-out", type=Path, default=None)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Write requested status artifacts without printing per-cycle path notices.",
    )
    parser.add_argument("--watch-interval-seconds", type=float, default=None)
    parser.add_argument("--watch-count", type=int, default=None)
    parser.add_argument("--watch-stop-on-final", action="store_true")
    parser.add_argument("--watch-stop-on-process-exit", action="store_true")
    parser.add_argument("--watch-stop-on-disk-low", action="store_true")
    parser.add_argument("--watch-stop-on-training-guard", action="store_true")
    parser.add_argument("--watch-stop-on-eval-step", type=int, default=None)
    parser.add_argument("--watch-stop-on-checkpoint", default=None)
    args = parser.parse_args(argv)
    if not args.run_dir.exists():
        parser.error(f"run_dir does not exist: {args.run_dir}")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.eval_steps is not None and args.eval_steps <= 0:
        parser.error("--eval-steps must be positive")
    if args.save_steps is not None and args.save_steps <= 0:
        parser.error("--save-steps must be positive")
    if args.save_total_limit is not None and args.save_total_limit <= 0:
        parser.error("--save-total-limit must be positive")
    if args.min_free_disk_gb is not None and args.min_free_disk_gb < 0.0:
        parser.error("--min-free-disk-gb must be non-negative")
    if args.tail_evals < 0:
        parser.error("--tail-evals must be non-negative")
    if args.log_tail_bytes <= 0:
        parser.error("--log-tail-bytes must be positive")
    if args.watch_interval_seconds is not None and args.watch_interval_seconds <= 0.0:
        parser.error("--watch-interval-seconds must be positive")
    if args.watch_count is not None and args.watch_count <= 0:
        parser.error("--watch-count must be positive")
    if args.watch_stop_on_eval_step is not None and args.watch_stop_on_eval_step < 0:
        parser.error("--watch-stop-on-eval-step must be non-negative")
    if args.watch_stop_on_checkpoint is not None:
        args.watch_stop_on_checkpoint = args.watch_stop_on_checkpoint.strip()
        if not args.watch_stop_on_checkpoint:
            parser.error("--watch-stop-on-checkpoint must be non-empty")
    if args.trace_jsonl is None:
        args.trace_jsonl = args.run_dir / DEFAULT_TRACE_NAME
    if args.run_card is None:
        args.run_card = args.run_dir / DEFAULT_RUN_CARD_NAME
    if args.pid_file is None:
        args.pid_file = args.run_dir / "ft.pid"
    if args.log_file is None:
        args.log_file = args.run_dir / DEFAULT_LOG_NAME
    if args.final_checkpoint is None and args.max_steps is not None:
        args.final_checkpoint = f"checkpoint-{args.max_steps}"
    return args


def _number_text(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _read_pid(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None
    return pid if pid > 0 else None


def _process_status(pid: int | None) -> str:
    if pid is None:
        return "unknown"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "exited"
    except PermissionError:
        return "alive"
    return "alive"


def _process_command_args(pid: int | None) -> list[str]:
    if pid is None:
        return []
    try:
        result = subprocess.run(
            ["ps", "-p", str(int(pid)), "-o", "command="],
            capture_output=True,
            check=False,
            text=True,
        )
    except (OSError, ValueError):
        return []
    if result.returncode != 0:
        return []
    command = result.stdout.strip()
    return command.split() if command else []


def _command_flag_value(command: list[str], flag: str) -> str | None:
    for index, item in enumerate(command):
        if item == flag and index + 1 < len(command):
            return command[index + 1]
        if item.startswith(f"{flag}="):
            return item.split("=", 1)[1]
    return None


def _positive_int_flag(command: list[str], flag: str) -> int | None:
    value = _command_flag_value(command, flag)
    if value is None:
        return None
    try:
        parsed = int(float(value))
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _nonnegative_float_flag(command: list[str], flag: str) -> float | None:
    value = _command_flag_value(command, flag)
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if parsed >= 0.0 else None


def _read_tail(path: Path, max_bytes: int) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes), os.SEEK_SET)
        return handle.read().decode("utf-8", errors="ignore")


def _parse_tqdm_duration_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    parts = value.strip().split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None
    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes * 60.0 + seconds
    hours, minutes, seconds = numbers
    return hours * 3600.0 + minutes * 60.0 + seconds


def _log_progress(
    log_file: Path, max_bytes: int, expected_max_steps: int | None = None
) -> dict[str, Any]:
    text = _read_tail(log_file, max_bytes)
    latest_step = None
    latest_max_steps = None
    latest_elapsed_seconds = None
    latest_remaining_seconds = None
    for match in PROGRESS_RE.finditer(text):
        max_steps = int(match.group("max_steps"))
        if expected_max_steps is not None and max_steps != expected_max_steps:
            continue
        latest_step = int(match.group("step"))
        latest_max_steps = max_steps
        latest_elapsed_seconds = _parse_tqdm_duration_seconds(match.group("elapsed"))
        latest_remaining_seconds = _parse_tqdm_duration_seconds(
            match.group("remaining")
        )
    progress = None
    if latest_step is not None and latest_max_steps:
        progress = min(max(float(latest_step) / float(latest_max_steps), 0.0), 1.0)
    return {
        "log_file": str(log_file),
        "log_status": "ok" if log_file.is_file() else "missing",
        "log_latest_step": latest_step,
        "log_max_steps": latest_max_steps,
        "log_progress": progress,
        "log_elapsed_seconds": latest_elapsed_seconds,
        "log_remaining_seconds": latest_remaining_seconds,
    }


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _resolved_max_steps(trace: dict[str, Any], explicit: int | None) -> int | None:
    if explicit is not None:
        return explicit
    inferred = _int_value(trace.get("trace_max_steps"))
    return inferred if inferred is not None and inferred > 0 else None


def _log_progress_with_trace_fallback(
    log_progress: dict[str, Any],
    trace: dict[str, Any],
    max_steps: int | None,
) -> dict[str, Any]:
    if log_progress.get("log_latest_step") is not None:
        return log_progress
    trace_step = _int_value(trace.get("trace_max_global_step"))
    if trace_step is None:
        return log_progress
    updated = dict(log_progress)
    updated["log_latest_step"] = trace_step
    updated["log_max_steps"] = max_steps
    updated["log_progress"] = (
        min(max(float(trace_step) / float(max_steps), 0.0), 1.0)
        if max_steps is not None and max_steps > 0
        else None
    )
    updated["log_status"] = (
        "fallback_trace"
        if updated.get("log_status") == "ok"
        else updated.get("log_status")
    )
    return updated


def _last_eval_loss_step(trace: dict[str, Any]) -> int | None:
    eval_points = trace.get("trace_eval_loss_points")
    if not isinstance(eval_points, list):
        return None
    for point in reversed(eval_points):
        if not isinstance(point, dict):
            continue
        step = point.get("step")
        if isinstance(step, int):
            return step
    return None


def _eval_step_reached(trace: dict[str, Any], target_step: int) -> bool:
    latest = _last_eval_loss_step(trace)
    return latest is not None and latest >= target_step


def _trace_summary(trace_jsonl: Path, max_steps: int | None) -> dict[str, Any]:
    if not trace_jsonl.is_file():
        return {
            "trace_jsonl": str(trace_jsonl),
            "trace_status": "missing",
            "trace_event_count": 0,
            "trace_max_global_step": None,
            "progress": None,
            "training_loss_guard_count": 0,
        }
    rows = st.load_hf_gpt2_finetune_trainer_trace(trace_jsonl)
    summary = st.summarize_hf_gpt2_finetune_trainer_trace(rows, max_steps=max_steps)
    summary["trace_last_eval_loss_step"] = _last_eval_loss_step(summary)
    resolved_max_steps = _resolved_max_steps(summary, max_steps)
    step = summary.get("trace_max_global_step")
    progress = None
    if resolved_max_steps is not None and isinstance(step, (int, float)):
        progress = min(max(float(step) / float(resolved_max_steps), 0.0), 1.0)
    summary.update(
        {
            "trace_jsonl": str(trace_jsonl),
            "trace_status": "ok",
            "max_steps": resolved_max_steps,
            "progress": progress,
            "training_loss_guard_count": sum(
                1 for row in rows if row.get("training_loss_guard")
            ),
        }
    )
    return summary


def _infer_eval_steps(trace: dict[str, Any]) -> int | None:
    eval_points = trace.get("trace_eval_loss_points")
    if not isinstance(eval_points, list):
        return None
    steps = []
    for point in eval_points:
        if not isinstance(point, dict):
            continue
        step = point.get("step")
        if isinstance(step, int) and step > 0:
            steps.append(step)
    steps = sorted(set(steps))
    if len(steps) < 2:
        return None
    intervals = [right - left for left, right in zip(steps, steps[1:]) if right > left]
    if not intervals:
        return None
    return intervals[-1]


def _eval_progress(
    trace: dict[str, Any],
    log_progress: dict[str, Any],
    eval_steps: int | None,
    max_steps: int | None,
) -> dict[str, Any]:
    interval = eval_steps or _infer_eval_steps(trace)
    trace_step = trace.get("trace_max_global_step")
    log_step = log_progress.get("log_latest_step")
    next_eval_step = None
    trace_steps_until_next_eval = None
    log_steps_until_next_eval = None
    latest_due_eval_step = None
    latest_due_eval_ready = None
    pending_eval_step = None
    log_steps_since_pending_eval = None
    trace_steps_since_pending_eval = None
    if interval is not None:
        basis = log_step if isinstance(log_step, int) else trace_step
        if isinstance(basis, (int, float)):
            latest_due_eval_step = int(int(basis) // interval) * interval
            if latest_due_eval_step <= 0:
                latest_due_eval_step = None
            next_eval_step = int(((int(basis) // interval) + 1) * interval)
            if max_steps is not None:
                next_eval_step = min(next_eval_step, max_steps)
                if latest_due_eval_step is not None:
                    latest_due_eval_step = min(latest_due_eval_step, max_steps)
            last_eval_step = _last_eval_loss_step(trace)
            if latest_due_eval_step is not None:
                latest_due_eval_ready = (
                    last_eval_step is not None and last_eval_step >= latest_due_eval_step
                )
                if not latest_due_eval_ready:
                    pending_eval_step = latest_due_eval_step
        if next_eval_step is not None and isinstance(trace_step, (int, float)):
            trace_steps_until_next_eval = max(int(next_eval_step) - int(trace_step), 0)
        if next_eval_step is not None and isinstance(log_step, int):
            log_steps_until_next_eval = max(int(next_eval_step) - log_step, 0)
        if pending_eval_step is not None and isinstance(log_step, int):
            log_steps_since_pending_eval = max(log_step - int(pending_eval_step), 0)
        if pending_eval_step is not None and isinstance(trace_step, (int, float)):
            trace_steps_since_pending_eval = max(
                int(trace_step) - int(pending_eval_step), 0
            )
    return {
        "eval_steps": interval,
        "next_eval_step": next_eval_step,
        "trace_steps_until_next_eval": trace_steps_until_next_eval,
        "log_steps_until_next_eval": log_steps_until_next_eval,
        "latest_due_eval_step": latest_due_eval_step,
        "latest_due_eval_ready": latest_due_eval_ready,
        "pending_eval_step": pending_eval_step,
        "log_steps_since_pending_eval": log_steps_since_pending_eval,
        "trace_steps_since_pending_eval": trace_steps_since_pending_eval,
    }


def _step_interval_progress(
    *,
    interval: int | None,
    log_progress: dict[str, Any],
    trace: dict[str, Any],
    max_steps: int | None,
    label: str,
) -> dict[str, Any]:
    next_step = None
    trace_steps_until_next = None
    log_steps_until_next = None
    if interval is not None:
        trace_step = trace.get("trace_max_global_step")
        log_step = log_progress.get("log_latest_step")
        basis = log_step if isinstance(log_step, int) else trace_step
        if isinstance(basis, (int, float)):
            next_step = int(((int(basis) // interval) + 1) * interval)
            if max_steps is not None:
                next_step = min(next_step, max_steps)
        if next_step is not None and isinstance(trace_step, (int, float)):
            trace_steps_until_next = max(int(next_step) - int(trace_step), 0)
        if next_step is not None and isinstance(log_step, int):
            log_steps_until_next = max(int(next_step) - log_step, 0)
    return {
        f"{label}_steps": interval,
        f"next_{label}_step": next_step,
        f"trace_steps_until_next_{label}": trace_steps_until_next,
        f"log_steps_until_next_{label}": log_steps_until_next,
    }


def _checkpoint_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("checkpoint-*")):
        if not path.is_dir():
            continue
        token = path.name.removeprefix("checkpoint-")
        try:
            step: int | None = int(token)
        except ValueError:
            step = None
        ready = (path / "model.safetensors").is_file()
        rows.append(
            {
                "name": path.name,
                "path": str(path),
                "step": step,
                "model_safetensors_ready": ready,
            }
        )
    return rows


def summarize_run(args: argparse.Namespace) -> dict[str, Any]:
    pid = _read_pid(args.pid_file)
    process_command = _process_command_args(pid)
    command_max_steps = _positive_int_flag(process_command, "--max-steps")
    command_eval_steps = _positive_int_flag(process_command, "--eval-steps")
    command_save_steps = _positive_int_flag(process_command, "--save-steps")
    command_save_total_limit = _positive_int_flag(
        process_command, "--save-total-limit"
    )
    command_min_free_disk_gb = _nonnegative_float_flag(
        process_command, "--min-free-disk-gb"
    )
    initial_max_steps = args.max_steps or command_max_steps
    trace = _trace_summary(args.trace_jsonl, initial_max_steps)
    max_steps = _resolved_max_steps(trace, initial_max_steps)
    eval_steps = args.eval_steps or command_eval_steps
    save_steps = args.save_steps or command_save_steps
    save_total_limit = args.save_total_limit or command_save_total_limit or 1
    min_free_disk_gb = (
        args.min_free_disk_gb
        if args.min_free_disk_gb is not None
        else command_min_free_disk_gb
    )
    log_progress = _log_progress(
        args.log_file,
        int(args.log_tail_bytes),
        expected_max_steps=max_steps,
    )
    log_progress = _log_progress_with_trace_fallback(
        log_progress,
        trace,
        max_steps,
    )
    eval_progress = _eval_progress(
        trace,
        log_progress,
        eval_steps,
        max_steps,
    )
    checkpoint_progress = _step_interval_progress(
        interval=save_steps,
        log_progress=log_progress,
        trace=trace,
        max_steps=max_steps,
        label="checkpoint",
    )
    run_card = _load_json(args.run_card)
    checkpoint_card = _load_json(args.checkpoint_card)
    checkpoints = _checkpoint_rows(args.run_dir)
    final_checkpoint_path = None
    final_checkpoint_ready = None
    final_checkpoint = args.final_checkpoint
    if final_checkpoint is None and max_steps is not None:
        final_checkpoint = f"checkpoint-{max_steps}"
    if final_checkpoint:
        final_checkpoint_path = args.run_dir / str(final_checkpoint)
        final_checkpoint_ready = (final_checkpoint_path / "model.safetensors").is_file()
    disk = shutil.disk_usage(args.run_dir)
    disk_free_gb = disk.free / (1024.0**3)
    disk_margin_gb = (
        None
        if min_free_disk_gb is None
        else disk_free_gb - float(min_free_disk_gb)
    )
    disk_status = (
        "unchecked"
        if min_free_disk_gb is None
        else "ok"
        if disk_margin_gb is not None and disk_margin_gb >= 0.0
        else "low"
    )
    latest_checkpoint = checkpoints[-1] if checkpoints else None
    latest_checkpoint_path = (
        latest_checkpoint.get("path") if isinstance(latest_checkpoint, dict) else None
    )
    checkpoint_headroom = st.hf_gpt2_finetune_disk_headroom_plan(
        args.run_dir,
        resume_from_checkpoint=latest_checkpoint_path,
        save_total_limit=save_total_limit,
    )
    return {
        "row_type": "hf_gpt2_finetune_run_status",
        "time_unix_s": time.time(),
        "run_dir": str(args.run_dir),
        "trace": trace,
        "log_progress": log_progress,
        "eval_progress": eval_progress,
        "checkpoint_progress": checkpoint_progress,
        "run_card_path": str(args.run_card),
        "run_card_status": (run_card or {}).get("status"),
        "runtime_settings": {
            "max_steps": max_steps,
            "eval_steps": eval_steps,
            "save_steps": save_steps,
            "save_total_limit": save_total_limit,
            "min_free_disk_gb": min_free_disk_gb,
            "process_command_available": bool(process_command),
        },
        "pid_file": str(args.pid_file),
        "pid": pid,
        "process_status": _process_status(pid),
        "checkpoint_card_path": str(args.checkpoint_card)
        if args.checkpoint_card is not None
        else None,
        "checkpoint_card_status": (checkpoint_card or {}).get("status"),
        "checkpoint_card_process_wait": (checkpoint_card or {}).get("process_wait"),
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "latest_checkpoint": latest_checkpoint,
        "save_total_limit": save_total_limit,
        "checkpoint_headroom": checkpoint_headroom,
        "final_checkpoint": final_checkpoint,
        "final_checkpoint_path": str(final_checkpoint_path)
        if final_checkpoint_path is not None
        else None,
        "final_checkpoint_ready": final_checkpoint_ready,
        "min_free_disk_gb": min_free_disk_gb,
        "disk_free_gb": disk_free_gb,
        "disk_margin_gb": disk_margin_gb,
        "disk_status": disk_status,
        "disk_total_gb": disk.total / (1024.0**3),
        "watch_stop_eval_step": args.watch_stop_on_eval_step,
        "watch_stop_eval_ready": None
        if args.watch_stop_on_eval_step is None
        else _eval_step_reached(trace, args.watch_stop_on_eval_step),
    }


def status_lines(status: dict[str, Any], *, tail_evals: int) -> list[str]:
    trace = status.get("trace") if isinstance(status.get("trace"), dict) else {}
    runtime_settings = (
        status.get("runtime_settings")
        if isinstance(status.get("runtime_settings"), dict)
        else {}
    )
    latest_checkpoint = status.get("latest_checkpoint")
    checkpoint_headroom = (
        status.get("checkpoint_headroom")
        if isinstance(status.get("checkpoint_headroom"), dict)
        else {}
    )
    latest_checkpoint_name = (
        latest_checkpoint.get("name") if isinstance(latest_checkpoint, dict) else None
    )
    lines = [
        (
            "hf_gpt2_ft_run_status "
            f"run_dir={status.get('run_dir')} "
            f"process={_number_text(status.get('process_status'))} "
            f"pid={_number_text(status.get('pid'))} "
            f"latest_step={_number_text(trace.get('trace_max_global_step'))} "
            f"max_steps={_number_text(trace.get('max_steps'))} "
            f"progress={_number_text(trace.get('progress'))} "
            f"runtime_eval_steps={_number_text(runtime_settings.get('eval_steps'))} "
            f"runtime_save_steps={_number_text(runtime_settings.get('save_steps'))} "
            f"runtime_process_command={_number_text(runtime_settings.get('process_command_available'))} "
            f"log_latest_step={_number_text((status.get('log_progress') or {}).get('log_latest_step'))} "
            f"log_progress={_number_text((status.get('log_progress') or {}).get('log_progress'))} "
            f"log_remaining_seconds={_number_text((status.get('log_progress') or {}).get('log_remaining_seconds'))} "
            f"next_eval_step={_number_text((status.get('eval_progress') or {}).get('next_eval_step'))} "
            f"log_steps_until_next_eval={_number_text((status.get('eval_progress') or {}).get('log_steps_until_next_eval'))} "
            f"latest_due_eval_step={_number_text((status.get('eval_progress') or {}).get('latest_due_eval_step'))} "
            f"latest_due_eval_ready={_number_text((status.get('eval_progress') or {}).get('latest_due_eval_ready'))} "
            f"pending_eval_step={_number_text((status.get('eval_progress') or {}).get('pending_eval_step'))} "
            f"log_steps_since_pending_eval={_number_text((status.get('eval_progress') or {}).get('log_steps_since_pending_eval'))} "
            f"next_checkpoint_step={_number_text((status.get('checkpoint_progress') or {}).get('next_checkpoint_step'))} "
            f"log_steps_until_next_checkpoint={_number_text((status.get('checkpoint_progress') or {}).get('log_steps_until_next_checkpoint'))} "
            f"last_loss={_number_text(trace.get('trace_last_loss'))} "
            f"last_eval_loss={_number_text(trace.get('trace_last_eval_loss'))} "
            f"last_eval_step={_number_text(trace.get('trace_last_eval_loss_step'))} "
            f"min_eval_loss={_number_text(trace.get('trace_min_eval_loss'))} "
            f"best_eval_loss_step={_number_text(trace.get('trace_best_eval_loss_step'))} "
            f"eval_loss_improvement={_number_text(trace.get('trace_eval_loss_improvement'))} "
            f"eval_loss_last_delta={_number_text(trace.get('trace_eval_loss_last_delta'))} "
            f"eval_loss_last_improvement_per_step={_number_text(trace.get('trace_eval_loss_last_improvement_per_step'))} "
            f"eval_loss_projected_final={_number_text(trace.get('trace_eval_loss_projected_final_loss'))} "
            f"eval_loss_monotonic={_number_text(trace.get('trace_eval_loss_monotonic_nonincreasing'))} "
            f"guard_count={_number_text(trace.get('training_loss_guard_count'))} "
            f"checkpoint_card={_number_text(status.get('checkpoint_card_status'))} "
            f"checkpoints={_number_text(status.get('checkpoint_count'))} "
            f"latest_checkpoint={_number_text(latest_checkpoint_name)} "
            f"save_total_limit={_number_text(status.get('save_total_limit'))} "
            f"checkpoint_headroom_checkpoint_gb={_number_text(checkpoint_headroom.get('resume_checkpoint_gb'))} "
            f"checkpoint_headroom_peak_gb={_number_text(checkpoint_headroom.get('estimated_peak_checkpoint_gb'))} "
            f"checkpoint_headroom_free_after_gb={_number_text(checkpoint_headroom.get('free_after_estimated_peak_gb'))} "
            f"final_ready={_number_text(status.get('final_checkpoint_ready'))} "
            f"disk_free_gb={_number_text(status.get('disk_free_gb'))} "
            f"min_free_disk_gb={_number_text(status.get('min_free_disk_gb'))} "
            f"disk_margin_gb={_number_text(status.get('disk_margin_gb'))} "
            f"disk_status={_number_text(status.get('disk_status'))} "
            f"watch_stop_eval_step={_number_text(status.get('watch_stop_eval_step'))} "
            f"watch_stop_eval_ready={_number_text(status.get('watch_stop_eval_ready'))} "
            f"watch_stop_reason={_number_text(status.get('watch_stop_reason'))}"
        )
    ]
    eval_points = trace.get("trace_eval_loss_points")
    if isinstance(eval_points, list) and tail_evals > 0:
        for point in eval_points[-tail_evals:]:
            if not isinstance(point, dict):
                continue
            lines.append(
                (
                    "hf_gpt2_ft_run_eval "
                    f"step={_number_text(point.get('step'))} "
                    f"eval_loss={_number_text(point.get('eval_loss'))} "
                    f"eval_runtime={_number_text(point.get('eval_runtime'))}"
                )
            )
    process_wait = status.get("checkpoint_card_process_wait")
    if isinstance(process_wait, dict):
        lines.append(
            (
                "hf_gpt2_ft_run_wait "
                f"status={_number_text(process_wait.get('status'))} "
                f"pid={_number_text(process_wait.get('pid'))} "
                f"waited_seconds={_number_text(process_wait.get('waited_seconds'))}"
            )
        )
    return lines


def _emit_status(args: argparse.Namespace, status: dict[str, Any]) -> None:
    lines = status_lines(status, tail_evals=args.tail_evals)
    payload = json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        if not args.quiet:
            print(f"hf_gpt2_ft_run_status_json {args.out}")
    if args.lines_out is not None:
        args.lines_out.parent.mkdir(parents=True, exist_ok=True)
        args.lines_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"hf_gpt2_ft_run_status_lines {args.lines_out}")
    if args.jsonl_out is not None:
        args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl_out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(status, ensure_ascii=False, sort_keys=True) + "\n")
        if not args.quiet:
            print(f"hf_gpt2_ft_run_status_jsonl {args.jsonl_out}")
    if args.out is None and args.lines_out is None and args.jsonl_out is None:
        print("\n".join(lines))


def _should_stop_watch(args: argparse.Namespace, status: dict[str, Any]) -> bool:
    return _watch_stop_reason(args, status) is not None


def _watch_stop_reason(args: argparse.Namespace, status: dict[str, Any]) -> str | None:
    if args.watch_stop_on_final and status.get("final_checkpoint_ready") is True:
        return "final_checkpoint_ready"
    if args.watch_stop_on_checkpoint is not None and _checkpoint_target_ready(
        status, args.watch_stop_on_checkpoint
    ):
        return "checkpoint_ready"
    if args.watch_stop_on_process_exit and status.get("process_status") == "exited":
        return "process_exited"
    if args.watch_stop_on_disk_low and status.get("disk_status") == "low":
        return "disk_low"
    if args.watch_stop_on_training_guard:
        trace = status.get("trace") if isinstance(status.get("trace"), dict) else {}
        guard_count = trace.get("training_loss_guard_count")
        if isinstance(guard_count, int) and guard_count > 0:
            return "training_loss_guard"
    if args.watch_stop_on_eval_step is not None:
        trace = status.get("trace") if isinstance(status.get("trace"), dict) else {}
        if _eval_step_reached(trace, args.watch_stop_on_eval_step):
            return "eval_step_reached"
    return None


def _checkpoint_target_ready(status: dict[str, Any], target: str) -> bool:
    checkpoints = status.get("checkpoints")
    if not isinstance(checkpoints, list):
        return False
    token = target.removeprefix("checkpoint-")
    target_step = int(token) if token.isdigit() else None
    target_name = f"checkpoint-{target_step}" if target_step is not None else target
    for row in checkpoints:
        if not isinstance(row, dict):
            continue
        if row.get("model_safetensors_ready") is not True:
            continue
        if row.get("name") == target or row.get("name") == target_name:
            return True
        if target_step is not None and row.get("step") == target_step:
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.watch_interval_seconds is None:
        _emit_status(args, summarize_run(args))
        return 0
    count = 0
    while True:
        if count and not args.quiet:
            print()
        status = summarize_run(args)
        stop_reason = _watch_stop_reason(args, status)
        if stop_reason is not None:
            status["watch_stop_reason"] = stop_reason
        _emit_status(args, status)
        count += 1
        if stop_reason is not None:
            break
        if args.watch_count is not None and count >= args.watch_count:
            break
        time.sleep(float(args.watch_interval_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
