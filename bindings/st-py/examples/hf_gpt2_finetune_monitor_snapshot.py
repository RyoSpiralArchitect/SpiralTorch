#!/usr/bin/env python3
"""Build a one-page monitor snapshot for a long SpiralTorch GPT-2 FT run."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--next-run-dir", type=Path, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--run-status-json", type=Path, default=None)
    parser.add_argument("--run-status-history-jsonl", type=Path, default=None)
    parser.add_argument("--eval-history-jsonl", type=Path, default=None)
    parser.add_argument("--checkpoint-history-jsonl", type=Path, default=None)
    parser.add_argument("--final-history-jsonl", type=Path, default=None)
    parser.add_argument("--wait-launch-history-jsonl", type=Path, default=None)
    parser.add_argument("--milestone-step", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--lines-out", type=Path, default=None)
    parser.add_argument("--require-final-ready", action="store_true")
    parser.add_argument("--require-milestone-ready", action="store_true")
    parser.add_argument("--require-wait-launched", action="store_true")
    args = parser.parse_args(argv)
    provided = [
        args.run_status_json,
        args.run_status_history_jsonl,
        args.eval_history_jsonl,
        args.checkpoint_history_jsonl,
        args.final_history_jsonl,
        args.wait_launch_history_jsonl,
    ]
    if args.run_dir is None and not any(provided):
        parser.error("provide run_dir or at least one history JSONL")
    if args.run_dir is not None and not args.run_dir.is_dir():
        parser.error(f"run_dir does not exist: {args.run_dir}")
    if args.next_run_dir is not None and not args.next_run_dir.is_dir():
        parser.error(f"next_run_dir does not exist: {args.next_run_dir}")
    if args.milestone_step is not None and args.milestone_step < 0:
        parser.error("--milestone-step must be non-negative")
    for path in provided:
        if path is not None and not path.is_file():
            parser.error(f"monitor input does not exist: {path}")
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


def _nested(row: dict[str, Any], section: str, field: str) -> Any:
    value = row.get(section)
    if not isinstance(value, dict):
        return None
    return value.get(field)


def _load_history(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"history row {line_number} is not an object")
        rows.append(payload)
    return rows


def _load_status_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("run status JSON is not an object")
    return payload


def _latest_path(directory: Path | None, patterns: list[str]) -> Path | None:
    if directory is None or not directory.is_dir():
        return None
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(path for path in directory.glob(pattern) if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))


def _last_eval_loss_step(row: dict[str, Any]) -> Any:
    explicit = _nested(row, "trace", "trace_last_eval_loss_step")
    if explicit is not None:
        return explicit
    eval_points = _nested(row, "trace", "trace_eval_loss_points")
    if not isinstance(eval_points, list):
        return None
    for point in reversed(eval_points):
        if not isinstance(point, dict):
            continue
        step = point.get("step")
        if isinstance(step, int):
            return step
    return None


def _checkpoint_name(row: dict[str, Any]) -> Any:
    checkpoint = row.get("latest_checkpoint")
    if isinstance(checkpoint, dict):
        return checkpoint.get("name")
    return None


def _checkpoint_names(row: dict[str, Any]) -> list[str]:
    names: list[str] = []
    latest = _checkpoint_name(row)
    if isinstance(latest, str):
        names.append(latest)
    checkpoints = row.get("checkpoints")
    if isinstance(checkpoints, list):
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict):
                continue
            name = checkpoint.get("name")
            if isinstance(name, str):
                names.append(name)
    return list(dict.fromkeys(names))


def _checkpoint_headroom(row: dict[str, Any]) -> dict[str, Any]:
    headroom = row.get("checkpoint_headroom")
    return headroom if isinstance(headroom, dict) else {}


def _runtime_settings(row: dict[str, Any]) -> dict[str, Any]:
    runtime = row.get("runtime_settings")
    return runtime if isinstance(runtime, dict) else {}


def _runtime_setting(row: dict[str, Any], field: str) -> Any:
    runtime = _runtime_settings(row)
    value = runtime.get(field)
    if value is not None:
        return value
    if field == "max_steps":
        return _nested(row, "trace", "max_steps") or _nested(
            row, "log_progress", "log_max_steps"
        )
    if field == "save_total_limit":
        return row.get("save_total_limit")
    if field == "min_free_disk_gb":
        return row.get("min_free_disk_gb")
    return None


def _launch_disk_guard(row: dict[str, Any]) -> dict[str, Any]:
    guard = row.get("launch_disk_guard")
    if isinstance(guard, dict):
        return guard
    return _reconstructed_launch_disk_guard(row)


def _command_flag_value(command: Any, flag: str) -> str | None:
    if not isinstance(command, list):
        return None
    values = [str(item) for item in command]
    for index, item in enumerate(values):
        if item == flag and index + 1 < len(values):
            return values[index + 1]
        if item.startswith(f"{flag}="):
            return item.split("=", 1)[1]
    return None


def _positive_int_flag(command: Any, flag: str) -> int | None:
    value = _command_flag_value(command, flag)
    if value is None:
        return None
    try:
        parsed = int(float(value))
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _nonnegative_float_flag(command: Any, flag: str) -> float | None:
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


def _checkpoint_step(path: Path) -> int | None:
    token = path.name.removeprefix("checkpoint-")
    try:
        return int(token)
    except ValueError:
        return None


def _checkpoint_size_estimate(path: Path) -> tuple[int | None, str | None]:
    exact = _path_size_bytes(path)
    if exact is not None:
        return exact, str(path)
    target_step = _checkpoint_step(path)
    if target_step is None or not path.parent.is_dir():
        return None, None
    candidates: list[tuple[int, Path]] = []
    for candidate in path.parent.glob("checkpoint-*"):
        if not candidate.is_dir():
            continue
        step = _checkpoint_step(candidate)
        if step is None or step > target_step:
            continue
        if not (candidate / "model.safetensors").is_file():
            continue
        candidates.append((step, candidate))
    if not candidates:
        return None, None
    _, source = max(candidates, key=lambda item: item[0])
    return _path_size_bytes(source), str(source)


def _reconstructed_launch_disk_guard(row: dict[str, Any]) -> dict[str, Any]:
    command = row.get("command")
    output_dir_value = _command_flag_value(command, "--output-dir")
    resume_value = _command_flag_value(command, "--resume-from-checkpoint")
    checkpoint_value = row.get("checkpoint")
    output_dir = Path(output_dir_value) if output_dir_value else None
    resume_checkpoint = (
        Path(resume_value)
        if resume_value is not None
        else Path(checkpoint_value)
        if isinstance(checkpoint_value, str)
        else None
    )
    if output_dir is None and resume_checkpoint is None:
        return {}
    gib = 1024.0**3
    min_free_gb = _nonnegative_float_flag(command, "--min-free-disk-gb")
    save_total_limit = _positive_int_flag(command, "--save-total-limit") or 1
    checkpoint_bytes, checkpoint_estimate_source = (
        (None, None)
        if resume_checkpoint is None
        else _checkpoint_size_estimate(resume_checkpoint)
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
        status = "reconstructed_unchecked"
        meets_min_free = None
    elif checked_free_gb is not None:
        meets_min_free = checked_free_gb >= float(min_free_gb)
        status = "reconstructed_ok" if meets_min_free else "reconstructed_blocked"
    elif free_bytes is not None:
        free_gb = float(free_bytes) / gib
        meets_min_free = free_gb >= float(min_free_gb)
        status = "reconstructed_ok" if meets_min_free else "reconstructed_blocked"
    else:
        status = "reconstructed_unknown"
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
        "resume_checkpoint_estimate_source": checkpoint_estimate_source,
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


def _eval_loss_points(row: dict[str, Any]) -> list[dict[str, Any]]:
    points = _nested(row, "trace", "trace_eval_loss_points")
    if not isinstance(points, list):
        return []
    clean: list[dict[str, Any]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        step = point.get("step")
        eval_loss = point.get("eval_loss")
        if not isinstance(step, int) or not isinstance(eval_loss, (int, float)):
            continue
        clean.append(
            {
                "step": step,
                "eval_loss": float(eval_loss),
                "eval_runtime": point.get("eval_runtime"),
                "time_unix_s": point.get("time_unix_s"),
            }
        )
    return clean


def _duration_seconds(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    first_time = rows[0].get("time_unix_s")
    last_time = rows[-1].get("time_unix_s")
    if isinstance(first_time, (int, float)) and isinstance(last_time, (int, float)):
        return float(last_time) - float(first_time)
    return None


def _delta_log_step(rows: list[dict[str, Any]]) -> int | None:
    if not rows:
        return None
    first_step = _nested(rows[0], "log_progress", "log_latest_step")
    last_step = _nested(rows[-1], "log_progress", "log_latest_step")
    if isinstance(first_step, int) and isinstance(last_step, int):
        return last_step - first_step
    return None


def _log_steps_per_second(rows: list[dict[str, Any]]) -> float | None:
    duration = _duration_seconds(rows)
    delta = _delta_log_step(rows)
    if duration is not None and duration > 0.0 and isinstance(delta, int):
        return float(delta) / duration
    return None


def _estimated_seconds(steps: Any, rows: list[dict[str, Any]]) -> float | None:
    rate = _log_steps_per_second(rows)
    if isinstance(steps, int) and rate is not None and rate > 0.0:
        return float(steps) / rate
    return None


def _estimated_seconds_with_fallback(
    steps: Any,
    rows: list[dict[str, Any]],
    *,
    total_steps: Any,
    total_seconds: Any,
) -> float | None:
    estimated = _estimated_seconds(steps, rows)
    if estimated is not None:
        return estimated
    if (
        isinstance(steps, int)
        and isinstance(total_steps, int)
        and total_steps > 0
        and isinstance(total_seconds, (int, float))
    ):
        return float(steps) * float(total_seconds) / float(total_steps)
    return None


def _status_watch_summary(
    rows: list[dict[str, Any]],
    *,
    name: str,
    history_jsonl: Path | None,
) -> dict[str, Any]:
    last = rows[-1] if rows else {}
    eval_losses = [
        _nested(row, "trace", "trace_last_eval_loss")
        for row in rows
        if isinstance(_nested(row, "trace", "trace_last_eval_loss"), (int, float))
    ]
    losses = [
        _nested(row, "trace", "trace_last_loss")
        for row in rows
        if isinstance(_nested(row, "trace", "trace_last_loss"), (int, float))
    ]
    steps_until_next_eval = _nested(
        last, "eval_progress", "log_steps_until_next_eval"
    )
    steps_until_next_checkpoint = _nested(
        last, "checkpoint_progress", "log_steps_until_next_checkpoint"
    )
    max_steps = _nested(last, "log_progress", "log_max_steps")
    log_step = _nested(last, "log_progress", "log_latest_step")
    steps_until_final = (
        max(int(max_steps) - int(log_step), 0)
        if isinstance(max_steps, int) and isinstance(log_step, int)
        else None
    )
    log_remaining_seconds = _nested(last, "log_progress", "log_remaining_seconds")
    eval_loss_points = _eval_loss_points(last)
    checkpoint_headroom = _checkpoint_headroom(last)
    return {
        "name": name,
        "history_jsonl": str(history_jsonl) if history_jsonl is not None else None,
        "row_count": len(rows),
        "last_time_unix_s": last.get("time_unix_s"),
        "duration_seconds": _duration_seconds(rows),
        "delta_log_step": _delta_log_step(rows),
        "log_steps_per_second": _log_steps_per_second(rows),
        "process_status": last.get("process_status"),
        "runtime_max_steps": _runtime_setting(last, "max_steps"),
        "runtime_eval_steps": _runtime_setting(last, "eval_steps"),
        "runtime_save_steps": _runtime_setting(last, "save_steps"),
        "runtime_save_total_limit": _runtime_setting(last, "save_total_limit"),
        "runtime_min_free_disk_gb": _runtime_setting(last, "min_free_disk_gb"),
        "runtime_process_command_available": _runtime_setting(
            last, "process_command_available"
        ),
        "log_latest_step": log_step,
        "log_max_steps": max_steps,
        "log_remaining_seconds": log_remaining_seconds,
        "steps_until_final": steps_until_final,
        "estimated_seconds_until_final": _estimated_seconds_with_fallback(
            steps_until_final,
            rows,
            total_steps=steps_until_final,
            total_seconds=log_remaining_seconds,
        ),
        "next_eval_step": _nested(last, "eval_progress", "next_eval_step"),
        "steps_until_next_eval": steps_until_next_eval,
        "estimated_seconds_until_next_eval": _estimated_seconds_with_fallback(
            steps_until_next_eval,
            rows,
            total_steps=steps_until_final,
            total_seconds=log_remaining_seconds,
        ),
        "latest_due_eval_step": _nested(
            last, "eval_progress", "latest_due_eval_step"
        ),
        "latest_due_eval_ready": _nested(
            last, "eval_progress", "latest_due_eval_ready"
        ),
        "pending_eval_step": _nested(last, "eval_progress", "pending_eval_step"),
        "log_steps_since_pending_eval": _nested(
            last, "eval_progress", "log_steps_since_pending_eval"
        ),
        "next_checkpoint_step": _nested(
            last, "checkpoint_progress", "next_checkpoint_step"
        ),
        "steps_until_next_checkpoint": steps_until_next_checkpoint,
        "estimated_seconds_until_next_checkpoint": _estimated_seconds_with_fallback(
            steps_until_next_checkpoint,
            rows,
            total_steps=steps_until_final,
            total_seconds=log_remaining_seconds,
        ),
        "last_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "last_eval_loss": _nested(last, "trace", "trace_last_eval_loss"),
        "last_eval_loss_step": _last_eval_loss_step(last),
        "eval_loss_points": eval_loss_points,
        "min_eval_loss": min(eval_losses) if eval_losses else None,
        "best_eval_loss_step": _nested(last, "trace", "trace_best_eval_loss_step"),
        "eval_loss_improvement": _nested(
            last, "trace", "trace_eval_loss_improvement"
        ),
        "eval_loss_last_delta": _nested(last, "trace", "trace_eval_loss_last_delta"),
        "eval_loss_last_improvement": _nested(
            last, "trace", "trace_eval_loss_last_improvement"
        ),
        "eval_loss_last_improvement_per_step": _nested(
            last, "trace", "trace_eval_loss_last_improvement_per_step"
        ),
        "eval_loss_mean_improvement_per_step": _nested(
            last, "trace", "trace_eval_loss_mean_improvement_per_step"
        ),
        "eval_loss_last_improvement_ratio_to_previous": _nested(
            last, "trace", "trace_eval_loss_last_improvement_ratio_to_previous"
        ),
        "eval_loss_projection_step": _nested(
            last, "trace", "trace_eval_loss_projection_step"
        ),
        "eval_loss_projection_remaining_steps": _nested(
            last, "trace", "trace_eval_loss_projection_remaining_steps"
        ),
        "eval_loss_projected_remaining_improvement": _nested(
            last, "trace", "trace_eval_loss_projected_remaining_improvement"
        ),
        "eval_loss_projected_final_loss": _nested(
            last, "trace", "trace_eval_loss_projected_final_loss"
        ),
        "eval_loss_monotonic_nonincreasing": _nested(
            last, "trace", "trace_eval_loss_monotonic_nonincreasing"
        ),
        "training_loss_guard_count": _nested(
            last, "trace", "training_loss_guard_count"
        ),
        "final_checkpoint_ready": last.get("final_checkpoint_ready"),
        "checkpoint_count": last.get("checkpoint_count"),
        "latest_checkpoint": _checkpoint_name(last),
        "checkpoint_names": _checkpoint_names(last),
        "save_total_limit": last.get("save_total_limit"),
        "checkpoint_headroom_checkpoint_gb": checkpoint_headroom.get(
            "resume_checkpoint_gb"
        ),
        "checkpoint_headroom_peak_gb": checkpoint_headroom.get(
            "estimated_peak_checkpoint_gb"
        ),
        "checkpoint_headroom_free_after_gb": checkpoint_headroom.get(
            "free_after_estimated_peak_gb"
        ),
        "disk_free_gb": last.get("disk_free_gb"),
        "disk_margin_gb": last.get("disk_margin_gb"),
        "disk_status": last.get("disk_status"),
        "watch_stop_eval_step": last.get("watch_stop_eval_step"),
        "watch_stop_eval_ready": last.get("watch_stop_eval_ready"),
        "watch_stop_reason": last.get("watch_stop_reason"),
    }


def _wait_launch_summary(
    rows: list[dict[str, Any]],
    *,
    history_jsonl: Path | None,
) -> dict[str, Any]:
    last = rows[-1] if rows else {}
    launch_disk_guard = _launch_disk_guard(last)
    launched_rows = [
        row
        for row in rows
        if row.get("launched_pid") is not None
        or row.get("status") in {"launching", "launched", "finished", "launch_error"}
    ]
    return {
        "history_jsonl": str(history_jsonl) if history_jsonl is not None else None,
        "row_count": len(rows),
        "last_time_unix_s": last.get("time_unix_s"),
        "duration_seconds": _duration_seconds(rows),
        "status": last.get("status"),
        "process_alive": last.get("process_alive"),
        "checkpoint_ready": last.get("checkpoint_ready"),
        "status_card_status": last.get("status_card_status"),
        "launched": bool(launched_rows),
        "launched_pid": last.get("launched_pid"),
        "returncode": last.get("returncode"),
        "launch_error": last.get("launch_error"),
        "launch_disk_status": launch_disk_guard.get("status"),
        "launch_disk_min_free_gb": launch_disk_guard.get("min_free_gb"),
        "launch_disk_peak_gb": launch_disk_guard.get(
            "estimated_peak_checkpoint_gb"
        ),
        "launch_disk_free_after_gb": launch_disk_guard.get(
            "free_after_estimated_peak_gb"
        ),
    }


def _latest_status_watch(watches: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        watch for watch in watches.values() if isinstance(watch.get("row_count"), int)
    ]
    candidates = [watch for watch in candidates if int(watch.get("row_count") or 0) > 0]
    if not candidates:
        return {}
    return max(
        candidates,
        key=lambda watch: (
            watch.get("last_time_unix_s")
            if isinstance(watch.get("last_time_unix_s"), (int, float))
            else -1.0,
            watch.get("name") or "",
        ),
    )


def _watch_field_with_direct_fallback(
    primary: dict[str, Any],
    direct: dict[str, Any],
    field: str,
) -> Any:
    value = primary.get(field)
    return direct.get(field) if value is None else value


def _eval_point_for_step(
    watches: dict[str, dict[str, Any]], milestone_step: int
) -> dict[str, Any] | None:
    for watch_name in ("direct", "eval", "checkpoint", "final"):
        watch = watches.get(watch_name)
        if not isinstance(watch, dict):
            continue
        points = watch.get("eval_loss_points")
        if not isinstance(points, list):
            continue
        for point in points:
            if not isinstance(point, dict):
                continue
            if point.get("step") == milestone_step:
                return {**point, "watch": watch_name}
    return None


def _has_checkpoint_for_step(
    watches: dict[str, dict[str, Any]], milestone_step: int
) -> bool:
    checkpoint_name = f"checkpoint-{milestone_step}"
    for watch in watches.values():
        if not isinstance(watch, dict):
            continue
        names = watch.get("checkpoint_names")
        if isinstance(names, list) and checkpoint_name in names:
            return True
        if watch.get("latest_checkpoint") == checkpoint_name:
            return True
    return False


def _milestone_summary(
    *,
    milestone_step: int | None,
    primary: dict[str, Any],
    watches: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if milestone_step is None:
        return {}
    log_step = primary.get("log_latest_step")
    step_reached = isinstance(log_step, int) and log_step >= milestone_step
    steps_until = (
        max(milestone_step - log_step, 0) if isinstance(log_step, int) else None
    )
    eval_point = _eval_point_for_step(watches, milestone_step)
    eval_ready = eval_point is not None
    checkpoint_ready = _has_checkpoint_for_step(watches, milestone_step)
    if eval_ready and checkpoint_ready:
        status = "ready"
    elif not step_reached:
        status = "waiting_for_step"
    elif not eval_ready:
        status = "waiting_for_eval"
    elif not checkpoint_ready:
        status = "waiting_for_checkpoint"
    else:
        status = "unknown"
    return {
        "milestone_step": milestone_step,
        "milestone_status": status,
        "milestone_ready": eval_ready and checkpoint_ready,
        "milestone_step_reached": step_reached,
        "milestone_steps_until": steps_until,
        "milestone_eval_ready": eval_ready,
        "milestone_eval_loss": eval_point.get("eval_loss")
        if eval_point is not None
        else None,
        "milestone_eval_watch": eval_point.get("watch")
        if eval_point is not None
        else None,
        "milestone_checkpoint_ready": checkpoint_ready,
        "milestone_checkpoint": f"checkpoint-{milestone_step}",
    }


def _resolve_sources(args: argparse.Namespace) -> dict[str, Path | None]:
    return {
        "direct": args.run_status_history_jsonl
        or args.run_status_json
        or _latest_path(
            args.run_dir,
            [
                "*run-status-history.jsonl",
                "run-status-history.jsonl",
                "*run-status.json",
                "run-status.json",
                "status.json",
            ],
        ),
        "eval": args.eval_history_jsonl
        or _latest_path(
            args.run_dir,
            ["watch-*-eval*-history.jsonl", "*eval*-history.jsonl"],
        ),
        "checkpoint": args.checkpoint_history_jsonl
        or _latest_path(
            args.run_dir,
            ["watch-*-checkpoint*-history.jsonl", "*checkpoint*-history.jsonl"],
        ),
        "final": args.final_history_jsonl
        or _latest_path(
            args.run_dir,
            ["watch-*-final*-history.jsonl", "*final*-history.jsonl"],
        ),
        "wait_launch": args.wait_launch_history_jsonl
        or _latest_path(
            args.next_run_dir,
            ["*-wait-launch-history.jsonl", "*wait*launch*history.jsonl"],
        )
        or _latest_path(
            args.run_dir,
            ["*-wait-launch-history.jsonl", "*wait*launch*history.jsonl"],
        ),
    }


def build_monitor_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    paths = _resolve_sources(args)
    loaded: dict[str, list[dict[str, Any]]] = {}
    for name, path in paths.items():
        if path is None:
            loaded[name] = []
        elif name == "direct" and path.suffix != ".jsonl":
            loaded[name] = [_load_status_json(path)]
        else:
            loaded[name] = _load_history(path)
    watches = {
        name: _status_watch_summary(
            loaded[name],
            name=name,
            history_jsonl=paths[name],
        )
        for name in ("direct", "eval", "checkpoint", "final")
    }
    wait_launch = _wait_launch_summary(
        loaded["wait_launch"], history_jsonl=paths["wait_launch"]
    )
    primary = _latest_status_watch(watches)
    all_times = [
        value
        for value in [
            *(watch.get("last_time_unix_s") for watch in watches.values()),
            wait_launch.get("last_time_unix_s"),
        ]
        if isinstance(value, (int, float))
    ]
    eval_watch = watches["eval"]
    checkpoint_watch = watches["checkpoint"]
    final_watch = watches["final"]
    direct_watch = watches["direct"]
    snapshot = {
        "row_type": "hf_gpt2_ft_monitor_snapshot",
        "label": args.label,
        "run_dir": str(args.run_dir) if args.run_dir is not None else None,
        "next_run_dir": str(args.next_run_dir)
        if args.next_run_dir is not None
        else None,
        "time_unix_s": max(all_times) if all_times else None,
        "primary_watch": primary.get("name"),
        "process_status": primary.get("process_status"),
        "runtime_max_steps": _watch_field_with_direct_fallback(
            primary, direct_watch, "runtime_max_steps"
        ),
        "runtime_eval_steps": _watch_field_with_direct_fallback(
            primary, direct_watch, "runtime_eval_steps"
        ),
        "runtime_save_steps": _watch_field_with_direct_fallback(
            primary, direct_watch, "runtime_save_steps"
        ),
        "runtime_save_total_limit": _watch_field_with_direct_fallback(
            primary, direct_watch, "runtime_save_total_limit"
        ),
        "runtime_min_free_disk_gb": _watch_field_with_direct_fallback(
            primary, direct_watch, "runtime_min_free_disk_gb"
        ),
        "runtime_process_command_available": _watch_field_with_direct_fallback(
            primary, direct_watch, "runtime_process_command_available"
        ),
        "log_latest_step": primary.get("log_latest_step"),
        "log_max_steps": primary.get("log_max_steps"),
        "log_remaining_seconds": primary.get("log_remaining_seconds"),
        "steps_until_final": primary.get("steps_until_final"),
        "estimated_seconds_until_final": primary.get(
            "estimated_seconds_until_final"
        ),
        "last_loss": primary.get("last_loss"),
        "min_loss": primary.get("min_loss"),
        "last_eval_loss": primary.get("last_eval_loss"),
        "last_eval_loss_step": primary.get("last_eval_loss_step"),
        "min_eval_loss": primary.get("min_eval_loss"),
        "best_eval_loss_step": primary.get("best_eval_loss_step"),
        "eval_loss_improvement": primary.get("eval_loss_improvement"),
        "eval_loss_last_delta": primary.get("eval_loss_last_delta"),
        "eval_loss_last_improvement": primary.get("eval_loss_last_improvement"),
        "eval_loss_last_improvement_per_step": primary.get(
            "eval_loss_last_improvement_per_step"
        ),
        "eval_loss_mean_improvement_per_step": primary.get(
            "eval_loss_mean_improvement_per_step"
        ),
        "eval_loss_last_improvement_ratio_to_previous": primary.get(
            "eval_loss_last_improvement_ratio_to_previous"
        ),
        "eval_loss_projection_step": primary.get("eval_loss_projection_step"),
        "eval_loss_projection_remaining_steps": primary.get(
            "eval_loss_projection_remaining_steps"
        ),
        "eval_loss_projected_remaining_improvement": primary.get(
            "eval_loss_projected_remaining_improvement"
        ),
        "eval_loss_projected_final_loss": primary.get(
            "eval_loss_projected_final_loss"
        ),
        "eval_loss_monotonic_nonincreasing": primary.get(
            "eval_loss_monotonic_nonincreasing"
        ),
        "next_eval_step": primary.get("next_eval_step"),
        "steps_until_next_eval": primary.get("steps_until_next_eval"),
        "estimated_seconds_until_next_eval": primary.get(
            "estimated_seconds_until_next_eval"
        ),
        "latest_due_eval_step": _watch_field_with_direct_fallback(
            primary, direct_watch, "latest_due_eval_step"
        ),
        "latest_due_eval_ready": _watch_field_with_direct_fallback(
            primary, direct_watch, "latest_due_eval_ready"
        ),
        "pending_eval_step": _watch_field_with_direct_fallback(
            primary, direct_watch, "pending_eval_step"
        ),
        "log_steps_since_pending_eval": _watch_field_with_direct_fallback(
            primary, direct_watch, "log_steps_since_pending_eval"
        ),
        "next_checkpoint_step": primary.get("next_checkpoint_step"),
        "steps_until_next_checkpoint": primary.get("steps_until_next_checkpoint"),
        "estimated_seconds_until_next_checkpoint": primary.get(
            "estimated_seconds_until_next_checkpoint"
        ),
        "training_loss_guard_count": primary.get("training_loss_guard_count"),
        "final_checkpoint_ready": primary.get("final_checkpoint_ready"),
        "checkpoint_count": primary.get("checkpoint_count"),
        "latest_checkpoint": primary.get("latest_checkpoint"),
        "save_total_limit": _watch_field_with_direct_fallback(
            primary, direct_watch, "save_total_limit"
        ),
        "checkpoint_headroom_checkpoint_gb": _watch_field_with_direct_fallback(
            primary, direct_watch, "checkpoint_headroom_checkpoint_gb"
        ),
        "checkpoint_headroom_peak_gb": _watch_field_with_direct_fallback(
            primary, direct_watch, "checkpoint_headroom_peak_gb"
        ),
        "checkpoint_headroom_free_after_gb": _watch_field_with_direct_fallback(
            primary, direct_watch, "checkpoint_headroom_free_after_gb"
        ),
        "disk_free_gb": primary.get("disk_free_gb"),
        "disk_margin_gb": primary.get("disk_margin_gb"),
        "disk_status": primary.get("disk_status"),
        "direct_status_available": bool(direct_watch.get("row_count")),
        "eval_watch_ready": eval_watch.get("watch_stop_eval_ready"),
        "eval_watch_step": eval_watch.get("watch_stop_eval_step"),
        "checkpoint_watch_reason": checkpoint_watch.get("watch_stop_reason"),
        "checkpoint_watch_final_ready": checkpoint_watch.get(
            "final_checkpoint_ready"
        ),
        "final_watch_reason": final_watch.get("watch_stop_reason"),
        "final_watch_ready": final_watch.get("final_checkpoint_ready"),
        "wait_launch_status": wait_launch.get("status"),
        "wait_launch_checkpoint_ready": wait_launch.get("checkpoint_ready"),
        "wait_launch_launched": wait_launch.get("launched"),
        "wait_launch_launched_pid": wait_launch.get("launched_pid"),
        "wait_launch_disk_status": wait_launch.get("launch_disk_status"),
        "wait_launch_disk_free_after_gb": wait_launch.get(
            "launch_disk_free_after_gb"
        ),
        "watches": watches,
        "wait_launch": wait_launch,
    }
    snapshot.update(
        _milestone_summary(
            milestone_step=args.milestone_step,
            primary=primary,
            watches=watches,
        )
    )
    return snapshot


def snapshot_lines(snapshot: dict[str, Any]) -> list[str]:
    label = snapshot.get("label") or "monitor"
    lines = [
        (
            "hf_gpt2_ft_monitor_snapshot "
            f"label={label} "
            f"primary={_number_text(snapshot.get('primary_watch'))} "
            f"process={_number_text(snapshot.get('process_status'))} "
            f"log_step={_number_text(snapshot.get('log_latest_step'))} "
            f"max_steps={_number_text(snapshot.get('log_max_steps'))} "
            f"runtime_max_steps={_number_text(snapshot.get('runtime_max_steps'))} "
            f"runtime_eval_steps={_number_text(snapshot.get('runtime_eval_steps'))} "
            f"runtime_save_steps={_number_text(snapshot.get('runtime_save_steps'))} "
            f"runtime_save_total_limit={_number_text(snapshot.get('runtime_save_total_limit'))} "
            f"runtime_min_free_disk_gb={_number_text(snapshot.get('runtime_min_free_disk_gb'))} "
            f"runtime_process_command={_number_text(snapshot.get('runtime_process_command_available'))} "
            f"log_remaining_seconds={_number_text(snapshot.get('log_remaining_seconds'))} "
            f"steps_until_final={_number_text(snapshot.get('steps_until_final'))} "
            f"estimated_seconds_until_final={_number_text(snapshot.get('estimated_seconds_until_final'))} "
            f"last_loss={_number_text(snapshot.get('last_loss'))} "
            f"min_loss={_number_text(snapshot.get('min_loss'))} "
            f"last_eval_step={_number_text(snapshot.get('last_eval_loss_step'))} "
            f"last_eval_loss={_number_text(snapshot.get('last_eval_loss'))} "
            f"min_eval_loss={_number_text(snapshot.get('min_eval_loss'))} "
            f"best_eval_loss_step={_number_text(snapshot.get('best_eval_loss_step'))} "
            f"eval_loss_improvement={_number_text(snapshot.get('eval_loss_improvement'))} "
            f"eval_loss_last_delta={_number_text(snapshot.get('eval_loss_last_delta'))} "
            f"eval_loss_last_improvement_per_step={_number_text(snapshot.get('eval_loss_last_improvement_per_step'))} "
            f"eval_loss_projected_final={_number_text(snapshot.get('eval_loss_projected_final_loss'))} "
            f"eval_loss_monotonic={_number_text(snapshot.get('eval_loss_monotonic_nonincreasing'))} "
            f"next_eval_step={_number_text(snapshot.get('next_eval_step'))} "
            f"steps_until_next_eval={_number_text(snapshot.get('steps_until_next_eval'))} "
            f"latest_due_eval_step={_number_text(snapshot.get('latest_due_eval_step'))} "
            f"latest_due_eval_ready={_number_text(snapshot.get('latest_due_eval_ready'))} "
            f"pending_eval_step={_number_text(snapshot.get('pending_eval_step'))} "
            f"log_steps_since_pending_eval={_number_text(snapshot.get('log_steps_since_pending_eval'))} "
            f"next_checkpoint_step={_number_text(snapshot.get('next_checkpoint_step'))} "
            f"steps_until_next_checkpoint={_number_text(snapshot.get('steps_until_next_checkpoint'))} "
            f"final_ready={_number_text(snapshot.get('final_checkpoint_ready'))} "
            f"latest_checkpoint={_number_text(snapshot.get('latest_checkpoint'))} "
            f"save_total_limit={_number_text(snapshot.get('save_total_limit'))} "
            f"checkpoint_headroom_peak_gb={_number_text(snapshot.get('checkpoint_headroom_peak_gb'))} "
            f"checkpoint_headroom_free_after_gb={_number_text(snapshot.get('checkpoint_headroom_free_after_gb'))} "
            f"disk_status={_number_text(snapshot.get('disk_status'))} "
            f"disk_margin_gb={_number_text(snapshot.get('disk_margin_gb'))} "
            f"guard_count={_number_text(snapshot.get('training_loss_guard_count'))} "
            f"direct_status_available={_number_text(snapshot.get('direct_status_available'))} "
            f"eval_watch_step={_number_text(snapshot.get('eval_watch_step'))} "
            f"eval_watch_ready={_number_text(snapshot.get('eval_watch_ready'))} "
            f"milestone_step={_number_text(snapshot.get('milestone_step'))} "
            f"milestone_status={_number_text(snapshot.get('milestone_status'))} "
            f"milestone_ready={_number_text(snapshot.get('milestone_ready'))} "
            f"milestone_steps_until={_number_text(snapshot.get('milestone_steps_until'))} "
            f"milestone_eval_ready={_number_text(snapshot.get('milestone_eval_ready'))} "
            f"milestone_eval_loss={_number_text(snapshot.get('milestone_eval_loss'))} "
            f"milestone_checkpoint_ready={_number_text(snapshot.get('milestone_checkpoint_ready'))} "
            f"checkpoint_watch_reason={_number_text(snapshot.get('checkpoint_watch_reason'))} "
            f"final_watch_reason={_number_text(snapshot.get('final_watch_reason'))} "
            f"wait_status={_number_text(snapshot.get('wait_launch_status'))} "
            f"wait_checkpoint_ready={_number_text(snapshot.get('wait_launch_checkpoint_ready'))} "
            f"wait_disk_status={_number_text(snapshot.get('wait_launch_disk_status'))} "
            f"wait_disk_free_after_gb={_number_text(snapshot.get('wait_launch_disk_free_after_gb'))} "
            f"wait_launched={_number_text(snapshot.get('wait_launch_launched'))}"
        )
    ]
    watches = snapshot.get("watches")
    if isinstance(watches, dict):
        for name in ("direct", "eval", "checkpoint", "final"):
            watch = watches.get(name)
            if not isinstance(watch, dict):
                continue
            lines.append(
                (
                    "hf_gpt2_ft_monitor_watch "
                    f"name={name} "
                    f"rows={_number_text(watch.get('row_count'))} "
                    f"log_step={_number_text(watch.get('log_latest_step'))} "
                    f"runtime_max_steps={_number_text(watch.get('runtime_max_steps'))} "
                    f"runtime_eval_steps={_number_text(watch.get('runtime_eval_steps'))} "
                    f"runtime_save_steps={_number_text(watch.get('runtime_save_steps'))} "
                    f"runtime_save_total_limit={_number_text(watch.get('runtime_save_total_limit'))} "
                    f"runtime_min_free_disk_gb={_number_text(watch.get('runtime_min_free_disk_gb'))} "
                    f"runtime_process_command={_number_text(watch.get('runtime_process_command_available'))} "
                    f"last_eval_step={_number_text(watch.get('last_eval_loss_step'))} "
                    f"last_eval_loss={_number_text(watch.get('last_eval_loss'))} "
                    f"eval_loss_projected_final={_number_text(watch.get('eval_loss_projected_final_loss'))} "
                    f"eval_loss_monotonic={_number_text(watch.get('eval_loss_monotonic_nonincreasing'))} "
                    f"next_eval_step={_number_text(watch.get('next_eval_step'))} "
                    f"pending_eval_step={_number_text(watch.get('pending_eval_step'))} "
                    f"next_checkpoint_step={_number_text(watch.get('next_checkpoint_step'))} "
                    f"final_ready={_number_text(watch.get('final_checkpoint_ready'))} "
                    f"checkpoint_headroom_peak_gb={_number_text(watch.get('checkpoint_headroom_peak_gb'))} "
                    f"checkpoint_headroom_free_after_gb={_number_text(watch.get('checkpoint_headroom_free_after_gb'))} "
                    f"watch_stop_eval_ready={_number_text(watch.get('watch_stop_eval_ready'))} "
                    f"watch_stop_reason={_number_text(watch.get('watch_stop_reason'))}"
                )
            )
    wait_launch = snapshot.get("wait_launch")
    if isinstance(wait_launch, dict):
        lines.append(
            (
                "hf_gpt2_ft_monitor_wait_launch "
                f"rows={_number_text(wait_launch.get('row_count'))} "
                f"status={_number_text(wait_launch.get('status'))} "
                f"process_alive={_number_text(wait_launch.get('process_alive'))} "
                f"checkpoint_ready={_number_text(wait_launch.get('checkpoint_ready'))} "
                f"status_card_status={_number_text(wait_launch.get('status_card_status'))} "
                f"launch_disk_status={_number_text(wait_launch.get('launch_disk_status'))} "
                f"launch_disk_free_after_gb={_number_text(wait_launch.get('launch_disk_free_after_gb'))} "
                f"launched={_number_text(wait_launch.get('launched'))} "
                f"launched_pid={_number_text(wait_launch.get('launched_pid'))} "
                f"returncode={_number_text(wait_launch.get('returncode'))}"
            )
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        snapshot = build_monitor_snapshot(args)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"failed to build monitor snapshot: {exc}", file=sys.stderr)
        return 1
    if args.require_final_ready and not snapshot.get("final_checkpoint_ready"):
        print("final checkpoint is not ready yet", file=sys.stderr)
        return 2
    if args.require_milestone_ready and not snapshot.get("milestone_ready"):
        print("milestone is not ready yet", file=sys.stderr)
        return 4
    if args.require_wait_launched and not snapshot.get("wait_launch_launched"):
        print("wait-launch has not launched a command yet", file=sys.stderr)
        return 3
    lines = snapshot_lines(snapshot)
    payload = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"hf_gpt2_ft_monitor_snapshot_json {args.out}")
    if args.lines_out is not None:
        args.lines_out.parent.mkdir(parents=True, exist_ok=True)
        args.lines_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"hf_gpt2_ft_monitor_snapshot_lines {args.lines_out}")
    if args.out is None and args.lines_out is None:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
