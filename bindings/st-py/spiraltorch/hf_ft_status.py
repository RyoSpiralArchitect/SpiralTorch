"""Reusable status-history helpers for SpiralTorch Hugging Face FT runs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "hf_gpt2_finetune_monitor_lines",
    "hf_gpt2_finetune_monitor_report",
    "hf_gpt2_finetune_milestone_capture_lines",
    "hf_gpt2_finetune_milestone_capture_report",
    "hf_gpt2_finetune_milestone_handoff_execution_lines",
    "hf_gpt2_finetune_milestone_handoff_execution_report",
    "hf_gpt2_finetune_milestone_handoff_lines",
    "hf_gpt2_finetune_milestone_handoff_report",
    "hf_gpt2_finetune_milestone_runtime_artifact_paths",
    "hf_gpt2_finetune_milestone_runtime_from_run_dir_archive",
    "hf_gpt2_finetune_milestone_runtime_from_run_dir_report",
    "hf_gpt2_finetune_milestone_runtime_lines",
    "hf_gpt2_finetune_milestone_runtime_report",
    "hf_gpt2_finetune_milestone_runtime_sources",
    "hf_gpt2_finetune_status_history_lines",
    "load_hf_gpt2_finetune_status_history",
    "main",
    "parse_args",
    "summarize_hf_gpt2_finetune_status_history",
    "write_hf_gpt2_finetune_milestone_runtime_report",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize SpiralTorch GPT-2 fine-tuning run status JSONL history."
    )
    parser.add_argument("history_jsonl", type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument("--tail", type=int, default=3)
    parser.add_argument(
        "--tail-evals",
        type=int,
        default=0,
        help="print the last N eval-loss points from the latest status row",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--lines-out", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.history_jsonl.is_file():
        parser.error(f"history_jsonl does not exist: {args.history_jsonl}")
    if args.tail < 0:
        parser.error("--tail must be non-negative")
    if args.tail_evals < 0:
        parser.error("--tail-evals must be non-negative")
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


def load_hf_gpt2_finetune_status_history(path: str | Path) -> list[dict[str, Any]]:
    history_path = Path(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        history_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"history row {line_number} is not an object")
        rows.append(payload)
    return rows


def _nested(row: dict[str, Any], section: str, field: str) -> Any:
    value = row.get(section)
    if not isinstance(value, dict):
        return None
    return value.get(field)


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _latest_checkpoint_name(row: dict[str, Any]) -> Any:
    checkpoint = row.get("latest_checkpoint")
    if isinstance(checkpoint, dict):
        return checkpoint.get("name")
    if isinstance(checkpoint, str):
        return Path(checkpoint).name
    return None


def _checkpoint_name(value: Any) -> str | None:
    if isinstance(value, Mapping):
        name = value.get("name")
        if isinstance(name, str) and name:
            return name
        path = value.get("path")
        if isinstance(path, (str, Path)) and str(path):
            return Path(path).name
    if isinstance(value, (str, Path)) and str(value):
        return Path(value).name
    return None


def _checkpoint_names(row: dict[str, Any]) -> list[str]:
    names: list[str] = []
    raw_names = row.get("checkpoint_names")
    if isinstance(raw_names, Sequence) and not isinstance(raw_names, (str, bytes)):
        names.extend(str(name) for name in raw_names if isinstance(name, str))
    checkpoints = row.get("checkpoints")
    if isinstance(checkpoints, Sequence) and not isinstance(checkpoints, (str, bytes)):
        for checkpoint in checkpoints:
            name = _checkpoint_name(checkpoint)
            if name is not None:
                names.append(name)
    latest = _latest_checkpoint_name(row)
    if isinstance(latest, str) and latest:
        names.append(latest)
    final_checkpoint = row.get("final_checkpoint")
    if row.get("final_checkpoint_ready") is True and isinstance(final_checkpoint, str):
        names.append(final_checkpoint)
    return list(dict.fromkeys(names))


def _checkpoint_headroom(row: dict[str, Any]) -> dict[str, Any]:
    headroom = row.get("checkpoint_headroom")
    return headroom if isinstance(headroom, dict) else {}


def _launch_disk_guard(row: dict[str, Any]) -> dict[str, Any]:
    guard = row.get("launch_disk_guard")
    if isinstance(guard, dict):
        return guard
    return {
        "status": row.get("launch_disk_status"),
        "min_free_gb": row.get("launch_disk_min_free_gb"),
        "estimated_peak_checkpoint_gb": row.get("launch_disk_peak_gb"),
        "free_after_estimated_peak_gb": row.get("launch_disk_free_after_gb"),
    }


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


def _coerce_status_rows(source: Any) -> tuple[list[dict[str, Any]], Path | None]:
    if source is None:
        return [], None
    if isinstance(source, (str, Path)):
        path = Path(source)
        text = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, Mapping):
            return [dict(payload)], path
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            return [dict(row) for row in payload if isinstance(row, Mapping)], path
        return load_hf_gpt2_finetune_status_history(path), path
    if isinstance(source, Mapping):
        return [dict(source)], None
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        return [dict(row) for row in source if isinstance(row, Mapping)], None
    raise TypeError("status source must be a path, mapping, sequence, or None")


def _first_log_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if isinstance(_nested(row, "log_progress", "log_latest_step"), int):
            return row
    return rows[0] if rows else {}


def _last_log_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in reversed(rows):
        if isinstance(_nested(row, "log_progress", "log_latest_step"), int):
            return row
    return rows[-1] if rows else {}


def _last_eval_loss_step(row: dict[str, Any]) -> Any:
    effective = _nested(row, "trace", "trace_effective_last_eval_loss_step")
    if effective is not None:
        return effective
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


def _eval_loss_points(row: dict[str, Any]) -> list[dict[str, Any]]:
    trace = row.get("trace")
    if not isinstance(trace, dict):
        return []
    points = trace.get("trace_eval_loss_points")
    if not isinstance(points, list):
        return []
    resume_baseline_eval_step = _int_value(
        trace.get("trace_resume_baseline_eval_step")
    )
    rows: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        if not isinstance(point, dict):
            continue
        raw_step = _int_value(point.get("step"))
        step = (
            resume_baseline_eval_step
            if raw_step == 0 and resume_baseline_eval_step is not None
            else raw_step
        )
        rows.append(
            {
                "index": index,
                "step": step,
                "raw_step": raw_step if step != raw_step else None,
                "eval_loss": point.get("eval_loss"),
                "eval_runtime": point.get("eval_runtime"),
                "time_unix_s": point.get("time_unix_s"),
            }
        )
    return rows


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


def _estimated_seconds_with_fallback(
    steps: Any,
    rows: list[dict[str, Any]],
    *,
    total_steps: Any,
    total_seconds: Any,
) -> float | None:
    rate = _log_steps_per_second(rows)
    if isinstance(steps, int) and rate is not None and rate > 0.0:
        return float(steps) / rate
    if (
        isinstance(steps, int)
        and isinstance(total_steps, int)
        and total_steps > 0
        and isinstance(total_seconds, (int, float))
    ):
        return float(steps) * float(total_seconds) / float(total_steps)
    return None


def _monitor_watch_summary(
    rows: list[dict[str, Any]],
    *,
    name: str,
    source_path: Path | None,
) -> dict[str, Any]:
    last = rows[-1] if rows else {}
    losses = [
        _nested(row, "trace", "trace_last_loss")
        for row in rows
        if isinstance(_nested(row, "trace", "trace_last_loss"), (int, float))
    ]
    eval_losses = [
        _nested(row, "trace", "trace_last_eval_loss")
        for row in rows
        if isinstance(_nested(row, "trace", "trace_last_eval_loss"), (int, float))
    ]
    log_step = _nested(last, "log_progress", "log_latest_step")
    max_steps = _nested(last, "log_progress", "log_max_steps")
    steps_until_final = (
        max(int(max_steps) - int(log_step), 0)
        if isinstance(max_steps, int) and isinstance(log_step, int)
        else None
    )
    log_remaining_seconds = _nested(last, "log_progress", "log_remaining_seconds")
    steps_until_next_eval = _nested(
        last, "eval_progress", "log_steps_until_next_eval"
    )
    steps_until_next_checkpoint = _nested(
        last, "checkpoint_progress", "log_steps_until_next_checkpoint"
    )
    checkpoint_headroom = _checkpoint_headroom(last)
    return {
        "name": name,
        "source_path": str(source_path) if source_path is not None else None,
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
        "eval_loss_points": _eval_loss_points(last),
        "min_eval_loss": min(eval_losses) if eval_losses else None,
        "best_eval_loss_step": _nested(last, "trace", "trace_best_eval_loss_step"),
        "eval_loss_improvement": _nested(
            last, "trace", "trace_eval_loss_improvement"
        ),
        "eval_loss_last_delta": _nested(last, "trace", "trace_eval_loss_last_delta"),
        "eval_loss_last_improvement_per_step": _nested(
            last, "trace", "trace_eval_loss_last_improvement_per_step"
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
        "latest_checkpoint": _latest_checkpoint_name(last),
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
    source_path: Path | None,
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
        "source_path": str(source_path) if source_path is not None else None,
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
        watch
        for watch in watches.values()
        if isinstance(watch.get("row_count"), int)
        and int(watch.get("row_count") or 0) > 0
    ]
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


def _watch_nullable_field_with_direct_fallback(
    primary: dict[str, Any],
    direct: dict[str, Any],
    field: str,
) -> Any:
    if int(primary.get("row_count") or 0) > 0 and field in primary:
        return primary.get(field)
    return direct.get(field)


def hf_gpt2_finetune_monitor_report(
    *,
    direct: Any = None,
    eval_watch: Any = None,
    checkpoint_watch: Any = None,
    final_watch: Any = None,
    wait_launch: Any = None,
    milestone_step: int | None = None,
    label: str | None = None,
    run_dir: str | Path | None = None,
    next_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build an importable monitor snapshot from FT status/watch artifacts."""

    sources = {
        "direct": _coerce_status_rows(direct),
        "eval": _coerce_status_rows(eval_watch),
        "checkpoint": _coerce_status_rows(checkpoint_watch),
        "final": _coerce_status_rows(final_watch),
    }
    watches = {
        name: _monitor_watch_summary(rows, name=name, source_path=source_path)
        for name, (rows, source_path) in sources.items()
    }
    wait_rows, wait_source = _coerce_status_rows(wait_launch)
    wait_summary = _wait_launch_summary(wait_rows, source_path=wait_source)
    primary = _latest_status_watch(watches)
    direct_watch = watches["direct"]
    all_times = [
        value
        for value in [
            *(watch.get("last_time_unix_s") for watch in watches.values()),
            wait_summary.get("last_time_unix_s"),
        ]
        if isinstance(value, (int, float))
    ]
    snapshot = {
        "row_type": "hf_gpt2_finetune_monitor_report",
        "label": label,
        "run_dir": None if run_dir is None else str(run_dir),
        "next_run_dir": None if next_run_dir is None else str(next_run_dir),
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
        "eval_loss_last_improvement_per_step": primary.get(
            "eval_loss_last_improvement_per_step"
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
        "pending_eval_step": _watch_nullable_field_with_direct_fallback(
            primary, direct_watch, "pending_eval_step"
        ),
        "log_steps_since_pending_eval": _watch_nullable_field_with_direct_fallback(
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
        "eval_watch_ready": watches["eval"].get("watch_stop_eval_ready"),
        "eval_watch_step": watches["eval"].get("watch_stop_eval_step"),
        "checkpoint_watch_reason": watches["checkpoint"].get("watch_stop_reason"),
        "checkpoint_watch_final_ready": watches["checkpoint"].get(
            "final_checkpoint_ready"
        ),
        "final_watch_reason": watches["final"].get("watch_stop_reason"),
        "final_watch_ready": watches["final"].get("final_checkpoint_ready"),
        "wait_launch_status": wait_summary.get("status"),
        "wait_launch_checkpoint_ready": wait_summary.get("checkpoint_ready"),
        "wait_launch_launched": wait_summary.get("launched"),
        "wait_launch_launched_pid": wait_summary.get("launched_pid"),
        "wait_launch_disk_status": wait_summary.get("launch_disk_status"),
        "wait_launch_disk_free_after_gb": wait_summary.get(
            "launch_disk_free_after_gb"
        ),
        "watches": watches,
        "wait_launch": wait_summary,
    }
    if milestone_step is not None:
        from .hf_ft import hf_gpt2_finetune_milestone_report

        milestone = hf_gpt2_finetune_milestone_report(
            {
                **primary,
                "watches": watches,
                "row_type": "hf_gpt2_finetune_monitor_primary",
            },
            milestone_step=milestone_step,
            label=label,
        )
        snapshot.update(
            {key: value for key, value in milestone.items() if key.startswith("milestone_")}
        )
    return snapshot


def hf_gpt2_finetune_monitor_lines(snapshot: dict[str, Any]) -> list[str]:
    """Render compact lines from an FT monitor report."""

    label = snapshot.get("label") or "monitor"
    lines = [
        (
            "hf_gpt2_ft_monitor "
            f"label={label} "
            f"primary={_number_text(snapshot.get('primary_watch'))} "
            f"process={_number_text(snapshot.get('process_status'))} "
            f"log_step={_number_text(snapshot.get('log_latest_step'))} "
            f"max_steps={_number_text(snapshot.get('log_max_steps'))} "
            f"runtime_max_steps={_number_text(snapshot.get('runtime_max_steps'))} "
            f"runtime_eval_steps={_number_text(snapshot.get('runtime_eval_steps'))} "
            f"runtime_save_steps={_number_text(snapshot.get('runtime_save_steps'))} "
            f"log_remaining_seconds={_number_text(snapshot.get('log_remaining_seconds'))} "
            f"estimated_seconds_until_final={_number_text(snapshot.get('estimated_seconds_until_final'))} "
            f"last_eval_step={_number_text(snapshot.get('last_eval_loss_step'))} "
            f"last_eval_loss={_number_text(snapshot.get('last_eval_loss'))} "
            f"eval_loss_projected_final={_number_text(snapshot.get('eval_loss_projected_final_loss'))} "
            f"next_eval_step={_number_text(snapshot.get('next_eval_step'))} "
            f"latest_due_eval_ready={_number_text(snapshot.get('latest_due_eval_ready'))} "
            f"pending_eval_step={_number_text(snapshot.get('pending_eval_step'))} "
            f"next_checkpoint_step={_number_text(snapshot.get('next_checkpoint_step'))} "
            f"final_ready={_number_text(snapshot.get('final_checkpoint_ready'))} "
            f"latest_checkpoint={_number_text(snapshot.get('latest_checkpoint'))} "
            f"disk_status={_number_text(snapshot.get('disk_status'))} "
            f"disk_margin_gb={_number_text(snapshot.get('disk_margin_gb'))} "
            f"milestone_step={_number_text(snapshot.get('milestone_step'))} "
            f"milestone_status={_number_text(snapshot.get('milestone_status'))} "
            f"milestone_ready={_number_text(snapshot.get('milestone_ready'))} "
            f"milestone_eval_ready={_number_text(snapshot.get('milestone_eval_ready'))} "
            f"milestone_checkpoint_ready={_number_text(snapshot.get('milestone_checkpoint_ready'))} "
            f"wait_status={_number_text(snapshot.get('wait_launch_status'))} "
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
                    f"last_eval_step={_number_text(watch.get('last_eval_loss_step'))} "
                    f"last_eval_loss={_number_text(watch.get('last_eval_loss'))} "
                    f"next_eval_step={_number_text(watch.get('next_eval_step'))} "
                    f"next_checkpoint_step={_number_text(watch.get('next_checkpoint_step'))} "
                    f"final_ready={_number_text(watch.get('final_checkpoint_ready'))} "
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
                f"launch_disk_status={_number_text(wait_launch.get('launch_disk_status'))} "
                f"launch_disk_free_after_gb={_number_text(wait_launch.get('launch_disk_free_after_gb'))} "
                f"launched={_number_text(wait_launch.get('launched'))} "
                f"launched_pid={_number_text(wait_launch.get('launched_pid'))}"
            )
        )
    return lines


def _capture_next_action(snapshot: Mapping[str, Any]) -> str:
    if snapshot.get("milestone_ready") is True:
        return "handoff"
    if snapshot.get("process_status") in {None, "alive"}:
        return "keep_watching"
    return "inspect_run"


def hf_gpt2_finetune_milestone_capture_report(
    monitor_or_status: Any,
    *,
    milestone_step: int | None = None,
    label: str | None = None,
    iteration: int | None = None,
    commands: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the small milestone capture state used by long FT handoffs."""

    if (
        isinstance(monitor_or_status, Mapping)
        and monitor_or_status.get("row_type")
        in {"hf_gpt2_finetune_monitor_report", "hf_gpt2_ft_monitor_snapshot"}
    ):
        snapshot = dict(monitor_or_status)
    else:
        if milestone_step is None:
            if isinstance(monitor_or_status, Mapping):
                milestone_step = _int_value(monitor_or_status.get("milestone_step"))
            if milestone_step is None:
                raise ValueError("milestone_step is required for non-monitor inputs")
        snapshot = hf_gpt2_finetune_monitor_report(
            direct=monitor_or_status,
            milestone_step=milestone_step,
            label=label,
        )
    if milestone_step is None:
        milestone_step = _int_value(snapshot.get("milestone_step"))
    label_value = label or snapshot.get("label")
    next_action = _capture_next_action(snapshot)
    state = {
        "row_type": "hf_gpt2_finetune_milestone_capture",
        "label": label_value,
        "status": snapshot.get("milestone_status"),
        "milestone_ready": snapshot.get("milestone_ready"),
        "milestone_step": milestone_step,
        "milestone_steps_until": snapshot.get("milestone_steps_until"),
        "milestone_eval_ready": snapshot.get("milestone_eval_ready"),
        "milestone_eval_loss": snapshot.get("milestone_eval_loss"),
        "milestone_checkpoint_ready": snapshot.get(
            "milestone_checkpoint_ready"
        ),
        "milestone_checkpoint": snapshot.get("milestone_checkpoint"),
        "process_status": snapshot.get("process_status"),
        "log_latest_step": snapshot.get("log_latest_step"),
        "last_eval_loss_step": snapshot.get("last_eval_loss_step"),
        "last_eval_loss": snapshot.get("last_eval_loss"),
        "next_eval_step": snapshot.get("next_eval_step"),
        "next_checkpoint_step": snapshot.get("next_checkpoint_step"),
        "latest_checkpoint": snapshot.get("latest_checkpoint"),
        "disk_status": snapshot.get("disk_status"),
        "disk_free_gb": snapshot.get("disk_free_gb"),
        "disk_margin_gb": snapshot.get("disk_margin_gb"),
        "wait_launch_status": snapshot.get("wait_launch_status"),
        "wait_launch_launched": snapshot.get("wait_launch_launched"),
        "next_action": next_action,
        "should_continue_watch": next_action == "keep_watching",
        "monitor": snapshot,
    }
    if iteration is not None:
        state["iteration"] = iteration
    if commands is not None:
        state["commands"] = [dict(command) for command in commands]
    return state


def hf_gpt2_finetune_milestone_capture_lines(
    report_or_monitor: Mapping[str, Any],
    *,
    milestone_step: int | None = None,
    label: str | None = None,
) -> list[str]:
    """Render compact lines from an FT milestone capture report."""

    if report_or_monitor.get("row_type") == "hf_gpt2_finetune_milestone_capture":
        state = dict(report_or_monitor)
    else:
        state = hf_gpt2_finetune_milestone_capture_report(
            report_or_monitor,
            milestone_step=milestone_step,
            label=label,
        )
    return [
        (
            "hf_gpt2_ft_milestone_capture "
            f"label={_number_text(state.get('label'))} "
            f"status={_number_text(state.get('status'))} "
            f"milestone_ready={_number_text(state.get('milestone_ready'))} "
            f"milestone_step={_number_text(state.get('milestone_step'))} "
            f"steps_until={_number_text(state.get('milestone_steps_until'))} "
            f"eval_ready={_number_text(state.get('milestone_eval_ready'))} "
            f"eval_loss={_number_text(state.get('milestone_eval_loss'))} "
            f"checkpoint_ready={_number_text(state.get('milestone_checkpoint_ready'))} "
            f"process={_number_text(state.get('process_status'))} "
            f"log_step={_number_text(state.get('log_latest_step'))} "
            f"last_eval_step={_number_text(state.get('last_eval_loss_step'))} "
            f"next_eval_step={_number_text(state.get('next_eval_step'))} "
            f"next_checkpoint_step={_number_text(state.get('next_checkpoint_step'))} "
            f"disk_status={_number_text(state.get('disk_status'))} "
            f"next_action={_number_text(state.get('next_action'))}"
        )
    ]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _checkpoint_from_capture(capture: Mapping[str, Any]) -> str | None:
    checkpoint = capture.get("milestone_checkpoint")
    if isinstance(checkpoint, str) and checkpoint:
        return checkpoint
    step = _int_value(capture.get("milestone_step"))
    if step is not None:
        return f"checkpoint-{step}"
    latest = capture.get("latest_checkpoint")
    if isinstance(latest, str) and latest:
        return latest
    return None


def hf_gpt2_finetune_milestone_handoff_report(
    capture_or_monitor: Mapping[str, Any],
    *,
    run_dir: str | Path | None = None,
    checkpoint: str | None = None,
    label_prefix: str | None = None,
    python: str = "python3",
    script: str | Path = "bindings/st-py/examples/hf_gpt2_ft_checkpoint_generation_control.py",
    compare_with_sweep: Sequence[str | Path] | str | Path | None = None,
    compare_with_label: Sequence[str] | str | None = None,
    curve_out: str | Path | None = None,
    curve_lines_out: str | Path | None = None,
    trainer_trace_jsonl: str | Path | None = None,
    run_card: str | Path | None = None,
    top_n: int = 3,
    wait: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Plan the checkpoint-generation handoff after a milestone capture is ready."""

    if capture_or_monitor.get("row_type") == "hf_gpt2_finetune_milestone_capture":
        capture = dict(capture_or_monitor)
    else:
        capture = hf_gpt2_finetune_milestone_capture_report(capture_or_monitor)
    resolved_checkpoint = checkpoint or _checkpoint_from_capture(capture)
    resolved_run_dir = Path(run_dir) if run_dir is not None else None
    checkpoint_path = (
        None
        if resolved_run_dir is None or resolved_checkpoint is None
        else resolved_run_dir / resolved_checkpoint
    )
    milestone_ready = capture.get("milestone_ready") is True
    checkpoint_ready = capture.get("milestone_checkpoint_ready") is True
    if checkpoint_path is not None:
        checkpoint_ready = checkpoint_ready or (checkpoint_path / "model.safetensors").is_file()
    status = (
        "ready"
        if milestone_ready and checkpoint_ready and resolved_checkpoint is not None
        else "waiting_for_milestone"
        if not milestone_ready
        else "waiting_for_checkpoint"
        if not checkpoint_ready
        else "missing_checkpoint"
    )
    prefix = label_prefix or str(capture.get("label") or "").strip() or None
    compare_paths = [str(path) for path in _as_list(compare_with_sweep)]
    compare_labels = [str(label) for label in _as_list(compare_with_label)]
    resolved_curve_out = (
        Path(curve_out)
        if curve_out is not None
        else resolved_run_dir / f"{resolved_checkpoint}-generation-curve.json"
        if resolved_run_dir is not None and resolved_checkpoint is not None
        else None
    )
    resolved_curve_lines_out = (
        Path(curve_lines_out)
        if curve_lines_out is not None
        else resolved_run_dir / f"{resolved_checkpoint}-generation-curve.txt"
        if resolved_run_dir is not None and resolved_checkpoint is not None
        else None
    )
    resolved_trainer_trace = (
        Path(trainer_trace_jsonl) if trainer_trace_jsonl is not None else None
    )
    if resolved_trainer_trace is None and resolved_run_dir is not None:
        default_trace = resolved_run_dir / "spiraltorch-hf-gpt2-ft-trainer-trace.jsonl"
        if default_trace.is_file():
            resolved_trainer_trace = default_trace
    resolved_curve_run_card = Path(run_card) if run_card is not None else None
    if resolved_curve_run_card is None and resolved_run_dir is not None:
        default_run_card = resolved_run_dir / "spiraltorch-hf-gpt2-ft-run-card.json"
        if default_run_card.is_file():
            resolved_curve_run_card = default_run_card
    package_kwargs: dict[str, Any] = {
        "compare_with_sweep": compare_paths,
        "compare_with_label": compare_labels,
        "top_n": top_n,
        "wait": wait,
        "dry_run": dry_run,
    }
    if resolved_run_dir is not None:
        package_kwargs["run_dir"] = str(resolved_run_dir)
    if resolved_checkpoint is not None:
        package_kwargs["checkpoint"] = resolved_checkpoint
    if prefix:
        package_kwargs["label_prefix"] = prefix
    if resolved_curve_out is not None:
        package_kwargs["curve_out"] = str(resolved_curve_out)
    if resolved_curve_lines_out is not None:
        package_kwargs["curve_lines_out"] = str(resolved_curve_lines_out)
    if resolved_trainer_trace is not None:
        package_kwargs["curve_trainer_trace_jsonl"] = str(resolved_trainer_trace)
    if resolved_curve_run_card is not None:
        package_kwargs["curve_run_card"] = str(resolved_curve_run_card)
    command: list[str] = [str(python), str(script)]
    if resolved_run_dir is not None:
        command.extend(["--run-dir", str(resolved_run_dir)])
    if resolved_checkpoint is not None:
        command.extend(["--checkpoint", resolved_checkpoint])
    if prefix:
        command.extend(["--label-prefix", prefix])
    for path in compare_paths:
        command.extend(["--compare-with-sweep", path])
    for label in compare_labels:
        command.extend(["--compare-with-label", label])
    if resolved_curve_out is not None:
        command.extend(["--curve-out", str(resolved_curve_out)])
    if resolved_curve_lines_out is not None:
        command.extend(["--curve-lines-out", str(resolved_curve_lines_out)])
    if resolved_trainer_trace is not None:
        command.extend(["--curve-trainer-trace-jsonl", str(resolved_trainer_trace)])
    if resolved_curve_run_card is not None:
        command.extend(["--curve-run-card", str(resolved_curve_run_card)])
    command.extend(["--top-n", str(top_n)])
    if wait:
        command.append("--wait")
    if dry_run:
        command.append("--dry-run")
    return {
        "row_type": "hf_gpt2_finetune_milestone_handoff",
        "status": status,
        "ready": status == "ready",
        "action": "checkpoint_generation_control",
        "label": capture.get("label"),
        "label_prefix": prefix,
        "milestone_step": capture.get("milestone_step"),
        "milestone_ready": capture.get("milestone_ready"),
        "milestone_eval_loss": capture.get("milestone_eval_loss"),
        "milestone_checkpoint_ready": capture.get("milestone_checkpoint_ready"),
        "checkpoint": resolved_checkpoint,
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "run_dir": None if resolved_run_dir is None else str(resolved_run_dir),
        "compare_with_sweep": compare_paths,
        "compare_with_label": compare_labels,
        "package_function": "spiraltorch.zspace_checkpoint_generation_control_report",
        "package_kwargs": package_kwargs,
        "command": command,
        "command_display": shlex.join(command),
        "capture": capture,
    }


def hf_gpt2_finetune_milestone_handoff_lines(
    report_or_capture: Mapping[str, Any],
    **kwargs: Any,
) -> list[str]:
    """Render compact lines from a milestone handoff plan."""

    if report_or_capture.get("row_type") == "hf_gpt2_finetune_milestone_handoff":
        report = dict(report_or_capture)
    else:
        report = hf_gpt2_finetune_milestone_handoff_report(
            report_or_capture,
            **kwargs,
        )
    return [
        (
            "hf_gpt2_ft_milestone_handoff "
            f"status={_number_text(report.get('status'))} "
            f"ready={_number_text(report.get('ready'))} "
            f"action={_number_text(report.get('action'))} "
            f"label={_number_text(report.get('label'))} "
            f"step={_number_text(report.get('milestone_step'))} "
            f"eval_loss={_number_text(report.get('milestone_eval_loss'))} "
            f"checkpoint={_number_text(report.get('checkpoint'))} "
            f"checkpoint_ready={_number_text(report.get('milestone_checkpoint_ready'))} "
            f"run_dir={_number_text(report.get('run_dir'))} "
            f"compare_count={len(report.get('compare_with_sweep') or [])} "
            f"command={_number_text(report.get('command_display'))}"
        )
    ]


HandoffRunner = Callable[..., subprocess.CompletedProcess[str] | None]
HandoffPackageRunner = Callable[..., Mapping[str, Any]]


def _handoff_report_from_value(
    report_or_capture: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    if report_or_capture.get("row_type") == "hf_gpt2_finetune_milestone_handoff":
        if not kwargs:
            return dict(report_or_capture)
        capture = report_or_capture.get("capture")
        if isinstance(capture, Mapping):
            return hf_gpt2_finetune_milestone_handoff_report(capture, **kwargs)
        raise ValueError(
            "handoff kwargs require a handoff report with an embedded capture"
        )
    return hf_gpt2_finetune_milestone_handoff_report(report_or_capture, **kwargs)


def _merged_env(env: Mapping[str, Any] | None) -> dict[str, str] | None:
    if env is None:
        return None
    merged = os.environ.copy()
    merged.update({str(key): str(value) for key, value in env.items()})
    return merged


def _bounded_text(value: Any, *, max_chars: int | None) -> tuple[str | None, bool]:
    if value is None:
        return None, False
    text = value if isinstance(value, str) else str(value)
    if max_chars is None or max_chars < 0 or len(text) <= max_chars:
        return text, False
    return text[-max_chars:], True


def _json_from_text(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(value[start : end + 1])
    except json.JSONDecodeError:
        return None


def hf_gpt2_finetune_milestone_handoff_execution_report(
    report_or_capture: Mapping[str, Any],
    *,
    run: bool = False,
    use_package_api: bool = False,
    cwd: str | Path | None = None,
    env: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    capture_output: bool = True,
    check: bool = False,
    runner: HandoffRunner | None = None,
    package_runner: HandoffPackageRunner | None = None,
    out: str | Path | None = None,
    lines_out: str | Path | None = None,
    max_output_chars: int | None = 20_000,
    **handoff_kwargs: Any,
) -> dict[str, Any]:
    """Plan or execute a milestone handoff command through the package API.

    By default this is deliberately non-invasive: it records the command that
    would run. Pass ``run=True`` to execute the handoff command and capture a
    small auditable process report.
    """

    handoff = _handoff_report_from_value(report_or_capture, **handoff_kwargs)
    command = [str(part) for part in handoff.get("command") or []]
    package_kwargs = dict(handoff.get("package_kwargs") or {})
    package_function = handoff.get("package_function")
    execution_backend = "package_api" if use_package_api else "command"
    started_unix_s = time.time()
    report: dict[str, Any] = {
        "row_type": "hf_gpt2_finetune_milestone_handoff_execution",
        "status": (
            "planned" if not run and handoff.get("ready") else handoff.get("status")
        ),
        "run": bool(run),
        "ready": handoff.get("ready"),
        "execution_backend": execution_backend,
        "handoff_status": handoff.get("status"),
        "action": handoff.get("action"),
        "label": handoff.get("label"),
        "milestone_step": handoff.get("milestone_step"),
        "milestone_eval_loss": handoff.get("milestone_eval_loss"),
        "checkpoint": handoff.get("checkpoint"),
        "checkpoint_path": handoff.get("checkpoint_path"),
        "run_dir": handoff.get("run_dir"),
        "cwd": None if cwd is None else str(cwd),
        "command": command,
        "command_display": shlex.join(command),
        "package_function": package_function,
        "package_kwargs": package_kwargs,
        "started_unix_s": started_unix_s,
        "handoff": handoff,
    }
    if use_package_api and not package_kwargs:
        report["status"] = "missing_package_plan"
        report["error"] = "handoff report did not include package kwargs"
    elif use_package_api and (
        "run_dir" not in package_kwargs or "checkpoint" not in package_kwargs
    ):
        report["status"] = "missing_package_plan"
        report["error"] = "package kwargs require run_dir and checkpoint"
    elif not use_package_api and not command:
        report["status"] = "missing_command"
        report["error"] = "handoff report did not include a command"
    elif not run:
        report["completed_unix_s"] = time.time()
    elif handoff.get("ready") is not True:
        report["status"] = "not_ready"
        report["error"] = f"handoff is not ready: {handoff.get('status')}"
        report["completed_unix_s"] = time.time()
    elif use_package_api:
        try:
            if package_runner is None:
                from .hf_generation import zspace_checkpoint_generation_control_report

                command_report = zspace_checkpoint_generation_control_report(
                    **package_kwargs
                )
            else:
                command_report = dict(package_runner(**package_kwargs))
        except Exception as exc:  # pragma: no cover - exercised by runtime callers
            report["status"] = "error"
            report["error"] = f"{exc.__class__.__name__}: {exc}"
            report["completed_unix_s"] = time.time()
        else:
            report.update(
                {
                    "status": "complete",
                    "returncode": 0,
                    "command_report": command_report,
                    "completed_unix_s": time.time(),
                }
            )
    else:
        process_cwd = None if cwd is None else str(cwd)
        process_env = _merged_env(env)
        process_runner = runner or subprocess.run
        try:
            result = process_runner(
                command,
                cwd=process_cwd,
                env=process_env,
                text=True,
                capture_output=capture_output,
                check=check,
                timeout=timeout,
            )
        except Exception as exc:  # pragma: no cover - exercised by runtime callers
            report["status"] = "error"
            report["error"] = f"{exc.__class__.__name__}: {exc}"
            report["completed_unix_s"] = time.time()
        else:
            returncode = getattr(result, "returncode", 0)
            stdout, stdout_truncated = _bounded_text(
                getattr(result, "stdout", None),
                max_chars=max_output_chars,
            )
            stderr, stderr_truncated = _bounded_text(
                getattr(result, "stderr", None),
                max_chars=max_output_chars,
            )
            command_report = _json_from_text(stdout)
            report.update(
                {
                    "status": "complete" if returncode == 0 else "failed",
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                    "command_report": command_report,
                    "completed_unix_s": time.time(),
                }
            )
    report["duration_seconds"] = (
        float(report["completed_unix_s"]) - started_unix_s
        if isinstance(report.get("completed_unix_s"), (int, float))
        else None
    )
    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report["out"] = str(out_path)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if lines_out is not None:
        lines_path = Path(lines_out)
        lines_path.parent.mkdir(parents=True, exist_ok=True)
        report["lines_out"] = str(lines_path)
        lines = hf_gpt2_finetune_milestone_handoff_execution_lines(report)
        lines_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def hf_gpt2_finetune_milestone_handoff_execution_lines(
    report_or_handoff: Mapping[str, Any],
    **kwargs: Any,
) -> list[str]:
    """Render compact lines from a milestone handoff execution report."""

    if (
        report_or_handoff.get("row_type")
        == "hf_gpt2_finetune_milestone_handoff_execution"
    ):
        report = dict(report_or_handoff)
    else:
        report = hf_gpt2_finetune_milestone_handoff_execution_report(
            report_or_handoff,
            **kwargs,
        )
    command_report = report.get("command_report")
    command_report_status = (
        command_report.get("status") if isinstance(command_report, Mapping) else None
    )
    return [
        (
            "hf_gpt2_ft_milestone_handoff_execution "
            f"status={_number_text(report.get('status'))} "
            f"run={_number_text(report.get('run'))} "
            f"backend={_number_text(report.get('execution_backend'))} "
            f"ready={_number_text(report.get('ready'))} "
            f"handoff_status={_number_text(report.get('handoff_status'))} "
            f"action={_number_text(report.get('action'))} "
            f"step={_number_text(report.get('milestone_step'))} "
            f"eval_loss={_number_text(report.get('milestone_eval_loss'))} "
            f"checkpoint={_number_text(report.get('checkpoint'))} "
            f"returncode={_number_text(report.get('returncode'))} "
            f"command_report_status={_number_text(command_report_status)} "
            f"duration_seconds={_number_text(report.get('duration_seconds'))} "
            f"command={_number_text(report.get('command_display'))}"
        )
    ]


def hf_gpt2_finetune_milestone_runtime_report(
    *,
    direct: Any = None,
    eval_watch: Any = None,
    checkpoint_watch: Any = None,
    final_watch: Any = None,
    wait_launch: Any = None,
    milestone_step: int | None = None,
    label: str | None = None,
    run_dir: str | Path | None = None,
    next_run_dir: str | Path | None = None,
    capture_iteration: int | None = None,
    capture_commands: Sequence[Mapping[str, Any]] | None = None,
    checkpoint: str | None = None,
    label_prefix: str | None = None,
    python: str = "python3",
    script: str | Path = "bindings/st-py/examples/hf_gpt2_ft_checkpoint_generation_control.py",
    compare_with_sweep: Sequence[str | Path] | str | Path | None = None,
    compare_with_label: Sequence[str] | str | None = None,
    curve_out: str | Path | None = None,
    curve_lines_out: str | Path | None = None,
    trainer_trace_jsonl: str | Path | None = None,
    run_card: str | Path | None = None,
    top_n: int = 3,
    wait: bool = False,
    dry_run: bool = True,
    execute: bool = False,
    use_package_api: bool = True,
    cwd: str | Path | None = None,
    env: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    capture_output: bool = True,
    check: bool = False,
    runner: HandoffRunner | None = None,
    package_runner: HandoffPackageRunner | None = None,
    out: str | Path | None = None,
    lines_out: str | Path | None = None,
    max_output_chars: int | None = 20_000,
) -> dict[str, Any]:
    """Build one importable FT milestone monitor/capture/handoff runtime report."""

    monitor = hf_gpt2_finetune_monitor_report(
        direct=direct,
        eval_watch=eval_watch,
        checkpoint_watch=checkpoint_watch,
        final_watch=final_watch,
        wait_launch=wait_launch,
        milestone_step=milestone_step,
        label=label,
        run_dir=run_dir,
        next_run_dir=next_run_dir,
    )
    capture = hf_gpt2_finetune_milestone_capture_report(
        monitor,
        iteration=capture_iteration,
        commands=capture_commands,
    )
    resolved_run_dir = run_dir if run_dir is not None else capture.get("monitor", {}).get("run_dir")
    handoff = hf_gpt2_finetune_milestone_handoff_report(
        capture,
        run_dir=resolved_run_dir,
        checkpoint=checkpoint,
        label_prefix=label_prefix,
        python=python,
        script=script,
        compare_with_sweep=compare_with_sweep,
        compare_with_label=compare_with_label,
        curve_out=curve_out,
        curve_lines_out=curve_lines_out,
        trainer_trace_jsonl=trainer_trace_jsonl,
        run_card=run_card,
        top_n=top_n,
        wait=wait,
        dry_run=dry_run,
    )
    execution = hf_gpt2_finetune_milestone_handoff_execution_report(
        handoff,
        run=execute,
        use_package_api=use_package_api,
        cwd=cwd,
        env=env,
        timeout=timeout,
        capture_output=capture_output,
        check=check,
        runner=runner,
        package_runner=package_runner,
        max_output_chars=max_output_chars,
    )
    execution_status = execution.get("status")
    status = (
        "executed"
        if execute and execution_status == "complete"
        else "ready"
        if handoff.get("ready") is True
        else "watching"
        if capture.get("should_continue_watch") is True
        else "inspect_run"
    )
    report: dict[str, Any] = {
        "row_type": "hf_gpt2_finetune_milestone_runtime",
        "status": status,
        "label": label or capture.get("label"),
        "milestone_step": capture.get("milestone_step"),
        "milestone_ready": capture.get("milestone_ready"),
        "next_action": capture.get("next_action"),
        "execute": bool(execute),
        "execution_backend": execution.get("execution_backend"),
        "execution_status": execution_status,
        "handoff_status": handoff.get("status"),
        "checkpoint": handoff.get("checkpoint"),
        "run_dir": handoff.get("run_dir") or monitor.get("run_dir"),
        "monitor": monitor,
        "capture": capture,
        "handoff": handoff,
        "execution": execution,
    }
    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report["out"] = str(out_path)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if lines_out is not None:
        lines_path = Path(lines_out)
        lines_path.parent.mkdir(parents=True, exist_ok=True)
        report["lines_out"] = str(lines_path)
        lines_path.write_text(
            "\n".join(hf_gpt2_finetune_milestone_runtime_lines(report)) + "\n",
            encoding="utf-8",
        )
    return report


def hf_gpt2_finetune_milestone_runtime_lines(
    report: Mapping[str, Any],
) -> list[str]:
    """Render compact lines from a milestone runtime report."""

    lines = [
        (
            "hf_gpt2_ft_milestone_runtime "
            f"status={_number_text(report.get('status'))} "
            f"label={_number_text(report.get('label'))} "
            f"step={_number_text(report.get('milestone_step'))} "
            f"milestone_ready={_number_text(report.get('milestone_ready'))} "
            f"next_action={_number_text(report.get('next_action'))} "
            f"handoff_status={_number_text(report.get('handoff_status'))} "
            f"execute={_number_text(report.get('execute'))} "
            f"execution_backend={_number_text(report.get('execution_backend'))} "
            f"execution_status={_number_text(report.get('execution_status'))} "
            f"checkpoint={_number_text(report.get('checkpoint'))} "
            f"run_dir={_number_text(report.get('run_dir'))}"
        )
    ]
    monitor = report.get("monitor")
    if isinstance(monitor, dict):
        lines.extend(hf_gpt2_finetune_monitor_lines(monitor)[:1])
    capture = report.get("capture")
    if isinstance(capture, Mapping):
        lines.extend(hf_gpt2_finetune_milestone_capture_lines(capture))
    handoff = report.get("handoff")
    if isinstance(handoff, Mapping):
        lines.extend(hf_gpt2_finetune_milestone_handoff_lines(handoff))
    execution = report.get("execution")
    if isinstance(execution, Mapping):
        lines.extend(hf_gpt2_finetune_milestone_handoff_execution_lines(execution))
    return lines


def _latest_runtime_path(root: str | Path | None, patterns: Sequence[str]) -> Path | None:
    if root is None:
        return None
    root_path = Path(root)
    if not root_path.is_dir():
        return None
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(path for path in root_path.glob(pattern) if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))


def hf_gpt2_finetune_milestone_runtime_sources(
    run_dir: str | Path,
    *,
    next_run_dir: str | Path | None = None,
    direct: str | Path | None = None,
    eval_watch: str | Path | None = None,
    checkpoint_watch: str | Path | None = None,
    final_watch: str | Path | None = None,
    wait_launch: str | Path | None = None,
) -> dict[str, str | None]:
    """Resolve standard status/watch artifacts for one FT run directory."""

    run_root = Path(run_dir)
    next_root = Path(next_run_dir) if next_run_dir is not None else None
    sources = {
        "direct": Path(direct)
        if direct is not None
        else _latest_runtime_path(
            run_root,
            [
                "direct-run-status-history.jsonl",
                "*run-status-history.jsonl",
                "run-status-history.jsonl",
                "direct-run-status.json",
                "*run-status.json",
                "run-status.json",
                "status.json",
            ],
        ),
        "eval": Path(eval_watch)
        if eval_watch is not None
        else _latest_runtime_path(
            run_root,
            ["watch-*-eval*-history.jsonl", "*eval*-history.jsonl"],
        ),
        "checkpoint": Path(checkpoint_watch)
        if checkpoint_watch is not None
        else _latest_runtime_path(
            run_root,
            ["watch-*-checkpoint*-history.jsonl", "*checkpoint*-history.jsonl"],
        ),
        "final": Path(final_watch)
        if final_watch is not None
        else _latest_runtime_path(
            run_root,
            ["watch-*-final*-history.jsonl", "*final*-history.jsonl"],
        ),
        "wait_launch": Path(wait_launch)
        if wait_launch is not None
        else (
            _latest_runtime_path(
                next_root,
                ["*-wait-launch-history.jsonl", "*wait*launch*history.jsonl"],
            )
            or _latest_runtime_path(
                run_root,
                ["*-wait-launch-history.jsonl", "*wait*launch*history.jsonl"],
            )
        ),
    }
    return {
        name: str(path) if isinstance(path, Path) else None
        for name, path in sources.items()
    }


def _milestone_step_from_filename(path: Path) -> int | None:
    parts = path.name.split("-")
    for index, part in enumerate(parts):
        if part == "milestone" and index + 1 < len(parts):
            return _int_value(parts[index + 1])
    return None


def _milestone_step_from_capture_file(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    for key in ("milestone_step", "step"):
        value = _int_value(payload.get(key))
        if value is not None:
            return value
    return _milestone_step_from_filename(path)


def _latest_milestone_capture_step(run_dir: str | Path) -> tuple[int | None, str | None]:
    root = Path(run_dir)
    if not root.is_dir():
        return None, None
    candidates = sorted(root.glob("milestone-*-capture.json"))
    if not candidates:
        candidates = sorted(root.glob("milestone-*"))
    ranked: list[tuple[float, str, int, Path]] = []
    for path in candidates:
        if not path.is_file():
            continue
        step = (
            _milestone_step_from_capture_file(path)
            if path.suffix == ".json"
            else _milestone_step_from_filename(path)
        )
        if step is None:
            continue
        ranked.append((path.stat().st_mtime, str(path), step, path))
    if not ranked:
        return None, None
    _, _, step, path = max(ranked)
    return step, str(path)


def _latest_eval_step_from_source(source: Any) -> int | None:
    try:
        rows, _ = _coerce_status_rows(source)
    except Exception:
        return None
    if not rows:
        return None
    step = _last_eval_loss_step(rows[-1])
    return _int_value(step)


def _infer_runtime_milestone_step(
    run_dir: str | Path,
    *,
    explicit_step: int | None,
    direct: Any = None,
) -> tuple[int | None, str | None]:
    if explicit_step is not None:
        return explicit_step, "explicit"
    capture_step, capture_path = _latest_milestone_capture_step(run_dir)
    if capture_step is not None:
        return capture_step, capture_path
    eval_step = _latest_eval_step_from_source(direct)
    if eval_step is not None:
        return eval_step, "direct_eval_history"
    return None, None


def _runtime_milestone_step_token(value: Any) -> str:
    parsed = _int_value(value)
    if parsed is not None:
        return str(parsed)
    if isinstance(value, str) and value.strip():
        token = "".join(
            ch if ch.isalnum() or ch in {"-", "_", "."} else "-"
            for ch in value.strip()
        )
        return token.strip("-") or "latest"
    return "latest"


def hf_gpt2_finetune_milestone_runtime_artifact_paths(
    run_dir: str | Path,
    *,
    milestone_step: int | str | None = None,
    report: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Return standard JSON/TXT artifact paths for a milestone runtime report."""

    step = milestone_step
    if step is None and report is not None:
        step = report.get("milestone_step")
    token = _runtime_milestone_step_token(step)
    root = Path(run_dir)
    return {
        "out": str(root / f"milestone-{token}-runtime.json"),
        "lines_out": str(root / f"milestone-{token}-runtime.txt"),
    }


def write_hf_gpt2_finetune_milestone_runtime_report(
    report: Mapping[str, Any],
    *,
    run_dir: str | Path | None = None,
    out: str | Path | None = None,
    lines_out: str | Path | None = None,
) -> dict[str, Any]:
    """Write a milestone runtime report using explicit or standard artifact paths."""

    archived = dict(report)
    root = Path(run_dir or archived.get("run_dir") or ".")
    defaults = hf_gpt2_finetune_milestone_runtime_artifact_paths(
        root,
        report=archived,
    )
    out_path = Path(out or defaults["out"])
    lines_path = Path(lines_out or defaults["lines_out"])
    archived["out"] = str(out_path)
    archived["lines_out"] = str(lines_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(archived, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines_path.write_text(
        "\n".join(hf_gpt2_finetune_milestone_runtime_lines(archived)) + "\n",
        encoding="utf-8",
    )
    return archived


def hf_gpt2_finetune_milestone_runtime_from_run_dir_report(
    run_dir: str | Path,
    *,
    next_run_dir: str | Path | None = None,
    direct: Any = None,
    eval_watch: Any = None,
    checkpoint_watch: Any = None,
    final_watch: Any = None,
    wait_launch: Any = None,
    out: str | Path | None = None,
    lines_out: str | Path | None = None,
    **runtime_kwargs: Any,
) -> dict[str, Any]:
    """Build a milestone runtime report by resolving standard artifacts from a run dir."""

    sources = hf_gpt2_finetune_milestone_runtime_sources(
        run_dir,
        next_run_dir=next_run_dir,
        direct=direct if isinstance(direct, (str, Path)) else None,
        eval_watch=eval_watch if isinstance(eval_watch, (str, Path)) else None,
        checkpoint_watch=checkpoint_watch
        if isinstance(checkpoint_watch, (str, Path))
        else None,
        final_watch=final_watch if isinstance(final_watch, (str, Path)) else None,
        wait_launch=wait_launch if isinstance(wait_launch, (str, Path)) else None,
    )
    milestone_step = _int_value(runtime_kwargs.get("milestone_step"))
    inferred_step, inferred_source = _infer_runtime_milestone_step(
        run_dir,
        explicit_step=milestone_step,
        direct=direct if direct is not None else sources["direct"],
    )
    runtime_kwargs = dict(runtime_kwargs)
    if inferred_step is not None:
        runtime_kwargs["milestone_step"] = inferred_step
    report = hf_gpt2_finetune_milestone_runtime_report(
        direct=direct if direct is not None else sources["direct"],
        eval_watch=eval_watch if eval_watch is not None else sources["eval"],
        checkpoint_watch=(
            checkpoint_watch if checkpoint_watch is not None else sources["checkpoint"]
        ),
        final_watch=final_watch if final_watch is not None else sources["final"],
        wait_launch=wait_launch if wait_launch is not None else sources["wait_launch"],
        run_dir=run_dir,
        next_run_dir=next_run_dir,
        **runtime_kwargs,
    )
    report["sources"] = sources
    report["source_count"] = sum(1 for value in sources.values() if value is not None)
    report["milestone_step_source"] = inferred_source
    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report["out"] = str(out_path)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if lines_out is not None:
        lines_path = Path(lines_out)
        lines_path.parent.mkdir(parents=True, exist_ok=True)
        report["lines_out"] = str(lines_path)
        lines_path.write_text(
            "\n".join(hf_gpt2_finetune_milestone_runtime_lines(report)) + "\n",
            encoding="utf-8",
        )
    return report


def hf_gpt2_finetune_milestone_runtime_from_run_dir_archive(
    run_dir: str | Path,
    *,
    next_run_dir: str | Path | None = None,
    direct: Any = None,
    eval_watch: Any = None,
    checkpoint_watch: Any = None,
    final_watch: Any = None,
    wait_launch: Any = None,
    out: str | Path | None = None,
    lines_out: str | Path | None = None,
    **runtime_kwargs: Any,
) -> dict[str, Any]:
    """Build and write a run-dir milestone runtime report with standard paths."""

    report = hf_gpt2_finetune_milestone_runtime_from_run_dir_report(
        run_dir,
        next_run_dir=next_run_dir,
        direct=direct,
        eval_watch=eval_watch,
        checkpoint_watch=checkpoint_watch,
        final_watch=final_watch,
        wait_launch=wait_launch,
        **runtime_kwargs,
    )
    return write_hf_gpt2_finetune_milestone_runtime_report(
        report,
        run_dir=run_dir,
        out=out,
        lines_out=lines_out,
    )


def summarize_hf_gpt2_finetune_status_history(
    rows: list[dict[str, Any]], *, label: str | None = None, history_jsonl: str | Path
) -> dict[str, Any]:
    first = rows[0] if rows else {}
    last = rows[-1] if rows else {}
    first_log = _first_log_row(rows)
    last_log = _last_log_row(rows)
    first_log_step = _nested(first_log, "log_progress", "log_latest_step")
    last_log_step = _nested(last_log, "log_progress", "log_latest_step")
    first_trace_step = _nested(first, "trace", "trace_max_global_step")
    last_trace_step = _nested(last, "trace", "trace_max_global_step")
    first_time = first.get("time_unix_s")
    last_time = last.get("time_unix_s")
    first_log_time = first_log.get("time_unix_s")
    last_log_time = last_log.get("time_unix_s")
    duration_seconds = (
        float(last_time) - float(first_time)
        if isinstance(first_time, (int, float)) and isinstance(last_time, (int, float))
        else None
    )
    log_duration_seconds = (
        float(last_log_time) - float(first_log_time)
        if isinstance(first_log_time, (int, float))
        and isinstance(last_log_time, (int, float))
        else None
    )
    delta_log_step = (
        int(last_log_step) - int(first_log_step)
        if isinstance(first_log_step, int) and isinstance(last_log_step, int)
        else None
    )
    delta_trace_step = (
        int(last_trace_step) - int(first_trace_step)
        if isinstance(first_trace_step, int) and isinstance(last_trace_step, int)
        else None
    )
    log_steps_per_second = (
        float(delta_log_step) / log_duration_seconds
        if isinstance(delta_log_step, int)
        and log_duration_seconds is not None
        and log_duration_seconds > 0.0
        else None
    )
    log_steps_until_next_eval = _nested(
        last, "eval_progress", "log_steps_until_next_eval"
    )
    log_steps_until_next_checkpoint = _nested(
        last, "checkpoint_progress", "log_steps_until_next_checkpoint"
    )
    last_log_max_steps = _nested(last, "log_progress", "log_max_steps")
    log_steps_until_final = (
        max(int(last_log_max_steps) - int(last_log_step), 0)
        if isinstance(last_log_max_steps, int) and isinstance(last_log_step, int)
        else None
    )
    estimated_seconds_until_next_eval = (
        float(log_steps_until_next_eval) / log_steps_per_second
        if isinstance(log_steps_until_next_eval, int)
        and log_steps_per_second is not None
        and log_steps_per_second > 0.0
        else None
    )
    estimated_seconds_until_next_checkpoint = (
        float(log_steps_until_next_checkpoint) / log_steps_per_second
        if isinstance(log_steps_until_next_checkpoint, int)
        and log_steps_per_second is not None
        and log_steps_per_second > 0.0
        else None
    )
    estimated_seconds_until_final = (
        float(log_steps_until_final) / log_steps_per_second
        if isinstance(log_steps_until_final, int)
        and log_steps_per_second is not None
        and log_steps_per_second > 0.0
        else None
    )
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
    disk_values = [
        row.get("disk_free_gb")
        for row in rows
        if isinstance(row.get("disk_free_gb"), (int, float))
    ]
    disk_margin_values = [
        row.get("disk_margin_gb")
        for row in rows
        if isinstance(row.get("disk_margin_gb"), (int, float))
    ]
    checkpoint_headroom = _checkpoint_headroom(last)
    return {
        "row_type": "hf_gpt2_ft_status_history_summary",
        "label": label,
        "history_jsonl": str(history_jsonl),
        "row_count": len(rows),
        "first_time_unix_s": first_time,
        "last_time_unix_s": last_time,
        "duration_seconds": duration_seconds,
        "log_duration_seconds": log_duration_seconds,
        "first_log_step": first_log_step,
        "last_log_step": last_log_step,
        "delta_log_step": delta_log_step,
        "log_steps_per_second": log_steps_per_second,
        "first_trace_step": first_trace_step,
        "last_trace_step": last_trace_step,
        "delta_trace_step": delta_trace_step,
        "last_runtime_max_steps": _runtime_setting(last, "max_steps"),
        "last_runtime_eval_steps": _runtime_setting(last, "eval_steps"),
        "last_runtime_save_steps": _runtime_setting(last, "save_steps"),
        "last_runtime_save_total_limit": _runtime_setting(last, "save_total_limit"),
        "last_runtime_min_free_disk_gb": _runtime_setting(last, "min_free_disk_gb"),
        "last_runtime_process_command_available": _runtime_setting(
            last, "process_command_available"
        ),
        "last_log_max_steps": last_log_max_steps,
        "last_log_remaining_seconds": _nested(
            last, "log_progress", "log_remaining_seconds"
        ),
        "last_log_steps_until_final": log_steps_until_final,
        "estimated_seconds_until_final": estimated_seconds_until_final,
        "last_next_eval_step": _nested(last, "eval_progress", "next_eval_step"),
        "last_log_steps_until_next_eval": log_steps_until_next_eval,
        "estimated_seconds_until_next_eval": estimated_seconds_until_next_eval,
        "last_latest_due_eval_step": _nested(
            last, "eval_progress", "latest_due_eval_step"
        ),
        "last_latest_due_eval_ready": _nested(
            last, "eval_progress", "latest_due_eval_ready"
        ),
        "last_pending_eval_step": _nested(last, "eval_progress", "pending_eval_step"),
        "last_log_steps_since_pending_eval": _nested(
            last, "eval_progress", "log_steps_since_pending_eval"
        ),
        "last_next_checkpoint_step": _nested(
            last, "checkpoint_progress", "next_checkpoint_step"
        ),
        "last_log_steps_until_next_checkpoint": log_steps_until_next_checkpoint,
        "estimated_seconds_until_next_checkpoint": (
            estimated_seconds_until_next_checkpoint
        ),
        "last_best_eval_loss_step": _nested(last, "trace", "trace_best_eval_loss_step"),
        "last_eval_loss_step": _last_eval_loss_step(last),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "loss_delta": (losses[-1] - losses[0]) if len(losses) >= 2 else None,
        "last_eval_loss": _nested(last, "trace", "trace_last_eval_loss"),
        "min_eval_loss": min(eval_losses) if eval_losses else None,
        "last_guard_count": _nested(last, "trace", "training_loss_guard_count"),
        "last_process_status": last.get("process_status"),
        "last_final_checkpoint_ready": last.get("final_checkpoint_ready"),
        "last_checkpoint_count": last.get("checkpoint_count"),
        "last_latest_checkpoint": _latest_checkpoint_name(last),
        "last_save_total_limit": last.get("save_total_limit"),
        "last_checkpoint_headroom_checkpoint_gb": checkpoint_headroom.get(
            "resume_checkpoint_gb"
        ),
        "last_checkpoint_headroom_peak_gb": checkpoint_headroom.get(
            "estimated_peak_checkpoint_gb"
        ),
        "last_checkpoint_headroom_free_after_gb": checkpoint_headroom.get(
            "free_after_estimated_peak_gb"
        ),
        "min_disk_free_gb": min(disk_values) if disk_values else None,
        "last_disk_free_gb": last.get("disk_free_gb"),
        "min_disk_margin_gb": min(disk_margin_values) if disk_margin_values else None,
        "last_disk_margin_gb": last.get("disk_margin_gb"),
        "last_disk_status": last.get("disk_status"),
        "last_watch_stop_eval_step": last.get("watch_stop_eval_step"),
        "last_watch_stop_eval_ready": last.get("watch_stop_eval_ready"),
        "last_watch_stop_reason": last.get("watch_stop_reason"),
    }


def hf_gpt2_finetune_status_history_lines(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    tail: int,
    tail_evals: int = 0,
) -> list[str]:
    label_text = summary.get("label") or "history"
    lines = [
        (
            "hf_gpt2_ft_status_history "
            f"label={label_text} "
            f"rows={_number_text(summary.get('row_count'))} "
            f"first_log_step={_number_text(summary.get('first_log_step'))} "
            f"last_log_step={_number_text(summary.get('last_log_step'))} "
            f"delta_log_step={_number_text(summary.get('delta_log_step'))} "
            f"duration_seconds={_number_text(summary.get('duration_seconds'))} "
            f"log_duration_seconds={_number_text(summary.get('log_duration_seconds'))} "
            f"log_steps_per_second={_number_text(summary.get('log_steps_per_second'))} "
            f"runtime_max_steps={_number_text(summary.get('last_runtime_max_steps'))} "
            f"runtime_eval_steps={_number_text(summary.get('last_runtime_eval_steps'))} "
            f"runtime_save_steps={_number_text(summary.get('last_runtime_save_steps'))} "
            f"runtime_save_total_limit={_number_text(summary.get('last_runtime_save_total_limit'))} "
            f"runtime_min_free_disk_gb={_number_text(summary.get('last_runtime_min_free_disk_gb'))} "
            f"runtime_process_command={_number_text(summary.get('last_runtime_process_command_available'))} "
            f"last_log_max_steps={_number_text(summary.get('last_log_max_steps'))} "
            f"last_log_remaining_seconds={_number_text(summary.get('last_log_remaining_seconds'))} "
            f"last_steps_until_final={_number_text(summary.get('last_log_steps_until_final'))} "
            f"estimated_seconds_until_final={_number_text(summary.get('estimated_seconds_until_final'))} "
            f"last_next_eval_step={_number_text(summary.get('last_next_eval_step'))} "
            f"last_steps_until_next_eval={_number_text(summary.get('last_log_steps_until_next_eval'))} "
            f"last_latest_due_eval_step={_number_text(summary.get('last_latest_due_eval_step'))} "
            f"last_latest_due_eval_ready={_number_text(summary.get('last_latest_due_eval_ready'))} "
            f"last_pending_eval_step={_number_text(summary.get('last_pending_eval_step'))} "
            f"last_steps_since_pending_eval={_number_text(summary.get('last_log_steps_since_pending_eval'))} "
            f"last_next_checkpoint_step={_number_text(summary.get('last_next_checkpoint_step'))} "
            f"last_steps_until_next_checkpoint={_number_text(summary.get('last_log_steps_until_next_checkpoint'))} "
            f"estimated_seconds_until_next_eval={_number_text(summary.get('estimated_seconds_until_next_eval'))} "
            f"estimated_seconds_until_next_checkpoint={_number_text(summary.get('estimated_seconds_until_next_checkpoint'))} "
            f"last_loss={_number_text(summary.get('last_loss'))} "
            f"min_loss={_number_text(summary.get('min_loss'))} "
            f"loss_delta={_number_text(summary.get('loss_delta'))} "
            f"last_eval_loss={_number_text(summary.get('last_eval_loss'))} "
            f"last_eval_step={_number_text(summary.get('last_eval_loss_step'))} "
            f"min_eval_loss={_number_text(summary.get('min_eval_loss'))} "
            f"best_eval_loss_step={_number_text(summary.get('last_best_eval_loss_step'))} "
            f"guard_count={_number_text(summary.get('last_guard_count'))} "
            f"process={_number_text(summary.get('last_process_status'))} "
            f"final_ready={_number_text(summary.get('last_final_checkpoint_ready'))} "
            f"last_save_total_limit={_number_text(summary.get('last_save_total_limit'))} "
            f"last_checkpoint_headroom_checkpoint_gb={_number_text(summary.get('last_checkpoint_headroom_checkpoint_gb'))} "
            f"last_checkpoint_headroom_peak_gb={_number_text(summary.get('last_checkpoint_headroom_peak_gb'))} "
            f"last_checkpoint_headroom_free_after_gb={_number_text(summary.get('last_checkpoint_headroom_free_after_gb'))} "
            f"last_disk_free_gb={_number_text(summary.get('last_disk_free_gb'))} "
            f"min_disk_free_gb={_number_text(summary.get('min_disk_free_gb'))} "
            f"last_disk_margin_gb={_number_text(summary.get('last_disk_margin_gb'))} "
            f"min_disk_margin_gb={_number_text(summary.get('min_disk_margin_gb'))} "
            f"disk_status={_number_text(summary.get('last_disk_status'))} "
            f"watch_stop_eval_step={_number_text(summary.get('last_watch_stop_eval_step'))} "
            f"watch_stop_eval_ready={_number_text(summary.get('last_watch_stop_eval_ready'))} "
            f"watch_stop_reason={_number_text(summary.get('last_watch_stop_reason'))}"
        )
    ]
    if tail > 0:
        start_index = max(len(rows) - tail, 0)
        for index, row in enumerate(rows[start_index:], start_index):
            lines.append(
                (
                    "hf_gpt2_ft_status_history_point "
                    f"index={index} "
                    f"log_step={_number_text(_nested(row, 'log_progress', 'log_latest_step'))} "
                    f"runtime_max_steps={_number_text(_runtime_setting(row, 'max_steps'))} "
                    f"runtime_eval_steps={_number_text(_runtime_setting(row, 'eval_steps'))} "
                    f"runtime_save_steps={_number_text(_runtime_setting(row, 'save_steps'))} "
                    f"runtime_save_total_limit={_number_text(_runtime_setting(row, 'save_total_limit'))} "
                    f"runtime_min_free_disk_gb={_number_text(_runtime_setting(row, 'min_free_disk_gb'))} "
                    f"runtime_process_command={_number_text(_runtime_setting(row, 'process_command_available'))} "
                    f"log_remaining_seconds={_number_text(_nested(row, 'log_progress', 'log_remaining_seconds'))} "
                    f"next_eval_step={_number_text(_nested(row, 'eval_progress', 'next_eval_step'))} "
                    f"steps_until_next_eval={_number_text(_nested(row, 'eval_progress', 'log_steps_until_next_eval'))} "
                    f"pending_eval_step={_number_text(_nested(row, 'eval_progress', 'pending_eval_step'))} "
                    f"next_checkpoint_step={_number_text(_nested(row, 'checkpoint_progress', 'next_checkpoint_step'))} "
                    f"steps_until_next_checkpoint={_number_text(_nested(row, 'checkpoint_progress', 'log_steps_until_next_checkpoint'))} "
                    f"last_loss={_number_text(_nested(row, 'trace', 'trace_last_loss'))} "
                    f"last_eval_loss={_number_text(_nested(row, 'trace', 'trace_last_eval_loss'))} "
                    f"last_eval_step={_number_text(_last_eval_loss_step(row))} "
                    f"best_eval_loss_step={_number_text(_nested(row, 'trace', 'trace_best_eval_loss_step'))} "
                    f"final_ready={_number_text(row.get('final_checkpoint_ready'))} "
                    f"save_total_limit={_number_text(row.get('save_total_limit'))} "
                    f"checkpoint_headroom_peak_gb={_number_text(_checkpoint_headroom(row).get('estimated_peak_checkpoint_gb'))} "
                    f"checkpoint_headroom_free_after_gb={_number_text(_checkpoint_headroom(row).get('free_after_estimated_peak_gb'))} "
                    f"disk_free_gb={_number_text(row.get('disk_free_gb'))} "
                    f"disk_margin_gb={_number_text(row.get('disk_margin_gb'))} "
                    f"disk_status={_number_text(row.get('disk_status'))} "
                    f"watch_stop_eval_step={_number_text(row.get('watch_stop_eval_step'))} "
                    f"watch_stop_eval_ready={_number_text(row.get('watch_stop_eval_ready'))} "
                    f"watch_stop_reason={_number_text(row.get('watch_stop_reason'))}"
                )
            )
    if tail_evals > 0 and rows:
        eval_points = _eval_loss_points(rows[-1])
        eval_start_index = max(len(eval_points) - tail_evals, 0)
        for point in eval_points[eval_start_index:]:
            raw_step_text = (
                f" raw_step={_number_text(point.get('raw_step'))}"
                if point.get("raw_step") is not None
                else ""
            )
            lines.append(
                (
                    "hf_gpt2_ft_status_history_eval "
                    f"index={_number_text(point.get('index'))} "
                    f"step={_number_text(point.get('step'))}"
                    f"{raw_step_text} "
                    f"eval_loss={_number_text(point.get('eval_loss'))} "
                    f"eval_runtime={_number_text(point.get('eval_runtime'))} "
                    f"time_unix_s={_number_text(point.get('time_unix_s'))}"
                )
            )
    return lines


def _load_history(path: Path) -> list[dict[str, Any]]:
    return load_hf_gpt2_finetune_status_history(path)


def summarize_history(
    rows: list[dict[str, Any]], *, label: str | None, history_jsonl: Path
) -> dict[str, Any]:
    return summarize_hf_gpt2_finetune_status_history(
        rows, label=label, history_jsonl=history_jsonl
    )


def history_lines(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    tail: int,
    tail_evals: int = 0,
) -> list[str]:
    return hf_gpt2_finetune_status_history_lines(
        summary, rows, tail=tail, tail_evals=tail_evals
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = load_hf_gpt2_finetune_status_history(args.history_jsonl)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"failed to load status history: {exc}", file=sys.stderr)
        return 1
    summary = summarize_hf_gpt2_finetune_status_history(
        rows, label=args.label, history_jsonl=args.history_jsonl
    )
    lines = hf_gpt2_finetune_status_history_lines(
        summary, rows, tail=args.tail, tail_evals=args.tail_evals
    )
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"hf_gpt2_ft_status_history_summary_json {args.out}")
    if args.lines_out is not None:
        args.lines_out.parent.mkdir(parents=True, exist_ok=True)
        args.lines_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"hf_gpt2_ft_status_history_summary_lines {args.lines_out}")
    if args.out is None and args.lines_out is None:
        print("\n".join(lines))
    return 0
