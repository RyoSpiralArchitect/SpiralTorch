#!/usr/bin/env python3
"""Summarize SpiralTorch GPT-2 fine-tuning run status JSONL history."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history_jsonl", type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument("--tail", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--lines-out", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.history_jsonl.is_file():
        parser.error(f"history_jsonl does not exist: {args.history_jsonl}")
    if args.tail < 0:
        parser.error("--tail must be non-negative")
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


def _nested(row: dict[str, Any], section: str, field: str) -> Any:
    value = row.get(section)
    if not isinstance(value, dict):
        return None
    return value.get(field)


def _latest_checkpoint_name(row: dict[str, Any]) -> Any:
    checkpoint = row.get("latest_checkpoint")
    if isinstance(checkpoint, dict):
        return checkpoint.get("name")
    return None


def summarize_history(
    rows: list[dict[str, Any]], *, label: str | None, history_jsonl: Path
) -> dict[str, Any]:
    first = rows[0] if rows else {}
    last = rows[-1] if rows else {}
    first_log_step = _nested(first, "log_progress", "log_latest_step")
    last_log_step = _nested(last, "log_progress", "log_latest_step")
    first_trace_step = _nested(first, "trace", "trace_max_global_step")
    last_trace_step = _nested(last, "trace", "trace_max_global_step")
    first_time = first.get("time_unix_s")
    last_time = last.get("time_unix_s")
    duration_seconds = (
        float(last_time) - float(first_time)
        if isinstance(first_time, (int, float)) and isinstance(last_time, (int, float))
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
        float(delta_log_step) / duration_seconds
        if isinstance(delta_log_step, int)
        and duration_seconds is not None
        and duration_seconds > 0.0
        else None
    )
    log_steps_until_next_eval = _nested(
        last, "eval_progress", "log_steps_until_next_eval"
    )
    estimated_seconds_until_next_eval = (
        float(log_steps_until_next_eval) / log_steps_per_second
        if isinstance(log_steps_until_next_eval, int)
        and log_steps_per_second is not None
        and log_steps_per_second > 0.0
        else None
    )
    eval_losses = [
        _nested(row, "trace", "trace_last_eval_loss")
        for row in rows
        if isinstance(_nested(row, "trace", "trace_last_eval_loss"), (int, float))
    ]
    disk_values = [
        row.get("disk_free_gb")
        for row in rows
        if isinstance(row.get("disk_free_gb"), (int, float))
    ]
    return {
        "row_type": "hf_gpt2_ft_status_history_summary",
        "label": label,
        "history_jsonl": str(history_jsonl),
        "row_count": len(rows),
        "first_time_unix_s": first_time,
        "last_time_unix_s": last_time,
        "duration_seconds": duration_seconds,
        "first_log_step": first_log_step,
        "last_log_step": last_log_step,
        "delta_log_step": delta_log_step,
        "log_steps_per_second": log_steps_per_second,
        "first_trace_step": first_trace_step,
        "last_trace_step": last_trace_step,
        "delta_trace_step": delta_trace_step,
        "last_next_eval_step": _nested(last, "eval_progress", "next_eval_step"),
        "last_log_steps_until_next_eval": log_steps_until_next_eval,
        "estimated_seconds_until_next_eval": estimated_seconds_until_next_eval,
        "last_eval_loss": _nested(last, "trace", "trace_last_eval_loss"),
        "min_eval_loss": min(eval_losses) if eval_losses else None,
        "last_guard_count": _nested(last, "trace", "training_loss_guard_count"),
        "last_process_status": last.get("process_status"),
        "last_final_checkpoint_ready": last.get("final_checkpoint_ready"),
        "last_checkpoint_count": last.get("checkpoint_count"),
        "last_latest_checkpoint": _latest_checkpoint_name(last),
        "min_disk_free_gb": min(disk_values) if disk_values else None,
        "last_disk_free_gb": last.get("disk_free_gb"),
    }


def history_lines(
    summary: dict[str, Any], rows: list[dict[str, Any]], *, tail: int
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
            f"log_steps_per_second={_number_text(summary.get('log_steps_per_second'))} "
            f"last_next_eval_step={_number_text(summary.get('last_next_eval_step'))} "
            f"last_steps_until_next_eval={_number_text(summary.get('last_log_steps_until_next_eval'))} "
            f"estimated_seconds_until_next_eval={_number_text(summary.get('estimated_seconds_until_next_eval'))} "
            f"last_eval_loss={_number_text(summary.get('last_eval_loss'))} "
            f"min_eval_loss={_number_text(summary.get('min_eval_loss'))} "
            f"guard_count={_number_text(summary.get('last_guard_count'))} "
            f"process={_number_text(summary.get('last_process_status'))} "
            f"final_ready={_number_text(summary.get('last_final_checkpoint_ready'))} "
            f"last_disk_free_gb={_number_text(summary.get('last_disk_free_gb'))} "
            f"min_disk_free_gb={_number_text(summary.get('min_disk_free_gb'))}"
        )
    ]
    if tail <= 0:
        return lines
    start_index = max(len(rows) - tail, 0)
    for index, row in enumerate(rows[start_index:], start_index):
        lines.append(
            (
                "hf_gpt2_ft_status_history_point "
                f"index={index} "
                f"log_step={_number_text(_nested(row, 'log_progress', 'log_latest_step'))} "
                f"next_eval_step={_number_text(_nested(row, 'eval_progress', 'next_eval_step'))} "
                f"steps_until_next_eval={_number_text(_nested(row, 'eval_progress', 'log_steps_until_next_eval'))} "
                f"last_eval_loss={_number_text(_nested(row, 'trace', 'trace_last_eval_loss'))} "
                f"final_ready={_number_text(row.get('final_checkpoint_ready'))} "
                f"disk_free_gb={_number_text(row.get('disk_free_gb'))}"
            )
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = _load_history(args.history_jsonl)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"failed to load status history: {exc}", file=sys.stderr)
        return 1
    summary = summarize_history(rows, label=args.label, history_jsonl=args.history_jsonl)
    lines = history_lines(summary, rows, tail=args.tail)
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


if __name__ == "__main__":
    raise SystemExit(main())
