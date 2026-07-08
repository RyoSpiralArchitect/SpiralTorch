#!/usr/bin/env python3
"""Summarize SpiralTorch GPT-2 FT wait-launch JSONL history."""

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
    parser.add_argument("--require-launched", action="store_true")
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


def _launch_disk_guard(row: dict[str, Any]) -> dict[str, Any]:
    guard = row.get("launch_disk_guard")
    return guard if isinstance(guard, dict) else {}


def summarize_history(
    rows: list[dict[str, Any]], *, label: str | None, history_jsonl: Path
) -> dict[str, Any]:
    first = rows[0] if rows else {}
    last = rows[-1] if rows else {}
    first_time = first.get("time_unix_s")
    last_time = last.get("time_unix_s")
    duration_seconds = (
        float(last_time) - float(first_time)
        if isinstance(first_time, (int, float)) and isinstance(last_time, (int, float))
        else None
    )
    launched_rows = [
        row
        for row in rows
        if row.get("launched_pid") is not None
        or row.get("status") in {"launching", "launched", "finished", "launch_error"}
    ]
    launch_disk_guard = _launch_disk_guard(last)
    return {
        "row_type": "hf_gpt2_ft_wait_launch_history_summary",
        "label": label,
        "history_jsonl": str(history_jsonl),
        "row_count": len(rows),
        "first_time_unix_s": first_time,
        "last_time_unix_s": last_time,
        "duration_seconds": duration_seconds,
        "first_status": first.get("status"),
        "last_status": last.get("status"),
        "last_process_alive": last.get("process_alive"),
        "last_checkpoint_ready": last.get("checkpoint_ready"),
        "last_status_card_status": last.get("status_card_status"),
        "last_launched_pid": last.get("launched_pid"),
        "last_returncode": last.get("returncode"),
        "last_launch_error": last.get("launch_error"),
        "last_launch_disk_status": launch_disk_guard.get("status"),
        "last_launch_disk_min_free_gb": launch_disk_guard.get("min_free_gb"),
        "last_launch_disk_free_gb": launch_disk_guard.get("free_gb"),
        "last_launch_disk_peak_gb": launch_disk_guard.get(
            "estimated_peak_checkpoint_gb"
        ),
        "last_launch_disk_free_after_gb": launch_disk_guard.get(
            "free_after_estimated_peak_gb"
        ),
        "launched": bool(launched_rows),
        "launched_row_count": len(launched_rows),
        "last_launched_pid_file": last.get("launched_pid_file"),
        "last_launched_log_file": last.get("launched_log_file"),
    }


def history_lines(
    summary: dict[str, Any], rows: list[dict[str, Any]], *, tail: int
) -> list[str]:
    label = summary.get("label") or "wait-launch"
    lines = [
        (
            "hf_gpt2_ft_wait_launch_history "
            f"label={label} "
            f"rows={_number_text(summary.get('row_count'))} "
            f"duration_seconds={_number_text(summary.get('duration_seconds'))} "
            f"first_status={_number_text(summary.get('first_status'))} "
            f"last_status={_number_text(summary.get('last_status'))} "
            f"process_alive={_number_text(summary.get('last_process_alive'))} "
            f"checkpoint_ready={_number_text(summary.get('last_checkpoint_ready'))} "
            f"status_card_status={_number_text(summary.get('last_status_card_status'))} "
            f"launched={_number_text(summary.get('launched'))} "
            f"launched_pid={_number_text(summary.get('last_launched_pid'))} "
            f"returncode={_number_text(summary.get('last_returncode'))} "
            f"launch_disk_status={_number_text(summary.get('last_launch_disk_status'))} "
            f"launch_disk_min_free_gb={_number_text(summary.get('last_launch_disk_min_free_gb'))} "
            f"launch_disk_peak_gb={_number_text(summary.get('last_launch_disk_peak_gb'))} "
            f"launch_disk_free_after_gb={_number_text(summary.get('last_launch_disk_free_after_gb'))} "
            f"launch_error={_number_text(summary.get('last_launch_error'))}"
        )
    ]
    if tail <= 0:
        return lines
    start_index = max(len(rows) - tail, 0)
    for index, row in enumerate(rows[start_index:], start_index):
        lines.append(
            (
                "hf_gpt2_ft_wait_launch_history_point "
                f"index={index} "
                f"status={_number_text(row.get('status'))} "
                f"process_alive={_number_text(row.get('process_alive'))} "
                f"checkpoint_ready={_number_text(row.get('checkpoint_ready'))} "
                f"status_card_status={_number_text(row.get('status_card_status'))} "
                f"launch_disk_status={_number_text(_launch_disk_guard(row).get('status'))} "
                f"launch_disk_free_after_gb={_number_text(_launch_disk_guard(row).get('free_after_estimated_peak_gb'))} "
                f"launched_pid={_number_text(row.get('launched_pid'))} "
                f"returncode={_number_text(row.get('returncode'))}"
            )
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = _load_history(args.history_jsonl)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"failed to load wait-launch history: {exc}", file=sys.stderr)
        return 1
    summary = summarize_history(rows, label=args.label, history_jsonl=args.history_jsonl)
    if args.require_launched and not summary["launched"]:
        print("wait-launch history has not launched a command yet", file=sys.stderr)
        return 2
    lines = history_lines(summary, rows, tail=args.tail)
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"hf_gpt2_ft_wait_launch_history_summary_json {args.out}")
    if args.lines_out is not None:
        args.lines_out.parent.mkdir(parents=True, exist_ok=True)
        args.lines_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"hf_gpt2_ft_wait_launch_history_summary_lines {args.lines_out}")
    if args.out is None and args.lines_out is None:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
