#!/usr/bin/env python3
"""Summarize SpiralTorch GPT-2 FT wait-launch JSONL history."""

from __future__ import annotations

import argparse
import json
import shutil
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


def _empty_model_metadata() -> dict[str, Any]:
    return {
        "model_profile_id": None,
        "model_profile_extends": None,
        "model_name": None,
        "tokenizer_name": None,
        "model_metadata_row_type": None,
        "model_metadata_source_index": None,
    }


def _payload_model_metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    empty = _empty_model_metadata()
    if not isinstance(payload, dict):
        return dict(empty)
    profile = payload.get("model_profile")
    profile_mapping = profile if isinstance(profile, dict) else {}
    command = payload.get("command")
    profile_id = (
        payload.get("model_profile_id")
        or payload.get("profile_id")
        or profile_mapping.get("profile_id")
        or profile_mapping.get("id")
        or _command_flag_value(command, "--model-profile")
    )
    profile_extends = (
        payload.get("model_profile_extends")
        or payload.get("profile_extends")
        or profile_mapping.get("extends")
    )
    model_name = (
        payload.get("model_name")
        or payload.get("hf_model_name")
        or profile_mapping.get("model_name")
        or _command_flag_value(command, "--model-name")
    )
    tokenizer_name = payload.get("tokenizer_name") or profile_mapping.get(
        "tokenizer_name"
    ) or _command_flag_value(command, "--tokenizer-name")
    return {
        "model_profile_id": None if profile_id is None else str(profile_id),
        "model_profile_extends": None
        if profile_extends is None
        else str(profile_extends),
        "model_name": None if model_name is None else str(model_name),
        "tokenizer_name": None if tokenizer_name is None else str(tokenizer_name),
        "model_metadata_row_type": payload.get("row_type"),
        "model_metadata_source_index": None,
    }


def _history_model_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = _empty_model_metadata()
    source_index: int | None = None
    for index, row in enumerate(rows):
        row_metadata = _payload_model_metadata(row)
        updated = False
        for key, value in row_metadata.items():
            if key == "model_metadata_source_index":
                continue
            if value is not None:
                metadata[key] = value
                updated = True
        if updated:
            source_index = index
    metadata["model_metadata_source_index"] = source_index
    return metadata


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
    model_metadata = _history_model_metadata(rows)
    return {
        "row_type": "hf_gpt2_ft_wait_launch_history_summary",
        "label": label,
        "history_jsonl": str(history_jsonl),
        "row_count": len(rows),
        "first_time_unix_s": first_time,
        "last_time_unix_s": last_time,
        "duration_seconds": duration_seconds,
        "model_profile_id": model_metadata.get("model_profile_id"),
        "model_profile_extends": model_metadata.get("model_profile_extends"),
        "model_name": model_metadata.get("model_name"),
        "tokenizer_name": model_metadata.get("tokenizer_name"),
        "model_metadata_source_index": model_metadata.get("model_metadata_source_index"),
        "model_metadata_row_type": model_metadata.get("model_metadata_row_type"),
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
            f"profile={_number_text(summary.get('model_profile_id'))} "
            f"extends={_number_text(summary.get('model_profile_extends'))} "
            f"model={_number_text(summary.get('model_name'))} "
            f"tokenizer={_number_text(summary.get('tokenizer_name'))} "
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
