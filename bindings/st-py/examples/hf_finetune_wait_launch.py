#!/usr/bin/env python3
"""Wait for a long Hugging Face FT handoff, then launch the next command."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

import hf_gpt2_finetune_wait_launch as _legacy  # noqa: E402
from hf_gpt2_finetune_wait_launch import *  # noqa: F401,F403,E402


def parse_args(argv=None):
    return _legacy.parse_args(argv)


def _genericize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    if payload.get("row_type") == "hf_gpt2_finetune_wait_launch":
        payload["row_type"] = "hf_finetune_wait_launch"
    guard = payload.get("launch_disk_guard")
    if isinstance(guard, dict):
        guard = dict(guard)
        if guard.get("row_type") == "hf_gpt2_ft_wait_launch_disk_guard":
            guard["row_type"] = "hf_ft_wait_launch_disk_guard"
        payload["launch_disk_guard"] = guard
    return payload


def _write_manifest(
    args,
    status: str,
    *,
    returncode: int | None = None,
    launch_error: str | None = None,
    launched_pid: int | None = None,
) -> dict[str, Any]:
    pid = _legacy._resolved_pid(args)
    launched_pid = (
        launched_pid
        if launched_pid is not None
        else _legacy._read_pid(args.launched_pid_file)
    )
    payload = _genericize_payload(
        {
            "row_type": "hf_gpt2_finetune_wait_launch",
            "status": status,
            "time_unix_s": time.time(),
            "pid": pid,
            "pid_file": None if args.pid_file is None else str(args.pid_file),
            "process_alive": _legacy._process_alive(pid),
            "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
            "checkpoint_ready_file": args.checkpoint_ready_file,
            "checkpoint_ready": _legacy._checkpoint_ready(args),
            "status_card": None if args.status_card is None else str(args.status_card),
            "status_card_status": _legacy._status_card_status(args),
            "launched_pid_file": (
                None
                if args.launched_pid_file is None
                else str(args.launched_pid_file)
            ),
            "launched_pid": launched_pid,
            "launched_log_file": (
                None
                if args.launched_log_file is None
                else str(args.launched_log_file)
            ),
            "launched_log_mode": args.launched_log_mode,
            "command": [str(item) for item in args.command],
            "launch_disk_guard": _legacy._launch_disk_guard(args),
            "detach": bool(args.detach),
            "dry_run": bool(args.dry_run),
            "returncode": returncode,
            "launch_error": launch_error,
        }
    )
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
        "hf_ft_wait_launch "
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


def run_wait_launch(args) -> dict[str, Any]:
    original = _legacy._write_manifest
    _legacy._write_manifest = _write_manifest
    try:
        return _genericize_payload(_legacy.run_wait_launch(args))
    finally:
        _legacy._write_manifest = original


def main(argv=None) -> int:
    args = parse_args(argv)
    payload = run_wait_launch(args)
    return int(payload.get("returncode") or 0)


if __name__ == "__main__":
    raise SystemExit(main())
