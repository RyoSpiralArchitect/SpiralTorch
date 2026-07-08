#!/usr/bin/env python3
"""Build a one-page monitor snapshot for a long Hugging Face FT run."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

import hf_gpt2_finetune_monitor_snapshot as _legacy  # noqa: E402
from hf_gpt2_finetune_monitor_snapshot import *  # noqa: F401,F403,E402


def parse_args(argv: list[str] | None = None):
    return _legacy.parse_args(argv)


def _genericize_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    snapshot = dict(snapshot)
    snapshot["row_type"] = "hf_finetune_monitor_snapshot"
    return snapshot


def build_monitor_snapshot(args) -> dict[str, Any]:
    return _genericize_snapshot(_legacy.build_monitor_snapshot(args))


def snapshot_lines(snapshot: dict[str, Any]) -> list[str]:
    replacements = {
        "hf_gpt2_ft_monitor_snapshot": "hf_ft_monitor_snapshot",
        "hf_gpt2_ft_monitor_watch": "hf_ft_monitor_watch",
        "hf_gpt2_ft_monitor_wait_launch": "hf_ft_monitor_wait_launch",
    }
    lines = _legacy.snapshot_lines(dict(snapshot))
    for before, after in replacements.items():
        lines = [
            line.replace(before, after, 1) if line.startswith(before) else line
            for line in lines
        ]
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
        print(f"hf_ft_monitor_snapshot_json {args.out}")
    if args.lines_out is not None:
        args.lines_out.parent.mkdir(parents=True, exist_ok=True)
        args.lines_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"hf_ft_monitor_snapshot_lines {args.lines_out}")
    if args.out is None and args.lines_out is None:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
