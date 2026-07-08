#!/usr/bin/env python3
"""Sweep SpiralTorch Z-Space generation controls on a Hugging Face CausalLM."""

from __future__ import annotations

import sys
import json
from collections.abc import Sequence
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

import hf_gpt2_zspace_generation_control_sweep as _legacy  # noqa: E402
from hf_gpt2_zspace_generation_control_sweep import *  # noqa: F401,F403,E402


DEFAULT_OUT = Path("runs/hf-zspace-generation-control-sweep.json")


def _genericize_report(report):
    payload = dict(report)
    if payload.get("row_type") == "hf_gpt2_zspace_generation_control_sweep":
        payload["row_type"] = "hf_zspace_generation_control_sweep"
    summary = payload.get("summary")
    if isinstance(summary, dict):
        summary_payload = dict(summary)
        if (
            summary_payload.get("row_type")
            == "hf_gpt2_zspace_generation_control_sweep_summary"
        ):
            summary_payload["row_type"] = "hf_zspace_generation_control_sweep_summary"
        payload["summary"] = summary_payload
    return payload


def _argv_has_option(raw_argv: Sequence[str], *names: str) -> bool:
    prefixes = tuple(f"{name}=" for name in names)
    return any(arg in names or arg.startswith(prefixes) for arg in raw_argv)


def parse_args(argv: Sequence[str] | None = None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _legacy.parse_args(argv)
    if not _argv_has_option(raw_argv, "--out"):
        args.out = DEFAULT_OUT
    return args


def run_sweep(args):
    return _genericize_report(_legacy.run_sweep(args))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_sweep(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"generation_control_sweep {args.out}")
    return 0 if report.get("status") in {"planned", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
