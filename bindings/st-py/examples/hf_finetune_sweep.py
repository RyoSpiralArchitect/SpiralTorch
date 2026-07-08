#!/usr/bin/env python3
"""Run Hugging Face fine-tune sweeps and compare SpiralTorch run cards."""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from hf_gpt2_finetune_sweep import *  # noqa: F401,F403,E402
from hf_gpt2_finetune_sweep import parse_args as _legacy_parse_args  # noqa: E402
from hf_gpt2_finetune_sweep import run_sweep  # noqa: E402

DEFAULT_BRIDGE = Path(__file__).resolve().with_name("hf_finetune_bridge.py")


def _argv_has_option(raw_argv: list[str], *names: str) -> bool:
    prefixes = tuple(f"{name}=" for name in names)
    return any(arg in names or arg.startswith(prefixes) for arg in raw_argv)


def parse_args(argv: list[str] | None = None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _legacy_parse_args(argv)
    if not _argv_has_option(raw_argv, "--bridge-script"):
        args.bridge_script = DEFAULT_BRIDGE
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_sweep(args)
    report_path = args.out_dir / "sweep-report.json"
    print(f"sweep_report {report_path}")
    scale_up_command = report.get("scale_up_command")
    if isinstance(scale_up_command, dict):
        print(
            "scale_up_command "
            f"{report.get('scale_up_command_path')} "
            f"status={scale_up_command.get('status')}"
        )
        command_display = scale_up_command.get("command_display")
        if command_display:
            print(f"scale_up_replay {command_display}")
    failed = int(report.get("failed_run_count") or 0)
    return 1 if failed and args.fail_fast else 0


if __name__ == "__main__":
    raise SystemExit(main())
