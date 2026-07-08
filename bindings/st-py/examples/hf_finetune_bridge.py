#!/usr/bin/env python3
"""Generic Hugging Face fine-tuning bridge with SpiralTorch Z-Space preflight."""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

import hf_gpt2_finetune_bridge as _legacy  # noqa: E402
from hf_gpt2_finetune_bridge import *  # noqa: F401,F403,E402
from hf_gpt2_finetune_bridge import parse_args as _legacy_parse_args  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("runs/hf-finetune")
DEFAULT_RUN_CARD_FILENAME = "spiraltorch-hf-finetune-run-card.json"
DEFAULT_TRAINER_TRACE_FILENAME = "spiraltorch-hf-finetune-trainer-trace.jsonl"


def _argv_has_option(raw_argv: list[str], *names: str) -> bool:
    prefixes = tuple(f"{name}=" for name in names)
    return any(arg in names or arg.startswith(prefixes) for arg in raw_argv)


def parse_args(argv: list[str] | None = None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _legacy_parse_args(argv)
    if not _argv_has_option(raw_argv, "--output-dir"):
        args.output_dir = DEFAULT_OUTPUT_DIR
    if not _argv_has_option(raw_argv, "--run-card"):
        args.run_card = args.output_dir / DEFAULT_RUN_CARD_FILENAME
    if not _argv_has_option(raw_argv, "--trainer-trace-jsonl"):
        args.trainer_trace_jsonl = args.output_dir / DEFAULT_TRAINER_TRACE_FILENAME
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    remote_access_report = _legacy._hf_remote_access_report(args)
    with _legacy._hf_remote_access(args):
        return _legacy._main_with_runtime_access(args, remote_access_report)


if __name__ == "__main__":
    raise SystemExit(main())
