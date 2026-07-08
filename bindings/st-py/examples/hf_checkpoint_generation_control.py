#!/usr/bin/env python3
"""Run Z-Space generation-control sweeps for Hugging Face FT checkpoints."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from hf_gpt2_ft_checkpoint_generation_control import *  # noqa: F401,F403,E402
from hf_gpt2_ft_checkpoint_generation_control import (  # noqa: E402
    parse_args as _legacy_parse_args,
)
from hf_gpt2_ft_checkpoint_generation_control import (  # noqa: E402
    run_checkpoint_generation_control,
)

SWEEP_SCRIPT = EXAMPLES_ROOT / "hf_zspace_generation_control_sweep.py"
COMPARE_SCRIPT = EXAMPLES_ROOT / "hf_zspace_generation_control_compare.py"
CURVE_SCRIPT = EXAMPLES_ROOT / "hf_finetune_generation_curve.py"


def _argv_has_option(raw_argv: Sequence[str], *names: str) -> bool:
    prefixes = tuple(f"{name}=" for name in names)
    return any(arg in names or arg.startswith(prefixes) for arg in raw_argv)


def parse_args(argv: Sequence[str] | None = None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _legacy_parse_args(argv)
    if not _argv_has_option(raw_argv, "--sweep-script"):
        args.sweep_script = SWEEP_SCRIPT
    if not _argv_has_option(raw_argv, "--compare-script"):
        args.compare_script = COMPARE_SCRIPT
    if not _argv_has_option(raw_argv, "--curve-script"):
        args.curve_script = CURVE_SCRIPT
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_checkpoint_generation_control(args)
    if args.dry_run and args.run_card is None:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
