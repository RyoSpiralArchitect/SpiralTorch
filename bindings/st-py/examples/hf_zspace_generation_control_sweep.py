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
DEFAULT_MODEL_PROFILE = "causal-lm-local-smoke"


def _generic_row_type(value):
    if not isinstance(value, str):
        return value
    if value == "hf_gpt2_zspace_generation_control_sweep":
        return "hf_zspace_generation_control_sweep"
    if value == "hf_gpt2_zspace_generation_control_sweep_summary":
        return "hf_zspace_generation_control_sweep_summary"
    if value.startswith("hf_gpt2_finetune_"):
        return "hf_finetune_" + value.removeprefix("hf_gpt2_finetune_")
    if value.startswith("hf_gpt2_ft_"):
        return "hf_ft_" + value.removeprefix("hf_gpt2_ft_")
    return value


def _genericize_payload(value):
    if isinstance(value, dict):
        return {
            key: (
                _generic_row_type(item)
                if key == "row_type"
                else _genericize_payload(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_genericize_payload(item) for item in value]
    return value


def _genericize_report(report):
    return _genericize_payload(dict(report))


def _argv_has_option(raw_argv: Sequence[str], *names: str) -> bool:
    prefixes = tuple(f"{name}=" for name in names)
    return any(arg in names or arg.startswith(prefixes) for arg in raw_argv)


def _with_default_model_profile(raw_argv: list[str]) -> list[str]:
    if _argv_has_option(raw_argv, "-h", "--help"):
        return raw_argv
    if _argv_has_option(raw_argv, "--model-configs", "--model-profile", "--model-name"):
        return raw_argv
    return ["--model-profile", DEFAULT_MODEL_PROFILE, *raw_argv]


def parse_args(argv: Sequence[str] | None = None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _legacy.parse_args(_with_default_model_profile(raw_argv))
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
