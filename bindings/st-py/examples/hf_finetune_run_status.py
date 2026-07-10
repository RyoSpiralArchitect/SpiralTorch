#!/usr/bin/env python3
"""Summarize a live SpiralTorch Hugging Face fine-tuning run directory."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

import spiraltorch as st  # noqa: E402
import hf_gpt2_finetune_run_status as _legacy  # noqa: E402
from hf_gpt2_finetune_run_status import *  # noqa: F401,F403,E402


def _argv_has_option(raw_argv: Sequence[str], *names: str) -> bool:
    prefixes = tuple(f"{name}=" for name in names)
    return any(arg in names or arg.startswith(prefixes) for arg in raw_argv)


def _preferred_run_file(run_dir: Path, generic_name: str, legacy_name: str) -> Path:
    generic_path = run_dir / generic_name
    legacy_path = run_dir / legacy_name
    if generic_path.exists() or not legacy_path.exists():
        return generic_path
    return legacy_path


def parse_args(argv: list[str] | None = None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    alias_parser = argparse.ArgumentParser(add_help=False)
    alias_parser.add_argument("--run-dir", action="append", default=[])
    aliases, legacy_argv = alias_parser.parse_known_args(raw_argv)
    if len(aliases.run_dir) > 1:
        alias_parser.error("--run-dir may be provided only once")
    if aliases.run_dir:
        legacy_argv.insert(0, aliases.run_dir[0])
    args = _legacy.parse_args(legacy_argv)
    if not _argv_has_option(raw_argv, "--trace-jsonl"):
        args.trace_jsonl = _preferred_run_file(
            args.run_dir,
            st.HF_FINETUNE_TRAINER_TRACE_FILENAME,
            _legacy.DEFAULT_TRACE_NAME,
        )
    if not _argv_has_option(raw_argv, "--run-card"):
        args.run_card = _preferred_run_file(
            args.run_dir,
            st.HF_FINETUNE_RUN_CARD_FILENAME,
            _legacy.DEFAULT_RUN_CARD_NAME,
        )
    return args


def _genericize_status(status: dict[str, Any]) -> dict[str, Any]:
    status = dict(status)
    status["row_type"] = "hf_finetune_run_status"
    trace = status.get("trace")
    if isinstance(trace, dict):
        trace["row_type"] = "hf_finetune_trainer_trace_summary"
    return status


def summarize_run(args) -> dict[str, Any]:
    return _genericize_status(_legacy.summarize_run(args))


def status_lines(status: dict[str, Any], *, tail_evals: int) -> list[str]:
    replacements = {
        "hf_gpt2_ft_run_status": "hf_ft_run_status",
        "hf_gpt2_ft_run_eval": "hf_ft_run_eval",
        "hf_gpt2_ft_run_wait": "hf_ft_run_wait",
    }
    lines = _legacy.status_lines(dict(status), tail_evals=tail_evals)
    for before, after in replacements.items():
        lines = [
            line.replace(before, after, 1) if line.startswith(before) else line
            for line in lines
        ]
    return lines


def _emit_status(args, status: dict[str, Any]) -> None:
    status = _genericize_status(status)
    lines = status_lines(status, tail_evals=args.tail_evals)
    payload = json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        if not args.quiet:
            print(f"hf_ft_run_status_json {args.out}")
    if args.lines_out is not None:
        args.lines_out.parent.mkdir(parents=True, exist_ok=True)
        args.lines_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"hf_ft_run_status_lines {args.lines_out}")
    if args.jsonl_out is not None:
        args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl_out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(status, ensure_ascii=False, sort_keys=True) + "\n")
        if not args.quiet:
            print(f"hf_ft_run_status_jsonl {args.jsonl_out}")
    if args.out is None and args.lines_out is None and args.jsonl_out is None:
        print("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.watch_interval_seconds is None:
        _emit_status(args, summarize_run(args))
        return 0
    count = 0
    while True:
        if count and not args.quiet:
            print()
        status = summarize_run(args)
        stop_reason = _legacy._watch_stop_reason(args, status)
        if stop_reason is not None:
            status["watch_stop_reason"] = stop_reason
        _emit_status(args, status)
        count += 1
        if stop_reason is not None:
            break
        if args.watch_count is not None and count >= args.watch_count:
            break
        time.sleep(float(args.watch_interval_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
