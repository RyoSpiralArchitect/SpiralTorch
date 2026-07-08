#!/usr/bin/env python3
"""Summarize SpiralTorch Hugging Face fine-tuning status JSONL history."""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

import hf_gpt2_finetune_status_history_summary as _legacy  # noqa: E402
from hf_gpt2_finetune_status_history_summary import *  # noqa: F401,F403,E402


def _genericize_summary(summary: dict[str, object]) -> dict[str, object]:
    payload = dict(summary)
    if payload.get("row_type") == "hf_gpt2_ft_status_history_summary":
        payload["row_type"] = "hf_ft_status_history_summary"
    return payload


def _genericize_lines(lines: list[str]) -> list[str]:
    replacements = {
        "hf_gpt2_ft_status_history": "hf_ft_status_history",
        "hf_gpt2_ft_status_history_point": "hf_ft_status_history_point",
        "hf_gpt2_ft_status_history_eval": "hf_ft_status_history_eval",
    }
    generic_lines = []
    for line in lines:
        for before, after in replacements.items():
            if line.startswith(before):
                line = line.replace(before, after, 1)
                break
        generic_lines.append(line)
    return generic_lines


def parse_args(argv: list[str] | None = None):
    return _legacy.parse_args(argv)


def _load_history(path: Path):
    return _legacy._load_history(path)


def summarize_history(rows, *, label, history_jsonl: Path):
    return _genericize_summary(
        _legacy.summarize_history(rows, label=label, history_jsonl=history_jsonl)
    )


def history_lines(
    summary: dict[str, object],
    rows: list[dict[str, object]],
    *,
    tail: int,
    tail_evals: int = 0,
) -> list[str]:
    legacy_summary = dict(summary)
    if legacy_summary.get("row_type") == "hf_ft_status_history_summary":
        legacy_summary["row_type"] = "hf_gpt2_ft_status_history_summary"
    return _genericize_lines(
        _legacy.history_lines(
            legacy_summary,
            rows,
            tail=tail,
            tail_evals=tail_evals,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = _load_history(args.history_jsonl)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"failed to load status history: {exc}", file=sys.stderr)
        return 1
    summary = summarize_history(
        rows,
        label=args.label,
        history_jsonl=args.history_jsonl,
    )
    lines = history_lines(
        summary,
        rows,
        tail=args.tail,
        tail_evals=args.tail_evals,
    )
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"hf_ft_status_history_summary_json {args.out}")
    if args.lines_out is not None:
        args.lines_out.parent.mkdir(parents=True, exist_ok=True)
        args.lines_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"hf_ft_status_history_summary_lines {args.lines_out}")
    if args.out is None and args.lines_out is None:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
