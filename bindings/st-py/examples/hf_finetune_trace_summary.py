#!/usr/bin/env python3
"""Summarize a SpiralTorch Hugging Face fine-tuning trainer trace."""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

import spiraltorch as st  # noqa: E402
import hf_gpt2_finetune_trace_summary as _legacy  # noqa: E402
from hf_gpt2_finetune_trace_summary import *  # noqa: F401,F403,E402


def parse_args(argv: list[str] | None = None):
    return _legacy.parse_args(argv)


def summarize_trace(args) -> dict[str, object]:
    rows = st.load_hf_finetune_trainer_trace(args.trace_jsonl)
    summary = st.summarize_hf_finetune_trainer_trace(
        rows,
        max_steps=args.max_steps,
    )
    summary["row_type"] = "hf_finetune_trainer_trace_summary"
    summary.update(
        {
            "trace_jsonl": str(args.trace_jsonl),
            "label": args.label,
            "max_steps": args.max_steps,
            "progress": _legacy._progress(summary, args.max_steps),
            "training_loss_guard_count": _legacy._guard_count(rows),
        }
    )
    return summary


def summary_lines(summary: dict[str, object], *, tail_evals: int) -> list[str]:
    replacements = {
        "hf_gpt2_ft_trace_summary": "hf_ft_trace_summary",
        "hf_gpt2_ft_trace_eval": "hf_ft_trace_eval",
    }
    lines = _legacy.summary_lines(dict(summary), tail_evals=tail_evals)
    for before, after in replacements.items():
        lines = [
            line.replace(before, after, 1) if line.startswith(before) else line
            for line in lines
        ]
    return lines


def validate_summary_gates(
    summary: dict[str, object],
    *,
    require_eval_loss_monotonic: bool,
    min_eval_loss_improvement: float | None,
) -> list[str]:
    return _legacy.validate_summary_gates(
        summary,
        require_eval_loss_monotonic=require_eval_loss_monotonic,
        min_eval_loss_improvement=min_eval_loss_improvement,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarize_trace(args)
    lines = summary_lines(summary, tail_evals=args.tail_evals)
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"hf_ft_trace_summary_json {args.out}")
    if args.lines_out is not None:
        args.lines_out.parent.mkdir(parents=True, exist_ok=True)
        args.lines_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"hf_ft_trace_summary_lines {args.lines_out}")
    if args.out is None and args.lines_out is None:
        print("\n".join(lines))
    failures = validate_summary_gates(
        summary,
        require_eval_loss_monotonic=args.require_eval_loss_monotonic,
        min_eval_loss_improvement=args.min_eval_loss_improvement,
    )
    if failures:
        print(
            "hf_ft_trace_summary_gate_failed " + ",".join(failures),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
