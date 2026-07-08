#!/usr/bin/env python3
"""Summarize a SpiralTorch GPT-2 fine-tuning trainer trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import spiraltorch as st


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_jsonl", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--lines-out", type=Path, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--tail-evals", type=int, default=6)
    parser.add_argument("--require-eval-loss-monotonic", action="store_true")
    parser.add_argument("--min-eval-loss-improvement", type=float, default=None)
    args = parser.parse_args(argv)
    if not args.trace_jsonl.is_file():
        parser.error(f"trace_jsonl does not exist: {args.trace_jsonl}")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.tail_evals < 0:
        parser.error("--tail-evals must be non-negative")
    if (
        args.min_eval_loss_improvement is not None
        and args.min_eval_loss_improvement < 0.0
    ):
        parser.error("--min-eval-loss-improvement must be non-negative")
    return args


def _number_text(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _guard_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("training_loss_guard"))


def _progress(summary: dict[str, Any], max_steps: int | None) -> float | None:
    if max_steps is None:
        return None
    step = summary.get("trace_max_global_step")
    if not isinstance(step, (int, float)):
        return None
    return min(max(float(step) / float(max_steps), 0.0), 1.0)


def summarize_trace(args: argparse.Namespace) -> dict[str, Any]:
    rows = st.load_hf_gpt2_finetune_trainer_trace(args.trace_jsonl)
    summary = st.summarize_hf_gpt2_finetune_trainer_trace(
        rows,
        max_steps=args.max_steps,
    )
    summary.update(
        {
            "trace_jsonl": str(args.trace_jsonl),
            "label": args.label,
            "max_steps": args.max_steps,
            "progress": _progress(summary, args.max_steps),
            "training_loss_guard_count": _guard_count(rows),
        }
    )
    return summary


def summary_lines(summary: dict[str, Any], *, tail_evals: int) -> list[str]:
    prefix = "hf_gpt2_ft_trace_summary"
    label = summary.get("label")
    label_part = f" label={label}" if label else ""
    lines = [
        (
            f"{prefix}{label_part} trace={summary.get('trace_jsonl')} "
            f"events={_number_text(summary.get('trace_event_count'))} "
            f"latest_step={_number_text(summary.get('trace_max_global_step'))} "
            f"max_steps={_number_text(summary.get('max_steps'))} "
            f"progress={_number_text(summary.get('progress'))} "
            f"last_loss={_number_text(summary.get('trace_last_loss'))} "
            f"last_eval_loss={_number_text(summary.get('trace_last_eval_loss'))} "
            f"min_eval_loss={_number_text(summary.get('trace_min_eval_loss'))} "
            f"eval_loss_improvement={_number_text(summary.get('trace_eval_loss_improvement'))} "
            f"eval_loss_last_delta={_number_text(summary.get('trace_eval_loss_last_delta'))} "
            f"eval_loss_last_improvement_per_step={_number_text(summary.get('trace_eval_loss_last_improvement_per_step'))} "
            f"eval_loss_projected_final={_number_text(summary.get('trace_eval_loss_projected_final_loss'))} "
            f"eval_loss_monotonic={_number_text(summary.get('trace_eval_loss_monotonic_nonincreasing'))} "
            f"guard_count={_number_text(summary.get('training_loss_guard_count'))} "
            f"steps_per_second_mean={_number_text(summary.get('trace_log_steps_per_second_mean'))}"
        )
    ]
    eval_points = summary.get("trace_eval_loss_points")
    if isinstance(eval_points, list) and tail_evals > 0:
        for point in eval_points[-tail_evals:]:
            if not isinstance(point, dict):
                continue
            lines.append(
                (
                    "hf_gpt2_ft_trace_eval "
                    f"step={_number_text(point.get('step'))} "
                    f"eval_loss={_number_text(point.get('eval_loss'))} "
                    f"eval_runtime={_number_text(point.get('eval_runtime'))}"
                )
            )
    return lines


def validate_summary_gates(
    summary: dict[str, Any],
    *,
    require_eval_loss_monotonic: bool,
    min_eval_loss_improvement: float | None,
) -> list[str]:
    failures: list[str] = []
    if require_eval_loss_monotonic:
        if summary.get("trace_eval_loss_monotonic_nonincreasing") is not True:
            failures.append("eval_loss_not_monotonic_nonincreasing")
    if min_eval_loss_improvement is not None:
        improvement = summary.get("trace_eval_loss_improvement")
        if not isinstance(improvement, (int, float)):
            failures.append("eval_loss_improvement_missing")
        elif float(improvement) < float(min_eval_loss_improvement):
            failures.append(
                "eval_loss_improvement_below_min:"
                f"{improvement}<{min_eval_loss_improvement}"
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarize_trace(args)
    lines = summary_lines(summary, tail_evals=args.tail_evals)
    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"hf_gpt2_ft_trace_summary_json {args.out}")
    if args.lines_out is not None:
        args.lines_out.parent.mkdir(parents=True, exist_ok=True)
        args.lines_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"hf_gpt2_ft_trace_summary_lines {args.lines_out}")
    if args.out is None and args.lines_out is None:
        print("\n".join(lines))
    failures = validate_summary_gates(
        summary,
        require_eval_loss_monotonic=args.require_eval_loss_monotonic,
        min_eval_loss_improvement=args.min_eval_loss_improvement,
    )
    if failures:
        print(
            "hf_gpt2_ft_trace_summary_gate_failed "
            + ",".join(failures),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
