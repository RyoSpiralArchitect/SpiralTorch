#!/usr/bin/env python3
"""Resolve, execute, and archive a GPT-2 FT milestone runtime handoff.

This is the small operational entrypoint around the importable
``spiraltorch`` milestone runtime API. By default it writes JSON/TXT
artifacts and keeps generation-control execution in dry-run mode so a live
training terminal can safely inspect the next handoff before running model
inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import spiraltorch as st


def _existing_file(
    path: Path | None, *, parser: argparse.ArgumentParser, name: str
) -> None:
    if path is not None and not path.is_file():
        parser.error(f"{name} does not exist: {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--run-dir", dest="run_dir_flag", type=Path, default=None)
    parser.add_argument("--next-run-dir", type=Path, default=None)
    parser.add_argument("--direct", type=Path, default=None)
    parser.add_argument("--eval-watch", type=Path, default=None)
    parser.add_argument("--checkpoint-watch", type=Path, default=None)
    parser.add_argument("--final-watch", type=Path, default=None)
    parser.add_argument("--wait-launch", type=Path, default=None)
    parser.add_argument("--milestone-step", type=int, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--label-prefix", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--compare-with-sweep", action="append", type=Path, default=None)
    parser.add_argument("--compare-with-label", action="append", default=None)
    parser.add_argument("--curve-out", type=Path, default=None)
    parser.add_argument("--curve-lines-out", type=Path, default=None)
    parser.add_argument("--trainer-trace-jsonl", type=Path, default=None)
    parser.add_argument("--run-card", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the handoff. Generation-control still stays dry-run by default.",
    )
    parser.add_argument(
        "--command-backend",
        choices=("package", "subprocess"),
        default="package",
        help="Use importable spiraltorch APIs or spawn the generated command.",
    )
    dry_group = parser.add_mutually_exclusive_group()
    dry_group.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="Plan generation-control without model inference (default).",
    )
    dry_group.add_argument(
        "--run-generation-control",
        dest="dry_run",
        action="store_false",
        help="Actually run generation-control model inference when --execute is set.",
    )
    parser.add_argument("--cwd", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--lines-out", type=Path, default=None)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the compact runtime lines after writing artifacts.",
    )
    args = parser.parse_args(argv)
    run_dir = args.run_dir_flag or args.run_dir
    if args.run_dir is not None and args.run_dir_flag is not None:
        parser.error("provide run_dir positionally or with --run-dir, not both")
    if run_dir is None:
        parser.error("provide run_dir or --run-dir")
    if not run_dir.is_dir():
        parser.error(f"run_dir does not exist: {run_dir}")
    if args.next_run_dir is not None and not args.next_run_dir.is_dir():
        parser.error(f"next_run_dir does not exist: {args.next_run_dir}")
    if args.milestone_step is not None and args.milestone_step < 0:
        parser.error("--milestone-step must be non-negative")
    if args.top_n < 0:
        parser.error("--top-n must be non-negative")
    if args.timeout is not None and args.timeout < 0.0:
        parser.error("--timeout must be non-negative")
    if args.compare_with_label and (
        len(args.compare_with_label) != len(args.compare_with_sweep or [])
    ):
        parser.error("--compare-with-label must match --compare-with-sweep count")
    for index, path in enumerate(args.compare_with_sweep or [], 1):
        _existing_file(path, parser=parser, name=f"--compare-with-sweep #{index}")
    for path, name in (
        (args.direct, "--direct"),
        (args.eval_watch, "--eval-watch"),
        (args.checkpoint_watch, "--checkpoint-watch"),
        (args.final_watch, "--final-watch"),
        (args.wait_launch, "--wait-launch"),
        (args.trainer_trace_jsonl, "--trainer-trace-jsonl"),
        (args.run_card, "--run-card"),
    ):
        _existing_file(path, parser=parser, name=name)
    args.run_dir = run_dir
    return args


def build_report(
    args: argparse.Namespace,
    *,
    package_runner: Callable[..., dict[str, Any]] | None = None,
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    return st.hf_gpt2_finetune_milestone_runtime_from_run_dir_report(
        args.run_dir,
        next_run_dir=args.next_run_dir,
        direct=args.direct,
        eval_watch=args.eval_watch,
        checkpoint_watch=args.checkpoint_watch,
        final_watch=args.final_watch,
        wait_launch=args.wait_launch,
        milestone_step=args.milestone_step,
        label=args.label,
        label_prefix=args.label_prefix,
        checkpoint=args.checkpoint,
        compare_with_sweep=args.compare_with_sweep,
        compare_with_label=args.compare_with_label,
        curve_out=args.curve_out,
        curve_lines_out=args.curve_lines_out,
        trainer_trace_jsonl=args.trainer_trace_jsonl,
        run_card=args.run_card,
        top_n=args.top_n,
        wait=args.wait,
        dry_run=args.dry_run,
        execute=args.execute,
        use_package_api=args.command_backend == "package",
        cwd=args.cwd,
        timeout=args.timeout,
        package_runner=package_runner,
        runner=runner,
    )


def write_report(
    report: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    archived = st.write_hf_gpt2_finetune_milestone_runtime_report(
        report,
        run_dir=args.run_dir,
        out=args.out,
        lines_out=args.lines_out,
    )
    report.clear()
    report.update(archived)
    out_path = Path(archived["out"])
    lines_path = Path(archived["lines_out"])
    return out_path, lines_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    out_path, lines_path = write_report(report, args)
    lines = st.hf_gpt2_finetune_milestone_runtime_lines(report)
    if not args.quiet:
        print("\n".join(lines))
    print(f"hf_gpt2_ft_milestone_runtime_json {out_path}")
    print(f"hf_gpt2_ft_milestone_runtime_lines {lines_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
