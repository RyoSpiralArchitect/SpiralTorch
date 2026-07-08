#!/usr/bin/env python3
"""Archive a Hugging Face FT run ops snapshot with a recommended next action."""

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
    parser.add_argument("--generation-limit", type=int, default=12)
    parser.add_argument("--checkpoint-limit", type=int, default=12)
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
        help="Execute the recommended milestone handoff when it is ready.",
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
    parser.add_argument("--quiet", action="store_true")
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
    if args.generation_limit < 0:
        parser.error("--generation-limit must be non-negative")
    if args.checkpoint_limit < 0:
        parser.error("--checkpoint-limit must be non-negative")
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
    return st.hf_finetune_run_ops_snapshot_report(
        args.run_dir,
        next_run_dir=args.next_run_dir,
        generation_limit=args.generation_limit,
        checkpoint_limit=args.checkpoint_limit,
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
        execute=args.execute,
        dry_run=args.dry_run,
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
    archived = st.write_hf_finetune_run_ops_snapshot(
        report,
        run_dir=args.run_dir,
        out=args.out,
        lines_out=args.lines_out,
    )
    report.clear()
    report.update(archived)
    return Path(archived["out"]), Path(archived["lines_out"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    out_path, lines_path = write_report(report, args)
    lines = st.hf_finetune_run_ops_snapshot_lines(report)
    if not args.quiet:
        print("\n".join(lines))
    print(f"hf_ft_run_ops_snapshot_json {out_path}")
    print(f"hf_ft_run_ops_snapshot_lines {lines_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
