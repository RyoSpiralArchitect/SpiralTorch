#!/usr/bin/env python3
"""Archive a lightweight manifest of Hugging Face FT run artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import spiraltorch as st


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--run-dir", dest="run_dir_flag", type=Path, default=None)
    parser.add_argument("--next-run-dir", type=Path, default=None)
    parser.add_argument("--generation-limit", type=int, default=12)
    parser.add_argument("--checkpoint-limit", type=int, default=12)
    parser.add_argument("--top-n", type=int, default=5)
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
    if args.top_n < 0:
        parser.error("--top-n must be non-negative")
    args.run_dir = run_dir
    return args


def build_report(args: argparse.Namespace) -> dict[str, object]:
    return st.hf_finetune_run_artifact_manifest(
        args.run_dir,
        next_run_dir=args.next_run_dir,
        generation_limit=args.generation_limit,
        checkpoint_limit=args.checkpoint_limit,
    )


def write_report(
    report: dict[str, object],
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    archived = st.write_hf_finetune_run_artifact_manifest(
        report,
        run_dir=args.run_dir,
        out=args.out,
        lines_out=args.lines_out,
        top_n=args.top_n,
    )
    report.clear()
    report.update(archived)
    return Path(archived["out"]), Path(archived["lines_out"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    out_path, lines_path = write_report(report, args)
    lines = st.hf_finetune_run_artifact_manifest_lines(
        report,
        top_n=args.top_n,
    )
    if not args.quiet:
        print("\n".join(lines))
    print(f"hf_ft_run_artifact_manifest_json {out_path}")
    print(f"hf_ft_run_artifact_manifest_lines {lines_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
