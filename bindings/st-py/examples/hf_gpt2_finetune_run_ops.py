#!/usr/bin/env python3
"""Compatibility wrapper for the generic Hugging Face FT run ops snapshot CLI."""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
for root in (EXAMPLES_ROOT, PACKAGE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import spiraltorch as st
from hf_finetune_run_ops import build_report, parse_args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    archived = st.write_hf_gpt2_finetune_run_ops_snapshot(
        report,
        run_dir=args.run_dir,
        out=args.out,
        lines_out=args.lines_out,
    )
    report.clear()
    report.update(archived)
    out_path = Path(archived["out"])
    lines_path = Path(archived["lines_out"])
    lines = st.hf_gpt2_finetune_run_ops_snapshot_lines(report)
    if not args.quiet:
        print("\n".join(lines))
    print(f"hf_gpt2_ft_run_ops_snapshot_json {out_path}")
    print(f"hf_gpt2_ft_run_ops_snapshot_lines {lines_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
