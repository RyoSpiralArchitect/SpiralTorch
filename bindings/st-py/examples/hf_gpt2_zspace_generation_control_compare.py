#!/usr/bin/env python3
"""Compare GPT-2 Z-Space generation-control sweep artifacts."""

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
    parser.add_argument(
        "sweeps",
        nargs="+",
        type=Path,
        help="Generation-control sweep JSON artifacts to compare.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Optional label for the matching sweep path. May be repeated.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--lines-out", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args(argv)
    if args.top_n < 0:
        parser.error("--top-n must be non-negative")
    if args.label and len(args.label) != len(args.sweeps):
        parser.error("--label must be repeated exactly once per sweep path")
    missing = [path for path in args.sweeps if not path.is_file()]
    if missing:
        parser.error("sweep artifact does not exist: " + ", ".join(map(str, missing)))
    return args


def _comparison_sources(args: argparse.Namespace) -> dict[str, Path] | list[Path]:
    if args.label:
        return {
            str(label): path
            for label, path in zip(args.label, args.sweeps, strict=True)
        }
    return list(args.sweeps)


def compare_sweeps(args: argparse.Namespace) -> dict[str, Any]:
    return st.compare_zspace_generation_control_sweeps(
        _comparison_sources(args),
        top_n=int(args.top_n),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    comparison = compare_sweeps(args)
    lines = st.summarize_zspace_generation_control_sweep_comparison_lines(
        comparison,
        top_n=int(args.top_n),
    )
    payload = json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"generation_control_compare {args.out}")
    else:
        print(payload, end="")
    if args.lines_out is not None:
        args.lines_out.parent.mkdir(parents=True, exist_ok=True)
        args.lines_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"generation_control_compare_lines {args.lines_out}")
    else:
        for line in lines:
            print(line, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
