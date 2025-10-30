"""Utility script to summarize scripts available under the tools directory.

The script recursively scans the tools directory, collecting information about
Python scripts and presenting a concise summary for quick discovery. It is
useful when the directory grows large and contains many utilities whose purpose
is not immediately obvious.

Example:
    python tools/list_tools.py

"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Iterable

from tool_summary import ToolSummary, collect_tool_summaries


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the CLI."""

    parser = argparse.ArgumentParser(
        description="List Python tools with their short descriptions."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Root directory to scan. Defaults to the directory of this script.",
    )
    parser.add_argument(
        "--wrap",
        type=int,
        default=70,
        help="Maximum character width for the description column.",
    )
    parser.add_argument(
        "--include-non-python",
        action="store_true",
        help="Include non-Python files in the listing with placeholder summaries.",
    )
    return parser.parse_args()


def print_summaries(summaries: Iterable[ToolSummary], wrap: int) -> None:
    """Pretty-print the collected *summaries* to stdout."""

    summaries = list(summaries)
    if not summaries:
        print("No tool scripts were found.")
        return

    max_path_length = max(len(summary.relative_path) for summary in summaries)
    for summary in summaries:
        description_lines = textwrap.wrap(summary.summary, width=max(wrap, 20)) or [""]
        first_line, *rest = description_lines
        print(f"{summary.relative_path.ljust(max_path_length)}  {first_line}")
        for line in rest:
            print(f"{' ' * max_path_length}  {line}")


def main() -> None:
    args = parse_args()
    summaries = collect_tool_summaries(
        args.root.resolve(), include_non_python=args.include_non_python
    )
    print_summaries(summaries, wrap=args.wrap)


if __name__ == "__main__":
    main()
