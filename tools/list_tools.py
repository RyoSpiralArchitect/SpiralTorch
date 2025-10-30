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
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Render the results as a table or as JSON for automation.",
    )
    parser.add_argument(
        "--filter",
        help="Only show tools whose path or summary contains the given substring.",
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Include the longer detail text when available.",
    )
    parser.add_argument(
        "--show-source",
        action="store_true",
        help="Display where each description was obtained (docstring, comment, …).",
    )
    return parser.parse_args()


def _filter_summaries(
    summaries: Iterable[ToolSummary], needle: str | None
) -> list[ToolSummary]:
    if not needle:
        return list(summaries)
    needle_lower = needle.lower()
    return [
        summary
        for summary in summaries
        if needle_lower in summary.relative_path.lower()
        or needle_lower in summary.summary.lower()
        or (summary.detail and needle_lower in summary.detail.lower())
    ]


def _print_table(
    summaries: list[ToolSummary],
    *,
    wrap: int,
    show_details: bool,
    show_source: bool,
) -> None:
    if not summaries:
        print("No tool scripts were found.")
        return

    max_path_length = max(len(summary.relative_path) for summary in summaries)
    max_lang_length = max(len(summary.language) for summary in summaries)
    if show_source:
        max_source_length = max(len(summary.description_source) for summary in summaries)
    else:
        max_source_length = 0

    header_path = "Path".ljust(max_path_length)
    header_lang = "Lang".ljust(max_lang_length)
    if show_source:
        header_src = "Source".ljust(max_source_length)
        print(f"{header_path}  {header_lang}  {header_src}  Description")
        print("-" * (max_path_length + max_lang_length + max_source_length + 18))
    else:
        print(f"{header_path}  {header_lang}  Description")
        print("-" * (max_path_length + max_lang_length + 15))
    for summary in summaries:
        description_lines = textwrap.wrap(summary.summary, width=max(wrap, 20)) or [""]
        first_line, *rest = description_lines
        language = summary.language.ljust(max_lang_length)
        prefix = summary.relative_path.ljust(max_path_length)
        if show_source:
            source = summary.description_source.ljust(max_source_length)
            print(f"{prefix}  {language}  {source}  {first_line}")
            spacer = f"{' ' * max_path_length}  {' ' * max_lang_length}  {' ' * max_source_length}  "
        else:
            print(f"{prefix}  {language}  {first_line}")
            spacer = f"{' ' * max_path_length}  {' ' * max_lang_length}  "
        for line in rest:
            print(f"{spacer}{line}")
        if show_details and summary.detail:
            detail_lines = textwrap.wrap(summary.detail, width=max(wrap, 20))
            for line in detail_lines:
                print(f"{spacer}{line}")


def _print_json(summaries: list[ToolSummary]) -> None:
    import json

    payload = [summary.as_dict() for summary in summaries]
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    summaries = collect_tool_summaries(
        args.root.resolve(), include_non_python=args.include_non_python
    )
    filtered = _filter_summaries(summaries, args.filter)
    if args.format == "json":
        _print_json(filtered)
    else:
        _print_table(
            filtered,
            wrap=args.wrap,
            show_details=args.show_details,
            show_source=args.show_source,
        )


if __name__ == "__main__":
    main()
