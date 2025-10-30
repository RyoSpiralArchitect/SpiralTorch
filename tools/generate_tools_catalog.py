"""Generate a Markdown or JSON catalog for the tools directory.

This script builds upon :mod:`tool_summary` to provide a more featureful
overview of the utilities available under ``tools/``.  In addition to printing a
table to stdout it can also emit machine readable JSON so that other automation
can ingest the catalog.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from tool_summary import ToolSummary, collect_tool_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Root directory to scan. Defaults to the directory of this script.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format for the catalog.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="If provided, write the catalog to this file instead of stdout.",
    )
    parser.add_argument(
        "--include-non-python",
        action="store_true",
        help="Include non-Python files in the catalog with placeholder summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    summaries = collect_tool_summaries(root, include_non_python=args.include_non_python)

    if args.format == "json":
        content = _format_json(summaries)
    else:
        content = _format_markdown(root.name, summaries)

    if args.output:
        args.output.write_text(content, encoding="utf8")
    else:
        print(content)


def _format_json(summaries: Iterable[ToolSummary]) -> str:
    data = [
        {
            "path": summary.relative_path,
            "summary": summary.summary,
        }
        for summary in summaries
    ]
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _format_markdown(title: str, summaries: Iterable[ToolSummary]) -> str:
    summaries = list(summaries)
    if not summaries:
        return f"# {title}\n\n_No tools were discovered._\n"

    lines = [f"# {title} tools catalog", "", "| Path | Description |", "| --- | --- |"]
    for summary in summaries:
        lines.append(f"| `{summary.relative_path}` | {summary.summary} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

