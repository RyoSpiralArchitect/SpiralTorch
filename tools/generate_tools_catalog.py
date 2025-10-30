"""Generate a Markdown or JSON catalog for the tools directory.

This script builds upon :mod:`tool_summary` to provide a more featureful
overview of the utilities available under ``tools/``.  In addition to printing a
table to stdout it can also emit machine readable JSON so that other automation
can ingest the catalog.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
    parser.add_argument(
        "--group-by",
        choices=("flat", "directory"),
        default="flat",
        help="Group the markdown catalog by directory for easier browsing.",
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Include the longer detail text in the markdown output when present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    summaries = collect_tool_summaries(root, include_non_python=args.include_non_python)

    if args.format == "json":
        content = _format_json(summaries)
    else:
        content = _format_markdown(
            root.name,
            summaries,
            group_by=args.group_by,
            include_details=args.show_details,
        )

    if args.output:
        args.output.write_text(content, encoding="utf8")
    else:
        print(content)


def _format_json(summaries: Iterable[ToolSummary]) -> str:
    data = [summary.as_dict() for summary in summaries]
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _format_markdown(
    title: str,
    summaries: Iterable[ToolSummary],
    *,
    group_by: str,
    include_details: bool,
) -> str:
    summaries = list(summaries)
    if not summaries:
        return f"# {title}\n\n_No tools were discovered._\n"

    lines = [f"# {title} tools catalog", ""]
    if group_by == "directory":
        grouped: dict[str, list[ToolSummary]] = defaultdict(list)
        for summary in summaries:
            directory = str(Path(summary.relative_path).parent) or "."
            grouped[directory].append(summary)
        for directory in sorted(grouped):
            lines.append(f"## `{directory}`")
            lines.extend(
                _format_markdown_table(grouped[directory], include_details=include_details)
            )
            lines.append("")
    else:
        lines.extend(_format_markdown_table(summaries, include_details=include_details))

    return "\n".join(line for line in lines if line is not None).rstrip() + "\n"


def _format_markdown_table(
    summaries: Iterable[ToolSummary], *, include_details: bool
) -> list[str]:
    rows = ["| Path | Language | Description |", "| --- | --- | --- |"]
    for summary in sorted(summaries, key=lambda s: s.relative_path):
        description = summary.summary
        if include_details and summary.detail:
            detail_html = summary.detail.replace("\n", "<br>")
            description = f"{description}<br><br>{detail_html}"
        rows.append(
            f"| `{summary.relative_path}` | {summary.language} | {description} |"
        )
    rows.append("")
    return rows


if __name__ == "__main__":
    main()

