"""Generate a Markdown or JSON catalog for the tools directory.

This script builds upon :mod:`tool_summary` to provide a more featureful
overview of the utilities available under ``tools/``.  In addition to printing a
table to stdout it can also emit machine readable JSON so that other automation
can ingest the catalog.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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
    parser.add_argument(
        "--show-tags",
        action="store_true",
        help="Include derived tags for each tool in the markdown output.",
    )
    parser.add_argument(
        "--show-imports",
        action="store_true",
        help="Display the discovered import set for Python tools in markdown output.",
    )
    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Show line counts and definition metrics in the markdown table.",
    )
    parser.add_argument(
        "--show-cli",
        action="store_true",
        help="Include CLI framework and argument columns in the markdown table.",
    )
    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Include description source, TODO counts, and last modified metadata.",
    )
    parser.add_argument(
        "--filter-language",
        action="append",
        default=[],
        metavar="LANG",
        help="Only include tools that match the given language. Can be used multiple times.",
    )
    parser.add_argument(
        "--filter-tag",
        action="append",
        default=[],
        metavar="TAG",
        help="Only include tools that contain all of the specified tags.",
    )
    parser.add_argument(
        "--filter-framework",
        action="append",
        default=[],
        metavar="FRAMEWORK",
        help="Only include tools that use the specified CLI frameworks.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Prepend an aggregate summary section to the markdown output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    summaries = collect_tool_summaries(root, include_non_python=args.include_non_python)
    summaries = _apply_filters(
        summaries,
        languages=set(lang.lower() for lang in args.filter_language),
        tags=set(tag.lower() for tag in args.filter_tag),
        frameworks=set(framework.lower() for framework in args.filter_framework),
    )

    if args.format == "json":
        content = _format_json(summaries)
    else:
        content = _format_markdown(
            root.name,
            summaries,
            group_by=args.group_by,
            include_details=args.show_details,
            show_tags=args.show_tags,
            show_imports=args.show_imports,
            show_metrics=args.show_metrics,
            show_cli=args.show_cli,
            show_metadata=args.show_metadata,
            include_summary=args.summary,
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
    show_tags: bool,
    show_imports: bool,
    show_metrics: bool,
    show_cli: bool,
    show_metadata: bool,
    include_summary: bool,
) -> str:
    summaries = list(summaries)
    if not summaries:
        return f"# {title}\n\n_No tools were discovered._\n"

    lines = [f"# {title} tools catalog", ""]
    if include_summary:
        lines.extend(_format_summary_section(summaries))
        lines.append("")
    if group_by == "directory":
        grouped: dict[str, list[ToolSummary]] = defaultdict(list)
        for summary in summaries:
            directory = str(Path(summary.relative_path).parent) or "."
            grouped[directory].append(summary)
        for directory in sorted(grouped):
            lines.append(f"## `{directory}`")
            lines.extend(
                _format_markdown_table(
                    grouped[directory],
                    include_details=include_details,
                    show_tags=show_tags,
                    show_imports=show_imports,
                    show_metrics=show_metrics,
                    show_cli=show_cli,
                    show_metadata=show_metadata,
                )
            )
            lines.append("")
    else:
        lines.extend(
            _format_markdown_table(
                summaries,
                include_details=include_details,
                show_tags=show_tags,
                show_imports=show_imports,
                show_metrics=show_metrics,
                show_cli=show_cli,
                show_metadata=show_metadata,
            )
        )

    return "\n".join(line for line in lines if line is not None).rstrip() + "\n"


def _format_markdown_table(
    summaries: Iterable[ToolSummary], *, include_details: bool, show_tags: bool, show_imports: bool, show_metrics: bool, show_cli: bool, show_metadata: bool
) -> list[str]:
    header = ["Path", "Language", "Description"]
    if show_metrics:
        header.extend(["Lines", "Functions", "Classes"])
    if show_tags:
        header.append("Tags")
    if show_imports:
        header.append("Imports")
    if show_cli:
        header.extend(["CLI frameworks", "CLI args", "Has __main__"])
    if show_metadata:
        header.extend(["Source", "TODOs", "Last modified"])

    rows = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for summary in sorted(summaries, key=lambda s: s.relative_path):
        description = summary.summary
        if include_details and summary.detail:
            detail_html = summary.detail.replace("\n", "<br>")
            description = f"{description}<br><br>{detail_html}"
        cells = [
            f"`{summary.relative_path}`",
            summary.language,
            description,
        ]
        if show_metrics:
            cells.extend(
                [
                    str(summary.lines_of_code),
                    str(summary.function_count),
                    str(summary.class_count),
                ]
            )
        if show_tags:
            tag_display = "<br>".join(summary.tags) if summary.tags else "—"
            cells.append(tag_display)
        if show_imports:
            import_display = "<br>".join(summary.imports) if summary.imports else "—"
            cells.append(import_display)
        if show_cli:
            framework_display = (
                "<br>".join(summary.cli_frameworks) if summary.cli_frameworks else "—"
            )
            cells.append(framework_display)
            cells.append(str(summary.cli_argument_count))
            cells.append("✅" if summary.has_main_guard else "—")
        if show_metadata:
            cells.append(summary.description_source)
            cells.append(str(summary.todo_count))
            cells.append(summary.last_modified or "—")
        rows.append("| " + " | ".join(cells) + " |")
    rows.append("")
    return rows


def _format_summary_section(summaries: Iterable[ToolSummary]) -> list[str]:
    """Generate a high-level summary for the provided *summaries*."""

    summaries = list(summaries)
    language_counts = Counter(summary.language for summary in summaries)
    tag_counts = Counter(tag for summary in summaries for tag in summary.tags)
    framework_counts = Counter(
        framework for summary in summaries for framework in summary.cli_frameworks
    )
    total_loc = sum(summary.lines_of_code for summary in summaries)
    average_loc = total_loc / len(summaries)
    total_todos = sum(summary.todo_count for summary in summaries)
    main_guards = sum(1 for summary in summaries if summary.has_main_guard)

    lines = ["## Overview", ""]
    lines.append(f"* **Tools discovered:** {len(summaries)}")
    lines.append(f"* **Total non-empty lines:** {total_loc}")
    lines.append(f"* **Average non-empty lines:** {average_loc:.1f}")
    if language_counts:
        formatted = ", ".join(f"{lang}: {count}" for lang, count in language_counts.most_common())
        lines.append(f"* **Languages:** {formatted}")
    if tag_counts:
        formatted = ", ".join(
            f"{tag}: {count}" for tag, count in tag_counts.most_common(10)
        )
        lines.append(f"* **Top tags:** {formatted}")
    if framework_counts:
        formatted = ", ".join(
            f"{framework}: {count}" for framework, count in framework_counts.most_common()
        )
        lines.append(f"* **CLI frameworks:** {formatted}")
    if main_guards:
        lines.append(f"* **Modules with __main__ guard:** {main_guards}")
    if total_todos:
        lines.append(f"* **TODO markers detected:** {total_todos}")
    return lines


def _apply_filters(
    summaries: list[ToolSummary], *, languages: set[str], tags: set[str], frameworks: set[str]
) -> list[ToolSummary]:
    """Filter *summaries* by the provided sets."""

    if not languages and not tags and not frameworks:
        return summaries

    filtered: list[ToolSummary] = []
    for summary in summaries:
        if languages and summary.language.lower() not in languages:
            continue
        summary_tags = {tag.lower() for tag in summary.tags}
        if tags and not tags.issubset(summary_tags):
            continue
        summary_frameworks = {framework.lower() for framework in summary.cli_frameworks}
        if frameworks and not frameworks.issubset(summary_frameworks):
            continue
        filtered.append(summary)
    return filtered


if __name__ == "__main__":
    main()

