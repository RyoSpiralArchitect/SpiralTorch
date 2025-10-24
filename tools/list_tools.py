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
import ast
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class ToolSummary:
    """Represents a summarized entry for a tool script."""

    path: Path
    summary: str

    @property
    def relative_path(self) -> str:
        """Return the path relative to the tools directory as a string."""

        return str(self.path)


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


def collect_tool_summaries(
    root: Path, include_non_python: bool = False
) -> list[ToolSummary]:
    """Walk the tree below *root* collecting summaries of Python scripts."""

    summaries: list[ToolSummary] = []
    for path in sorted(_iter_tool_files(root, include_non_python)):
        summaries.append(ToolSummary(path=path.relative_to(root), summary=_summarize(path)))
    return summaries


def _iter_tool_files(root: Path, include_non_python: bool) -> Iterator[Path]:
    """Yield files beneath *root* that should be summarised."""

    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if path.name.endswith(".py"):
            yield path
        elif include_non_python:
            yield path


def _summarize(path: Path) -> str:
    """Generate a human-friendly summary for *path*."""

    if path.suffix == ".py":
        doc = _extract_docstring(path)
        if doc:
            first_line = doc.strip().splitlines()[0].strip()
            if first_line:
                return first_line
        top_comment = _extract_leading_comment(path)
        if top_comment:
            return top_comment
        return "(no docstring found)"

    return "(non-Python file)"


def _extract_docstring(path: Path) -> str | None:
    """Return the module level docstring for the file at *path*."""

    try:
        module = ast.parse(path.read_text(encoding="utf8"))
    except (SyntaxError, UnicodeDecodeError):
        return None
    return ast.get_docstring(module)


def _extract_leading_comment(path: Path) -> str | None:
    """Return the first leading comment in the file, if any."""

    try:
        lines = path.read_text(encoding="utf8").splitlines()
    except UnicodeDecodeError:
        return None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            comment = stripped.lstrip("#").strip()
            if comment:
                return comment
            continue
        break
    return None


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
