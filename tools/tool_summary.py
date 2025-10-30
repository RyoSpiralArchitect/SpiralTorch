"""Shared utilities for summarizing scripts in the :mod:`tools` directory.

The tools folder has been steadily growing and we now maintain common helper
functions for discovering scripts and extracting their brief description.  By
centralising the logic in this module we can easily build new commands on top
of the same foundation without duplicating code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import ast


@dataclass
class ToolSummary:
    """Represents a summarized entry for a tool script."""

    path: Path
    summary: str

    @property
    def relative_path(self) -> str:
        """Return the path relative to the tools directory as a string."""

        return str(self.path)


def collect_tool_summaries(
    root: Path, include_non_python: bool = False
) -> list[ToolSummary]:
    """Walk the tree below *root* collecting summaries of tool scripts."""

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


__all__ = ["ToolSummary", "collect_tool_summaries"]

