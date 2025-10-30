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
import re


LANGUAGE_BY_SUFFIX: dict[str, str] = {
    ".py": "Python",
    ".rs": "Rust",
    ".sh": "Shell",
    ".toml": "TOML",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".json": "JSON",
    ".md": "Markdown",
}


@dataclass
class ToolSummary:
    """Represents a summarized entry for a tool script."""

    path: Path
    summary: str
    detail: str | None = None
    language: str = "Unknown"
    description_source: str = "unknown"

    @property
    def relative_path(self) -> str:
        """Return the path relative to the tools directory as a string."""

        return str(self.path)

    def as_dict(self) -> dict[str, str | None]:
        """Return a JSON-serialisable representation of the summary."""

        return {
            "path": self.relative_path,
            "summary": self.summary,
            "detail": self.detail,
            "language": self.language,
            "description_source": self.description_source,
        }


def collect_tool_summaries(
    root: Path, include_non_python: bool = False
) -> list[ToolSummary]:
    """Walk the tree below *root* collecting summaries of tool scripts."""

    summaries: list[ToolSummary] = []
    for path in sorted(_iter_tool_files(root, include_non_python)):
        summary, detail, source = _summarize(path)
        summaries.append(
            ToolSummary(
                path=path.relative_to(root),
                summary=summary,
                detail=detail,
                language=_detect_language(path),
                description_source=source,
            )
        )
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


def _summarize(path: Path) -> tuple[str, str | None, str]:
    """Generate a human-friendly summary for *path*.

    Returns a tuple of ``(summary, detail, description_source)``.
    """

    if path.suffix == ".py":
        module = _parse_module(path)
        doc = _extract_docstring(module)
        if doc:
            summary, detail = _split_docstring(doc)
            if summary:
                return summary, detail, "docstring"
        assignment = _extract_named_constant(module, {"SUMMARY", "DESCRIPTION"})
        if assignment:
            summary, detail = _split_docstring(assignment)
            return summary, detail, "module constant"
        parser_desc = _extract_argparse_description(module)
        if parser_desc:
            summary, detail = _split_docstring(parser_desc)
            return summary, detail, "argparse description"
        top_comment = _extract_leading_comment(path)
        if top_comment:
            summary, detail = _split_docstring(top_comment)
            return summary, detail, "leading comment"
        return "(no description found)", None, "missing"

    top_comment = _extract_leading_comment(path)
    if top_comment:
        summary, detail = _split_docstring(top_comment)
        return summary or "(non-Python file)", detail, "leading comment"
    return "(non-Python file)", None, "missing"


def _extract_docstring(module: ast.Module | None) -> str | None:
    """Return the module level docstring for *module*."""

    if module is None:
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


def _extract_named_constant(module: ast.Module | None, names: set[str]) -> str | None:
    """Return the value of a module level string constant with one of *names*."""

    if module is None:
        return None

    for node in module.body:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                value = _literal_to_str(node.value)
                if value:
                    return value
    return None


def _extract_argparse_description(module: ast.Module | None) -> str | None:
    """Attempt to find an ``argparse.ArgumentParser`` description string."""

    if module is None:
        return None

    for node in ast.walk(module):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                name = f"{getattr(func.value, 'id', '')}.{func.attr}"
            elif isinstance(func, ast.Name):
                name = func.id
            else:
                name = None

            if name not in {"ArgumentParser", "argparse.ArgumentParser"}:
                continue

            for keyword in node.keywords:
                if keyword.arg == "description":
                    value = _literal_to_str(keyword.value)
                    if value:
                        return value
    return None


def _literal_to_str(node: ast.AST) -> str | None:
    """Return the string representation of *node* if it is a literal."""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, (ast.JoinedStr, ast.BinOp)):
        try:
            value = ast.literal_eval(node)
        except Exception:  # pragma: no cover - best effort only
            return None
        if isinstance(value, str):
            return value
    return None


def _split_docstring(text: str) -> tuple[str, str | None]:
    """Split *text* into a short summary line and a detailed remainder."""

    lines = [line.rstrip() for line in text.strip().splitlines()]
    if not lines:
        return "", None
    summary = lines[0].strip()
    detail_lines = [line for line in lines[1:] if line.strip()]
    detail = "\n".join(detail_lines) if detail_lines else None
    return summary, detail


def _detect_language(path: Path) -> str:
    """Infer the language of *path* from its suffix or shebang."""

    language = LANGUAGE_BY_SUFFIX.get(path.suffix.lower())
    if language:
        return language

    try:
        first_line = path.read_text(encoding="utf8").splitlines()[0]
    except (UnicodeDecodeError, IndexError):
        return "Unknown"

    match = re.match(r"#!\s*/usr/bin/env\s+(\w+)", first_line)
    if match:
        shebang_lang = match.group(1).lower()
        if shebang_lang.startswith("python"):
            return "Python"
        if shebang_lang == "bash":
            return "Shell"
    return "Unknown"


def _parse_module(path: Path) -> ast.Module | None:
    """Parse *path* as a Python module, returning ``None`` on failure."""

    try:
        return ast.parse(path.read_text(encoding="utf8"))
    except (SyntaxError, UnicodeDecodeError):
        return None


__all__ = ["ToolSummary", "collect_tool_summaries"]

