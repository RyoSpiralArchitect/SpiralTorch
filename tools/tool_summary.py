"""Shared utilities for summarizing scripts in the :mod:`tools` directory.

The tools folder has been steadily growing and we now maintain common helper
functions for discovering scripts and extracting their brief description.  By
centralising the logic in this module we can easily build new commands on top
of the same foundation without duplicating code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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
    lines_of_code: int = 0
    function_count: int = 0
    class_count: int = 0
    imports: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    cli_frameworks: tuple[str, ...] = ()
    cli_argument_count: int = 0
    has_main_guard: bool = False
    todo_count: int = 0
    last_modified: str | None = None

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
            "lines_of_code": self.lines_of_code,
            "function_count": self.function_count,
            "class_count": self.class_count,
            "imports": list(self.imports),
            "tags": list(self.tags),
            "cli_frameworks": list(self.cli_frameworks),
            "cli_argument_count": self.cli_argument_count,
            "has_main_guard": self.has_main_guard,
            "todo_count": self.todo_count,
            "last_modified": self.last_modified,
        }


def collect_tool_summaries(
    root: Path, include_non_python: bool = False
) -> list[ToolSummary]:
    """Walk the tree below *root* collecting summaries of tool scripts."""

    summaries: list[ToolSummary] = []
    for path in sorted(_iter_tool_files(root, include_non_python)):
        text = _read_file(path)
        module = _parse_module(path) if path.suffix == ".py" else None
        summary, detail, source = _summarize(path, module=module)
        imports = tuple(_extract_imports(module))
        functions, classes = _count_definitions(module)
        cli_frameworks = tuple(_detect_cli_frameworks(module, imports))
        cli_argument_count = _count_cli_arguments(module)
        has_main_guard = _module_has_main_guard(module)
        todo_count = _count_todos(text)
        tags = tuple(
            _infer_tags(
                path,
                module,
                imports,
                text,
                cli_frameworks,
                has_main_guard,
                todo_count,
                cli_argument_count,
            )
        )
        summaries.append(
            ToolSummary(
                path=path.relative_to(root),
                summary=summary,
                detail=detail,
                language=_detect_language(path),
                description_source=source,
                lines_of_code=_count_lines_of_code(path, text),
                function_count=functions,
                class_count=classes,
                imports=imports,
                tags=tags,
                cli_frameworks=cli_frameworks,
                cli_argument_count=cli_argument_count,
                has_main_guard=has_main_guard,
                todo_count=todo_count,
                last_modified=_last_modified_timestamp(path),
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


def _summarize(
    path: Path, *, module: ast.Module | None = None
) -> tuple[str, str | None, str]:
    """Generate a human-friendly summary for *path*.

    Returns a tuple of ``(summary, detail, description_source)``.
    """

    if path.suffix == ".py":
        if module is None:
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

    text = _read_file(path)
    if text is None:
        return None

    lines = text.splitlines()
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


def _extract_imports(module: ast.Module | None) -> list[str]:
    """Return a sorted list of imported top-level modules for *module*."""

    if module is None:
        return []

    modules: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split(".")[0])
    return sorted(modules)


def _count_definitions(module: ast.Module | None) -> tuple[int, int]:
    """Return ``(function_count, class_count)`` for *module*."""

    if module is None:
        return 0, 0

    function_count = 0
    class_count = 0
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_count += 1
        elif isinstance(node, ast.ClassDef):
            class_count += 1
    return function_count, class_count


def _count_lines_of_code(path: Path, text: str | None) -> int:
    """Return a rough non-empty line count for *path*."""

    if text is None:
        return 0

    count = 0
    comment_prefix = "#" if path.suffix == ".py" else None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if comment_prefix and stripped.startswith(comment_prefix):
            continue
        count += 1
    return count


def _infer_tags(
    path: Path,
    module: ast.Module | None,
    imports: tuple[str, ...] | list[str],
    text: str | None,
    cli_frameworks: tuple[str, ...] | list[str],
    has_main_guard: bool,
    todo_count: int,
    cli_argument_count: int,
) -> list[str]:
    """Heuristically derive descriptive tags for *path*."""

    tags: set[str] = set()
    suffix = path.suffix.lower()

    if suffix == ".py":
        normalized_imports = {name.lower() for name in imports}
        if cli_frameworks:
            tags.add("cli")
            tags.update(f"cli:{framework}" for framework in cli_frameworks)
        elif any(
            name in normalized_imports for name in {"argparse", "click", "typer", "fire"}
        ):
            tags.add("cli")
        if any(name in imports for name in {"pytest", "unittest"}):
            tags.add("test")
        if module is not None:
            async_functions = any(
                isinstance(node, ast.AsyncFunctionDef) for node in ast.walk(module)
            )
            if async_functions:
                tags.add("async")
        if has_main_guard:
            tags.add("entrypoint")
        if cli_argument_count >= 10:
            tags.add("cli:large")
        elif cli_argument_count >= 5:
            tags.add("cli:medium")
    elif suffix in {".sh", ".bash"}:
        tags.add("shell")
    elif suffix in {".toml", ".yaml", ".yml", ".json"}:
        tags.add("config")

    if todo_count:
        tags.add("todo")
    if path.name.startswith("test_") or path.name.endswith("_test.py"):
        tags.add("test")

    return sorted(tags)


def _is_main_guard(test: ast.AST) -> bool:
    """Return ``True`` if *test* matches ``__name__ == '__main__'``."""

    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    left = test.left
    right = test.comparators[0]
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    )


def _read_file(path: Path) -> str | None:
    """Return the contents of *path* as UTF-8 text if possible."""

    try:
        return path.read_text(encoding="utf8")
    except UnicodeDecodeError:
        return None


def _detect_cli_frameworks(
    module: ast.Module | None, imports: tuple[str, ...] | list[str]
) -> list[str]:
    """Return a sorted list of CLI frameworks heuristically detected."""

    frameworks: set[str] = set()
    normalized_imports = {name.lower() for name in imports}
    mapping = {
        "argparse": "argparse",
        "click": "click",
        "typer": "typer",
        "fire": "fire",
    }
    for name, label in mapping.items():
        if name in normalized_imports:
            frameworks.add(label)

    if module is not None:
        for node in ast.walk(module):
            qualname = _qualname(node) if isinstance(node, ast.Call) else None
            if qualname is None and isinstance(node, ast.Attribute):
                qualname = _qualname(node)
            if not qualname:
                continue
            root = qualname.split(".")[0]
            if root in {"click", "typer", "fire"}:
                frameworks.add(root)
    return sorted(frameworks)


def _count_cli_arguments(module: ast.Module | None) -> int:
    """Estimate the number of CLI arguments/options defined in *module*."""

    if module is None:
        return 0

    count = 0
    for node in ast.walk(module):
        if isinstance(node, ast.Call):
            qualname = _qualname(node.func)
            if not qualname:
                continue
            qualname_lower = qualname.lower()
            if qualname_lower.endswith("add_argument") or qualname_lower.endswith(
                "add_option"
            ):
                count += 1
            elif qualname_lower.endswith("add_argument_group"):
                count += 1
            elif qualname_lower.endswith(".option") or qualname_lower.endswith(
                ".argument"
            ):
                count += 1
        elif isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                qualname = _qualname(decorator)
                if not qualname:
                    continue
                qualname_lower = qualname.lower()
                if qualname_lower.endswith(".option") or qualname_lower.endswith(
                    ".argument"
                ):
                    count += 1
    return count


def _module_has_main_guard(module: ast.Module | None) -> bool:
    """Return ``True`` if the module defines an ``if __name__ == '__main__'`` guard."""

    if module is None:
        return False
    for node in module.body:
        if isinstance(node, ast.If) and _is_main_guard(node.test):
            return True
    return False


def _count_todos(text: str | None) -> int:
    """Count TODO-like markers in *text*."""

    if not text:
        return 0
    pattern = re.compile(r"\b(TODO|FIXME|XXX)\b")
    return sum(1 for _ in pattern.finditer(text))


def _qualname(node: ast.AST | None) -> str | None:
    """Return a dotted path representation for *node* if possible."""

    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualname(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    if isinstance(node, ast.Call):
        return _qualname(node.func)
    return None


def _last_modified_timestamp(path: Path) -> str | None:
    """Return the ISO formatted last modified timestamp for *path*."""

    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


__all__ = ["ToolSummary", "collect_tool_summaries"]

