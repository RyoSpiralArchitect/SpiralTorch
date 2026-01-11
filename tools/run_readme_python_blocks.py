#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PythonBlock:
    start_line: int
    language: str
    code: str


def _parse_python_blocks(markdown: str) -> list[PythonBlock]:
    blocks: list[PythonBlock] = []
    fence: str | None = None
    language: str | None = None
    start_line: int | None = None
    buf: list[str] = []

    for lineno, raw_line in enumerate(markdown.splitlines(), start=1):
        line = raw_line.rstrip("\n")
        stripped = line.lstrip()

        if fence is None:
            if stripped.startswith("```"):
                fence = "```"
                language = stripped[3:].strip().lower()
                start_line = lineno + 1
                buf = []
            continue

        if stripped.startswith(fence):
            lang = (language or "").strip()
            if lang in {"python", "py", "python3"}:
                blocks.append(
                    PythonBlock(
                        start_line=start_line or 1,
                        language=lang,
                        code="\n".join(buf).rstrip() + "\n",
                    )
                )
            fence = None
            language = None
            start_line = None
            buf = []
            continue

        buf.append(line)

    return blocks


def _should_skip(code: str) -> bool:
    markers = ("ST_SKIP", "DOCS_SKIP", "doctest: +SKIP")
    return any(marker in code for marker in markers)


def _run_block(
    block: PythonBlock,
    *,
    python: str,
    cwd: Path,
    env: dict[str, str],
    index: int,
    total: int,
) -> None:
    header = f"[README python] block {index}/{total} (starts at README.md:{block.start_line})"
    print(header, flush=True)
    if _should_skip(block.code):
        print(f"{header} -> skipped", flush=True)
        return

    with tempfile.TemporaryDirectory(prefix="st-readme-") as tmpdir:
        path = Path(tmpdir) / f"readme_block_{index}.py"
        path.write_text(block.code, encoding="utf-8")
        subprocess.run(
            [python, str(path)],
            cwd=str(cwd),
            env=env,
            check=True,
            text=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute all ```python fenced blocks in README.md (each in a fresh process)."
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("README.md"),
        help="Path to README.md",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use (default: current interpreter)",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=Path("."),
        help="Working directory for executing blocks (default: repo root)",
    )
    args = parser.parse_args()

    readme_path: Path = args.readme
    if not readme_path.exists():
        print(f"README not found: {readme_path}", file=sys.stderr)
        return 2

    markdown = readme_path.read_text(encoding="utf-8")
    blocks = _parse_python_blocks(markdown)
    if not blocks:
        print("No python blocks found.", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")

    cwd = args.cwd.resolve()
    for idx, block in enumerate(blocks, start=1):
        _run_block(
            block,
            python=args.python,
            cwd=cwd,
            env=env,
            index=idx,
            total=len(blocks),
        )

    print(f"[README python] OK ({len(blocks)} blocks)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

