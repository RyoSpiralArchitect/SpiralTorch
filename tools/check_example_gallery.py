#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileRef:
    line: int
    path: str


_FILE_RE = re.compile(r"\*\*File:\*\*\s*`([^`]+)`")


def _parse_refs(markdown: str) -> list[FileRef]:
    refs: list[FileRef] = []
    for lineno, line in enumerate(markdown.splitlines(), start=1):
        match = _FILE_RE.search(line)
        if not match:
            continue
        refs.append(FileRef(line=lineno, path=match.group(1).strip()))
    return refs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate that docs/example-gallery.md only references files that exist."
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=Path("docs/example-gallery.md"),
        help="Path to the example gallery markdown file",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root (paths are resolved relative to this)",
    )
    args = parser.parse_args()

    doc_path: Path = args.doc
    if not doc_path.exists():
        print(f"[example-gallery] missing doc: {doc_path}", file=sys.stderr)
        return 2

    repo_root = args.repo_root.resolve()
    markdown = doc_path.read_text(encoding="utf-8")
    refs = _parse_refs(markdown)
    if not refs:
        print("[example-gallery] no **File:** references found", file=sys.stderr)
        return 1

    missing: list[FileRef] = []
    for ref in refs:
        rel = Path(ref.path)
        candidate = rel if rel.is_absolute() else repo_root / rel
        if not candidate.exists():
            missing.append(ref)

    if missing:
        for ref in missing:
            print(
                f"[example-gallery] {doc_path}:{ref.line} missing file: {ref.path}",
                file=sys.stderr,
            )
        print(f"[example-gallery] FAILED ({len(missing)} missing)", file=sys.stderr)
        return 1

    print(f"[example-gallery] OK ({len(refs)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

