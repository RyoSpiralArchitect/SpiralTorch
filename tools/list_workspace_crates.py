#!/usr/bin/env python3
"""List SpiralTorch Cargo workspace packages and default-member coverage."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[assignment]


@dataclass(frozen=True)
class WorkspaceCrate:
    name: str
    path: str
    version: str
    default_member: bool
    description: str
    tests: int
    examples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("Cargo.toml"),
        help="Path to the workspace Cargo.toml. Defaults to ./Cargo.toml.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json", "markdown"),
        default="table",
        help="Output format.",
    )
    parser.add_argument(
        "--non-default-only",
        action="store_true",
        help="Only list packages that are workspace members but not default-members.",
    )
    doc_group = parser.add_mutually_exclusive_group()
    doc_group.add_argument(
        "--check-doc",
        type=Path,
        metavar="PATH",
        help="Fail if PATH's generated workspace inventory is out of date.",
    )
    doc_group.add_argument(
        "--write-doc",
        type=Path,
        metavar="PATH",
        help="Refresh PATH's generated workspace inventory in place.",
    )
    return parser.parse_args()


def load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_cargo_metadata(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    try:
        completed = subprocess.run(
            [
                "cargo",
                "metadata",
                "--locked",
                "--no-deps",
                "--format-version",
                "1",
                "--manifest-path",
                str(manifest_path),
            ],
            cwd=manifest_path.parent,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError("cargo is required to resolve the workspace inventory") from error
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"cargo metadata failed: {message}")
    try:
        metadata = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"cargo metadata returned invalid JSON: {error}") from error
    if not isinstance(metadata, dict):
        raise RuntimeError("cargo metadata returned a non-object document")
    return metadata


def _member_sort_key(path: str, declared_members: list[str]) -> tuple[int, str]:
    normalized = path.removeprefix("./").rstrip("/")
    for index, pattern in enumerate(declared_members):
        pattern = pattern.removeprefix("./").rstrip("/")
        if normalized == pattern or fnmatch.fnmatchcase(normalized, pattern):
            return index, normalized
    # Cargo can auto-enroll an in-tree path dependency that is not declared.
    return len(declared_members), normalized


def collect_workspace_crates(manifest_path: Path) -> list[WorkspaceCrate]:
    root_manifest = load_toml(manifest_path)
    workspace = root_manifest.get("workspace", {})
    root = manifest_path.resolve().parent
    declared_members = [str(member) for member in workspace.get("members", [])]
    metadata = load_cargo_metadata(manifest_path)
    packages = {
        str(package["id"]): package for package in metadata.get("packages", [])
    }
    workspace_members = [str(member) for member in metadata.get("workspace_members", [])]
    default_members = {
        str(member) for member in metadata.get("workspace_default_members", [])
    }

    resolved: list[tuple[str, dict[str, Any]]] = []
    for member_id in workspace_members:
        package = packages.get(member_id)
        if package is None:
            raise RuntimeError(
                f"cargo metadata omitted workspace package '{member_id}'"
            )
        package_manifest = Path(str(package["manifest_path"])).resolve()
        package_dir = package_manifest.parent
        member_path = Path(os.path.relpath(package_dir, root)).as_posix()
        resolved.append((member_path, package))
    resolved.sort(key=lambda item: _member_sort_key(item[0], declared_members))

    crates: list[WorkspaceCrate] = []
    for member, package in resolved:
        targets = package.get("targets", [])
        crates.append(
            WorkspaceCrate(
                name=str(package.get("name", member)),
                path=member,
                version=str(package.get("version", "")),
                default_member=str(package["id"]) in default_members,
                description=str(package.get("description") or ""),
                tests=sum("test" in target.get("kind", []) for target in targets),
                examples=sum("example" in target.get("kind", []) for target in targets),
            )
        )
    return crates


_DESC_MAX = 50  # Maximum characters shown for description in table output


def _truncate(s: str, max_len: int) -> str:
    if max_len < 3:
        return s[:max_len]
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def print_table(crates: list[WorkspaceCrate]) -> None:
    if not crates:
        print("No workspace crates found.")
        return

    truncated_descs = [_truncate(c.description, _DESC_MAX) for c in crates]
    widths = {
        "name": max(len("Package"), max(len(crate.name) for crate in crates)),
        "path": max(len("Path"), max(len(crate.path) for crate in crates)),
        "version": max(len("Version"), max(len(crate.version) for crate in crates)),
        "default": len("Default"),
        "tests": len("Tests"),
        "examples": len("Examples"),
        "description": max(len("Description"), max(len(d) for d in truncated_descs)),
    }
    print(
        f"{'Package'.ljust(widths['name'])}  "
        f"{'Version'.ljust(widths['version'])}  "
        f"{'Default'.ljust(widths['default'])}  "
        f"{'Tests'.rjust(widths['tests'])}  "
        f"{'Examples'.rjust(widths['examples'])}  "
        f"{'Path'.ljust(widths['path'])}  "
        f"Description"
    )
    print(
        "-"
        * (
            widths["name"]
            + widths["version"]
            + widths["default"]
            + widths["tests"]
            + widths["examples"]
            + widths["path"]
            + widths["description"]
            + 14
        )
    )
    for crate, desc in zip(crates, truncated_descs):
        print(
            f"{crate.name.ljust(widths['name'])}  "
            f"{crate.version.ljust(widths['version'])}  "
            f"{('yes' if crate.default_member else 'no').ljust(widths['default'])}  "
            f"{str(crate.tests).rjust(widths['tests'])}  "
            f"{str(crate.examples).rjust(widths['examples'])}  "
            f"{crate.path.ljust(widths['path'])}  "
            f"{desc}"
        )


_DOC_START = "<!-- workspace-crates:start -->"
_DOC_END = "<!-- workspace-crates:end -->"


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render_markdown(crates: list[WorkspaceCrate]) -> str:
    package_label = "package" if len(crates) == 1 else "packages"
    lines = [
        f"Current inventory: **{len(crates)} Cargo workspace {package_label}.**",
        "",
        "| Package | Path | Default member | Description |",
        "|---|---|---:|---|",
    ]
    for crate in crates:
        lines.append(
            f"| `{_markdown_cell(crate.name)}` | "
            f"`{_markdown_cell(crate.path)}` | "
            f"{'yes' if crate.default_member else 'no'} | "
            f"{_markdown_cell(crate.description)} |"
        )
    return "\n".join(lines)


def refresh_workspace_doc(contents: str, crates: list[WorkspaceCrate]) -> str:
    if contents.count(_DOC_START) != 1 or contents.count(_DOC_END) != 1:
        raise ValueError(
            "workspace inventory document must contain exactly one start and end marker"
        )
    start = contents.index(_DOC_START)
    end_start = contents.index(_DOC_END)
    if end_start < start:
        raise ValueError("workspace inventory end marker appears before start marker")
    end = end_start + len(_DOC_END)
    replacement = (
        f"{_DOC_START}\n"
        "<!-- Generated by tools/list_workspace_crates.py; do not edit by hand. -->\n"
        f"{render_markdown(crates)}\n"
        f"{_DOC_END}"
    )
    return contents[:start] + replacement + contents[end:]


def main() -> None:
    args = parse_args()
    crates = collect_workspace_crates(args.manifest_path)
    if args.non_default_only:
        crates = [crate for crate in crates if not crate.default_member]

    doc_path = args.check_doc or args.write_doc
    if doc_path is not None:
        if args.non_default_only:
            raise SystemExit("--non-default-only cannot be used with document modes")
        contents = doc_path.read_text(encoding="utf-8")
        refreshed = refresh_workspace_doc(contents, crates)
        if args.check_doc:
            if refreshed != contents:
                raise SystemExit(
                    f"{doc_path} is out of date; refresh it with "
                    f"{Path(__file__).as_posix()} --write-doc {doc_path}"
                )
            print(f"{doc_path}: workspace inventory is current ({len(crates)} packages)")
        else:
            doc_path.write_text(refreshed, encoding="utf-8")
            print(f"{doc_path}: refreshed workspace inventory ({len(crates)} packages)")
    elif args.format == "json":
        print(json.dumps([asdict(crate) for crate in crates], indent=2))
    elif args.format == "markdown":
        print(render_markdown(crates))
    else:
        print_table(crates)


if __name__ == "__main__":
    main()
