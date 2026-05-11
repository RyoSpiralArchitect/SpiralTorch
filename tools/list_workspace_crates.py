#!/usr/bin/env python3
"""List SpiralTorch Cargo workspace packages and default-member coverage."""

from __future__ import annotations

import argparse
import json
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
        choices=("table", "json"),
        default="table",
        help="Output format.",
    )
    parser.add_argument(
        "--non-default-only",
        action="store_true",
        help="Only list packages that are workspace members but not default-members.",
    )
    return parser.parse_args()


def load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def collect_workspace_crates(manifest_path: Path) -> list[WorkspaceCrate]:
    root_manifest = load_toml(manifest_path)
    workspace = root_manifest.get("workspace", {})
    root = manifest_path.resolve().parent
    members = workspace.get("members", [])
    default_members = set(workspace.get("default-members", []))

    crates: list[WorkspaceCrate] = []
    for member in members:
        package_manifest = root / member / "Cargo.toml"
        package_data = load_toml(package_manifest)
        package = package_data.get("package", {})
        package_dir = package_manifest.parent
        crates.append(
            WorkspaceCrate(
                name=str(package.get("name", member)),
                path=member,
                version=str(package.get("version", "")),
                default_member=member in default_members,
                description=str(package.get("description", "")),
                tests=sum(1 for _ in package_dir.glob("tests/**/*.rs")),
                examples=sum(1 for _ in package_dir.glob("examples/**/*.rs")),
            )
        )
    return crates


_DESC_MAX = 50  # Maximum characters shown for description in table output


def _truncate(s: str, max_len: int) -> str:
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


def main() -> None:
    args = parse_args()
    crates = collect_workspace_crates(args.manifest_path)
    if args.non_default_only:
        crates = [crate for crate in crates if not crate.default_member]

    if args.format == "json":
        print(json.dumps([asdict(crate) for crate in crates], indent=2))
    else:
        print_table(crates)


if __name__ == "__main__":
    main()
