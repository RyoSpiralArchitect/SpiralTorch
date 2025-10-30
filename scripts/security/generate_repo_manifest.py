#!/usr/bin/env python3
"""Produce a signed-ready manifest of the SpiralTorch repository and enforce AGPL compliance invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    import tomli as tomllib  # type: ignore[assignment]


DEFAULT_CANONICAL_LICENSE = "LICENSE .txt"
REQUIRED_LICENSE_TOKEN = "AGPL-3.0-or-later"


@dataclass
class FileEntry:
    path: Path
    sha256: str
    sha512: str
    size: int

    @classmethod
    def from_path(cls, path: Path) -> "FileEntry":
        return cls(path=path, sha256=digest(path, "sha256"), sha512=digest(path, "sha512"), size=path.stat().st_size)


@dataclass
class PackageRecord:
    manifest: Path
    name: str
    version: str | None
    license_expression: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root to scan. Defaults to the CWD.")
    parser.add_argument(
        "--canonical-license",
        type=Path,
        default=None,
        help="Path to the canonical AGPL license text. Defaults to 'LICENSE .txt' under the repo root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional destination for the manifest JSON. Defaults to <repo-root>/spiraltorch-repo-license-manifest.json.",
    )
    parser.add_argument(
        "--allow-untracked",
        action="store_true",
        help="Allow untracked files in the working tree. By default the presence of untracked files aborts generation.",
    )
    return parser.parse_args()


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def git_ls_files(root: Path) -> Iterable[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    entries = result.stdout.decode("utf-8").split("\0")
    for entry in entries:
        if not entry:
            continue
        yield Path(entry)


def ensure_clean_tree(root: Path, allow_untracked: bool) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    lines = [line for line in result.stdout.decode("utf-8").splitlines() if line]
    if not lines:
        return
    untracked = [line for line in lines if line.startswith("??")]
    others = [line for line in lines if not line.startswith("??")]
    if others:
        raise SystemExit("Working tree has modifications. Commit or stash changes before generating the manifest.")
    if untracked and not allow_untracked:
        raise SystemExit("Untracked files detected. Commit, clean, or re-run with --allow-untracked.")


def validate_canonical_license(license_path: Path, relative_path: Path) -> dict[str, str | int]:
    if not license_path.is_file():
        raise SystemExit(f"Canonical AGPL license not found at {license_path}")
    content = license_path.read_text(encoding="utf-8", errors="ignore")
    if REQUIRED_LICENSE_TOKEN not in content:
        raise SystemExit(
            f"Canonical license file at {license_path} does not reference {REQUIRED_LICENSE_TOKEN}."
        )
    return {
        "path": relative_path.as_posix(),
        "sha256": digest(license_path, "sha256"),
        "sha512": digest(license_path, "sha512"),
        "size": license_path.stat().st_size,
    }


def validate_notice(root: Path) -> None:
    notice_path = root / "NOTICE"
    if not notice_path.is_file():
        raise SystemExit("NOTICE file is missing from the repository root.")
    notice_text = notice_path.read_text(encoding="utf-8", errors="ignore")
    if "AGPL" not in notice_text:
        raise SystemExit("NOTICE file must explicitly reference the AGPL obligations.")


def record_cargo_packages(root: Path, manifest_path: Path) -> PackageRecord | None:
    text = manifest_path.read_text(encoding="utf-8")
    data = tomllib.loads(text)
    package = data.get("package")
    workspace_package = data.get("workspace", {}).get("package")
    if not package and not workspace_package:
        return None

    pkg = package or workspace_package or {}
    license_field = pkg.get("license")
    license_file = pkg.get("license-file")
    if license_field:
        if REQUIRED_LICENSE_TOKEN not in license_field:
            raise SystemExit(
                f"{manifest_path}: license expression '{license_field}' does not contain {REQUIRED_LICENSE_TOKEN}."
            )
        license_expression = license_field
    elif license_file:
        license_file_path = (manifest_path.parent / license_file).resolve()
        if not license_file_path.is_file():
            raise SystemExit(f"{manifest_path}: referenced license file '{license_file}' is missing.")
        embedded = license_file_path.read_text(encoding="utf-8", errors="ignore")
        if REQUIRED_LICENSE_TOKEN not in embedded:
            raise SystemExit(
                f"{manifest_path}: referenced license file '{license_file}' does not mention {REQUIRED_LICENSE_TOKEN}."
            )
        license_expression = f"file:{license_file}"
    else:
        raise SystemExit(f"{manifest_path}: package section is missing an AGPL license declaration.")

    return PackageRecord(
        manifest=manifest_path.relative_to(root),
        name=pkg.get("name", ""),
        version=pkg.get("version"),
        license_expression=license_expression,
    )


def record_pyproject(root: Path, manifest_path: Path) -> PackageRecord | None:
    text = manifest_path.read_text(encoding="utf-8")
    data = tomllib.loads(text)
    project = data.get("project")
    if not project:
        return None

    license_field = project.get("license")
    license_expression: str | None = None
    if isinstance(license_field, dict):
        text_value = license_field.get("text")
        if text_value:
            license_expression = text_value
    elif isinstance(license_field, str):
        license_expression = license_field

    if not license_expression:
        raise SystemExit(f"{manifest_path}: project license metadata is missing or malformed.")

    if REQUIRED_LICENSE_TOKEN not in license_expression:
        raise SystemExit(
            f"{manifest_path}: project license metadata '{license_expression}' does not contain {REQUIRED_LICENSE_TOKEN}."
        )

    return PackageRecord(
        manifest=manifest_path.relative_to(root),
        name=str(project.get("name", "")),
        version=str(project.get("version")) if project.get("version") else None,
        license_expression=license_expression,
    )


def gather_compliance_metadata(root: Path, tracked_files: Iterable[Path]) -> dict[str, list[dict[str, str]]]:
    cargo_records: list[dict[str, str]] = []
    python_records: list[dict[str, str]] = []

    for relative in tracked_files:
        path = root / relative
        if path.name == "Cargo.toml":
            record = record_cargo_packages(root, path)
            if record:
                cargo_records.append(
                    {
                        "manifest": str(record.manifest),
                        "name": record.name,
                        "version": record.version or "",
                        "license": record.license_expression,
                    }
                )
        elif path.name == "pyproject.toml":
            record = record_pyproject(root, path)
            if record:
                python_records.append(
                    {
                        "manifest": str(record.manifest),
                        "name": record.name,
                        "version": record.version or "",
                        "license": record.license_expression,
                    }
                )

    return {
        "cargo": sorted(cargo_records, key=lambda entry: entry["manifest"]),
        "python": sorted(python_records, key=lambda entry: entry["manifest"]),
    }


def build_manifest(root: Path, tracked_files: list[Path], canonical_license: dict[str, str | int]) -> dict:
    file_entries = []
    for relative in tracked_files:
        full_path = root / relative
        if not full_path.is_file():
            continue
        entry = FileEntry.from_path(full_path)
        file_entries.append(
            {
                "path": str(relative.as_posix()),
                "size": entry.size,
                "sha256": entry.sha256,
                "sha512": entry.sha512,
            }
        )

    metadata = gather_compliance_metadata(root, tracked_files)

    return {
        "schema": "https://spiraltorch.org/security/repo-license-manifest/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": root.resolve().as_posix(),
        "canonical_license": canonical_license,
        "files": sorted(file_entries, key=lambda item: item["path"]),
        "compliance": metadata,
    }


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if not (repo_root / ".git").exists():
        raise SystemExit(f"{repo_root} does not appear to be a Git repository (missing .git directory).")

    ensure_clean_tree(repo_root, allow_untracked=args.allow_untracked)

    canonical_path = args.canonical_license or repo_root / DEFAULT_CANONICAL_LICENSE
    if not canonical_path.is_absolute():
        canonical_path = (repo_root / canonical_path).resolve()
    try:
        relative_canonical = canonical_path.relative_to(repo_root)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise SystemExit("Canonical license must reside inside the repository root.") from exc

    canonical_metadata = validate_canonical_license(canonical_path, relative_canonical)

    validate_notice(repo_root)

    tracked_files = list(git_ls_files(repo_root))
    relative_str = relative_canonical.as_posix()
    if relative_str not in {path.as_posix() for path in tracked_files}:
        raise SystemExit("Canonical license file is not tracked by Git.")

    manifest = build_manifest(repo_root, tracked_files, canonical_metadata)

    output_path = args.output or repo_root / "spiraltorch-repo-license-manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote repository manifest to {output_path}")


if __name__ == "__main__":
    main()
