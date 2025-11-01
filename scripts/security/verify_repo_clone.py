#!/usr/bin/env python3
"""Verify a local repository clone against the signed SpiralTorch license manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable


REQUIRED_LICENSE_TOKEN = "AGPL-3.0-or-later"


SCHEMA = "https://spiraltorch.org/security/compliance-seal/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a trusted spiraltorch-repo-license-manifest.json file.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository clone root to validate. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--require-extra",
        action="append",
        dest="required_paths",
        default=[],
        help="Additional file paths that must exist in the clone. Relative to the repository root.",
    )
    parser.add_argument(
        "--seal",
        type=Path,
        default=None,
        help="Optional compliance seal to enforce commit provenance and manifest digests.",
    )
    return parser.parse_args()


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Required file is missing: {path}")


def current_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - relies on git availability
        raise SystemExit(
            "Unable to resolve HEAD commit for clone; ensure you are inside a git checkout."
        ) from exc
    commit = result.stdout.decode("utf-8").strip()
    if not commit:
        raise SystemExit("Unable to determine HEAD commit for the repository clone.")
    return commit


def validate_seal(seal_data: dict, seal_path: Path, manifest_path: Path, repo_root: Path) -> None:
    if seal_data.get("schema") != SCHEMA:
        raise SystemExit(f"Unsupported compliance seal schema in {seal_path}")

    manifest_info = seal_data.get("manifest") or {}
    expected_sha256 = manifest_info.get("sha256")
    expected_sha512 = manifest_info.get("sha512")
    if not expected_sha256 or not expected_sha512:
        raise SystemExit("Compliance seal is missing manifest digests.")

    actual_sha256 = digest(manifest_path, "sha256")
    actual_sha512 = digest(manifest_path, "sha512")
    if actual_sha256 != expected_sha256 or actual_sha512 != expected_sha512:
        raise SystemExit("Compliance seal does not match the supplied manifest file.")

    canonical = seal_data.get("canonical_license") or {}
    for key in ["path", "sha256", "sha512"]:
        if not canonical.get(key):
            raise SystemExit("Compliance seal is missing canonical license metadata.")

    clause = seal_data.get("agpl_clause", "")
    if REQUIRED_LICENSE_TOKEN not in clause:
        raise SystemExit("Compliance seal clause fails to reference AGPL obligations.")

    repo_head = current_commit(repo_root)
    seal_commit = seal_data.get("commit")
    if not seal_commit:
        raise SystemExit("Compliance seal omitted the source commit hash.")
    if repo_head != seal_commit:
        raise SystemExit(
            "Repository clone is not checked out at the sealed commit."
            " Ensure you are inspecting the official AGPL state before use."
        )

    for required in seal_data.get("required_files", []) or []:
        ensure_exists(repo_root / required)


def validate_clone(manifest: dict, repo_root: Path, required_paths: Iterable[str]) -> None:
    files = manifest.get("files") or []
    if not files:
        raise SystemExit("Manifest contained no file entries to compare.")

    repo_root = repo_root.resolve()
    failures: list[str] = []

    for entry in files:
        relative = entry.get("path")
        if not relative:
            failures.append("Manifest entry missing path field.")
            continue
        sha256 = entry.get("sha256")
        sha512 = entry.get("sha512")
        size = entry.get("size")
        if not all([sha256, sha512, size is not None]):
            failures.append(f"Manifest entry for {relative} is incomplete.")
            continue

        local_path = repo_root / relative
        if not local_path.is_file():
            failures.append(f"Missing tracked file: {relative}")
            continue

        local_sha256 = digest(local_path, "sha256")
        local_sha512 = digest(local_path, "sha512")
        if local_sha256 != sha256:
            failures.append(f"SHA256 mismatch for {relative}")
        if local_sha512 != sha512:
            failures.append(f"SHA512 mismatch for {relative}")

        local_size = local_path.stat().st_size
        if local_size != size:
            failures.append(f"Size mismatch for {relative}: expected {size}, found {local_size}")

    canonical = manifest.get("canonical_license") or {}
    canonical_path = canonical.get("path")
    canonical_sha256 = canonical.get("sha256")
    canonical_sha512 = canonical.get("sha512")
    if not canonical_path or not canonical_sha256 or not canonical_sha512:
        failures.append("Manifest did not include canonical license metadata.")
    else:
        license_path = repo_root / canonical_path
        if not license_path.is_file():
            failures.append(f"Canonical license missing from clone: {canonical_path}")
        else:
            if digest(license_path, "sha256") != canonical_sha256:
                failures.append("Canonical license SHA256 mismatch.")
            if digest(license_path, "sha512") != canonical_sha512:
                failures.append("Canonical license SHA512 mismatch.")
            license_text = license_path.read_text(encoding="utf-8", errors="ignore")
            if REQUIRED_LICENSE_TOKEN not in license_text:
                failures.append("Canonical license file does not reference the AGPL obligations.")

    compliance = manifest.get("compliance") or {}
    for section, entries in compliance.items():
        for entry in entries or []:
            license_expression = entry.get("license", "")
            if REQUIRED_LICENSE_TOKEN not in license_expression:
                failures.append(
                    f"{section} manifest {entry.get('manifest', '<unknown>')} missing AGPL expression: {license_expression}"
                )
            manifest_path = entry.get("manifest")
            if manifest_path:
                local_manifest = repo_root / manifest_path
                if not local_manifest.is_file():
                    failures.append(f"Compliance manifest missing from clone: {manifest_path}")

    for extra in required_paths:
        ensure_exists(repo_root / extra)

    if failures:
        formatted = "\n - ".join([""] + failures)
        raise SystemExit("Repository clone failed license verification:" + formatted)


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.seal:
        seal_path = args.seal.resolve()
        if not seal_path.is_file():
            raise SystemExit(f"Compliance seal not found: {seal_path}")
        seal_data = json.loads(seal_path.read_text(encoding="utf-8"))
        validate_seal(seal_data, seal_path, manifest_path, args.repo_root)
    validate_clone(manifest, args.repo_root, args.required_paths)
    print("Repository clone matches the signed manifest and preserves AGPL declarations.")


if __name__ == "__main__":
    main()
