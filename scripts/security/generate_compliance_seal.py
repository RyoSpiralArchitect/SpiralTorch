#!/usr/bin/env python3
"""Generate a compliance seal tying a repository manifest to a specific commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_LICENSE_TOKEN = "AGPL-3.0-or-later"
SCHEMA = "https://spiraltorch.org/security/compliance-seal/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the spiraltorch-repo-license-manifest.json file to seal.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve manifest and determine the current commit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination for the generated seal JSON. Defaults to <manifest-dir>/spiraltorch-compliance-seal.json.",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default=None,
        help="Explicit commit hash to embed. Defaults to the HEAD commit of --repo-root.",
    )
    return parser.parse_args()


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_commit(repo_root: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    commit = result.stdout.decode("utf-8").strip()
    if not commit:
        raise SystemExit("Unable to determine the repository HEAD commit.")
    return commit


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical = manifest_data.get("canonical_license") or {}
    canonical_path = canonical.get("path")
    canonical_sha256 = canonical.get("sha256")
    canonical_sha512 = canonical.get("sha512")
    if not canonical_path or not canonical_sha256 or not canonical_sha512:
        raise SystemExit("Manifest is missing canonical license metadata required for the seal.")

    commit = resolve_commit(args.repo_root, args.commit)
    manifest_sha256 = digest(manifest_path, "sha256")
    manifest_sha512 = digest(manifest_path, "sha512")

    agpl_clause = (
        "This seal affirms that SpiralTorch remains governed by the AGPL-3.0-or-later; "
        "redistributors must provide complete corresponding source and preserve this manifest and license."
    )
    if REQUIRED_LICENSE_TOKEN not in agpl_clause:
        raise SystemExit("Compliance seal clause must reference the AGPL obligations explicitly.")

    seal = {
        "schema": SCHEMA,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "commit": commit,
        "manifest": {
            "path": manifest_path.name,
            "sha256": manifest_sha256,
            "sha512": manifest_sha512,
        },
        "canonical_license": {
            "path": canonical_path,
            "sha256": canonical_sha256,
            "sha512": canonical_sha512,
        },
        "compliance_checks": manifest_data.get("compliance", {}),
        "agpl_clause": agpl_clause,
        "required_files": [canonical_path, manifest_path.name, "NOTICE"],
    }

    output_path = args.output or manifest_path.parent / "spiraltorch-compliance-seal.json"
    output_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote compliance seal to {output_path}")


if __name__ == "__main__":
    main()
