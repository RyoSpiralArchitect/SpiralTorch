#!/usr/bin/env python3
"""Validate that built wheels embed the canonical AGPL license text and emit a provenance report."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True, help="Directory containing built wheel artifacts.")
    parser.add_argument(
        "--license", type=Path, required=True, help="Path to the canonical AGPL license text that must be embedded."
    )
    parser.add_argument("--tag", required=True, help="Release tag associated with the artifacts.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON report. Defaults to dist/spiraltorch-<tag>-license-report.json.",
    )
    return parser.parse_args()


def compute_digest(data: bytes, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    digest.update(data)
    return digest.hexdigest()


def discover_wheels(dist_dir: Path) -> Iterable[Path]:
    yield from sorted(dist_dir.rglob("*.whl"))


def inspect_wheel(wheel_path: Path, canonical_digest: str) -> dict:
    entries: list[dict] = []
    with zipfile.ZipFile(wheel_path) as zf:
        for name in zf.namelist():
            normalized = name.replace("\\", "/")
            leaf = normalized.split("/")[-1]
            if "license" not in leaf.lower():
                continue
            data = zf.read(name)
            sha256 = compute_digest(data)
            entries.append(
                {
                    "path": normalized,
                    "sha256": sha256,
                    "matches_canonical": sha256 == canonical_digest,
                }
            )
    return {
        "filename": wheel_path.name,
        "embedded_licenses": entries,
        "status": "ok" if any(entry["matches_canonical"] for entry in entries) else "missing",
    }


def main() -> None:
    args = parse_args()
    dist_dir = args.dist
    if not dist_dir.is_dir():
        raise SystemExit(f"Distribution directory not found: {dist_dir}")

    license_path = args.license
    if not license_path.is_file():
        raise SystemExit(f"Canonical license file not found: {license_path}")

    canonical_bytes = license_path.read_bytes()
    canonical_digest = compute_digest(canonical_bytes)

    wheels = [inspect_wheel(path, canonical_digest) for path in discover_wheels(dist_dir)]
    if not wheels:
        raise SystemExit(f"No wheel artifacts discovered under {dist_dir}")

    failures = [wheel for wheel in wheels if wheel["status"] != "ok"]
    if failures:
        missing = ", ".join(wheel["filename"] for wheel in failures)
        raise SystemExit(f"Wheel(s) missing canonical AGPL license payload: {missing}")

    report = {
        "schema": "https://spiraltorch.org/security/release-license-report/v1",
        "tag": args.tag,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "license": {
            "path": str(license_path),
            "sha256": canonical_digest,
        },
        "wheels": wheels,
    }

    output_path = args.output or dist_dir / f"spiraltorch-{args.tag}-license-report.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
