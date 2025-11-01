#!/usr/bin/env python3
"""Generate manifest, checksums, and SBOM for release artifacts.

This script is executed from CI to harden the release pipeline:
- creates deterministic SHA-256 checksums for built wheels
- captures dependency metadata for auditing/resale disputes
- emits a signed manifest tying artifacts to repository provenance
- writes verification instructions so downstream consumers can validate authenticity
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from typing import Dict, Iterable, List, Tuple

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - legacy fallback
    import tomli as tomllib  # type: ignore[assignment]

REPO_URL = "https://github.com/spiraltorch/spiraltorch"
SCHEMA_URL = "https://spiraltorch.dev/security/release-manifest-v1.json"


class ArtifactError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=pathlib.Path, required=True, help="Path containing release artifacts")
    parser.add_argument("--manifest", type=pathlib.Path, required=True, help="Destination for the release manifest JSON")
    parser.add_argument("--checksums", type=pathlib.Path, required=True, help="Destination for aggregate SHA-256 checksums")
    parser.add_argument("--sbom", type=pathlib.Path, required=True, help="Destination for dependency SBOM JSON")
    parser.add_argument("--verify-doc", type=pathlib.Path, required=True, help="Destination for verification instructions")
    return parser.parse_args()


def sha256_file(path: pathlib.Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_artifacts(dist_path: pathlib.Path) -> List[Dict[str, object]]:
    resolved_root = dist_path.resolve()
    artifacts: List[Dict[str, object]] = []
    for wheel in sorted(resolved_root.glob("**/*.whl")):
        if wheel.is_file():
            digest = sha256_file(wheel)
            try:
                relative = wheel.relative_to(resolved_root)
            except ValueError:
                relative = pathlib.Path(wheel.name)
            artifacts.append(
                {
                    "path": wheel,
                    "relative": relative.as_posix(),
                    "sha256": digest,
                    "size": wheel.stat().st_size,
                }
            )
    if not artifacts:
        raise ArtifactError(f"No wheel artifacts discovered under {dist_path!s}")
    return artifacts


def write_checksums(artifacts: Iterable[Dict[str, object]], destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for artifact in artifacts:
        digest = artifact["sha256"]
        rel = artifact["relative"]
        lines.append(f"{digest}  {rel}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_commit() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - CI fallbacks
        return "unknown"
    return completed.stdout.strip()


def git_tag() -> str:
    ref = os.environ.get("GITHUB_REF")
    if ref and ref.startswith("refs/tags/"):
        return ref.split("/", 2)[-1]
    return os.environ.get("GITHUB_REF_NAME", "")


def generate_manifest(artifacts: Iterable[Dict[str, object]]) -> Dict[str, object]:
    ts = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    manifest_artifacts: List[Dict[str, object]] = []
    for artifact in artifacts:
        manifest_artifacts.append(
            {
                "path": artifact["relative"],
                "sha256": artifact["sha256"],
                "size": artifact["size"],
            }
        )
    return {
        "$schema": SCHEMA_URL,
        "generated_at": ts,
        "repository": REPO_URL,
        "git_commit": git_commit(),
        "git_tag": git_tag(),
        "artifacts": manifest_artifacts,
    }


def load_cargo_lock(lock_path: pathlib.Path = pathlib.Path("Cargo.lock")) -> Dict[str, object]:
    try:
        with lock_path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ArtifactError("Cargo.lock not found for SBOM generation") from exc


def workspace_members(manifest_path: pathlib.Path = pathlib.Path("Cargo.toml")) -> List[str]:
    try:
        with manifest_path.open("rb") as fh:
            manifest = tomllib.load(fh)
    except FileNotFoundError:
        return []
    workspace = manifest.get("workspace")
    if not isinstance(workspace, dict):
        package = manifest.get("package")
        if isinstance(package, dict):
            name = package.get("name")
            return [name] if isinstance(name, str) else []
        return []
    members = workspace.get("members", [])
    if isinstance(members, list):
        return [m for m in members if isinstance(m, str)]
    return []


def generate_sbom(lock_data: Dict[str, object], members: List[str]) -> Dict[str, object]:
    packages: List[Dict[str, object]] = []
    for pkg in lock_data.get("package", []):
        if not isinstance(pkg, dict):
            continue
        packages.append(
            {
                "name": pkg.get("name"),
                "version": pkg.get("version"),
                "source": pkg.get("source"),
                "checksum": pkg.get("checksum"),
                "license": pkg.get("license"),
            }
        )
    metadata_table = lock_data.get("metadata")
    root = metadata_table.get("root") if isinstance(metadata_table, dict) else None
    root_deps: List[str] = []
    if isinstance(root, dict):
        deps = root.get("dependencies")
        if isinstance(deps, list):
            root_deps = [dep for dep in deps if isinstance(dep, str)]
    return {
        "sbom_version": 1,
        "generated_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "workspace_members": members,
        "packages": packages,
        "root_dependencies": root_deps,
    }


def write_json(payload: Dict[str, object], destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_verify_doc(destination: pathlib.Path, manifest_path: pathlib.Path, checksums_path: pathlib.Path, sbom_path: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SpiralTorch Release Verification",
        "",
        "Every official SpiralTorch build ships with cryptographic materials to make tampering and grey-market redistribution obvious.",
        "",
        "1. Validate the provenance attestations published with the GitHub release to ensure the wheels were built inside GitHub Actions.",
        f"2. Verify the Sigstore signature for each artifact, including `{manifest_path.name}` and `{checksums_path.name}`, against the SpiralTorch organization.",
        f"3. Compute SHA-256 checksums for downloaded files and compare them with the entries in `{checksums_path.name}`.",
        f"4. Inspect `{manifest_path.name}` for the commit and tag you expect. Any mismatch indicates the build is not from {REPO_URL}.",
        f"5. Review `{sbom_path.name}` to confirm the dependency graph matches the upstream project and no unexpected packages were injected.",
        "",
        "If any of the above validations fail, treat the package as untrusted. Report suspicious mirrors or resale attempts to kishkavsesvit@icloud.com immediately.",
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    artifacts = collect_artifacts(args.dist)
    write_checksums(artifacts, args.checksums)
    manifest = generate_manifest(artifacts)
    write_json(manifest, args.manifest)
    lock_data = load_cargo_lock()
    sbom = generate_sbom(lock_data, workspace_members())
    write_json(sbom, args.sbom)
    write_verify_doc(args.verify_doc, args.manifest, args.checksums, args.sbom)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
