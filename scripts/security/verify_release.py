#!/usr/bin/env python3
"""Verify the integrity of SpiralTorch release artifacts hosted on GitHub."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict

import requests


CHUNK_SIZE = 1024 * 1024
REQUIRED_LICENSE_TOKEN = "AGPL-3.0-or-later"
COMPLIANCE_SEAL_NAME = "spiraltorch-compliance-seal.json"
COMPLIANCE_SEAL_SCHEMA = "https://spiraltorch.org/security/compliance-seal/v1"
REPO_LICENSE_MANIFEST_NAME = "spiraltorch-repo-license-manifest.json"
RELEASE_MANIFEST_SCHEMA = "https://spiraltorch.org/security/release-manifest/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="GitHub repository slug (OWNER/REPO). Defaults to $GITHUB_REPOSITORY.",
    )
    parser.add_argument(
        "--tag",
        help="Specific release tag to verify. Defaults to the latest published release.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Directory for downloaded artifacts. A temporary directory is used when omitted.",
    )
    parser.add_argument(
        "--allow-workflow-dispatch-ref",
        action="append",
        default=[],
        help=(
            "Also accept Sigstore GitHub identities signed by the Release Wheels workflow "
            "via workflow_dispatch on this ref, e.g. refs/heads/main. Repeat to allow multiple refs."
        ),
    )
    return parser.parse_args()


def github_headers(token: str | None, accept: str | None = None) -> Dict[str, str]:
    headers = {"Accept": accept or "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def github_json(repo: str, path: str, token: str | None) -> Dict:
    url = f"https://api.github.com/repos/{repo}/{path}"
    response = requests.get(url, headers=github_headers(token))
    if response.status_code == 404:
        raise SystemExit(f"GitHub resource not found: {path}")
    response.raise_for_status()
    return response.json()


def download_asset(asset: Dict, destination: Path, token: str | None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = asset["url"]
    response = requests.get(url, headers=github_headers(token, "application/octet-stream"), stream=True)
    response.raise_for_status()
    with destination.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            fh.write(chunk)
    return destination


def download_sigstore_companions(asset_name: str, asset_index: Dict[str, Dict], work_dir: Path, token: str | None) -> None:
    bundle_name = asset_name + ".sigstore.json"
    bundle_asset = asset_index.get(bundle_name)
    if bundle_asset:
        download_asset(bundle_asset, work_dir / bundle_name, token)
        return

    legacy_names = [asset_name + ".sig", asset_name + ".crt"]
    missing = [name for name in legacy_names if name not in asset_index]
    if missing:
        raise SystemExit(f"Expected Sigstore companion asset missing for {asset_name}: {', '.join(missing)}")
    for name in legacy_names:
        download_asset(asset_index[name], work_dir / name, token)


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_ref(value: str) -> str:
    if value.startswith("refs/"):
        return value
    return f"refs/heads/{value}"


def sigstore_bundle_path(file_path: Path) -> Path:
    return file_path.with_name(file_path.name + ".sigstore.json")


def legacy_sigstore_paths(file_path: Path) -> tuple[Path, Path]:
    certificate = file_path.with_suffix(file_path.suffix + ".crt")
    signature = file_path.with_suffix(file_path.suffix + ".sig")
    return certificate, signature


def sigstore_verify_command(file_path: Path, repo: str, ref: str, trigger: str) -> list[str]:
    bundle = sigstore_bundle_path(file_path)
    certificate, signature = legacy_sigstore_paths(file_path)
    material_args: list[str]
    if bundle.exists():
        material_args = ["--bundle", str(bundle)]
    elif certificate.exists() and signature.exists():
        material_args = ["--certificate", str(certificate), "--signature", str(signature)]
    else:
        raise SystemExit(f"Missing Sigstore bundle or legacy metadata for {file_path.name}")

    return [
        sys.executable,
        "-m",
        "sigstore",
        "verify",
        "github",
        *material_args,
        "--repository",
        repo,
        "--ref",
        ref,
        "--name",
        "Release Wheels",
        "--trigger",
        trigger,
        str(file_path),
    ]


def run_sigstore_verify(file_path: Path, repo: str, tag: str, allowed_workflow_dispatch_refs: list[str] | None = None) -> None:
    bundle = sigstore_bundle_path(file_path)
    certificate, signature = legacy_sigstore_paths(file_path)
    if not bundle.exists() and not (certificate.exists() and signature.exists()):
        raise SystemExit(f"Missing Sigstore bundle or legacy metadata for {file_path.name}")

    identities = [(f"refs/tags/{tag}", "push")]
    for ref in allowed_workflow_dispatch_refs or []:
        identities.append((normalize_ref(ref), "workflow_dispatch"))

    failures: list[str] = []
    for ref, trigger in identities:
        cmd = sigstore_verify_command(file_path, repo, ref, trigger)
        completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode == 0:
            print(f"Sigstore verified {file_path.name} with trigger={trigger} ref={ref}.")
            return
        detail = (completed.stderr or completed.stdout).strip()
        failures.append(f"trigger={trigger} ref={ref}: {detail or f'exit {completed.returncode}'}")

    summary = "\n - ".join(failures)
    raise SystemExit(f"Sigstore verification failed for {file_path.name}:\n - {summary}")


def validate_compliance_seal(seal_path: Path, manifest_path: Path, license_report: Dict | None, failures: list[str]) -> None:
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("schema") != COMPLIANCE_SEAL_SCHEMA:
        failures.append(f"Compliance seal schema mismatch in {seal_path.name}")
        return

    manifest_info = seal.get("manifest") or {}
    expected_manifest_path = manifest_info.get("path")
    if expected_manifest_path and expected_manifest_path != manifest_path.name:
        failures.append("Compliance seal references an unexpected manifest filename.")

    for algorithm, key in (("sha256", "sha256"), ("sha512", "sha512")):
        expected = manifest_info.get(key)
        if not expected:
            failures.append(f"Compliance seal missing manifest {key} digest.")
            continue
        actual = file_digest(manifest_path, algorithm)
        if actual != expected:
            failures.append(f"Compliance seal manifest {key} digest mismatch.")

    canonical = seal.get("canonical_license") or {}
    for field in ("path", "sha256", "sha512"):
        if not canonical.get(field):
            failures.append("Compliance seal missing canonical license metadata.")
            break

    clause = seal.get("agpl_clause", "")
    if REQUIRED_LICENSE_TOKEN not in clause:
        failures.append("Compliance seal clause does not enforce AGPL obligations.")

    if license_report:
        canonical_report = (license_report.get("license") or {}).get("sha256")
        if canonical_report and canonical.get("sha256") != canonical_report:
            failures.append("Compliance seal canonical license digest disagrees with license report.")

    commit = seal.get("commit")
    if not commit or len(commit) < 7:
        failures.append("Compliance seal omitted a valid commit identifier.")

    required_files = seal.get("required_files") or []
    for expected in (manifest_path.name, canonical.get("path"), "NOTICE"):
        if expected and expected not in required_files:
            failures.append(f"Compliance seal required_files missing {expected}.")


def verify_release(args: argparse.Namespace) -> None:
    if not args.repo:
        raise SystemExit("A repository slug must be provided via --repo or $GITHUB_REPOSITORY.")

    token = os.environ.get("GITHUB_TOKEN")

    if args.tag:
        release = github_json(args.repo, f"releases/tags/{args.tag}", token)
    else:
        release = github_json(args.repo, "releases/latest", token)

    tag_name = release.get("tag_name")
    if not tag_name:
        raise SystemExit("Release response did not include a tag_name field.")

    assets = release.get("assets", [])
    if not assets:
        raise SystemExit(f"Release {tag_name} does not expose any downloadable assets.")

    asset_index = {asset["name"]: asset for asset in assets}
    manifest_name = f"spiraltorch-{tag_name}-manifest.json"
    manifest_asset = asset_index.get(manifest_name)
    if not manifest_asset:
        manifest_candidates = sorted(
            asset["name"] for asset in assets if asset["name"].endswith("-manifest.json")
        )
        candidate_summary = ", ".join(manifest_candidates) if manifest_candidates else "<none>"
        raise SystemExit(
            "No authenticated release manifest was found in the release assets. "
            f"Expected {manifest_name}; saw manifest-like assets: {candidate_summary}."
        )

    with contextlib.ExitStack() as stack:
        if args.work_dir:
            work_dir = args.work_dir
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            tmp_dir = stack.enter_context(tempfile.TemporaryDirectory(prefix="spiraltorch-release-"))
            work_dir = Path(tmp_dir)

        manifest_path = work_dir / manifest_asset["name"]
        download_asset(manifest_asset, manifest_path, token)

        # Download accompanying Sigstore materials for the manifest itself.
        download_sigstore_companions(manifest_asset["name"], asset_index, work_dir, token)

        # Verify the manifest signature before trusting its contents.
        run_sigstore_verify(manifest_path, args.repo, tag_name, args.allow_workflow_dispatch_ref)

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != RELEASE_MANIFEST_SCHEMA:
            raise SystemExit(f"Release manifest schema mismatch in {manifest_path.name}.")
        if manifest.get("tag") != tag_name:
            raise SystemExit(f"Release manifest tag mismatch in {manifest_path.name}.")

        manifest_files = manifest.get("files", [])
        if not manifest_files:
            raise SystemExit("Release manifest did not contain any file entries to validate.")

        failures: list[str] = []

        license_report_name = f"spiraltorch-{tag_name}-license-report.json"
        license_report_asset = asset_index.get(license_report_name)
        license_report = None
        if not license_report_asset:
            failures.append(f"Missing license provenance report asset: {license_report_name}")
        else:
            license_report_path = work_dir / license_report_name
            download_asset(license_report_asset, license_report_path, token)
            license_report = json.loads(license_report_path.read_text(encoding="utf-8"))

            canonical_sha256 = license_report.get("license", {}).get("sha256")
            if not canonical_sha256:
                failures.append("License report missing canonical license digest.")

            wheel_reports = license_report.get("wheels") or []
            if not wheel_reports:
                failures.append("License report did not enumerate any wheel artifacts.")

            for entry in wheel_reports:
                filename = entry.get("filename", "<unknown>")
                embedded = entry.get("embedded_licenses") or []
                if not embedded:
                    failures.append(f"License report for {filename} did not list any embedded license payloads.")
                    continue
                if not any(item.get("matches_canonical") for item in embedded):
                    failures.append(f"No canonical AGPL license match found inside {filename}.")

        compliance_seal_path: Path | None = None
        repo_license_manifest_path: Path | None = None

        for entry in manifest_files:
            asset_name = entry.get("asset")
            if not asset_name:
                failures.append("Manifest entry is missing an 'asset' field.")
                continue

            asset = asset_index.get(asset_name)
            if not asset:
                failures.append(f"Manifest references unknown asset: {asset_name}")
                continue

            asset_path = work_dir / asset_name
            download_asset(asset, asset_path, token)

            if asset_name == COMPLIANCE_SEAL_NAME:
                compliance_seal_path = asset_path
            elif asset_name == REPO_LICENSE_MANIFEST_NAME:
                repo_license_manifest_path = asset_path

            expected_sha256 = entry.get("sha256")
            expected_sha512 = entry.get("sha512")
            expected_size = entry.get("size")

            actual_size = asset_path.stat().st_size
            if expected_size is not None and actual_size != expected_size:
                failures.append(f"Size mismatch for {asset_name}: expected {expected_size}, saw {actual_size}")

            if expected_sha256:
                actual_sha256 = file_digest(asset_path, "sha256")
                if actual_sha256 != expected_sha256:
                    failures.append(f"sha256 mismatch for {asset_name}")

            if expected_sha512:
                actual_sha512 = file_digest(asset_path, "sha512")
                if actual_sha512 != expected_sha512:
                    failures.append(f"sha512 mismatch for {asset_name}")

            try:
                download_sigstore_companions(asset_name, asset_index, work_dir, token)
                run_sigstore_verify(asset_path, args.repo, tag_name, args.allow_workflow_dispatch_ref)
            except SystemExit as exc:
                failures.append(str(exc))

        if compliance_seal_path is None:
            failures.append(f"Release missing compliance seal asset: {COMPLIANCE_SEAL_NAME}")
        elif repo_license_manifest_path is None:
            failures.append(f"Release missing repo license manifest asset: {REPO_LICENSE_MANIFEST_NAME}")
        else:
            validate_compliance_seal(
                compliance_seal_path,
                repo_license_manifest_path,
                license_report,
                failures,
            )

        if failures:
            summary = "\n - ".join(failures)
            raise SystemExit(f"Release verification failed:\n - {summary}")

        print(
            "Release {tag} passed integrity, signature, and AGPL provenance checks for {count} assets.".format(
                tag=tag_name,
                count=len(manifest_files),
            )
        )


def main() -> None:
    verify_release(parse_args())


if __name__ == "__main__":
    main()
