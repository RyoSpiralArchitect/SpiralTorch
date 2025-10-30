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


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_sigstore_verify(file_path: Path, repo: str, tag: str) -> None:
    certificate = file_path.with_suffix(file_path.suffix + ".crt")
    signature = file_path.with_suffix(file_path.suffix + ".sig")
    if not certificate.exists() or not signature.exists():
        raise SystemExit(f"Missing Sigstore metadata for {file_path.name}")

    cmd = [
        sys.executable,
        "-m",
        "sigstore",
        "verify",
        "github",
        "--certificate",
        str(certificate),
        "--signature",
        str(signature),
        "--repository",
        repo,
        "--ref",
        f"refs/tags/{tag}",
        "--name",
        "Release Wheels",
        "--trigger",
        "push",
        str(file_path),
    ]
    subprocess.run(cmd, check=True)


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

    manifest_asset = next((asset for asset in assets if asset["name"].endswith("-manifest.json")), None)
    if not manifest_asset:
        raise SystemExit(
            "No authenticated manifest was found in the release assets. Ensure the release workflow has run successfully."
        )

    asset_index = {asset["name"]: asset for asset in assets}

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
        for suffix in (".sig", ".crt"):
            sig_asset = asset_index.get(manifest_asset["name"] + suffix)
            if not sig_asset:
                raise SystemExit(f"Expected manifest companion asset missing: {manifest_asset['name'] + suffix}")
            download_asset(sig_asset, work_dir / sig_asset["name"], token)

        # Verify the manifest signature before trusting its contents.
        run_sigstore_verify(manifest_path, args.repo, tag_name)

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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

            sig_path = asset_path.with_suffix(asset_path.suffix + ".sig")
            crt_path = asset_path.with_suffix(asset_path.suffix + ".crt")
            sig_asset = asset_index.get(sig_path.name)
            crt_asset = asset_index.get(crt_path.name)
            if sig_asset and crt_asset:
                download_asset(sig_asset, sig_path, token)
                download_asset(crt_asset, crt_path, token)
                run_sigstore_verify(asset_path, args.repo, tag_name)
            else:
                failures.append(f"Missing Sigstore signature or certificate for {asset_name}")

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
