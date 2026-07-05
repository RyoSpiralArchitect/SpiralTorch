#!/usr/bin/env python3
"""Verify that PyPI wheels match the signed GitHub Release wheel manifest."""

from __future__ import annotations

import argparse
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_PACKAGE = "spiraltorch"
DEFAULT_REPO = "RyoSpiralArchitect/SpiralTorch"


class VerifyError(RuntimeError):
    """Raised for release verification failures with user-actionable messages."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default=DEFAULT_PACKAGE, help="PyPI package name. Default: spiraltorch.")
    parser.add_argument("--version", required=True, help="PyPI package version to verify.")
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO),
        help="GitHub repository slug. Defaults to $GITHUB_REPOSITORY or RyoSpiralArchitect/SpiralTorch.",
    )
    parser.add_argument(
        "--release-tag",
        help="GitHub Release tag to compare against. Defaults to v{version}.",
    )
    parser.add_argument(
        "--github-token-env",
        default="GITHUB_TOKEN",
        help="Environment variable used for authenticated GitHub Release asset reads. Default: GITHUB_TOKEN.",
    )
    parser.add_argument(
        "--expected-wheels",
        type=int,
        help="Require this exact number of wheel files on both PyPI and the GitHub Release.",
    )
    parser.add_argument(
        "--require-latest",
        action="store_true",
        help="Also require the PyPI package's latest version to equal --version.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=240.0,
        help="Seconds to wait for PyPI JSON to expose the expected wheels. Default: 240.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Seconds between PyPI JSON polling attempts. Default: 10.",
    )
    return parser.parse_args()


def download_text(url: str, *, token: str | None = None) -> str:
    headers = {"Accept": "application/octet-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError, UnicodeDecodeError) as exc:
        raise VerifyError(f"Unable to download text from {url}") from exc


def download_json(url: str) -> dict:
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifyError(f"Unable to download JSON from {url}") from exc


def parse_sha256_lines(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 2:
            raise VerifyError(f"Malformed wheels.sha256 line: {line!r}")
        digest, name = parts
        if len(digest) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in digest):
            raise VerifyError(f"Malformed sha256 digest for {name}: {digest}")
        if name in entries:
            raise VerifyError(f"Duplicate wheels.sha256 entry for {name}")
        entries[name] = digest.lower()
    return entries


def github_release_wheel_digests(repo: str, tag: str, token_env: str) -> dict[str, str]:
    token = os.environ.get(token_env) or None
    url = f"https://github.com/{repo}/releases/download/{tag}/wheels.sha256"
    entries = parse_sha256_lines(download_text(url, token=token))
    wheels = {name: digest for name, digest in entries.items() if name.endswith(".whl")}
    if not wheels:
        raise VerifyError(f"GitHub Release {repo}@{tag} did not expose wheel entries in wheels.sha256")
    return wheels


def pypi_latest_version(package: str) -> str | None:
    payload = download_json(f"https://pypi.org/pypi/{package}/json")
    info = payload.get("info") or {}
    latest = info.get("version")
    return latest if isinstance(latest, str) else None


def pypi_wheel_digests(package: str, version: str) -> dict[str, str]:
    payload = download_json(f"https://pypi.org/pypi/{package}/{version}/json")
    urls = payload.get("urls")
    if not isinstance(urls, list):
        raise VerifyError(f"PyPI JSON for {package}=={version} did not include a urls list")

    wheels: dict[str, str] = {}
    for file_info in urls:
        if not isinstance(file_info, dict):
            continue
        filename = file_info.get("filename")
        if not isinstance(filename, str) or not filename.endswith(".whl"):
            continue
        digests = file_info.get("digests") or {}
        sha256 = digests.get("sha256")
        if not isinstance(sha256, str):
            raise VerifyError(f"PyPI file {filename} did not expose a sha256 digest")
        if filename in wheels:
            raise VerifyError(f"Duplicate PyPI wheel entry for {filename}")
        wheels[filename] = sha256.lower()
    return wheels


def wait_for_pypi_wheels(
    package: str,
    version: str,
    *,
    expected_wheels: int | None,
    timeout: float,
    poll_interval: float,
) -> dict[str, str]:
    deadline = time.time() + timeout
    wheels: dict[str, str] = {}
    while True:
        wheels = pypi_wheel_digests(package, version)
        print(f"pypi_wheels_for_{version}={len(wheels)}", flush=True)
        if expected_wheels is None or len(wheels) >= expected_wheels:
            return wheels
        if time.time() >= deadline:
            raise VerifyError(
                f"Timed out waiting for {expected_wheels} PyPI wheel(s) for {package}=={version}; "
                f"last_count={len(wheels)}"
            )
        time.sleep(poll_interval)


def compare_digests(release: dict[str, str], pypi: dict[str, str]) -> None:
    release_names = set(release)
    pypi_names = set(pypi)
    missing = sorted(release_names - pypi_names)
    extra = sorted(pypi_names - release_names)
    mismatched = sorted(name for name in release_names & pypi_names if release[name] != pypi[name])
    if missing or extra or mismatched:
        details = {"missing": missing, "extra": extra, "mismatched": mismatched}
        raise VerifyError("PyPI wheels do not match GitHub Release wheels.sha256: " + json.dumps(details, sort_keys=True))


def main() -> int:
    args = parse_args()
    tag = args.release_tag or f"v{args.version}"
    release = github_release_wheel_digests(args.repo, tag, args.github_token_env)

    if args.expected_wheels is not None and len(release) != args.expected_wheels:
        raise VerifyError(
            f"GitHub Release {args.repo}@{tag} exposes {len(release)} wheel(s); expected {args.expected_wheels}"
        )

    pypi = wait_for_pypi_wheels(
        args.package,
        args.version,
        expected_wheels=args.expected_wheels,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
    )

    if args.expected_wheels is not None and len(pypi) != args.expected_wheels:
        raise VerifyError(f"PyPI exposes {len(pypi)} wheel(s) for {args.version}; expected {args.expected_wheels}")
    if args.require_latest:
        latest = pypi_latest_version(args.package)
        print(f"pypi_latest={latest}", flush=True)
    else:
        latest = None
    if args.require_latest and latest != args.version:
        raise VerifyError(f"PyPI latest version is {latest!r}; expected {args.version!r}")

    compare_digests(release, pypi)
    print(
        f"pypi_release_digests=ok package={args.package} version={args.version} "
        f"repo={args.repo} tag={tag} wheels={len(pypi)}",
        flush=True,
    )
    for name in sorted(pypi):
        print(f"match {name} {pypi[name]}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerifyError as exc:
        raise SystemExit(f"error: {exc}") from exc
