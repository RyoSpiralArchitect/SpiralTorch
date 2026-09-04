#!/usr/bin/env python3
"""Verify that PyPI wheels match the signed GitHub Release wheel manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import ssl
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_PACKAGE = "spiraltorch"
DEFAULT_REPO = "RyoSpiralArchitect/SpiralTorch"
SIMPLE_JSON_TYPE = "application/vnd.pypi.simple.v1+json"
# Match pip's content negotiation to avoid checking a different cache variant.
SIMPLE_ACCEPT = (
    SIMPLE_JSON_TYPE + ", application/vnd.pypi.simple.v1+html; q=0.1, text/html; q=0.01"
)


class VerifyError(RuntimeError):
    """Raised for release verification failures with user-actionable messages."""


class PublicationPending(VerifyError):
    """A read-only availability failure that may be retried within the budget."""


def positive_seconds(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be positive finite seconds")
    return result


def positive_count(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive wheel count")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        default=DEFAULT_PACKAGE,
        help="PyPI package name. Default: spiraltorch.",
    )
    parser.add_argument(
        "--version", required=True, help="PyPI package version to verify."
    )
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
        type=positive_count,
        help="Require this exact number of wheel files on both PyPI and the GitHub Release.",
    )
    parser.add_argument(
        "--require-latest",
        action="store_true",
        help="Also require the PyPI package's latest version to equal --version.",
    )
    parser.add_argument(
        "--require-simple-index",
        action="store_true",
        help="Also wait for matching, non-yanked wheel hashes in pip's Simple API.",
    )
    parser.add_argument(
        "--pip-requirements",
        type=Path,
        help="After verification, write the exact version and approved hashes for pip --require-hashes.",
    )
    parser.add_argument(
        "--timeout",
        type=positive_seconds,
        default=240.0,
        help=(
            "Retry budget for PyPI JSON, optional latest-version and Simple API "
            "readiness. Socket timeouts are capped by the remaining budget. Default: 240."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=positive_seconds,
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


def download_json(url: str, *, simple: bool = False, timeout: float = 30.0) -> dict:
    headers = {"Accept": SIMPLE_ACCEPT if simple else "application/json"}
    if simple:
        headers["Cache-Control"] = "max-age=0"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            if simple:
                content_type = (
                    response.headers.get("Content-Type", "")
                    .split(";")[0]
                    .strip()
                    .lower()
                )
                if content_type != SIMPLE_JSON_TYPE:
                    raise VerifyError(
                        f"Unexpected Simple API content type: {content_type!r}"
                    )
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code in {404, 408, 429} or 500 <= exc.code <= 599:
            raise PublicationPending(
                f"PyPI read temporarily unavailable: HTTP {exc.code}"
            ) from exc
        raise VerifyError(
            f"Unable to download JSON from {url}: HTTP {exc.code}"
        ) from exc
    except URLError as exc:
        if isinstance(exc.reason, ssl.SSLCertVerificationError):
            raise VerifyError(f"TLS verification failed for {url}") from exc
        raise PublicationPending(f"PyPI read temporarily unavailable at {url}") from exc
    except TimeoutError as exc:
        raise PublicationPending(f"PyPI read timed out at {url}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifyError(f"Unable to download JSON from {url}") from exc
    if not isinstance(payload, dict):
        raise VerifyError(f"Expected a JSON object from {url}")
    return payload


def normalize_package(package: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", package):
        raise VerifyError("Invalid PyPI package name")
    return re.sub(r"[-_.]+", "-", package).lower()


def sha256_digest(value: object, filename: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise VerifyError(f"Malformed sha256 digest for {filename}")
    return value.lower()


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
        if len(digest) != 64 or any(
            ch not in "0123456789abcdefABCDEF" for ch in digest
        ):
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
        raise VerifyError(
            f"GitHub Release {repo}@{tag} did not expose wheel entries in wheels.sha256"
        )
    return wheels


def pypi_latest_version(package: str, *, timeout: float = 30.0) -> str:
    payload = download_json(
        f"https://pypi.org/pypi/{normalize_package(package)}/json", timeout=timeout
    )
    info = payload.get("info")
    if not isinstance(info, dict):
        raise VerifyError("PyPI project JSON did not include an info object")
    latest = info.get("version")
    if not isinstance(latest, str) or not latest:
        raise VerifyError("PyPI project JSON did not include a version string")
    return latest


def pypi_wheel_digests(
    package: str, version: str, *, timeout: float = 30.0
) -> dict[str, str]:
    payload = download_json(
        f"https://pypi.org/pypi/{normalize_package(package)}/{quote(version, safe='')}/json",
        timeout=timeout,
    )
    urls = payload.get("urls")
    if not isinstance(urls, list):
        raise VerifyError(
            f"PyPI JSON for {package}=={version} did not include a urls list"
        )

    wheels: dict[str, str] = {}
    for file_info in urls:
        if not isinstance(file_info, dict):
            raise VerifyError("Malformed PyPI release file entry")
        filename = file_info.get("filename")
        if not isinstance(filename, str) or not filename:
            raise VerifyError("Malformed PyPI release filename")
        if not filename.endswith(".whl"):
            continue
        digests = file_info.get("digests")
        if not isinstance(digests, dict):
            raise VerifyError(f"PyPI file {filename} did not expose a digests object")
        if filename in wheels:
            raise VerifyError(f"Duplicate PyPI wheel entry for {filename}")
        wheels[filename] = sha256_digest(digests.get("sha256"), filename)
    return wheels


def pypi_simple_wheel_digests(
    package: str, expected: dict[str, str], *, timeout: float = 30.0
) -> dict[str, str]:
    package = normalize_package(package)
    payload = download_json(
        f"https://pypi.org/simple/{package}/", simple=True, timeout=timeout
    )
    meta = payload.get("meta")
    api_version = meta.get("api-version") if isinstance(meta, dict) else None
    if not isinstance(api_version, str) or not re.fullmatch(r"1\.[0-9]+", api_version):
        raise VerifyError("Unsupported or malformed Simple API version")
    if payload.get("name") != package or not isinstance(payload.get("files"), list):
        raise VerifyError("Malformed or wrong-project Simple API response")
    wheels: dict[str, str] = {}
    for file_info in payload["files"]:
        if not isinstance(file_info, dict) or not isinstance(
            file_info.get("filename"), str
        ):
            raise VerifyError("Malformed Simple API file entry")
        filename = file_info["filename"]
        if filename not in expected:
            continue
        if filename in wheels:
            raise VerifyError(f"Duplicate Simple API wheel entry for {filename}")
        yanked = file_info.get("yanked", False)
        if yanked is not False:
            raise VerifyError(
                f"Simple API wheel {filename} is yanked or has invalid yank metadata"
            )
        hashes = file_info.get("hashes")
        if not isinstance(hashes, dict):
            raise VerifyError(f"Simple API wheel {filename} did not expose hashes")
        wheels[filename] = sha256_digest(hashes.get("sha256"), filename)
        if wheels[filename] != expected[filename]:
            raise VerifyError(f"Simple API wheel hash mismatched for {filename}")
    return wheels


def wait_for_pypi_wheels(
    package: str,
    version: str,
    *,
    expected_wheels: int | None,
    require_latest: bool = False,
    require_simple_index: bool = False,
    expected_digests: dict[str, str] | None = None,
    timeout: float,
    poll_interval: float,
) -> dict[str, str]:
    if (
        not math.isfinite(timeout)
        or timeout <= 0
        or not math.isfinite(poll_interval)
        or poll_interval <= 0
    ):
        raise VerifyError("timeout and poll_interval must be positive finite seconds")
    if expected_wheels is not None and (
        type(expected_wheels) is not int or expected_wheels <= 0
    ):
        raise VerifyError("expected_wheels must be positive")
    if require_simple_index and not expected_digests:
        raise VerifyError(
            "Simple API readiness requires the expected release wheel hashes"
        )
    if expected_digests is not None and (
        not expected_digests
        or (expected_wheels is not None and len(expected_digests) != expected_wheels)
    ):
        raise VerifyError("Expected release wheel set does not match expected_wheels")
    if expected_digests is not None:
        expected_digests = {
            name: sha256_digest(digest, name)
            for name, digest in expected_digests.items()
        }
    deadline = time.monotonic() + timeout
    wheels: dict[str, str] = {}
    latest: str | None = None
    reason = "publication not yet visible"

    def request_timeout() -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PublicationPending("readiness budget expired")
        return min(30.0, remaining)

    while True:
        try:
            wheels = pypi_wheel_digests(package, version, timeout=request_timeout())
            print(f"pypi_wheels_for_{version}={len(wheels)}", flush=True)
            if expected_digests is not None:
                for name, digest in wheels.items():
                    if name not in expected_digests or expected_digests[name] != digest:
                        raise VerifyError(
                            f"Unexpected or mismatched PyPI wheel: {name}"
                        )
            if expected_wheels is not None and len(wheels) > expected_wheels:
                raise VerifyError("PyPI exposes more wheels than the expected release")
            wheels_ready = bool(wheels) and (
                expected_wheels is None or len(wheels) == expected_wheels
            )
            if expected_digests is not None:
                wheels_ready = wheels == expected_digests
            if require_latest:
                latest = pypi_latest_version(package, timeout=request_timeout())
                print(f"pypi_latest={latest}", flush=True)
            ready = wheels_ready and (not require_latest or latest == version)
            reason = "release JSON or latest-version metadata is not ready"
            if ready and require_simple_index:
                assert expected_digests is not None
                simple = pypi_simple_wheel_digests(
                    package, expected_digests, timeout=request_timeout()
                )
                ready = simple == expected_digests
                reason = f"Simple API exposes {len(simple)}/{len(expected_digests)} release wheels"
                print(f"pypi_simple_wheels_for_{version}={len(simple)}", flush=True)
            if ready and time.monotonic() < deadline:
                return wheels
        except PublicationPending as exc:
            reason = str(exc)
            print(f"pypi_publication_pending={reason}", flush=True)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise VerifyError(
                f"Timed out waiting for PyPI publication of {package}=={version}; "
                f"expected_wheels={expected_wheels} last_count={len(wheels)} "
                f"require_latest={require_latest} latest={latest!r}; {reason}"
            )
        time.sleep(min(poll_interval, remaining))


def compare_digests(release: dict[str, str], pypi: dict[str, str]) -> None:
    release_names = set(release)
    pypi_names = set(pypi)
    missing = sorted(release_names - pypi_names)
    extra = sorted(pypi_names - release_names)
    mismatched = sorted(
        name for name in release_names & pypi_names if release[name] != pypi[name]
    )
    if missing or extra or mismatched:
        details = {"missing": missing, "extra": extra, "mismatched": mismatched}
        raise VerifyError(
            "PyPI wheels do not match GitHub Release wheels.sha256: "
            + json.dumps(details, sort_keys=True)
        )


def write_pip_requirements(
    path: Path, package: str, version: str, release: dict[str, str]
) -> None:
    package = normalize_package(package)
    if not re.fullmatch(r"[0-9][A-Za-z0-9.!+_-]*", version):
        raise VerifyError("Unsafe or unsupported version in pip requirements output")
    if not release:
        raise VerifyError("Cannot write pip requirements without approved wheel hashes")
    hashes = sorted({sha256_digest(value, name) for name, value in release.items()})
    requirement = f"{package}=={version} " + " ".join(
        f"--hash=sha256:{value}" for value in hashes
    )
    path.write_text(requirement + "\n", encoding="utf-8")


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
        require_latest=args.require_latest,
        require_simple_index=args.require_simple_index,
        expected_digests=release,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
    )

    if args.expected_wheels is not None and len(pypi) != args.expected_wheels:
        raise VerifyError(
            f"PyPI exposes {len(pypi)} wheel(s) for {args.version}; expected {args.expected_wheels}"
        )
    compare_digests(release, pypi)
    if args.pip_requirements is not None:
        write_pip_requirements(
            args.pip_requirements, args.package, args.version, release
        )
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
