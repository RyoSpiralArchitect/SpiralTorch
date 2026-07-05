#!/usr/bin/env python3
"""Safely publish SpiralTorch wheel artifacts to PyPI.

The helper is intentionally conservative: it never prints the PyPI token,
validates wheel metadata before upload, checks the current PyPI release state,
and can run as a dry-run while waiting for credentials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PACKAGE = "spiraltorch"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE}/json"


class PublishError(RuntimeError):
    """Raised for user-actionable release publishing failures."""


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True, help="Directory containing wheel artifacts.")
    parser.add_argument(
        "--expected-version",
        help="Require every discovered wheel to use this exact package version.",
    )
    parser.add_argument(
        "--token-source",
        choices=("clipboard", "env", "none"),
        default="clipboard",
        help="Where to read the PyPI API token from before upload. Default: clipboard via pbpaste.",
    )
    parser.add_argument(
        "--token-env",
        default="PYPI_API_TOKEN",
        help="Environment variable used when --token-source=env. Default: PYPI_API_TOKEN.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate wheels and report PyPI/token readiness without uploading.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Pass --skip-existing to twine upload and allow already-present PyPI files.",
    )
    parser.add_argument(
        "--repository-url",
        help="Optional twine repository URL. Omit for production PyPI.",
    )
    parser.add_argument(
        "--github-release-tag",
        help=(
            "Optional GitHub Release tag whose wheels.sha256 must exactly match "
            "the local wheel set before upload."
        ),
    )
    parser.add_argument(
        "--github-release-repo",
        default=os.environ.get("GITHUB_REPOSITORY", "RyoSpiralArchitect/SpiralTorch"),
        help=(
            "Repository slug used with --github-release-tag. "
            "Defaults to $GITHUB_REPOSITORY or RyoSpiralArchitect/SpiralTorch."
        ),
    )
    parser.add_argument(
        "--github-token-env",
        default="GITHUB_TOKEN",
        help="Optional environment variable for authenticated release asset download. Default: GITHUB_TOKEN.",
    )
    parser.add_argument(
        "--no-smoke",
        action="store_true",
        help="Skip the post-upload pip install/import smoke test.",
    )
    parser.add_argument(
        "--verify-timeout",
        type=float,
        default=180.0,
        help="Seconds to wait for PyPI JSON to expose the uploaded files. Default: 180.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for twine and smoke tests. Default: current interpreter.",
    )
    return parser.parse_args()


def run(
    args: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    log("+ " + " ".join(args))
    return subprocess.run(args, check=True, env=env, cwd=cwd, text=True)


def discover_wheels(dist: Path) -> list[Path]:
    wheels = sorted(path for path in dist.rglob(f"{PACKAGE}-*.whl") if path.is_file())
    if not wheels:
        raise PublishError(f"No {PACKAGE} wheels found under {dist}")
    return wheels


def wheel_version(path: Path) -> str:
    parts = path.name.split("-")
    if len(parts) < 2 or parts[0] != PACKAGE:
        raise PublishError(f"Unexpected wheel filename: {path.name}")
    return parts[1]


def validate_versions(wheels: Iterable[Path], expected: str | None) -> str:
    versions = {wheel_version(path) for path in wheels}
    if len(versions) != 1:
        raise PublishError(f"Wheel artifacts disagree on versions: {sorted(versions)}")
    version = versions.pop()
    if expected is not None and version != expected:
        raise PublishError(f"Wheel version {version} does not match --expected-version {expected}")
    return version


def twine_check(python: str, wheels: Sequence[Path]) -> None:
    run([python, "-m", "twine", "check", *(str(path) for path in wheels)])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_text(url: str, *, token: str | None = None) -> str:
    headers = {"Accept": "application/octet-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError, UnicodeDecodeError) as exc:
        raise PublishError(f"Unable to download release metadata from {url}") from exc


def parse_sha256_lines(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 2:
            raise PublishError(f"Malformed wheels.sha256 line: {line!r}")
        digest, name = parts
        if len(digest) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in digest):
            raise PublishError(f"Malformed sha256 digest for {name}: {digest}")
        entries[name] = digest.lower()
    return entries


def verify_github_release_wheel_checksums(
    wheels: Sequence[Path],
    *,
    repo: str,
    tag: str,
    token_env: str,
) -> None:
    url = f"https://github.com/{repo}/releases/download/{tag}/wheels.sha256"
    token = os.environ.get(token_env) or None
    expected = parse_sha256_lines(download_text(url, token=token))
    local = {path.name: file_sha256(path) for path in wheels}

    expected_wheels = {name: digest for name, digest in expected.items() if name.endswith(".whl")}
    if not expected_wheels:
        raise PublishError(f"Release {tag} did not expose any wheel entries in wheels.sha256")

    missing = sorted(set(expected_wheels) - set(local))
    extra = sorted(set(local) - set(expected_wheels))
    mismatched = sorted(name for name in set(local) & set(expected_wheels) if local[name] != expected_wheels[name])

    if missing or extra or mismatched:
        details = {
            "missing": missing,
            "extra": extra,
            "mismatched": mismatched,
        }
        raise PublishError(
            f"Local wheels do not match GitHub Release {repo}@{tag} wheels.sha256: "
            + json.dumps(details, sort_keys=True)
        )

    log(f"github_release_wheel_checksums=ok repo={repo} tag={tag} wheels={len(local)}")


def read_clipboard() -> str:
    try:
        return subprocess.check_output(["pbpaste"], text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PublishError("Unable to read clipboard with pbpaste") from exc


def read_token(source: str, env_name: str) -> tuple[str, dict[str, object]]:
    raw = ""
    if source == "clipboard":
        raw = read_clipboard()
    elif source == "env":
        raw = os.environ.get(env_name, "")

    token = raw.strip()
    metadata = {
        "source": source,
        "chars": len(token),
        "trimmed": token != raw,
        "starts_with_pypi": token.startswith("pypi-"),
        "is_ascii": all(ord(ch) < 128 for ch in token),
        "contains_whitespace": any(ch.isspace() for ch in token),
    }
    return token, metadata


def token_is_upload_ready(metadata: dict[str, object]) -> bool:
    return bool(metadata["starts_with_pypi"] and metadata["is_ascii"] and not metadata["contains_whitespace"])


def require_upload_token(token: str, metadata: dict[str, object]) -> None:
    if not token:
        raise PublishError(f"PyPI token is empty (source={metadata['source']})")
    if not token_is_upload_ready(metadata):
        raise PublishError(
            "PyPI token is not upload-ready: "
            f"starts_with_pypi={metadata['starts_with_pypi']} "
            f"is_ascii={metadata['is_ascii']} "
            f"contains_whitespace={metadata['contains_whitespace']} "
            f"chars={metadata['chars']}"
        )


def pypi_release_file_count(version: str) -> int:
    try:
        with urlopen(PYPI_JSON_URL, timeout=20) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise PublishError(f"Unable to read PyPI JSON from {PYPI_JSON_URL}") from exc
    releases = payload.get("releases")
    if not isinstance(releases, dict):
        raise PublishError("PyPI JSON did not include a releases object")
    files = releases.get(version, [])
    return len(files) if isinstance(files, list) else 0


def wait_for_pypi_files(version: str, expected_at_least: int, timeout: float) -> int:
    deadline = time.monotonic() + timeout
    last_count = 0
    while True:
        last_count = pypi_release_file_count(version)
        if last_count >= expected_at_least:
            return last_count
        if time.monotonic() >= deadline:
            raise PublishError(
                f"Timed out waiting for PyPI to expose {expected_at_least} files for {version}; last_count={last_count}"
            )
        time.sleep(5.0)


def twine_upload(args: argparse.Namespace, wheels: Sequence[Path], token: str) -> None:
    command = [args.python, "-m", "twine", "upload", "--non-interactive"]
    if args.skip_existing:
        command.append("--skip-existing")
    if args.repository_url:
        command.extend(["--repository-url", args.repository_url])
    command.extend(str(path) for path in wheels)

    env = os.environ.copy()
    env["TWINE_USERNAME"] = "__token__"
    env["TWINE_PASSWORD"] = token
    run(command, env=env)


def smoke_pypi_install(python: str, version: str) -> None:
    script = textwrap.dedent(
        f"""
        import spiraltorch as st
        import os
        from pathlib import Path
        from spiraltorch.nn import Linear

        smoke_target = Path(os.environ["SPIRALTORCH_SMOKE_TARGET"]).resolve()
        module_path = Path(st.__file__).resolve()
        assert smoke_target in module_path.parents, module_path

        assert st.__version__ == {version!r}, st.__version__
        layer = Linear(2, 2, name="pypi_smoke")
        state = dict(layer.state_dict())
        assert "pypi_smoke::weight" in state
        assert "pypi_smoke::bias" in state
        info = st.build_info()
        assert info.get("package") == "spiraltorch-py", info
        assert info.get("version") == {version!r}, info
        print("pypi smoke ok", st.__version__, sorted(state), flush=True)
        """
    )
    with tempfile.TemporaryDirectory(prefix="spiraltorch-pypi-smoke-") as target:
        run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--target",
                target,
                f"{PACKAGE}=={version}",
            ]
        )
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONPATH"] = target
        env["SPIRALTORCH_SMOKE_TARGET"] = target
        with tempfile.TemporaryDirectory(prefix="spiraltorch-pypi-smoke-cwd-") as run_cwd:
            run([python, "-c", script], env=env, cwd=Path(run_cwd))


def main() -> int:
    args = parse_args()
    wheels = discover_wheels(args.dist)
    version = validate_versions(wheels, args.expected_version)

    log(f"discovered_wheels={len(wheels)}")
    log(f"version={version}")
    for wheel in wheels:
        log(f"wheel={wheel}")

    twine_check(args.python, wheels)
    if args.github_release_tag:
        verify_github_release_wheel_checksums(
            wheels,
            repo=args.github_release_repo,
            tag=args.github_release_tag,
            token_env=args.github_token_env,
        )

    token, token_metadata = read_token(args.token_source, args.token_env)
    redacted_token_metadata = dict(token_metadata)
    log("token_metadata=" + json.dumps(redacted_token_metadata, sort_keys=True))

    pypi_files = pypi_release_file_count(version)
    log(f"pypi_files_for_{version}={pypi_files}")
    if pypi_files and not args.skip_existing:
        raise PublishError(
            f"PyPI already exposes {pypi_files} file(s) for {version}; rerun with --skip-existing to tolerate this."
        )

    if args.dry_run:
        log(f"dry_run=true upload_ready={token_is_upload_ready(token_metadata)}")
        return 0

    require_upload_token(token, token_metadata)
    twine_upload(args, wheels, token)
    visible_files = wait_for_pypi_files(version, len(wheels), args.verify_timeout)
    log(f"pypi_visible_files_for_{version}={visible_files}")
    if not args.no_smoke:
        smoke_pypi_install(args.python, version)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PublishError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
