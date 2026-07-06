#!/usr/bin/env python3
"""Summarize SpiralTorch release readiness without exposing secrets."""
from __future__ import annotations

import argparse
from email.parser import Parser
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

PACKAGE = "spiraltorch"
DEFAULT_REPO = os.environ.get("GITHUB_REPOSITORY", "RyoSpiralArchitect/SpiralTorch")
DEFAULT_SECRET_ENVIRONMENT = "pypi"
DEFAULT_TOKEN_SECRET = "PYPI_API_TOKEN"
REQUIRED_WHEEL_PAYLOADS = (
    "spiraltorch/__init__.pyi",
    "spiraltorch/py.typed",
    "spiraltorch/spiralk.pyi",
)


class StatusError(RuntimeError):
    """Raised when release status cannot be computed from local inputs."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default=PACKAGE, help="PyPI package name. Default: spiraltorch.")
    parser.add_argument(
        "--version",
        help="Release version. Defaults to bindings/st-py/pyproject.toml project.version.",
    )
    parser.add_argument(
        "--release-tag",
        help="GitHub Release tag. Defaults to v{version}.",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="GitHub repository slug. Default: $GITHUB_REPOSITORY or RyoSpiralArchitect/SpiralTorch.",
    )
    parser.add_argument(
        "--expected-wheels",
        type=int,
        default=3,
        help="Expected number of release/PyPI wheel files. Default: 3.",
    )
    parser.add_argument(
        "--dist",
        type=Path,
        help=(
            "Optional local wheel directory to inspect for version metadata and "
            "required type payloads before publishing."
        ),
    )
    parser.add_argument(
        "--token-env",
        default=DEFAULT_TOKEN_SECRET,
        help="Local token environment variable to inspect without printing. Default: PYPI_API_TOKEN.",
    )
    parser.add_argument(
        "--github-secret-environment",
        default=DEFAULT_SECRET_ENVIRONMENT,
        help="GitHub Actions environment whose secrets should be checked with gh. Default: pypi.",
    )
    parser.add_argument(
        "--no-clipboard",
        action="store_true",
        help="Skip the pbpaste token-shape probe.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the status object as JSON instead of human-readable lines.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root used for local version checks. Default: this script's repo.",
    )
    return parser.parse_args()


def toml_section_value(path: Path, section: str, key: str) -> str | None:
    """Read a simple quoted string value from a top-level TOML section."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise StatusError(f"Unable to read {path}") from exc
    current_section: str | None = None
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line.strip("[]").strip()
            continue
        if current_section != section or "=" not in line:
            continue
        raw_key, raw_value = line.split("=", 1)
        if raw_key.strip() != key:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            return value[1:-1]
    return None


def local_versions(root: Path) -> dict[str, Any]:
    pyproject_version = toml_section_value(root / "bindings" / "st-py" / "pyproject.toml", "project", "version")
    cargo_version = toml_section_value(root / "bindings" / "st-py" / "Cargo.toml", "package", "version")
    return {
        "pyproject": pyproject_version,
        "cargo": cargo_version,
        "consistent": bool(pyproject_version and pyproject_version == cargo_version),
    }


def download_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "spiraltorch-release-status"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise StatusError(f"Unable to download JSON from {url}") from exc
    if not isinstance(payload, dict):
        raise StatusError(f"{url} did not return a JSON object")
    return payload


def pypi_status(package: str, version: str, expected_wheels: int) -> dict[str, Any]:
    payload = download_json(f"https://pypi.org/pypi/{package}/json")
    info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
    releases = payload.get("releases") if isinstance(payload.get("releases"), dict) else {}
    files = releases.get(version, []) if isinstance(releases, dict) else []
    if not isinstance(files, list):
        files = []
    filenames = sorted(
        file_info["filename"]
        for file_info in files
        if isinstance(file_info, dict) and isinstance(file_info.get("filename"), str)
    )
    wheels = [name for name in filenames if isinstance(name, str) and name.endswith(".whl")]
    return {
        "latest": info.get("version") if isinstance(info.get("version"), str) else None,
        "file_count": len(filenames),
        "wheel_count": len(wheels),
        "wheel_names": wheels,
        "published": len(wheels) >= expected_wheels,
    }


def local_wheel_payload_status(
    dist: Path | None,
    *,
    package: str,
    version: str,
    expected_wheels: int,
) -> dict[str, Any]:
    if dist is None:
        return {
            "checked": False,
            "ready": None,
            "wheel_count": 0,
            "wheel_names": [],
            "required_payloads": list(REQUIRED_WHEEL_PAYLOADS),
            "missing_payloads": {},
            "version_mismatches": {},
            "metadata_errors": {},
        }

    if not dist.is_dir():
        return {
            "checked": True,
            "ready": False,
            "error": f"dist directory not found: {dist}",
            "wheel_count": 0,
            "wheel_names": [],
            "required_payloads": list(REQUIRED_WHEEL_PAYLOADS),
            "missing_payloads": {},
            "version_mismatches": {},
            "metadata_errors": {},
        }

    wheels = sorted(path for path in dist.rglob(f"{package}-*.whl") if path.is_file())
    missing_payloads: dict[str, list[str]] = {}
    version_mismatches: dict[str, str | None] = {}
    metadata_errors: dict[str, str] = {}
    for wheel in wheels:
        try:
            with zipfile.ZipFile(wheel) as archive:
                names = set(archive.namelist())
                metadata_names = [
                    name for name in names
                    if name.endswith(".dist-info/METADATA")
                ]
                if len(metadata_names) != 1:
                    metadata_errors[wheel.name] = (
                        f"expected one METADATA file, found {len(metadata_names)}"
                    )
                    continue
                missing = sorted(set(REQUIRED_WHEEL_PAYLOADS) - names)
                if missing:
                    missing_payloads[wheel.name] = missing
                metadata = Parser().parsestr(archive.read(metadata_names[0]).decode("utf-8"))
        except (OSError, zipfile.BadZipFile, UnicodeDecodeError) as exc:
            metadata_errors[wheel.name] = f"{exc.__class__.__name__}: {exc}"
            continue

        actual = metadata.get("Version")
        if actual != version:
            version_mismatches[wheel.name] = actual

    ready = (
        len(wheels) == expected_wheels
        and not missing_payloads
        and not version_mismatches
        and not metadata_errors
    )
    return {
        "checked": True,
        "ready": ready,
        "wheel_count": len(wheels),
        "wheel_names": [wheel.name for wheel in wheels],
        "required_payloads": list(REQUIRED_WHEEL_PAYLOADS),
        "missing_payloads": missing_payloads,
        "version_mismatches": version_mismatches,
        "metadata_errors": metadata_errors,
    }


def github_release_status(repo: str, tag: str, package: str, version: str, expected_wheels: int) -> dict[str, Any]:
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    try:
        payload = download_json(url)
    except StatusError as exc:
        return {
            "exists": False,
            "error": str(exc),
            "wheel_count": 0,
            "wheel_names": [],
            "has_wheels_sha256": False,
            "ready": False,
        }
    assets = payload.get("assets") if isinstance(payload.get("assets"), list) else []
    asset_names = sorted(
        asset["name"] for asset in assets if isinstance(asset, dict) and isinstance(asset.get("name"), str)
    )
    wheel_prefix = f"{package}-{version}-"
    wheels = [
        name
        for name in asset_names
        if isinstance(name, str) and name.startswith(wheel_prefix) and name.endswith(".whl")
    ]
    has_wheels_sha256 = "wheels.sha256" in asset_names
    is_draft = bool(payload.get("draft"))
    is_prerelease = bool(payload.get("prerelease"))
    ready = len(wheels) == expected_wheels and has_wheels_sha256 and not is_draft and not is_prerelease
    return {
        "exists": True,
        "draft": is_draft,
        "prerelease": is_prerelease,
        "asset_count": len(asset_names),
        "wheel_count": len(wheels),
        "wheel_names": wheels,
        "has_wheels_sha256": has_wheels_sha256,
        "ready": ready,
    }


def token_metadata(raw: str, *, source: str) -> dict[str, Any]:
    token = raw.strip()
    contains_whitespace = any(ch.isspace() for ch in token)
    is_ascii = all(ord(ch) < 128 for ch in token)
    starts_with_pypi = token.startswith("pypi-")
    return {
        "source": source,
        "chars": len(token),
        "trimmed": token != raw,
        "starts_with_pypi": starts_with_pypi,
        "is_ascii": is_ascii,
        "contains_whitespace": contains_whitespace,
        "upload_ready": bool(token and starts_with_pypi and is_ascii and not contains_whitespace),
    }


def env_token_status(env_name: str) -> dict[str, Any]:
    return token_metadata(os.environ.get(env_name, ""), source=f"env:{env_name}")


def clipboard_token_status() -> dict[str, Any]:
    try:
        proc = subprocess.run(["pbpaste"], check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "source": "clipboard:pbpaste",
            "available": False,
            "error": str(exc),
            "upload_ready": False,
        }
    metadata = token_metadata(proc.stdout, source="clipboard:pbpaste")
    metadata["available"] = proc.returncode == 0
    if proc.returncode != 0:
        metadata["error"] = (proc.stderr or "").strip() or f"pbpaste exited {proc.returncode}"
    return metadata


def github_secret_status(secret_name: str, *, environment: str) -> dict[str, Any]:
    command = ["gh", "secret", "list", "--env", environment, "--app", "actions"]
    try:
        proc = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "source": f"github-env:{environment}",
            "available": False,
            "present": False,
            "error": str(exc),
        }
    names = []
    if proc.returncode == 0:
        for line in proc.stdout.splitlines():
            fields = line.split()
            if fields:
                names.append(fields[0])
    return {
        "source": f"github-env:{environment}",
        "available": proc.returncode == 0,
        "present": secret_name in names,
        "secret_name": secret_name,
        "error": "" if proc.returncode == 0 else (proc.stderr or "").strip(),
    }


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def token_secret_setup_command(*, secret_name: str, environment: str) -> str:
    command = [
        "python",
        "scripts/configure_pypi_token_secret.py",
        "--token-source",
        "prompt",
    ]
    if secret_name != DEFAULT_TOKEN_SECRET:
        command.extend(["--secret-name", secret_name])
    if environment != DEFAULT_SECRET_ENVIRONMENT:
        command.extend(["--environment", environment])
    return shell_join(command)


def token_secret_setup_stdin_command(*, secret_name: str, environment: str) -> str:
    command = [
        "python",
        "scripts/configure_pypi_token_secret.py",
        "--token-source",
        "stdin",
    ]
    if secret_name != DEFAULT_TOKEN_SECRET:
        command.extend(["--secret-name", secret_name])
    if environment != DEFAULT_SECRET_ENVIRONMENT:
        command.extend(["--environment", environment])
    return (
        "( old_stty=$(stty -g); "
        "trap 'stty \"$old_stty\"; unset PYPI_TOKEN' EXIT; "
        "printf 'PyPI token for spiraltorch (hidden): '; "
        "stty -echo; "
        "IFS= read -r PYPI_TOKEN; "
        "stty \"$old_stty\"; "
        "printf '\\n'; "
        "printf '%s' \"$PYPI_TOKEN\" | "
        f"{shell_join(command)} )"
    )


def publish_from_release_command(*, tag: str, expected_wheels: int, publish_method: str) -> str:
    return shell_join(
        [
            "gh",
            "workflow",
            "run",
            "publish_pypi_from_release.yml",
            "--ref",
            "main",
            "-f",
            f"release_tag={tag}",
            "-f",
            f"expected_wheels={expected_wheels}",
            "-f",
            f"publish_method={publish_method}",
            "-f",
            "skip_existing=true",
        ]
    )


def trusted_publisher_claims(
    repo: str,
    *,
    environment: str,
    workflow: str = "publish_pypi_from_release.yml",
) -> dict[str, str]:
    return {
        "sub": f"repo:{repo}:environment:{environment}",
        "repository": repo,
        "workflow_ref": f"{repo}/.github/workflows/{workflow}@refs/heads/main",
        "environment": environment,
    }


def build_status(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    versions = local_versions(root)
    version = args.version or versions["pyproject"]
    if not version:
        raise StatusError("Unable to infer --version from bindings/st-py/pyproject.toml")
    tag = args.release_tag or f"v{version}"
    release = github_release_status(args.repo, tag, args.package, version, args.expected_wheels)
    local_wheels = local_wheel_payload_status(
        args.dist,
        package=args.package,
        version=version,
        expected_wheels=args.expected_wheels,
    )
    pypi = pypi_status(args.package, version, args.expected_wheels)
    tokens: dict[str, Any] = {
        "env": env_token_status(args.token_env),
        "github_environment_secret": github_secret_status(args.token_env, environment=args.github_secret_environment),
        "prompt_available": True,
    }
    if not args.no_clipboard:
        tokens["clipboard"] = clipboard_token_status()

    version_consistent = bool(versions["consistent"] and versions["pyproject"] == version)
    local_ready = bool(version_consistent)
    local_wheel_payloads_ready = bool(
        local_wheels.get("ready")
        if local_wheels.get("checked")
        else True
    )
    release_ready = bool(release.get("ready"))
    pypi_published = bool(pypi.get("published"))
    token_ready = bool(tokens["env"].get("upload_ready")) or bool(tokens.get("clipboard", {}).get("upload_ready"))
    github_secret_ready = bool(tokens["github_environment_secret"].get("present"))
    token_secret_command = token_secret_setup_command(
        secret_name=args.token_env,
        environment=args.github_secret_environment,
    )
    token_secret_stdin_command = token_secret_setup_stdin_command(
        secret_name=args.token_env,
        environment=args.github_secret_environment,
    )
    token_publish_command = publish_from_release_command(
        tag=tag,
        expected_wheels=args.expected_wheels,
        publish_method="token",
    )
    trusted_publish_command = publish_from_release_command(
        tag=tag,
        expected_wheels=args.expected_wheels,
        publish_method="trusted",
    )
    if pypi_published:
        next_action = "verify published PyPI wheels against the GitHub Release manifest"
    elif not local_ready:
        next_action = "align local pyproject/Cargo versions before publishing"
    elif not local_wheel_payloads_ready:
        next_action = "fix local wheel metadata/type payloads before publishing"
    elif not release_ready:
        next_action = "fix or rebuild the GitHub Release wheel assets before publishing"
    elif github_secret_ready:
        next_action = token_publish_command
    elif token_ready:
        next_action = "run scripts/publish_pypi_wheels.py with the ready local token source"
    else:
        next_action = f"{token_secret_command} OR token_secret_setup_stdin OR configure PyPI Trusted Publishing"

    return {
        "package": args.package,
        "version": version,
        "release_tag": tag,
        "repo": args.repo,
        "expected_wheels": args.expected_wheels,
        "local_versions": versions,
        "local_wheels": local_wheels,
        "github_release": release,
        "pypi": pypi,
        "tokens": tokens,
        "commands": {
            "token_secret_setup": token_secret_command,
            "token_secret_setup_stdin": token_secret_stdin_command,
            "publish_token_workflow": token_publish_command,
            "publish_trusted_workflow": trusted_publish_command,
        },
        "trusted_publisher": trusted_publisher_claims(args.repo, environment=args.github_secret_environment),
        "ready": {
            "local_versions": local_ready,
            "local_wheel_payloads": local_wheel_payloads_ready,
            "github_release": release_ready,
            "pypi_published": pypi_published,
            "local_token": token_ready,
            "github_token_secret": github_secret_ready,
        },
        "next_action": next_action,
    }


def yes_no(value: object) -> str:
    return "yes" if value else "no"


def print_text(status: dict[str, Any]) -> None:
    versions = status["local_versions"]
    local_wheels = status["local_wheels"]
    release = status["github_release"]
    pypi = status["pypi"]
    tokens = status["tokens"]
    ready = status["ready"]
    print(f"release_status package={status['package']} version={status['version']} tag={status['release_tag']}")
    print(
        "local_versions "
        f"pyproject={versions.get('pyproject')} cargo={versions.get('cargo')} "
        f"consistent={yes_no(ready['local_versions'])}"
    )
    if local_wheels.get("checked"):
        print(
            "local_wheels "
            f"ready={yes_no(ready['local_wheel_payloads'])} "
            f"wheels={local_wheels.get('wheel_count', 0)}/{status['expected_wheels']} "
            f"type_payloads={len(local_wheels.get('required_payloads', []))} "
            f"missing_payload_wheels={len(local_wheels.get('missing_payloads', {}))} "
            f"version_mismatch_wheels={len(local_wheels.get('version_mismatches', {}))} "
            f"metadata_error_wheels={len(local_wheels.get('metadata_errors', {}))}"
        )
    print(
        "github_release "
        f"exists={yes_no(release.get('exists'))} ready={yes_no(ready['github_release'])} "
        f"wheels={release.get('wheel_count', 0)}/{status['expected_wheels']} "
        f"wheels_sha256={yes_no(release.get('has_wheels_sha256'))} "
        f"draft={yes_no(release.get('draft'))} prerelease={yes_no(release.get('prerelease'))}"
    )
    print(
        "pypi "
        f"latest={pypi.get('latest')} files_for_version={pypi.get('file_count')} "
        f"wheels_for_version={pypi.get('wheel_count')}/{status['expected_wheels']} "
        f"published={yes_no(ready['pypi_published'])}"
    )
    env_token = tokens["env"]
    clipboard = tokens.get("clipboard")
    gh_secret = tokens["github_environment_secret"]
    clipboard_state = "skipped"
    if isinstance(clipboard, dict):
        clipboard_state = f"ready={yes_no(clipboard.get('upload_ready'))} chars={clipboard.get('chars', 0)}"
    print(
        "token_readiness "
        f"env_ready={yes_no(env_token.get('upload_ready'))} env_chars={env_token.get('chars', 0)} "
        f"clipboard_{clipboard_state} "
        f"github_env_secret_present={yes_no(gh_secret.get('present'))} "
        "prompt_available=yes"
    )
    commands = status.get("commands", {})
    if isinstance(commands, dict):
        token_secret_setup = commands.get("token_secret_setup")
        token_secret_setup_stdin = commands.get("token_secret_setup_stdin")
        publish_token_workflow = commands.get("publish_token_workflow")
        publish_trusted_workflow = commands.get("publish_trusted_workflow")
        if token_secret_setup:
            print(f"token_secret_setup: {token_secret_setup}")
        if token_secret_setup_stdin:
            print(f"token_secret_setup_stdin: {token_secret_setup_stdin}")
        if publish_token_workflow:
            print(f"publish_token_workflow: {publish_token_workflow}")
        if publish_trusted_workflow:
            print(f"publish_trusted_workflow: {publish_trusted_workflow}")
    trusted_publisher = status.get("trusted_publisher", {})
    if isinstance(trusted_publisher, dict):
        print(
            "trusted_publisher "
            f"sub={trusted_publisher.get('sub')} "
            f"workflow_ref={trusted_publisher.get('workflow_ref')} "
            f"environment={trusted_publisher.get('environment')}"
        )
    print(f"next_action: {status['next_action']}")


def main() -> int:
    args = parse_args()
    try:
        status = build_status(args)
    except StatusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print_text(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
