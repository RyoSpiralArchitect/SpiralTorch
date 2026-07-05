#!/usr/bin/env python3
"""Install a PyPI API token as the GitHub Actions `pypi` environment secret.

The helper is intentionally narrow: it validates only token shape, never prints
the token value, and passes the secret to `gh secret set` through stdin.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys


DEFAULT_REPO = os.environ.get("GITHUB_REPOSITORY", "RyoSpiralArchitect/SpiralTorch")
DEFAULT_ENVIRONMENT = "pypi"
DEFAULT_SECRET_NAME = "PYPI_API_TOKEN"


class SecretSetupError(RuntimeError):
    """Raised for user-actionable secret setup failures."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"GitHub repository slug. Default: {DEFAULT_REPO}.")
    parser.add_argument(
        "--environment",
        default=DEFAULT_ENVIRONMENT,
        help=f"GitHub Actions environment for the secret. Default: {DEFAULT_ENVIRONMENT}.",
    )
    parser.add_argument(
        "--secret-name",
        default=DEFAULT_SECRET_NAME,
        help=f"GitHub Actions secret name. Default: {DEFAULT_SECRET_NAME}.",
    )
    parser.add_argument(
        "--token-source",
        choices=("prompt", "env", "clipboard", "stdin"),
        default="prompt",
        help="Where to read the PyPI API token from. Default: hidden prompt.",
    )
    parser.add_argument(
        "--token-env",
        default=DEFAULT_SECRET_NAME,
        help=f"Environment variable used when --token-source=env. Default: {DEFAULT_SECRET_NAME}.",
    )
    parser.add_argument("--gh", default="gh", help="GitHub CLI executable. Default: gh.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate token shape and print the target secret location without storing it.",
    )
    return parser.parse_args()


def token_metadata(raw: str, *, source: str) -> dict[str, object]:
    token = raw.strip()
    contains_whitespace = any(ch.isspace() for ch in token)
    is_ascii = all(ord(ch) < 128 for ch in token)
    return {
        "source": source,
        "chars": len(token),
        "trimmed": token != raw,
        "starts_with_pypi": token.startswith("pypi-"),
        "is_ascii": is_ascii,
        "contains_whitespace": contains_whitespace,
        "upload_ready": bool(token and token.startswith("pypi-") and is_ascii and not contains_whitespace),
    }


def read_token(source: str, env_name: str) -> tuple[str, dict[str, object]]:
    if source == "prompt":
        raw = getpass.getpass("PyPI token for spiraltorch (hidden): ")
    elif source == "env":
        raw = os.environ.get(env_name, "")
    elif source == "clipboard":
        try:
            proc = subprocess.run(["pbpaste"], check=False, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError) as exc:
            raise SecretSetupError(f"Unable to read clipboard with pbpaste: {exc}") from exc
        if proc.returncode != 0:
            error = (proc.stderr or "").strip() or f"pbpaste exited {proc.returncode}"
            raise SecretSetupError(f"Unable to read clipboard with pbpaste: {error}")
        raw = proc.stdout
    elif source == "stdin":
        raw = sys.stdin.read()
    else:
        raise SecretSetupError(f"Unsupported token source: {source}")
    token = raw.strip()
    return token, token_metadata(raw, source=source if source != "env" else f"env:{env_name}")


def require_upload_token(metadata: dict[str, object]) -> None:
    if metadata.get("upload_ready"):
        return
    raise SecretSetupError(
        "PyPI token is not upload-ready: "
        f"starts_with_pypi={metadata.get('starts_with_pypi')} "
        f"is_ascii={metadata.get('is_ascii')} "
        f"contains_whitespace={metadata.get('contains_whitespace')} "
        f"chars={metadata.get('chars')}"
    )


def secret_set_command(gh: str, *, repo: str, environment: str, secret_name: str) -> list[str]:
    return [
        gh,
        "secret",
        "set",
        secret_name,
        "--repo",
        repo,
        "--env",
        environment,
        "--app",
        "actions",
    ]


def set_github_secret(
    token: str,
    *,
    gh: str,
    repo: str,
    environment: str,
    secret_name: str,
) -> subprocess.CompletedProcess[str]:
    command = secret_set_command(gh, repo=repo, environment=environment, secret_name=secret_name)
    return subprocess.run(command, input=token, text=True, capture_output=True, check=False)


def sanitized_failure(proc: subprocess.CompletedProcess[str]) -> str:
    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    payload = {
        "returncode": proc.returncode,
        "stdout_chars": len(stdout),
        "stderr": stderr[:500],
    }
    return json.dumps(payload, sort_keys=True)


def main() -> int:
    args = parse_args()
    token, metadata = read_token(args.token_source, args.token_env)
    print("token_metadata=" + json.dumps(metadata, sort_keys=True), flush=True)
    require_upload_token(metadata)

    target = {
        "repo": args.repo,
        "environment": args.environment,
        "secret_name": args.secret_name,
    }
    if args.dry_run:
        print("secret_store=dry_run " + json.dumps(target, sort_keys=True), flush=True)
        return 0

    proc = set_github_secret(
        token,
        gh=args.gh,
        repo=args.repo,
        environment=args.environment,
        secret_name=args.secret_name,
    )
    if proc.returncode != 0:
        raise SecretSetupError("gh secret set failed: " + sanitized_failure(proc))
    print("secret_store=ok " + json.dumps(target, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SecretSetupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
