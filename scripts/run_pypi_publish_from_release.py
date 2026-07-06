#!/usr/bin/env python3
"""Preflight and dispatch the GitHub Release-to-PyPI publish workflow.

The default mode is a safe dry-run. Token publishing is refused unless the
`pypi` GitHub Actions environment can see `PYPI_API_TOKEN`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import release_status


WORKFLOW = "publish_pypi_from_release.yml"


class PublishRunError(RuntimeError):
    """Raised for user-actionable workflow dispatch failures."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default=release_status.PACKAGE, help="PyPI package name. Default: spiraltorch.")
    parser.add_argument("--version", help="Release version. Defaults to bindings/st-py/pyproject.toml.")
    parser.add_argument("--release-tag", help="GitHub Release tag. Defaults to v{version}.")
    parser.add_argument(
        "--repo",
        default=release_status.DEFAULT_REPO,
        help="GitHub repository slug. Default: $GITHUB_REPOSITORY or RyoSpiralArchitect/SpiralTorch.",
    )
    parser.add_argument("--ref", default="main", help="Git ref used to run the workflow. Default: main.")
    parser.add_argument("--expected-wheels", type=int, default=3, help="Expected wheel count. Default: 3.")
    parser.add_argument(
        "--publish-method",
        choices=("dry-run", "token", "trusted"),
        default="dry-run",
        help="Workflow publish_method input. Default: dry-run.",
    )
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Pass skip_existing=true to the workflow. Default.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Pass skip_existing=false to the workflow.",
    )
    parser.add_argument(
        "--allow-published",
        action="store_true",
        help="Allow dispatch even when PyPI already exposes this version.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch the dispatched workflow run and return its exit status.",
    )
    parser.add_argument("--watch-interval", type=int, default=10, help="Seconds between watch polls. Default: 10.")
    parser.add_argument(
        "--run-list-delay",
        type=float,
        default=3.0,
        help="Seconds to wait before locating the dispatched run. Default: 3.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Run preflight and print the workflow command without dispatching.",
    )
    parser.add_argument(
        "--no-clipboard",
        action="store_true",
        default=True,
        help="Skip clipboard token probing during preflight. Default.",
    )
    parser.add_argument(
        "--check-clipboard",
        dest="no_clipboard",
        action="store_false",
        help="Allow release_status to inspect pbpaste token shape during preflight.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root used for local version checks. Default: this script's repo.",
    )
    return parser.parse_args()


def status_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        package=args.package,
        version=args.version,
        release_tag=args.release_tag,
        repo=args.repo,
        expected_wheels=args.expected_wheels,
        token_env=release_status.DEFAULT_TOKEN_SECRET,
        github_secret_environment=release_status.DEFAULT_SECRET_ENVIRONMENT,
        no_clipboard=args.no_clipboard,
        dist=None,
        json=False,
        root=args.root,
    )


def ensure_publish_ready(status: dict[str, Any], *, publish_method: str, allow_published: bool) -> None:
    ready = status["ready"]
    if not ready.get("local_versions"):
        raise PublishRunError("Local pyproject/Cargo versions are not aligned.")
    if not ready.get("github_release"):
        raise PublishRunError("GitHub Release assets are not ready for publishing.")
    if publish_method != "dry-run" and ready.get("pypi_published") and not allow_published:
        raise PublishRunError(
            "PyPI already exposes this version; rerun with --allow-published if you intentionally want a workflow rerun."
        )
    if publish_method == "token" and not ready.get("github_token_secret"):
        raise PublishRunError(
            "Token publishing requires the pypi environment secret first: "
            + status["commands"]["token_secret_setup"]
        )


def workflow_run_command(
    *,
    repo: str,
    ref: str,
    tag: str,
    expected_wheels: int,
    publish_method: str,
    skip_existing: bool,
) -> list[str]:
    return [
        "gh",
        "workflow",
        "run",
        WORKFLOW,
        "--repo",
        repo,
        "--ref",
        ref,
        "-f",
        f"release_tag={tag}",
        "-f",
        f"expected_wheels={expected_wheels}",
        "-f",
        f"publish_method={publish_method}",
        "-f",
        f"skip_existing={str(skip_existing).lower()}",
    ]


def run(command: Sequence[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+ " + release_status.shell_join(list(command)), flush=True)
    return subprocess.run(
        list(command),
        check=True,
        capture_output=capture,
        text=True,
    )


def latest_run_id(repo: str, *, workflow: str, branch: str) -> tuple[int, str]:
    proc = run(
        [
            "gh",
            "run",
            "list",
            "--repo",
            repo,
            "--workflow",
            workflow,
            "--branch",
            branch,
            "--limit",
            "1",
            "--json",
            "databaseId,url",
        ],
        capture=True,
    )
    runs = json.loads(proc.stdout)
    if not runs:
        raise PublishRunError(f"No recent runs found for workflow={workflow} branch={branch}")
    run_id = int(runs[0]["databaseId"])
    return run_id, str(runs[0].get("url", ""))


def watch_run(repo: str, run_id: int, *, interval: int) -> None:
    run(
        [
            "gh",
            "run",
            "watch",
            str(run_id),
            "--repo",
            repo,
            "--interval",
            str(interval),
            "--exit-status",
        ]
    )


def main() -> int:
    args = parse_args()
    status = release_status.build_status(status_args(args))
    release_status.print_text(status)
    ensure_publish_ready(status, publish_method=args.publish_method, allow_published=args.allow_published)

    command = workflow_run_command(
        repo=args.repo,
        ref=args.ref,
        tag=status["release_tag"],
        expected_wheels=status["expected_wheels"],
        publish_method=args.publish_method,
        skip_existing=args.skip_existing,
    )
    print("workflow_dispatch_command: " + release_status.shell_join(command), flush=True)
    if args.print_only:
        print("workflow_dispatch=print_only", flush=True)
        return 0

    run(command)
    if args.watch:
        time.sleep(args.run_list_delay)
        run_id, url = latest_run_id(args.repo, workflow=WORKFLOW, branch=args.ref)
        print(f"workflow_run id={run_id} url={url}", flush=True)
        watch_run(args.repo, run_id, interval=args.watch_interval)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PublishRunError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
