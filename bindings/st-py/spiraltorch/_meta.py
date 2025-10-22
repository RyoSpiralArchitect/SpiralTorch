"""Build metadata helpers for SpiralTorch Python bindings."""

from __future__ import annotations

import datetime as _datetime
import getpass as _getpass
import hashlib as _hashlib
import json as _json
import os as _os
import platform as _platform
import subprocess as _subprocess
import sys as _sys
import uuid as _uuid
from pathlib import Path as _Path
from types import MappingProxyType as _MappingProxyType
from typing import Any as _Any

_BUILD_PREFIX = "RyoST"
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]


def _utc_timestamp() -> str:
    return _datetime.datetime.utcnow().replace(tzinfo=_datetime.timezone.utc).isoformat()


def _git_capture(*args: str) -> str | None:
    try:
        result = _subprocess.run(
            ("git", *args),
            cwd=_PROJECT_ROOT,
            check=False,
            stdout=_subprocess.PIPE,
            stderr=_subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )
    except (FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    output = (result.stdout or "").strip()
    return output or None


def _git_is_dirty() -> bool | None:
    status = _git_capture("status", "--porcelain")
    if status is None:
        return None
    return any(line.strip() for line in status.splitlines())


_timestamp = _utc_timestamp()
_seed = _uuid.uuid4().hex[:12]

BUILD_ID = f"{_BUILD_PREFIX}-{_platform.node()}-{_timestamp}-{_seed}"

_git_commit = _git_capture("rev-parse", "HEAD")
_git_tag = _git_capture("describe", "--tags", "--always")
_git_dirty = _git_is_dirty()

_pkg_version = None
try:  # pragma: no cover - metadata lookup optional
    from importlib import metadata as _metadata

    try:
        _pkg_version = _metadata.version("spiraltorch")
    except _metadata.PackageNotFoundError:
        _pkg_version = None
except Exception:  # pragma: no cover - very defensive
    _pkg_version = None

_BUILD_MANIFEST_MUT: dict[str, _Any] = {
    "id": BUILD_ID,
    "timestamp": _timestamp,
    "seed": _seed,
    "user": _getpass.getuser(),
    "host": _platform.node(),
    "platform": {
        "python": _sys.version.split()[0],
        "implementation": _platform.python_implementation(),
        "platform": _platform.platform(),
    },
    "pkg": {
        "name": "spiraltorch",
        "version": _pkg_version,
    },
    "git": {
        "commit": _git_commit,
        "describe": _git_tag,
        "dirty": _git_dirty,
    },
    "environment": {
        "build_host": _os.environ.get("HOSTNAME") or _os.environ.get("COMPUTERNAME"),
        "build_user": _os.environ.get("USER") or _os.environ.get("USERNAME"),
    },
}

BUILD_MANIFEST = _MappingProxyType(_BUILD_MANIFEST_MUT)
BUILD_MANIFEST_JSON = _json.dumps(_BUILD_MANIFEST_MUT, sort_keys=True, separators=(",", ":"))
BUILD_FINGERPRINT = "sha256:" + _hashlib.sha256(BUILD_MANIFEST_JSON.encode("utf-8")).hexdigest()

_LICENSE_SENTINEL = "SpiralTorch::Generated under AGPL-3.0-or-later (c) Ryo SpiralArchitect, 2025"

__all__ = [
    "BUILD_ID",
    "BUILD_FINGERPRINT",
    "BUILD_MANIFEST",
    "BUILD_MANIFEST_JSON",
]
