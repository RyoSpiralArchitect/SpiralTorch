"""Bounded local transport for native SGD examples. Rust verifies model semantics."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat


def read_limited(path: Path, limit: int) -> bytes:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("expected a regular file, not a link")
    flags = getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with open(path, "rb", opener=lambda p, f: os.open(p, f | flags)) as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= limit:
            raise ValueError("expected a regular file within the native recovery budget")
        raw = stream.read(info.st_size)
        if len(raw) != info.st_size or stream.read(1):
            raise ValueError("file size changed during native recovery")
        return raw


def read_pinned_session(checkpoint: Path, expected_id: str, metadata: Path):
    """Return exact bytes and the caller-selected boundary for from_session_checkpoint_at."""
    from . import NativeCausalLmTrainingSession

    meta = json.loads(read_limited(metadata, 1_048_576))
    keys = ("recorded_data_id", "post_lora_id", "post_geometry_id")
    if (not isinstance(meta, dict) or meta.get("checkpoint_id") != expected_id
            or "completed_updates" not in meta or not all(k in meta for k in keys)):
        raise ValueError("session boundary is missing or belongs to another checkpoint")
    raw = read_limited(checkpoint, NativeCausalLmTrainingSession.MAX_CHECKPOINT_BYTES)
    if "sha256:" + hashlib.sha256(raw).hexdigest() != expected_id:
        raise ValueError("session checkpoint differs from pinned identity")
    return raw, dict(update=meta["completed_updates"], **{k: meta[k] for k in keys})
