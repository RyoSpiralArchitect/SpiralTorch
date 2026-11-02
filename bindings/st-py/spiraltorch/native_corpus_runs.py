"""Local recovery points for Rust-owned, observed native causal-LM updates.

Use a private, caller-owned directory on a filesystem supporting hard links.
The trainer and directory must each have one writer. No implicit retry, latest
checkpoint selection, model download, or foreign training backend.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

__all__ = [
    "step_and_save_observation",
    "restore_observed_checkpoint",
    "load_observed_model",
]

_SCHEMA = "spiraltorch.native_corpus.observed_checkpoint.v1"
_OBSERVED = "spiraltorch.native_lm.observed_training_step.v1"
_RECEIPT_LIMIT = 4 * 1024 * 1024
_CHECKPOINT_LIMIT = 128 * 1024 * 1024


def _pin(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _paths(directory, update):
    if type(update) is not int or not 1 <= update < 2**64:
        raise ValueError("select a positive completed update")
    root = Path(directory)
    mode = root.lstat().st_mode
    if not stat.S_ISDIR(mode) or (os.name == "posix" and mode & 0o077):
        raise ValueError("use a private run directory (0700 on POSIX)")
    return (
        root / f"checkpoint-{update:06d}.json",
        root / f"observation-{update:06d}.json",
    )


def _publish(path, raw):
    # Only a fully closed file can acquire the final name; never replace even
    # an orphan left by an earlier interrupted attempt.
    fd, pending = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(raw)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.link(pending, path)
    finally:
        os.unlink(pending)


def _read(path, limit):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("expected a regular file, not a link")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("file exceeds the native recovery budget")
    return raw


def step_and_save_observation(trainer, directory):
    """Perform ONE observed step, save its post-update state, then publish its receipt.

    Returns (ordinary native step, file/ID metadata), or None when exhausted.
    Both files are private. After an I/O error the in-memory step may already
    have advanced: stop, do not retry, and select a previously completed pair.
    A checkpoint alone is NOT a completed pair.
    """
    if trainer.exhausted:
        return None
    cursor = trainer.completed_updates + 1
    checkpoint, receipt = _paths(directory, cursor)
    if any(os.path.lexists(p) for p in (checkpoint, receipt)):
        raise FileExistsError("observed update output already exists")
    observed = trainer.step_observed()
    report = observed.step
    if report.update != cursor or trainer.completed_updates != cursor:
        raise ValueError("observed update does not own the selected trainer cursor")
    payload, checkpoint_id = trainer.checkpoint_with_id()
    state = payload.encode()
    if len(state) > _CHECKPOINT_LIMIT or _pin(state) != checkpoint_id:
        raise ValueError("invalid native checkpoint bytes or identity")
    saved = {"file": checkpoint.name, "checkpoint_id": checkpoint_id, "updates": cursor}
    record = {
        "schema": _SCHEMA,
        "checkpoint": saved,
        "observation": json.loads(observed.to_json()),
    }
    raw = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(raw) > _RECEIPT_LIMIT:
        raise ValueError("observation exceeds receipt budget")
    _publish(checkpoint, state)
    _publish(receipt, raw)
    return report, {
        **saved,
        "observation_file": receipt.name,
        "observation_id": _pin(raw),
    }


def _observed_pair(directory, update, expected_observation_id, base):
    checkpoint, receipt = _paths(directory, update)
    raw = _read(receipt, _RECEIPT_LIMIT)
    if _pin(raw) != expected_observation_id:
        raise ValueError("observation receipt differs from its pin")
    record = json.loads(raw)
    try:
        saved, observation = record["checkpoint"], record["observation"]
        step = observation["step"]
        cid, data_id, post_id = (
            saved["checkpoint_id"],
            observation["data_id"],
            step["post_geometry_artifact_id"],
        )
        schema, observed_schema = record["schema"], observation["schema"]
        file, cursor, step_cursor, base_id = (
            saved["file"],
            saved["updates"],
            step["update"],
            observation["base_model_identity"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid observation receipt structure") from exc
    if (
        schema != _SCHEMA
        or observed_schema != _OBSERVED
        or file != checkpoint.name
        or type(cursor) is not int
        or type(step_cursor) is not int
        or cursor != update
        or step_cursor != update
        or base_id != base.base_model_identity
        or not all(isinstance(v, str) for v in (cid, data_id, post_id))
    ):
        raise ValueError("observation receipt does not identify this update and base")
    return _read(checkpoint, _CHECKPOINT_LIMIT).decode(), cid, data_id, post_id


def load_observed_model(directory, update, *, expected_observation_id, base):
    """Load a model for inference without reading training tokens or claiming corpus verification."""
    from . import NativeCausalLm

    raw, cid, data_id, post_id = _observed_pair(
        directory, update, expected_observation_id, base
    )
    return NativeCausalLm.from_training_checkpoint_at(
        base,
        raw,
        cid,
        update=update,
        recorded_data_id=data_id,
        post_geometry_id=post_id,
    )


def restore_observed_checkpoint(
    directory, update, *, expected_observation_id, base, token_documents, labels=None
):
    """Restore an explicitly selected, receipt-pinned pair and verify the supplied ordered corpus."""
    from . import NativeCausalLmTrainer

    raw, cid, data_id, post_id = _observed_pair(
        directory, update, expected_observation_id, base
    )
    trainer = NativeCausalLmTrainer.from_json(
        base, token_documents, raw, cid, labels=labels
    )
    if (
        trainer.completed_updates != update
        or trainer.data_id != data_id
        or trainer.model.geometry_artifact_id != post_id
    ):
        raise ValueError("observed step and native checkpoint do not agree")
    return trainer
