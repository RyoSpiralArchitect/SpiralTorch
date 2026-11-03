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

from .native_session_io import read_limited as _read

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


def step_and_save_observation(trainer, directory):
    """Perform ONE observed step, save its post-update state, then publish its receipt.

    Returns (ordinary native step, file/ID metadata), or None when exhausted.
    Both files are private. Rust commits the live state only after publishing
    both. After an I/O error stop and select a previously completed pair.
    A checkpoint alone is NOT a completed pair.
    """
    if trainer.exhausted:
        return None
    reports, entry = trainer.advance_to_observed_checkpoint_new(trainer.completed_updates + 1, directory)
    return reports[0], entry


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
    return _read(checkpoint, _CHECKPOINT_LIMIT), cid, data_id, post_id


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
