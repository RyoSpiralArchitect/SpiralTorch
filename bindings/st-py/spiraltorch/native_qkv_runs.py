"""Private file orchestration; Rust owns QKV updates, data identity and restart.

Use one writer and a filesystem supporting the native no-clobber publication
contract. Stop on any error; never infer a latest checkpoint or retry a segment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

__all__ = ["run_qkv_corpus"]


def _open_new(path):
    return open(path, "x", encoding="utf-8", opener=lambda p, f: os.open(p, f, 0o600))


def _append(stream, record):
    stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def run_qkv_corpus(trainer, directory, *, stop_after=None, save_every=10):
    """Save an initial anchor, native committed segments, then an inference snapshot.

    `stop_after` is an absolute update boundary. A new directory is mandatory.
    Checkpoint publication precedes report/index writes; if either log write
    fails, stop and verify an explicitly selected checkpoint before continuing.
    Missing result.json means this invocation did not finish its output protocol.
    """
    from . import NativeCausalLmQkvTrainer

    if not isinstance(trainer, NativeCausalLmQkvTrainer):
        raise TypeError("expected a native QKV corpus trainer")
    if type(save_every) is not int or not 1 <= save_every <= trainer.MAX_CHECKPOINT_INTERVAL:
        raise ValueError("save_every must be within the native checkpoint interval")
    stop = trainer.total_updates if stop_after is None else stop_after
    if type(stop) is not int or not trainer.updates <= stop <= trainer.total_updates:
        raise ValueError("stop_after must be between current and final update")
    root = Path(directory)
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    start = trainer.updates

    def name(update):
        return f"checkpoint-{update:09d}.bin"

    with _open_new(root / "checkpoints.jsonl") as index, _open_new(root / "steps.jsonl") as log:
        latest = dict(file=name(start), checkpoint_id=trainer.save_checkpoint_new(root / name(start)), updates=start)
        _append(index, latest)
        while trainer.updates < stop:
            end = min(stop, (trainer.updates // save_every + 1) * save_every)
            reports, pin = trainer.advance_to_checkpoint_new(end, root / name(end))
            latest = dict(file=name(end), checkpoint_id=pin, updates=end)
            _append(index, latest)
            for report in reports:
                _append(log, report)
    snapshot = trainer.optimizer_snapshot()
    qkv_pin = snapshot.save_checkpoint_new(root / "qkv.bin")
    result = dict(schema="spiraltorch.native_qkv_run.v1", data_id=trainer.data_id,
                  config=json.loads(trainer.config_json()), initial_update=start,
                  updates=trainer.updates, total_updates=trainer.total_updates,
                  active_windows=trainer.active_windows, exhausted=trainer.updates == trainer.total_updates,
                  final_checkpoint=latest, checkpoint_index="checkpoints.jsonl", step_log="steps.jsonl",
                  qkv=dict(file="qkv.bin", checkpoint_id=qkv_pin, projection_id=snapshot.serving_id,
                           optimizer_steps=snapshot.steps, layer=snapshot.layer),
                  scope="native CPU single-QKV training with frozen base/adapters; not quality evidence")
    # Create this marker only after all required data files have been published.
    # Retain it on I/O errors; callers must still verify its JSON and pinned files.
    with _open_new(root / "result.json") as stream:
        _append(stream, result)
    if os.name == "posix":
        fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return result
