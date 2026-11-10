"""Private file orchestration; Rust owns QKV updates, data identity and restart.

Use one writer and a filesystem supporting the native no-clobber publication
contract. Stop on any error; never infer a latest checkpoint or retry a segment.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

__all__ = ["run_qkv_corpus", "verify_qkv_run"]


def _read_regular(root, name, limit):
    if not stat.S_ISREG((root / name).lstat().st_mode):
        raise ValueError(f"invalid QKV run file: {name}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(root / name, flags)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError(f"invalid QKV run file: {name}")
        data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError(f"oversize QKV run file: {name}")
        return data


def _records(root, name):
    if not stat.S_ISREG((root / name).lstat().st_mode):
        raise ValueError(f"invalid QKV run log: {name}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(root / name, flags)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 1 << 30:
            raise ValueError(f"invalid QKV run log: {name}")
        while raw := stream.readline((1 << 20) + 1):
            if len(raw) > 1 << 20 or not raw.endswith(b"\n"):
                raise ValueError(f"truncated or oversize QKV run log row: {name}")
            row = _json(raw)
            if not isinstance(row, dict):
                raise ValueError(f"invalid QKV run log row: {name}")
            yield row


def _json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate QKV run field: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"nonfinite QKV run JSON value: {value}")

    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid_constant)


def _number(value):
    if type(value) is not int or not 0 <= value <= (1 << 64) - 1:
        raise ValueError("invalid QKV run update counter")
    return value


def _pinned(data, expected):
    actual = "sha256:" + hashlib.sha256(data).hexdigest()
    if expected != actual:
        raise ValueError("QKV run file identity mismatch")


def _checkpoint(data, pin, result, update):
    _pinned(data, pin)
    if len(data) < 4:
        raise ValueError("truncated QKV trainer checkpoint")
    length = int.from_bytes(data[:4], "little")
    if length > 16384 or 4 + length > len(data):
        raise ValueError("invalid QKV trainer checkpoint header")
    header = _json(data[4:4 + length])
    if (header.get("schema") != "spiraltorch.native_causal_lm_qkv_trainer.v1"
            or header.get("data_id") != result["data_id"]
            or header.get("config") != result["config"]
            or _number(header.get("updates")) != update):
        raise ValueError("QKV trainer checkpoint disagrees with run")
    inner = data[4 + length:]
    _pinned(inner, header.get("optimizer_id"))
    if len(inner) < 4:
        raise ValueError("truncated QKV optimizer checkpoint")
    inner_length = int.from_bytes(inner[:4], "little")
    if inner_length > 16384 or 4 + inner_length > len(inner):
        raise ValueError("invalid QKV optimizer checkpoint header")
    optimizer = _json(inner[4:4 + inner_length])
    if (optimizer.get("schema") != "spiraltorch.qkv_adamw.v1"
            or _number(optimizer.get("steps")) != _number(header.get("initial_steps")) + update):
        raise ValueError("QKV optimizer clock disagrees with run")
    return inner, optimizer


def verify_qkv_run(directory):
    """Check a completed run's local pins, order and checkpoint/export agreement.

    This is an integrity check, not authentication or model-quality evidence.
    It does not load a model or trust an incomplete run without result.json.
    """
    root = Path(directory)
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError("QKV run root must be a directory, not a link")
    marker = _read_regular(root, "result.json", 1 << 20)
    if not marker.endswith(b"\n"):
        raise ValueError("truncated QKV run completion marker")
    result = _json(marker)
    if not isinstance(result, dict):
        raise ValueError("invalid QKV run result")
    start = _number(result.get("initial_update"))
    end = _number(result.get("updates"))
    total = _number(result.get("total_updates"))
    if (result.get("schema") != "spiraltorch.native_qkv_run.v1"
            or not isinstance(result.get("data_id"), str)
            or not isinstance(result.get("config"), dict)
            or not start <= end <= total
            or result.get("exhausted") is not (end == total)
            or result.get("checkpoint_index") != "checkpoints.jsonl"
            or result.get("step_log") != "steps.jsonl"):
        raise ValueError("invalid QKV run result")
    previous = None
    final_inner = None
    final_optimizer = None
    final_row = None
    for row in _records(root, "checkpoints.jsonl"):
        update = _number(row.get("updates"))
        if (update < start or update > end or (previous is None and update != start)
                or (previous is not None and update <= previous)
                or row.get("file") != f"checkpoint-{update:09d}.bin"):
            raise ValueError("invalid QKV checkpoint index order or name")
        data = _read_regular(root, row["file"], 1 << 26)
        final_inner, final_optimizer = _checkpoint(data, row.get("checkpoint_id"), result, update)
        previous = update
        final_row = row
    if previous != end or final_row != result.get("final_checkpoint"):
        raise ValueError("QKV final checkpoint disagrees with index")
    expected = start
    for row in _records(root, "steps.jsonl"):
        expected += 1
        if _number(row.get("update")) != expected or expected > end:
            raise ValueError("QKV step log has a gap or extra update")
    if expected != end:
        raise ValueError("QKV step log is incomplete")
    qkv = result.get("qkv")
    if not isinstance(qkv, dict) or qkv.get("file") != "qkv.bin":
        raise ValueError("invalid QKV export reference")
    exported = _read_regular(root, "qkv.bin", 1 << 26)
    _pinned(exported, qkv.get("checkpoint_id"))
    if (exported != final_inner
            or _number(qkv.get("optimizer_steps")) != final_optimizer["steps"]
            or _number(qkv.get("layer")) != _number(final_optimizer.get("layer"))):
        raise ValueError("QKV export differs from final trainer checkpoint")
    return result


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
