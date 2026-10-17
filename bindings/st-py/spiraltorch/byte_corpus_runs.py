"""Private, append-only segment storage for Rust-owned byte-LM studies.

This is a synchronous execution primitive, not a scheduler or promotion policy.
The durable store currently requires a local POSIX filesystem with hard links
and directory fsync. Parent directories must be trusted by the caller.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import stat
import time
from typing import TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from . import nn

__all__ = [
    "ByteCorpusSavedSegment",
    "load_byte_corpus_segment",
    "run_byte_corpus_segments",
]

_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_METADATA_BYTES = 64 * 1024
_CLAIM_SCHEMA = "spiraltorch.byte_corpus.segment_launch.v2"
_SEGMENT_SCHEMA = "spiraltorch.byte_corpus.saved_segment.v2"


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _hash(value) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("expected a lowercase SHA-256 hex digest")
    return value


def _study_type():
    from . import nn
    return nn.ByteCorpusStudy


@contextmanager
def _directory(path: Path, *, create=False):
    if os.name != "posix":
        raise NotImplementedError("byte-corpus run storage requires a POSIX filesystem")
    if create:
        # This is the exclusive launch claim. Never reclaim a failed directory.
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
        with _directory(path.parent) as parent:
            os.fsync(parent)
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        yield descriptor
    finally:
        os.close(descriptor)


def _read(directory: int, name: str, limit: int) -> bytes:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError(f"{name} must be a regular file of at most {limit} bytes")
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"{name} exceeds its byte limit")
    return raw


def _write_new(directory: int, name: str, raw: bytes) -> str:
    temporary = f".pending-{uuid.uuid4().hex}"
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                         0o600, dir_fd=directory)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic, no-replace publication. A seal is never visible half-written.
        os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory,
                follow_symlinks=False)
        os.fsync(directory)
    finally:
        os.unlink(temporary, dir_fd=directory)
        os.fsync(directory)
    return _digest(raw)


def _metadata(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate metadata key")
            result[key] = value
        return result

    def invalid_constant(_):
        raise ValueError("non-finite metadata value")

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("segment metadata must be a JSON object")
    return value


def _stops(stops, study, resume=None) -> list[int]:
    stops = list(islice(stops, 33))
    if (not stops or len(stops) > 32
            or any(type(stop) is not int for stop in stops)
            or stops != sorted(set(stops)) or stops[-1] != study.total_updates
            or (resume is not None and stops[0] <= resume.completed_updates)):
        raise ValueError("declare 1..32 increasing cursors ending at total_updates, after resume")
    for stop in stops:
        study.validate_segment(stop, resume=resume)
    return stops


@dataclass(frozen=True)
class ByteCorpusSavedSegment:
    """Verified bytes plus Rust-validated study/checkpoint; no device acquisition.

    Digests bind local bytes, not authenticity, quality, or past execution.
    The report remains opaque so model floats are never reserialized by Python.
    """

    study: nn.ByteCorpusStudy
    checkpoint: nn.ByteCorpusStudyCheckpoint
    report_json: str
    receipt_sha256: str
    claim_sha256: str
    checkpoint_sha256: str


def load_byte_corpus_segment(
    source: str | Path, stop_after: int, *, expected_receipt_sha256: str | None = None,
) -> ByteCorpusSavedSegment:
    """Load one explicitly selected v2 seal; never guess the latest checkpoint.

    Unsealed partial files, symlinks, oversized files, hash mismatches and Rust
    checkpoint violations fail closed. Other segments need not be loaded.
    """
    if type(stop_after) is not int or stop_after < 0:
        raise ValueError("stop_after must be a nonnegative integer")
    if expected_receipt_sha256 is not None:
        _hash(expected_receipt_sha256)
    with _directory(Path(source)) as directory:
        claim_raw = _read(directory, "claim.json", _MAX_METADATA_BYTES)
        claim = _metadata(claim_raw)
        if claim.get("schema") != _CLAIM_SCHEMA:
            raise ValueError("expected a v2 segment launch claim; legacy runs are not auto-upgraded")
        raw = _read(directory, "request.json", _MAX_JSON_BYTES)
        if claim.get("request_sha256") != _digest(raw):
            raise ValueError("request hash does not match launch claim")
        study = _study_type().from_json(raw)
        stops = _stops(claim.get("stops", []), study)
        if stop_after not in stops or claim.get("total_updates") != study.total_updates:
            raise ValueError("segment cursor does not match launch claim")
        receipt_raw = _read(directory, f"segment-{stop_after}.json", _MAX_METADATA_BYTES)
        receipt_hash = _digest(receipt_raw)
        if expected_receipt_sha256 is not None and receipt_hash != expected_receipt_sha256:
            raise ValueError("segment receipt does not match the expected hash")
        receipt = _metadata(receipt_raw)
        if (receipt.get("schema") != _SEGMENT_SCHEMA
                or receipt.get("claim_sha256") != _digest(claim_raw)
                or receipt.get("request_sha256") != study.request_sha256
                or type(receipt.get("stop_after")) is not int
                or receipt["stop_after"] != stop_after):
            raise ValueError("segment receipt does not match launch claim or cursor")
        report_raw = _read(directory, f"report-{stop_after}.json", _MAX_JSON_BYTES)
        checkpoint_raw = _read(directory, f"checkpoint-{stop_after}.json", _MAX_JSON_BYTES)
    if (_digest(report_raw) != receipt.get("report_sha256")
            or _digest(checkpoint_raw) != receipt.get("checkpoint_sha256")):
        raise ValueError("segment content hash mismatch")
    checkpoint = study.checkpoint_from_json(checkpoint_raw)
    if (checkpoint.completed_updates != stop_after
            or checkpoint.request_sha256 != study.request_sha256):
        raise ValueError("Rust checkpoint does not match segment cursor or request")
    return ByteCorpusSavedSegment(study, checkpoint, report_raw.decode("utf-8"),
                                 receipt_hash, _digest(claim_raw), _digest(checkpoint_raw))


def run_byte_corpus_segments(
    request: str | Path, output: str | Path, stops, *,
    resume_from: str | Path | None = None, resume_after: int | None = None,
    expected_resume_sha256: str | None = None,
) -> list[dict]:
    """Execute declared absolute cursors through Rust, sealing each full segment.

    Recovery requires both a source run and its explicit cursor, and always
    creates a new private output directory. There is no automatic retry, lock
    stealing, promotion, or change to the request/rates/geometry policy.
    """
    if (resume_from is None) != (resume_after is None):
        raise ValueError("resume_from and resume_after must be supplied together")
    if resume_from is None and expected_resume_sha256 is not None:
        raise ValueError("expected_resume_sha256 requires a resume source")
    request, output = Path(request), Path(output)
    with _directory(request.parent) as directory:
        raw = _read(directory, request.name, _MAX_JSON_BYTES)
    study = _study_type().from_json(raw)
    saved = None
    if resume_from is not None:
        saved = load_byte_corpus_segment(resume_from, resume_after,
                                        expected_receipt_sha256=expected_resume_sha256)
        if saved.study.request_sha256 != study.request_sha256:
            raise ValueError("resume requires the exact original request bytes")
    checkpoint = saved.checkpoint if saved is not None else None
    stops = _stops(stops, study, checkpoint)
    parent = None if saved is None else dict(
        claim_sha256=saved.claim_sha256, segment_sha256=saved.receipt_sha256,
        checkpoint_sha256=saved.checkpoint_sha256, completed_updates=resume_after)
    claim = dict(schema=_CLAIM_SCHEMA, run_id=uuid.uuid4().hex, pid=os.getpid(),
                 request_sha256=study.request_sha256, total_updates=study.total_updates,
                 stops=stops, parent=parent, before_gpu_launch=True,
                 runner_source_sha256=_digest(Path(__file__).read_bytes()))
    receipts = []
    with _directory(output, create=True) as directory:
        try:
            claim_hash = _write_new(directory, "claim.json", _json_bytes(claim))
            _write_new(directory, "request.json", raw)
            for stop in stops:
                print(json.dumps(dict(event="segment_started", stop_after=stop)), flush=True)
                started = time.monotonic()
                segment = study.advance_wgpu(stop, resume=checkpoint)
                candidate = segment.checkpoint
                if (candidate.completed_updates != stop
                        or candidate.request_sha256 != study.request_sha256):
                    raise ValueError("Rust segment returned different cursor or request")
                report = segment.report_json.encode("utf-8")
                encoded = candidate.to_json().encode("utf-8")
                if max(len(report), len(encoded)) > _MAX_JSON_BYTES:
                    raise ValueError("segment JSON exceeds the store's 64 MiB byte limit")
                report_hash = _write_new(directory, f"report-{stop}.json", report)
                checkpoint_hash = _write_new(directory, f"checkpoint-{stop}.json", encoded)
                receipt = dict(schema=_SEGMENT_SCHEMA, claim_sha256=claim_hash,
                               request_sha256=study.request_sha256, stop_after=stop,
                               seconds=time.monotonic() - started,
                               report_sha256=report_hash, checkpoint_sha256=checkpoint_hash)
                _write_new(directory, f"segment-{stop}.json", _json_bytes(receipt))
                receipts.append(receipt)
                checkpoint = candidate
                print(json.dumps(dict(event="segment_completed", **receipt)), flush=True)
            _write_new(directory, "completed.json", _json_bytes(dict(
                schema="spiraltorch.byte_corpus.segment_completion.v2",
                claim_sha256=claim_hash, request_sha256=study.request_sha256,
                completed_updates=checkpoint.completed_updates, segments=receipts)))
        except BaseException as error:
            try:
                _write_new(directory, "failed.json", _json_bytes(dict(
                    type=type(error).__name__, message=str(error), completed_segments=receipts)))
            except Exception:
                # Failure to record an I/O failure must not hide the original error.
                pass
            raise
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stop", type=int, action="append", required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--resume-after", type=int)
    parser.add_argument("--expected-resume-sha256")
    args = parser.parse_args()
    run_byte_corpus_segments(args.request, args.output, args.stop,
                            resume_from=args.resume_from, resume_after=args.resume_after,
                            expected_resume_sha256=args.expected_resume_sha256)


if __name__ == "__main__":
    main()
