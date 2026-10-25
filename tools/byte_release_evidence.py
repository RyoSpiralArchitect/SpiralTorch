"""Opt-in retention of already available Python/native bytes, never GPU capture."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import re

ACTIVE = ContextVar("byte_release_evidence", default=None)
MAX_FILE = 64 * 1024 * 1024
MAX_TOTAL = 512 * 1024 * 1024
MAX_EVENTS = 512


def raw_save(path, value):
    if type(value) is str:
        value = value.encode("utf-8")
    if type(value) is not bytes or len(value) > MAX_FILE:
        raise ValueError("retention requires bounded exact bytes")
    with Path(path).open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    return {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")


class Recorder:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.events, self.total = [], 0

    def retain(self, label, factory):
        if not re.fullmatch("[a-z][a-z0-9_-]{0,63}", label) or len(self.events) >= MAX_EVENTS:
            raise ValueError("invalid or excessive retention event")
        directory = self.directory / f"{len(self.events):04d}-{label}"
        directory.mkdir()
        event = {"label": label, "directory": directory.name, "files": {}, "complete": False}
        self.events.append(event)
        payloads = factory()
        if type(payloads) is not dict or not payloads or len(payloads) > 256:
            raise ValueError("expected a bounded named payload set")
        for name, value in payloads.items():
            if type(name) is not str or not re.fullmatch(r"[a-z0-9_-]+\.(json|bin|txt)", name):
                raise ValueError("unsafe retention filename")
            if type(value) is str:
                value = value.encode("utf-8")
            if type(value) is not bytes or self.total + len(value) > MAX_TOTAL:
                raise ValueError("retained payload limit exceeded")
            event["files"][name] = raw_save(directory / name, value)
            self.total += len(value)
        event["complete"] = True

    def close(self):
        raw_save(self.directory / "index.json", encoded({
            "schema": "spiraltorch.byte_release.retention.v1", "events": self.events,
            "bytes": self.total, "release_ready": False,
        }))


@contextmanager
def recording(directory):
    if ACTIVE.get() is not None:
        raise ValueError("retention scopes must not overlap")
    recorder = Recorder(directory)
    token = ACTIVE.set(recorder)
    try:
        yield recorder
    finally:
        ACTIVE.reset(token)
        recorder.close()


def retain(label, factory):
    """The factory is not even evaluated outside an explicit retained test scope."""
    recorder = ACTIVE.get()
    if recorder is not None:
        recorder.retain(label, factory)
