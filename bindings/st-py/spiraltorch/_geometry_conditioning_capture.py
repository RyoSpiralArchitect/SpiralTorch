"""Scoped transport of native scalar reports, never geometric mathematics."""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy

_ACTIVE = ContextVar("spiraltorch_native_conditioning", default=None)


class NativeConditioningCapture:
    def __init__(self):
        self.reports = []
        self.omitted = 0

    def append(self, report):
        # A reentrant forward can consume the last slot while an outer sink
        # is still pending. Enforce the bound at the actual append as well.
        if len(self.reports) >= 16:
            self.omitted += 1
            return
        self.reports.append(deepcopy(report))


@contextmanager
def capture_native_conditioning():
    capture = NativeConditioningCapture()
    token = _ACTIVE.set(capture)
    try:
        yield capture
    finally:
        _ACTIVE.reset(token)


def native_conditioning_sink():
    capture = _ACTIVE.get()
    if capture is None:
        return None
    if len(capture.reports) >= 16:
        capture.omitted += 1
        return None
    return capture.append
