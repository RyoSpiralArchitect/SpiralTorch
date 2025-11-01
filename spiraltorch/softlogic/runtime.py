"""Runtime helpers for applying SpiralK directives to a session."""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from .compiler import (
    Backend,
    Document,
    FeedbackBlock,
    Precision,
    RefractBlock,
    RefractOpPolicy,
    TargetSpec,
    compile_spiralk,
)

logger = logging.getLogger(__name__)


def apply_spiralk(session: Any, src: str, *, run_id: Optional[str] = None) -> Document:
    """Parse the given SpiralK document and attempt to apply it to ``session``.

    The function always returns the parsed :class:`Document`.  If the session
    exposes a kernel DSL backend (``kdsl_handle`` or ``kdsl`` attribute) the
    refract directives are forwarded to it.  Synchronisation and feedback blocks
    are currently logged so that integrators can hook them into their planner
    and telemetry stacks.
    """

    document = compile_spiralk(src)
    backend = _resolve_kdsl_backend(session)

    if backend is None:
        logger.warning("SoftLogic: no kernel DSL backend available; returning document only")
        return document

    lowering = _PythonRefractLowering(backend)
    for block in document.refracts:
        lowering.apply(block)

    if document.syncs:
        logger.info("SoftLogic: parsed %d sync block(s); planner integration pending", len(document.syncs))

    if document.feedbacks:
        emitter = _FeedbackEmitter(run_id or getattr(session, "run_id", "softlogic"))
        for block in document.feedbacks:
            emitter.emit(block)

    return document


def _resolve_kdsl_backend(session: Any) -> Optional[Any]:
    handle = getattr(session, "kdsl_handle", None)
    if callable(handle):
        try:
            backend = handle()
            if backend is not None:
                return backend
        except Exception:  # pragma: no cover - defensive
            logger.exception("SoftLogic: kdsl_handle() failed")
    backend = getattr(session, "kdsl", None)
    if backend is not None:
        return backend
    return None


class _PythonRefractLowering:
    def __init__(self, backend: Any):
        self._backend = backend

    def apply(self, block: RefractBlock) -> None:
        self._apply_target(block.target)
        if block.precision is not None:
            self._call_backend(("set_precision",), (block.precision,), fallback_args=(block.precision.value,))
        if block.layout is not None:
            self._call_backend(("set_layout",), (block.layout,), fallback_args=(block.layout.value,))
        if block.schedule is not None:
            self._call_backend(("set_schedule", "configure_schedule"), (block.schedule,))
        if block.backend is not None:
            self._call_backend(("select_backend", "set_backend"), (block.backend,), fallback_args=(block.backend.value,))
        for policy in block.policies:
            self._apply_policy(policy)

    def _apply_target(self, target: TargetSpec) -> None:
        if self._call_backend(("select_target", "bind_target"), (target,)):
            return
        self._call_backend(("select_target", "bind_target"), ((target.kind, target.name),))

    def _apply_policy(self, policy: RefractOpPolicy) -> None:
        if self._call_backend(("tune_op", "configure_op"), (policy.op, policy.flags)):
            return
        payload = {"op": policy.op, "flags": list(policy.flags)}
        self._call_backend(("tune_op", "configure_op"), (payload,))

    def _call_backend(self, names: Iterable[str], args: tuple[Any, ...], *, fallback_args: Optional[tuple[Any, ...]] = None) -> bool:
        if _maybe_call(self._backend, names, *args):
            return True
        if fallback_args is not None:
            return _maybe_call(self._backend, names, *fallback_args)
        return False


class _FeedbackEmitter:
    def __init__(self, run_id: str):
        self._run_id = run_id

    def emit(self, block: FeedbackBlock) -> None:  # pragma: no cover - logging only
        logger.info(
            "SoftLogic feedback: run_id=%s export=%s metrics=%s",
            self._run_id,
            block.export_path,
            block.metrics,
        )


def _maybe_call(obj: Any, names: Iterable[str], *args: Any) -> bool:
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                fn(*args)
            except TypeError:
                continue
            except Exception:  # pragma: no cover - surface backend issues
                logger.exception("SoftLogic: error while calling %%s", name)
                return True
            return True
    return False
