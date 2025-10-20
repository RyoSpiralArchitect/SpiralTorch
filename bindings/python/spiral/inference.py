"""Safety-aware inference helpers built on top of the SpiralTorch runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from spiraltorch import inference as _native

SafetyViolation = _native.SafetyViolation
SafetyVerdict = _native.SafetyVerdict
InferenceResult = _native.InferenceResult
AuditEvent = _native.AuditEvent
AuditLog = _native.AuditLog


@dataclass(frozen=True)
class SafetyEvent:
    """Snapshot of a safety audit event returned by the native runtime."""

    channel: str
    verdict: SafetyVerdict
    content_preview: str
    timestamp: str

    @classmethod
    def from_native(cls, event: AuditEvent) -> "SafetyEvent":
        return cls(
            channel=event.channel,
            verdict=event.verdict,
            content_preview=event.content_preview,
            timestamp=event.timestamp,
        )


class InferenceClient:
    """High-level wrapper that injects policy enforcement before returning results."""

    def __init__(self, *, refusal_threshold: float | None = None) -> None:
        self._runtime = _native.InferenceRuntime(refusal_threshold=refusal_threshold)

    def generate(
        self,
        prompt: str,
        *,
        candidate: str | None = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InferenceResult:
        payload: Dict[str, Any] = dict(metadata or {})
        if candidate is not None:
            payload.setdefault("candidate", candidate)
        return self._runtime.generate(prompt, metadata=payload or None)

    @property
    def audit_log(self) -> AuditLog:
        return self._runtime.audit_log

    def audit_events(self) -> list[SafetyEvent]:
        return [SafetyEvent.from_native(event) for event in self.audit_log.entries()]


__all__ = [
    "AuditEvent",
    "AuditLog",
    "InferenceClient",
    "InferenceResult",
    "SafetyEvent",
    "SafetyVerdict",
    "SafetyViolation",
]
