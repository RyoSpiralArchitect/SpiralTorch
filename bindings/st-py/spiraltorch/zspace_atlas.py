from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .zspace_trace import load_zspace_trace_events

__all__ = [
    "zspace_trace_to_atlas_route",
    "zspace_trace_event_to_atlas_frame",
]


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _normalise_event(obj: Mapping[str, Any], *, event_type: str = "ZSpaceTrace") -> dict[str, Any] | None:
    if "kind" in obj:
        return dict(obj)

    if "payload" in obj:
        record_type = obj.get("event_type") or obj.get("type")
        if record_type not in (None, event_type):
            return None
        payload = obj.get("payload")
        if not isinstance(payload, Mapping):
            return None
        if len(payload) != 1:
            return None
        (kind, body), = payload.items()
        event: dict[str, Any] = {"kind": str(kind)}
        if isinstance(body, Mapping):
            event.update(body)
        else:
            event["data"] = body
        if "ts" in obj:
            event["ts"] = obj["ts"]
        return event

    if len(obj) == 1:
        (kind, body), = obj.items()
        event = {"kind": str(kind)}
        if isinstance(body, Mapping):
            event.update(body)
        else:
            event["data"] = body
        return event

    return None


def zspace_trace_event_to_atlas_frame(
    event: Mapping[str, Any],
    *,
    district: str = "Concourse",
    timestamp_base: float | None = None,
    step_seconds: float = 0.001,
) -> Any | None:
    """Convert one ZSpaceTrace event (normalised dict or plugin record) into an AtlasFrame.

    Returns `None` when the event cannot be converted.
    """

    normalised = _normalise_event(event)
    if normalised is None:
        return None

    ts = _as_float(normalised.get("ts"))
    if ts is None:
        base = time.time() if timestamp_base is None else float(timestamp_base)
        step = _as_float(normalised.get("step")) or 0.0
        ts = base + step * max(0.0, float(step_seconds))

    import spiraltorch as st

    fragment = st.telemetry.AtlasFragment(timestamp=ts)
    kind = str(normalised.get("kind") or "ZSpaceTrace")
    fragment.push_note(f"zspace.trace.kind={kind}")

    step_val = _as_float(normalised.get("step"))
    if step_val is not None:
        fragment.push_metric("zspace.trace.step", float(step_val), district)

    coherence = normalised.get("coherence")
    if isinstance(coherence, Sequence):
        coh_values = [_as_float(v) for v in coherence]
        coh = [v for v in coh_values if v is not None]
        fragment.push_metric("coherence_channels", float(len(coh)), district)
        fragment.push_metric("coherence_response_mean", _mean(coh), district)
        fragment.push_metric("coherence_response_peak", max(coh) if coh else 0.0, district)

    diagnostics = normalised.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        mean_coherence = _as_float(diagnostics.get("mean_coherence"))
        entropy = _as_float(diagnostics.get("entropy"))
        energy_ratio = _as_float(diagnostics.get("energy_ratio"))
        fractional_order = _as_float(diagnostics.get("fractional_order"))
        z_bias = _as_float(diagnostics.get("z_bias"))
        preserved = _as_float(diagnostics.get("preserved_channels"))
        discarded = _as_float(diagnostics.get("discarded_channels"))
        dominant = _as_float(diagnostics.get("dominant_channel"))
        label = diagnostics.get("label")

        if mean_coherence is not None:
            fragment.push_metric("coherence_mean", mean_coherence, district)
            fragment.push_metric("speed", math.tanh(mean_coherence), district)
        if entropy is not None:
            fragment.push_metric("coherence_entropy", entropy, district)
            fragment.push_metric("stability", math.tanh(1.0 - entropy), district)
        if energy_ratio is not None:
            fragment.push_metric("coherence_energy_ratio", energy_ratio, district)
            fragment.push_metric("drs", math.tanh(energy_ratio - 0.5), district)
        if fractional_order is not None:
            fragment.push_metric("coherence_fractional_order", fractional_order, district)
            fragment.push_metric("frac", math.tanh(fractional_order), district)
        if z_bias is not None:
            fragment.push_metric("coherence_z_bias", z_bias, district)
            fragment.push_metric("memory", math.tanh(z_bias), district)
        if preserved is not None:
            fragment.push_metric("coherence_preserved", preserved, district)
        if discarded is not None:
            fragment.push_metric("coherence_discarded", discarded, district)
        if dominant is not None:
            fragment.push_metric("coherence_dominant", dominant, district)
        if label is not None:
            fragment.push_note(f"zspace.trace.label={label}")

    return fragment.to_frame()


def zspace_trace_to_atlas_route(
    trace: str | Path | Iterable[Mapping[str, Any]],
    *,
    district: str = "Concourse",
    bound: int = 512,
    event_type: str = "ZSpaceTrace",
    timestamp_base: float | None = None,
    step_seconds: float = 0.001,
) -> Any:
    """Convert a ZSpaceTrace JSONL trace (or iterable of events) into a telemetry.AtlasRoute."""

    import spiraltorch as st

    if isinstance(trace, (str, Path)):
        events: list[dict[str, Any]] = load_zspace_trace_events(trace, event_type=event_type)
    else:
        events = []
        for item in trace:
            if isinstance(item, Mapping):
                normalised = _normalise_event(item, event_type=event_type)
                if normalised is not None:
                    events.append(normalised)

    route = st.telemetry.AtlasRoute()
    base = time.time() if timestamp_base is None else float(timestamp_base)
    for idx, event in enumerate(events):
        frame = zspace_trace_event_to_atlas_frame(
            event,
            district=district,
            timestamp_base=base + float(idx) * max(0.0, float(step_seconds)),
            step_seconds=step_seconds,
        )
        if frame is None:
            continue
        route.push_bounded(frame, bound=int(bound))
    return route

