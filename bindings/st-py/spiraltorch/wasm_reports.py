from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
from typing import Any

from .zspace_inference import ZSpacePartialBundle

__all__ = [
    "compare_wasm_reports",
    "load_wasm_report",
    "summarize_wasm_report",
    "wasm_report_to_zspace_partial",
]


def load_wasm_report(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a JSON report exported by a SpiralTorch WASM browser demo."""

    report_path = Path(path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("WASM report must be a JSON object")
    payload.setdefault("artifact_path", str(report_path))
    return payload


def _coerce_report(report: str | os.PathLike[str] | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(report, (str, os.PathLike)):
        return load_wasm_report(report)
    return dict(report)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _path(source: Mapping[str, Any], *keys: str) -> Any:
    value: Any = source
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _number_at(source: Mapping[str, Any], *keys: str) -> float | None:
    return _as_float(_path(source, *keys))


def _stats_summary(stats: Any) -> dict[str, Any] | None:
    if not isinstance(stats, Mapping):
        return None
    out: dict[str, Any] = {}
    for key in ("count", "finiteCount", "min", "max", "mean", "rms", "l1", "linf"):
        value = _as_float(stats.get(key))
        if value is not None:
            out[key] = value
    return out or None


def _loss_trace_stats(trace: Any) -> dict[str, Any] | None:
    values: list[float] = []
    for row in _sequence(trace):
        if not isinstance(row, Mapping):
            continue
        value = _as_float(row.get("loss"))
        if value is not None:
            values.append(value)
    if not values:
        return None
    first = values[0]
    last = values[-1]
    return {
        "count": len(values),
        "first": first,
        "last": last,
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "delta": last - first,
        "improved": last <= first,
    }


def _family(schema: str, kind: str) -> str:
    needle = f"{schema} {kind}".lower()
    if "mellin" in needle:
        return "mellin"
    if "canvas" in needle or "hypertrain" in needle:
        return "canvas"
    return "unknown"


def _runtime_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _mapping(report.get("runtime"))
    return {
        "wasm": bool(runtime.get("wasm")),
        "webgpu_available": bool(runtime.get("webgpuAvailable")),
        "webgpu_device_ready": bool(runtime.get("webgpuDeviceReady")),
        "webgpu_init_failed": bool(runtime.get("webgpuInitFailed")),
        "webgpu_trail_ready": bool(runtime.get("webgpuTrailReady")),
        "webgpu_trainer_ready": bool(runtime.get("webgpuTrainerReady")),
        "webgpu_fft_ready": bool(runtime.get("webgpuFftReady")),
    }


def _mellin_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    training = _mapping(report.get("training"))
    inference = _mapping(report.get("inferenceProbe"))
    target = _mapping(report.get("target"))
    learned = _mapping(report.get("learned"))
    grid = _mapping(report.get("grid")) or _mapping(target.get("grid"))
    plan = _mapping(report.get("plan"))
    trace_stats = _loss_trace_stats(training.get("trace"))
    abs_diff = _stats_summary(inference.get("absDiffStats"))

    return {
        "grid": {
            "len": _number_at(grid, "len"),
            "hilbert_norm": _number_at(grid, "hilbertNorm"),
            "sample_stats": _stats_summary(grid.get("sampleStats")),
        },
        "plan": {
            "len": _number_at(plan, "len"),
            "shape": list(_sequence(plan.get("shape"))),
        },
        "learning": {
            "steps": _number_at(training, "steps"),
            "lr": _number_at(training, "lr"),
            "duration_ms": _number_at(training, "durationMs"),
            "final_loss": _number_at(training, "finalLoss"),
            "trace": trace_stats,
            "target_magnitude": _stats_summary(target.get("magnitudeStats")),
            "learned_magnitude": _stats_summary(learned.get("magnitudeStats")),
            "abs_diff": abs_diff,
        },
        "inference_probe": {
            "mode": inference.get("mode"),
            "s_real": _as_float(inference.get("sReal")),
            "magnitude": _stats_summary(inference.get("magnitudeStats")),
            "abs_diff": abs_diff,
        },
    }


def _canvas_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _mapping(report.get("metrics"))
    current = _mapping(report.get("currentFrame"))
    gradients = _mapping(current.get("gradients"))
    desire = _mapping(current.get("desire"))
    control = _mapping(current.get("learningControl"))
    target = _mapping(report.get("target"))

    return {
        "frame": {
            "width": _number_at(current, "width"),
            "height": _number_at(current, "height"),
            "relation": _stats_summary(current.get("relationStats")),
            "field": _stats_summary(current.get("fieldStats")),
            "trail": _stats_summary(current.get("trailStats")),
        },
        "learning": {
            "step": _number_at(metrics, "step"),
            "history_length": _number_at(metrics, "historyLength"),
            "truncated": bool(metrics.get("truncated")),
            "last_loss": _number_at(_mapping(metrics.get("last")), "loss"),
            "loss": _stats_summary(metrics.get("lossStats")),
            "hyper_rms": _stats_summary(metrics.get("hyperRmsStats")),
            "real_rms": _stats_summary(metrics.get("realRmsStats")),
            "lr": _stats_summary(metrics.get("lrStats")),
        },
        "gradients": {
            "hypergrad_rms": _number_at(gradients, "hypergradRms"),
            "realgrad_rms": _number_at(gradients, "realgradRms"),
            "hypergrad_count": _number_at(gradients, "hypergradCount"),
            "realgrad_count": _number_at(gradients, "realgradCount"),
        },
        "desire": {
            "balance": _number_at(desire, "balance"),
            "stability": _number_at(desire, "stability"),
            "saturation": _number_at(desire, "saturation"),
            "events_mask": _number_at(desire, "eventsMask"),
        },
        "learning_control": {
            "hyper_lr_scale": _number_at(control, "hyperLearningRateScale"),
            "real_lr_scale": _number_at(control, "realLearningRateScale"),
            "operator_mix": _number_at(control, "operatorMix"),
            "operator_gain": _number_at(control, "operatorGain"),
        },
        "target": {
            "present": bool(target),
            "dims": target.get("dims") if isinstance(target.get("dims"), Mapping) else None,
            "stats": _stats_summary(target.get("stats")),
        },
    }


def summarize_wasm_report(
    report: str | os.PathLike[str] | Mapping[str, Any],
) -> dict[str, Any]:
    """Return a compact summary for a SpiralTorch WASM report."""

    payload = _coerce_report(report)
    schema = str(payload.get("schema", ""))
    kind = str(payload.get("kind", ""))
    family = _family(schema, kind)
    summary: dict[str, Any] = {
        "schema": schema,
        "kind": kind,
        "family": family,
        "created_at": payload.get("createdAt"),
        "artifact_path": payload.get("artifact_path"),
        "runtime": _runtime_summary(payload),
    }
    if family == "mellin":
        summary.update(_mellin_summary(payload))
    elif family == "canvas":
        summary.update(_canvas_summary(payload))
    return summary


def _summary_loss(summary: Mapping[str, Any]) -> float | None:
    learning = _mapping(summary.get("learning"))
    for key in ("final_loss", "last_loss"):
        value = _as_float(learning.get(key))
        if value is not None:
            return value
    loss_stats = _mapping(learning.get("loss"))
    return _as_float(loss_stats.get("mean"))


def _summary_stability(summary: Mapping[str, Any]) -> float | None:
    desire = _mapping(summary.get("desire"))
    value = _as_float(desire.get("stability"))
    if value is not None:
        return value
    loss = _summary_loss(summary)
    if loss is None:
        return None
    return 1.0 / (1.0 + max(0.0, loss))


def compare_wasm_reports(
    reports: (
        Mapping[str, str | os.PathLike[str] | Mapping[str, Any]]
        | Sequence[str | os.PathLike[str] | Mapping[str, Any]]
        | str
        | os.PathLike[str]
    ),
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare several WASM reports by final/last loss and stability."""

    items: list[tuple[str, str | os.PathLike[str] | Mapping[str, Any]]] = []
    if isinstance(reports, Mapping):
        items = [(str(label), report) for label, report in reports.items()]
    elif isinstance(reports, (str, os.PathLike)):
        items = [(str(Path(reports).stem), reports)]
    else:
        values = list(reports)
        if labels is not None and len(labels) != len(values):
            raise ValueError("labels length must match reports length")
        items = [
            (
                str(labels[index]) if labels is not None else f"run_{index}",
                value,
            )
            for index, value in enumerate(values)
        ]

    rows: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    for label, report in items:
        summary = summarize_wasm_report(report)
        family = str(summary.get("family", "unknown"))
        family_counts[family] = family_counts.get(family, 0) + 1
        rows.append(
            {
                "label": label,
                "schema": summary.get("schema"),
                "kind": summary.get("kind"),
                "family": family,
                "loss": _summary_loss(summary),
                "stability": _summary_stability(summary),
                "summary": summary,
            }
        )

    best_loss = min(
        (row for row in rows if row.get("loss") is not None),
        key=lambda row: float(row["loss"]),
        default=None,
    )
    best_stability = max(
        (row for row in rows if row.get("stability") is not None),
        key=lambda row: float(row["stability"]),
        default=None,
    )
    return {
        "kind": "spiraltorch.wasm_report_comparison",
        "count": len(rows),
        "families": family_counts,
        "best_loss": None if best_loss is None else dict(best_loss),
        "best_stability": None if best_stability is None else dict(best_stability),
        "reports": rows,
    }


def _work_units(summary: Mapping[str, Any]) -> float:
    family = summary.get("family")
    if family == "mellin":
        grid = _mapping(summary.get("grid"))
        plan = _mapping(summary.get("plan"))
        return max(
            1.0,
            _as_float(grid.get("len")) or 0.0,
            _as_float(plan.get("len")) or 0.0,
        )
    if family == "canvas":
        frame = _mapping(summary.get("frame"))
        relation = _mapping(frame.get("relation"))
        return max(1.0, _as_float(relation.get("count")) or 0.0)
    return 1.0


def _gradient_from_summary(summary: Mapping[str, Any], dim: int) -> list[float]:
    family = summary.get("family")
    learning = _mapping(summary.get("learning"))
    runtime = _mapping(summary.get("runtime"))
    values: list[float] = []
    if family == "mellin":
        trace = _mapping(learning.get("trace"))
        abs_diff = _mapping(learning.get("abs_diff"))
        values.extend(
            [
                _as_float(learning.get("final_loss")) or 0.0,
                _as_float(trace.get("delta")) or 0.0,
                _as_float(abs_diff.get("rms")) or 0.0,
                _as_float(abs_diff.get("linf")) or 0.0,
            ]
        )
    elif family == "canvas":
        gradients = _mapping(summary.get("gradients"))
        desire = _mapping(summary.get("desire"))
        control = _mapping(summary.get("learning_control"))
        values.extend(
            [
                _as_float(learning.get("last_loss")) or 0.0,
                _as_float(gradients.get("hypergrad_rms")) or 0.0,
                _as_float(gradients.get("realgrad_rms")) or 0.0,
                _as_float(desire.get("balance")) or 0.0,
                _as_float(desire.get("saturation")) or 0.0,
                _as_float(control.get("operator_mix")) or 0.0,
            ]
        )
    values.extend(
        [
            1.0 if runtime.get("webgpu_available") else 0.0,
            1.0 if runtime.get("webgpu_device_ready") else 0.0,
        ]
    )
    if len(values) < dim:
        values.extend(0.0 for _ in range(dim - len(values)))
    return values[:dim]


def wasm_report_to_zspace_partial(
    report: str | os.PathLike[str] | Mapping[str, Any],
    *,
    bundle_weight: float = 1.0,
    origin: str | None = None,
    telemetry_prefix: str = "wasm",
    gradient_dim: int = 8,
) -> ZSpacePartialBundle:
    """Convert a WASM report summary into a Z-space partial observation."""

    summary = summarize_wasm_report(report)
    family = str(summary.get("family", "unknown"))
    runtime = _mapping(summary.get("runtime"))
    loss = _summary_loss(summary)
    stability_hint = _summary_stability(summary)
    work_units = _work_units(summary)

    speed = 1.0 / (1.0 + math.log1p(work_units) / 12.0)
    memory = 1.0 / (1.0 + math.log1p(work_units) / 16.0)
    if runtime.get("webgpu_device_ready"):
        speed = _clamp01(speed + 0.08)
    elif runtime.get("webgpu_available"):
        speed = _clamp01(speed + 0.03)
    stability = _clamp01(stability_hint if stability_hint is not None else 0.5)
    drs = _clamp01(loss if loss is not None else 1.0 - stability)
    gradient = _gradient_from_summary(summary, max(1, int(gradient_dim)))

    telemetry: dict[str, Any] = {
        f"{telemetry_prefix}.family_mellin": 1.0 if family == "mellin" else 0.0,
        f"{telemetry_prefix}.family_canvas": 1.0 if family == "canvas" else 0.0,
        f"{telemetry_prefix}.webgpu_available": 1.0
        if runtime.get("webgpu_available")
        else 0.0,
        f"{telemetry_prefix}.webgpu_device_ready": 1.0
        if runtime.get("webgpu_device_ready")
        else 0.0,
        f"{telemetry_prefix}.work_units": work_units,
    }
    if loss is not None:
        telemetry[f"{telemetry_prefix}.loss"] = loss
    if stability_hint is not None:
        telemetry[f"{telemetry_prefix}.stability_hint"] = stability_hint

    return ZSpacePartialBundle(
        {
            "speed": speed,
            "memory": memory,
            "stability": stability,
            "drs": drs,
            "gradient": gradient,
        },
        weight=max(0.0, float(bundle_weight)),
        origin=origin or f"wasm:{family}",
        telemetry=telemetry,
    )
