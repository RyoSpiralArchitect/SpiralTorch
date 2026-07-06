from __future__ import annotations

from collections.abc import Mapping, Sequence
import glob
import json
import math
import os
from pathlib import Path
from typing import Any

from .zspace_inference import ZSpacePartialBundle

__all__ = [
    "audit_wasm_report",
    "audit_wasm_report_context",
    "build_wasm_report_context",
    "build_wasm_report_context_artifact",
    "compare_wasm_reports",
    "collect_wasm_report_paths",
    "load_wasm_report_context_artifact",
    "load_wasm_report",
    "summarize_wasm_report",
    "wasm_report_to_zspace_partial",
    "write_wasm_report_context_artifact",
]

_CONTEXT_ARTIFACT_SCHEMA = "spiraltorch.wasm_report_context.v1"


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


def _is_report_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and ("schema" in value or "kind" in value)


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _dedupe_paths(paths: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        key = str(Path(path).expanduser())
        if key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _report_label(index: int, report: Any) -> str:
    if isinstance(report, (str, os.PathLike)):
        return str(Path(report).stem)
    artifact_path = report.get("artifact_path") if isinstance(report, Mapping) else None
    if isinstance(artifact_path, (str, os.PathLike)):
        return str(Path(artifact_path).stem)
    family_hint = str(report.get("kind", "")) if isinstance(report, Mapping) else ""
    if family_hint:
        return family_hint
    return f"run_{index}"


def _iter_report_items(
    reports: Any,
    *,
    labels: Sequence[str] | None = None,
) -> list[tuple[str, str | os.PathLike[str] | Mapping[str, Any]]]:
    if reports is None:
        return []
    if isinstance(reports, (str, os.PathLike)):
        return [(_report_label(0, reports), reports)]
    if _is_report_mapping(reports):
        return [(_report_label(0, reports), reports)]
    if isinstance(reports, Mapping):
        return [(str(label), report) for label, report in reports.items()]
    values = list(reports)
    if labels is not None and len(labels) != len(values):
        raise ValueError("labels length must match reports length")
    return [
        (
            str(labels[index]) if labels is not None else _report_label(index, value),
            value,
        )
        for index, value in enumerate(values)
    ]


def _dedupe_report_items(
    items: Sequence[tuple[str, str | os.PathLike[str] | Mapping[str, Any]]],
) -> list[tuple[str, str | os.PathLike[str] | Mapping[str, Any]]]:
    seen_paths: set[str] = set()
    result: list[tuple[str, str | os.PathLike[str] | Mapping[str, Any]]] = []
    for label, report in items:
        if isinstance(report, (str, os.PathLike)):
            key = str(Path(report).expanduser())
            if key in seen_paths:
                continue
            seen_paths.add(key)
        result.append((label, report))
    return result


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
        "absolute_improvement": first - last,
        "relative_improvement": (first - last) / abs(first) if first != 0.0 else 0.0,
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
            "trace": _loss_trace_stats(metrics.get("tail")),
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


def _runtime_readiness(summary: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _mapping(summary.get("runtime"))
    family = str(summary.get("family", "unknown"))
    component_ready = [
        bool(runtime.get("webgpu_device_ready")),
        bool(runtime.get("webgpu_trainer_ready")),
        bool(runtime.get("webgpu_fft_ready")),
        bool(runtime.get("webgpu_trail_ready")),
    ]
    observed_components = sum(1 for ready in component_ready if ready)
    webgpu_available = bool(runtime.get("webgpu_available"))
    wasm_ready = bool(runtime.get("wasm"))

    score = 0.0
    if wasm_ready:
        score += 0.35
    if webgpu_available:
        score += 0.20
    if observed_components:
        score += 0.35
    elif family == "mellin" and webgpu_available:
        # The Mellin demo currently reports WebGPU availability but may not open
        # a device for the CPU/WASM training path.
        score += 0.15
    if not runtime.get("webgpu_init_failed"):
        score += 0.10

    if observed_components:
        status = "webgpu_ready"
    elif webgpu_available:
        status = "webgpu_available"
    elif wasm_ready:
        status = "wasm_only"
    else:
        status = "missing_runtime"

    return {
        "status": status,
        "score": _clamp01(score),
        "wasm": wasm_ready,
        "webgpu_available": webgpu_available,
        "webgpu_device_ready": bool(runtime.get("webgpu_device_ready")),
        "webgpu_component_ready_count": observed_components,
        "webgpu_init_failed": bool(runtime.get("webgpu_init_failed")),
    }


def _learning_progress(summary: Mapping[str, Any]) -> dict[str, Any]:
    learning = _mapping(summary.get("learning"))
    trace = _mapping(learning.get("trace"))
    loss_stats = _mapping(learning.get("loss"))
    loss = _summary_loss(summary)

    source = "missing"
    first_loss: float | None = None
    last_loss = loss
    absolute_improvement: float | None = None
    relative_improvement: float | None = None
    improved: bool | None = None

    if trace:
        source = "trace"
        first_loss = _as_float(trace.get("first"))
        trace_last = _as_float(trace.get("last"))
        if trace_last is not None:
            last_loss = trace_last
        absolute_improvement = _as_float(trace.get("absolute_improvement"))
        relative_improvement = _as_float(trace.get("relative_improvement"))
        improved_value = trace.get("improved")
        improved = bool(improved_value) if isinstance(improved_value, bool) else None
    elif loss_stats:
        # Browser reports sometimes keep aggregate loss stats but not the full
        # tail. Approximate progress from the observed max down to the selected
        # final/last loss.
        source = "loss_stats"
        first_loss = _as_float(loss_stats.get("max"))
        last_loss = loss
        if first_loss is not None and last_loss is not None:
            absolute_improvement = first_loss - last_loss
            relative_improvement = (
                absolute_improvement / abs(first_loss) if first_loss != 0.0 else 0.0
            )
            improved = absolute_improvement >= 0.0

    progress_score = 0.5
    if relative_improvement is not None:
        progress_score = _clamp01(relative_improvement)
    elif improved is not None:
        progress_score = 1.0 if improved else 0.0

    return {
        "source": source,
        "loss": loss,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "absolute_improvement": absolute_improvement,
        "relative_improvement": relative_improvement,
        "improved": improved,
        "progress_score": progress_score,
    }


def _audit_from_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _runtime_readiness(summary)
    progress = _learning_progress(summary)
    loss = _summary_loss(summary)
    stability = _summary_stability(summary)
    learning = _mapping(summary.get("learning"))

    loss_score = 0.5 if loss is None else 1.0 / (1.0 + max(0.0, loss))
    stability_score = _clamp01(stability if stability is not None else 0.5)
    readiness_score = _clamp01(
        0.35 * float(runtime["score"])
        + 0.25 * stability_score
        + 0.25 * loss_score
        + 0.15 * float(progress["progress_score"])
    )

    risk_flags: list[str] = []
    if not runtime["wasm"]:
        risk_flags.append("wasm_runtime_missing")
    if not runtime["webgpu_available"]:
        risk_flags.append("webgpu_unavailable")
    if runtime["webgpu_init_failed"]:
        risk_flags.append("webgpu_init_failed")
    if (
        runtime["webgpu_available"]
        and runtime["webgpu_component_ready_count"] == 0
        and summary.get("family") == "canvas"
    ):
        risk_flags.append("canvas_webgpu_components_not_ready")
    if loss is None:
        risk_flags.append("loss_not_observed")
    if progress["improved"] is False:
        risk_flags.append("loss_not_improved")
    if bool(learning.get("truncated")):
        risk_flags.append("metrics_history_truncated")

    if readiness_score >= 0.78 and "loss_not_improved" not in risk_flags:
        status = "ready"
    elif readiness_score >= 0.58:
        status = "usable"
    else:
        status = "needs_attention"

    recommendations: list[str] = []
    if not runtime["wasm"]:
        recommendations.append("rerun the browser demo after the WASM package loads")
    if runtime["webgpu_init_failed"]:
        recommendations.append("inspect the browser WebGPU initialization failure")
    elif not runtime["webgpu_available"]:
        recommendations.append("capture the report in a browser with WebGPU available")
    if progress["improved"] is False:
        recommendations.append("increase training steps or reduce the learning rate")
    if bool(learning.get("truncated")):
        recommendations.append("download a fresh report before the metrics tail truncates")
    if status == "ready":
        recommendations.append("promote this report as a Z-space runtime context candidate")
    elif status == "usable":
        recommendations.append("compare against nearby runs before promotion")

    return {
        "status": status,
        "readiness_score": readiness_score,
        "runtime": runtime,
        "learning": progress,
        "loss_score": loss_score,
        "stability_score": stability_score,
        "risk_flags": risk_flags,
        "recommendations": recommendations,
    }


def audit_wasm_report(
    report: str | os.PathLike[str] | Mapping[str, Any],
) -> dict[str, Any]:
    """Return readiness, learning-progress, and risk diagnostics for a WASM report."""

    summary = summarize_wasm_report(report)
    audit = _audit_from_summary(summary)
    return {
        "kind": "spiraltorch.wasm_report_audit",
        "schema": summary.get("schema"),
        "report_kind": summary.get("kind"),
        "family": summary.get("family"),
        "artifact_path": summary.get("artifact_path"),
        **audit,
    }


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

    items = _iter_report_items(reports, labels=labels)

    rows: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    for label, report in items:
        summary = summarize_wasm_report(report)
        audit = _audit_from_summary(summary)
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
                "readiness_score": audit["readiness_score"],
                "audit_status": audit["status"],
                "risk_flags": audit["risk_flags"],
                "audit": audit,
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
    best_readiness = max(
        rows,
        key=lambda row: (
            float(row.get("readiness_score") or 0.0),
            -(float(row["loss"]) if row.get("loss") is not None else 1.0e9),
            str(row.get("label")),
        ),
        default=None,
    )
    return {
        "kind": "spiraltorch.wasm_report_comparison",
        "count": len(rows),
        "families": family_counts,
        "best_loss": None if best_loss is None else dict(best_loss),
        "best_stability": None if best_stability is None else dict(best_stability),
        "best_readiness": None if best_readiness is None else dict(best_readiness),
        "reports": rows,
    }


def collect_wasm_report_paths(
    reports: (
        Sequence[str | os.PathLike[str]]
        | str
        | os.PathLike[str]
        | None
    ) = None,
    *,
    globs: Sequence[str] | None = None,
    dirs: Sequence[str | os.PathLike[str]] | None = None,
    recursive: bool = False,
) -> list[str]:
    """Collect browser-exported WASM report JSON paths from paths, globs, or dirs."""

    paths: list[str] = []
    if reports is None:
        pass
    elif isinstance(reports, (str, os.PathLike)):
        paths.append(str(reports))
    else:
        paths.extend(str(path) for path in reports if str(path))

    for pattern in globs or ():
        paths.extend(sorted(glob.glob(str(pattern), recursive=True)))

    for raw_dir in dirs or ():
        report_dir = Path(raw_dir)
        iterator = report_dir.rglob("*.json") if recursive else report_dir.glob("*.json")
        paths.extend(str(path) for path in sorted(iterator))

    return _dedupe_paths(paths)


def _select_wasm_report_indices(
    summaries: Sequence[Mapping[str, Any]],
    max_reports: int | None,
) -> list[int]:
    if max_reports is None or max_reports <= 0 or max_reports >= len(summaries):
        return list(range(len(summaries)))

    def _rank_key(index: int) -> tuple[bool, float, int]:
        loss = _summary_loss(summaries[index])
        return (loss is None, loss if loss is not None else 0.0, index)

    ranked = sorted(range(len(summaries)), key=_rank_key)
    return sorted(ranked[:max_reports])


def _compact_report_summary(
    label: str,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = _mapping(summary.get("runtime"))
    audit = _audit_from_summary(summary)
    learning = _mapping(audit.get("learning"))
    return {
        "label": label,
        "schema": summary.get("schema"),
        "kind": summary.get("kind"),
        "family": summary.get("family"),
        "artifact_path": summary.get("artifact_path"),
        "loss": _summary_loss(summary),
        "stability": _summary_stability(summary),
        "readiness_score": audit.get("readiness_score"),
        "audit_status": audit.get("status"),
        "learning_progress_score": learning.get("progress_score"),
        "learning_relative_improvement": learning.get("relative_improvement"),
        "risk_flags": list(_sequence(audit.get("risk_flags"))),
        "webgpu_available": runtime.get("webgpu_available"),
        "webgpu_device_ready": runtime.get("webgpu_device_ready"),
    }


def _compact_wasm_comparison(comparison: Mapping[str, Any]) -> dict[str, Any]:
    def _row(row: Any) -> dict[str, Any] | None:
        if not isinstance(row, Mapping):
            return None
        return {
            "label": row.get("label"),
            "family": row.get("family"),
            "loss": row.get("loss"),
            "stability": row.get("stability"),
            "readiness_score": row.get("readiness_score"),
            "audit_status": row.get("audit_status"),
            "risk_flags": row.get("risk_flags"),
        }

    return {
        "kind": comparison.get("kind"),
        "count": comparison.get("count"),
        "families": comparison.get("families"),
        "best_loss": _row(comparison.get("best_loss")),
        "best_stability": _row(comparison.get("best_stability")),
        "best_readiness": _row(comparison.get("best_readiness")),
    }


def audit_wasm_report_context(
    reports: (
        Mapping[str, str | os.PathLike[str] | Mapping[str, Any]]
        | Sequence[str | os.PathLike[str] | Mapping[str, Any]]
        | str
        | os.PathLike[str]
    ),
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Audit a set of WASM reports for promotion into runtime context."""

    items = _iter_report_items(reports, labels=labels)
    rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    risk_counts: dict[str, int] = {}
    for label, report in items:
        audit = audit_wasm_report(report)
        status = str(audit.get("status", "unknown"))
        family = str(audit.get("family", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1
        for flag in _sequence(audit.get("risk_flags")):
            flag_label = str(flag)
            risk_counts[flag_label] = risk_counts.get(flag_label, 0) + 1
        rows.append(
            {
                "label": label,
                "family": family,
                "status": status,
                "readiness_score": audit.get("readiness_score"),
                "loss": _mapping(audit.get("learning")).get("loss"),
                "learning_progress_score": _mapping(audit.get("learning")).get(
                    "progress_score"
                ),
                "risk_flags": audit.get("risk_flags"),
                "audit": audit,
            }
        )

    ranked = sorted(
        rows,
        key=lambda row: (
            float(row.get("readiness_score") or 0.0),
            -(float(row["loss"]) if row.get("loss") is not None else 1.0e9),
            str(row.get("label")),
        ),
        reverse=True,
    )
    best = ranked[0] if ranked else None
    recommendations: list[str] = []
    if best is not None:
        recommendations.append(
            f"prefer {best['label']} for the strongest audited WASM context"
        )
    if risk_counts:
        top_risks = ", ".join(
            key
            for key, _ in sorted(
                risk_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:3]
        )
        recommendations.append(f"review common WASM report risks: {top_risks}")
    if len(family_counts) > 1:
        recommendations.append(
            "compare Mellin and Canvas report families separately before mixing"
        )

    return {
        "kind": "spiraltorch.wasm_report_context_audit",
        "count": len(rows),
        "families": family_counts,
        "statuses": status_counts,
        "risk_counts": risk_counts,
        "best": best,
        "ranked": ranked,
        "recommendations": recommendations,
    }


def build_wasm_report_context(
    reports: (
        Mapping[str, str | os.PathLike[str] | Mapping[str, Any]]
        | Sequence[str | os.PathLike[str] | Mapping[str, Any]]
        | str
        | os.PathLike[str]
        | Mapping[str, Any]
        | None
    ) = None,
    *,
    report_globs: Sequence[str] | None = None,
    report_dirs: Sequence[str | os.PathLike[str]] | None = None,
    max_reports: int | None = None,
    recursive: bool = False,
    bundle_weight: float = 1.0,
    telemetry_prefix: str = "wasm",
    gradient_dim: int = 8,
) -> tuple[list[ZSpacePartialBundle], dict[str, Any]]:
    """Discover, compare, select, and convert WASM reports into Z-space context."""

    explicit_items = _iter_report_items(reports)
    discovered_paths = collect_wasm_report_paths(
        None,
        globs=report_globs,
        dirs=report_dirs,
        recursive=recursive,
    )
    discovered_items = [
        (_report_label(index, path), path)
        for index, path in enumerate(discovered_paths, start=len(explicit_items))
    ]
    candidate_items = _dedupe_report_items([*explicit_items, *discovered_items])
    metadata: dict[str, Any] = {
        "candidate_count": len(candidate_items),
        "report_count": 0,
        "gradient_dim": int(gradient_dim),
        "bundle_weight": float(bundle_weight),
        "telemetry_prefix": telemetry_prefix,
        "selection": {
            "max_reports": max_reports,
            "recursive": bool(recursive),
        },
        "candidate_reports": [],
        "reports": [],
        "context_origins": [],
        "comparison": None,
        "audit": None,
    }
    if not candidate_items:
        return [], metadata

    labels = [label for label, _ in candidate_items]
    report_values = [report for _, report in candidate_items]
    summaries = [summarize_wasm_report(report) for report in report_values]
    selected_indices = _select_wasm_report_indices(summaries, max_reports)
    selected_items = [candidate_items[index] for index in selected_indices]
    selected_summaries = [summaries[index] for index in selected_indices]
    comparison = compare_wasm_reports(report_values, labels=labels)
    context_audit = audit_wasm_report_context(report_values, labels=labels)
    context_partials = [
        wasm_report_to_zspace_partial(
            report,
            bundle_weight=bundle_weight,
            telemetry_prefix=telemetry_prefix,
            gradient_dim=gradient_dim,
        )
        for _, report in selected_items
    ]

    metadata["report_count"] = len(selected_items)
    metadata["candidate_reports"] = [
        _compact_report_summary(label, summary)
        for label, summary in zip(labels, summaries)
    ]
    metadata["reports"] = [
        _compact_report_summary(label, summary)
        for (label, _), summary in zip(selected_items, selected_summaries)
    ]
    metadata["context_origins"] = [partial.origin for partial in context_partials]
    metadata["comparison"] = _compact_wasm_comparison(comparison)
    metadata["audit"] = {
        "kind": context_audit.get("kind"),
        "count": context_audit.get("count"),
        "families": context_audit.get("families"),
        "statuses": context_audit.get("statuses"),
        "risk_counts": context_audit.get("risk_counts"),
        "best": None
        if not isinstance(context_audit.get("best"), Mapping)
        else {
            "label": _mapping(context_audit["best"]).get("label"),
            "family": _mapping(context_audit["best"]).get("family"),
            "status": _mapping(context_audit["best"]).get("status"),
            "readiness_score": _mapping(context_audit["best"]).get("readiness_score"),
            "loss": _mapping(context_audit["best"]).get("loss"),
        },
        "recommendations": context_audit.get("recommendations"),
    }
    return context_partials, metadata


def _partial_payload(partial: ZSpacePartialBundle) -> dict[str, Any]:
    telemetry = partial.telemetry_payload()
    return {
        "metrics": partial.resolved(),
        "weight": partial.weight,
        "origin": partial.origin,
        "telemetry": None if telemetry is None else dict(telemetry),
    }


def _partial_from_payload(payload: Any) -> ZSpacePartialBundle:
    if not isinstance(payload, Mapping):
        raise ValueError("WASM context partial rows must be JSON objects")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("WASM context partial rows must contain metrics")
    telemetry = payload.get("telemetry")
    if telemetry is not None and not isinstance(telemetry, Mapping):
        raise ValueError("WASM context partial telemetry must be an object")
    return ZSpacePartialBundle(
        dict(metrics),
        weight=float(payload.get("weight", 1.0)),
        origin=None if payload.get("origin") is None else str(payload.get("origin")),
        telemetry=None if telemetry is None else dict(telemetry),
    )


def build_wasm_report_context_artifact(
    reports: (
        Mapping[str, str | os.PathLike[str] | Mapping[str, Any]]
        | Sequence[str | os.PathLike[str] | Mapping[str, Any]]
        | str
        | os.PathLike[str]
        | Mapping[str, Any]
        | None
    ) = None,
    *,
    report_globs: Sequence[str] | None = None,
    report_dirs: Sequence[str | os.PathLike[str]] | None = None,
    max_reports: int | None = None,
    recursive: bool = False,
    bundle_weight: float = 1.0,
    telemetry_prefix: str = "wasm",
    gradient_dim: int = 8,
) -> dict[str, Any]:
    """Return a portable JSON-ready artifact for selected WASM context partials."""

    context_partials, metadata = build_wasm_report_context(
        reports,
        report_globs=report_globs,
        report_dirs=report_dirs,
        max_reports=max_reports,
        recursive=recursive,
        bundle_weight=bundle_weight,
        telemetry_prefix=telemetry_prefix,
        gradient_dim=gradient_dim,
    )
    return {
        "schema": _CONTEXT_ARTIFACT_SCHEMA,
        "kind": "spiraltorch.wasm_report_context",
        "metadata": metadata,
        "context_partials": [_partial_payload(partial) for partial in context_partials],
    }


def write_wasm_report_context_artifact(
    path: str | os.PathLike[str],
    reports: (
        Mapping[str, str | os.PathLike[str] | Mapping[str, Any]]
        | Sequence[str | os.PathLike[str] | Mapping[str, Any]]
        | str
        | os.PathLike[str]
        | Mapping[str, Any]
        | None
    ) = None,
    *,
    report_globs: Sequence[str] | None = None,
    report_dirs: Sequence[str | os.PathLike[str]] | None = None,
    max_reports: int | None = None,
    recursive: bool = False,
    bundle_weight: float = 1.0,
    telemetry_prefix: str = "wasm",
    gradient_dim: int = 8,
) -> str:
    """Write a selected WASM context handoff artifact and return its path."""

    artifact = build_wasm_report_context_artifact(
        reports,
        report_globs=report_globs,
        report_dirs=report_dirs,
        max_reports=max_reports,
        recursive=recursive,
        bundle_weight=bundle_weight,
        telemetry_prefix=telemetry_prefix,
        gradient_dim=gradient_dim,
    )
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(out_path)


def load_wasm_report_context_artifact(
    path: str | os.PathLike[str],
) -> tuple[list[ZSpacePartialBundle], dict[str, Any]]:
    """Load a WASM context handoff artifact into partials plus metadata."""

    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("WASM context artifact must be a JSON object")
    partial_rows = payload.get("context_partials")
    if not isinstance(partial_rows, Sequence) or isinstance(
        partial_rows, (str, bytes, bytearray)
    ):
        raise ValueError("WASM context artifact must contain context_partials")
    metadata = payload.get("metadata")
    if metadata is None:
        metadata_payload: dict[str, Any] = {}
    elif isinstance(metadata, Mapping):
        metadata_payload = dict(metadata)
    else:
        raise ValueError("WASM context artifact metadata must be an object")
    metadata_payload.setdefault("artifact_path", str(artifact_path))
    metadata_payload.setdefault("artifact_schema", payload.get("schema"))
    return [_partial_from_payload(row) for row in partial_rows], metadata_payload


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
