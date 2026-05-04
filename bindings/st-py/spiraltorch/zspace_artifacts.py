from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "load_zspace_artifact_manifest",
    "build_zspace_downstream_hook",
    "build_desire_adapter_from_downstream_hook",
    "desire_step_from_downstream_hook",
    "run_desire_geometry_bias_validation",
]


def load_zspace_artifact_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("artifact manifest must be a JSON object")
    payload.setdefault("artifact_manifest", str(manifest_path))
    return payload


def _coerce_manifest(manifest_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(manifest_or_path, Mapping):
        return dict(manifest_or_path)
    return load_zspace_artifact_manifest(manifest_or_path)


def _coerce_focus_items(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        return []
    return [dict(item) for item in payload if isinstance(item, Mapping)]


def _phase_hint(stage_focus: Sequence[Mapping[str, Any]]) -> str | None:
    ranked = []
    for item in stage_focus:
        latest = item.get("latest")
        try:
            latest_value = float(latest)
        except (TypeError, ValueError):
            latest_value = 0.0
        ranked.append((latest_value, str(item.get("stage") or "")))
    ranked.sort(reverse=True)
    for _, stage in ranked:
        if stage:
            return stage
    return None


def build_zspace_downstream_hook(
    manifest_or_path: Mapping[str, Any] | str | Path,
    *,
    top_k: int = 6,
) -> dict[str, Any]:
    manifest = _coerce_manifest(manifest_or_path)
    perspective = manifest.get("noncollapse_perspective")
    perspective_payload = dict(perspective) if isinstance(perspective, Mapping) else {}
    focus_items = _coerce_focus_items(perspective_payload.get("focus", []))
    stage_focus = []
    metric_focus = []
    for item in focus_items:
        name = str(item.get("name", ""))
        if name.startswith("noncollapse.stage."):
            enriched = dict(item)
            enriched.setdefault("stage", name.removeprefix("noncollapse.stage."))
            stage_focus.append(enriched)
        else:
            metric_focus.append(dict(item))
    if top_k > 0:
        metric_focus = metric_focus[: int(top_k)]

    summary = manifest.get("summary")
    summary_payload = dict(summary) if isinstance(summary, Mapping) else {}
    guidance = str(perspective_payload.get("guidance", ""))
    top_metric = metric_focus[0]["name"] if metric_focus else None
    phase_hint = _phase_hint(stage_focus)

    return {
        "kind": "spiraltorch.zspace_artifact_hook",
        "views": {
            "trace_jsonl": manifest.get("trace_jsonl"),
            "trace_html": manifest.get("trace_html"),
            "atlas_noncollapse_html": manifest.get("atlas_noncollapse_html"),
            "artifact_manifest": manifest.get("artifact_manifest"),
        },
        "summary": {
            "frames": int(summary_payload.get("frames", 0) or 0),
            "total_notes": int(summary_payload.get("total_notes", 0) or 0),
            "guidance": guidance,
        },
        "signals": {
            "coverage": int(perspective_payload.get("coverage", 0) or 0),
            "mean": float(perspective_payload.get("mean", 0.0) or 0.0),
            "latest": float(perspective_payload.get("latest", 0.0) or 0.0),
            "delta": float(perspective_payload.get("delta", 0.0) or 0.0),
            "momentum": float(perspective_payload.get("momentum", 0.0) or 0.0),
            "volatility": float(perspective_payload.get("volatility", 0.0) or 0.0),
            "stability": float(perspective_payload.get("stability", 0.0) or 0.0),
        },
        "stage_focus": stage_focus,
        "top_focus": metric_focus,
        "desire_candidate": {
            "experimental": True,
            "stability_signal": float(perspective_payload.get("stability", 0.0) or 0.0),
            "momentum_signal": float(perspective_payload.get("momentum", 0.0) or 0.0),
            "delta_signal": float(perspective_payload.get("delta", 0.0) or 0.0),
            "phase_hint": phase_hint,
            "focus_metric": top_metric,
            "guidance": guidance,
        },
    }


def _coerce_downstream_hook(hook_or_manifest: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    payload = _coerce_manifest(hook_or_manifest)
    if payload.get("kind") == "spiraltorch.zspace_artifact_hook":
        return payload
    downstream = payload.get("downstream_hook")
    if isinstance(downstream, Mapping):
        return dict(downstream)
    return build_zspace_downstream_hook(payload)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_desire_adapter_from_downstream_hook(
    hook_or_manifest: Mapping[str, Any] | str | Path,
    *,
    base_gain: float = 1.0,
    min_gain: float = 0.45,
    max_gain: float = 1.8,
    stability_weight: float = 0.45,
    momentum_weight: float = 0.2,
    delta_weight: float = 0.35,
    phase_bias: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    hook = _coerce_downstream_hook(hook_or_manifest)
    candidate = hook.get("desire_candidate")
    candidate_payload = dict(candidate) if isinstance(candidate, Mapping) else {}
    phase_bias_payload = {
        "observation": -0.18,
        "injection": 0.0,
        "integration": 0.16,
    }
    if isinstance(phase_bias, Mapping):
        for key, value in phase_bias.items():
            try:
                phase_bias_payload[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

    stability = _clamp(float(candidate_payload.get("stability_signal", 0.0) or 0.0), 0.0, 1.0)
    momentum_raw = float(candidate_payload.get("momentum_signal", 0.0) or 0.0)
    delta_raw = max(0.0, float(candidate_payload.get("delta_signal", 0.0) or 0.0))
    phase_hint = candidate_payload.get("phase_hint")
    phase_hint_str = str(phase_hint) if phase_hint is not None else None
    momentum = math.tanh(momentum_raw)
    delta = math.tanh(delta_raw)
    phase_term = phase_bias_payload.get(phase_hint_str or "", 0.0)

    gain_factor = (
        1.0
        + stability_weight * ((stability - 0.5) * 2.0)
        + momentum_weight * momentum
        - delta_weight * delta
        + phase_term
    )
    gain = _clamp(float(base_gain) * gain_factor, float(min_gain), float(max_gain))
    temperature_scale = _clamp(1.0 / gain if gain > 1e-6 else float(max_gain), 0.6, 1.8)

    top_focus = hook.get("top_focus")
    top_focus_items = _coerce_focus_items(top_focus)
    top_metric = top_focus_items[0]["name"] if top_focus_items else candidate_payload.get("focus_metric")

    return {
        "kind": "spiraltorch.desire_adapter",
        "gain": gain,
        "temperature_scale": temperature_scale,
        "phase_hint": phase_hint_str,
        "stability_signal": stability,
        "momentum_signal": momentum_raw,
        "delta_signal": delta_raw,
        "focus_metric": top_metric,
        "geometry_bias_signal": [
            stability,
            momentum,
            delta,
            gain / float(max_gain) if float(max_gain) > 1e-6 else gain,
        ],
        "guidance": candidate_payload.get("guidance") or hook.get("summary", {}).get("guidance"),
    }


def desire_step_from_downstream_hook(
    pipeline: Any,
    logits: Sequence[float],
    previous_token: int,
    hook_or_manifest: Mapping[str, Any] | str | Path,
    *,
    concept: Sequence[float] | None = None,
    window: Sequence[tuple[int, float]] | None = None,
    base_gain: float = 1.0,
    min_gain: float = 0.45,
    max_gain: float = 1.8,
    stability_weight: float = 0.45,
    momentum_weight: float = 0.2,
    delta_weight: float = 0.35,
    phase_bias: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    adapter = build_desire_adapter_from_downstream_hook(
        hook_or_manifest,
        base_gain=base_gain,
        min_gain=min_gain,
        max_gain=max_gain,
        stability_weight=stability_weight,
        momentum_weight=momentum_weight,
        delta_weight=delta_weight,
        phase_bias=phase_bias,
    )
    ingest_geometry_bias = getattr(pipeline, "ingest_geometry_bias", None)
    geometry_bias_ingested = False
    if callable(ingest_geometry_bias):
        ingest_geometry_bias(adapter["geometry_bias_signal"], source="zspace")
        geometry_bias_ingested = True
    scaled_logits = [float(value) * float(adapter["gain"]) for value in logits]
    result = pipeline.step(
        scaled_logits,
        int(previous_token),
        concept=None if concept is None else list(concept),
        window=None if window is None else list(window),
    )
    payload = dict(result) if isinstance(result, Mapping) else {"result": result}
    payload["downstream_adapter"] = adapter
    payload["scaled_logits_gain"] = adapter["gain"]
    payload["geometry_bias_ingested"] = geometry_bias_ingested
    geometry_bias_metrics = getattr(pipeline, "geometry_bias_metrics", None)
    if callable(geometry_bias_metrics) and "geometry_bias_metrics" not in payload:
        payload["geometry_bias_metrics"] = geometry_bias_metrics()
    geometry_bias_coherence = getattr(pipeline, "geometry_bias_coherence", None)
    if callable(geometry_bias_coherence) and "geometry_bias_coherence" not in payload:
        payload["geometry_bias_coherence"] = geometry_bias_coherence()
    return payload


def _coerce_modes(modes: Sequence[str] | None) -> list[str]:
    if modes is None:
        return ["inference", "training"]
    out = [str(mode) for mode in modes if str(mode)]
    return out or ["inference"]


def _is_finite_number(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def _non_negative_number(value: Any) -> bool:
    return _is_finite_number(value) and float(value) >= 0.0


def _validate_step_payload(payload: Mapping[str, Any]) -> dict[str, bool]:
    adapter = payload.get("downstream_adapter")
    adapter_payload = dict(adapter) if isinstance(adapter, Mapping) else {}
    signal = adapter_payload.get("geometry_bias_signal")
    signal_values = list(signal) if isinstance(signal, Sequence) and not isinstance(signal, (str, bytes)) else []

    checks = {
        "adapter_kind": adapter_payload.get("kind") == "spiraltorch.desire_adapter",
        "gain_finite": _is_finite_number(adapter_payload.get("gain")),
        "temperature_scale_finite": _is_finite_number(adapter_payload.get("temperature_scale")),
        "geometry_bias_signal_finite": bool(signal_values)
        and all(_is_finite_number(value) for value in signal_values),
        "geometry_bias_ingested": payload.get("geometry_bias_ingested") is True,
    }

    if "geometry_bias_metrics" in payload:
        metrics = payload.get("geometry_bias_metrics")
        metrics_payload = dict(metrics) if isinstance(metrics, Mapping) else {}
        checks["geometry_bias_metrics_available"] = bool(metrics_payload)
        checks["geometry_bias_metrics_non_negative"] = (
            _non_negative_number(metrics_payload.get("accuracy_mean"))
            and _non_negative_number(metrics_payload.get("fairness_mean"))
        )

    if "geometry_bias_coherence" in payload:
        coherence = payload.get("geometry_bias_coherence")
        coherence_payload = dict(coherence) if isinstance(coherence, Mapping) else {}
        checks["geometry_bias_coherence_available"] = bool(coherence_payload)
        checks["geometry_bias_coherence_non_negative"] = (
            _non_negative_number(coherence_payload.get("composite_energy"))
            and _non_negative_number(coherence_payload.get("z_energy"))
        )

    return checks


def run_desire_geometry_bias_validation(
    pipeline_factory: Any,
    logits: Sequence[float],
    previous_token: int,
    hook_or_manifest: Mapping[str, Any] | str | Path,
    *,
    concept: Sequence[float] | None = None,
    window: Sequence[tuple[int, float]] | None = None,
    modes: Sequence[str] = ("inference", "training"),
    base_gain: float = 1.0,
    min_gain: float = 0.45,
    max_gain: float = 1.8,
    stability_weight: float = 0.45,
    momentum_weight: float = 0.2,
    delta_weight: float = 0.35,
    phase_bias: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if not callable(pipeline_factory):
        raise TypeError("pipeline_factory must be callable")

    mode_list = _coerce_modes(modes)
    adapter = build_desire_adapter_from_downstream_hook(
        hook_or_manifest,
        base_gain=base_gain,
        min_gain=min_gain,
        max_gain=max_gain,
        stability_weight=stability_weight,
        momentum_weight=momentum_weight,
        delta_weight=delta_weight,
        phase_bias=phase_bias,
    )

    results: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    for mode in mode_list:
        pipeline = pipeline_factory()
        set_bias_context = getattr(pipeline, "set_bias_context", None)
        bias_context_applied = False
        if callable(set_bias_context):
            set_bias_context(mode)
            bias_context_applied = True

        try:
            payload = desire_step_from_downstream_hook(
                pipeline,
                logits,
                previous_token,
                hook_or_manifest,
                concept=concept,
                window=window,
                base_gain=base_gain,
                min_gain=min_gain,
                max_gain=max_gain,
                stability_weight=stability_weight,
                momentum_weight=momentum_weight,
                delta_weight=delta_weight,
                phase_bias=phase_bias,
            )
        except Exception as exc:  # noqa: BLE001 - return a structured validation report.
            failure = {
                "mode": mode,
                "error": f"{exc.__class__.__name__}: {exc}",
            }
            failures.append(failure)
            results[mode] = {
                "ok": False,
                "passed": False,
                "bias_context_applied": bias_context_applied,
                "error": failure["error"],
            }
            continue

        checks = _validate_step_payload(payload)
        ok = all(checks.values())
        if not ok:
            failures.append(
                {
                    "mode": mode,
                    "failed_checks": [
                        name for name, passed in sorted(checks.items()) if not passed
                    ],
                }
            )
        results[mode] = {
            "ok": ok,
            "passed": ok,
            "bias_context_applied": bias_context_applied,
            "checks": checks,
            "result": payload,
        }

    ok = not failures
    return {
        "kind": "spiraltorch.desire_geometry_bias_validation",
        "ok": ok,
        "passed": ok,
        "modes": mode_list,
        "adapter": adapter,
        "results": results,
        "failures": failures,
        "summary": {
            "total": len(mode_list),
            "passed": sum(1 for result in results.values() if result.get("ok") is True),
            "failed": len(failures),
        },
    }
