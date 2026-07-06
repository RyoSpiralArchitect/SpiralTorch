from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "log_z_series_partial",
    "log_z_series_probe",
    "log_z_series_probe_to_zspace_partial",
]


def _optional_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _finite_float(value: Any, default: float = 0.0) -> float:
    numeric = _optional_float(value)
    return default if numeric is None else numeric


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _normalised(value: float) -> float:
    value = abs(float(value))
    return value / (1.0 + value)


def _complex_parts(value: Any) -> tuple[float, float]:
    try:
        sample = complex(value)
    except (TypeError, ValueError):
        return 0.0, 0.0
    return float(sample.real), float(sample.imag)


def _complex_abs(re: float, im: float) -> float:
    return math.hypot(re, im)


def _complex_phase(re: float, im: float) -> float:
    return math.atan2(im, re)


def _scalar_stats(values: Sequence[Any]) -> dict[str, Any]:
    finite = [_finite_float(value) for value in values]
    if not finite:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "energy": 0.0}
    count = len(finite)
    return {
        "count": count,
        "mean": sum(finite) / count,
        "min": min(finite),
        "max": max(finite),
        "energy": sum(value * value for value in finite) / count,
    }


def _projection_stats(values: Sequence[Any], preview_len: int) -> dict[str, Any]:
    parts = [_complex_parts(value) for value in values]
    abs_values = [_complex_abs(re, im) for re, im in parts]
    count = len(parts)
    mean_abs = sum(abs_values) / count if count else 0.0
    max_abs = max(abs_values) if abs_values else 0.0
    energy = sum(value * value for value in abs_values) / count if count else 0.0
    phase_drift = 0.0
    if count > 1:
        phase_drift = _complex_phase(*parts[-1]) - _complex_phase(*parts[0])
    preview_count = max(0, int(preview_len))
    preview = [
        {
            "index": idx,
            "re": re,
            "im": im,
            "abs": _complex_abs(re, im),
            "phase": _complex_phase(re, im),
        }
        for idx, (re, im) in enumerate(parts[:preview_count])
    ]
    return {
        "count": count,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "energy": energy,
        "phase_drift": phase_drift,
        "stability_score": 1.0 - _normalised(max_abs - mean_abs),
        "preview_count": min(preview_count, count),
        "preview": preview,
    }


def _series_attr(series: Any, name: str, default: Any = None) -> Any:
    value = getattr(series, name, default)
    return value() if callable(value) and name in {"len"} else value


def _probe_gradient(probe: Mapping[str, Any], dim: int) -> list[float]:
    sample_stats = probe.get("sample_stats")
    weight_stats = probe.get("weight_stats")
    projection = probe.get("projection")
    values: list[float] = []
    if isinstance(sample_stats, Mapping):
        values.extend(
            [
                _normalised(_finite_float(sample_stats.get("mean"))),
                _normalised(_finite_float(sample_stats.get("energy"))),
                _normalised(_finite_float(sample_stats.get("max"))),
            ]
        )
    if isinstance(weight_stats, Mapping):
        values.extend(
            [
                _normalised(_finite_float(weight_stats.get("mean"))),
                _normalised(_finite_float(weight_stats.get("energy"))),
            ]
        )
    if isinstance(projection, Mapping):
        values.extend(
            [
                _normalised(_finite_float(projection.get("mean_abs"))),
                _normalised(_finite_float(projection.get("energy"))),
                _finite_float(projection.get("stability_score")) - 0.5,
                _normalised(_finite_float(projection.get("phase_drift"))),
            ]
        )
        preview = projection.get("preview")
        if isinstance(preview, Sequence) and not isinstance(preview, (str, bytes, bytearray)):
            for row in preview:
                if isinstance(row, Mapping):
                    values.append(_normalised(_finite_float(row.get("abs"))))

    target = max(1, int(dim))
    if not values:
        return [0.0] * target
    if len(values) < target:
        repeats = (target + len(values) - 1) // len(values)
        values = (values * repeats)[:target]
    return [_clamp01(0.5 + value) * 2.0 - 1.0 for value in values[:target]]


def log_z_series_probe(
    series: Any,
    z_values: Sequence[complex],
    *,
    preview_len: int = 8,
) -> dict[str, Any]:
    """Summarise a ``LogZSeries`` projection as a portable WASM-compatible probe."""

    if not callable(getattr(series, "evaluate_many_z", None)):
        raise TypeError("series must expose evaluate_many_z(z_values)")
    samples = list(_series_attr(series, "samples", []))
    weights = list(_series_attr(series, "weights", []))
    projection = list(series.evaluate_many_z(list(z_values)))
    log_start = _finite_float(_series_attr(series, "log_start", 0.0))
    log_step = _finite_float(_series_attr(series, "log_step", 0.0))
    length = int(_finite_float(_series_attr(series, "len", len(samples)), len(samples)))
    support_end = log_start + log_step * max(0, length - 1)

    return {
        "kind": "spiraltorch.wasm_log_z_series_probe",
        "source_crate": "st-frac::cosmology",
        "mode": "log_z_series",
        "log_lattice": {
            "log_start": log_start,
            "log_step": log_step,
            "len": length,
            "support": [log_start, support_end],
        },
        "options": {
            "window": str(_series_attr(series, "window", "unknown")),
            "normalisation": str(_series_attr(series, "normalisation", "unknown")),
        },
        "sample_count": length,
        "sample_stats": _scalar_stats(samples),
        "weight_stats": _scalar_stats(weights),
        "z_count": len(z_values),
        "projection": _projection_stats(projection, preview_len),
    }


def log_z_series_probe_to_zspace_partial(
    probe: Mapping[str, Any],
    *,
    bundle_weight: float = 1.0,
    origin: str | None = None,
    telemetry_prefix: str = "log_z_series",
    gradient_dim: int = 8,
) -> Any:
    """Convert a Log-Z probe payload into a Z-space partial observation."""

    from .zspace_inference import ZSpacePartialBundle

    sample_stats = probe.get("sample_stats")
    weight_stats = probe.get("weight_stats")
    projection = probe.get("projection")
    lattice = probe.get("log_lattice")
    sample_energy = (
        _finite_float(sample_stats.get("energy")) if isinstance(sample_stats, Mapping) else 0.0
    )
    weight_energy = (
        _finite_float(weight_stats.get("energy")) if isinstance(weight_stats, Mapping) else 0.0
    )
    projection_energy = (
        _finite_float(projection.get("energy")) if isinstance(projection, Mapping) else 0.0
    )
    projection_mean = (
        _finite_float(projection.get("mean_abs")) if isinstance(projection, Mapping) else 0.0
    )
    projection_stability = (
        _clamp01(_finite_float(projection.get("stability_score"), 1.0))
        if isinstance(projection, Mapping)
        else 1.0
    )
    phase_drift = (
        _finite_float(projection.get("phase_drift")) if isinstance(projection, Mapping) else 0.0
    )
    sample_count = max(0.0, _finite_float(probe.get("sample_count")))
    z_count = max(0.0, _finite_float(probe.get("z_count")))
    roughness = _clamp01(
        0.45 * (1.0 - projection_stability)
        + 0.30 * _normalised(phase_drift)
        + 0.25 * _normalised(projection_energy - projection_mean)
    )

    partial = {
        "speed": _clamp01(0.55 * projection_stability + 0.45 * _normalised(z_count)),
        "memory": _clamp01(
            0.45 * _normalised(sample_energy)
            + 0.30 * _normalised(weight_energy)
            + 0.25 * _normalised(sample_count)
        ),
        "stability": _clamp01(0.70 * projection_stability + 0.30 * (1.0 - roughness)),
        "drs": roughness,
        "gradient": _probe_gradient(probe, gradient_dim),
    }

    prefix = telemetry_prefix or "log_z_series"
    telemetry: dict[str, Any] = {
        f"{prefix}.sample_count": sample_count,
        f"{prefix}.z_count": z_count,
        f"{prefix}.sample_energy": sample_energy,
        f"{prefix}.weight_energy": weight_energy,
        f"{prefix}.projection_energy": projection_energy,
        f"{prefix}.projection_mean_abs": projection_mean,
        f"{prefix}.projection_stability": projection_stability,
        f"{prefix}.phase_drift": phase_drift,
    }
    if isinstance(lattice, Mapping):
        for key in ("log_start", "log_step", "len"):
            telemetry[f"{prefix}.lattice.{key}"] = _finite_float(lattice.get(key))

    return ZSpacePartialBundle(
        partial,
        weight=max(0.0, float(bundle_weight)),
        origin=origin or "log_z_series:projection",
        telemetry=telemetry,
    )


def log_z_series_partial(
    samples: Sequence[float],
    log_start: float,
    log_step: float,
    z_values: Sequence[complex],
    *,
    window: str = "rectangular",
    normalisation: str = "l1",
    preview_len: int = 8,
    bundle_weight: float = 1.0,
    origin: str | None = None,
    telemetry_prefix: str = "log_z_series",
    gradient_dim: int = 8,
) -> Any:
    """Create a native Log-Z probe and return it as a Z-space partial bundle."""

    from . import frac as frac_module

    series_type = getattr(frac_module, "LogZSeries", None)
    if not callable(series_type):
        raise RuntimeError("log_z_series_partial requires native LogZSeries support")
    series = series_type(
        float(log_start),
        float(log_step),
        [float(sample) for sample in samples],
        window,
        normalisation,
    )
    probe = log_z_series_probe(series, z_values, preview_len=preview_len)
    return log_z_series_probe_to_zspace_partial(
        probe,
        bundle_weight=bundle_weight,
        origin=origin,
        telemetry_prefix=telemetry_prefix,
        gradient_dim=gradient_dim,
    )
