from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "fractal_field_partial",
    "fractal_field_probe",
    "fractal_field_probe_to_zspace_partial",
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


def _sample_abs(re: float, im: float) -> float:
    return math.hypot(re, im)


def _sample_phase(re: float, im: float) -> float:
    return math.atan2(im, re)


def _generator_config(generator: Any) -> dict[str, Any]:
    return {
        "octaves": int(_finite_float(getattr(generator, "octaves", 0), 0.0)),
        "lacunarity": _finite_float(getattr(generator, "lacunarity", 0.0)),
        "gain": _finite_float(getattr(generator, "gain", 0.0)),
        "iterations": int(_finite_float(getattr(generator, "iterations", 0), 0.0)),
    }


def _probe_gradient(probe: Mapping[str, Any], dim: int) -> list[float]:
    samples = probe.get("samples")
    values = [
        _normalised(_finite_float(probe.get("energy"))),
        _normalised(_finite_float(probe.get("mean_abs"))),
        _normalised(_finite_float(probe.get("max_abs"))),
        _normalised(_finite_float(probe.get("phase_drift"))),
        _normalised(_finite_float(probe.get("total_variation"))),
        _finite_float(probe.get("coherence_score")) - 0.5,
    ]
    if isinstance(samples, Sequence) and not isinstance(samples, (str, bytes, bytearray)):
        for sample in samples:
            if isinstance(sample, Mapping):
                values.extend(
                    [
                        _finite_float(sample.get("re")),
                        _finite_float(sample.get("im")),
                        _normalised(_finite_float(sample.get("abs"))),
                    ]
                )

    target = max(1, int(dim))
    if not values:
        return [0.0] * target
    if len(values) < target:
        repeats = (target + len(values) - 1) // len(values)
        values = (values * repeats)[:target]
    return [_clamp01(0.5 + value) * 2.0 - 1.0 for value in values[:target]]


def fractal_field_probe(
    generator: Any,
    log_start: float,
    log_step: float,
    length: int,
    *,
    preview_len: int = 8,
) -> dict[str, Any]:
    """Summarise a ``FractalFieldGenerator.branching_field`` as a portable probe."""

    if not callable(getattr(generator, "branching_field", None)):
        raise TypeError("generator must expose branching_field(log_start, log_step, length)")
    raw_field = list(generator.branching_field(log_start, log_step, int(length)))
    field = [_complex_parts(sample) for sample in raw_field]
    count = len(field)

    abs_values = [_sample_abs(re, im) for re, im in field]
    energy = sum(value * value for value in abs_values) / count if count else 0.0
    mean_abs = sum(abs_values) / count if count else 0.0
    max_abs = max(abs_values) if abs_values else 0.0
    mean_real = sum(re for re, _ in field) / count if count else 0.0
    mean_imag = sum(im for _, im in field) / count if count else 0.0
    total_variation = 0.0
    if count > 1:
        total_variation = sum(
            _sample_abs(field[idx][0] - field[idx - 1][0], field[idx][1] - field[idx - 1][1])
            for idx in range(1, count)
        ) / (count - 1)
    phase_drift = 0.0
    if count > 1:
        phase_drift = _sample_phase(*field[-1]) - _sample_phase(*field[0])

    preview_count = max(0, int(preview_len))
    samples = [
        {
            "index": idx,
            "log": float(log_start) + float(log_step) * idx,
            "re": re,
            "im": im,
            "abs": _sample_abs(re, im),
            "phase": _sample_phase(re, im),
        }
        for idx, (re, im) in enumerate(field[:preview_count])
    ]
    support_end = float(log_start) + float(log_step) * max(0, count - 1)

    return {
        "kind": "spiraltorch.wasm_fractal_field_probe",
        "source_crate": "st-frac::fractal_field",
        "mode": "branching_field",
        "generator": _generator_config(generator),
        "log_lattice": {
            "log_start": float(log_start),
            "log_step": float(log_step),
            "len": count,
            "support": [float(log_start), support_end],
        },
        "sample_count": count,
        "preview_count": min(preview_count, count),
        "energy": energy,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "mean_real": mean_real,
        "mean_imag": mean_imag,
        "phase_drift": phase_drift,
        "total_variation": total_variation,
        "coherence_score": 1.0 - _normalised(total_variation),
        "samples": samples,
    }


def fractal_field_probe_to_zspace_partial(
    probe: Mapping[str, Any],
    *,
    bundle_weight: float = 1.0,
    origin: str | None = None,
    telemetry_prefix: str = "fractal_field",
    gradient_dim: int = 8,
) -> Any:
    """Convert a fractal-field probe payload into a Z-space partial observation."""

    from .zspace_inference import ZSpacePartialBundle

    energy = _finite_float(probe.get("energy"))
    mean_abs = _finite_float(probe.get("mean_abs"))
    max_abs = _finite_float(probe.get("max_abs"))
    phase_drift = _finite_float(probe.get("phase_drift"))
    total_variation = _finite_float(probe.get("total_variation"))
    sample_count = max(0.0, _finite_float(probe.get("sample_count")))
    coherence = _clamp01(_finite_float(probe.get("coherence_score"), 1.0))
    roughness = _clamp01(
        0.55 * _normalised(total_variation)
        + 0.25 * _normalised(phase_drift)
        + 0.20 * (1.0 - coherence)
    )

    partial = {
        "speed": _clamp01(
            0.45 * coherence
            + 0.35 * _normalised(mean_abs)
            + 0.20 * _normalised(phase_drift)
        ),
        "memory": _clamp01(
            0.50 * _normalised(energy)
            + 0.25 * _normalised(max_abs)
            + 0.25 * _normalised(sample_count)
        ),
        "stability": _clamp01(0.75 * coherence + 0.25 * (1.0 - roughness)),
        "drs": roughness,
        "gradient": _probe_gradient(probe, gradient_dim),
    }

    generator = probe.get("generator")
    lattice = probe.get("log_lattice")
    prefix = telemetry_prefix or "fractal_field"
    telemetry: dict[str, Any] = {
        f"{prefix}.sample_count": sample_count,
        f"{prefix}.energy": energy,
        f"{prefix}.mean_abs": mean_abs,
        f"{prefix}.max_abs": max_abs,
        f"{prefix}.phase_drift": phase_drift,
        f"{prefix}.total_variation": total_variation,
        f"{prefix}.coherence_score": coherence,
    }
    if isinstance(generator, Mapping):
        for key in ("octaves", "lacunarity", "gain", "iterations"):
            telemetry[f"{prefix}.generator.{key}"] = _finite_float(generator.get(key))
    if isinstance(lattice, Mapping):
        for key in ("log_start", "log_step", "len"):
            telemetry[f"{prefix}.lattice.{key}"] = _finite_float(lattice.get(key))

    return ZSpacePartialBundle(
        partial,
        weight=max(0.0, float(bundle_weight)),
        origin=origin or "fractal_field:branching_field",
        telemetry=telemetry,
    )


def fractal_field_partial(
    octaves: int,
    log_start: float,
    log_step: float,
    length: int,
    *,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    iterations: int = 16,
    preview_len: int = 8,
    bundle_weight: float = 1.0,
    origin: str | None = None,
    telemetry_prefix: str = "fractal_field",
    gradient_dim: int = 8,
) -> Any:
    """Create a native fractal field probe and return it as a Z-space partial bundle."""

    from . import frac as frac_module

    generator_type = getattr(frac_module, "FractalFieldGenerator", None)
    if not callable(generator_type):
        raise RuntimeError("fractal_field_partial requires native FractalFieldGenerator support")
    generator = generator_type(int(octaves), float(lacunarity), float(gain), int(iterations))
    probe = fractal_field_probe(
        generator,
        log_start,
        log_step,
        length,
        preview_len=preview_len,
    )
    return fractal_field_probe_to_zspace_partial(
        probe,
        bundle_weight=bundle_weight,
        origin=origin,
        telemetry_prefix=telemetry_prefix,
        gradient_dim=gradient_dim,
    )
