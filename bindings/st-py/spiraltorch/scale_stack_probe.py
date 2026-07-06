from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from importlib import import_module
from typing import Any

__all__ = [
    "scale_stack_probe",
    "scale_stack_probe_to_zspace_partial",
    "scalar_scale_stack_partial",
    "scalar_scale_stack_probe",
    "semantic_scale_stack_partial",
    "semantic_scale_stack_probe",
]


def _spiraltorch() -> Any:
    return import_module("spiraltorch")


def _value_or_call(value: Any, *args: Any) -> Any:
    return value(*args) if callable(value) else value


def _pairs(value: Any) -> list[tuple[float, float]]:
    rows = _value_or_call(value)
    return [(float(scale), float(gate)) for scale, gate in rows]


def _triples(value: Any) -> list[tuple[float, float, float]]:
    rows = _value_or_call(value)
    return [(float(low), float(high), float(mass)) for low, high, mass in rows]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric == numeric else None


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _finite_float(value: Any, default: float = 0.0) -> float:
    numeric = _optional_float(value)
    return default if numeric is None else numeric


def _mean(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if _optional_float(value) is not None]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _normalised(value: float) -> float:
    value = abs(float(value))
    return value / (1.0 + value)


def _probe_gradient(probe: Mapping[str, Any], dim: int) -> list[float]:
    samples = probe.get("samples")
    persistence = probe.get("persistence")
    coherence_profile = probe.get("coherence_profile")
    values: list[float] = []

    if isinstance(samples, Sequence):
        for sample in samples:
            if isinstance(sample, Mapping):
                values.append(_finite_float(sample.get("gate_mean")) - 0.5)
    if isinstance(persistence, Sequence):
        for bin_payload in persistence:
            if isinstance(bin_payload, Mapping):
                values.append(_finite_float(bin_payload.get("mass")))
    if isinstance(coherence_profile, Sequence):
        for row in coherence_profile:
            if isinstance(row, Mapping):
                scale = _optional_float(row.get("scale"))
                values.append(0.0 if scale is None else 1.0 / (1.0 + max(0.0, scale)))

    values.extend(
        [
            _finite_float(probe.get("interface_density")) - 0.5,
            _normalised(_finite_float(probe.get("moment_1"))),
            _normalised(_finite_float(probe.get("moment_2"))),
        ]
    )

    target = max(1, int(dim))
    if not values:
        return [0.0] * target
    if len(values) < target:
        repeats = (target + len(values) - 1) // len(values)
        values = (values * repeats)[:target]
    return [_clamp01(0.5 + value) * 2.0 - 1.0 for value in values[:target]]


def _break_scales(probe: Mapping[str, Any]) -> list[float]:
    profile = probe.get("coherence_profile")
    if not isinstance(profile, Sequence):
        return []
    scales = []
    for row in profile:
        if isinstance(row, Mapping):
            scale = _optional_float(row.get("scale"))
            if scale is not None:
                scales.append(scale)
    return scales


def scale_stack_probe(
    stack: Any,
    *,
    ambient_dim: float,
    dimension_window: int = 3,
    levels: Sequence[float] = (0.25, 0.5, 0.75),
) -> dict[str, Any]:
    """Return the Python-side equivalent of the WASM scale-stack probe payload."""

    samples = _pairs(getattr(stack, "samples"))
    persistence = _triples(getattr(stack, "persistence"))
    coherence_profile = []
    break_scale = getattr(stack, "coherence_break_scale")
    for level in levels:
        coherence_profile.append(
            {
                "level": float(level),
                "scale": _optional_float(_value_or_call(break_scale, float(level))),
            }
        )

    return {
        "kind": "spiraltorch.wasm_scale_stack_probe",
        "source_crate": "st-frac::scale_stack",
        "mode": str(_value_or_call(getattr(stack, "mode"))),
        "threshold": float(_value_or_call(getattr(stack, "threshold"))),
        "sample_count": len(samples),
        "samples": [
            {"scale": scale, "gate_mean": gate_mean}
            for scale, gate_mean in samples
        ],
        "persistence": [
            {"scale_low": low, "scale_high": high, "mass": mass}
            for low, high, mass in persistence
        ],
        "interface_density": _optional_float(
            _value_or_call(getattr(stack, "interface_density"))
        ),
        "moment_0": float(_value_or_call(getattr(stack, "moment"), 0)),
        "moment_1": float(_value_or_call(getattr(stack, "moment"), 1)),
        "moment_2": float(_value_or_call(getattr(stack, "moment"), 2)),
        "boundary_dimension": _optional_float(
            _value_or_call(
                getattr(stack, "boundary_dimension"),
                float(ambient_dim),
                int(dimension_window),
            )
        ),
        "coherence_profile": coherence_profile,
    }


def scale_stack_probe_to_zspace_partial(
    probe: Mapping[str, Any],
    *,
    bundle_weight: float = 1.0,
    origin: str | None = None,
    telemetry_prefix: str = "scale_stack",
    gradient_dim: int = 8,
) -> Any:
    """Convert a scale-stack probe payload into a Z-space partial observation."""

    from .zspace_inference import ZSpacePartialBundle

    mode = str(probe.get("mode", "unknown"))
    moment_0 = _finite_float(probe.get("moment_0"))
    moment_1 = _finite_float(probe.get("moment_1"))
    moment_2 = _finite_float(probe.get("moment_2"))
    density = _clamp01(_finite_float(probe.get("interface_density"), moment_0))
    boundary_dimension = _optional_float(probe.get("boundary_dimension"))
    sample_count = max(0.0, _finite_float(probe.get("sample_count")))
    break_scales = _break_scales(probe)
    mean_break = _mean(break_scales)

    coherence_pressure = 0.0 if mean_break is None else 1.0 / (1.0 + max(0.0, mean_break))
    dimension_pressure = 0.0 if boundary_dimension is None else _normalised(boundary_dimension)
    persistence_memory = _normalised(moment_1)
    roughness = _clamp01(0.55 * density + 0.30 * dimension_pressure + 0.15 * _normalised(moment_2))

    partial = {
        "speed": _clamp01(0.60 * coherence_pressure + 0.40 * density),
        "memory": _clamp01(
            0.55 * persistence_memory
            + 0.25 * dimension_pressure
            + 0.20 * _normalised(sample_count)
        ),
        "stability": _clamp01(1.0 - 0.65 * roughness),
        "drs": roughness,
        "gradient": _probe_gradient(probe, gradient_dim),
    }

    prefix = telemetry_prefix or "scale_stack"
    telemetry: dict[str, Any] = {
        f"{prefix}.sample_count": sample_count,
        f"{prefix}.interface_density": density,
        f"{prefix}.moment_0": moment_0,
        f"{prefix}.moment_1": moment_1,
        f"{prefix}.moment_2": moment_2,
        f"{prefix}.coherence_break_count": float(len(break_scales)),
        f"{prefix}.mode_scalar": 1.0 if mode == "scalar" else 0.0,
        f"{prefix}.mode_semantic": 1.0 if mode.startswith("semantic::") else 0.0,
    }
    if boundary_dimension is not None:
        telemetry[f"{prefix}.boundary_dimension"] = boundary_dimension
    if mean_break is not None:
        telemetry[f"{prefix}.coherence_break_mean"] = mean_break
        telemetry[f"{prefix}.coherence_break_min"] = min(break_scales)
        telemetry[f"{prefix}.coherence_break_max"] = max(break_scales)

    return ZSpacePartialBundle(
        partial,
        weight=max(0.0, float(bundle_weight)),
        origin=origin or f"scale_stack:{mode}",
        telemetry=telemetry,
    )


def scalar_scale_stack_probe(
    field: Iterable[float],
    shape: Sequence[int],
    scales: Sequence[float],
    threshold: float,
    *,
    ambient_dim: float | None = None,
    dimension_window: int = 3,
    levels: Sequence[float] = (0.25, 0.5, 0.75),
) -> dict[str, Any]:
    """Build a native scalar ScaleStack and return the shared WASM/Python probe shape."""

    st = _spiraltorch()
    stack = st.scalar_scale_stack(
        [float(value) for value in field],
        [int(value) for value in shape],
        [float(value) for value in scales],
        float(threshold),
    )
    return scale_stack_probe(
        stack,
        ambient_dim=float(len(shape) if ambient_dim is None else ambient_dim),
        dimension_window=dimension_window,
        levels=levels,
    )


def scalar_scale_stack_partial(
    field: Iterable[float],
    shape: Sequence[int],
    scales: Sequence[float],
    threshold: float,
    *,
    ambient_dim: float | None = None,
    dimension_window: int = 3,
    levels: Sequence[float] = (0.25, 0.5, 0.75),
    bundle_weight: float = 1.0,
    origin: str | None = None,
    telemetry_prefix: str = "scale_stack",
    gradient_dim: int = 8,
) -> Any:
    """Build a scalar scale-stack probe and return it as a Z-space partial bundle."""

    probe = scalar_scale_stack_probe(
        field,
        shape,
        scales,
        threshold,
        ambient_dim=ambient_dim,
        dimension_window=dimension_window,
        levels=levels,
    )
    return scale_stack_probe_to_zspace_partial(
        probe,
        bundle_weight=bundle_weight,
        origin=origin,
        telemetry_prefix=telemetry_prefix,
        gradient_dim=gradient_dim,
    )


def semantic_scale_stack_probe(
    embeddings: Sequence[Sequence[float]],
    scales: Sequence[float],
    threshold: float,
    *,
    metric: str = "euclidean",
    ambient_dim: float = 1.0,
    dimension_window: int = 3,
    levels: Sequence[float] = (0.25, 0.5, 0.75),
) -> dict[str, Any]:
    """Build a native semantic ScaleStack and return the shared WASM/Python probe shape."""

    st = _spiraltorch()
    stack = st.semantic_scale_stack(
        [[float(value) for value in row] for row in embeddings],
        [float(value) for value in scales],
        float(threshold),
        metric,
    )
    return scale_stack_probe(
        stack,
        ambient_dim=float(ambient_dim),
        dimension_window=dimension_window,
        levels=levels,
    )


def semantic_scale_stack_partial(
    embeddings: Sequence[Sequence[float]],
    scales: Sequence[float],
    threshold: float,
    *,
    metric: str = "euclidean",
    ambient_dim: float = 1.0,
    dimension_window: int = 3,
    levels: Sequence[float] = (0.25, 0.5, 0.75),
    bundle_weight: float = 1.0,
    origin: str | None = None,
    telemetry_prefix: str = "scale_stack",
    gradient_dim: int = 8,
) -> Any:
    """Build a semantic scale-stack probe and return it as a Z-space partial bundle."""

    probe = semantic_scale_stack_probe(
        embeddings,
        scales,
        threshold,
        metric=metric,
        ambient_dim=ambient_dim,
        dimension_window=dimension_window,
        levels=levels,
    )
    return scale_stack_probe_to_zspace_partial(
        probe,
        bundle_weight=bundle_weight,
        origin=origin,
        telemetry_prefix=telemetry_prefix,
        gradient_dim=gradient_dim,
    )
