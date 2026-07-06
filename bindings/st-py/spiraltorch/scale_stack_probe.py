from __future__ import annotations

from collections.abc import Iterable, Sequence
from importlib import import_module
from typing import Any

__all__ = [
    "scale_stack_probe",
    "scalar_scale_stack_probe",
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
