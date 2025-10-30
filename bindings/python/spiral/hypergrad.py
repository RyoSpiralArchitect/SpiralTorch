"""High-level helpers for building and analysing SpiralTorch hypergrad tapes."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Mapping

import spiraltorch as st

HypergradTape = Any
HypergradSummary = Any

__all__ = [
    "HypergradTape",
    "HypergradSummary",
    "hypergrad_session",
    "hypergrad_summary_dict",
    "suggest_hypergrad_operator",
]


def _callable_attr(obj: Any, name: str) -> Callable[[], Any]:
    attr = getattr(obj, name, None)
    if attr is None:
        raise AttributeError(f"object {obj!r} is missing required attribute '{name}'")
    if not callable(attr):
        raise TypeError(f"attribute '{name}' on {obj!r} is not callable")
    return attr  # type: ignore[return-value]


@contextmanager
def hypergrad_session(
    *shape_args: Any,
    curvature: float = -1.0,
    learning_rate: float = 0.05,
    topos: Any | None = None,
    auto_reset: bool = True,
    apply: Callable[[HypergradTape], None] | None = None,
    **kwargs: Any,
) -> Iterator[HypergradTape]:
    """Construct a hypergrad tape and ensure it is reset once released.

    Parameters
    ----------
    shape_args:
        Positional shape arguments forwarded to :func:`spiraltorch.hypergrad`.
    curvature:
        Curvature parameter for the tape. Defaults to ``-1.0``.
    learning_rate:
        Base learning rate for the tape. Defaults to ``0.05``.
    topos:
        Optional open-cartesian guard forwarded to the native runtime.
    auto_reset:
        When ``True`` the tape's :py:meth:`reset` method is invoked on exit.
    apply:
        Optional callback invoked after the context manager yields but before
        any automatic reset occurs. This is useful when callers want to push
        the accumulated hypergrad into model weights without writing an
        explicit ``try``/``finally`` block.
    kwargs:
        Additional keyword arguments forwarded to :func:`spiraltorch.hypergrad`.
    """

    tape = st.hypergrad(
        *shape_args,
        curvature=curvature,
        learning_rate=learning_rate,
        topos=topos,
        **kwargs,
    )
    try:
        yield tape
        if apply is not None:
            apply(tape)
    finally:
        if auto_reset and hasattr(tape, "reset"):
            tape.reset()


def hypergrad_summary_dict(
    tape: HypergradTape,
    *,
    include_gradient: bool = False,
    extra: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    """Return a dictionary representation of a hypergrad tape's statistics."""

    summary = _callable_attr(tape, "summary")()
    shape = tuple(int(value) for value in _callable_attr(tape, "shape")())
    metrics: Dict[str, Any] = {
        "shape": shape,
        "curvature": float(_callable_attr(tape, "curvature")()),
        "learning_rate": float(_callable_attr(tape, "learning_rate")()),
        "summary": {
            "l1": float(_callable_attr(summary, "l1")()),
            "l2": float(_callable_attr(summary, "l2")()),
            "linf": float(_callable_attr(summary, "linf")()),
            "mean_abs": float(_callable_attr(summary, "mean_abs")()),
            "rms": float(_callable_attr(summary, "rms")()),
            "count": int(_callable_attr(summary, "count")()),
            "sum_squares": float(_callable_attr(summary, "sum_squares")()),
            "sum": float(_callable_attr(summary, "sum")()),
            "sum_cubes": float(_callable_attr(summary, "sum_cubes")()),
            "sum_quartic": float(_callable_attr(summary, "sum_quartic")()),
            "mean": float(_callable_attr(summary, "mean")()),
            "variance": float(_callable_attr(summary, "variance")()),
            "std": float(_callable_attr(summary, "std")()),
            "skewness": float(_callable_attr(summary, "skewness")()),
            "kurtosis": float(_callable_attr(summary, "kurtosis")()),
        },
    }

    telemetry_attr = getattr(tape, "telemetry", None)
    if callable(telemetry_attr):
        telemetry = telemetry_attr()
        metrics["telemetry"] = {
            "shape": tuple(int(value) for value in _callable_attr(telemetry, "shape")()),
            "volume": int(_callable_attr(telemetry, "volume")()),
            "curvature": float(_callable_attr(telemetry, "curvature")()),
            "learning_rate": float(_callable_attr(telemetry, "learning_rate")()),
            "saturation": float(_callable_attr(telemetry, "saturation")()),
            "porosity": float(_callable_attr(telemetry, "porosity")()),
            "tolerance": float(_callable_attr(telemetry, "tolerance")()),
            "max_depth": int(_callable_attr(telemetry, "max_depth")()),
            "max_volume": int(_callable_attr(telemetry, "max_volume")()),
        }

    if include_gradient:
        gradient = _callable_attr(tape, "gradient")()
        metrics["gradient"] = [float(value) for value in gradient]

    if extra:
        metrics.setdefault("summary", {}).update({k: float(v) for k, v in extra.items()})

    return metrics


def suggest_hypergrad_operator(
    tape: HypergradTape | Mapping[str, Any],
    *,
    clamp: bool = True,
    min_mix: float = 0.1,
    max_mix: float = 0.9,
    min_gain: float = 0.5,
    max_gain: float = 3.0,
) -> Dict[str, float]:
    """Derive WGSL operator hints from a hypergrad tape or summary mapping."""

    if isinstance(tape, Mapping):
        payload = tape
    else:
        payload = hypergrad_summary_dict(tape)

    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise TypeError("hypergrad summary payload must contain a mapping under 'summary'")

    rms = float(summary.get("rms", 0.0))
    mean_abs = float(summary.get("mean_abs", 0.0))
    std = float(summary.get("std", rms))
    skewness = float(summary.get("skewness", 0.0))
    kurtosis = float(summary.get("kurtosis", 3.0))
    l2 = float(summary.get("l2", 0.0))
    linf = float(summary.get("linf", 0.0))
    count = max(1, int(summary.get("count", 0)))

    ratio = mean_abs / (rms + 1e-6)
    spread = linf / (mean_abs + 1e-6)
    tail = max(0.0, kurtosis - 3.0)
    skew_factor = 1.0 + min(2.0, abs(skewness)) * 0.1
    mix = ratio / (1.0 + 0.25 * tail)
    gain = (std / (l2 + 1e-6)) * skew_factor

    if clamp:
        mix = min(max_mix, max(min_mix, mix))
        gain = min(max_gain, max(min_gain, gain))

    return {
        "mix": float(mix),
        "gain": float(gain),
        "ratio": float(ratio),
        "spread": float(spread),
        "std": float(std),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "count": float(count),
    }
