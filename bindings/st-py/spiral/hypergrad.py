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
            "min": float(_callable_attr(summary, "min")()),
            "max": float(_callable_attr(summary, "max")()),
            "support_width": float(_callable_attr(summary, "support_width")()),
            "positive_count": int(_callable_attr(summary, "positive_count")()),
            "negative_count": int(_callable_attr(summary, "negative_count")()),
            "zero_count": int(_callable_attr(summary, "zero_count")()),
            "near_zero_count": int(_callable_attr(summary, "near_zero_count")()),
            "positive_fraction": float(_callable_attr(summary, "positive_fraction")()),
            "negative_fraction": float(_callable_attr(summary, "negative_fraction")()),
            "zero_fraction": float(_callable_attr(summary, "zero_fraction")()),
            "near_zero_fraction": float(_callable_attr(summary, "near_zero_fraction")()),
            "activation": float(_callable_attr(summary, "activation")()),
            "sign_lean": float(_callable_attr(summary, "sign_lean")()),
            "sign_entropy": float(_callable_attr(summary, "sign_entropy")()),
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
            "finite_count": int(_callable_attr(telemetry, "finite_count")()),
            "non_finite_count": int(_callable_attr(telemetry, "non_finite_count")()),
            "non_finite_ratio": float(_callable_attr(telemetry, "non_finite_ratio")()),
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
    activation = float(summary.get("activation", 0.0))
    sign_entropy = float(summary.get("sign_entropy", 0.0))
    lean = float(summary.get("sign_lean", 0.0))
    near_zero_fraction = float(summary.get("near_zero_fraction", 0.0))
    support_width = float(summary.get("support_width", 0.0))

    ratio = mean_abs / (rms + 1e-6)
    spread = linf / (mean_abs + 1e-6)
    tail = max(0.0, kurtosis - 3.0)
    skew_factor = 1.0 + min(2.0, abs(skewness)) * 0.1
    mix = ratio / (1.0 + 0.25 * tail)
    gain = (std / (l2 + 1e-6)) * skew_factor

    activation = max(0.0, min(1.0, activation))
    sign_entropy = max(0.0, min(1.0, sign_entropy))
    dormancy = 1.0 - activation
    mix *= (0.85 + 0.15 * activation) * (0.9 + 0.1 * sign_entropy)
    gain *= (0.9 + 0.2 * activation) * (0.95 + 0.1 * (sign_entropy - 0.5))
    spread = max(spread, abs(support_width) / (abs(mean_abs) + 1e-6))

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
        "activation": float(activation),
        "dormancy": float(dormancy),
        "sign_entropy": float(sign_entropy),
        "sign_lean": float(lean),
        "near_zero_fraction": float(near_zero_fraction),
        "support_width": float(support_width),
    }

