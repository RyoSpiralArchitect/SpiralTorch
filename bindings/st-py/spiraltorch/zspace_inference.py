"""Inference helpers that reconstruct Z-space metrics from partial observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any, Dict, Mapping, Sequence
from types import MappingProxyType

__all__ = [
    "ZSpaceDecoded",
    "ZSpaceInference",
    "ZSpacePosterior",
    "ZSpaceInferenceRuntime",
    "decode_zspace_embedding",
    "infer_from_partial",
    "compile_inference",
    "canvas_partial_from_snapshot",
    "infer_canvas_snapshot",
    "infer_canvas_transformer",
    "coherence_partial_from_diagnostics",
    "infer_coherence_diagnostics",
    "infer_coherence_from_sequencer",
]


_METRIC_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "speed": "speed",
        "velocity": "speed",
        "mem": "memory",
        "memory": "memory",
        "stab": "stability",
        "stability": "stability",
        "frac": "frac",
        "frac_reg": "frac",
        "fractality": "frac",
        "drs": "drs",
        "drift": "drs",
        "gradient": "gradient",
        "canvas_energy": "canvas_energy",
        "canvas_mean": "canvas_mean",
        "canvas_peak": "canvas_peak",
        "canvas_balance": "canvas_balance",
        "canvas_l1": "canvas_l1",
        "canvas_l2": "canvas_l2",
        "canvas_linf": "canvas_linf",
        "canvas_pixels": "canvas_pixels",
        "canvas_patch_energy": "canvas_patch_energy",
        "canvas_patch_mean": "canvas_patch_mean",
        "canvas_patch_peak": "canvas_patch_peak",
        "canvas_patch_pixels": "canvas_patch_pixels",
        "canvas_patch_balance": "canvas_patch_balance",
        "hypergrad_norm": "hypergrad_norm",
        "hypergrad_balance": "hypergrad_balance",
        "hypergrad_mean": "hypergrad_mean",
        "hypergrad_l1": "hypergrad_l1",
        "hypergrad_l2": "hypergrad_l2",
        "hypergrad_linf": "hypergrad_linf",
        "realgrad_norm": "realgrad_norm",
        "realgrad_balance": "realgrad_balance",
        "realgrad_mean": "realgrad_mean",
        "realgrad_l1": "realgrad_l1",
        "realgrad_l2": "realgrad_l2",
        "realgrad_linf": "realgrad_linf",
        "coherence_mean": "coherence_mean",
        "coherence_entropy": "coherence_entropy",
        "coherence_energy_ratio": "coherence_energy_ratio",
        "coherence_z_bias": "coherence_z_bias",
        "coherence_fractional_order": "coherence_fractional_order",
        "coherence_channels": "coherence_channels",
        "coherence_preserved": "coherence_preserved",
        "coherence_discarded": "coherence_discarded",
        "coherence_dominant": "coherence_dominant",
        "coherence_peak": "coherence_peak",
        "coherence_weight_entropy": "coherence_weight_entropy",
        "coherence_response_peak": "coherence_response_peak",
        "coherence_response_mean": "coherence_response_mean",
        "coherence_strength": "coherence_strength",
        "coherence_prosody": "coherence_prosody",
        "coherence_articulation": "coherence_articulation",
    }
)


def _softplus(value: float) -> float:
    if value > 20.0:
        return value
    if value < -20.0:
        return math.exp(value)
    return math.log1p(math.exp(value))


def _ensure_vector(z_state: Sequence[float]) -> list[float]:
    vector = [float(v) for v in z_state]
    if not vector:
        raise ValueError("z_state must contain at least one value")
    return vector


def _rfft(values: Sequence[float]) -> list[complex]:
    n = len(values)
    if n == 0:
        return []
    freq: list[complex] = []
    for k in range(n // 2 + 1):
        real = 0.0
        imag = 0.0
        for t, val in enumerate(values):
            angle = -2.0 * math.pi * k * t / max(1, n)
            real += val * math.cos(angle)
            imag += val * math.sin(angle)
        freq.append(complex(real, imag))
    return freq


def _fractional_energy(values: Sequence[float], alpha: float) -> float:
    spectrum = _rfft(values)
    n = len(spectrum)
    if n <= 1:
        return 0.0
    acc = 0.0
    for idx, coeff in enumerate(spectrum):
        omega = idx / max(1, n - 1)
        weight = omega ** (2.0 * alpha)
        acc += weight * abs(coeff) ** 2
    return acc / n


def _normalise_gradient(values: Sequence[float], length: int) -> list[float]:
    grad = [float(v) for v in values]
    if len(grad) < length:
        grad.extend(0.0 for _ in range(length - len(grad)))
    elif len(grad) > length:
        grad = grad[:length]
    scale = max(1.0, max(abs(v) for v in grad) if grad else 1.0)
    return [math.tanh(v / scale) for v in grad]


def _canonicalise_inputs(partial: Mapping[str, Any] | None) -> dict[str, Any]:
    if partial is None:
        return {}
    if not isinstance(partial, Mapping):
        raise TypeError("partial observations must be provided as a mapping")
    resolved: dict[str, Any] = {}
    for key, value in partial.items():
        canonical = _METRIC_ALIASES.get(key.lower())
        if canonical is None:
            raise KeyError(f"unknown metric '{key}'")
        resolved[canonical] = value
    return resolved


def _barycentric_from_metrics(metrics: Mapping[str, float]) -> tuple[float, float, float]:
    speed = float(metrics.get("speed", 0.0))
    memory = float(metrics.get("memory", 0.0))
    stability = float(metrics.get("stability", 0.0))
    weights = [_softplus(speed), _softplus(memory), _softplus(stability)]
    total = sum(weights)
    if total <= 0.0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return tuple(weight / total for weight in weights)


def _compute_gradient(values: Sequence[float]) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    grad: list[float] = []
    for idx in range(n):
        left = values[idx] - values[idx - 1] if idx > 0 else values[idx]
        right = values[idx + 1] - values[idx] if idx + 1 < n else -values[idx]
        grad.append(0.5 * (left + right))
    return grad


def _decode_metrics(z_state: Sequence[float], alpha: float) -> tuple[dict[str, float], list[float], tuple[float, float, float], float, float]:
    vector = _ensure_vector(z_state)
    n = len(vector)
    diffs = [vector[i + 1] - vector[i] for i in range(n - 1)]
    curvature = [vector[i + 1] - 2.0 * vector[i] + vector[i - 1] for i in range(1, n - 1)]
    mean_velocity = sum(abs(v) for v in diffs) / max(1, len(diffs))
    curvature_energy = sum(abs(v) for v in curvature) / max(1, len(curvature))
    l2 = math.sqrt(sum(value * value for value in vector))
    centre = sum(vector) / n
    frac_energy = _fractional_energy(vector, alpha)
    total_energy = sum(value * value for value in vector)
    gradient = _normalise_gradient(_compute_gradient(vector), n)
    speed = math.tanh(mean_velocity + 0.25 * l2 / max(1, n))
    memory = math.tanh(centre + 0.25 * total_energy / max(1, n))
    smoothness = 1.0 / (1.0 + curvature_energy)
    drift = 1.0 / (1.0 + mean_velocity)
    stability = math.tanh((smoothness + drift) * 1.5 - 1.0)
    spectrum = _rfft(vector)
    if len(spectrum) > 1:
        half = max(1, len(spectrum) // 2)
        high = sum(abs(coeff) ** 2 for coeff in spectrum[half:])
        low = sum(abs(coeff) ** 2 for coeff in spectrum[:half])
        drs = math.tanh((high - low) / (high + low + 1e-9))
    else:
        drs = 0.0
    frac = math.tanh(frac_energy / (total_energy + 1e-9))
    metrics = {
        "speed": speed,
        "memory": memory,
        "stability": stability,
        "frac": frac,
        "drs": drs,
    }
    barycentric = _barycentric_from_metrics(metrics)
    return metrics, gradient, barycentric, total_energy, frac_energy


@dataclass(frozen=True)
class ZSpaceDecoded:
    """Full set of metrics reconstructed from a latent Z vector."""

    z_state: tuple[float, ...]
    metrics: Mapping[str, float]
    gradient: tuple[float, ...]
    barycentric: tuple[float, float, float]
    energy: float
    frac_energy: float

    def as_dict(self) -> dict[str, Any]:
        data = {
            "z_state": list(self.z_state),
            "metrics": dict(self.metrics),
            "gradient": list(self.gradient),
            "barycentric": self.barycentric,
            "energy": self.energy,
            "frac_energy": self.frac_energy,
        }
        return data


@dataclass(frozen=True)
class ZSpaceInference:
    """Inference result after fusing partial observations with the decoded state."""

    metrics: Mapping[str, float]
    gradient: tuple[float, ...]
    barycentric: tuple[float, float, float]
    residual: float
    confidence: float
    prior: ZSpaceDecoded
    applied: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "gradient": list(self.gradient),
            "barycentric": self.barycentric,
            "residual": self.residual,
            "confidence": self.confidence,
            "applied": dict(self.applied),
            "prior": self.prior.as_dict(),
        }


class ZSpacePosterior:
    """Posterior over Z-space metrics conditioned on a latent state."""

    def __init__(self, z_state: Sequence[float], *, alpha: float = 0.35) -> None:
        self._z_state = tuple(_ensure_vector(z_state))
        self._alpha = max(1e-6, float(alpha))
        self._decoded: ZSpaceDecoded | None = None

    @property
    def z_state(self) -> list[float]:
        return list(self._z_state)

    @property
    def alpha(self) -> float:
        return self._alpha

    def decode(self) -> ZSpaceDecoded:
        if self._decoded is None:
            metrics, gradient, barycentric, energy, frac_energy = _decode_metrics(
                self._z_state, self._alpha
            )
            self._decoded = ZSpaceDecoded(
                z_state=self._z_state,
                metrics=MappingProxyType(dict(metrics)),
                gradient=tuple(gradient),
                barycentric=barycentric,
                energy=energy,
                frac_energy=frac_energy,
            )
        return self._decoded

    def project(
        self,
        partial: Mapping[str, Any] | None,
        *,
        smoothing: float = 0.35,
    ) -> ZSpaceInference:
        decoded = self.decode()
        metrics = dict(decoded.metrics)
        gradient = list(decoded.gradient)
        applied: Dict[str, Any] = {}
        updates = _canonicalise_inputs(partial)
        if "gradient" in updates:
            gradient = _normalise_gradient(updates["gradient"], len(self._z_state))
            applied["gradient"] = list(gradient)
        for key, value in updates.items():
            if key == "gradient":
                continue
            metrics[key] = float(value)
            applied[key] = metrics[key]
        override_bary = _barycentric_from_metrics(metrics)
        base_bary = decoded.barycentric
        blend = max(0.0, min(1.0, float(smoothing)))
        barycentric = tuple(
            blend * base + (1.0 - blend) * override
            for base, override in zip(base_bary, override_bary)
        )
        norm = sum(barycentric)
        if norm <= 0.0:
            barycentric = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        else:
            barycentric = tuple(value / norm for value in barycentric)
        diff = 0.0
        for name, base_value in decoded.metrics.items():
            diff += (metrics[name] - base_value) ** 2
        residual = math.sqrt(diff / len(decoded.metrics)) if decoded.metrics else 0.0
        confidence = math.exp(-residual)
        return ZSpaceInference(
            metrics=MappingProxyType(dict(metrics)),
            gradient=tuple(gradient),
            barycentric=barycentric,
            residual=residual,
            confidence=confidence,
            prior=decoded,
            applied=MappingProxyType(dict(applied)),
        )


class ZSpaceInferenceRuntime:
    """Stateful helper that incrementally fuses observations into a latent posterior."""

    def __init__(
        self,
        z_state: Sequence[float],
        *,
        alpha: float = 0.35,
        smoothing: float = 0.35,
        accumulate: bool = True,
    ) -> None:
        self._posterior = ZSpacePosterior(z_state, alpha=alpha)
        self._smoothing = float(smoothing)
        self._accumulate = bool(accumulate)
        self._cached: dict[str, Any] = {}

    @property
    def posterior(self) -> ZSpacePosterior:
        """Return the underlying posterior instance."""

        return self._posterior

    @property
    def smoothing(self) -> float:
        """Smoothing factor used when mixing barycentric coordinates."""

        return self._smoothing

    @property
    def accumulate(self) -> bool:
        """Whether successive updates reuse previously supplied observations."""

        return self._accumulate

    @property
    def cached_observations(self) -> Mapping[str, Any]:
        """Return the currently cached observation map."""

        return MappingProxyType(dict(self._cached))

    def clear(self) -> None:
        """Forget any cached observations."""

        self._cached.clear()

    def _merge(self, partial: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if partial is None:
            if not self._cached:
                return None
            return self._cached
        updates = _canonicalise_inputs(partial)
        if not self._accumulate:
            self._cached = {}
        if "gradient" in updates:
            gradient = updates.pop("gradient")
            if gradient is not None:
                self._cached["gradient"] = gradient
            else:
                self._cached.pop("gradient", None)
        for key, value in updates.items():
            self._cached[key] = value
        return self._cached

    def update(self, partial: Mapping[str, Any] | None = None) -> ZSpaceInference:
        """Fuse *partial* with any cached observations and produce an inference."""

        merged = self._merge(partial)
        return self._posterior.project(merged, smoothing=self._smoothing)

    def infer(self, partial: Mapping[str, Any] | None = None) -> ZSpaceInference:
        """Alias for :meth:`update` to mirror the functional helpers."""

        return self.update(partial)


def decode_zspace_embedding(z_state: Sequence[float], *, alpha: float = 0.35) -> ZSpaceDecoded:
    """Decode latent coordinates into a structured metric bundle."""

    return ZSpacePosterior(z_state, alpha=alpha).decode()


def infer_from_partial(
    z_state: Sequence[float],
    partial: Mapping[str, Any] | None,
    *,
    alpha: float = 0.35,
    smoothing: float = 0.35,
) -> ZSpaceInference:
    """Fuse partial metric observations with a latent state to complete Z-space inference."""

    posterior = ZSpacePosterior(z_state, alpha=alpha)
    return posterior.project(partial, smoothing=smoothing)


def compile_inference(
    fn=None,
    *,
    alpha: float = 0.35,
    smoothing: float = 0.35,
):
    """Wrap a callable so it automatically feeds its output into Z-space inference.

    The returned callable expects a latent ``z_state`` as its first argument and
    delegates any additional positional and keyword arguments to *fn*.  The
    original callable must return either ``None`` (indicating no new
    observations) or a mapping of partial observations compatible with
    :func:`infer_from_partial`.

    The helper can be used directly::

        def collect_metrics(data):
            return {"speed": data["speed"]}

        infer_speed = compile_inference(collect_metrics)
        result = infer_speed(z_state, sample)

    or as a decorator::

        @compile_inference(alpha=0.5)
        def analyze(sample):
            return {"memory": sample.mean()}

    """

    if fn is None:
        return lambda actual: compile_inference(
            actual, alpha=alpha, smoothing=smoothing
        )

    if not callable(fn):
        raise TypeError("compile_inference expects a callable or to be used as a decorator")

    def _compiled(z_state: Sequence[float], *args, **kwargs) -> ZSpaceInference:
        partial = fn(*args, **kwargs)
        if partial is not None and not isinstance(partial, Mapping):
            raise TypeError("compiled inference callable must return a mapping or None")
        return infer_from_partial(
            z_state,
            partial,
            alpha=alpha,
            smoothing=smoothing,
        )

    _compiled.__name__ = getattr(fn, "__name__", "compiled_inference")
    _compiled.__doc__ = fn.__doc__
    return _compiled


def _maybe_call(value: Any) -> Any:
    if callable(value):
        try:
            return value()
        except TypeError:
            return value
    return value


def _matrix_stats(matrix: Any) -> dict[str, float]:
    matrix = _maybe_call(matrix)
    if matrix is None or not isinstance(matrix, Iterable):
        return {"l1": 0.0, "l2": 0.0, "linf": 0.0, "mean": 0.0, "count": 0.0}
    flat: list[float] = []
    for row in matrix:
        row = _maybe_call(row)
        if row is None or not isinstance(row, Iterable):
            continue
        for value in row:
            try:
                flat.append(float(value))
            except (TypeError, ValueError):
                continue
    if not flat:
        return {"l1": 0.0, "l2": 0.0, "linf": 0.0, "mean": 0.0, "count": 0.0}
    l1 = sum(abs(value) for value in flat)
    l2 = math.sqrt(sum(value * value for value in flat))
    linf = max(abs(value) for value in flat)
    mean = sum(flat) / len(flat)
    return {"l1": l1, "l2": l2, "linf": linf, "mean": mean, "count": float(len(flat))}


def _merge_summary(stats: dict[str, float], summary: Mapping[str, Any] | None) -> dict[str, float]:
    if not isinstance(summary, Mapping):
        return stats
    merged = dict(stats)
    for key, value in summary.items():
        try:
            merged[key] = float(value)
        except (TypeError, ValueError):
            continue
    return merged


def _canvas_snapshot_stats(snapshot: Any) -> dict[str, dict[str, float]]:
    canvas = _maybe_call(getattr(snapshot, "canvas", None))
    hypergrad = _maybe_call(getattr(snapshot, "hypergrad", None))
    realgrad = _maybe_call(getattr(snapshot, "realgrad", None))
    summary = _maybe_call(getattr(snapshot, "summary", None))
    patch = _maybe_call(getattr(snapshot, "patch", None))
    canvas_stats = _matrix_stats(canvas)
    hyper_stats = _matrix_stats(hypergrad)
    real_stats = _matrix_stats(realgrad)
    if isinstance(summary, Mapping):
        hyper_stats = _merge_summary(hyper_stats, summary.get("hypergrad"))
        real_stats = _merge_summary(real_stats, summary.get("realgrad"))
    patch_stats = _matrix_stats(patch) if patch is not None else None
    stats: dict[str, dict[str, float]] = {
        "canvas": canvas_stats,
        "hypergrad": hyper_stats,
        "realgrad": real_stats,
    }
    if patch_stats is not None:
        stats["patch"] = patch_stats
    return stats


def canvas_partial_from_snapshot(
    snapshot: Any,
    *,
    hyper_gain: float = 2.5,
    memory_gain: float = 2.0,
    stability_gain: float = 2.5,
    patch_gain: float = 1.5,
) -> dict[str, float]:
    """Derive Z-space friendly metrics from a Canvas snapshot."""

    stats = _canvas_snapshot_stats(snapshot)
    canvas = stats.get("canvas", {})
    hyper = stats.get("hypergrad", {})
    real = stats.get("realgrad", {})
    patch = stats.get("patch")

    canvas_norm = float(canvas.get("l2", 0.0))
    hyper_norm = float(hyper.get("l2", 0.0))
    real_norm = float(real.get("l2", 0.0))
    patch_norm = float(patch.get("l2", 0.0)) if patch else 0.0
    total = canvas_norm + hyper_norm + real_norm + 1e-9
    canvas_ratio = canvas_norm / total
    hyper_ratio = hyper_norm / total
    real_ratio = real_norm / total
    patch_ratio = patch_norm / (patch_norm + canvas_norm + 1e-9)

    hyper_gain = max(0.0, float(hyper_gain))
    memory_gain = max(0.0, float(memory_gain))
    stability_gain = max(0.0, float(stability_gain))
    patch_gain = max(0.0, float(patch_gain))

    speed = math.tanh(hyper_gain * hyper_ratio + 0.5 * float(hyper.get("mean", 0.0)))
    memory = math.tanh(memory_gain * canvas_ratio + float(canvas.get("mean", 0.0)))
    stability = math.tanh(
        stability_gain * (1.0 - abs(hyper_ratio - real_ratio)) - 0.5 * stability_gain
    )
    frac_source = float(patch.get("linf", canvas.get("linf", 0.0))) if patch else float(
        canvas.get("linf", 0.0)
    )
    frac = math.tanh(patch_gain * frac_source)
    drs = math.tanh((hyper_ratio - real_ratio) * 2.5)

    partial: dict[str, float] = {
        "speed": speed,
        "memory": memory,
        "stability": stability,
        "frac": frac,
        "drs": drs,
        "canvas_energy": canvas_norm,
        "canvas_mean": float(canvas.get("mean", 0.0)),
        "canvas_peak": float(canvas.get("linf", 0.0)),
        "canvas_l1": float(canvas.get("l1", 0.0)),
        "canvas_l2": canvas_norm,
        "canvas_linf": float(canvas.get("linf", 0.0)),
        "canvas_balance": canvas_ratio,
        "canvas_pixels": float(canvas.get("count", 0.0)),
        "hypergrad_norm": hyper_norm,
        "hypergrad_mean": float(hyper.get("mean", 0.0)),
        "hypergrad_l1": float(hyper.get("l1", 0.0)),
        "hypergrad_l2": hyper_norm,
        "hypergrad_linf": float(hyper.get("linf", 0.0)),
        "hypergrad_balance": hyper_ratio,
        "realgrad_norm": real_norm,
        "realgrad_mean": float(real.get("mean", 0.0)),
        "realgrad_l1": float(real.get("l1", 0.0)),
        "realgrad_l2": real_norm,
        "realgrad_linf": float(real.get("linf", 0.0)),
        "realgrad_balance": real_ratio,
    }
    if patch is not None:
        partial.update(
            {
                "canvas_patch_energy": patch_norm,
                "canvas_patch_mean": float(patch.get("mean", 0.0)),
                "canvas_patch_peak": float(patch.get("linf", 0.0)),
                "canvas_patch_balance": patch_ratio,
                "canvas_patch_pixels": float(patch.get("count", 0.0)),
            }
        )
    return partial


def infer_canvas_snapshot(
    z_state: Sequence[float],
    snapshot: Any,
    *,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    hyper_gain: float = 2.5,
    memory_gain: float = 2.0,
    stability_gain: float = 2.5,
    patch_gain: float = 1.5,
) -> ZSpaceInference:
    """Project a Canvas snapshot into Z-space inference."""

    partial = canvas_partial_from_snapshot(
        snapshot,
        hyper_gain=hyper_gain,
        memory_gain=memory_gain,
        stability_gain=stability_gain,
        patch_gain=patch_gain,
    )
    return infer_from_partial(z_state, partial, alpha=alpha, smoothing=smoothing)


def infer_canvas_transformer(
    z_state: Sequence[float],
    canvas: Any,
    *,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    hyper_gain: float = 2.5,
    memory_gain: float = 2.0,
    stability_gain: float = 2.5,
    patch_gain: float = 1.5,
) -> ZSpaceInference:
    """Capture a CanvasTransformer snapshot and feed it into inference."""

    snapshot = _maybe_call(getattr(canvas, "snapshot", None))
    if snapshot is None:
        raise AttributeError("canvas object must expose a snapshot() method or property")
    return infer_canvas_snapshot(
        z_state,
        snapshot,
        alpha=alpha,
        smoothing=smoothing,
        hyper_gain=hyper_gain,
        memory_gain=memory_gain,
        stability_gain=stability_gain,
        patch_gain=patch_gain,
    )


def _sequence_floats(values: Any) -> list[float]:
    values = _maybe_call(values)
    if values is None:
        return []
    if isinstance(values, Mapping):
        values = values.values()
    if not isinstance(values, Iterable):
        return []
    out: list[float] = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def coherence_partial_from_diagnostics(
    diagnostics: Any,
    *,
    coherence: Any = None,
    contour: Any = None,
    speed_gain: float = 1.0,
    stability_gain: float = 1.0,
    frac_gain: float = 1.0,
    drs_gain: float = 1.0,
) -> dict[str, float]:
    """Convert coherence diagnostics into Z-space partial observations."""

    speed_gain = max(0.0, float(speed_gain))
    stability_gain = max(0.0, float(stability_gain))
    frac_gain = max(0.0, float(frac_gain))
    drs_gain = max(0.0, float(drs_gain))

    mean_coherence = float(_maybe_call(getattr(diagnostics, "mean_coherence", 0.0)) or 0.0)
    entropy = float(_maybe_call(getattr(diagnostics, "coherence_entropy", 0.0)) or 0.0)
    energy_ratio = float(_maybe_call(getattr(diagnostics, "energy_ratio", 0.0)) or 0.0)
    z_bias = float(_maybe_call(getattr(diagnostics, "z_bias", 0.0)) or 0.0)
    fractional_raw = _maybe_call(getattr(diagnostics, "fractional_order", 0.0))
    fractional_order = float(fractional_raw) if fractional_raw is not None else 0.0
    weights = _sequence_floats(getattr(diagnostics, "normalized_weights", []))
    preserved_raw = _maybe_call(getattr(diagnostics, "preserved_channels", None))
    preserved = float(preserved_raw) if preserved_raw is not None else float(len(weights))
    discarded_raw = _maybe_call(getattr(diagnostics, "discarded_channels", None))
    discarded = float(discarded_raw) if discarded_raw is not None else 0.0
    dominant = _maybe_call(getattr(diagnostics, "dominant_channel", None))

    response = _sequence_floats(coherence)

    partial: dict[str, float] = {
        "speed": math.tanh(speed_gain * mean_coherence),
        "memory": math.tanh(z_bias),
        "stability": math.tanh(stability_gain * (1.0 - entropy)),
        "frac": math.tanh(frac_gain * fractional_order),
        "drs": math.tanh(drs_gain * (energy_ratio - 0.5)),
        "coherence_mean": mean_coherence,
        "coherence_entropy": entropy,
        "coherence_energy_ratio": energy_ratio,
        "coherence_z_bias": z_bias,
        "coherence_fractional_order": fractional_order,
        "coherence_channels": float(len(weights)),
        "coherence_preserved": preserved,
        "coherence_discarded": discarded,
    }
    if dominant is not None:
        try:
            partial["coherence_dominant"] = float(dominant)
        except (TypeError, ValueError):
            partial["coherence_dominant"] = -1.0

    if weights:
        partial["coherence_peak"] = max(weights)
        weight_entropy = -sum(
            weight * math.log(max(weight, 1e-9)) for weight in weights if weight > 0.0
        )
        partial["coherence_weight_entropy"] = weight_entropy
    else:
        partial["coherence_peak"] = 0.0
        partial["coherence_weight_entropy"] = 0.0

    if response:
        partial["coherence_response_peak"] = max(response)
        partial["coherence_response_mean"] = sum(response) / len(response)
    else:
        partial["coherence_response_peak"] = 0.0
        partial["coherence_response_mean"] = 0.0

    if contour is not None:
        for key, attr in (
            ("coherence_strength", "coherence_strength"),
            ("coherence_prosody", "prosody_index"),
            ("coherence_articulation", "articulation_bias"),
        ):
            value = _maybe_call(getattr(contour, attr, None))
            if value is None:
                continue
            try:
                partial[key] = float(value)
            except (TypeError, ValueError):
                partial[key] = 0.0

    return partial


def infer_coherence_diagnostics(
    z_state: Sequence[float],
    diagnostics: Any,
    *,
    coherence: Any = None,
    contour: Any = None,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    speed_gain: float = 1.0,
    stability_gain: float = 1.0,
    frac_gain: float = 1.0,
    drs_gain: float = 1.0,
) -> ZSpaceInference:
    """Fuse coherence diagnostics with a latent state."""

    partial = coherence_partial_from_diagnostics(
        diagnostics,
        coherence=coherence,
        contour=contour,
        speed_gain=speed_gain,
        stability_gain=stability_gain,
        frac_gain=frac_gain,
        drs_gain=drs_gain,
    )
    return infer_from_partial(z_state, partial, alpha=alpha, smoothing=smoothing)


def infer_coherence_from_sequencer(
    z_state: Sequence[float],
    sequencer: Any,
    tensor: Any,
    *,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    method: str = "forward_with_diagnostics",
    include_contour: bool = False,
    return_outputs: bool = False,
    speed_gain: float = 1.0,
    stability_gain: float = 1.0,
    frac_gain: float = 1.0,
    drs_gain: float = 1.0,
):
    """Run a sequencer forward pass and project its diagnostics into Z-space."""

    forward = getattr(sequencer, method, None)
    if forward is None:
        raise AttributeError(f"sequencer has no method '{method}'")
    outputs = forward(tensor)
    if not isinstance(outputs, tuple) or len(outputs) < 3:
        raise ValueError(
            "sequencer forward method must return (tensor, coherence, diagnostics)"
        )
    _, coherence, diagnostics = outputs[:3]
    contour = None
    if include_contour:
        contour_getter = getattr(sequencer, "emit_linguistic_contour", None)
        if callable(contour_getter):
            contour = contour_getter(tensor)
    inference = infer_coherence_diagnostics(
        z_state,
        diagnostics,
        coherence=coherence,
        contour=contour,
        alpha=alpha,
        smoothing=smoothing,
        speed_gain=speed_gain,
        stability_gain=stability_gain,
        frac_gain=frac_gain,
        drs_gain=drs_gain,
    )
    if return_outputs:
        return inference, outputs
    return inference


