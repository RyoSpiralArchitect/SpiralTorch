"""Inference helpers that reconstruct Z-space metrics from partial observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Iterable
from importlib import import_module
import sys
from typing import Any, Dict, Mapping, MutableMapping, Sequence
from types import MappingProxyType

__all__ = [
    "ZSpaceDecoded",
    "ZSpaceInference",
    "ZSpacePosterior",
    "ZSpacePartialBundle",
    "ZSpaceInferenceRuntime",
    "ZSpaceInferencePipeline",
    "decode_zspace_embedding",
    "infer_from_partial",
    "infer_with_partials",
    "infer_with_psi",
    "compile_inference",
    "blend_zspace_partials",
    "canvas_partial_from_snapshot",
    "canvas_coherence_partial",
    "infer_canvas_snapshot",
    "infer_canvas_transformer",
    "coherence_partial_from_diagnostics",
    "infer_coherence_diagnostics",
    "infer_coherence_from_sequencer",
    "infer_canvas_with_coherence",
    "weights_partial_from_tensor",
    "weights_partial_from_dlpack",
    "infer_weights_from_dlpack",
    "psi_partial_from_reading",
    "psi_partial_from_advisory",
    "psi_partial_from_tuning",
    "fetch_latest_psi_telemetry",
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


def _is_dynamic_metric_key(candidate: str) -> bool:
    candidate = candidate.lower()
    if candidate.startswith("psi_"):
        return True
    if "weight_" in candidate:
        return True
    if candidate.startswith("telemetry_"):
        return True
    return False


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


@dataclass(frozen=True)
class ZSpacePartialBundle:
    """Container describing a partial observation and its relative weight."""

    metrics: Mapping[str, Any]
    weight: float = 1.0
    origin: str | None = None

    def resolved(self) -> dict[str, Any]:
        """Return the canonicalised metric mapping."""

        return _canonicalise_inputs(self.metrics)


def _canonicalise_inputs(partial: Mapping[str, Any] | None) -> dict[str, Any]:
    if partial is None:
        return {}
    if not isinstance(partial, Mapping):
        raise TypeError("partial observations must be provided as a mapping")
    resolved: dict[str, Any] = {}
    for key, value in partial.items():
        lower = key.lower()
        canonical = _METRIC_ALIASES.get(lower)
        if canonical is None:
            if _is_dynamic_metric_key(lower):
                canonical = lower
            else:
                raise KeyError(f"unknown metric '{key}'")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            if canonical == "gradient":
                resolved[canonical] = value
                continue
            raise KeyError(f"unknown metric '{key}'")
        resolved[canonical] = numeric
    return resolved


def _ensure_iterable(values: Any) -> list[float]:
    if isinstance(values, Mapping):
        values = values.values()
    if not isinstance(values, Iterable):
        return []
    result: list[float] = []
    for value in values:
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            continue
    return result


def _resolve_partial(
    partial: Mapping[str, Any] | ZSpacePartialBundle | None,
    *,
    fallback_weight: float = 1.0,
) -> tuple[dict[str, Any], float] | None:
    if partial is None:
        return None
    weight = fallback_weight
    if isinstance(partial, ZSpacePartialBundle):
        weight = float(partial.weight)
        mapping = partial.resolved()
    else:
        mapping = _canonicalise_inputs(partial)
    if weight <= 0.0:
        return None
    return mapping, weight


def _merge_with_psi(
    partial: Mapping[str, Any] | ZSpacePartialBundle | None,
    psi: Any,
) -> Mapping[str, Any] | None:
    psi_metrics = _resolve_psi_partial(psi)
    base: Mapping[str, Any] | None
    if isinstance(partial, ZSpacePartialBundle):
        base = partial.resolved()
    else:
        base = partial
    if not psi_metrics:
        return base
    if base is None:
        return psi_metrics
    merged: MutableMapping[str, Any] = dict(base)
    merged.update(psi_metrics)
    return merged


def blend_zspace_partials(
    partials: Sequence[Mapping[str, Any] | ZSpacePartialBundle | None],
    *,
    weights: Sequence[float] | None = None,
    strategy: str = "mean",
) -> dict[str, Any]:
    """Fuse several partial observations into a single mapping.

    Parameters
    ----------
    partials:
        Sequence of mappings or :class:`ZSpacePartialBundle` instances. ``None``
        entries are ignored.
    weights:
        Optional per-partial weighting that overrides the bundle's intrinsic
        weight. Negative or zero weights suppress that partial.
    strategy:
        Reduction strategy used when multiple partials define the same metric.
        Supported values are ``"mean"`` (default), ``"last"``, ``"max"`` and
        ``"min"``.
    """

    if not isinstance(partials, Sequence):
        raise TypeError("partials must be provided as a sequence")

    def _reduce(values: list[tuple[float, float]]) -> float:
        if not values:
            return 0.0
        if strategy == "last":
            return values[-1][0]
        if strategy == "max":
            return max(value for value, _ in values)
        if strategy == "min":
            return min(value for value, _ in values)
        # default: weighted mean
        total_weight = sum(weight for _, weight in values)
        if total_weight <= 0.0:
            return values[-1][0]
        return sum(value * weight for value, weight in values) / total_weight

    aggregated: dict[str, list[tuple[float, float]]] = {}
    gradients: list[tuple[list[float], float]] = []
    default_weight = 1.0
    for index, partial in enumerate(partials):
        weight_override = None
        if weights is not None:
            try:
                weight_override = float(weights[index])
            except (IndexError, TypeError, ValueError):
                weight_override = None
        resolved = _resolve_partial(
            partial, fallback_weight=weight_override if weight_override is not None else default_weight
        )
        if resolved is None:
            continue
        mapping, weight = resolved
        gradient = mapping.pop("gradient", None)
        for key, value in mapping.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            aggregated.setdefault(key, []).append((numeric, weight))
        if gradient is not None:
            gradients.append((_ensure_iterable(gradient), weight))

    merged = {key: _reduce(values) for key, values in aggregated.items()}

    if gradients:
        length = max((len(values) for values, _ in gradients), default=0)
        if length:
            total_weight = 0.0
            accumulator = [0.0] * length
            for values, weight in gradients:
                padded = list(values) + [0.0] * (length - len(values))
                for idx in range(length):
                    accumulator[idx] += padded[idx] * weight
                total_weight += weight
            if total_weight > 0.0:
                merged["gradient"] = [value / total_weight for value in accumulator]
            else:
                merged["gradient"] = gradients[-1][0]
    return merged


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

    def update(
        self, partial: Mapping[str, Any] | None = None, *, psi: Any | None = None
    ) -> ZSpaceInference:
        """Fuse *partial* with any cached observations and produce an inference."""

        merged = self._merge(partial)
        payload = _merge_with_psi(merged, psi)
        return self._posterior.project(payload, smoothing=self._smoothing)

    def infer(
        self, partial: Mapping[str, Any] | None = None, *, psi: Any | None = None
    ) -> ZSpaceInference:
        """Alias for :meth:`update` to mirror the functional helpers."""

        return self.update(partial, psi=psi)


class ZSpaceInferencePipeline:
    """Composable pipeline that blends heterogeneous partials before inference."""

    def __init__(
        self,
        z_state: Sequence[float],
        *,
        alpha: float = 0.35,
        smoothing: float = 0.35,
        strategy: str = "mean",
        psi: Any | None = None,
    ) -> None:
        self._runtime = ZSpaceInferenceRuntime(
            z_state, alpha=alpha, smoothing=smoothing, accumulate=False
        )
        self._strategy = strategy
        self._partials: list[ZSpacePartialBundle] = []
        self._psi_source: Any | None = psi

    @property
    def strategy(self) -> str:
        """Return the blending strategy used for partial fusion."""

        return self._strategy

    @property
    def posterior(self) -> ZSpacePosterior:
        """Expose the underlying :class:`ZSpacePosterior`."""

        return self._runtime.posterior

    @property
    def smoothing(self) -> float:
        """Smoothing factor applied during barycentric blending."""

        return self._runtime.smoothing

    def add_partial(
        self,
        partial: Mapping[str, Any] | ZSpacePartialBundle,
        *,
        weight: float | None = None,
        origin: str | None = None,
    ) -> ZSpacePartialBundle:
        """Register a new partial observation to be included in the next inference."""

        if isinstance(partial, ZSpacePartialBundle):
            bundle = partial
        else:
            bundle = ZSpacePartialBundle(
                partial, weight=1.0 if weight is None else weight, origin=origin
            )
        self._partials.append(bundle)
        return bundle

    def add_canvas_snapshot(self, snapshot: Any, **kwargs: Any) -> ZSpacePartialBundle:
        """Derive and register metrics from a Canvas snapshot."""

        partial = canvas_partial_from_snapshot(snapshot, **kwargs)
        return self.add_partial(partial, origin="canvas")

    def add_coherence_diagnostics(
        self, diagnostics: Any, **kwargs: Any
    ) -> ZSpacePartialBundle:
        """Derive and register metrics from coherence diagnostics."""

        partial = coherence_partial_from_diagnostics(diagnostics, **kwargs)
        return self.add_partial(partial, origin="coherence")

    def clear(self) -> None:
        """Discard any buffered partial observations."""

        self._partials.clear()

    @property
    def psi_source(self) -> Any | None:
        """Return the default PSI telemetry source consulted during inference."""

        return self._psi_source

    def set_psi_source(self, psi: Any | None) -> None:
        """Update the PSI telemetry source consulted during inference."""

        self._psi_source = psi

    def infer(
        self,
        *,
        strategy: str | None = None,
        weights: Sequence[float] | None = None,
        clear: bool = True,
        psi: Any | None = None,
    ) -> ZSpaceInference:
        """Blend registered partials and compute the Z-space inference."""

        chosen_strategy = strategy or self._strategy
        blended = blend_zspace_partials(
            self._partials, strategy=chosen_strategy, weights=weights
        )
        psi_source = self._psi_source if psi is None else psi
        payload = _merge_with_psi(blended, psi_source)
        inference = self._runtime.posterior.project(
            payload, smoothing=self._runtime.smoothing
        )
        if clear:
            self.clear()
        return inference


def decode_zspace_embedding(z_state: Sequence[float], *, alpha: float = 0.35) -> ZSpaceDecoded:
    """Decode latent coordinates into a structured metric bundle."""

    return ZSpacePosterior(z_state, alpha=alpha).decode()


def infer_from_partial(
    z_state: Sequence[float],
    partial: Mapping[str, Any] | None,
    *,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    psi: Any | None = None,
) -> ZSpaceInference:
    """Fuse partial metric observations with a latent state to complete Z-space inference."""

    posterior = ZSpacePosterior(z_state, alpha=alpha)
    payload = _merge_with_psi(partial, psi)
    return posterior.project(payload, smoothing=smoothing)


def infer_with_partials(
    z_state: Sequence[float],
    *partials: Mapping[str, Any] | ZSpacePartialBundle | None,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    strategy: str = "mean",
    weights: Sequence[float] | None = None,
    psi: Any | None = None,
) -> ZSpaceInference:
    """Infer Z-space metrics from multiple partial observations."""

    blended = blend_zspace_partials(partials, weights=weights, strategy=strategy)
    return infer_from_partial(
        z_state,
        blended,
        alpha=alpha,
        smoothing=smoothing,
        psi=psi,
    )


def infer_with_psi(
    z_state: Sequence[float],
    partial: Mapping[str, Any] | ZSpacePartialBundle | None = None,
    *,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    psi: Any = True,
) -> ZSpaceInference:
    """Convenience wrapper that injects PSI telemetry before projecting."""

    return infer_from_partial(
        z_state,
        partial,
        alpha=alpha,
        smoothing=smoothing,
        psi=psi,
    )


def compile_inference(
    fn=None,
    *,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    psi: Any | None = None,
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
            actual, alpha=alpha, smoothing=smoothing, psi=psi
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
            psi=psi,
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


def _iter_values(value: Any) -> list[Any]:
    value = _maybe_call(value)
    if value is None:
        return []
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _maybe_mapping(value: Any) -> Mapping[Any, Any] | None:
    value = _maybe_call(value)
    if isinstance(value, Mapping):
        return value
    return None


def _extract_attr(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping) and name in obj:
        return obj[name]
    return getattr(obj, name, None)


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _psi_component_name(component: Any) -> str:
    if isinstance(component, str):
        label = component
    else:
        label = getattr(component, "name", None) or getattr(component, "value", None)
        if label is None:
            label = str(component)
    label = label.replace("PsiComponent::", "")
    slug = "".join(ch if ch.isalnum() else "_" for ch in label.lower()).strip("_")
    return slug or "component"


def _flatten_numeric(values: Any) -> list[float]:
    stack = [_maybe_call(values)]
    flat: list[float] = []
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        current = _maybe_call(current)
        if current is None:
            continue
        if isinstance(current, (str, bytes, bytearray)):
            numeric = _maybe_float(current)
            if numeric is not None:
                flat.append(numeric)
            continue
        if isinstance(current, Mapping):
            ident = id(current)
            if ident in seen:
                continue
            seen.add(ident)
            stack.extend(reversed(list(current.values())))
            continue
        if isinstance(current, Iterable):
            ident = id(current)
            if ident in seen:
                continue
            seen.add(ident)
            stack.extend(reversed(list(current)))
            continue
        numeric = _maybe_float(current)
        if numeric is not None:
            flat.append(numeric)
            continue
        tolist = getattr(current, "tolist", None)
        if callable(tolist):
            try:
                stack.append(tolist())
                continue
            except Exception:
                pass
        array = getattr(current, "__array__", None)
        if callable(array):
            try:
                stack.append(array())
                continue
            except Exception:
                pass
        iterator = getattr(current, "__iter__", None)
        if callable(iterator):
            try:
                stack.append(list(iterator()))
                continue
            except Exception:
                pass
    return flat


def _capture_tensor_like(value: Any, *, allow_compat: bool = True) -> Any:
    candidate = _maybe_call(value)
    if candidate is None:
        return None
    module = sys.modules.get("spiraltorch")
    if module is not None:
        from_dlpack = getattr(module, "from_dlpack", None)
        if callable(from_dlpack) and (
            hasattr(candidate, "__dlpack__")
            or getattr(candidate, "__capsule__", None) is not None
            or type(candidate).__name__ == "PyCapsule"
        ):
            try:
                return from_dlpack(candidate)
            except Exception:
                pass
        if allow_compat:
            compat = getattr(module, "compat", None)
            capture = getattr(compat, "capture", None) if compat is not None else None
            if callable(capture):
                try:
                    return capture(candidate)
                except Exception:
                    pass
    return candidate


def weights_partial_from_tensor(
    weights: Any,
    *,
    prefix: str = "weight",
) -> dict[str, float]:
    """Summarise tensor weights into canonical scalar metrics."""

    flat = _flatten_numeric(weights)
    if not flat:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_l1": 0.0,
            f"{prefix}_l2": 0.0,
            f"{prefix}_linf": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_var": 0.0,
            f"{prefix}_energy": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_spread": 0.0,
        }
    count = float(len(flat))
    total = sum(flat)
    mean = total / len(flat)
    variance = sum((value - mean) ** 2 for value in flat) / len(flat)
    l2_sq = sum(value * value for value in flat)
    energy = l2_sq / len(flat)
    metrics = {
        f"{prefix}_count": count,
        f"{prefix}_l1": sum(abs(value) for value in flat),
        f"{prefix}_l2": math.sqrt(l2_sq),
        f"{prefix}_linf": max(abs(value) for value in flat),
        f"{prefix}_mean": mean,
        f"{prefix}_var": variance,
        f"{prefix}_energy": energy,
        f"{prefix}_min": min(flat),
        f"{prefix}_max": max(flat),
        f"{prefix}_spread": max(flat) - min(flat),
    }
    return metrics


def weights_partial_from_dlpack(
    weights: Any,
    *,
    prefix: str = "weight",
    allow_compat: bool = True,
) -> dict[str, float]:
    """Capture DLPack/compat tensors and summarise them for inference."""

    tensor = _capture_tensor_like(weights, allow_compat=allow_compat)
    return weights_partial_from_tensor(tensor, prefix=prefix)


def infer_weights_from_dlpack(
    z_state: Sequence[float],
    weights: Any,
    *,
    prefix: str = "weight",
    allow_compat: bool = True,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    psi: Any | None = None,
) -> ZSpaceInference:
    """Project imported weights through the Z-space inference helpers."""

    partial = weights_partial_from_dlpack(weights, prefix=prefix, allow_compat=allow_compat)
    return infer_from_partial(
        z_state,
        partial,
        alpha=alpha,
        smoothing=smoothing,
        psi=psi,
    )


def psi_partial_from_advisory(
    advisory: Any,
    *,
    prefix: str = "psi_spiral",
) -> dict[str, float]:
    """Extract scalar PSI advisory diagnostics into inference metrics."""

    advisory = _maybe_call(advisory)
    if advisory is None:
        return {}
    metrics: dict[str, float] = {}
    for attr in ("mu_eff0", "alpha3", "audit_container_gap", "audit_cluster", "container_cluster"):
        value = _maybe_float(_extract_attr(advisory, attr))
        if value is not None:
            metrics[f"{prefix}_{attr}"] = value
    regime = _extract_attr(advisory, "regime")
    if regime is not None:
        label = _psi_component_name(regime)
        mapping = {
            "supercritical": 0.0,
            "degenerate": 0.5,
            "subcritical": 1.0,
        }
        metrics[f"{prefix}_regime"] = mapping.get(label, 0.75)
        metrics[f"{prefix}_regime_flag_{label}"] = 1.0
    try:
        stability = advisory.stability_score()  # type: ignore[attr-defined]
    except Exception:
        stability = None
    stability_value = _maybe_float(stability)
    if stability_value is not None:
        metrics[f"{prefix}_stability"] = stability_value
    try:
        audit = advisory.audit_overbias()  # type: ignore[attr-defined]
    except Exception:
        audit = None
    if isinstance(audit, bool):
        metrics[f"{prefix}_audit_overbias"] = 1.0 if audit else 0.0
    elif audit is not None:
        metrics[f"{prefix}_audit_overbias"] = float(bool(audit))
    try:
        reinforcement = advisory.container_reinforcement()  # type: ignore[attr-defined]
    except Exception:
        reinforcement = None
    reinforcement_value = _maybe_float(reinforcement)
    if reinforcement_value is not None:
        metrics[f"{prefix}_container_reinforcement"] = reinforcement_value
    return metrics


def psi_partial_from_tuning(
    tuning: Any,
    *,
    prefix: str = "psi_tuning",
) -> dict[str, float]:
    """Translate PSI tuning plans into Z-space partial metrics."""

    tuning = _maybe_call(tuning)
    if tuning is None:
        return {}
    metrics: dict[str, float] = {}
    required = _extract_attr(tuning, "required_components")
    components = _iter_values(required)
    if components:
        metrics[f"{prefix}_required_components"] = float(len(components))
    increments = _maybe_mapping(_extract_attr(tuning, "weight_increments"))
    if increments:
        total = 0.0
        for key, value in increments.items():
            numeric = _maybe_float(value)
            if numeric is None:
                continue
            metrics[f"{prefix}_weight_{_psi_component_name(key)}"] = numeric
            total += abs(numeric)
        metrics[f"{prefix}_weight_total"] = total
    thresholds = _maybe_mapping(_extract_attr(tuning, "threshold_shifts"))
    if thresholds:
        total = 0.0
        for key, value in thresholds.items():
            numeric = _maybe_float(value)
            if numeric is None:
                continue
            metrics[f"{prefix}_threshold_{_psi_component_name(key)}"] = numeric
            total += abs(numeric)
        metrics[f"{prefix}_threshold_total"] = total
    return metrics


def psi_partial_from_reading(
    reading: Any,
    *,
    events: Any = None,
    advisory: Any | None = None,
    tuning: Any | None = None,
    prefix: str = "psi",
) -> dict[str, float]:
    """Build inference metrics from PSI telemetry readings."""

    metrics: dict[str, float] = {}
    reading = _maybe_call(reading)
    if reading is not None:
        total = _maybe_float(_extract_attr(reading, "total"))
        if total is not None:
            metrics[f"{prefix}_total"] = total
        step = _maybe_float(_extract_attr(reading, "step"))
        if step is not None:
            metrics[f"{prefix}_step"] = step
        breakdown = _maybe_mapping(_extract_attr(reading, "breakdown"))
        if breakdown:
            count = 0
            for component, value in breakdown.items():
                numeric = _maybe_float(value)
                if numeric is None:
                    continue
                metrics[f"{prefix}_component_{_psi_component_name(component)}"] = numeric
                count += 1
            metrics[f"{prefix}_component_count"] = float(count)
    event_list = [item for item in (_maybe_call(evt) for evt in _iter_values(events)) if item is not None]
    if event_list:
        up = down = 0.0
        intensity = 0.0
        for event in event_list:
            direction = _extract_attr(event, "up")
            if isinstance(direction, bool):
                if direction:
                    up += 1.0
                else:
                    down += 1.0
            elif direction is not None:
                try:
                    if bool(direction):
                        up += 1.0
                    else:
                        down += 1.0
                except Exception:
                    pass
            value = _maybe_float(_extract_attr(event, "value"))
            if value is not None:
                intensity += abs(value)
        metrics[f"{prefix}_event_count"] = float(len(event_list))
        metrics[f"{prefix}_event_up"] = up
        metrics[f"{prefix}_event_down"] = down
        metrics[f"{prefix}_event_intensity"] = intensity
    if advisory is not None:
        metrics.update(psi_partial_from_advisory(advisory, prefix=f"{prefix}_spiral"))
    if tuning is not None:
        metrics.update(psi_partial_from_tuning(tuning, prefix=f"{prefix}_tuning"))
    return metrics


def fetch_latest_psi_telemetry() -> tuple[Any | None, list[Any], Any | None, Any | None]:
    """Fetch cached PSI telemetry from the runtime hub if available."""

    module = sys.modules.get("spiraltorch.telemetry.hub")
    if module is None:
        try:
            module = import_module("spiraltorch.telemetry.hub")
        except Exception:
            module = None
    if module is None:
        return None, [], None, None

    def _safe_call(name: str) -> Any:
        attr = getattr(module, name, None)
        if callable(attr):
            try:
                return attr()
            except Exception:
                return None
        return None

    reading = _safe_call("get_last_psi")
    events = _safe_call("get_last_psi_events") or []
    advisory = _safe_call("get_last_psi_spiral") or _safe_call("get_last_psi_spiral_advisory")
    tuning = _safe_call("get_last_psi_spiral_tuning") or _safe_call("get_last_psi_spiral_tuning_plan")
    return reading, list(_iter_values(events)), advisory, tuning


def _resolve_psi_partial(psi: Any) -> dict[str, float]:
    if psi is None or psi is False:
        return {}
    psi = _maybe_call(psi)
    if psi is True:
        reading, events, advisory, tuning = fetch_latest_psi_telemetry()
        return psi_partial_from_reading(
            reading,
            events=events,
            advisory=advisory,
            tuning=tuning,
        )
    if isinstance(psi, Mapping):
        return _canonicalise_inputs(psi)
    if isinstance(psi, (list, tuple)):
        reading = psi[0] if len(psi) > 0 else None
        events = psi[1] if len(psi) > 1 else None
        advisory = psi[2] if len(psi) > 2 else None
        tuning = psi[3] if len(psi) > 3 else None
        return psi_partial_from_reading(
            reading,
            events=events,
            advisory=advisory,
            tuning=tuning,
        )
    return psi_partial_from_reading(psi)


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


def canvas_coherence_partial(
    snapshot: Any,
    diagnostics: Any,
    *,
    coherence: Any = None,
    contour: Any = None,
    strategy: str = "mean",
    weights: Sequence[float] | None = None,
    canvas_kwargs: Mapping[str, Any] | None = None,
    coherence_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Blend Canvas and coherence-derived partials into a single mapping."""

    canvas_kwargs = dict(canvas_kwargs or {})
    coherence_kwargs = dict(coherence_kwargs or {})
    if coherence is not None:
        coherence_kwargs.setdefault("coherence", coherence)
    if contour is not None:
        coherence_kwargs.setdefault("contour", contour)
    canvas_partial = canvas_partial_from_snapshot(snapshot, **canvas_kwargs)
    coherence_partial = coherence_partial_from_diagnostics(
        diagnostics, **coherence_kwargs
    )
    bundles = [
        ZSpacePartialBundle(canvas_partial, origin="canvas"),
        ZSpacePartialBundle(coherence_partial, origin="coherence"),
    ]
    return blend_zspace_partials(bundles, strategy=strategy, weights=weights)


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


def infer_canvas_with_coherence(
    z_state: Sequence[float],
    snapshot: Any,
    diagnostics: Any,
    *,
    coherence: Any = None,
    contour: Any = None,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    strategy: str = "mean",
    weights: Sequence[float] | None = None,
    canvas_kwargs: Mapping[str, Any] | None = None,
    coherence_kwargs: Mapping[str, Any] | None = None,
) -> ZSpaceInference:
    """Fuse Canvas and coherence diagnostics before projecting into Z-space."""

    partial = canvas_coherence_partial(
        snapshot,
        diagnostics,
        coherence=coherence,
        contour=contour,
        strategy=strategy,
        weights=weights,
        canvas_kwargs=canvas_kwargs,
        coherence_kwargs=coherence_kwargs,
    )
    return infer_from_partial(z_state, partial, alpha=alpha, smoothing=smoothing)


