"""Inference helpers that reconstruct Z-space metrics from partial observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence
from types import MappingProxyType

__all__ = [
    "ZSpaceDecoded",
    "ZSpaceInference",
    "ZSpacePosterior",
    "decode_zspace_embedding",
    "infer_from_partial",
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


