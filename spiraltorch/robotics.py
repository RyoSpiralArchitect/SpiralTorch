"""High-level robotics orchestration utilities for SpiralTorch."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence
import time

from .rl import PolicyGradient


NumberSequence = Sequence[float] | Iterable[float]


@dataclass
class SensorChannel:
    """Descriptor for a single sensor modality.

    Each channel wraps an ``encoder`` that transforms raw sensor payloads into a
    numeric vector compatible with SpiralTorch's Z-space abstractions.  Channels
    expose light-weight calibration through ``bias`` and ``scale`` so robotics
    stacks can tweak per-modality alignment without recompiling kernels.
    """

    name: str
    encoder: Callable[[Any], NumberSequence]
    weight: float = 1.0
    bias: tuple[float, ...] | None = None
    scale: float = 1.0

    def encode(self, payload: Any) -> tuple[float, ...]:
        values = self.encoder(payload)
        if isinstance(values, Mapping):
            vector = [float(values[key]) for key in values]
        else:
            vector = [float(value) for value in values]
        if self.bias is not None:
            if len(self.bias) != len(vector):
                raise ValueError(
                    f"channel {self.name} bias length {len(self.bias)}"
                    f" does not match encoded vector length {len(vector)}"
                )
            vector = [value - offset for value, offset in zip(vector, self.bias)]
        if self.scale != 1.0:
            vector = [value * float(self.scale) for value in vector]
        return tuple(vector)


@dataclass
class ZSpaceFrame:
    """Container representing a fused multi-modal robotics observation."""

    coordinates: dict[str, tuple[float, ...]]
    weights: dict[str, float]
    timestamp: float | None = None

    def norm(self, channel: str | None = None) -> float:
        """Return the Euclidean norm of a specific channel or the entire frame."""

        if channel is not None:
            vector = self.coordinates.get(channel)
            if vector is None:
                return 0.0
            return math.sqrt(sum(value * value for value in vector))
        accum = 0.0
        for vector in self.coordinates.values():
            accum += sum(value * value for value in vector)
        return math.sqrt(accum)

    def as_vector(self, *, weighted: bool = True) -> tuple[float, ...]:
        """Flatten frame coordinates into a single vector.

        Weighted vectors multiply each modality by its associated channel weight,
        mirroring how kernels combine modalities before dispatching to WGPU.
        """

        flattened: list[float] = []
        for name in sorted(self.coordinates):
            vector = self.coordinates[name]
            weight = self.weights.get(name, 1.0)
            factor = weight if weighted else 1.0
            flattened.extend(value * factor for value in vector)
        return tuple(flattened)

    def summary(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "coordinates": {name: tuple(values) for name, values in self.coordinates.items()},
            "weights": dict(self.weights),
        }


@dataclass
class Desire:
    """Represent a robotics instinct encoded as a potential field."""

    target_norm: float = 0.0
    tolerance: float = 0.0
    weight: float = 1.0

    def energy(self, vector: Sequence[float]) -> float:
        magnitude = math.sqrt(sum(value * value for value in vector))
        excess = max(0.0, magnitude - float(self.target_norm))
        if self.tolerance > 0.0:
            excess = max(0.0, excess - float(self.tolerance))
        return float(self.weight) * excess * excess


@dataclass
class DesireEnergy:
    total: float
    per_channel: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {"total": self.total, "per_channel": dict(self.per_channel)}


class DesireLagrangianField:
    """Aggregate instinctive potentials across registered desires."""

    def __init__(self, desires: Mapping[str, Desire] | None = None) -> None:
        self._desires: dict[str, Desire] = dict(desires or {})

    def configure(self, name: str, desire: Desire) -> None:
        self._desires[name] = desire

    def remove(self, name: str) -> None:
        self._desires.pop(name, None)

    def energy(self, frame: ZSpaceFrame) -> DesireEnergy:
        per_channel: dict[str, float] = {}
        total = 0.0
        for name, desire in self._desires.items():
            vector = frame.coordinates.get(name)
            if vector is None:
                continue
            contribution = desire.energy(vector)
            per_channel[name] = contribution
            total += contribution
        return DesireEnergy(total=total, per_channel=per_channel)

    def summary(self) -> dict[str, Any]:
        return {
            name: {
                "target_norm": desire.target_norm,
                "tolerance": desire.tolerance,
                "weight": desire.weight,
            }
            for name, desire in self._desires.items()
        }


class SensorFusionHub:
    """Manage multi-modal robotics sensor ingestion and fusion."""

    def __init__(self) -> None:
        self._channels: MutableMapping[str, SensorChannel] = {}

    def register_channel(
        self,
        name: str,
        encoder: Callable[[Any], NumberSequence],
        *,
        weight: float = 1.0,
    ) -> None:
        if name in self._channels:
            raise ValueError(f"channel {name!r} already registered")
        channel = SensorChannel(name=name, encoder=encoder, weight=float(weight))
        self._channels[name] = channel

    def calibrate(self, name: str, *, bias: Sequence[float] | None = None, scale: float | None = None) -> None:
        channel = self._channels.get(name)
        if channel is None:
            raise KeyError(f"unknown channel {name!r}")
        if bias is not None:
            channel.bias = tuple(float(value) for value in bias)
        if scale is not None:
            channel.scale = float(scale)

    def fuse(self, payloads: Mapping[str, Any] | Any, *, timestamp: float | None = None) -> ZSpaceFrame:
        if not self._channels:
            raise RuntimeError("no sensor channels registered")
        coordinates: dict[str, tuple[float, ...]] = {}
        weights: dict[str, float] = {}
        for name, channel in self._channels.items():
            raw = None
            if isinstance(payloads, Mapping):
                raw = payloads.get(name)
            else:
                raw = getattr(payloads, name, None)
            if raw is None:
                continue
            vector = channel.encode(raw)
            coordinates[name] = vector
            weights[name] = channel.weight
        if not coordinates:
            raise ValueError("no sensor payloads matched registered channels")
        stamp = time.time() if timestamp is None else float(timestamp)
        return ZSpaceFrame(coordinates=coordinates, weights=weights, timestamp=stamp)

    def stream(
        self,
        samples: Iterable[Mapping[str, Any] | Any],
    ) -> Iterator[ZSpaceFrame]:
        for sample in samples:
            yield self.fuse(sample)

    def summary(self) -> dict[str, Any]:
        return {
            name: {
                "weight": channel.weight,
                "bias": channel.bias,
                "scale": channel.scale,
            }
            for name, channel in self._channels.items()
        }


@dataclass
class TelemetryReport:
    stability: float
    energy_trend: float
    anomalies: tuple[str, ...]
    failsafe: bool
    frame_norm: float
    energy_total: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "stability": self.stability,
            "energy_trend": self.energy_trend,
            "anomalies": list(self.anomalies),
            "failsafe": self.failsafe,
            "frame_norm": self.frame_norm,
            "energy_total": self.energy_total,
        }


class PsiTelemetry:
    """Monitor runtime vitals and emit intervention signals."""

    def __init__(
        self,
        *,
        window: int = 12,
        stability_threshold: float = 0.25,
        failure_energy: float = 5.0,
        norm_limit: float = 25.0,
    ) -> None:
        self.window = max(int(window), 3)
        self.stability_threshold = float(stability_threshold)
        self.failure_energy = float(failure_energy)
        self.norm_limit = float(norm_limit)
        self._norms: deque[float] = deque(maxlen=self.window)
        self._energies: deque[float] = deque(maxlen=self.window)

    def observe(self, frame: ZSpaceFrame, energy: DesireEnergy) -> TelemetryReport:
        norm = frame.norm()
        self._norms.append(norm)
        self._energies.append(float(energy.total))
        stability = self._estimate_stability()
        trend = self._estimate_trend()
        anomalies: list[str] = []
        failsafe = False
        if stability < self.stability_threshold:
            anomalies.append("stability_drop")
            failsafe = True
        if energy.total > self.failure_energy:
            anomalies.append("excess_energy")
            failsafe = True
        if norm > self.norm_limit:
            anomalies.append("norm_overflow")
            failsafe = True
        return TelemetryReport(
            stability=stability,
            energy_trend=trend,
            anomalies=tuple(anomalies),
            failsafe=failsafe,
            frame_norm=norm,
            energy_total=float(energy.total),
        )

    def _estimate_stability(self) -> float:
        if len(self._norms) < 2:
            return 1.0
        diffs = [abs(b - a) for a, b in zip(self._norms, list(self._norms)[1:])]
        if not diffs:
            return 1.0
        max_diff = max(diffs)
        mean_diff = sum(diffs) / len(diffs)
        denominator = mean_diff + max_diff + 1e-6
        return 1.0 / (1.0 + denominator)

    def _estimate_trend(self) -> float:
        if len(self._energies) < 2:
            return 0.0
        first = self._energies[0]
        last = self._energies[-1]
        return (last - first) / max(len(self._energies) - 1, 1)


@dataclass
class RuntimeStep:
    frame: ZSpaceFrame
    energy: DesireEnergy
    telemetry: TelemetryReport
    commands: dict[str, float]
    halted: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame.summary(),
            "energy": self.energy.as_dict(),
            "telemetry": self.telemetry.as_dict(),
            "commands": dict(self.commands),
            "halted": self.halted,
        }


class RoboticsRuntime:
    """Coordinate sensor fusion, instinctive planning, and telemetry."""

    def __init__(
        self,
        *,
        sensors: SensorFusionHub | None = None,
        desires: DesireLagrangianField | None = None,
        telemetry: PsiTelemetry | None = None,
    ) -> None:
        self.sensors = sensors or SensorFusionHub()
        self.desires = desires or DesireLagrangianField()
        self.telemetry = telemetry or PsiTelemetry()
        self._policy_cb: Callable[[ZSpaceFrame, DesireEnergy, TelemetryReport], dict[str, float]] | None = None
        self._policy_gradient: PolicyGradient | None = None
        self._last_step: RuntimeStep | None = None

    @property
    def last_step(self) -> RuntimeStep | None:
        return self._last_step

    def attach_policy(self, callback: Callable[[ZSpaceFrame, DesireEnergy, TelemetryReport], dict[str, float]]) -> None:
        self._policy_cb = callback

    def attach_policy_gradient(self, policy: PolicyGradient) -> None:
        self._policy_gradient = policy

    def step(
        self,
        payloads: Mapping[str, Any] | Any,
        *,
        timestamp: float | None = None,
        returns: Iterable[float] | None = None,
        baseline: float = 0.0,
    ) -> RuntimeStep:
        frame = self.sensors.fuse(payloads, timestamp=timestamp)
        energy = self.desires.energy(frame)
        report = self.telemetry.observe(frame, energy)
        commands: dict[str, float]
        halted = report.failsafe
        if report.failsafe:
            commands = {"halt": 1.0, "energy_total": energy.total}
        elif self._policy_cb is not None:
            commands = dict(self._policy_cb(frame, energy, report))
        elif self._policy_gradient is not None:
            reward_stream: Iterable[float]
            if returns is not None:
                reward_stream = returns
            else:
                reward_stream = (-energy.total,)
            commands = self._policy_gradient.step(reward_stream, baseline=baseline)
        else:
            commands = {}
        step = RuntimeStep(frame=frame, energy=energy, telemetry=report, commands=commands, halted=halted)
        self._last_step = step
        return step


__all__ = [
    "Desire",
    "DesireEnergy",
    "DesireLagrangianField",
    "PsiTelemetry",
    "RoboticsRuntime",
    "RuntimeStep",
    "SensorChannel",
    "SensorFusionHub",
    "TelemetryReport",
    "ZSpaceFrame",
]
