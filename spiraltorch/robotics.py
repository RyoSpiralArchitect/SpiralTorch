"""High-level robotics orchestration utilities for SpiralTorch."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import time
from typing import Iterable, Mapping, MutableMapping, Sequence


@dataclass
class ExponentialSmoother:
    """Simple exponential moving-average smoother."""

    alpha: float
    state: float | None = None

    def update(self, value: float) -> float:
        if self.state is None:
            self.state = float(value)
        else:
            self.state = self.alpha * float(value) + (1.0 - self.alpha) * self.state
        return self.state


@dataclass
class SensorChannel:
    """Descriptor for a registered sensor modality."""

    dimension: int
    bias: tuple[float, ...]
    scale: float
    smoothers: list[ExponentialSmoother] = field(default_factory=list)

    def configure_smoothing(self, smoothing: float | None) -> None:
        if smoothing is None:
            self.smoothers = []
            return
        alpha = float(smoothing)
        if not 0.0 < alpha <= 1.0:
            raise ValueError(
                f"smoothing coefficient must be in the range (0, 1]; got {alpha}"
            )
        self.smoothers = [ExponentialSmoother(alpha=alpha) for _ in range(self.dimension)]

    def apply(self, payload: Sequence[float] | Iterable[float]) -> tuple[float, ...]:
        values = tuple(float(value) for value in payload)
        if len(values) != self.dimension:
            raise ValueError(
                f"payload for channel must contain {self.dimension} values; got {len(values)}"
            )
        adjusted: list[float] = []
        for idx, value in enumerate(values):
            calibrated = (value - self.bias[idx]) * self.scale
            if idx < len(self.smoothers):
                calibrated = self.smoothers[idx].update(calibrated)
            adjusted.append(calibrated)
        return tuple(adjusted)


@dataclass
class FusedFrame:
    """Container representing a fused multi-modal observation."""

    coordinates: dict[str, tuple[float, ...]]
    timestamp: float

    def norm(self, channel: str) -> float | None:
        vector = self.coordinates.get(channel)
        if vector is None:
            return None
        return math.sqrt(sum(value * value for value in vector))


@dataclass
class Desire:
    """Represent an instinct encoded as a potential field."""

    target_norm: float = 0.0
    tolerance: float = 0.0
    weight: float = 1.0

    def energy(self, norm: float) -> float:
        delta = abs(norm - float(self.target_norm)) - float(self.tolerance)
        penalty = max(delta, 0.0)
        return float(self.weight) * penalty


@dataclass
class EnergyReport:
    total: float
    per_channel: dict[str, float]


class DesireLagrangianField:
    """Aggregate instinctive potentials across registered desires."""

    def __init__(self, desires: Mapping[str, Desire] | None = None) -> None:
        self._desires: dict[str, Desire] = dict(desires or {})

    def energy(self, frame: FusedFrame) -> EnergyReport:
        per_channel: dict[str, float] = {}
        total = 0.0
        for name, desire in self._desires.items():
            norm = frame.norm(name)
            if norm is None:
                continue
            value = desire.energy(norm)
            per_channel[name] = value
            total += value
        return EnergyReport(total=total, per_channel=per_channel)


class SensorFusionHub:
    """Manage multi-modal sensor ingestion and calibration."""

    def __init__(self) -> None:
        self._channels: MutableMapping[str, SensorChannel] = {}

    def register_channel(
        self, name: str, dimension: int, *, smoothing: float | None = None
    ) -> None:
        if name in self._channels:
            raise ValueError(f"channel {name!r} already registered")
        if dimension <= 0:
            raise ValueError("channel dimension must be positive")
        channel = SensorChannel(
            dimension=int(dimension),
            bias=tuple(0.0 for _ in range(dimension)),
            scale=1.0,
        )
        if smoothing is not None:
            channel.configure_smoothing(smoothing)
        self._channels[name] = channel

    def calibrate(
        self, name: str, *, bias: Sequence[float] | None = None, scale: float | None = None
    ) -> None:
        channel = self._channels.get(name)
        if channel is None:
            raise KeyError(f"unknown channel {name!r}")
        if bias is not None:
            vector = tuple(float(value) for value in bias)
            if len(vector) != channel.dimension:
                raise ValueError(
                    f"bias for channel {name!r} must contain {channel.dimension} values; got {len(vector)}"
                )
            channel.bias = vector
        if scale is not None:
            channel.scale = float(scale)

    def configure_smoothing(self, name: str, smoothing: float | None) -> None:
        channel = self._channels.get(name)
        if channel is None:
            raise KeyError(f"unknown channel {name!r}")
        channel.configure_smoothing(smoothing)

    def fuse(self, payloads: Mapping[str, Sequence[float] | Iterable[float]]) -> FusedFrame:
        if not self._channels:
            raise RuntimeError("no sensor channels registered")
        coordinates: dict[str, tuple[float, ...]] = {}
        for name, raw in payloads.items():
            channel = self._channels.get(name)
            if channel is None:
                raise KeyError(f"unknown channel {name!r}")
            coordinates[name] = channel.apply(raw)
        return FusedFrame(coordinates=coordinates, timestamp=time.time())


@dataclass
class TelemetryReport:
    energy: float
    stability: float
    failsafe: bool
    anomalies: tuple[str, ...]


class PsiTelemetry:
    """Monitor runtime vitals and emit intervention signals."""

    def __init__(
        self,
        *,
        window: int = 8,
        stability_threshold: float = 0.5,
        failure_energy: float = 5.0,
        norm_limit: float = 10.0,
    ) -> None:
        self.window = max(int(window), 1)
        self.stability_threshold = float(stability_threshold)
        self.failure_energy = float(failure_energy)
        self.norm_limit = float(norm_limit)
        self._history: deque[float] = deque(maxlen=self.window)

    def observe(self, frame: FusedFrame, energy: EnergyReport) -> TelemetryReport:
        self._history.append(float(energy.total))
        if len(self._history) < 2:
            stability = 1.0
        else:
            mean = sum(self._history) / len(self._history)
            variance = sum((value - mean) ** 2 for value in self._history) / len(self._history)
            stability = 1.0 / (1.0 + math.sqrt(variance))

        anomalies: list[str] = []
        if stability < self.stability_threshold:
            anomalies.append("instability")
        if energy.total > self.failure_energy:
            anomalies.append("energy_overflow")
        for vector in frame.coordinates.values():
            norm = math.sqrt(sum(value * value for value in vector))
            if norm > self.norm_limit:
                anomalies.append("norm_overflow")
        anomalies = sorted(set(anomalies))
        failsafe = any(tag.startswith("norm_overflow") for tag in anomalies) or (
            energy.total > self.failure_energy
        )
        return TelemetryReport(
            energy=float(energy.total),
            stability=stability,
            failsafe=failsafe,
            anomalies=tuple(anomalies),
        )


@dataclass
class PolicyGradientController:
    base_learning_rate: float = 0.05
    smoothing: float = 0.7
    gauge: float = field(default=0.0, init=False)

    def update(self, energy: EnergyReport, telemetry: TelemetryReport) -> dict[str, float]:
        effective = self.base_learning_rate / (1.0 + float(max(energy.total, 0.0)))
        self.gauge = self.gauge * self.smoothing + telemetry.stability * (1.0 - self.smoothing)
        return {"learning_rate": effective, "gauge": self.gauge}


@dataclass
class RuntimeStep:
    frame: FusedFrame
    energy: EnergyReport
    telemetry: TelemetryReport
    commands: dict[str, float]
    halted: bool


class RoboticsRuntime:
    """Coordinate sensor fusion, instinctive planning, and telemetry."""

    def __init__(
        self,
        *,
        sensors: SensorFusionHub,
        desires: DesireLagrangianField,
        telemetry: PsiTelemetry | None = None,
    ) -> None:
        self.sensors = sensors
        self.desires = desires
        self.telemetry = telemetry or PsiTelemetry()
        self._policy: PolicyGradientController | None = None

    def attach_policy_gradient(self, controller: PolicyGradientController) -> None:
        self._policy = controller

    def step(self, payloads: Mapping[str, Sequence[float] | Iterable[float]]) -> RuntimeStep:
        frame = self.sensors.fuse(payloads)
        energy = self.desires.energy(frame)
        report = self.telemetry.observe(frame, energy)
        commands: dict[str, float] = {}
        if self._policy is not None:
            commands.update(self._policy.update(energy, report))
        if report.failsafe:
            commands["halt"] = 1.0
        return RuntimeStep(frame=frame, energy=energy, telemetry=report, commands=commands, halted=report.failsafe)


__all__ = [
    "Desire",
    "DesireLagrangianField",
    "EnergyReport",
    "FusedFrame",
    "PolicyGradientController",
    "PsiTelemetry",
    "RoboticsRuntime",
    "RuntimeStep",
    "SensorFusionHub",
    "TelemetryReport",
]
