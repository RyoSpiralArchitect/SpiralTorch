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
class ChannelHealth:
    """Health metadata for a channel in a fused frame."""

    stale: bool
    optional: bool


@dataclass
class SensorChannel:
    """Descriptor for a registered sensor modality."""

    dimension: int
    bias: tuple[float, ...]
    scale: float
    optional: bool = False
    max_staleness: float | None = None
    last_timestamp: float | None = None
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

    def mark_staleness(self, now: float) -> bool:
        if self.max_staleness is None:
            return False
        if self.last_timestamp is None:
            return True
        return (now - self.last_timestamp) > self.max_staleness

    def apply(
        self,
        payload: Sequence[float] | Iterable[float],
        *,
        timestamp: float,
    ) -> tuple[float, ...]:
        values = tuple(float(value) for value in payload)
        if len(values) != self.dimension:
            raise ValueError(
                f"payload for channel must contain {self.dimension} values; got {len(values)}"
            )
        stale = self.mark_staleness(timestamp)
        adjusted: list[float] = []
        for idx, value in enumerate(values):
            calibrated = (value - self.bias[idx]) * self.scale
            if idx < len(self.smoothers):
                calibrated = self.smoothers[idx].update(calibrated)
            adjusted.append(calibrated)
        self.last_timestamp = timestamp
        return tuple(adjusted), stale


@dataclass
class FusedFrame:
    """Container representing a fused multi-modal observation."""

    coordinates: dict[str, tuple[float, ...]]
    timestamp: float
    health: dict[str, ChannelHealth]

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
        self,
        name: str,
        dimension: int,
        *,
        smoothing: float | None = None,
        optional: bool = False,
        max_staleness: float | None = None,
    ) -> None:
        if name in self._channels:
            raise ValueError(f"channel {name!r} already registered")
        if dimension <= 0:
            raise ValueError("channel dimension must be positive")
        if max_staleness is not None and max_staleness <= 0.0:
            raise ValueError("staleness threshold must be positive")
        channel = SensorChannel(
            dimension=int(dimension),
            bias=tuple(0.0 for _ in range(dimension)),
            scale=1.0,
            optional=bool(optional),
            max_staleness=float(max_staleness) if max_staleness is not None else None,
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
        now = time.time()
        coordinates: dict[str, tuple[float, ...]] = {}
        health: dict[str, ChannelHealth] = {}
        for name, channel in self._channels.items():
            raw = payloads.get(name)
            if raw is None:
                if not channel.optional:
                    raise KeyError(f"payload for required channel {name!r} missing")
                stale = channel.mark_staleness(now)
                health[name] = ChannelHealth(stale=stale, optional=True)
                continue
            adjusted, stale = channel.apply(raw, timestamp=now)
            coordinates[name] = adjusted
            health[name] = ChannelHealth(stale=stale, optional=channel.optional)
        for name in payloads:
            if name not in self._channels:
                raise KeyError(f"unknown channel {name!r}")
        return FusedFrame(coordinates=coordinates, timestamp=now, health=health)


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
        for name, health in frame.health.items():
            if health.stale:
                anomalies.append(f"stale:{name}")
                stability *= 0.5
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
        self._trajectory: deque[RuntimeStep] | None = None

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
        step = RuntimeStep(
            frame=frame,
            energy=energy,
            telemetry=report,
            commands=commands,
            halted=report.failsafe,
        )
        if self._trajectory is not None:
            self._trajectory.append(step)
        return step

    def enable_recording(self, capacity: int) -> None:
        cap = max(int(capacity), 1)
        self._trajectory = deque(maxlen=cap)

    def recording_len(self) -> int:
        if self._trajectory is None:
            return 0
        return len(self._trajectory)

    def drain_trajectory(self) -> list[RuntimeStep]:
        if self._trajectory is None:
            return []
        result = list(self._trajectory)
        self._trajectory.clear()
        return result


__all__ = [
    "ChannelHealth",
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
