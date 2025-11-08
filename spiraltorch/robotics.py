"""High-level robotics orchestration utilities for SpiralTorch."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import time
from typing import Iterable, Mapping, MutableMapping, Sequence


def _safe_mean(values: Sequence[float]) -> float:
    """Return the mean of *values* or 0.0 when empty."""

    return sum(values) / len(values) if values else 0.0


class ZSpaceGeometry:
    """Geometry helper for computing norms in Z-space."""

    def __init__(
        self,
        *,
        kind: str,
        curvature: float = 0.0,
        metric: Sequence[Sequence[float]] | None = None,
        time_dilation: float = 1.0,
    ) -> None:
        self._kind = kind
        self._curvature = float(curvature)
        if metric is not None:
            self._metric = tuple(tuple(float(value) for value in row) for row in metric)
        else:
            self._metric = tuple()
        self._time_dilation = max(float(time_dilation), 1e-6)

    @staticmethod
    def euclidean() -> "ZSpaceGeometry":
        return ZSpaceGeometry(kind="euclidean")

    @staticmethod
    def non_euclidean(curvature: float) -> "ZSpaceGeometry":
        return ZSpaceGeometry(kind="non_euclidean", curvature=curvature)

    @staticmethod
    def general_relativity(
        metric: Sequence[Sequence[float]],
        time_dilation: float = 1.0,
    ) -> "ZSpaceGeometry":
        return ZSpaceGeometry(
            kind="general_relativity", metric=metric, time_dilation=time_dilation
        )

    def metric_norm(self, vector: Sequence[float]) -> float:
        if not vector:
            return 0.0
        euclidean = math.sqrt(sum(float(value) * float(value) for value in vector))
        if self._kind == "euclidean":
            return euclidean
        if self._kind == "non_euclidean":
            curvature = self._curvature
            if abs(curvature) < 1e-6:
                return euclidean
            if curvature > 0.0:
                adjustment = 1.0 + curvature * euclidean * euclidean / 6.0
                return euclidean * adjustment
            contraction = 1.0 + abs(curvature) * euclidean * euclidean / 6.0
            return euclidean / max(contraction, 1e-6)
        # General relativity fallback
        metric = self._metric
        if not metric:
            return euclidean * self._time_dilation
        accumulator = 0.0
        for i, value_i in enumerate(vector):
            row = metric[i] if i < len(metric) else ()
            for j, value_j in enumerate(vector):
                coefficient = (
                    row[j]
                    if j < len(row)
                    else (1.0 if i == j else 0.0)
                )
                accumulator += float(value_i) * coefficient * float(value_j)
        spatial = math.sqrt(abs(accumulator))
        return spatial * self._time_dilation

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"ZSpaceGeometry(kind={self._kind!r})"

    @property
    def kind(self) -> str:
        return self._kind


@dataclass
class GravityWell:
    mass: float
    regime: str = "newtonian"
    speed_of_light: float = 1.0

    @staticmethod
    def newtonian(mass: float) -> "GravityWell":
        return GravityWell(mass=float(mass), regime="newtonian")

    @staticmethod
    def relativistic(mass: float, speed_of_light: float) -> "GravityWell":
        return GravityWell(
            mass=float(mass), regime="relativistic", speed_of_light=float(speed_of_light)
        )


@dataclass
class GravityField:
    constant: float = 6.67430e-11
    wells: dict[str, GravityWell] = field(default_factory=dict)

    def add_well(self, channel: str, well: GravityWell) -> None:
        self.wells[channel] = well

    def potential(self, channel: str, radius: float) -> float | None:
        well = self.wells.get(channel)
        if well is None:
            return None
        radius = float(radius)
        if radius <= 1e-6:
            return 0.0
        base = -self.constant * float(well.mass) / radius
        if well.regime == "relativistic":
            c = max(float(well.speed_of_light), 1e-6)
            base *= 1.0 / (1.0 + radius / c)
        return base


@dataclass
class ZSpaceDynamics:
    geometry: ZSpaceGeometry = field(default_factory=ZSpaceGeometry.euclidean)
    gravity: GravityField | None = None


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
    gravitational: float
    gravitational_per_channel: dict[str, float]


class DesireLagrangianField:
    """Aggregate instinctive potentials across registered desires."""

    def __init__(
        self,
        desires: Mapping[str, Desire] | None = None,
        *,
        dynamics: ZSpaceDynamics | None = None,
    ) -> None:
        self._desires: dict[str, Desire] = dict(desires or {})
        self._dynamics = dynamics or ZSpaceDynamics()

    def energy(self, frame: FusedFrame) -> EnergyReport:
        per_channel: dict[str, float] = {}
        gravitational: dict[str, float] = {}
        total = 0.0
        for name, vector in frame.coordinates.items():
            desire = self._desires.get(name)
            if desire is not None:
                norm = self._dynamics.geometry.metric_norm(vector)
                value = desire.energy(norm)
                per_channel[name] = value
                total += value
            if self._dynamics.gravity is not None:
                radius = self._dynamics.geometry.metric_norm(vector)
                potential = self._dynamics.gravity.potential(name, radius)
                if potential is not None:
                    gravitational[name] = potential
                    total += abs(potential)
        gravity_total = sum(gravitational.values()) if gravitational else 0.0
        return EnergyReport(
            total=total,
            per_channel=per_channel,
            gravitational=gravity_total,
            gravitational_per_channel=gravitational,
        )

    @property
    def dynamics(self) -> ZSpaceDynamics:
        return self._dynamics

    def set_dynamics(self, dynamics: ZSpaceDynamics) -> None:
        self._dynamics = dynamics


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

    def stability_margin(self, threshold: float) -> float:
        return float(self.stability) - float(threshold)


@dataclass
class TelemetryInsight:
    stability_margin: float
    energy_trend: float
    stability_trend: float
    anomaly_pressure: float
    energy_baseline: float

    def integration_factor(self) -> float:
        margin = max(self.stability_margin, 0.0)
        relief = max(-self.energy_trend, 0.0)
        resilience = max(self.stability_trend, 0.0)
        return margin + 0.5 * relief + 0.25 * resilience

    def to_dict(self, prefix: str = "telemetry") -> dict[str, float]:
        return {
            f"{prefix}.stability_margin": self.stability_margin,
            f"{prefix}.energy_trend": self.energy_trend,
            f"{prefix}.stability_trend": self.stability_trend,
            f"{prefix}.anomaly_pressure": self.anomaly_pressure,
            f"{prefix}.energy_baseline": self.energy_baseline,
        }


@dataclass
class AtlasSnapshot:
    """Short-lived container capturing telemetry, harmony, and synergy factors."""

    stability_margin: float
    synergy_index: float
    harmony: float
    resonance: float
    drift_penalty: float
    anomaly_pressure: float
    adaptation_readiness: float
    energy_trend: float
    curvature: float

    def coherence_score(self) -> float:
        margin = max(self.stability_margin, 0.0)
        synergy = max(self.synergy_index, 0.0)
        resonance = max(self.resonance, 0.0)
        drift = max(self.drift_penalty, 0.0)
        attenuation = 1.0 - min(drift, 1.0)
        return (margin + 0.5 * synergy + 0.25 * resonance) * attenuation

    def resilience_score(self) -> float:
        readiness = max(self.adaptation_readiness, 0.0)
        anomaly = max(self.anomaly_pressure, 0.0)
        return readiness * (1.0 - min(anomaly, 1.0))


class TelemetryAtlas:
    """Maintain a rolling window of telemetry-driven systemic summaries."""

    def __init__(self, window: int = 8) -> None:
        self.window = max(int(window), 1)
        self._history: deque[AtlasSnapshot] = deque(maxlen=self.window)

    def record(
        self,
        *,
        stability_margin: float,
        synergy_index: float,
        harmony: float,
        resonance: float,
        drift_penalty: float,
        anomaly_pressure: float,
        adaptation_readiness: float,
        energy_trend: float,
        curvature: float,
    ) -> dict[str, float]:
        snapshot = AtlasSnapshot(
            stability_margin=float(stability_margin),
            synergy_index=float(synergy_index),
            harmony=float(harmony),
            resonance=float(resonance),
            drift_penalty=float(drift_penalty),
            anomaly_pressure=float(anomaly_pressure),
            adaptation_readiness=float(adaptation_readiness),
            energy_trend=float(energy_trend),
            curvature=float(curvature),
        )
        self._history.append(snapshot)
        return self.summary()

    def summary(self) -> dict[str, float]:
        snapshots = list(self._history)
        if not snapshots:
            return {
                "ecosystem_cohesion": 0.0,
                "ecosystem_resilience": 0.0,
                "atlas_memory_strength": 0.0,
                "atlas_window_fill": 0.0,
                "atlas_curvature_span": 0.0,
                "atlas_margin_mean": 0.0,
                "atlas_synergy_mean": 0.0,
                "atlas_resonance_mean": 0.0,
                "atlas_drift_mean": 0.0,
                "atlas_anomaly_pressure_mean": 0.0,
                "atlas_adaptation_mean": 0.0,
                "atlas_energy_trend_mean": 0.0,
                "atlas_harmony_trend": 0.0,
            }

        margin_mean = _safe_mean([entry.stability_margin for entry in snapshots])
        synergy_mean = _safe_mean([entry.synergy_index for entry in snapshots])
        resonance_mean = _safe_mean([entry.resonance for entry in snapshots])
        drift_mean = _safe_mean([entry.drift_penalty for entry in snapshots])
        anomaly_mean = _safe_mean([entry.anomaly_pressure for entry in snapshots])
        adaptation_mean = _safe_mean([entry.adaptation_readiness for entry in snapshots])
        energy_trend_mean = _safe_mean([entry.energy_trend for entry in snapshots])
        cohesion = _safe_mean([entry.coherence_score() for entry in snapshots])
        resilience = _safe_mean([entry.resilience_score() for entry in snapshots])
        harmony_trend = (
            snapshots[-1].harmony - snapshots[0].harmony if len(snapshots) > 1 else 0.0
        )
        curvature_span = (
            snapshots[-1].curvature - snapshots[0].curvature if len(snapshots) > 1 else 0.0
        )
        memory_strength = max(
            0.0,
            min(
                1.0,
                0.5 * resilience + 0.5 * max(snapshots[-1].harmony, 0.0),
            ),
        )

        return {
            "ecosystem_cohesion": cohesion,
            "ecosystem_resilience": resilience,
            "atlas_memory_strength": memory_strength,
            "atlas_window_fill": len(snapshots) / float(self.window),
            "atlas_curvature_span": curvature_span,
            "atlas_margin_mean": margin_mean,
            "atlas_synergy_mean": synergy_mean,
            "atlas_resonance_mean": resonance_mean,
            "atlas_drift_mean": drift_mean,
            "atlas_anomaly_pressure_mean": anomaly_mean,
            "atlas_adaptation_mean": adaptation_mean,
            "atlas_energy_trend_mean": energy_trend_mean,
            "atlas_harmony_trend": harmony_trend,
        }

    def history_size(self) -> int:
        return len(self._history)

class PsiTelemetry:
    """Monitor runtime vitals and emit intervention signals."""

    def __init__(
        self,
        *,
        window: int = 8,
        stability_threshold: float = 0.5,
        failure_energy: float = 5.0,
        norm_limit: float = 10.0,
        geometry: ZSpaceGeometry | None = None,
    ) -> None:
        self.window = max(int(window), 1)
        self.stability_threshold = float(stability_threshold)
        self.failure_energy = float(failure_energy)
        self.norm_limit = float(norm_limit)
        self._history: deque[float] = deque(maxlen=self.window)
        self._stability_history: deque[float] = deque(maxlen=self.window)
        self._geometry = geometry or ZSpaceGeometry.euclidean()

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
        if abs(energy.gravitational) > self.failure_energy:
            anomalies.append("gravity_overflow")
        for vector in frame.coordinates.values():
            norm = self._geometry.metric_norm(vector)
            if norm > self.norm_limit:
                anomalies.append("norm_overflow")
        for name, health in frame.health.items():
            if health.stale:
                anomalies.append(f"stale:{name}")
                stability *= 0.5
        anomalies = sorted(set(anomalies))
        failsafe = any(tag.startswith("norm_overflow") for tag in anomalies) or (
            energy.total > self.failure_energy
        ) or (abs(energy.gravitational) > self.failure_energy)
        report = TelemetryReport(
            energy=float(energy.total),
            stability=stability,
            failsafe=failsafe,
            anomalies=tuple(anomalies),
        )
        self._stability_history.append(report.stability)
        return report

    def insight(self, report: TelemetryReport) -> TelemetryInsight:
        if self._history:
            history_values = list(self._history)
            energy_trend = history_values[-1] - history_values[0]
            energy_baseline = sum(history_values) / len(history_values)
        else:
            energy_trend = 0.0
            energy_baseline = report.energy
        if self._stability_history:
            stability_values = list(self._stability_history)
            stability_mean = sum(stability_values) / len(stability_values)
            stability_trend = stability_values[-1] - stability_mean
        else:
            stability_trend = 0.0
        anomaly_pressure = min(1.0, len(report.anomalies) / max(self.window, 1))
        margin = report.stability_margin(self.stability_threshold)
        return TelemetryInsight(
            stability_margin=margin,
            energy_trend=energy_trend,
            stability_trend=stability_trend,
            anomaly_pressure=anomaly_pressure,
            energy_baseline=energy_baseline,
        )

    def set_geometry(self, geometry: ZSpaceGeometry) -> None:
        self._geometry = geometry

    @property
    def geometry(self) -> ZSpaceGeometry:
        return self._geometry

    @property
    def window_size(self) -> int:
        return self.window

    @property
    def stability_threshold_value(self) -> float:
        return self.stability_threshold

    @property
    def failure_energy_limit(self) -> float:
        return self.failure_energy

    @property
    def norm_limit_value(self) -> float:
        return self.norm_limit


@dataclass
class PolicyGradientController:
    base_learning_rate: float = 0.05
    smoothing: float = 0.7
    gauge: float = field(default=0.0, init=False)

    def update(self, energy: EnergyReport, telemetry: TelemetryReport) -> dict[str, float]:
        effective = self.base_learning_rate / (1.0 + float(max(energy.total, 0.0)))
        self.gauge = self.gauge * self.smoothing + telemetry.stability * (1.0 - self.smoothing)
        return {"learning_rate": effective, "gauge": self.gauge}


class SafetyPlugin:
    """Interface for integrating spiral-safety style plugins."""

    def review(
        self,
        frame: "FusedFrame",
        energy: "EnergyReport",
        telemetry: TelemetryReport,
    ) -> "SafetyReview":
        raise NotImplementedError


@dataclass
class SafetyMetrics:
    frame_hazards: dict[str, float]
    safe_radii: dict[str, float]
    existence_load: float
    chi: int
    strict_mode: bool


@dataclass
class SafetyReview:
    hazard_total: float
    refused: bool
    flagged_frames: tuple[str, ...]
    metrics: SafetyMetrics


class DriftSafetyPlugin(SafetyPlugin):
    """Fallback drift-response style safety plugin."""

    def __init__(self, word_name: str = "Robotics", hazard_cut: float = 0.8) -> None:
        self.word_name = str(word_name)
        self.hazard_cut = max(float(hazard_cut), 0.0)
        self._thresholds: dict[str, float] = {}

    def set_threshold(self, channel: str, hazard: float) -> None:
        self._thresholds[str(channel)] = float(hazard)

    def review(
        self,
        frame: "FusedFrame",
        energy: "EnergyReport",
        telemetry: TelemetryReport,
    ) -> SafetyReview:
        hazards: dict[str, float] = {}
        radii: dict[str, float] = {}
        flagged: list[str] = []
        for name, vector in frame.coordinates.items():
            threshold = self._thresholds.setdefault(name, self.hazard_cut)
            channel_energy = abs(energy.per_channel.get(name, 0.0))
            gravitational = abs(energy.gravitational_per_channel.get(name, 0.0))
            radius = math.sqrt(sum(float(value) * float(value) for value in vector))
            stability = max(0.0, min(1.0, float(telemetry.stability)))
            phi = 1.0 - stability
            anomaly_prefix = f"stale:{name}"
            anomaly_hits = sum(
                1
                for tag in telemetry.anomalies
                if tag in {"instability", "energy_overflow", "gravity_overflow", "norm_overflow"}
                or tag.startswith(anomaly_prefix)
            )
            penalty = anomaly_hits + (1 if telemetry.failsafe else 0)
            hazard = (channel_energy + gravitational + radius + penalty) * (1.0 + phi)
            hazards[name] = hazard
            radii[name] = (1.0 / (1.0 + hazard)) if hazard > 0.0 else float("inf")
            if hazard >= threshold:
                flagged.append(name)
        hazard_total = sum(hazards.values())
        chi = len(flagged)
        existence_load = hazard_total
        strict = telemetry.failsafe or chi > 0 or existence_load >= 1.0
        metrics = SafetyMetrics(
            frame_hazards=hazards,
            safe_radii=radii,
            existence_load=existence_load,
            chi=chi,
            strict_mode=strict,
        )
        refused = strict
        return SafetyReview(
            hazard_total=hazard_total,
            refused=refused,
            flagged_frames=tuple(flagged),
            metrics=metrics,
        )


@dataclass
class RuntimeStep:
    frame: FusedFrame
    energy: EnergyReport
    telemetry: TelemetryReport
    commands: dict[str, float]
    halted: bool
    safety: tuple[SafetyReview, ...]


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
        self._safety_plugins: list[SafetyPlugin] = []
        self.telemetry.set_geometry(self.desires.dynamics.geometry)

    def attach_policy_gradient(self, controller: PolicyGradientController) -> None:
        self._policy = controller

    def attach_safety_plugin(self, plugin: SafetyPlugin) -> None:
        self._safety_plugins.append(plugin)

    def clear_safety_plugins(self) -> None:
        self._safety_plugins.clear()

    def step(self, payloads: Mapping[str, Sequence[float] | Iterable[float]]) -> RuntimeStep:
        frame = self.sensors.fuse(payloads)
        energy = self.desires.energy(frame)
        report = self.telemetry.observe(frame, energy)
        commands: dict[str, float] = {}
        if self._policy is not None:
            commands.update(self._policy.update(energy, report))
        safety_reviews: list[SafetyReview] = []
        halted = report.failsafe
        for plugin in self._safety_plugins:
            review = plugin.review(frame, energy, report)
            safety_reviews.append(review)
            if review.refused:
                halted = True
        if halted:
            commands["halt"] = 1.0
        step = RuntimeStep(
            frame=frame,
            energy=energy,
            telemetry=report,
            commands=commands,
            halted=halted,
            safety=tuple(safety_reviews),
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

    def configure_dynamics(self, dynamics: ZSpaceDynamics) -> None:
        self.desires.set_dynamics(dynamics)
        self.telemetry.set_geometry(self.desires.dynamics.geometry)


def _summarise_canvas_vectors(
    vectors: Sequence[Sequence[float]],
) -> tuple[float, float, tuple[float, float, float]]:
    energy_sum = 0.0
    energy_sq = 0.0
    chroma_r = 0.0
    chroma_g = 0.0
    chroma_b = 0.0
    count = 0
    for vector in vectors:
        if len(vector) != 4:
            raise ValueError("canvas vectors must contain four components [energy, r, g, b]")
        energy = float(vector[0])
        energy_sum += energy
        energy_sq += energy * energy
        chroma_r += float(vector[1])
        chroma_g += float(vector[2])
        chroma_b += float(vector[3])
        count += 1
    if count == 0:
        return 0.0, 0.0, (0.0, 0.0, 0.0)
    inv = 1.0 / count
    mean = energy_sum * inv
    rms = math.sqrt(energy_sq * inv)
    return mean, rms, (chroma_r * inv, chroma_g * inv, chroma_b * inv)


@dataclass
class VisionFeedbackSnapshot:
    channel: str
    timestamp: float
    sensor: tuple[float, ...]
    sensor_norm: float
    canvas_mean_energy: float
    canvas_rms_energy: float
    chroma: tuple[float, float, float]
    alignment: float

    def metrics(self) -> dict[str, float]:
        return {
            "vision.energy.mean": self.canvas_mean_energy,
            "vision.energy.rms": self.canvas_rms_energy,
            "vision.alignment": self.alignment,
            "vision.sensor.norm": self.sensor_norm,
            "vision.chroma.r": self.chroma[0],
            "vision.chroma.g": self.chroma[1],
            "vision.chroma.b": self.chroma[2],
        }

    def gradient_component(self) -> list[float]:
        return [self.alignment, self.canvas_rms_energy]


class VisionFeedbackSynchronizer:
    """Pair sensor channels with CanvasProjector vector feedback."""

    def __init__(
        self,
        channel: str,
        *,
        coherence: float = 1.0,
        tension: float = 1.0,
        depth: int = 1,
    ) -> None:
        self._channel = channel
        self._coherence = float(coherence)
        self._tension = float(tension)
        self._depth = max(int(depth), 1)

    @property
    def channel(self) -> str:
        return self._channel

    def set_patch(self, coherence: float, tension: float, depth: int) -> None:
        self._coherence = float(coherence)
        self._tension = float(tension)
        self._depth = max(int(depth), 1)

    def sync(
        self,
        step: RuntimeStep,
        vectors: Sequence[Sequence[float]],
    ) -> VisionFeedbackSnapshot:
        sensor = step.frame.coordinates.get(self._channel)
        if sensor is None:
            raise ValueError(f"channel {self._channel!r} missing from fused frame")
        sensor_tuple = tuple(float(value) for value in sensor)
        sensor_norm = math.sqrt(sum(value * value for value in sensor_tuple))
        mean_energy, rms_energy, chroma = _summarise_canvas_vectors(vectors)
        alignment = mean_energy / sensor_norm if sensor_norm > 0.0 else 0.0
        return VisionFeedbackSnapshot(
            channel=self._channel,
            timestamp=float(step.frame.timestamp),
            sensor=sensor_tuple,
            sensor_norm=sensor_norm,
            canvas_mean_energy=mean_energy,
            canvas_rms_energy=rms_energy,
            chroma=tuple(chroma),
            alignment=alignment,
        )


@dataclass
class ZSpacePartialObservation:
    metrics: dict[str, float]
    commands: dict[str, float]
    gradient: list[float]
    weight: float


@dataclass
class TemporalFeedbackSample:
    timestamp: float
    energy_total: float
    gravitational_total: float
    psi_stability: float
    psi_failsafe: bool
    psi_anomalies: tuple[str, ...]
    per_channel_energy: dict[str, float]
    per_channel_gravity: dict[str, float]
    channel_norms: dict[str, float]
    average_norm: float
    commands: dict[str, float]
    halted: bool
    vision_energy: float | None
    vision_rms: float | None
    vision_alignment: float | None

    @classmethod
    def from_step(
        cls,
        step: RuntimeStep,
        vision: VisionFeedbackSnapshot | None = None,
    ) -> "TemporalFeedbackSample":
        channel_norms = {
            name: math.sqrt(sum(value * value for value in vector))
            for name, vector in step.frame.coordinates.items()
        }
        average_norm = (
            sum(channel_norms.values()) / len(channel_norms)
            if channel_norms
            else 0.0
        )
        return cls(
            timestamp=float(step.frame.timestamp),
            energy_total=float(step.energy.total),
            gravitational_total=float(step.energy.gravitational),
            psi_stability=float(step.telemetry.stability),
            psi_failsafe=bool(step.telemetry.failsafe),
            psi_anomalies=tuple(step.telemetry.anomalies),
            per_channel_energy=dict(step.energy.per_channel),
            per_channel_gravity=dict(step.energy.gravitational_per_channel),
            channel_norms=channel_norms,
            average_norm=average_norm,
            commands=dict(step.commands),
            halted=bool(step.halted),
            vision_energy=vision.canvas_mean_energy if vision else None,
            vision_rms=vision.canvas_rms_energy if vision else None,
            vision_alignment=vision.alignment if vision else None,
        )

    def anomaly_score(self) -> float:
        score = 1.0 if self.psi_failsafe else 0.0
        score += float(len(self.psi_anomalies))
        if self.halted:
            score += 1.0
        return score

    def gradient(self) -> list[float]:
        pairs = []
        for name, value in self.per_channel_energy.items():
            gravity = self.per_channel_gravity.get(name, 0.0)
            pairs.append((name, value - gravity))
        pairs.sort(key=lambda item: item[0])
        gradient = [delta for _, delta in pairs]
        if self.vision_alignment is not None:
            gradient.append(self.vision_alignment)
        if self.vision_rms is not None:
            gradient.append(self.vision_rms)
        return gradient

    def to_partial(self) -> ZSpacePartialObservation:
        metrics: dict[str, float] = {
            "psi.energy": self.energy_total,
            "psi.stability": self.psi_stability,
            "psi.failsafe": 1.0 if self.psi_failsafe else 0.0,
            "psi.anomalies": float(len(self.psi_anomalies)),
            "desire.energy.total": self.energy_total,
            "gravity.energy.total": self.gravitational_total,
            "sensor.norm.mean": self.average_norm,
        }
        for name, value in self.per_channel_energy.items():
            metrics[f"desire.energy.{name}"] = value
        for name, value in self.per_channel_gravity.items():
            metrics[f"gravity.energy.{name}"] = value
        for name, value in self.channel_norms.items():
            metrics[f"sensor.norm.{name}"] = value
        if self.vision_energy is not None:
            metrics["vision.energy.mean"] = self.vision_energy
        if self.vision_rms is not None:
            metrics["vision.energy.rms"] = self.vision_rms
        if self.vision_alignment is not None:
            metrics["vision.alignment"] = self.vision_alignment
        return ZSpacePartialObservation(
            metrics=metrics,
            commands=dict(self.commands),
            gradient=self.gradient(),
            weight=1.0,
        )


@dataclass
class TrainerMetrics:
    speed: float
    memory: float
    stability: float
    gradient: list[float]
    drs: float


@dataclass
class TemporalFeedbackSummary:
    discounted_energy: float
    discounted_gravity: float
    discounted_stability: float
    discounted_alignment: float | None
    commands: dict[str, float]
    partial: ZSpacePartialObservation
    latest_sensor_norm: float
    anomaly_score: float


class TemporalFeedbackLearner:
    """Aggregate runtime steps into discounted feedback signals."""

    def __init__(self, horizon: int, *, discount: float = 0.9) -> None:
        if horizon <= 0:
            raise ValueError("temporal horizon must be positive")
        if not 0.0 < float(discount) <= 1.0:
            raise ValueError("discount factor must lie in (0, 1]")
        self._horizon = int(horizon)
        self._discount = float(discount)
        self._buffer: deque[TemporalFeedbackSample] = deque(maxlen=self._horizon)

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def discount(self) -> float:
        return self._discount

    def push(
        self,
        step: RuntimeStep,
        vision: VisionFeedbackSnapshot | None = None,
    ) -> TemporalFeedbackSummary:
        sample = TemporalFeedbackSample.from_step(step, vision)
        if len(self._buffer) == self._horizon:
            self._buffer.popleft()
        self._buffer.append(sample)

        energy = 0.0
        gravity = 0.0
        stability = 0.0
        alignment_total = 0.0
        alignment_weight = 0.0
        commands: dict[str, float] = {}
        weight = 0.0
        factor = 1.0
        for snapshot in reversed(self._buffer):
            energy += factor * snapshot.energy_total
            gravity += factor * snapshot.gravitational_total
            stability += factor * snapshot.psi_stability
            if snapshot.vision_alignment is not None:
                alignment_total += factor * snapshot.vision_alignment
                alignment_weight += factor
            for name, value in snapshot.commands.items():
                commands[name] = commands.get(name, 0.0) + factor * value
            weight += factor
            factor *= self._discount
        if weight > 0.0:
            inv = 1.0 / weight
            energy *= inv
            gravity *= inv
            stability *= inv
            commands = {name: value * inv for name, value in commands.items()}
        alignment = (
            alignment_total / alignment_weight if alignment_weight > 0.0 else None
        )
        partial = self._buffer[-1].to_partial()
        partial.weight = max(weight, 1.0)
        partial.metrics["feedback.energy.discounted"] = energy
        partial.metrics["feedback.gravity.discounted"] = gravity
        partial.metrics["feedback.stability.discounted"] = stability
        if alignment is not None:
            partial.metrics["vision.alignment.discounted"] = alignment
        for name, value in commands.items():
            partial.metrics[f"feedback.command.{name}"] = value
        summary = TemporalFeedbackSummary(
            discounted_energy=energy,
            discounted_gravity=gravity,
            discounted_stability=stability,
            discounted_alignment=alignment,
            commands=commands,
            partial=partial,
            latest_sensor_norm=self._buffer[-1].average_norm,
            anomaly_score=self._buffer[-1].anomaly_score(),
        )
        return summary


@dataclass
class ZSpaceTrainerSample:
    metrics: TrainerMetrics
    partial: ZSpacePartialObservation


@dataclass
class TrainerEpisode:
    samples: list[ZSpaceTrainerSample]
    average_memory: float
    average_stability: float
    average_drs: float
    length: int


class ZSpaceTrainerBridge:
    """Project robotics rollouts into ZSpaceTrainer metrics."""

    def __init__(self, horizon: int, *, discount: float = 0.9) -> None:
        self._learner = TemporalFeedbackLearner(horizon, discount=discount)

    @property
    def horizon(self) -> int:
        return self._learner.horizon

    @property
    def discount(self) -> float:
        return self._learner.discount

    def push(
        self,
        step: RuntimeStep,
        vision: VisionFeedbackSnapshot | None = None,
    ) -> ZSpaceTrainerSample:
        summary = self._learner.push(step, vision)
        gradient = list(summary.partial.gradient)
        if not gradient:
            gradient.append(summary.discounted_energy - summary.discounted_gravity)
        metrics = TrainerMetrics(
            speed=summary.latest_sensor_norm,
            memory=summary.discounted_energy + summary.discounted_gravity,
            stability=summary.discounted_stability,
            gradient=gradient,
            drs=summary.anomaly_score,
        )
        return ZSpaceTrainerSample(metrics=metrics, partial=summary.partial)


class ZSpaceTrainerEpisodeBuilder:
    """Collect discounted trainer samples across a robotics episode."""

    def __init__(
        self,
        horizon: int,
        *,
        discount: float = 0.9,
        capacity: int = 64,
    ) -> None:
        if capacity <= 0:
            raise ValueError("episode capacity must be positive")
        self._bridge = ZSpaceTrainerBridge(horizon, discount=discount)
        self._capacity = int(capacity)
        self._buffer: list[ZSpaceTrainerSample] = []
        self._memory_sum = 0.0
        self._stability_sum = 0.0
        self._drs_sum = 0.0

    @property
    def horizon(self) -> int:
        return self._bridge.horizon

    @property
    def discount(self) -> float:
        return self._bridge.discount

    @property
    def capacity(self) -> int:
        return self._capacity

    def push(
        self,
        step: RuntimeStep,
        vision: VisionFeedbackSnapshot | None = None,
        *,
        end_episode: bool,
    ) -> TrainerEpisode | None:
        if len(self._buffer) >= self._capacity:
            raise ValueError("episode capacity exceeded before flush")
        sample = self._bridge.push(step, vision)
        self._buffer.append(_clone_trainer_sample(sample))
        self._memory_sum += sample.metrics.memory
        self._stability_sum += sample.metrics.stability
        self._drs_sum += sample.metrics.drs
        if end_episode:
            return self._finish_episode()
        return None

    def flush(self) -> TrainerEpisode | None:
        if not self._buffer:
            return None
        return self._finish_episode()

    def _finish_episode(self) -> TrainerEpisode:
        length = len(self._buffer)
        samples = self._buffer
        self._buffer = []
        normaliser = float(length if length > 0 else 1)
        episode = TrainerEpisode(
            samples=samples,
            average_memory=self._memory_sum / normaliser,
            average_stability=self._stability_sum / normaliser,
            average_drs=self._drs_sum / normaliser,
            length=length,
        )
        self._memory_sum = 0.0
        self._stability_sum = 0.0
        self._drs_sum = 0.0
        return episode


def _clone_trainer_sample(sample: ZSpaceTrainerSample) -> ZSpaceTrainerSample:
    metrics = TrainerMetrics(
        speed=sample.metrics.speed,
        memory=sample.metrics.memory,
        stability=sample.metrics.stability,
        gradient=list(sample.metrics.gradient),
        drs=sample.metrics.drs,
    )
    partial = ZSpacePartialObservation(
        metrics=dict(sample.partial.metrics),
        commands=dict(sample.partial.commands),
        gradient=list(sample.partial.gradient),
        weight=sample.partial.weight,
    )
    return ZSpaceTrainerSample(metrics=metrics, partial=partial)


def _validate_metric(components: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    matrix = []
    for row in components:
        vector = tuple(float(value) for value in row)
        matrix.append(vector)
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("metric tensor must be 4x4")
    return tuple(matrix)


def _time_dilation_from_metric(matrix: Sequence[Sequence[float]]) -> float:
    g_tt = float(matrix[0][0])
    shift = math.sqrt(sum(float(matrix[0][i]) ** 2 for i in range(1, 4)))
    base = math.sqrt(abs(g_tt)) if g_tt < 0.0 else 1.0
    return max(base / (1.0 + shift), 1e-6)


def relativity_geometry_from_metric(
    components: Sequence[Sequence[float]],
) -> ZSpaceGeometry:
    matrix = _validate_metric(components)
    spatial = tuple(
        tuple(matrix[i + 1][j + 1] for j in range(3))
        for i in range(3)
    )
    dilation = _time_dilation_from_metric(matrix)
    return ZSpaceGeometry.general_relativity(spatial, dilation)


def relativity_dynamics_from_metric(
    components: Sequence[Sequence[float]],
    gravity: GravityField | None = None,
) -> ZSpaceDynamics:
    geometry = relativity_geometry_from_metric(components)
    return ZSpaceDynamics(geometry=geometry, gravity=gravity)


def _seed_metric_from_ansatz(ansatz: str) -> tuple[tuple[float, ...], ...]:
    kind = ansatz.strip().lower()
    if kind == "static_spherical":
        return (
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    if kind == "homogeneous_isotropic":
        return (
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    return (
        (-1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def relativity_dynamics_from_ansatz(
    ansatz: str,
    *,
    scale: float = 1.0,
    gravity: GravityField | None = None,
) -> ZSpaceDynamics:
    seed = _seed_metric_from_ansatz(ansatz)
    scaled = tuple(
        tuple(value * float(scale) for value in row)
        for row in seed
    )
    return relativity_dynamics_from_metric(scaled, gravity)


__all__ = [
    "GravityField",
    "GravityWell",
    "ZSpaceDynamics",
    "ZSpaceGeometry",
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
    "VisionFeedbackSnapshot",
    "VisionFeedbackSynchronizer",
    "ZSpacePartialObservation",
    "TemporalFeedbackLearner",
    "TemporalFeedbackSummary",
    "TrainerMetrics",
    "TrainerEpisode",
    "ZSpaceTrainerBridge",
    "ZSpaceTrainerEpisodeBuilder",
    "ZSpaceTrainerSample",
    "relativity_geometry_from_metric",
    "relativity_dynamics_from_metric",
    "relativity_dynamics_from_ansatz",
]
