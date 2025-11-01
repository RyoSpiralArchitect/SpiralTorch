"""High-level robotics orchestration utilities for SpiralTorch."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import time
from typing import Iterable, Mapping, MutableMapping, Sequence


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
        return TelemetryReport(
            energy=float(energy.total),
            stability=stability,
            failsafe=failsafe,
            anomalies=tuple(anomalies),
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
    "relativity_geometry_from_metric",
    "relativity_dynamics_from_metric",
    "relativity_dynamics_from_ansatz",
]
