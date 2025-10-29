"""Pure-Python fallbacks mirroring the quantum overlay bindings."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

__all__ = [
    "QuantumOverlayConfig",
    "QuantumMeasurement",
    "QuantumRealityStudio",
    "ZOverlayCircuit",
    "ZResonance",
]


def _clamp_unit(value: float, *, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):  # noqa: BLE001 - user-facing API surface
        return default
    return min(max(numeric, 0.0), 1.0)


def _float_or_default(value: float, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):  # noqa: BLE001 - user-facing API surface
        return default


def _extract_field(container: object, name: str, default: float = 0.0) -> float:
    if isinstance(container, Mapping):
        return _float_or_default(container.get(name, default), default=default)
    return _float_or_default(getattr(container, name, default), default=default)


def _extract_band_energy(container: object) -> Sequence[float]:
    if isinstance(container, Mapping):
        candidate = container.get("band_energy")
    else:
        candidate = getattr(container, "band_energy", None)
    if isinstance(candidate, Sequence) and len(candidate) >= 3:
        return candidate
    return (0.0, 0.0, 0.0)


@dataclass
class QuantumOverlayConfig:
    """Configuration snapshot for quantum overlay synthesis."""

    curvature: float = -1.0
    qubits: int = 24
    packing_bias: float = 0.35
    leech_shells: int = 24

    def __post_init__(self) -> None:  # pragma: no cover - pure normalisation
        self._normalise()

    def _normalise(self) -> None:
        curvature = _float_or_default(self.curvature, default=-1.0)
        if not math.isfinite(curvature) or curvature >= 0.0:
            curvature = -1.0
        self.curvature = curvature
        self.qubits = max(int(self.qubits), 1)
        self.packing_bias = _clamp_unit(self.packing_bias, default=0.35)
        self.leech_shells = max(int(self.leech_shells), 1)

    def update(
        self,
        *,
        curvature: float | None = None,
        qubits: int | None = None,
        packing_bias: float | None = None,
        leech_shells: int | None = None,
    ) -> None:
        if curvature is not None:
            self.curvature = curvature
        if qubits is not None:
            self.qubits = qubits
        if packing_bias is not None:
            self.packing_bias = packing_bias
        if leech_shells is not None:
            self.leech_shells = leech_shells
        self._normalise()


@dataclass
class ZResonance:
    """Simplified resonance snapshot used for overlay synthesis."""

    spectrum: list[float] = field(default_factory=list)
    eta_hint: float = 0.0
    shell_weights: list[float] = field(default_factory=list)

    @classmethod
    def from_pulses(cls, pulses: Iterable[object]) -> ZResonance:
        spectrum: list[float] = []
        shells: list[float] = []
        eta_acc = 0.0
        count = 0
        for pulse in pulses:
            mean = _extract_field(pulse, "mean")
            band_energy = _extract_band_energy(pulse)
            shell_energy = sum(_float_or_default(value) for value in band_energy[:3])
            if not math.isfinite(shell_energy) or shell_energy <= 0.0:
                shell_energy = abs(mean)
            spectrum.append(mean)
            shells.append(abs(shell_energy))
            eta_acc += abs(_extract_field(pulse, "z_bias"))
            count += 1
        eta_hint = 0.0 if count == 0 else min(eta_acc / count, 2.0)
        return cls(spectrum=spectrum, eta_hint=eta_hint, shell_weights=shells)

    @classmethod
    def from_spectrum(
        cls,
        spectrum: Iterable[float],
        eta_hint: float = 0.0,
    ) -> ZResonance:
        values = [_float_or_default(value) for value in spectrum]
        shells = [abs(value) * (idx + 1) for idx, value in enumerate(values)]
        return cls(spectrum=values, eta_hint=max(eta_hint, 0.0), shell_weights=shells)


@dataclass
class QuantumMeasurement:
    """Collapsed measurement emitted by :class:`ZOverlayCircuit`."""

    active_qubits: list[int]
    eta_bar: float
    policy_logits: list[float]
    packing_pressure: float


class ZOverlayCircuit:
    """Hyperbolic overlay synthesised from a resonance snapshot."""

    def __init__(self, config: QuantumOverlayConfig, resonance: ZResonance) -> None:
        self._config = QuantumOverlayConfig(
            curvature=config.curvature,
            qubits=config.qubits,
            packing_bias=config.packing_bias,
            leech_shells=config.leech_shells,
        )
        shells = self._normalize_shells(resonance.shell_weights, self._config.qubits)
        curvature = abs(self._config.curvature)
        self._weights: list[float] = []
        leech_period = max(self._config.leech_shells, 1)
        for idx in range(self._config.qubits):
            shell = shells[idx % len(shells)]
            hyper = math.tanh(shell * curvature)
            leech_phase = math.sin(2.0 * math.pi * (idx / leech_period))
            weight = (
                hyper * (1.0 - self._config.packing_bias)
                + abs(leech_phase) * self._config.packing_bias
            )
            self._weights.append(weight)
        packing_pressure = sum(shells) / len(shells)
        self._eta = abs(math.tanh(resonance.eta_hint + packing_pressure))
        self._packing = float(packing_pressure)

    @staticmethod
    def _normalize_shells(shells: Sequence[float], qubits: int) -> list[float]:
        if not shells:
            return [1.0 / (idx + 1) for idx in range(max(qubits, 1))]
        weights = [abs(value) for value in shells]
        while len(weights) < qubits:
            weights.append(weights[len(weights) % len(weights)])
        total = sum(weights) or 1.0
        return [value / total for value in weights]

    def weights(self) -> list[float]:
        return list(self._weights)

    def eta_bar(self) -> float:
        return float(self._eta)

    def packing_pressure(self) -> float:
        return float(self._packing)

    def measure(self, threshold: float) -> QuantumMeasurement:
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):  # noqa: BLE001 - user-facing API surface
            threshold = 0.0
        indexed = sorted(
            enumerate(self._weights),
            key=lambda item: item[1],
            reverse=True,
        )
        active: list[int] = []
        logits: list[float] = []
        for index, weight in indexed:
            if weight >= threshold or not active:
                active.append(index)
            logits.append(float(weight))
        return QuantumMeasurement(
            active_qubits=active,
            eta_bar=self._eta,
            policy_logits=logits,
            packing_pressure=self._packing,
        )


class QuantumRealityStudio:
    """Minimal Python mirror of the Rust quantum overlay studio."""

    def __init__(
        self,
        curvature: float = -1.0,
        qubits: int = 24,
        packing_bias: float = 0.35,
        leech_shells: int = 24,
    ) -> None:
        self._config = QuantumOverlayConfig(
            curvature=curvature,
            qubits=qubits,
            packing_bias=packing_bias,
            leech_shells=leech_shells,
        )

    @property
    def config(self) -> QuantumOverlayConfig:
        return self._config

    def configure(
        self,
        *,
        curvature: float | None = None,
        qubits: int | None = None,
        packing_bias: float | None = None,
        leech_shells: int | None = None,
    ) -> None:
        self._config.update(
            curvature=curvature,
            qubits=qubits,
            packing_bias=packing_bias,
            leech_shells=leech_shells,
        )

    def overlay_zspace(self, resonance: ZResonance) -> ZOverlayCircuit:
        return ZOverlayCircuit(self._config, resonance)

    def overlay(self, resonance: ZResonance) -> ZOverlayCircuit:
        return self.overlay_zspace(resonance)

    def record_quantum_policy(
        self,
        pulses: Iterable[object],
        *,
        threshold: float = 0.0,
    ) -> QuantumMeasurement:
        resonance = ZResonance.from_pulses(pulses)
        circuit = self.overlay_zspace(resonance)
        return circuit.measure(threshold)
