"""Quantum overlay helpers for SpiralTorch Python workflows."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Sequence


@dataclass
class QuantumOverlayConfig:
    """Configuration for hyperbolic quantum overlays."""

    curvature: float = -1.0
    qubits: int = 24
    packing_bias: float = 0.35

    def __post_init__(self) -> None:
        if not math.isfinite(self.curvature) or self.curvature >= 0.0:
            self.curvature = -1.0
        self.qubits = max(int(self.qubits), 1)
        self.packing_bias = float(min(max(self.packing_bias, 0.0), 1.0))


@dataclass
class ZResonance:
    """Simplified resonance snapshot used for overlay synthesis."""

    spectrum: list[float] = field(default_factory=list)
    eta_hint: float = 0.0
    shell_weights: list[float] = field(default_factory=list)

    @classmethod
    def from_pulses(cls, pulses: Iterable[dict[str, float]]) -> "ZResonance":
        spectrum: list[float] = []
        shell: list[float] = []
        eta_acc = 0.0
        count = 0
        for pulse in pulses:
            mean = float(pulse.get("mean", 0.0))
            energies = pulse.get("band_energy")
            if isinstance(energies, Sequence) and len(energies) >= 3:
                shell_energy = float(energies[0]) + float(energies[1]) + float(energies[2])
            else:
                shell_energy = abs(mean)
            spectrum.append(mean)
            shell.append(max(shell_energy, 0.0))
            eta_acc += abs(float(pulse.get("z_bias", 0.0)))
            count += 1
        eta_hint = 0.0 if count == 0 else min(eta_acc / count, 2.0)
        return cls(spectrum=spectrum, eta_hint=eta_hint, shell_weights=shell)

    @classmethod
    def from_spectrum(cls, spectrum: Iterable[float], eta_hint: float = 0.0) -> "ZResonance":
        values = [float(value) for value in spectrum]
        shell = [abs(value) * (idx + 1) for idx, value in enumerate(values)]
        return cls(spectrum=values, eta_hint=max(eta_hint, 0.0), shell_weights=shell)


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
        self._config = config
        shells = self._normalize_shells(resonance.shell_weights, config.qubits)
        curvature = abs(config.curvature)
        self._weights: list[float] = []
        for idx in range(config.qubits):
            shell = shells[idx % len(shells)]
            hyper = math.tanh(shell * curvature)
            leech_phase = math.sin(2.0 * math.pi * (idx / max(config.qubits, 1)))
            weight = hyper * (1.0 - config.packing_bias) + abs(leech_phase) * config.packing_bias
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
            weights.append(weights[len(weights) % len(shells)])
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
        except (TypeError, ValueError):  # noqa: BLE001 - user facing API
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

    def __init__(self, curvature: float = -1.0, qubits: int = 24) -> None:
        self._config = QuantumOverlayConfig(curvature=curvature, qubits=qubits)

    def configure(self, **kwargs: float) -> None:
        if "curvature" in kwargs:
            self._config.curvature = float(kwargs["curvature"])
            if self._config.curvature >= 0.0:
                self._config.curvature = -abs(self._config.curvature or 1.0)
        if "qubits" in kwargs:
            self._config.qubits = max(int(kwargs["qubits"]), 1)
        if "packing_bias" in kwargs:
            self._config.packing_bias = min(max(float(kwargs["packing_bias"]), 0.0), 1.0)

    def overlay_zspace(self, resonance: ZResonance) -> ZOverlayCircuit:
        return ZOverlayCircuit(self._config, resonance)

    def overlay(self, resonance: ZResonance) -> ZOverlayCircuit:
        return self.overlay_zspace(resonance)

    def record_quantum_policy(self, pulses: Iterable[dict[str, float]]) -> QuantumMeasurement:
        resonance = ZResonance.from_pulses(pulses)
        circuit = self.overlay_zspace(resonance)
        return circuit.measure(0.0)
