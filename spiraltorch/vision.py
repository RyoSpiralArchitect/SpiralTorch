"""Fractal Z-space helpers for SpiralTorch Python notebooks."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence


@dataclass
class InfiniteZPatch:
    dimension: float
    zoom: float
    support: tuple[float, float]
    density: List[float]

    def eta_bar(self) -> float:
        if not self.density:
            return 0.0
        mean = sum(self.density) / len(self.density)
        return abs(math.tanh(mean))

    def density_mean(self) -> float:
        if not self.density:
            return 0.0
        return sum(self.density) / len(self.density)

    def spectral_compactness(self) -> float:
        if not self.density:
            return 0.0
        mean = self.density_mean()
        variance = sum((value - mean) ** 2 for value in self.density) / len(self.density)
        return 1.0 / (1.0 + variance)

    def harmonic_resonance(self) -> float:
        if not self.density:
            return 0.0
        return sum(abs(math.sin(value * math.pi)) for value in self.density) / len(
            self.density
        )

    def drift_index(self) -> float:
        if len(self.density) < 2:
            return 0.0
        head = self.density[0]
        tail = self.density[-1]
        return min(1.0, abs(head - tail))


class FractalCanvas:
    def __init__(self, dim: float = 2.0) -> None:
        self.dimension = float(dim)

    def emit_infinite_z(self, zoom: float = math.inf, steps: int = 64) -> InfiniteZPatch:
        steps = max(int(steps), 4)
        zoom = float(zoom)
        if not math.isfinite(zoom) or zoom <= 0.0:
            zoom = 1024.0
        log_start = -math.log(zoom)
        log_step = (math.log(zoom) + 1.0) / steps
        weights: List[float] = []
        density: List[float] = []
        for idx in range(steps):
            t = log_start + log_step * idx
            radius = math.sqrt(abs(self.dimension) + abs(t))
            phase = math.tanh(self.dimension * t)
            weight = abs(math.sin(phase)) + 1e-6
            weights.append(weight)
            density.append(radius * weight)
        total = sum(weights) or 1.0
        density = [value / total for value in density]
        support = (math.exp(log_start), math.exp(log_start + log_step * (steps - 1)))
        return InfiniteZPatch(
            dimension=self.dimension,
            zoom=zoom,
            support=support,
            density=density,
        )

    def emit_zspace_infinite(self, dim: float | None = None) -> InfiniteZPatch:
        if dim is not None:
            self.dimension = float(dim)
        return self.emit_infinite_z()

    def emit_resonant_series(
        self,
        depth: int = 3,
        *,
        base_zoom: float | None = None,
        steps: int = 48,
    ) -> list[InfiniteZPatch]:
        depth = max(int(depth), 1)
        series: list[InfiniteZPatch] = []
        baseline = float(base_zoom) if base_zoom is not None else (2.0 + abs(self.dimension))
        for index in range(depth):
            zoom = baseline ** (1.0 + index * 0.5)
            patch = self.emit_infinite_z(zoom=zoom, steps=steps + index * 8)
            series.append(patch)
        return series

    def synthesize_signature(
        self, patches: Sequence[InfiniteZPatch] | Iterable[InfiniteZPatch]
    ) -> "FractalSignature":
        collection = list(patches)
        if not collection:
            collection.append(self.emit_infinite_z())
        eta_values = [patch.eta_bar() for patch in collection]
        density_means = [patch.density_mean() for patch in collection]
        compactness = [patch.spectral_compactness() for patch in collection]
        resonance = [patch.harmonic_resonance() for patch in collection]
        drift = [patch.drift_index() for patch in collection]
        eta_mean = sum(eta_values) / len(eta_values)
        density_mean = sum(density_means) / len(density_means)
        compactness_mean = sum(compactness) / len(compactness)
        resonance_mean = sum(resonance) / len(resonance)
        drift_mean = sum(drift) / len(drift)
        return FractalSignature(
            eta_mean=eta_mean,
            density_mean=density_mean,
            compactness=compactness_mean,
            resonance=resonance_mean,
            drift_index=drift_mean,
        )


@dataclass
class FractalSignature:
    eta_mean: float
    density_mean: float
    compactness: float
    resonance: float
    drift_index: float

    def harmony(self) -> float:
        drift_factor = max(0.0, 1.0 - abs(self.drift_index))
        return max(0.0, min(1.0, 0.5 * self.compactness + 0.5 * drift_factor))

    def adaptivity(self) -> float:
        return max(0.0, min(1.0, 0.6 * self.resonance + 0.4 * self.compactness))

    def to_summary(self, prefix: str = "fractal") -> dict[str, float]:
        return {
            f"{prefix}_eta_mean": self.eta_mean,
            f"{prefix}_density_mean": self.density_mean,
            f"{prefix}_compactness": self.compactness,
            f"{prefix}_resonance": self.resonance,
            f"{prefix}_drift_index": self.drift_index,
        }
