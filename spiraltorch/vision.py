"""Fractal Z-space helpers for SpiralTorch Python notebooks."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List


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
