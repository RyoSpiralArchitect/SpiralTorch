"""Python-side helpers for the Julia Z-space bridge."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, List


@dataclass
class ZTigerOptim:
    curvature: float = -1.0
    gain: float = 1.0
    history: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not math.isfinite(self.curvature) or self.curvature >= 0.0:
            self.curvature = -1.0
        self.gain = float(max(self.gain, 1.0))

    def update(self, lora_pid: float, resonance: Iterable[float]) -> float:
        pid = max(float(lora_pid), 1e-6)
        samples = [abs(float(value)) for value in resonance]
        if not samples:
            samples = [0.0]
        weighted = [value * (idx + 1) for idx, value in enumerate(samples)]
        mean = sum(weighted) / len(weighted)
        self.history.append(mean)
        if len(self.history) > 32:
            self.history.pop(0)
        smooth = sum(self.history) / len(self.history)
        self.gain = abs(math.tanh(smooth * abs(math.tanh(self.curvature)) / pid)) + 1.0
        return self.gain
