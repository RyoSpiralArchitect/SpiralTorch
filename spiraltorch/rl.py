"""Simplified reinforcement-learning helpers mirroring SpiralTorch APIs."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional


@dataclass
class LossStdTrigger:
    std_threshold: float = 0.1
    decay: float = 0.8
    max_ratio: float = 3.0
    warmup: int = 4
    geometry_eta: float = 0.0
    geometry_curvature: float = -1.0

    _ema: float = 0.0
    _seen: int = 0

    def observe(self, value: float) -> Optional[float]:
        if not math.isfinite(value):
            return None
        self._seen += 1
        if self._seen == 1:
            self._ema = value
        else:
            self._ema = self.decay * self._ema + (1.0 - self.decay) * value
        if self._seen <= self.warmup:
            return None
        if self._ema <= self.std_threshold:
            return None
        ratio = (self._ema / self.std_threshold) - 1.0
        if self.geometry_eta > 0.0:
            ratio *= 1.0 + self.geometry_eta * 0.5 + abs(math.tanh(self.geometry_curvature))
        return max(0.0, min(ratio, self.max_ratio))


class PolicyGradient:
    def __init__(self) -> None:
        self._hyper_trigger: LossStdTrigger | None = None
        self._geometry_feedback: dict[str, float] | None = None

    def attach_hyper_surprise(self, trigger: LossStdTrigger) -> None:
        self._hyper_trigger = trigger

    def attach_geometry_feedback(self, feedback: dict[str, float]) -> None:
        self._geometry_feedback = dict(feedback)

    def step(self, returns: Iterable[float], baseline: float = 0.0) -> dict[str, float]:
        returns = [float(value) for value in returns]
        if not returns:
            return {"learning_rate": 1.0, "gauge": 1.0}
        variance = sum((value - baseline) ** 2 for value in returns) / len(returns)
        std = math.sqrt(max(variance, 0.0))
        ratio = None
        if self._hyper_trigger is not None:
            ratio = self._hyper_trigger.observe(std)
        gauge = 1.0 + (ratio or 0.0)
        learning_rate = 1.0 + (ratio or 0.0)
        if self._geometry_feedback:
            learning_rate *= self._geometry_feedback.get("min_learning_rate_scale", 1.0)
            gauge *= self._geometry_feedback.get("max_learning_rate_scale", 1.0)
        return {"learning_rate": learning_rate, "gauge": gauge}
