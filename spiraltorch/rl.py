"""Simplified reinforcement-learning helpers mirroring SpiralTorch APIs."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
import math
from typing import Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .qr import QuantumMeasurement
    from .qr import QuantumRealityStudio
    from .vision import InfiniteZPatch


def _resolve_quantum_fractal_bridge():
    candidates: list[str] = []
    if "." in __name__:
        base = __name__.rsplit(".", 1)[0]
        candidates.append(f"{base}.qr")
    if __name__.endswith("rl"):
        candidates.append(f"{__name__[:-2]}qr")
    candidates.append("spiraltorch.qr")
    for name in candidates:
        module = sys.modules.get(name)
        if module is None:
            try:
                module = importlib.import_module(name)
            except Exception:  # noqa: BLE001 - optional import path
                continue
        bridge = getattr(module, "quantum_measurement_from_fractal", None)
        if bridge is not None:
            return bridge
    raise ImportError("quantum_measurement_from_fractal bridge is unavailable")


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
        self._last_quantum: dict[str, float] | None = None

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

    def update_from_quantum(
        self,
        measurement: "QuantumMeasurement",
        *,
        base_rate: float = 1.0,
        returns: Iterable[float] | None = None,
        baseline: float = 0.0,
    ) -> dict[str, float]:
        eta = float(getattr(measurement, "eta_bar", 0.0))
        packing = float(getattr(measurement, "packing_pressure", 0.0))
        activation_density = float(
            getattr(measurement, "activation_density", lambda: 0.0)()
            if hasattr(measurement, "activation_density")
            else 0.0
        )
        geometry_feedback = {
            "min_learning_rate_scale": 1.0 + max(activation_density, 0.0),
            "max_learning_rate_scale": 1.0 + max(eta, 0.0),
        }
        self._geometry_feedback = geometry_feedback
        if self._hyper_trigger is not None:
            self._hyper_trigger.geometry_eta = eta
            self._hyper_trigger.geometry_curvature = -abs(packing) if packing != 0.0 else -1.0
        update: dict[str, float]
        if hasattr(measurement, "to_policy_update"):
            raw_update = measurement.to_policy_update(base_rate=base_rate)
            update = {key: float(value) for key, value in raw_update.items()}
        else:
            novelty = abs(eta) + abs(packing) * 0.5
            update = {
                "learning_rate": max(float(base_rate), 0.0) + novelty,
                "gauge": max(float(base_rate), 0.0) + activation_density,
                "eta_bar": eta,
                "packing_pressure": packing,
                "activation_density": activation_density,
            }
        if returns is not None:
            rl_update = self.step(returns, baseline=baseline)
            update["learning_rate"] *= rl_update.get("learning_rate", 1.0)
            update["gauge"] *= rl_update.get("gauge", 1.0)
        self._last_quantum = dict(update)
        return update

    @property
    def last_quantum_update(self) -> dict[str, float] | None:
        if self._last_quantum is None:
            return None
        return dict(self._last_quantum)


def update_policy_from_fractal(
    policy: "PolicyGradient",
    studio: "QuantumRealityStudio",
    patch: "InfiniteZPatch",
    *,
    base_rate: float = 1.0,
    threshold: float = 0.0,
    eta_scale: float = 1.0,
    returns: Iterable[float] | None = None,
    baseline: float = 0.0,
) -> dict[str, float]:
    """Route a fractal patch through the quantum studio and update the policy."""

    measurement_fn = _resolve_quantum_fractal_bridge()
    measurement = measurement_fn(
        studio,
        patch,
        threshold=threshold,
        eta_scale=eta_scale,
    )
    return policy.update_from_quantum(
        measurement,
        base_rate=base_rate,
        returns=returns,
        baseline=baseline,
    )
