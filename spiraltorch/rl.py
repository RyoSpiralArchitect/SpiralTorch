"""Simplified reinforcement-learning helpers mirroring SpiralTorch APIs."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import importlib
import sys
import math
from typing import Deque, Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .qr import QuantumMeasurement
    from .qr import QuantumRealityStudio
    from .qr import ZResonance
    from .vision import InfiniteZPatch


def _resolve_quantum_module():
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
        return module
    raise ImportError("quantum overlay module is unavailable")


def _resolve_quantum_fractal_bridge():
    module = _resolve_quantum_module()
    bridge = getattr(module, "quantum_measurement_from_fractal", None)
    if bridge is None:
        raise ImportError("quantum_measurement_from_fractal bridge is unavailable")
    return bridge


def _resolve_quantum_fractal_sequence_bridge():
    module = _resolve_quantum_module()
    bridge = getattr(module, "quantum_measurement_from_fractal_sequence", None)
    if bridge is None:
        raise ImportError("quantum_measurement_from_fractal_sequence bridge is unavailable")
    return bridge


def _resolve_fractal_session_cls():
    module = _resolve_quantum_module()
    session = getattr(module, "FractalQuantumSession", None)
    if session is None:
        raise ImportError("FractalQuantumSession is unavailable")
    return session


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

    def build_fractal_session(
        self,
        studio: "QuantumRealityStudio",
        *,
        threshold: float = 0.0,
        eta_scale: float = 1.0,
    ):
        session_cls = _resolve_fractal_session_cls()
        return session_cls(studio, threshold=threshold, eta_scale=eta_scale)

    def update_from_fractal_stream(
        self,
        studio: "QuantumRealityStudio",
        patches: Iterable["InfiniteZPatch"],
        *,
        weights: Iterable[float] | None = None,
        base_rate: float = 1.0,
        threshold: float = 0.0,
        eta_scale: float = 1.0,
        returns: Iterable[float] | None = None,
        baseline: float = 0.0,
    ) -> dict[str, float]:
        measurement_fn = _resolve_quantum_fractal_sequence_bridge()
        patch_seq = tuple(patches)
        weights_seq = None if weights is None else tuple(float(value) for value in weights)
        measurement = measurement_fn(
            studio,
            patch_seq,
            weights=weights_seq,
            threshold=threshold,
            eta_scale=eta_scale,
        )
        return self.update_from_quantum(
            measurement,
            base_rate=base_rate,
            returns=returns,
            baseline=baseline,
        )

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


def update_policy_from_fractal_stream(
    policy: "PolicyGradient",
    studio: "QuantumRealityStudio",
    patches: Iterable["InfiniteZPatch"],
    *,
    weights: Iterable[float] | None = None,
    base_rate: float = 1.0,
    threshold: float = 0.0,
    eta_scale: float = 1.0,
    returns: Iterable[float] | None = None,
    baseline: float = 0.0,
) -> dict[str, float]:
    """Aggregate fractal patches before updating the policy from quantum feedback."""

    return policy.update_from_fractal_stream(
        studio,
        patches,
        weights=weights,
        base_rate=base_rate,
        threshold=threshold,
        eta_scale=eta_scale,
        returns=returns,
        baseline=baseline,
    )


@dataclass
class FractalQuantumTrainer:
    """Coordinate fractal Z-patches, quantum overlays, and policy feedback."""

    studio: "QuantumRealityStudio"
    policy: PolicyGradient
    threshold: float = 0.0
    eta_scale: float = 1.0
    base_rate: float = 1.0
    window: int = 6

    _session: "FractalQuantumSession" = field(init=False, repr=False)
    _history: Deque[dict[str, float]] = field(init=False, repr=False)
    _returns: Deque[float] = field(init=False, repr=False)
    _last_measurement: "QuantumMeasurement | None" = field(
        default=None, init=False, repr=False
    )
    _golden_ratio: float = field(
        default=1.618033988749895, init=False, repr=False
    )
    _ingested: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        session_cls = _resolve_fractal_session_cls()
        self._session = session_cls(
            self.studio,
            threshold=self.threshold,
            eta_scale=self.eta_scale,
        )
        window = max(int(self.window), 1)
        self._history = deque(maxlen=window)
        self._returns = deque(maxlen=window * 4)
        self.base_rate = float(self.base_rate)
        self.threshold = float(self.threshold)
        self.eta_scale = float(self.eta_scale)

    @property
    def session(self) -> "FractalQuantumSession":
        return self._session

    @property
    def last_update(self) -> dict[str, float] | None:
        if not self._history:
            return None
        return dict(self._history[-1])

    @property
    def last_measurement(self) -> "QuantumMeasurement | None":
        return self._last_measurement

    def ingest_patch(self, patch: "InfiniteZPatch", *, weight: float = 1.0) -> "ZResonance":
        resonance = self._session.ingest(patch, weight=float(weight))
        self._ingested = self._session.ingested
        return resonance

    def accumulate_returns(self, values: Iterable[float]) -> None:
        for value in values:
            try:
                numeric = float(value)
            except (TypeError, ValueError):  # noqa: BLE001 - user data surface
                continue
            if math.isfinite(numeric):
                self._returns.append(numeric)

    def _compute_golden_feedback(self) -> dict[str, float]:
        if not self._history:
            return {}
        weight = 1.0
        accum: dict[str, float] = {}
        total = 0.0
        for update in reversed(self._history):
            for key, value in update.items():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):  # noqa: BLE001 - defensive cast
                    continue
                accum[key] = accum.get(key, 0.0) + numeric * weight
            total += weight
            weight *= self._golden_ratio
        if total <= 0.0:
            return {}
        return {key: value / total for key, value in accum.items()}

    @property
    def golden_feedback(self) -> dict[str, float]:
        return self._compute_golden_feedback()

    def peek_resonance(self) -> "ZResonance":
        return self._session.resonance()

    def flush(self, *, baseline: float = 0.0) -> dict[str, float]:
        measurement = self._session.measure(threshold=self.threshold)
        returns = list(self._returns)
        update = self.policy.update_from_quantum(
            measurement,
            base_rate=self.base_rate,
            returns=returns or None,
            baseline=baseline,
        )
        self._history.append(dict(update))
        self._last_measurement = measurement
        self._session.clear()
        self._returns.clear()
        self._ingested = 0
        return update

    def summary(self) -> dict[str, object]:
        return {
            "pending": self._ingested,
            "window": self._history.maxlen,
            "history": [dict(update) for update in self._history],
            "golden_feedback": self.golden_feedback,
            "returns": list(self._returns),
            "threshold": self.threshold,
            "eta_scale": self.eta_scale,
            "base_rate": self.base_rate,
            "last_measurement": self._last_measurement,
        }

    def reconfigure(
        self,
        *,
        threshold: float | None = None,
        eta_scale: float | None = None,
        base_rate: float | None = None,
        window: int | None = None,
    ) -> None:
        if threshold is not None:
            self.threshold = float(threshold)
        if eta_scale is not None:
            self.eta_scale = float(eta_scale)
        if base_rate is not None:
            self.base_rate = float(base_rate)
        if window is not None and window > 0:
            self.window = int(window)
            self._history = deque(self._history, maxlen=self.window)
            self._returns = deque(self._returns, maxlen=self.window * 4)
        session_cls = _resolve_fractal_session_cls()
        self._session = session_cls(
            self.studio,
            threshold=self.threshold,
            eta_scale=self.eta_scale,
        )
        self._ingested = 0
