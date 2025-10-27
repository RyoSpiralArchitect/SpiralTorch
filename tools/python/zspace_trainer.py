# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
# Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

"""Pure Python meta-optimizer for Z-space training metrics.

The helper mirrors the design sketched in the user request but avoids NumPy
and PyTorch so it can run in constrained environments.  A tiny Adam
implementation keeps the Z vector stable while a fractional FFT regulariser
penalises high-frequency drift.  The trainer also accepts Drift-Response
Linguistics (DRL) penalties so language batches can react to high-risk
vocabulary without rewriting the Rust stack.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass
class ZMetrics:
    speed: float
    memory: float
    stability: float
    gradient: Optional[Sequence[float]] = None
    drs: float = 0.0
    telemetry: Optional[Dict[str, float]] = None


class ZSpaceTrainer:
    def __init__(
        self,
        z_dim: int = 4,
        alpha: float = 0.35,
        lam_speed: float = 0.5,
        lam_mem: float = 0.3,
        lam_stab: float = 0.2,
        lam_frac: float = 0.1,
        lam_drs: float = 0.0,
        lr: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        if z_dim <= 0:
            raise ValueError("z_dim must be positive")
        self.z: List[float] = [0.0] * z_dim
        self._alpha = max(1e-6, alpha)
        self._lam = (lam_speed, lam_mem, lam_stab, lam_frac, lam_drs)
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._m: List[float] = [0.0] * z_dim
        self._v: List[float] = [0.0] * z_dim
        self._t = 0

    @property
    def state(self) -> List[float]:
        return list(self.z)

    def _rfft(self, values: Sequence[float]) -> List[complex]:
        n = len(values)
        freq: List[complex] = []
        for k in range(n // 2 + 1):
            total = 0.0j
            for t, val in enumerate(values):
                angle = -2.0 * math.pi * k * t / n
                total += complex(val, 0.0) * cmath.exp(1j * angle)
            freq.append(total)
        return freq

    def _frac_reg(self, values: Sequence[float]) -> float:
        spectrum = self._rfft(values)
        n = len(spectrum)
        if n <= 1:
            return 0.0
        acc = 0.0
        for idx, coeff in enumerate(spectrum):
            omega = idx / max(1, n - 1)
            weight = omega ** (2.0 * self._alpha)
            acc += weight * abs(coeff) ** 2
        return acc / n

    def _frac_grad(self) -> List[float]:
        grad: List[float] = []
        base = self._frac_reg(self.z)
        step = 1e-4
        for i in range(len(self.z)):
            original = self.z[i]
            self.z[i] = original + step
            plus = self._frac_reg(self.z)
            self.z[i] = original - step
            minus = self._frac_reg(self.z)
            self.z[i] = original
            grad.append((plus - minus) / (2.0 * step))
        # ensure numerical noise does not explode
        scale = max(1.0, max(abs(g) for g in grad))
        return [g / scale for g in grad]

    def _normalise(self, value: float) -> float:
        return math.tanh(value)

    def _normalise_gradient(self, grad: Sequence[float]) -> List[float]:
        if not grad:
            return [0.0] * len(self.z)
        if len(grad) == len(self.z):
            return [self._normalise(g) for g in grad]
        # Tile or truncate to match dimensionality.
        out: List[float] = []
        for idx in range(len(self.z)):
            out.append(self._normalise(grad[idx % len(grad)]))
        return out

    def step(self, metrics: Dict[str, float] | ZMetrics) -> float:
        if isinstance(metrics, ZMetrics):
            speed = metrics.speed
            memory = metrics.memory
            stability = metrics.stability
            gradient = metrics.gradient
        else:
            speed = float(metrics.get("speed", 0.0))
            memory = float(metrics.get("mem", metrics.get("memory", 0.0)))
            stability = float(metrics.get("stab", metrics.get("stability", 0.0)))
            grad = metrics.get("gradient")
            gradient = grad if isinstance(grad, Sequence) else None
        lam_speed, lam_mem, lam_stab, lam_frac, lam_drs = self._lam

        penalty = (
            lam_speed * self._normalise(speed)
            + lam_mem * self._normalise(memory)
            + lam_stab * self._normalise(stability)
        )
        drs_signal = metrics.drs if isinstance(metrics, ZMetrics) else float(metrics.get("drs", 0.0))
        if lam_drs:
            penalty += lam_drs * self._normalise(drs_signal)
        frac_reg = self._frac_reg(self.z)
        loss = penalty + lam_frac * frac_reg

        grad_metric = self._normalise_gradient(gradient or [])
        frac_grad = self._frac_grad()
        grad_total = []
        for idx in range(len(self.z)):
            total = grad_metric[idx] if idx < len(grad_metric) else grad_metric[idx % len(grad_metric)]
            total += lam_frac * frac_grad[idx]
            grad_total.append(total)

        self._adam_update(grad_total)
        return loss

    def _adam_update(self, grad: Sequence[float]) -> None:
        self._t += 1
        lr = self._lr
        beta1 = self._beta1
        beta2 = self._beta2
        eps = self._eps
        for i, g in enumerate(grad):
            self._m[i] = beta1 * self._m[i] + (1.0 - beta1) * g
            self._v[i] = beta2 * self._v[i] + (1.0 - beta2) * (g * g)
            m_hat = self._m[i] / (1.0 - beta1 ** self._t)
            v_hat = self._v[i] / (1.0 - beta2 ** self._t)
            self.z[i] -= lr * m_hat / (math.sqrt(v_hat) + eps)


def step_many(trainer: ZSpaceTrainer, samples: Iterable[Dict[str, float]]) -> List[float]:
    for metrics in samples:
        trainer.step(metrics)
    return trainer.state


if __name__ == "__main__":  # pragma: no cover - smoke example
    trainer = ZSpaceTrainer(z_dim=4)
    history = [
        {"speed": 0.8, "mem": 0.5, "stab": 0.6, "gradient": [0.1, -0.2, 0.3, -0.1]},
        {"speed": 0.6, "mem": 0.4, "stab": 0.7, "gradient": [0.05, -0.05, 0.1, 0.0]},
    ]
    final_state = step_many(trainer, history)
    print("Final Z state:", [round(v, 4) for v in final_state])
