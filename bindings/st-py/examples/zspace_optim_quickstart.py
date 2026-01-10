# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
# Unauthorized derivative works or closed redistribution prohibited under AGPL §13.
"""Minimal "Z-space + optim" loop for SpiralTorch Python wheels.

Run from the repository root after installing the wheel (or `maturin develop`):

  python bindings/st-py/examples/zspace_optim_quickstart.py

The script demonstrates:

* `spiraltorch.optim.Amegagrad` (Hypergrad + Realgrad) as a step-based optimiser.
* `spiraltorch.ZSpaceTrainer` consuming gradient-derived metrics.
"""

from __future__ import annotations

from typing import Iterable

import spiraltorch as st


def _flatten(matrix: Iterable[Iterable[float]]) -> list[float]:
    return [float(value) for row in matrix for value in row]


def main(steps: int = 6) -> None:
    if not hasattr(st, "optim") or not hasattr(st.optim, "Amegagrad"):
        raise RuntimeError(
            "spiraltorch.optim requires the compiled SpiralTorch extension. "
            "Build it via `maturin develop -m bindings/st-py/Cargo.toml` or install a wheel."
        )

    opt = st.optim.Amegagrad(
        (1, 4),
        curvature=-0.9,
        hyper_learning_rate=0.03,
        real_learning_rate=0.02,
        gain=1.0,
    )
    weights = st.Tensor(1, 4, [0.25, -0.1, 0.05, 0.3])
    target = [0.4, -0.35, 0.2, 0.1]

    ztrainer = st.ZSpaceTrainer(z_dim=4, lr=0.05, lam_frac=0.05)

    print("== SpiralTorch quickstart: zspace + optim ==")
    print("initial weights:", _flatten(weights.tolist()))

    for step in range(int(steps)):
        opt.zero_grad()

        current = _flatten(weights.tolist())
        signal = [t - c for t, c in zip(target, current)]
        opt.accumulate_wave(st.Tensor(1, 4, signal))

        summary = opt.real.summary()
        metrics = {
            "speed": summary.mean_abs(),
            "memory": summary.l2(),
            "stability": 1.0 / (1.0 + summary.l2()),
            "gradient": opt.real.gradient(),
        }
        z_loss = ztrainer.step(metrics)

        control = opt.desire_control()
        opt.step(weights, tune=True, control=control)

        print(
            f"step {step:02d} | z_loss={z_loss:.5f} | "
            f"hyper_scale={float(control.hyper_rate_scale()):.3f} "
            f"real_scale={float(control.real_rate_scale()):.3f} | "
            f"weights={_flatten(weights.tolist())}"
        )


if __name__ == "__main__":
    main()

