# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
# Unauthorized derivative works or closed redistribution prohibited under AGPL §13.
"""Text → optim → zspace quickstart for SpiralTorch wheels.

Run from the repository root after installing the wheel (or `maturin develop`):

  python bindings/st-py/examples/text_optim_zspace_quickstart.py

This script demonstrates the "unique stack" loop:

* `LanguageWaveEncoder` turns text into a Z-space tensor.
* `spiraltorch.optim.Amegagrad` absorbs the text (padding/truncation included) and updates `weights`.
* `ZSpaceTrainer` consumes gradient-derived metrics for a meta-control signal.
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
    if not hasattr(st, "LanguageWaveEncoder"):
        raise RuntimeError("LanguageWaveEncoder is unavailable in this build.")

    curvature = -1.0
    encoder = st.LanguageWaveEncoder(curvature, 0.5)
    probe = encoder.encode_z_space("SpiralTorch")
    rows, cols = probe.shape()

    opt = st.optim.Amegagrad(
        (rows, cols),
        curvature=float(encoder.curvature()),
        hyper_learning_rate=0.03,
        real_learning_rate=0.02,
        gain=1.0,
    )
    weights = st.Tensor(rows, cols, [0.0] * (rows * cols))
    ztrainer = st.ZSpaceTrainer(z_dim=4, lr=0.05, lam_frac=0.05)

    texts = [
        "SpiralTorch: Rust-first learning, Python-friendly bindings.",
        "Z-space as a steerable metric stack.",
        "Hypergrad + Realgrad tapes as optimisers.",
    ]

    print("== SpiralTorch quickstart: text → optim → zspace ==")
    print("encoder curvature:", encoder.curvature(), "shape:", (rows, cols))
    print("initial weights (head):", _flatten(weights.tolist())[:5])

    for step in range(int(steps)):
        text = texts[step % len(texts)]
        opt.zero_grad()
        opt.absorb_text(encoder, text)

        summary = opt.real.summary()
        metrics = {
            "speed": summary.mean_abs(),
            "memory": summary.l2(),
            "stability": 1.0 / (1.0 + summary.l2()),
            "gradient": opt.real.gradient(),
        }
        z_loss = ztrainer.step(metrics)

        control = opt.desire_control()
        before = _flatten(weights.tolist())
        opt.step(weights, tune=True, control=control)
        after = _flatten(weights.tolist())

        print(
            f"step {step:02d} | z_loss={z_loss:.5f} | "
            f"hyper_scale={float(control.hyper_rate_scale()):.3f} "
            f"real_scale={float(control.real_rate_scale()):.3f} | "
            f"Δw₀={after[0] - before[0]:+.6f}"
        )


if __name__ == "__main__":
    main()
