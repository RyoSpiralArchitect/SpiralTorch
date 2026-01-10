# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
# Unauthorized derivative works or closed redistribution prohibited under AGPL §13.
"""Z-space inference quickstart for SpiralTorch.

Run from the repository root after installing the wheel (or `maturin develop`):

  python bindings/st-py/examples/zspace_inference_quickstart.py

Highlights:
* `ZSpaceTrainer.step_partial(...)` fuses partial observations into the latent
  posterior and immediately updates the Z vector (one-liner).
"""

from __future__ import annotations

import spiraltorch as st


def main() -> None:
    trainer = st.ZSpaceTrainer(z_dim=4, lr=0.05, lam_frac=0.05)
    print("initial z:", trainer.state)

    partial = {
        "speed": 0.2,
        "memory": 0.1,
        "stability": 0.9,
        "gradient": [0.15, -0.05, 0.02, 0.0],
    }

    loss = trainer.step_partial(partial, smoothing=0.4)
    inference = trainer.last_inference
    print("loss:", loss)
    print("updated z:", trainer.state)
    if inference is not None:
        print(
            "inference:",
            {
                "residual": inference.residual,
                "confidence": inference.confidence,
                "barycentric": inference.barycentric,
            },
        )

    bundle = st.ZSpacePartialBundle(
        {"speed": 0.05, "stability": 0.95},
        weight=0.5,
        origin="demo",
        telemetry={"demo.energy": 1.0, "demo.focus": 0.25},
    )
    loss2 = trainer.step_partial(bundle, smoothing=0.25)
    print("loss2:", loss2)
    print("z2:", trainer.state)


if __name__ == "__main__":
    main()

