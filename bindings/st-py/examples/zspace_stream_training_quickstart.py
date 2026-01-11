# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
# Unauthorized derivative works or closed redistribution prohibited under AGPL §13.
"""Streaming Z-space trainer quickstart for SpiralTorch wheels.

Run from the repository root after installing the wheel:

  python bindings/st-py/examples/zspace_stream_training_quickstart.py

This script demonstrates:

* `ZSpaceTrainer` as a lightweight meta-controller for (speed, memory, stability) metrics.
* `stream_zspace_training(...)` to feed metrics as an iterator with an on-step callback.
* `decode_zspace_embedding(...)` to introspect the final state.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterator, Mapping

import spiraltorch as st


def _metric_stream(steps: int) -> Iterator[Mapping[str, object]]:
    for t in range(steps):
        phase = t * 0.15
        speed = 0.55 + 0.25 * math.sin(phase)
        memory = 0.45 + 0.25 * math.cos(phase * 0.9)
        stability = 0.85 - 0.35 * abs(speed - memory)
        gradient = [0.05 * math.sin(phase + i) for i in range(4)]
        yield {
            "speed": speed,
            "memory": memory,
            "stability": stability,
            "gradient": gradient,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--z-dim", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.02)
    args = parser.parse_args()

    trainer = st.ZSpaceTrainer(z_dim=args.z_dim, alpha=args.alpha, lr=args.lr)

    def on_step(i: int, state: list[float], loss: float) -> None:
        if i == 0 or (i + 1) % 10 == 0:
            head = ", ".join(f"{v:+.3f}" for v in state[:4])
            print(f"step {i+1:03d}: loss={loss:.4f} z[:4]=[{head}]")

    st.stream_zspace_training(trainer, _metric_stream(args.steps), on_step=on_step)
    print("\nfinal z:", [round(v, 4) for v in trainer.state])

    decoded = st.decode_zspace_embedding(trainer.state, alpha=float(args.alpha))
    print("decoded metrics:", decoded.metrics)


if __name__ == "__main__":
    main()
