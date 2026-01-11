# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
# Unauthorized derivative works or closed redistribution prohibited under AGPL §13.
"""Maxwell-coded envelopes → SpiralK hints quickstart for SpiralTorch wheels.

Run from the repository root after installing the wheel:

  python bindings/st-py/examples/maxwell_spiralk_bridge_quickstart.py

This script demonstrates:

* `spiraltorch.spiralk.SequentialZ` as a lightweight streaming Z-stat estimator.
* Converting a detected pulse into KDSl via `MaxwellSpiralKBridge`.
* `required_blocks(...)` to estimate block counts for a target Z threshold.

Note: the KDSl emitted here targets `maxwell.bias` (a runtime knob), not rank-plan overrides.
"""

from __future__ import annotations

import argparse
import math

import spiraltorch as st


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--target-z", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=0.9)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=96)
    args = parser.parse_args()

    seq = st.spiralk.SequentialZ()
    bridge = st.spiralk.MaxwellSpiralKBridge()

    print("required blocks estimate:", st.spiralk.required_blocks(args.target_z, args.sigma, args.kappa, args.lambda_))

    detected = False
    for i in range(args.max_samples):
        sample = 0.25 * math.sin(i * 0.25) + 0.05 * math.cos(i * 0.9)
        z = seq.push(sample)
        if z is None:
            continue

        if z >= args.target_z:
            detected = True
            se = seq.standard_error() or 0.0
            hint = bridge.push_pulse(
                channel="demo/attention",
                blocks=seq.len(),
                mean=seq.mean(),
                standard_error=se,
                z_score=float(z),
                band_energy=(0.4, 0.35, 0.25),
                z_bias=0.15,
            )
            print("\nZ threshold reached at blocks=", seq.len(), "z≈", float(z))
            print("hint line:\n", hint.script_line())
            break

    if not detected:
        print("\n(no detection) increase --max-samples or lower --target-z")

    script = bridge.script()
    print("\nbridge KDSl script:\n", script)


if __name__ == "__main__":
    main()

