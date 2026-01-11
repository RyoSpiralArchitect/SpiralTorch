# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
# Unauthorized derivative works or closed redistribution prohibited under AGPL §13.
"""SoT-3Dφ → TensorBiome → Hypergrad quickstart for SpiralTorch wheels.

Run from the repository root after installing the wheel:

  python bindings/st-py/examples/sot_biome_quickstart.py

This script demonstrates:

* `spiraltorch.sot.generate_plan` (SoT-3Dφ spiral plan synthesis).
* Converting a plan into tensors (`as_tensor` / `feature_tensor` / `role_tensor`).
* Growing a guarded `TensorBiome` via `SoT3DPlan.grow_biome(...)`.
* Feeding the canopy tensor into `spiral.hypergrad_session` and deriving operator hints.
"""

from __future__ import annotations

import argparse
from pprint import pprint

import spiraltorch as st

try:
    from spiral import hypergrad_session, hypergrad_summary_dict, suggest_hypergrad_operator
except Exception:  # pragma: no cover - spiral helpers should ship with the wheel
    hypergrad_session = None
    hypergrad_summary_dict = None
    suggest_hypergrad_operator = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--radial-growth", type=float, default=0.08)
    parser.add_argument("--curvature", type=float, default=-1.0)
    parser.add_argument("--max-volume", type=int, default=512)
    parser.add_argument(
        "--strict-biome",
        action="store_true",
        help="Fail if SoT3DPlan.grow_biome/canopy is unavailable (useful for CI).",
    )
    args = parser.parse_args()

    if not hasattr(st, "sot") or not hasattr(st.sot, "generate_plan"):
        raise RuntimeError("spiraltorch.sot is unavailable in this build.")

    plan = st.sot.generate_plan(args.steps, radial_growth=args.radial_growth)
    print("plan:", plan)
    print("polyline head:", plan.polyline()[:3])
    print("reflection points head:", plan.reflection_points()[:8])

    print("plan.as_tensor shape:", plan.as_tensor().shape())
    print("plan.feature_tensor shape:", plan.feature_tensor().shape())
    print("plan.role_tensor shape:", plan.role_tensor().shape())

    topos = st.hypergrad_topos(curvature=args.curvature, max_volume=args.max_volume)
    biome = None
    canopy = None
    try:
        biome = plan.grow_biome(topos, label_prefix="sot_demo")
        canopy = biome.canopy()
    except Exception as exc:
        if args.strict_biome:
            raise
        print("\nSoT3DPlan.grow_biome(...) failed:", exc)
        print("Falling back to `plan.feature_tensor()`; install a newer wheel for biome support.")
        canopy = plan.feature_tensor()

    if biome is not None:
        print("\nbiome shoots:", len(biome), "total weight:", biome.total_weight())
    print("canopy shape:", canopy.shape())

    if hypergrad_session is None:
        print("\nspiral.hypergrad_session is unavailable; skipping hypergrad demo.")
        return

    with hypergrad_session(
        *canopy.shape(),
        curvature=args.curvature,
        learning_rate=0.05,
        topos=topos,
    ) as tape:
        tape.accumulate_wave(canopy)
        summary = hypergrad_summary_dict(tape)  # type: ignore[misc]
        hints = suggest_hypergrad_operator(summary)  # type: ignore[misc]

    print("\nhypergrad summary:")
    pprint(summary["summary"])
    print("suggested WGSL operator hint:", {"mix": hints["mix"], "gain": hints["gain"]})


if __name__ == "__main__":
    main()
