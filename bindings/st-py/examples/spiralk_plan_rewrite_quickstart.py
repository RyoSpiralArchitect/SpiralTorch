# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
# Unauthorized derivative works or closed redistribution prohibited under AGPL §13.
"""SpiralK KDSl plan rewrite quickstart for SpiralTorch wheels.

Run from the repository root after installing the wheel:

  python bindings/st-py/examples/spiralk_plan_rewrite_quickstart.py

This script demonstrates:

* `spiraltorch.plan(kind="fft", ...)` to build a rank plan.
* Generating a baseline KDSl hint with `RankPlan.fft_spiralk_hint()`.
* Applying KDSl overrides via `RankPlan.rewrite_with_spiralk(...)`.
"""

from __future__ import annotations

import argparse

import spiraltorch as st


def _summarize(plan: st.RankPlan) -> str:
    return (
        f"kind={plan.kind} tile={plan.fft_tile} radix={plan.fft_radix} segments={plan.fft_segments} "
        f"workgroup={plan.workgroup} lanes={plan.lanes} subgroup={plan.subgroup}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--tile-cols", type=int, default=1024)
    args = parser.parse_args()

    plan = st.plan(
        kind="fft",
        rows=args.rows,
        cols=args.cols,
        k=args.k,
        backend="auto",
        lane_width=32,
        subgroup=True,
    )
    print("base:", _summarize(plan))

    base_hint = plan.fft_spiralk_hint()
    print("\nbase KDSl hint:\n" + base_hint)

    program = f"if c >= {args.tile_cols} {{ tile_cols: {args.tile_cols}; }}\n"
    rewritten = plan.rewrite_with_spiralk(program)
    print("\noverride program:\n" + program)
    print("rewritten:", _summarize(rewritten))

    if _summarize(rewritten) == _summarize(plan):
        print("(note) plan did not change; try a different --tile-cols / --cols / --k.")


if __name__ == "__main__":
    main()
