#!/usr/bin/env python3
"""Compare the installed native core with a built Node wasm32 client, offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import spiraltorch as st


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--wasm-module", type=Path)
    destination.add_argument("--output-fixture", type=Path)
    parser.add_argument("--node", default="node")
    args = parser.parse_args()
    records = []
    for normalization in ("active_positions", "eligible_targets"):
        for schedule in (
            {"kind": "constant"},
            {
                "kind": "linear_decay",
                "start_update": 2,
                "end_update": 9,
                "final_scale": 0.1,
            },
        ):
            for slot in (0, 2, 3, 5, 9, 100):
                for active, eligible in ((0, 0), (0, 7), (1, 7), (7, 7)):
                    request = {
                        "config": {
                            "normalization": normalization,
                            "schedule": schedule,
                        },
                        "base_strength": 0.2,
                        "completed_update_slots": slot,
                        "active_position_count": active,
                        "eligible_target_count": eligible,
                    }
                    records.append(
                        {
                            "request": request,
                            "expected": st.zspace_repetition_objective_control(
                                **request
                            ),
                        }
                    )
    if args.output_fixture is not None:
        args.output_fixture.parent.mkdir(parents=True, exist_ok=True)
        with args.output_fixture.open("x", encoding="utf-8") as handle:
            json.dump(records, handle, allow_nan=False)
        print(f"Exported {len(records)} native Rust objective controls")
        return
    with tempfile.TemporaryDirectory(prefix="spiraltorch-objective-") as directory:
        fixture = Path(directory) / "fixture.json"
        fixture.write_text(json.dumps(records, allow_nan=False), encoding="utf-8")
        subprocess.run(
            [
                args.node,
                str(
                    Path(__file__).resolve().parents[1]
                    / "bindings/st-wasm/tests/repetition_objective.cjs"
                ),
                str(args.wasm_module.resolve()),
                str(fixture),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
