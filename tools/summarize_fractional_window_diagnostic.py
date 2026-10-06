#!/usr/bin/env python3
"""Pure-stdlib seed averages; no checkpoint execution or quality inference."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def summarize(raw):
    report = json.loads(raw)
    if (report["status"] != "completed" or not report["frozen_manifest_inputs_preserved"]
            or not report["frozen_base_unchanged"] or report["training_updates"] != 0
            or report["optimizer_steps"] != 0):
        raise ValueError("expected a completed read-only diagnostic")
    runs = report["runs"]
    if (set(runs) != {f"{s}:{report['arm']}" for s in report["seeds"]}
            or not runs or not all(r["full_replay"]["exact"] for r in runs.values())):
        raise ValueError("incomplete replay gate")
    modes = {}
    for mode in report["windows"]:
        modes[mode] = {}
        for label in report["evaluation_block_hashes"]:
            means = [row["modes"][mode]["scores"][label]["mean"] for row in runs.values()]
            contrasts = [row["modes"][mode]["scores"][label]["mean"]
                         - row["modes"]["full"]["scores"][label]["mean"] for row in runs.values()]
            if not all(math.isfinite(x) for x in means + contrasts):
                raise ValueError("nonfinite mean")
            modes[mode][label] = {"mean_ce": math.fsum(means) / len(means),
                                 "mean_delta": math.fsum(contrasts) / len(contrasts), "per_seed": contrasts}
    return {"schema": "spiraltorch.fractional_history_window_numeric_summary.v1",
            "aggregation": "binary64_math_fsum_divide_by_seed_count",
            "diagnostic_id": report["diagnostic_id"], "results_sha256": hashlib.sha256(raw).hexdigest(),
            "runs": len(runs), "blocks_per_seed": {k: len(v) for k, v in report["evaluation_block_hashes"].items()},
            "modes": modes, "scope": report["scope"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    encoded = json.dumps(summarize(args.input.read_bytes()), indent=2, allow_nan=False) + "\n"
    with args.output.open("x") as handle:
        handle.write(encoded)


if __name__ == "__main__":
    main()
