"""New route identities; shared NeRF numerical oracle, grid and aggregation."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nerf-single-submit"))
import protocol as base

SCHEMA = "spiraltorch.nerf_row_input_bench.v1"
SUMMARY = "spiraltorch.nerf_row_input_summary.v1"
COMPARISON = "packed_input_vs_row_addressing"
RENAMES = {
    "native_separate": "native_packed", "native_single": "native_rows",
    "browser_separate": "browser_packed", "browser_single": "browser_rows",
    "paired_separate_over_single": "paired_packed_over_rows",
    "geomean_separate_over_single": "geomean_packed_over_rows",
    "eager_mps_over_single_median_ratio": "eager_mps_over_rows_median_ratio",
}


def adapt(report):
    if report["schema"] != SCHEMA or report["comparison"] != COMPARISON:
        raise ValueError("not a packed/row-input comparison")
    routes = {"packed": "separate", "rows": "single"}
    cases = []
    for case in report["cases"]:
        if any(i["route"] not in routes for i in case["intervals"]):
            raise ValueError("unknown input-addressing route")
        cases.append({**case, "intervals": [{**i, "route": routes[i["route"]]}
                                           for i in case["intervals"]]})
    result = {**report, "schema": base.SCHEMA, "comparison": base.COMPARISON, "cases": cases}
    base.normalize(result)
    return result


def rename(value, reverse=False):
    mapping = {v: k for k, v in RENAMES.items()} if reverse else RENAMES
    if isinstance(value, dict):
        return {mapping.get(k, k): rename(v, reverse) for k, v in value.items()}
    if isinstance(value, list):
        return [rename(v, reverse) for v in value]
    return mapping.get(value, value) if isinstance(value, str) else value


def analyze(native, browser, torch):
    result = rename(base.analyze([adapt(r) for r in native], [adapt(r) for r in browser], torch))
    result.update(schema=SUMMARY, comparison=COMPARISON, boundary=(
        "Same three-submission sample/NN/composite chain, parameters and owning terminal RGBA/guard "
        "boundary. Forced input packing versus automatic first-linear row addressing in the same "
        "current binary. Not a previous-binary or submission-count comparison. Row addressing "
        "avoids the position pack only when needed and representable; 1x1 is already contiguous. "
        "Three rotated serial runtime rounds, nine paired blocks per condition/round, bursts 1/4. "
        "Setup, validation and serialization excluded; per-render allocation, encoding, submission "
        "and final owning copy/map/completion included. Burst4 observes only its last output. "
        "Independent CPU f64 geometry/integration plus f32 NN oracle; timed Torch CPU/MPS eager f32 "
        "with stable thin alpha. Shared M4 desktop, coarse browser clock, descriptive ratios only; "
        "no isolated GPU timing, general PyTorch superiority, training or scene-quality claim. "
        "Temporary route-label adaptation reuses existing validation/statistics without altering raw reports."
    ))
    return result


def validate_summary(result):
    if result["schema"] != SUMMARY or result["comparison"] != COMPARISON:
        raise ValueError("row-input summary identity")
    adapted = rename(result, True)
    adapted.update(schema="spiraltorch.nerf_submit_summary.v1", comparison=base.COMPARISON)
    base.validate_summary(adapted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("torch").add_argument("source", type=Path)
    analysis = sub.add_parser("analyze")
    for family in ("native", "browser", "torch"):
        analysis.add_argument("--" + family, action="append", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "torch":
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        result = base.torch_control.run(base.normalize(adapt(json.loads(args.source.read_text()))))
        result["input_protocol"] = SCHEMA
    else:
        result = analyze(*[[json.loads(p.read_text()) for p in getattr(args, family)]
                           for family in ("native", "browser", "torch")])
        validate_summary(result)
    print(json.dumps(result, allow_nan=False))
