"""Pointwise-view identities and prelude; shared NeRF oracle/statistics."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nerf-single-submit"))
import protocol as base

SCHEMA = "spiraltorch.nerf_pointwise_input_bench.v1"
SUMMARY = "spiraltorch.nerf_pointwise_input_summary.v1"
COMPARISON = "packed_input_vs_pointwise_view"
RENAMES = {
    "native_separate": "native_packed", "native_single": "native_views",
    "browser_separate": "browser_packed", "browser_single": "browser_views",
    "paired_separate_over_single": "paired_packed_over_views",
    "geomean_separate_over_single": "geomean_packed_over_views",
    "eager_mps_over_single_median_ratio": "eager_mps_over_views_median_ratio",
}


def prelude(report):
    if report.get("input_prelude") != "relu" or any(
        c.get("input_prelude") != "relu" for c in report["cases"]
    ):
        raise ValueError("pointwise comparison requires the same explicit ReLU prelude everywhere")


def adapt(report):
    if report["schema"] != SCHEMA or report["comparison"] != COMPARISON:
        raise ValueError("not a packed/pointwise-view comparison")
    prelude(report)
    if type(report.get("view_cases")) is not int or report["view_cases"] != 8:
        raise ValueError("missing native/browser view admission")
    routes = {"packed": "separate", "views": "single"}
    cases = []
    for case in report["cases"]:
        if any(i["route"] not in routes for i in case["intervals"]):
            raise ValueError("unknown pointwise input route")
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
    for report in torch:
        prelude(report)
        if report.get("input_protocol") != SCHEMA:
            raise ValueError("Torch belongs to a different input protocol")
    result = rename(base.analyze([adapt(r) for r in native], [adapt(r) for r in browser], torch))
    result.update(schema=SUMMARY, comparison=COMPARISON, input_prelude="relu", boundary=(
        "Same three-submission sample/NN/composite chain, ReLU input prelude, parameters and owning "
        "terminal RGBA/guard boundary. Forced packing versus direct pointwise N-D view addressing "
        "in one current binary. No submission-count or previous-binary comparison. The 1x1 inputs "
        "are already contiguous noise controls. Three rotated serial runtime rounds, nine paired "
        "blocks per condition/round, bursts 1/4. Setup, validation and serialization excluded; "
        "per-render allocation, encoding, submission and final owning copy/map/completion included. "
        "Burst4 observes only its last output. Independent CPU f64 geometry/integration plus f32 NN "
        "oracle; eager f32 Torch CPU/MPS uses the SAME ReLU prelude and stable thin alpha. Shared "
        "M4 desktop and coarse browser clock, descriptive ratios only; no isolated GPU timing, "
        "general PyTorch superiority, training or scene-quality claim. Label adaptation is temporary."
    ))
    for case in result["cases"]:
        case["input_prelude"] = "relu"
    return result


def validate_summary(result):
    if result["schema"] != SUMMARY or result["comparison"] != COMPARISON:
        raise ValueError("pointwise-input summary identity")
    prelude(result)
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
        result = base.torch_control.run(
            base.normalize(adapt(json.loads(args.source.read_text()))), input_prelude="relu"
        )
        result["input_protocol"] = SCHEMA
    else:
        result = analyze(*[[json.loads(p.read_text()) for p in getattr(args, family)]
                           for family in ("native", "browser", "torch")])
        validate_summary(result)
    print(json.dumps(result, allow_nan=False))
