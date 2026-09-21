"""Submission comparison using the existing NeRF grid/oracle/admission engine.

Normalize route labels only inside shared validation/statistics. Raw and public
reports retain the separate/single schema; neither route is the old copy-based
stable-workspace control. The Torch numerical/timing implementation is shared.
"""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nerf-direct-graph"))
import analyze as base_analysis
import contract as base_contract
import torch_control

SCHEMA = "spiraltorch.nerf_submit_bench.v1"
COMPARISON = "separate_direct_vs_single_submission"
RENAMES = {
    "native_staged":"native_separate", "native_direct":"native_single",
    "browser_staged":"browser_separate", "browser_direct":"browser_single",
    "paired_staged_over_direct":"paired_separate_over_single",
    "geomean_staged_over_direct":"geomean_separate_over_single",
    "eager_mps_over_direct_median_ratio":"eager_mps_over_single_median_ratio",
}


def normalize(report):
    if report["schema"] != SCHEMA or report["comparison"] != COMPARISON:
        raise ValueError("not a separate/single submission comparison")
    routes = {"separate":"staged", "single":"direct"}
    cases = []
    for case in report["cases"]:
        if any(i["route"] not in routes for i in case["intervals"]):
            raise ValueError("unknown submission route")
        cases.append({**case, "intervals":[{**i, "route":routes[i["route"]]} for i in case["intervals"]]})
    adapted = {**report, "schema":"spiraltorch.nerf_direct_bench.v1", "cases":cases}
    base_contract.admit(adapted, "wgpu")
    return adapted


def rename(value, reverse=False):
    mapping = {v:k for k,v in RENAMES.items()} if reverse else RENAMES
    if isinstance(value, dict):
        return {mapping.get(k,k):rename(v,reverse) for k,v in value.items()}
    if isinstance(value, list):
        return [rename(v,reverse) for v in value]
    return mapping.get(value,value) if isinstance(value,str) else value


def analyze(native, browser, torch):
    result = rename(base_analysis.analyze([normalize(r) for r in native], [normalize(r) for r in browser], torch))
    result["schema"] = "spiraltorch.nerf_submit_summary.v1"
    result["comparison"] = COMPARISON
    result["boundary"] = (
        "Same input bytes, shaders and owning terminal RGBA/guard boundary; separate direct sample/NN/composite "
        "versus one composed render submission. Separate uses the same current graph implementation, not "
        "a previous binary or the copy-based staged route. Three rotated serial runtime rounds; nine paired "
        "blocks per condition and round. Setup, validation and serialization excluded; per-render allocation, "
        "encoding, submissions and final copy/map/completion included. Burst4 observes only its last output. "
        "Shared independent CPU f64 geometry/integration + f32 NN oracle; timed Torch CPU/MPS eager f32 "
        "with stable thin alpha. Structural submission counts are not measured GPU timestamps. Shared M4 "
        "desktop, not dedicated hardware. Descriptive ratios only; no universal speed, training, image-quality "
        "or cross-vendor claim. Historical route names are normalized only inside shared admission/statistics."
    )
    return result


def validate_summary(result):
    if result["schema"] != "spiraltorch.nerf_submit_summary.v1" or result["comparison"] != COMPARISON or result["status"] != "passed":
        raise ValueError("submission summary identity")
    if result["rounds"] != 3 or result["tolerance"] != {"atol":base_contract.ATOL,"rtol":base_contract.RTOL}:
        raise ValueError("submission protocol drift")
    cases = result["cases"]
    if len(cases) != len(base_contract.KEYS) or {base_contract.key(c) for c in cases} != base_contract.KEYS:
        raise ValueError("summary grid")
    for case in cases:
        if case["summary"] != rename(base_analysis.summarize(rename(case["intervals"], True))):
            raise ValueError("timing aggregation mismatch")
        if set(case["max_scaled_errors"]) != {"native_separate","native_single","browser_separate","browser_single","cpu","mps"}:
            raise ValueError("numerical routes")
        if any(type(v) not in (int,float) or not 0 <= v <= 1 for v in case["max_scaled_errors"].values()):
            raise ValueError("unaccepted numerical result")
    if result["descriptive_summary"] != rename(base_analysis.describe(rename(cases, True))):
        raise ValueError("aggregate summary mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode",required=True)
    sub.add_parser("torch").add_argument("source",type=Path)
    analyze_parser = sub.add_parser("analyze")
    for name in ["native","browser","torch"]:
        analyze_parser.add_argument("--"+name,action="append",type=Path,required=True)
    args = parser.parse_args()
    if args.mode == "torch":
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        result = torch_control.run(normalize(json.loads(args.source.read_text())))
        result["input_protocol"] = SCHEMA
    else:
        result = analyze(*[[json.loads(p.read_text()) for p in getattr(args,name)] for name in ["native","browser","torch"]])
        validate_summary(result)
    print(json.dumps(result,allow_nan=False))
