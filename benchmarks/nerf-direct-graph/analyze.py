"""Complete three-round comparison; never select only winning conditions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import ATOL, RTOL, KEYS, admit, close, f32, inputs


def summarize(intervals):
    expected = {(r, b, n, d) for r in range(3) for b in range(9)
                for n in (1, 4) for d in ("native_staged", "native_direct", "browser_staged", "browser_direct", "cpu", "mps")}
    index = {(x["round"], x["block"], x["burst"], x["route"]): x["elapsed_ms"] for x in intervals}
    if len(intervals) != len(expected) or set(index) != expected:
        raise ValueError("incomplete/duplicate published intervals")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in index.values()):
        raise ValueError("invalid published timing")
    result = []
    for burst in (1, 4):
        medians = {route: statistics.median(v for (_, _, n, d), v in index.items() if n == burst and d == route)
                   for route in ("native_staged", "native_direct", "browser_staged", "browser_direct", "cpu", "mps")}
        paired = {runtime: statistics.median(index[r, b, burst, runtime + "_staged"] / index[r, b, burst, runtime + "_direct"]
                                            for r in range(3) for b in range(9)) for runtime in ("native", "browser")}
        result.append({"burst":burst, "median_interval_ms":medians, "paired_staged_over_direct":paired,
                       "eager_mps_over_direct_median_ratio":{r:medians["mps"] / medians[r + "_direct"] for r in ("native", "browser")}})
    return result


def analyze(native, browser, torch):
    if any(len(reports) != 3 for reports in [native, browser, torch]):
        raise ValueError("exactly three complete rounds are required")
    admitted = [[admit(report, kind) for report in reports]
                for reports, kind in [(native, "wgpu"), (browser, "wgpu"), (torch, "torch")]]
    for report in browser:
        if report["page_errors"] or report["console_messages"] or report["browser_adapter_probe"]["is_fallback_adapter"] is not False:
            raise ValueError("browser runtime errors or fallback")
    records = []
    for ident in sorted(KEYS):
        fingerprint = inputs(admitted[0][0][ident])
        reference = admitted[2][0][ident]["reference"]
        errors, scaled, intervals = {}, {}, []
        for family, rounds in zip(["native", "browser", "torch"], admitted):
            routes = [family + "_staged", family + "_direct"] if family != "torch" else ["cpu", "mps"]
            for repeat, cases in enumerate(rounds):
                case = cases[ident]
                if inputs(case) != fingerprint:
                    raise ValueError("ray/parameter bytes drifted between routes or rounds")
                close(case["reference"], reference)
                for route, output in zip(routes, case["last_outputs"]):
                    errors[route] = max(errors.get(route, 0), close(output, reference))
                    scaled[route] = max(scaled.get(route, 0), max(abs(a-b)/(ATOL+RTOL*abs(b)) for a, b in zip(output, reference)))
                for interval in case["intervals"]:
                    route = interval["route"] if family == "torch" else family + "_" + interval["route"]
                    intervals.append({**interval, "route":route, "round":repeat})
        records.append({"rays":ident[0], "samples":ident[1], "hidden":ident[2], "input_sha256":fingerprint,
                        "oracle_sha256":hashlib.sha256(f32(reference)).hexdigest(), "max_abs_errors":errors,
                        "max_scaled_errors":scaled, "intervals":intervals, "summary":summarize(intervals)})
    return {"schema":"spiraltorch.nerf_direct_summary.v1", "status":"passed", "rounds":3,
            "tolerance":{"atol":ATOL,"rtol":RTOL}, "cases":records, "descriptive_summary":describe(records),
            "native_adapters":[r["adapter"] for r in native],
            "browser_metadata":[{k:r[k] for k in ["adapter", "browser_version", "browser_adapter_probe", "asset_sha256"]} for r in browser],
            "torch_metadata":[{k:r[k] for k in ["torch_version", "devices", "intra_op_threads", "inter_op_threads", "compiled", "thin_alpha"]} for r in torch],
            "boundary":"One M4 desktop, rotating serial runtime order over three rounds; nine rotated paired blocks per cell and round. Timings include host orchestration, allocations, GPU submissions, final RGBA/guard copy/map and completion; setup excluded. Burst4 observes only its final result. Every timed result checked against its route reference; final outputs and WGPU reference independently checked against CPU f64-integration/f32-NN oracle. Eager Torch uses stabilized thin alpha and vectorized f32 prefix sums, not identical kernels. Descriptive ratios, not uncertainty estimates, isolated GPU timings, general PyTorch superiority, training or scene-quality evidence."}


def describe(records):
    summary = {}
    for runtime in ["native", "browser"]:
        ratios = [s["paired_staged_over_direct"][runtime] for c in records for s in c["summary"]]
        summary[runtime] = {"geomean_staged_over_direct":math.exp(statistics.mean(math.log(r) for r in ratios)),
                            "min":min(ratios), "max":max(ratios), "cells_over_1":sum(r > 1 for r in ratios), "cells":len(ratios)}
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["native", "browser", "torch"]:
        parser.add_argument("--" + name, action="append", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(*[[json.loads(p.read_text()) for p in getattr(args, name)]
                               for name in ["native", "browser", "torch"]]), allow_nan=False))
