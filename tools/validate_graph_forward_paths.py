#!/usr/bin/env python3
"""Reaggregate matched forward timing captures; reject contract/fixture drift."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_graph_forward_paths import admit_native, close, digest, summarize, validate_samples


def admit_client(document, fixture, fixture_path, browser=False):
    schema = "spiraltorch.graph_forward_browser_bench.v1" if browser else "spiraltorch.graph_forward_paths.v1"
    if document.get("schema") != schema or document.get("status") != "passed":
        raise ValueError("client did not complete")
    if browser:
        if (document.get("source_fixture_sha256") != digest(fixture_path)
                or document.get("asset_sha256", {}).get("/fixture.json") != digest(fixture_path)
                or document.get("page_errors") != [] or document.get("warmup") != 3
                or document.get("samples_per_route") != 9 or document.get("burst") != 8
                or document.get("adapter", {}).get("backend") != "BrowserWebGpu"
                or document.get("adapter", {}).get("device_type") in (None, "Cpu")):
            raise ValueError("browser lineage or observation boundary differs")
        routes = ["scalar_h2h", "register_h2h", "scalar_burst", "register_burst"]
        if document.get("routes") != routes:
            raise ValueError("browser route matrix differs")
    else:
        if (document.get("sources", {}).get(str(fixture_path.resolve())) != digest(fixture_path)
                or document.get("devices") != ["cpu", "mps"] or document.get("build_info", {}).get("profile") != "release"
                or document.get("build_info", {}).get("features", {}).get("wgpu") is not True):
            raise ValueError("Python lineage, build or device matrix differs")
        routes = [f"python_{k}_{c}" for k in ("scalar", "register") for c in ("h2h", "burst")]
        routes += [f"torch_{d}_{c}" for d in ("cpu", "mps") for c in ("h2h", "burst")]
    if len(document.get("cases", [])) != len(fixture["cases"]):
        raise ValueError("incomplete client matrix")
    for case, frozen in zip(document["cases"], fixture["cases"]):
        if any(case.get(key) != frozen[key] for key in ("shape", "seed", "depth")):
            raise ValueError("client recipe changed")
        if not browser and case.get("routes") != routes:
            raise ValueError("Python routes changed")
        validate_samples(case, routes)
        if set(case["last_outputs"]) != set(routes):
            raise ValueError("missing client captures")
        for values in case["last_outputs"].values():
            close(values, frozen["reference"])
        if not browser and case.get("summary") != summarize(case["samples"]):
            raise ValueError("client summary differs from raw timings")
    return document["cases"]


def aggregate(baseline_native, candidate_native, baseline_python, candidate_python,
              baseline_browser, candidate_browser, fixture_path):
    admit_native(baseline_native)
    admit_native(candidate_native)
    for before, after in zip(baseline_native["cases"], candidate_native["cases"]):
        if any(before[key] != after[key] for key in ("shape", "depth", "seed", "plan", "input", "source_operations", "gpu_stages")):
            raise ValueError("candidate changed the frozen model/input")
        close(after["reference"], before["reference"])
    if baseline_native["adapter"] != candidate_native["adapter"]:
        raise ValueError("native adapters differ")
    for doc, browser in ((baseline_python, False), (candidate_python, False),
                         (baseline_browser, True), (candidate_browser, True)):
        admit_client(doc, baseline_native, fixture_path, browser)
    if (baseline_python["torch"] != candidate_python["torch"]
            or baseline_python["device_admission"] != candidate_python["device_admission"]
            or baseline_browser["browser_version"] != candidate_browser["browser_version"]
            or baseline_browser["adapter"] != candidate_browser["adapter"]):
        raise ValueError("comparison device/runtime changed")
    results = []
    pairs = (("native", baseline_native, candidate_native), ("python_torch", baseline_python, candidate_python),
             ("browser", baseline_browser, candidate_browser))
    for i, frozen in enumerate(baseline_native["cases"]):
        result = {key:frozen[key] for key in ("shape", "depth", "seed")}
        for family, before, after in pairs:
            b, a = [summarize(doc["cases"][i]["samples"]) for doc in (before, after)]
            result[family] = dict(baseline=b, candidate=a, candidate_over_baseline={
                route:a[route]["median_ms_per_forward"]/b[route]["median_ms_per_forward"] for route in b})
        n = result["native"]["candidate"]
        p = result["python_torch"]["candidate"]
        result["candidate_comparisons"] = dict(
            native_scalar_h2h_over_legacy=n[1]["median_ms_per_forward"]/n[0]["median_ms_per_forward"],
            python_scalar_h2h_over_torch_mps=p["python_scalar_h2h"]["median_ms_per_forward"]/p["torch_mps_h2h"]["median_ms_per_forward"],
            python_scalar_burst_over_torch_mps=p["python_scalar_burst"]["median_ms_per_forward"]/p["torch_mps_burst"]["median_ms_per_forward"])
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-native", "candidate-native", "baseline-python", "candidate-python", "baseline-browser", "candidate-browser", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    paths = [getattr(args, name) for name in ("baseline_native", "candidate_native", "baseline_python", "candidate_python", "baseline_browser", "candidate_browser")]
    report = dict(schema="spiraltorch.graph_forward_paths_comparison.v1", status="error",
                  scope="Ratios of per-case median ms/forward; lower is better. H2H and fixed-input burst are not interchangeable. Separate baseline/candidate runs, not isolated GPU timestamps; macOS contention and browser physical GPU UNKNOWN.")
    with args.output.open("x") as output:
        try:
            report["sources"] = {str(path.resolve()):digest(path) for path in paths}
            report["cases"] = aggregate(*(json.loads(path.read_bytes()) for path in paths), paths[0])
            report["status"] = "passed"
        except Exception as error:
            report["error"] = repr(error)
            raise
        finally:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")


if __name__ == "__main__":
    main()
