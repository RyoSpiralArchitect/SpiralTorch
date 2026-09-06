#!/usr/bin/env python3
"""Recheck frozen fixtures and summarize A/B/B/A without dropping slow controls."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / "tools"))
import bench_resident_rank_vs_torch as bench

ORDER = ["a1", "b1", "b2", "a2"]
BASE = "c4532d3e30b39b1fbb28cdde9b535fac797101f6"
CANDIDATE = "189a2f0e35d7347bc1763fc4a38d84db3ba39352"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_samples(samples, count):
    require(len(samples) == count, "timing sample count changed")
    require(all(math.isfinite(x) and x >= 0 for x in samples), "invalid timing")
    return statistics.median(samples) * 1000


def f32(value):
    return struct.pack("<f", value)


def verify_native(report, requests, commit, request_sha):
    require(report["status"] == "passed" and len(report["cases"]) == 243,
            "incomplete native comparison")
    require(report["source"]["commit"] == commit and not report["source"]["tracked_dirty"],
            "wrong or dirty source")
    require(report["provenance"]["valid"] and report["build_source_binding"]["valid"],
            "source/image admission failed")
    require(report["native_build_identity"]["manifest"]["git"]["commit"] == commit,
            "wrong embedded build")
    require(report["source"] == report["provenance"]["source_after"], "source changed")
    require(report["request_sha256"] == request_sha and report["resident_only_requested"],
            "wrong input or measurement boundary")
    for case, request in zip(report["cases"], requests):
        require(case["request"] == {k: v for k, v in request.items() if k != "input"},
                "case order or geometry changed")
        native = case["native"]
        bench.validate_native_result(native, request, True)
        require(native["adapter"]["name"] == report["torch_device"]
                == "NVIDIA GeForce RTX 5090", "unmatched device")
        require(native["adapter"]["backend"] == "Vulkan", "wrong backend")
        require(case["torch_operation"] == bench.cuda_rank_operation(request),
                "weaker CUDA ordering comparator")
        values, indices = [], []
        for row in range(request["rows"]):
            data = request["input"][row * request["cols"]:(row + 1) * request["cols"]]
            require(all(math.isfinite(x) for x in data), "native fixture must be finite")
            order = sorted(range(len(data)), key=lambda i:
                           ((-data[i] if request["kind"] == "topk" else data[i]), i))
            start = (len(data) - request["k"]) // 2 if request["kind"] == "midk" else 0
            selected = order[start:start + request["k"]]
            indices.extend(selected)
            values.extend(data[i] for i in selected)
        require(native["indices"] == indices, "canonical index mismatch")
        require(list(map(f32, native["values"])) == list(map(f32, values)),
                "exact fp32 value mismatch")
        require(native["resident_repetitions"] == 16, "wrong repetition boundary")
        finite_samples(native["samples_ms"]["resident_dispatch_fence_per_op"], 12)
        finite_samples(case["timings"]["torch_resident_per_op"]["samples_ms"], 12)


def main():
    requests = list(bench.requests_for(bench.audit.load_bench_module(), "active-lanes"))
    payload = "".join(json.dumps(r, allow_nan=False) + "\n" for r in requests)
    request_sha = hashlib.sha256(payload.encode()).hexdigest()
    reports, hashes = {}, {}
    for platform in ["native", "browser"]:
        reports[platform] = []
        for label in ORDER:
            file = ROOT / f"{platform}-{label}.json.gz"
            raw = file.read_bytes()
            hashes[file.name] = hashlib.sha256(raw).hexdigest()
            report = json.loads(gzip.decompress(raw))
            if platform == "native":
                verify_native(report, requests, BASE if label.startswith("a") else CANDIDATE,
                              request_sha)
            else:
                require(report["status"] == "passed" and not report["page_errors"],
                        "browser failure")
                require(report["fixture_suite"] == "active-lanes"
                        and len(report["cases"]) == 89 and len(report["assertions"]) == 1920,
                        "incomplete browser fixture")
                require(all(c["status"] == "passed" for c in report["cases"]),
                        "incomplete browser case")
                expected_wasm = (
                    "5b4e9bf7cdab721217e7097c5e24a93ff8d7775c0be69b915aef7bd982079263"
                    if label.startswith("a") else
                    "54776ab68fc09c6b23d52855a2ffabcbc991086194d8b6b66e25405fe6473634"
                )
                require(report["wasm_sha256"] == expected_wasm, "wrong browser assets")
            reports[platform].append(report)
    require(len({r["page_sha256"] for r in reports["browser"]}) == 1,
            "browser harness changed")
    native_table, browser_table = [], []
    for request in requests[:81]:
        key = {k: request[k] for k in ["kind", "cols", "k", "tile"]}
        row = dict(key, native_us=[], cuda_us=[])
        for report in reports["native"]:
            cases = [c for c in report["cases"]
                     if all(c["request"][k] == v for k, v in key.items())]
            require(len(cases) == 3, "missing native seed")
            row["native_us"].append(statistics.median(
                finite_samples(c["native"]["samples_ms"]["resident_dispatch_fence_per_op"], 12)
                for c in cases))
            row["cuda_us"].append(statistics.median(
                finite_samples(c["timings"]["torch_resident_per_op"]["samples_ms"], 12)
                for c in cases))
        native_table.append(row)
        browser = dict(key, browser_us=[])
        for report in reports["browser"]:
            cases = [c for c in report["cases"] if not c.get("validation_only")
                     and all(c["tile_cols" if k == "tile" else k] == v for k, v in key.items())]
            require(len(cases) == 1, "missing browser case")
            browser["browser_us"].append(finite_samples(cases[0]["samples_ms"], 10))
        browser_table.append(browser)
    print(json.dumps({
        "schema": "spiraltorch.active_lane_comparison.v1", "order": ORDER,
        "baseline": BASE, "candidate": CANDIDATE, "request_sha256": request_sha,
        "boundary": "Native medians of three seed medians; browser single mixed-finite fixture medians. No pooled speedup, confidence interval, GPU-event or full-model claim.",
        "artifact_sha256": hashes, "native_cases_per_run": 243,
        "browser_cases_per_run": 89, "native_table": native_table, "browser_table": browser_table,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
