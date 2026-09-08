#!/usr/bin/env python3
"""Recompute every retained N-D benchmark interval from the full gzip/JSON record.

This verifies internal source binding, fixture values, captures and aggregation;
it does not attest absent binary bytes or execute GPU kernels again.
"""
import argparse
import gzip
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_nd_tensor_vs_torch as bench


def validate_case(case, config, identity, index, pointwise=False):
    bench.validate_fixture(case["fixture"], config, identity)
    samples = case["intervals"]
    if len(samples) != 10:
        raise ValueError("missing or extra intervals")
    lanes = (
        ["sequential", "batched", "fused", "torch"] if pointwise else ["rust", "torch"]
    )
    for iteration, sample in enumerate(samples):
        order = (
            ["rust", "torch"] if (iteration + index + 1) % 2 == 0 else ["torch", "rust"]
        )
        if pointwise:
            offset = (iteration + index + 1) % 4
            order = lanes[offset:] + lanes[:offset]
            if (iteration // 4) % 2:
                order = list(reversed(order))
        if (
            type(sample.get("iteration")) is not int
            or sample["iteration"] != iteration
            or sample.get("warmup") is not (iteration < 2)
            or sample.get("order") != order
        ):
            raise ValueError("interval selection or ordering changed")
        for lane in lanes:
            bench.validate_sample(
                sample[lane], config["shape"], iteration == 2, lane != "torch"
            )
            if pointwise and lane != "torch" and sample[lane].get("execution") != lane:
                raise ValueError("execution mode differs")
    errors = {}
    for lane in lanes:
        if lane == "torch":
            continue
        pairs = list(zip(samples[2][lane]["values"], samples[2]["torch"]["values"]))
        if any(abs(a - b) > 1e-6 + 1e-4 * abs(b) for a, b in pairs):
            raise ValueError("captured values differ")
        errors[lane] = max(abs(a - b) for a, b in pairs)
    error = max(errors.values())
    if pointwise and case.get("lane_errors") != errors:
        raise ValueError("per-lane errors differ")
    if case.get("max_abs_error") != error:
        raise ValueError("published numeric error differs")
    medians = {
        lane: statistics.median(s[lane]["elapsed_ms"] for s in samples[2:])
        for lane in lanes
    }
    ratios = {
        lane: medians["torch"] / medians[lane] for lane in lanes if lane != "torch"
    }
    if case.get("median_ms") != medians or case.get("torch_over_rust") != (
        ratios if pointwise else ratios["rust"]
    ):
        raise ValueError("published median or ratio differs")
    return dict(
        config=config,
        median_ms=medians,
        rust_over_torch=(
            {lane: medians[lane] / medians["torch"] for lane in ratios}
            if pointwise
            else medians["rust"] / medians["torch"]
        ),
        max_abs_error=error,
        retained_per_lane=8,
    )


def validate(report):
    pointwise = report.get("schema") == "spiraltorch.nd_tensor.pointwise_bench.v1"
    if (
        (
            not pointwise
            and report.get("schema") != "spiraltorch.nd_tensor.torch_bench.v1"
        )
        or report.get("status") != "passed"
        or report.get("warmups") != 2
        or report.get("samples") != 8
        or report.get("fallback_enabled") is not False
    ):
        raise ValueError("wrong benchmark contract")
    if pointwise and report.get("lanes") != ["sequential", "batched", "fused", "torch"]:
        raise ValueError("benchmark lanes differ")
    binding = bench.audit.validate_source_binding(report["identity"], report["source"])
    if not binding["valid"] or report.get("source_binding") != binding:
        raise ValueError("source binding differs")
    if report["product"]["sha256"] != report["binary_sha256"]:
        raise ValueError("recorded binary hashes differ")
    configs = bench.recipes()
    if len(report["cases"]) != len(configs):
        raise ValueError("missing or extra recipes")
    result = []
    for index, (case, config) in enumerate(zip(report["cases"], configs)):
        if report["torch_device"] == "mps":
            bench.match_adapter(
                case["fixture"]["adapter"], "mps", report["device_admission"]["name"]
            )
        result.append(validate_case(case, config, report["identity"], index, pointwise))
    return dict(
        status="passed",
        cases=result,
        boundary="full record reaggregation and capture agreement, not independent GPU re-execution or absent-binary attestation",
    )


def load(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        raw = handle.read(64 * 1024 * 1024 + 1)
    if len(raw) > 64 * 1024 * 1024:
        raise ValueError("record exceeds decompressed fixture budget")
    return json.loads(raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(load(args.report)), indent=2, allow_nan=False))
