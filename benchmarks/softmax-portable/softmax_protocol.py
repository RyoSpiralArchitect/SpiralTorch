"""Finite-domain admission and complete, descriptive three-round comparisons."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

SCHEMA = "spiraltorch.softmax_portable_bench.v1"
TORCH = "spiraltorch.softmax_torch_bench.v1"
SUMMARY = "spiraltorch.softmax_portable_summary.v1"
COMPARISON = "redundant_vs_deduplicated_barriers"
KEYS = {(r, c, m) for r, c in [(1, 31), (1, 256), (17, 257), (65, 1025),
                                (128, 1025), (17, 4096)] for m in (0, 4)}
ATOL, RTOL = 2e-7, 2e-6
ROUTES = ("native_redundant", "native_deduplicated", "browser_redundant",
          "browser_deduplicated", "cpu", "mps")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def f32(values):
    if any(not finite(v) for v in values):
        raise ValueError("nonfinite or nonnumeric array")
    return struct.pack("<" + "f" * len(values), *values)


def key(case):
    result = tuple(case[k] for k in ("rows", "cols", "mode"))
    if any(type(v) is not int for v in result) or result not in KEYS:
        raise ValueError("unknown shape/mode")
    return result


def inputs(rows, cols):
    return [((i * 37 + 17) % 1009 - 504) / 128 for i in range(rows * cols)]


def oracle(values, rows, cols, mode):
    probabilities, peaks = [], []
    for row in range(rows):
        x = values[row * cols:(row + 1) * cols]
        top = max(x)
        exps = [math.exp(v - top) for v in x]
        total = math.fsum(exps)
        probabilities.extend(v / total for v in exps)
        if mode == 4:
            peaks.extend(float(v == top) for v in x)
    result = probabilities + peaks
    return list(struct.unpack("<" + "f" * len(result), f32(result)))


def close(actual, reference):
    if len(actual) != len(reference):
        raise ValueError("output length")
    maximum, scaled = 0., 0.
    for a, b in zip(actual, reference):
        if not finite(a) or not finite(b):
            raise ValueError("nonfinite output")
        error = abs(a - b)
        relative = error / (ATOL + RTOL * abs(b))
        if relative > 1:
            raise ValueError("softmax oracle mismatch")
        maximum, scaled = max(maximum, error), max(scaled, relative)
    return maximum, scaled


def intervals(case, routes, rounds=False):
    expected = {(r, b, n, d) for r in (range(3) if rounds else [0])
                for b in range(9) for n in (1, 4) for d in routes}
    index = {}
    for item in case["intervals"]:
        r, b, n, d = item.get("round", 0), item["block"], item["burst"], item["route"]
        if any(type(v) is not int for v in (r, b, n)):
            raise ValueError("noninteger interval identity")
        ident = r, b, n, d
        if ident not in expected or ident in index:
            raise ValueError("unknown/duplicate interval")
        if not finite(item["elapsed_ms"]) or item["elapsed_ms"] <= 0:
            raise ValueError("invalid interval duration")
        error = item["max_abs_error"]
        if not finite(error) or not 0 <= error <= ATOL + RTOL:
            raise ValueError("invalid interval error")
        parity = (b + 3 + case["rows"] + case["cols"] + case["mode"]) % 2
        order = [0, 1] if parity == 0 else [1, 0]
        if item["order"] != order or any(type(i) is not int for i in item["order"]):
            raise ValueError("paired route order")
        index[ident] = item["elapsed_ms"]
    if set(index) != expected:
        raise ValueError("incomplete intervals")
    return index


def admit(report, family):
    is_torch = family == "torch"
    if family not in ("native", "browser", "torch"):
        raise ValueError("unknown runtime")
    if (report["schema"] != (TORCH if is_torch else SCHEMA)
            or report["status"] != "passed" or report["warmup"] != 3
            or report["blocks"] != 9 or report["bursts"] != [1, 4]):
        raise ValueError("protocol identity")
    if is_torch:
        if (report["input_protocol"] != SCHEMA or report["devices"] != ["cpu", "mps"]
                or report["compiled"] is not False or report["preallocated_outputs"] is not True
                or report["intra_op_threads"] != 4 or report["inter_op_threads"] != 1):
            raise ValueError("Torch control identity")
    else:
        if (report["comparison"] != COMPARISON or report["finite_domain_fix_in_both"] is not True
                or report["domain_cases"] != 48 or type(report["subgroup_exercised"]) is not bool
                or not report["adapter"] or "device_type: Cpu" in report["adapter"]):
            raise ValueError("WGPU domain admission")
    if family == "browser":
        if (report["page_errors"] or report["console_messages"]
                or report["browser_adapter_probe"]["is_fallback_adapter"] is not False
                or not report["asset_sha256"] or not report["browser_version"]):
            raise ValueError("browser runtime/probe admission")
    routes = ("cpu", "mps") if is_torch else ("redundant", "deduplicated")
    cases = {}
    for case in report["cases"]:
        ident = key(case)
        if ident in cases:
            raise ValueError("duplicate condition")
        rows, cols, mode = ident
        expected_input = inputs(rows, cols)
        if f32(case["input"]) != f32(expected_input):
            raise ValueError("input bytes drift")
        reference = oracle(expected_input, rows, cols, mode)
        close(case["reference"], reference)
        if len(case["last_outputs"]) != 2:
            raise ValueError("route outputs")
        for output in case["last_outputs"]:
            close(output, reference)
        intervals(case, routes)
        cases[ident] = case
    if set(cases) != KEYS:
        raise ValueError("incomplete grid")
    return cases


def summarize(case):
    index = intervals(case, ROUTES, rounds=True)
    result = []
    for burst in (1, 4):
        medians = {d: statistics.median(v for (_, _, n, route), v in index.items()
                                       if n == burst and route == d) for d in ROUTES}
        ratios = {family: statistics.median(
            index[r, b, burst, family + "_redundant"] / index[r, b, burst, family + "_deduplicated"]
            for r in range(3) for b in range(9)) for family in ("native", "browser")}
        result.append({"burst": burst, "median_interval_ms": medians,
                       "paired_redundant_over_deduplicated": ratios})
    return result


def describe(cases):
    result = {}
    for family in ("native", "browser"):
        values = [s["paired_redundant_over_deduplicated"][family]
                  for c in cases for s in c["summary"]]
        result[family] = {"geomean_redundant_over_deduplicated":
                         math.exp(statistics.mean(math.log(v) for v in values)),
                         "min": min(values), "max": max(values),
                         "cells_over_1": sum(v > 1 for v in values), "cells": len(values)}
    return result


def analyze(native, browser, torch):
    reports = [native, browser, torch]
    if any(len(group) != 3 for group in reports):
        raise ValueError("exactly three complete rounds required")
    admitted = [[admit(r, family) for r in group]
                for family, group in zip(("native", "browser", "torch"), reports)]
    records = []
    for ident in sorted(KEYS):
        rows, cols, mode = ident
        reference = oracle(inputs(rows, cols), rows, cols, mode)
        record = dict(rows=rows, cols=cols, mode=mode, intervals=[],
                      input_sha256=hashlib.sha256(f32(inputs(rows, cols))).hexdigest(),
                      oracle_sha256=hashlib.sha256(f32(reference)).hexdigest(),
                      max_abs_errors={}, max_scaled_errors={})
        for family, rounds in zip(("native", "browser", "torch"), admitted):
            route_names = ("cpu", "mps") if family == "torch" else (
                family + "_redundant", family + "_deduplicated")
            for repeat, cases in enumerate(rounds):
                case = cases[ident]
                for route, output in zip(route_names, case["last_outputs"]):
                    absolute, scaled = close(output, reference)
                    for field, value in [("max_abs_errors", absolute), ("max_scaled_errors", scaled)]:
                        record[field][route] = max(record[field].get(route, 0), value)
                for item in case["intervals"]:
                    route = item["route"] if family == "torch" else family + "_" + item["route"]
                    record["intervals"].append({**item, "route": route, "round": repeat})
        record["summary"] = summarize(record)
        records.append(record)
    return {"schema": SUMMARY, "status": "passed", "comparison": COMPARISON, "rounds": 3,
            "tolerance": {"atol": ATOL, "rtol": RTOL}, "cases": records,
            "descriptive_summary": describe(records),
            "native_metadata": [{k: r[k] for k in ("adapter", "domain_cases", "subgroup_exercised")} for r in native],
            "browser_metadata": [{k: r[k] for k in ("adapter", "domain_cases", "subgroup_exercised",
                                 "browser_adapter_probe", "asset_sha256", "browser_version")} for r in browser],
            "torch_metadata": [{k: r[k] for k in ("torch_version", "devices", "intra_op_threads",
                               "inter_op_threads", "compiled", "preallocated_outputs")} for r in torch],
            "boundary": "Same finite-corrected workgroup shader and explicit layout in both WGPU routes; only two entry barriers differ. Prepared f32 inputs/outputs, per-operation dispatch and one final owning CPU copy/map/completion per burst1/4. Torch eager CPU/MPS uses reusable output/scratch tensors, softmax out= and all-tied-peak masks, not one-hot argmax. Setup, oracle, checks, list/JSON conversion excluded except Rust f32 Vec construction. Independent scalar f64 oracle and Torch CPU f64 cross-check. Three rotated serial runtime rounds on shared M4, coarse browser clock; descriptive application intervals, not kernel-only, torch.compile, training, quality, or universal PyTorch superiority."}


def validate_summary(result):
    if (result["schema"] != SUMMARY or result["status"] != "passed"
            or result["comparison"] != COMPARISON or result["rounds"] != 3
            or result["tolerance"] != {"atol": ATOL, "rtol": RTOL}):
        raise ValueError("summary identity")
    seen = set()
    for case in result["cases"]:
        ident = key(case)
        if ident in seen:
            raise ValueError("duplicate summary condition")
        seen.add(ident)
        for field in ("input_sha256", "oracle_sha256"):
            value = case[field]
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("summary content hash")
        for field, bound in (("max_abs_errors", ATOL + RTOL), ("max_scaled_errors", 1.)):
            if set(case[field]) != set(ROUTES) or any(
                    not finite(v) or not 0 <= v <= bound for v in case[field].values()):
                raise ValueError("summary numeric gate")
        if case["summary"] != summarize(case):
            raise ValueError("summary does not match all intervals")
    if seen != KEYS or result["descriptive_summary"] != describe(result["cases"]):
        raise ValueError("summary grid/aggregation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for family in ("native", "browser", "torch"):
        parser.add_argument("--" + family, action="append", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(*[[json.loads(p.read_text()) for p in getattr(args, family)]
                       for family in ("native", "browser", "torch")])
    validate_summary(result)
    print(json.dumps(result, allow_nan=False))
