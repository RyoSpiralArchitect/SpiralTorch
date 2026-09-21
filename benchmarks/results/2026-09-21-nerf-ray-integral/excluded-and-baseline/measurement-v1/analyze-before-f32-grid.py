"""Numerically validate local reports and produce payload-free condition rows."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import struct
import sys

GRID = list(itertools.product([1, 32, 256], [1, 8, 64], [False, True]))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(values):
    assert all(math.isfinite(x) for x in values)
    return hashlib.sha256(struct.pack("<" + "f" * len(values), *values)).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate(report, reference):
    assert len(report["cases"]) == len(GRID)
    assert report["warmups"] == 5 and report["intervals"] == 9 and report["repetitions"] == 4
    rows = []
    for expected, case, ref in zip(GRID, report["cases"], reference["cases"]):
        meta = case.get("metadata", case)
        key = tuple(meta[k] for k in ["batch", "samples", "varying"])
        assert key == expected
        if "metadata" in case:
            assert case["metadata"] == ref["metadata"]
        values = case["values"]
        assert len(values) == key[0] * 3 == len(ref["values"])
        assert all(math.isfinite(v) for v in values)
        errors = [abs(a-b) for a,b in zip(values, ref["values"])]
        assert max(errors) <= 3e-6, (key, max(errors))
        timings = case["elapsed_ns"]
        assert len(timings) == 9 and all(math.isfinite(x) and x > 0 for x in timings)
        rows.append({"batch": key[0], "samples": key[1], "varying": key[2],
            "elapsed_ns": timings, "median_ns": statistics.median(timings),
            "output_f32le_sha256": tensor_digest(values),
            "input_metadata_sha256": canonical_digest(ref["metadata"]),
            "max_abs_vs_native_preflight": max(errors)})
    if "contract" in report:
        contract = report["contract"]
        assert contract["layout_equal"] is True
        assert len(contract["constant_cases"]) == 12
        constant_max_abs, constant_max_relative = 0.0, 0.0
        seen = []
        for case in contract["constant_cases"]:
            seen.append((case["samples"], case["width"]))
            assert len(case["actual"]) == 3
            width = struct.unpack("<f", struct.pack("<f", case["width"]))[0]
            for value, c in zip(case["actual"], [0.4, 0.2, 0.1]):
                c32 = struct.unpack("<f", struct.pack("<f", c))[0]
                expected = c32 * -math.expm1(-2 * width)
                error = abs(value - expected)
                assert math.isfinite(value) and error <= 2e-6 * abs(expected)
                constant_max_abs = max(constant_max_abs, error)
                constant_max_relative = max(constant_max_relative, error / abs(expected) if expected else 0.0)
        assert seen == list(itertools.product([1, 8, 64], [0.0, 1e-8, 1.0, 20.0]))
        training = contract["training"]
        ref_training = reference["contract"]["training"]
        assert training["before"] == ref_training["before"]
        assert training["after"].keys() == ref_training["after"].keys()
        parameter_error = 0.0
        for name, p in training["after"].items():
            assert p["shape"] == ref_training["after"][name]["shape"]
            actual, expected = p["values"], ref_training["after"][name]["values"]
            assert len(actual) == len(expected)
            assert all(math.isfinite(x) for x in actual)
            parameter_error = max(parameter_error, max(abs(a-b) for a,b in zip(actual, expected)))
        assert parameter_error <= 3e-6
        assert abs(training["loss"] - ref_training["loss"]) <= 3e-6
        assert abs(training["avg_transmittance"] - ref_training["avg_transmittance"]) <= 3e-6
        contracts = {"constant_max_abs": constant_max_abs, "constant_max_relative": constant_max_relative,
            "layout_equal": True, "one_step_parameter_max_abs_vs_native": parameter_error,
            "loss": training["loss"], "avg_transmittance": training["avg_transmittance"],
            "parameters_after_sha256": canonical_digest(training["after"])}
    else:
        training = report["training"]
        ref_training = reference["contract"]["training"]
        actual_error = 0.0
        assert training["updates"].keys() == ref_training["after"].keys()
        for name, values in training["updates"].items():
            expected = ref_training["after"][name]["values"]
            assert len(values) == len(expected) and all(math.isfinite(v) for v in values)
            actual_error = max(actual_error, max(abs(a-b) for a,b in zip(values, expected)))
        assert actual_error <= 3e-6
        assert abs(training["loss"] - ref_training["loss"]) <= 3e-6
        assert abs(training["transmittance"] - ref_training["avg_transmittance"]) <= 3e-6
        contracts = {"one_step_parameter_max_abs_vs_native": actual_error, "loss": training["loss"],
                     "avg_transmittance": training["transmittance"], "updates_sha256": canonical_digest(training["updates"])}
    return {"runtime": report["runtime"], "cases": rows, "contracts": contracts,
            "environment": {k: report[k] for k in ["node", "torch", "torch_file", "python", "platform", "threads", "interop_threads"] if k in report}}


if __name__ == "__main__":
    reference = json.loads(Path(sys.argv[1]).read_text())
    for path in map(Path, sys.argv[2:]):
        result = validate(json.loads(path.read_text()), reference)
        print(json.dumps({"path": str(path), "sha256": digest(path), "compact": result}))
