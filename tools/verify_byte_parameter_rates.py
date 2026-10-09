#!/usr/bin/env python3
"""Compare local frozen-geometry traces; no training implementation or quality claim."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bits(value):
    require(type(value) in (int, float) and math.isfinite(value), "non-finite numeric scalar")
    try:
        return struct.pack("!f", value)
    except OverflowError as error:
        raise ValueError("outside float32 range") from error


def compare_values(actual, expected):
    require(len(actual) == len(expected) and len(actual) > 0, "vector length")
    maximum = 0.
    for a, e in zip(actual, expected):
        bits(a)
        bits(e)
        delta = abs(a - e)
        require(delta <= 3e-6 + 5e-5 * abs(e), "absolute/relative numeric gate")
        maximum = max(maximum, delta)
    return maximum


def relative_gradient(actual, expected):
    compare_values(actual, expected)
    norm = math.sqrt(sum(e * e for e in expected))
    actual_norm = math.sqrt(sum(a * a for a in actual))
    require(norm > 1e-8 and actual_norm > 0, "inert geometry derivative")
    relative = math.sqrt(sum((a - e) ** 2 for a, e in zip(actual, expected))) / norm
    require(relative <= .002, "geometry relative L2 gate")
    return relative, norm


def verify(reference, report):
    require(reference["schema"] == "spiraltorch.resident_byte_geometry_frozen.torch_fixture.v1", "reference schema")
    require(reference["device"] == "cpu" and reference["dtype"] == "float32"
            and type(reference["threads"]) is int and reference["threads"] == 1, "reference execution")
    require(reference["tolerance"] == {"atol": 3e-6, "rtol": 5e-5, "geometry_relative_l2": .002}, "changed gates")
    require(report["schema"] == "spiraltorch.resident_byte_parameter_rates.validation.v1"
            and report["passed"] is True, "runtime schema/status")
    require(len(reference["cases"]) == len(report["checks"]) == 2, "case coverage")
    summaries = []
    for index, (case, check) in enumerate(zip(reference["cases"], report["checks"])):
        descriptors = case["parameters"]
        count = len(descriptors)
        require(count == (23, 37)[index] and type(check["parameter_count"]) is int
                and check["parameter_count"] == count, "parameter count")
        names = [p["name"] for p in descriptors]
        shapes = [p["shape"] for p in descriptors]
        require(check["name"] == case["name"] and check["parameter_names"] == names
                and check["parameter_shapes"] == shapes, "case/layout identity")
        frozen = [i for i, name in enumerate(names) if name.startswith("geometry.")]
        require(frozen == list(range(2, 7 + index)), "geometry coverage")
        require(check["geometry_slots"] == [frozen[0], frozen[-1] + 1], "geometry layout")
        rates = [0. if i in frozen else .125 for i in range(count)]
        learning = case["learning"]
        require(check["rates"] == learning["rates"] == rates
                and all(type(r) in (int, float) for r in check["rates"]), "rate policy")
        require(learning["steps"] == 16 and len(learning["trace"]) == len(check["trace"]) == 16, "step coverage")
        require(all(check[k] is True for k in ("invalid_rates_preserve_tape", "frozen_parameter_bits", "embedding_parameters_learn")), "missing runtime controls")
        controls = check["metric_controls"]
        require(controls["detached_embedding_gradient_separation"] > 1e-8
                and math.isfinite(controls["detached_embedding_gradient_separation"]), "insensitive detach control")
        contrast = controls["pullback_contrast_relative_l2"]
        require(type(contrast) in (int, float) and math.isfinite(contrast) and 0 <= contrast <= .002,
                "detached pullback contrast")
        errors = {"loss_max_abs_error": 0., "parameter_max_abs_error": 0.,
                  "geometry_gradient_relative_l2": 0., "embedding_gradient_max_abs_error": 0.}
        minimum_norm = math.inf
        for step, (expected, actual) in enumerate(zip(learning["trace"], check["trace"]), 1):
            require(type(actual["revision"]) is int and actual["revision"] == step, "revision sequence")
            errors["loss_max_abs_error"] = max(errors["loss_max_abs_error"], compare_values([actual["loss"]], [expected["loss"]]))
            require(len(actual["parameters"]) == len(expected["parameters"]) == count, "parameter coverage")
            for i, (a, e, descriptor) in enumerate(zip(actual["parameters"], expected["parameters"], descriptors)):
                require(len(a) == len(e) == len(descriptor["values"]) == math.prod(descriptor["shape"]), "parameter shape")
                errors["parameter_max_abs_error"] = max(errors["parameter_max_abs_error"], compare_values(a, e))
                if i in frozen:
                    initial_bits = list(map(bits, descriptor["values"]))
                    require(list(map(bits, a)) == list(map(bits, e)) == initial_bits, "frozen parameter bits changed")
                elif i < 2:
                    require(list(map(bits, a)) != list(map(bits, descriptor["values"])), "embedding did not learn")
            require(len(actual["geometry_gradients"]) == len(expected["geometry_gradients"]) == len(frozen), "gradient coverage")
            for i, a, e in zip(frozen, actual["geometry_gradients"], expected["geometry_gradients"]):
                require(len(a) == len(e) == len(descriptors[i]["values"]), "gradient shape")
                error, norm = relative_gradient(a, e)
                errors["geometry_gradient_relative_l2"] = max(errors["geometry_gradient_relative_l2"], error)
                minimum_norm = min(minimum_norm, norm)
            errors["embedding_gradient_max_abs_error"] = max(errors["embedding_gradient_max_abs_error"],
                compare_values(actual["embedding_output_gradient"], expected["embedding_output_gradient"]))
        summaries.append({"name": case["name"], "parameter_tensors": count, "steps": 16,
                          "frozen_geometry_tensors": len(frozen), "frozen_parameter_bits": True,
                          "minimum_reference_geometry_gradient_l2": minimum_norm, **errors})
    return {"schema": "spiraltorch.resident_byte_parameter_rates.comparison.v1", "passed": True,
            "cases": summaries, "scope": "Synthetic numerical learner parity, not performance, quality, or runtime attestation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "report", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    reference, report = args.reference.read_bytes(), args.report.read_bytes()
    result = verify(json.loads(reference), json.loads(report))
    result["input_sha256"] = {"reference": hashlib.sha256(reference).hexdigest(),
                              "report": hashlib.sha256(report).hexdigest()}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
