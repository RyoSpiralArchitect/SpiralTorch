#!/usr/bin/env python3
"""Independently recompute calibration statistics and verify resident learning."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

spec = importlib.util.spec_from_file_location("flat", Path(__file__).with_name("verify_byte_flat_metric.py"))
flat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flat)
require, bits, exact_record = flat.require, flat.bits, flat.exact_record
TOLERANCE = 1e-5


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), "finite statistic")
    return value


def close_stat(actual, expected):
    require(abs(finite(actual) - expected) <= 1e-12 * max(abs(expected), 1e-12), "reported statistic mismatch")


def matrix(value, blocks):
    require(type(value) is list and len(value) == blocks, "statistic block coverage")
    require(all(type(row) is list and len(row) == 2 for row in value), "statistic head coverage")
    for row in value:
        for v in row:
            finite(v)
    return value


def moments(scores, blocks):
    """Fixed [2 calibration batches, blocks, 2B*2H*4T*4T] observations."""
    require(type(scores) is list and len(scores) == 2, "calibration batch coverage")
    terms = [[[] for _ in range(2)] for _ in range(blocks)]
    for batch in scores:
        require(type(batch) is list and len(batch) == blocks, "score block coverage")
        for block, values in enumerate(batch):
            require(type(values) is list and len(values) == 64, "score shape")
            for v in values:
                bits(v)  # Validate masked entries too.
            for b in range(2):
                for h in range(2):
                    for q in range(4):
                        start = ((b * 2 + h) * 4 + q) * 4
                        row = values[start:start + q + 1]
                        mean = math.fsum(row) / len(row)
                        terms[block][h].extend((x - mean) ** 2 for x in row)
    return [[math.sqrt(math.fsum(head) / 40) for head in block] for block in terms]


def softplus(value):
    return max(value, 0.) + math.log1p(math.exp(-abs(value)))


def validate_record(case, record, blocks, *, runtime):
    exact_record(record["shape"], [2, 2, 4, 4])
    exact_record(record["windows"], [case["windows"], [[13, 19, 5, 2, 255], [1, 3, 15, 8, 254]]])
    require(type(record["valid_pairs_per_head"]) is int and record["valid_pairs_per_head"] == 40,
            "valid-pair count")
    require(type(record["relative_tolerance"]) is float and record["relative_tolerance"] == TOLERANCE,
            "changed calibration gate")
    stats = {}
    for prefix in ("reference", "candidate", "fitted"):
        stats[prefix] = moments(record[prefix + "_scores"], blocks)
        reported = matrix(record[prefix + "_rms"], blocks)
        for actual, computed in zip(reported, stats[prefix]):
            for a, e in zip(actual, computed):
                close_stat(a, e)
                require(e > 0, "degenerate signal")
    old, fitted = matrix(record["old_raw_gains"], blocks), matrix(record["raw_gains"], blocks)
    expected_old = [p["values"] for p in case["uncalibrated_parameters"] if p["name"].startswith("geometry.raw_gain.")]
    require(len(expected_old) == blocks, "old gain coverage")
    flat.exact(old, expected_old)
    scales, errors, sensitivity = [], [], []
    for block in range(blocks):
        scales.append([])
        for h in range(2):
            bits(fitted[block][h])
            desired = stats["reference"][block][h] / stats["candidate"][block][h]
            realized = softplus(fitted[block][h]) / softplus(old[block][h])
            require(abs(realized / desired - 1.) <= TOLERANCE, "softplus gain scale")
            error = abs(stats["fitted"][block][h] / stats["reference"][block][h] - 1.)
            require(error <= TOLERANCE, "realized device RMS gate")
            scales[-1].append(desired)
            errors.append(error)
            # Matching forward strength does not match the raw-gain Jacobian.
            relative_slope = lambda r: (1. / (1. + math.exp(-r))) / softplus(r)
            sensitivity.append(relative_slope(fitted[block][h]) / relative_slope(old[block][h]))
    require(any(abs(scale - 1.) > 1e-4 for row in scales for scale in row), "insensitive calibration control")
    if runtime:
        require(record["name"] == case["name"], "calibration case identity")
        require(all(record[key] is True for key in ("no_updates_consumed", "targets_and_external_bias_excluded",
                                                   "ordinary_and_out_of_range_absent",
                                                   "positive_control")), "calibration runtime controls")
        reported = matrix(record["requested_scales"], blocks)
        for a, e in zip(reported, scales):
            for x, y in zip(a, e):
                close_stat(x, y)
        require(type(record["realized_relative_errors"]) is list
                and len(record["realized_relative_errors"]) == 2 * blocks, "error coverage")
        for a, e in zip(record["realized_relative_errors"], errors):
            # Absolute 1e-12 only for the near-zero difference of two RMS ratios.
            require(abs(finite(a) - e) <= 1e-12, "reported RMS error")
        initial = [p["values"] for p in case["uncalibrated_parameters"]]
        calibrated = copy.deepcopy(initial)
        for block in range(blocks):
            calibrated[6 + block] = fitted[block]
        for key, parameters in (("before_checkpoint_json", initial), ("calibrated_checkpoint_json", calibrated)):
            cp = json.loads(record[key], object_pairs_hook=flat.unique_object)
            exact_record(cp, flat.expected_checkpoint(case, parameters, 0))
    return {"name": case["name"], "valid_pairs_per_head": 40,
            "reference_rms": stats["reference"], "candidate_rms": stats["candidate"], "fitted_rms": stats["fitted"],
            "requested_scales": scales, "maximum_realized_relative_error": max(errors),
            "raw_gain_relative_sensitivity_ratio": sensitivity}


def verify(reference, report):
    require(reference["schema"] == "spiraltorch.resident_byte_bias_scale.torch_fixture.v1", "reference schema")
    require(report["schema"] == "spiraltorch.resident_byte_bias_scale.validation.v1" and report["passed"] is True,
            "runtime schema/status")
    require(type(reference["cases"]) is list and type(report["calibration"]) is list
            and len(reference["cases"]) == len(report["calibration"]) == 2, "calibration case coverage")
    summaries = []
    for index, (case, actual) in enumerate(zip(reference["cases"], report["calibration"])):
        blocks = index + 1
        c = case["config"]
        require((c["batch"], c["heads"], c["steps"]) == (2, 2, 4)
                and len(c["blocks"]) == blocks, "frozen calibration topology")
        before = case["uncalibrated_parameters"]
        require(len(before) == len(case["parameters"]) == (23, 37)[index], "initial parameter coverage")
        # The Torch fit may change gains only. The learner's expected outputs
        # were generated from this independently fitted initialization.
        expected = copy.deepcopy(before)
        for block in range(blocks):
            expected[6 + block]["values"] = case["calibration"]["raw_gains"][block]
        exact_record(case["parameters"], expected)
        validate_record(case, case["calibration"], blocks, runtime=False)
        summary = validate_record(case, actual, blocks, runtime=True)
        learner_initial = json.loads(report["learning"]["checks"][index]["initial_checkpoint_json"],
                                     object_pairs_hook=flat.unique_object)
        fitted_checkpoint = json.loads(actual["calibrated_checkpoint_json"], object_pairs_hook=flat.unique_object)
        exact_record(learner_initial, fitted_checkpoint)
        score_error = 0.
        for key in ("reference_scores", "candidate_scores", "fitted_scores"):
            for a_batch, e_batch in zip(actual[key], case["calibration"][key]):
                for a, e in zip(a_batch, e_batch):
                    score_error = max(score_error, flat.compare(a, e))
        gain_error = max(flat.compare(a, e) for a, e in zip(actual["raw_gains"], case["calibration"]["raw_gains"]))
        summary.update({"score_max_abs_error_vs_torch": score_error, "raw_gain_max_abs_error_vs_torch": gain_error,
                        "checkpoint_gain_only_exact": True, "no_updates_consumed": True,
                        "ordinary_and_out_of_range_absent": True,
                        "learner_initial_checkpoint_exact": True,
                        "targets_and_external_bias_excluded": True})
        summaries.append(summary)
    compatible = {**reference, "schema": "spiraltorch.resident_byte_geometry_flat.torch_fixture.v1"}
    learning = flat.verify(compatible, report["learning"])
    return {"schema": "spiraltorch.resident_byte_bias_scale.comparison.v1", "passed": True,
            "calibration": summaries, "learning": learning,
            "scope": "Independent engine-local initialization fits; full synthetic VJP/16-step parity at fixed gates, "
                     "same-runtime exact resume. Not bit-identical cross-engine initialization, equal distributions, "
                     "equal learning dynamics, corpus quality, speed, or runtime attestation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "report", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    reference, report = args.reference.read_bytes(), args.report.read_bytes()
    result = verify(json.loads(reference, object_pairs_hook=flat.unique_object),
                    json.loads(report, object_pairs_hook=flat.unique_object))
    result["input_sha256"] = {"reference": hashlib.sha256(reference).hexdigest(), "report": hashlib.sha256(report).hexdigest()}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
