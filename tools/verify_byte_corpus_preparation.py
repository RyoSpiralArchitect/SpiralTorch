#!/usr/bin/env python3
"""Verify a Rust preparation packet against retained v3 bytes; never fit weights."""
import argparse
import copy
import importlib.util
import math
from pathlib import Path
import struct

def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = load("byte_corpus_study")
resume = load("verify_byte_corpus_resume")
exact, stored_parameters = resume.exact, resume.stored_parameters


def require(ok, message):
    if not ok:
        raise ValueError(message)


def bits(values):
    require(type(values) is list and all(type(value) in (float, int) and math.isfinite(value) for value in values),
            "invalid checkpoint scalar type")
    return [struct.pack("<f", value) for value in values]


def same_parameters(left, right):
    for parameter in [*left, *right]:
        shape = parameter["shape"]
        require(type(shape) is list and bool(shape) and all(type(d) is int and d > 0 for d in shape)
                and math.prod(shape) == len(parameter["values"]), "invalid checkpoint shape type or size")
    return len(left) == len(right) and all(exact(a["shape"], b["shape"]) and bits(a["values"]) == bits(b["values"])
                                             for a, b in zip(left, right))


def verify(source_raw, packet):
    source = study.decode_json(source_raw)
    require(study.request_version(source) == 3, "preparation source must be v3")
    require(set(packet) == {"schema", "request_json", "report_json"}
            and packet["schema"] == "spiraltorch.byte_corpus.prepared.v1"
            and type(packet["request_json"]) is str and type(packet["report_json"]) is str, "preparation envelope")
    request_raw = packet["request_json"].encode()
    request = study.decode_json(request_raw)
    report = study.decode_json(packet["report_json"])
    require(study.request_version(request) == 4, "prepared request must be v4")
    require(report["schema"] == "spiraltorch.byte_corpus.bias_preparation.v1" and bool(report.get("adapter")), "preparation report")
    spec = request["bias_calibration"]
    require(exact(report["bias_calibration"], spec) and spec["source_request_sha256"] == study.digest(source_raw)
            and report["request_sha256"] == study.digest(request_raw), "source/prepared byte identity")
    header = copy.deepcopy(request)
    header.pop("bias_calibration")
    header["schema"] = source["schema"]
    header["cases"] = source["cases"]
    require(exact(header, source), "preparation changed data, selection, rate or configuration")
    n = len(source["cases"])
    for actual, expected in zip(request["cases"][:n], source["cases"]):
        original = copy.deepcopy(actual)
        require(original.pop("bias_initialization") == study.ORIGINAL, "source initialization relabeled")
        require(exact(original, expected), "source arm changed during preparation")
    seeds = sorted({c["seed"] for c in source["cases"]})
    require([r["seed"] for r in report["cases"]] == seeds, "calibration seed coverage/order")
    expected_extras = [c for c in source["cases"] if c.get("pair_metric") == study.FLAT]
    require(len(request["cases"][n:]) == len(expected_extras), "calibrated arm coverage")
    summaries = []
    for record in report["cases"]:
        seed = record["seed"]
        initial = next(c for c in expected_extras if c["seed"] == seed and not study.frozen(c))
        fitted = next(c for c in request["cases"] if c["seed"] == seed and study.matched(c) and not study.frozen(c))
        for field, case in [("before_checkpoint_json", initial), ("fitted_checkpoint_json", fitted)]:
            stored = stored_parameters(study.decode_json(record[field]), 0, study.FLAT)
            require(same_parameters(stored, case["parameters"]), "prepared checkpoint is not the actual request initialization")
        gains = [p["values"] for p in fitted["parameters"] if p["name"].startswith("geometry.raw_gain.")]
        require(len(gains) == len(record["raw_gains"]) and all(bits(a) == bits(b) for a, b in zip(gains, record["raw_gains"])), "fitted gain binding")
        cfg = source["config"]
        count = len(spec["train_batch_indices"]) * cfg["batch"] * cfg["steps"] * (cfg["steps"] + 1) // 2
        require(type(record["valid_pairs_per_head"]) is int and record["valid_pairs_per_head"] == count
                and record["no_updates_consumed"] is True, "calibration count or update ownership")
        errors = []
        for field in ("reference_rms", "candidate_rms", "fitted_rms", "realized_relative_errors"):
            value = record[field]
            require(len(value) == len(cfg["blocks"]) and all(len(row) == cfg["heads"] for row in value), "block/head coverage")
            require(all(type(x) in (float, int) and math.isfinite(x) and (x >= 0 if field == "realized_relative_errors" else x > 0)
                        for row in value for x in row), "nonfinite or zero calibration statistic")
        for a, b, stated in zip(record["reference_rms"], record["fitted_rms"], record["realized_relative_errors"]):
            for target, actual, reported_error in zip(a, b, stated):
                error = abs(actual / target - 1)
                require(error <= 1e-5 and abs(error - reported_error) <= 1e-14, "realized RMS gate or arithmetic")
                errors.append(error)
        summaries.append(dict(seed=seed, valid_pairs_per_head=count, max_relative_rms_error=max(errors),
                              reference_rms=record["reference_rms"], candidate_rms=record["candidate_rms"],
                              fitted_rms=record["fitted_rms"]))
    for actual, original in zip(request["cases"][n:], expected_extras):
        expected = copy.deepcopy(original)
        expected["name"] += "_rms_matched"
        expected["bias_initialization"] = study.MATCHED
        for before, after in zip(expected["parameters"], actual["parameters"]):
            if before["name"].startswith("geometry.raw_gain."):
                before["values"] = after["values"]
        require(exact(expected, actual), "calibration changed more than the permitted gain values")
    return dict(schema="spiraltorch.byte_corpus.preparation_verification.v1", passed=True,
                source_request_sha256=study.digest(source_raw), request_sha256=study.digest(request_raw),
                source_arms_unchanged=n, gain_only_transformation=True, initial_checkpoint_bits_bound=True,
                bias_calibration=spec, cases=summaries,
                scope="Artifact binding and reported RMS arithmetic; actual-device qualification remains the Rust run, not independent attestation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "packet", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    packet_raw = args.packet.read_bytes()
    packet = study.decode_json(packet_raw)
    result = verify(args.source.read_bytes(), packet)
    result["packet_sha256"] = study.digest(packet_raw)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, key in [("request.json", "request_json"), ("preparation.json", "report_json")]:
        with (args.output / name).open("xb") as stream:
            stream.write(packet[key].encode())
    study.write_new(args.output / "verification.json", result)
    print(study.encoded(result).decode(), end="")


if __name__ == "__main__":
    main()
