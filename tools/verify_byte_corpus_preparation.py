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
    version = study.request_version(request)
    require(version in (4, 5), "prepared request must be v4/v5")
    metrics = [study.FLAT, study.FISHER] if version == 5 else [study.FLAT]
    require(report["schema"] == f"spiraltorch.byte_corpus.bias_preparation.v{2 if version == 5 else 1}"
            and bool(report.get("adapter")), "preparation report")
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
    expected_keys = [(s, m) for s in seeds for m in metrics]
    require([(r["seed"], r.get("pair_metric", study.FLAT)) for r in report["cases"]] == expected_keys
            and all(type(r["seed"]) is int and ("pair_metric" in r if version == 5 else "pair_metric" not in r)
                    for r in report["cases"]), "calibration seed/metric coverage/order")
    expected_extras = []
    for metric in metrics:
        for case in source["cases"]:
            if case.get("pair_metric") != study.FLAT:
                continue
            candidate = copy.deepcopy(case)
            candidate["pair_metric"] = metric
            if metric == study.FISHER:
                candidate["name"] += "_fisher_rao"
            expected_extras.append(candidate)
    require(len(request["cases"][n:]) == len(expected_extras), "calibrated arm coverage")
    summaries, targets = [], {}
    for record in report["cases"]:
        seed = record["seed"]
        metric = record.get("pair_metric", study.FLAT)
        initial = next(c for c in expected_extras if c["seed"] == seed and c["pair_metric"] == metric and not study.frozen(c))
        fitted = next(c for c in request["cases"] if c["seed"] == seed and c.get("pair_metric") == metric and study.matched(c) and not study.frozen(c))
        for field, case in [("before_checkpoint_json", initial), ("fitted_checkpoint_json", fitted)]:
            stored = stored_parameters(study.decode_json(record[field]), 0, metric)
            require(same_parameters(stored, case["parameters"]), "prepared checkpoint is not the actual request initialization")
        old_gains = [p["values"] for p in initial["parameters"] if p["name"].startswith("geometry.raw_gain.")]
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
        require(record["reference_rms"] == targets.setdefault(seed, record["reference_rms"]),
                "metrics must share one Poincare RMS target per seed")
        require(len(old_gains) == len(gains) == len(cfg["blocks"])
                and all(len(row) == cfg["heads"] for row in old_gains + gains), "gain block/head coverage")
        for before, after, reference, candidate in zip(
                old_gains, gains, record["reference_rms"], record["candidate_rms"]):
            for old_gain, new_gain, target, original in zip(before, after, reference, candidate):
                old, new = (softplus(study.f32(value)) for value in (old_gain, new_gain))
                ratio = target / original
                require(math.isfinite(ratio) and ratio > 0. and old > 0.
                        and abs((new / old) / ratio - 1.) <= 1e-5,
                        "gain transformation differs from RMS scale")
        for a, b, stated in zip(record["reference_rms"], record["fitted_rms"], record["realized_relative_errors"]):
            for target, actual, reported_error in zip(a, b, stated):
                error = abs(actual / target - 1)
                require(error <= 1e-5 and abs(error - reported_error) <= 1e-14, "realized RMS gate or arithmetic")
                errors.append(error)
        summaries.append(dict(seed=seed, valid_pairs_per_head=count, max_relative_rms_error=max(errors),
                              reference_rms=record["reference_rms"], candidate_rms=record["candidate_rms"],
                              fitted_rms=record["fitted_rms"]))
        if version == 5:
            summaries[-1]["pair_metric"] = metric
    for actual, original in zip(request["cases"][n:], expected_extras):
        expected = copy.deepcopy(original)
        expected["name"] += "_rms_matched"
        expected["bias_initialization"] = study.MATCHED
        for before, after in zip(expected["parameters"], actual["parameters"]):
            if before["name"].startswith("geometry.raw_gain."):
                before["values"] = after["values"]
        require(exact(expected, actual), "calibration changed more than the permitted gain values")
    return dict(schema=f"spiraltorch.byte_corpus.preparation_verification.v{2 if version == 5 else 1}", passed=True,
                source_request_sha256=study.digest(source_raw), request_sha256=study.digest(request_raw),
                source_arms_unchanged=n, gain_only_transformation=True, initial_checkpoint_bits_bound=True,
                bias_calibration=spec, cases=summaries,
                scope="Artifact binding and reported RMS/gain arithmetic; actual-device qualification remains the Rust run, not independent attestation")


def softplus(value):
    return max(value, 0.) + math.log1p(math.exp(-abs(value)))


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
