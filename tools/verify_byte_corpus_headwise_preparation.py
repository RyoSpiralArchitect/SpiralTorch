#!/usr/bin/env python3
"""Bind a Rust-prepared headwise packet to unchanged v5 bytes and device RMS."""
import argparse
import importlib.util
import math
from pathlib import Path
import struct


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study, resume = load("byte_corpus_study"), load("verify_byte_corpus_resume")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify_topology(config, record):
    """Check the bounded study recipe, not arbitrary model execution semantics."""
    model = record["model"]
    layout = [config[k] for k in ("batch", "steps", "width")]
    require(resume.f32(model["geometry"]["curvature"]) == resume.f32(config["curvature"]), "checkpoint curvature")
    require(len(model["blocks"]) == len(config["blocks"]), "checkpoint block count")
    linear = lambda offset, gelu=False: dict(kind="linear", weight=offset, bias=offset+1, gelu=gelu)
    norm = dict(kind="layer_norm", gain=0, bias=1, epsilon=1e-5)

    def graph(actual, version, stages, roles):
        require(actual["schema"] == f"spiraltorch.nn.inference_plan.v{version}"
                and resume.exact(actual["input_shape"], layout), "checkpoint graph schema/layout")
        require(len(actual["stages"]) == len(stages), "checkpoint graph stage count")
        for a, e in zip(actual["stages"], stages):
            require(set(a) == set(e), "checkpoint graph stage fields")
            for key, value in e.items():
                require(resume.f32(a[key]) == resume.f32(value) if type(value) is float
                        else resume.exact(a[key], value), "checkpoint graph operation")
        require([p.get("role") for p in actual["parameters"]] == roles, "checkpoint graph parameter roles")

    graph(model["geometry"]["projection"], 2, [linear(0)], ["weight", "bias"])
    for actual, topos in zip(model["blocks"], config["blocks"]):
        require(resume.exact(actual["heads"], config["heads"]), "checkpoint head count")
        graph(actual["pre"], 3, [norm], ["gain", "bias"])
        for key in ("qkv", "output"):
            graph(actual[key], 2, [linear(0)], ["weight", "bias"])
        feed, roles = [norm, linear(2, True)], ["gain", "bias", "weight", "bias"]
        if topos:
            feed.append(dict(kind="topos_resonator",gate=4,coupling=.2,saturation=.12,porosity=.3,
                             iterations=4,max_volume=config["batch"]*config["steps"]*config["hidden"]))
            roles.append("gate")
        feed.append(linear(5 if topos else 4))
        graph(actual["feed_forward"], 5 if topos else 3, feed, roles+["weight", "bias"])
    graph(model["head"], 3, [norm, linear(2)], ["gain", "bias", "weight", "bias"])


def verify(source_raw, packet):
    require(type(packet) is dict and set(packet) == {"schema", "request_json", "report_json"}
            and packet["schema"] == "spiraltorch.byte_corpus.prepared.v1", "preparation envelope")
    source = study.decode_json(source_raw)
    request_raw = packet["request_json"].encode()
    request = study.decode_json(request_raw)
    record = study.decode_json(packet["report_json"])
    require(study.request_version(source) == 5 and study.request_version(request) == 6, "preparation schemas")
    require(request["headwise_controls"]["source_request_sha256"] == study.digest(source_raw), "source hash")
    require(set(request) == set(source) | {"headwise_controls"}, "source request fields")
    for key in source:
        if key not in ("schema", "cases"):
            require(resume.exact(source[key], request[key]), "source recipe changed")
    # JSON reserialization can turn an integer into an equivalent float. The
    # selected cases must keep both descriptors and uploaded float32 bits.
    def initial(case):
        copy = dict(case)
        copy["parameters"] = [(p["name"], p["shape"], [struct.pack("<f", v).hex() for v in p["values"]]) for p in case["parameters"]]
        return copy
    require(len(request["cases"]) == len(source["cases"]) * 21 // 9, "expanded coverage")
    require([initial(c) for c in request["cases"][:len(source["cases"])]] == [initial(c) for c in source["cases"]], "inherited initial bits")
    require(record["schema"] == "spiraltorch.byte_corpus.headwise_preparation.v1"
            and record["source_request_sha256"] == study.digest(source_raw)
            and record["request_sha256"] == study.digest(request_raw), "preparation identity")
    require(record.get("adapter") and all(record.get(k) is True for k in ("passed", "no_fitting", "no_updates_consumed")), "preparation state")
    require(resume.exact(record["train_batch_indices"], request["bias_calibration"]["train_batch_indices"])
            and type(record["relative_tolerance"]) in (float, int) and record["relative_tolerance"] == 1e-5, "frozen RMS recipe")
    expected = [c for seed in sorted({c["seed"] for c in request["cases"]}) for c in request["cases"]
                if c["seed"] == seed and "head_pair_metrics" in c and c["geometry_update"] == "train"]
    require(len(record["cases"]) == len(expected), "RMS route coverage")
    cfg = request["config"]
    count = len(record["train_batch_indices"]) * cfg["batch"] * cfg["steps"] * (cfg["steps"] + 1) // 2
    targets, summaries = {}, []
    for row, case in zip(record["cases"], expected):
        for key in ("name", "seed", "head_pair_metrics"):
            require(resume.exact(row[key], case[key]), "RMS case identity")
        model = study.decode_json(row["model_json"])
        parameters = resume.stored_parameters(model, 0, head_pair_metrics=case["head_pair_metrics"])
        verify_topology(cfg, model)
        require(len(parameters) == len(case["parameters"]), "checkpoint tensor coverage")
        for p, original in zip(parameters, case["parameters"]):
            require(resume.exact(p["shape"], original["shape"])
                    and list(map(resume.f32, p["values"])) == list(map(resume.f32, original["values"])), "initial checkpoint bits")
        gains = next(p["values"] for p in case["parameters"] if p["name"] == "geometry.raw_gain.0")
        require(list(map(resume.f32, row["raw_gains"])) == list(map(resume.f32, gains)), "selected gain bits")
        require(resume.exact(row["valid_pairs_per_head"], [count]), "RMS pair coverage")
        for key in ("target_rms", "actual_rms", "relative_errors"):
            require(type(row[key]) is list and len(row[key]) == 1 and type(row[key][0]) is list
                    and len(row[key][0]) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in row[key][0]), "RMS head coverage")
        require(row["target_rms"] == targets.setdefault(case["seed"], row["target_rms"]), "shared RMS target")
        for a, b, claimed in zip(row["target_rms"][0], row["actual_rms"][0], row["relative_errors"][0]):
            require(a > 0 and b > 0, "positive RMS")
            error = abs(b/a - 1)
            require(error <= 1e-5 and abs(error-claimed) <= 1e-14, "RMS arithmetic or gate")
        summaries.append({k:v for k,v in row.items() if k != "model_json" and k != "raw_gains"})
    return dict(schema="spiraltorch.byte_corpus.headwise_preparation_verification.v1", passed=True,
                source_request_sha256=study.digest(source_raw), request_sha256=study.digest(request_raw),
                inherited_cases=len(source["cases"]), all_initial_parameter_bits_checked=True,
                no_fitting=True, no_updates_consumed=True, cases=summaries,
                scope="Request/checkpoint binding and measured device RMS, not runtime attestation or quality evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "packet", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = verify(args.source.read_bytes(), study.decode_json(args.packet.read_bytes()))
    study.write_new(args.output, result)
    print(f"Verified {len(result['cases'])} headwise preparations")


if __name__ == "__main__":
    main()
