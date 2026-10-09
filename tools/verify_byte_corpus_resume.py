#!/usr/bin/env python3
"""Exact, within-runtime resume comparison; no speed or learning-quality claim."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

SPEC = importlib.util.spec_from_file_location("byte_corpus_study", Path(__file__).with_name("byte_corpus_study.py"))
STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STUDY)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def f32(value):
    require(type(value) in (int, float) and math.isfinite(value), "non-finite scalar")
    return struct.pack("!f", value)


def exact(left, right):
    """JSON value equality with strict types and signed-zero-sensitive f64 bits."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    if isinstance(left, float):
        return math.isfinite(left) and math.isfinite(right) and struct.pack("!d", left) == struct.pack("!d", right)
    return left == right


def stored_parameters(record, cursor):
    """Extract tensors in Rust's documented owner order; no model mathematics.

    Full topology/operation validation remains ByteCorpusStudy's Rust preflight.
    This verifier binds the saved parameter artifact to the paused readback.
    """
    require(set(record) == {"schema", "update_rule", "window_state", "attempted_revision", "model"},
            "incomplete model envelope")
    require(record["schema"] == "spiraltorch.nn.byte_decoder_checkpoint.v1", "model schema")
    require(record["update_rule"] == "stateless_sgd.v1", "model update rule")
    require(record["window_state"] == "reset_positions_and_geometry.v1", "model window state")
    require(record["attempted_revision"] == str(cursor), "model clock")
    model = record["model"]
    require(set(model) == {"token", "position", "geometry", "blocks", "head"}, "incomplete model")
    parameters = [model["token"], model["position"]]

    def graph(value):
        require(set(value) == {"schema", "input_shape", "parameters", "stages"}, "graph envelope")
        require(value["schema"] in {f"spiraltorch.nn.inference_plan.v{n}" for n in (2, 3, 4, 5)},
                "graph schema")
        parameters.extend(value["parameters"])

    geometry = model["geometry"]
    if geometry is not None:
        require(set(geometry) == {"projection", "curvature", "raw_decay", "raw_phase", "raw_gains"},
                "geometry envelope")
        graph(geometry["projection"])
        for values in [geometry["raw_decay"], geometry["raw_phase"], *geometry["raw_gains"]]:
            parameters.append({"shape": [len(values)], "values": values})
    for block in model["blocks"]:
        require(set(block) == {"pre", "heads", "qkv", "output", "feed_forward"}, "block envelope")
        for name in ("pre", "qkv", "output", "feed_forward"):
            graph(block[name])
    graph(model["head"])
    return parameters


def verify(request, request_hash, baseline, partial, resumed, checkpoint):
    version = STUDY.request_version(request)
    total = len(request["train_batches"])
    cursor = checkpoint["completed_updates"]
    require(type(cursor) is int and 0 <= cursor <= total, "invalid saved cursor")
    require(checkpoint["schema"] == f"spiraltorch.byte_corpus.checkpoint.v{version}", "checkpoint schema")
    require(checkpoint["request_sha256"] == request_hash, "checkpoint request identity")
    require(exact(resumed, baseline), "resumed report differs from uninterrupted report")
    require(baseline["schema"] == f"spiraltorch.byte_corpus.result.v{version}", "baseline is not complete")
    require(partial["adapter"] == baseline["adapter"], "partial runtime differs")
    require(partial["schema"] == (f"spiraltorch.byte_corpus.result.v{version}" if cursor == total
                                 else f"spiraltorch.byte_corpus.partial.v{version}"), "partial schema")
    for report in (baseline, partial, resumed):
        require(report["engine"] == "spiraltorch", "foreign engine")
        require(report["request_sha256"] == request_hash, "report request identity")
        require(len(report["cases"]) == len(request["cases"]), "missing report cases")
    require(len(checkpoint["cases"]) == len(request["cases"]), "missing checkpoint cases")
    summaries = []
    for case, full, prefix, resumed_case, saved in zip(request["cases"], baseline["cases"],
                                                    partial["cases"], resumed["cases"], checkpoint["cases"]):
        for report in (full, prefix, resumed_case):
            require(all(exact(report[key], case[key]) for key in ("name", "seed", "geometry")),
                    "case identity/order differs")
            if version == 2:
                require(case["geometry_update"] in {"train", "frozen"}
                        and report.get("geometry_update") == case["geometry_update"], "case update policy differs")
                trainable = sum(len(p["values"]) for p in case["parameters"]
                                if case["geometry_update"] != "frozen" or not p["name"].startswith("geometry."))
                require(exact(report.get("trainable_parameter_scalars"), trainable), "trainable count differs")
            require(exact(report["parameter_tensors"], len(case["parameters"])), "parameter count")
            require(exact(report["parameter_scalars"], sum(len(p["values"]) for p in case["parameters"])),
                    "parameter scalar count")
            require(len(report["final_parameters"]) == len(case["parameters"]), "missing weights")
            for values, parameter in zip(report["final_parameters"], case["parameters"]):
                require(len(values) == len(parameter["values"]), "weight shape")
                for value in values:
                    f32(value)
                if version == 2 and case["geometry_update"] == "frozen" and parameter["name"].startswith("geometry."):
                    require(list(map(f32, values)) == list(map(f32, parameter["values"])), "frozen weight bits differ")
        require(len(full["training"]) == total, "incomplete baseline training")
        require(exact([p["revision"] for p in full["training"]], list(range(1, total + 1))),
                "baseline revision sequence")
        for point in full["training"]:
            f32(point["ce"])
        evaluations = [n for n in range(total + 1)
                       if n % request["checkpoint_every"] == 0 or n == total]
        require(exact([p["revision"] for p in full["validation"]], evaluations), "evaluation schedule")
        for point in full["validation"]:
            require(len(point["batch_losses"]) == len(request["validation_batches"]), "evaluation coverage")
            require(exact(point["target_bytes"], request["config"]["batch"] * request["config"]["steps"]
                          * len(request["validation_batches"])), "evaluation target count")
            require(all(type(point[key]) in (int, float) and math.isfinite(point[key])
                        for key in ("mean_ce", "bits_per_byte")), "non-finite evaluation summary")
            for value in point["batch_losses"]:
                f32(value)
        require(exact(prefix["training"], full["training"][:cursor]), "training prefix differs")
        require(exact(prefix["validation"], [p for p in full["validation"] if p["revision"] <= cursor]),
                "evaluation prefix differs or extra pause evaluation")
        require(saved["name"] == case["name"], "saved case identity")
        model = STUDY.decode_json(saved["model_json"])
        stored = stored_parameters(model, cursor)
        require((model["model"]["geometry"] is not None) is case["geometry"], "saved geometry mode")
        require(len(stored) == len(case["parameters"]), "saved tensor count")
        for parameter, expected, readback in zip(stored, case["parameters"], prefix["final_parameters"]):
            require(exact(parameter["shape"], expected["shape"]), "saved tensor shape")
            require(list(map(f32, parameter["values"])) == list(map(f32, readback)),
                    "saved tensor bits differ from paused report")
        require(len(saved["training"]) == cursor, "saved training length")
        for point, expected in zip(saved["training"], prefix["training"]):
            require(exact(point["revision"], expected["revision"]) and f32(point["ce"]) == f32(expected["ce"]),
                    "saved training scalar differs")
        require(len(saved["validation"]) == len(prefix["validation"]), "saved evaluation length")
        for point, expected in zip(saved["validation"], prefix["validation"]):
            require(exact(point["revision"], expected["revision"]), "saved evaluation revision")
            require(list(map(f32, point["batch_losses"])) == list(map(f32, expected["batch_losses"])),
                    "saved evaluation scalar differs")
        summaries.append({"name": case["name"], "updates": total,
                          "pause": cursor, "evaluation_revisions": evaluations,
                          "final_mean_ce": full["validation"][-1]["mean_ce"]})
    return {"schema": f"spiraltorch.byte_corpus.resume_verification.v{version}", "passed": True,
            "request_sha256": request_hash, "exact_resumed_report_equal": True,
            "exact_training_and_evaluation_prefixes": True,
            "saved_parameter_bits_match_paused_report": True, "cases": summaries,
            "scope": "Within-runtime report equality and saved parameter/history consistency; full model validity belongs to Rust preflight, not runtime attestation, quality or speed evidence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "baseline", "partial", "resumed", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    names = ("request", "baseline", "partial", "resumed", "checkpoint")
    raw = {name: getattr(args, name).read_bytes() for name in names}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    data = {name: STUDY.decode_json(value) for name, value in raw.items()}
    result = verify(data["request"], hashes["request"], data["baseline"], data["partial"],
                    data["resumed"], data["checkpoint"])
    result["input_sha256"] = hashes
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Verified {len(result['cases'])} exact resumed cases")


if __name__ == "__main__":
    main()
