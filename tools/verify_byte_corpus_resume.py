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


def stored_parameters(record, cursor, pair_metric=None, head_pair_metrics=None, mixture=None, optimizer=None):
    """Extract tensors in Rust's documented owner order; no model mathematics.

    Full topology/operation validation remains ByteCorpusStudy's Rust preflight.
    This verifier binds the saved parameter artifact to the paused readback.
    """
    adamw = optimizer is not None and optimizer["kind"] == "adamw_zero_rate_freeze.v1"
    require(set(record) == ({"schema", "update_rule", "window_state", "attempted_revision", "model"}
                            | ({"model_schema", "optimizer"} if adamw else set())),
            "incomplete model envelope")
    metric_aware = pair_metric in (STUDY.FLAT, STUDY.FISHER)
    require(pair_metric in (None, STUDY.POINCARE, STUDY.FLAT, STUDY.FISHER), "saved metric identity")
    require(head_pair_metrics is None or pair_metric is None, "conflicting saved metric selectors")
    require(mixture is None or (head_pair_metrics is None and pair_metric is None), "conflicting saved mixture selector")
    query = mixture is not None and "query_projection" in mixture
    topology = f"spiraltorch.nn.byte_decoder_checkpoint.v{5 if query else 4 if mixture is not None else 3 if head_pair_metrics is not None else 2 if metric_aware else 1}"
    require(record["schema"] == ("spiraltorch.nn.byte_decoder_checkpoint.v6" if adamw else topology),
            "model schema")
    if adamw:
        require(record["model_schema"] == topology, "optimizer-wrapped topology")
    require(record["update_rule"] == ("adamw_zero_rate_freeze.v1" if adamw else "stateless_sgd.v1"),
            "model update rule")
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
    require(not (metric_aware or head_pair_metrics is not None or mixture is not None) or geometry is not None, "metric-aware model requires geometry")
    if geometry is not None:
        fields = {"projection", "curvature", "raw_decay", "raw_phase", "raw_gains"}
        if mixture is not None:
            fields.add("mixture")
            saved = geometry.get("mixture")
            require(type(saved) is dict and set(saved)==({"metrics","scales","raw_logits"} | ({"query_projections"} if query else set()))
                    and exact(saved["metrics"],mixture["metrics"])
                    and type(saved["scales"]) is list and len(saved["scales"])==len(mixture["scales"])
                    and all(list(map(f32,a))==list(map(f32,b)) for a,b in zip(saved["scales"],mixture["scales"])), "saved mixture topology differs")
            require(type(saved["raw_logits"]) is list and len(saved["raw_logits"])==len(geometry["raw_gains"]), "saved router block count")
        elif head_pair_metrics is not None:
            fields.add("head_pair_metrics")
            require(exact(geometry.get("head_pair_metrics"), head_pair_metrics), "saved headwise metric differs")
        elif metric_aware:
            fields.add("pair_metric")
            require(geometry.get("pair_metric") == pair_metric, "saved metric differs")
        require(set(geometry) == fields,
                "geometry envelope")
        graph(geometry["projection"])
        for values in [geometry["raw_decay"], geometry["raw_phase"], *geometry["raw_gains"]]:
            parameters.append({"shape": [len(values)], "values": values})
        if mixture is not None:
            for gains,values in zip(geometry["raw_gains"],geometry["mixture"]["raw_logits"]):
                require(type(values) is list and len(values)==len(gains)*len(mixture["metrics"]), "saved router shape")
                parameters.append({"shape":[len(gains),len(mixture["metrics"])],"values":values})
            if query:
                projections = geometry["mixture"]["query_projections"]
                require(type(projections) is list and len(projections) == len(geometry["raw_gains"]), "saved query block count")
                for projection in projections:
                    graph(projection)
    for block in model["blocks"]:
        require(set(block) == {"pre", "heads", "qkv", "output", "feed_forward"}, "block envelope")
        for name in ("pre", "qkv", "output", "feed_forward"):
            graph(block[name])
    graph(model["head"])
    return parameters


def verify(request, request_hash, baseline, partial, resumed, checkpoint):
    wire_version = STUDY.request_version(request)
    optimizer = STUDY.load_reference("byte_corpus_optimizer")
    controls = optimizer.recipe(request)
    version = STUDY.request_version(optimizer.design(request))
    total = len(request["train_batches"])
    cursor = checkpoint["completed_updates"]
    require(type(cursor) is int and 0 <= cursor <= total, "invalid saved cursor")
    require(checkpoint["schema"] == f"spiraltorch.byte_corpus.checkpoint.v{wire_version}", "checkpoint schema")
    require(checkpoint["request_sha256"] == request_hash, "checkpoint request identity")
    require(exact(resumed, baseline), "resumed report differs from uninterrupted report")
    if cursor == total:
        require(exact(partial, resumed), "completed checkpoint report differs from no-op resume")
    require(baseline["schema"] == f"spiraltorch.byte_corpus.result.v{wire_version}", "baseline is not complete")
    require(partial["adapter"] == baseline["adapter"], "partial runtime differs")
    require(partial["schema"] == (f"spiraltorch.byte_corpus.result.v{wire_version}" if cursor == total
                                 else f"spiraltorch.byte_corpus.partial.v{wire_version}"), "partial schema")
    for report in (baseline, partial, resumed):
        optimizer.metadata(request, report)
        require(report["engine"] == "spiraltorch", "foreign engine")
        require(report["request_sha256"] == request_hash, "report request identity")
        require(len(report["cases"]) == len(request["cases"]), "missing report cases")
        if version >= 4:
            require(exact(report.get("bias_calibration"), request["bias_calibration"]), "calibration recipe differs")
    require(len(checkpoint["cases"]) == len(request["cases"]), "missing checkpoint cases")
    summaries = []
    for case, full, prefix, resumed_case, saved in zip(request["cases"], baseline["cases"],
                                                    partial["cases"], resumed["cases"], checkpoint["cases"]):
        for report in (full, prefix, resumed_case):
            if wire_version == 9:
                require("optimizer_state" in report, "missing optimizer state")
                optimizer.validate_state(case, report["optimizer_state"], controls,
                                         cursor if report is prefix else total, STUDY.parameter_frozen)
            else:
                require("optimizer_state" not in report, "foreign optimizer state")
            require(all(exact(report[key], case[key]) for key in ("name", "seed", "geometry")),
                    "case identity/order differs")
            if version >= 2:
                require(case["geometry_update"] in {"train", "frozen"}
                        and report.get("geometry_update") == case["geometry_update"], "case update policy differs")
                trainable = STUDY.trainable_scalars(case)
                require(exact(report.get("trainable_parameter_scalars"), trainable), "trainable count differs")
            if "mixture" in case:
                require(exact(report.get("mixture"),case["mixture"]) and "pair_metric" not in report and "head_pair_metrics" not in report, "report mixture differs")
            elif "mixture" in report:
                raise ValueError("foreign mixture report")
            elif "head_pair_metrics" in case:
                require(exact(report.get("head_pair_metrics"), case["head_pair_metrics"]) and "pair_metric" not in report, "report headwise metric differs")
            elif version >= 3:
                require("head_pair_metrics" not in report, "unexpected report headwise metric")
                require("pair_metric" in report and exact(report["pair_metric"], case.get("pair_metric")), "report metric differs")
            if version >= 4:
                require(report.get("bias_initialization") == case["bias_initialization"], "initialization differs")
            require(exact(report["parameter_tensors"], len(case["parameters"])), "parameter count")
            require(exact(report["parameter_scalars"], sum(len(p["values"]) for p in case["parameters"])),
                    "parameter scalar count")
            require(len(report["final_parameters"]) == len(case["parameters"]), "missing weights")
            for values, parameter in zip(report["final_parameters"], case["parameters"]):
                require(len(values) == len(parameter["values"]), "weight shape")
                for value in values:
                    f32(value)
                if version >= 2 and STUDY.parameter_frozen(case,parameter["name"]):
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
        stored = stored_parameters(model, cursor, case.get("pair_metric"), case.get("head_pair_metrics"),
                                   case.get("mixture"), controls)
        if wire_version == 9:
            optimizer.validate_state(case, model.get("optimizer"), controls, cursor, STUDY.parameter_frozen)
            if controls["kind"] == optimizer.ADAMW:
                for a, b in zip(model["optimizer"]["slots"], prefix["optimizer_state"]["slots"]):
                    for field in ("first", "second"):
                        require(list(map(f32, a[field])) == list(map(f32, b[field])),
                                "saved optimizer moment bits differ from paused report")
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
        if version >= 3:
            summaries[-1]["geometry_update"] = case["geometry_update"]
            field = "mixture" if "mixture" in case else "head_pair_metrics" if "head_pair_metrics" in case else "pair_metric"
            summaries[-1][field] = case.get(field)
        if version >= 4:
            summaries[-1]["bias_initialization"] = case["bias_initialization"]
    result = {"schema": f"spiraltorch.byte_corpus.resume_verification.v{wire_version}", "passed": True,
            "request_sha256": request_hash, "exact_resumed_report_equal": True,
            "exact_training_and_evaluation_prefixes": True,
            "saved_parameter_bits_match_paused_report": True, "cases": summaries,
            "scope": "Within-runtime report equality and saved parameter/history consistency; full model validity belongs to Rust preflight, not runtime attestation, quality or speed evidence"}
    if wire_version == 9:
        result.update(design_schema=request["design_schema"], optimizer=controls,
                      saved_optimizer_bits_and_clocks_match=True)
    return result


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
