#!/usr/bin/env python3
"""Independently compare local flat-metric observations, including exact resume."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

spec = importlib.util.spec_from_file_location("rates", Path(__file__).with_name("verify_byte_parameter_rates.py"))
rates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rates)
require, bits, compare, relative = rates.require, rates.bits, rates.compare_values, rates.relative_gradient
METRIC = "euclidean_chord_squared.v1"
METRICS = {"flat": METRIC, "fisher_rao": "categorical_fisher_rao_squared.v1"}
HEADWISE = [
    [["poincare_squared.v1", "poincare_squared.v1", "categorical_fisher_rao_squared.v1", "poincare_squared.v1"]],
    [["categorical_fisher_rao_squared.v1", "euclidean_chord_squared.v1"],
     ["euclidean_chord_squared.v1", "poincare_squared.v1"]],
]
MIXTURE_METRICS = ["poincare_squared.v1", "euclidean_chord_squared.v1", "categorical_fisher_rao_squared.v1"]
MIXTURE_SCALES = [[1., .8, 24., 1., 1.2, 48.], [.7, 1.5, 36., 1.3, .6, 40.]]


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def exact_record(actual, expected):
    if type(expected) is dict:
        require(type(actual) is dict and actual.keys() == expected.keys(), "checkpoint fields")
        for key, value in expected.items():
            exact_record(actual[key], value)
    elif type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), "checkpoint shape")
        for a, e in zip(actual, expected):
            exact_record(a, e)
    elif type(expected) is float:
        require(bits(actual) == bits(expected), "checkpoint float32 bits")
    else:
        require(type(actual) is type(expected) and actual == expected, "checkpoint identity")


def expected_checkpoint(case, parameters, revision, metric=METRIC):
    """The two frozen fixture topologies, not a general model serializer."""
    c = case["config"]
    descriptors = case["parameters"]
    names = [p["name"] for p in descriptors]
    require(len(set(names)) == len(names) == len(parameters), "checkpoint parameter coverage")
    used = []

    def parameter(name, shape, role=None):
        require(name in names, "checkpoint parameter name")
        i = names.index(name)
        require(descriptors[i]["shape"] == shape and len(parameters[i]) == math.prod(shape), "checkpoint layout")
        for value in parameters[i]:
            bits(value)
        used.append(name)
        record = {"shape": shape, "values": [float(v) for v in parameters[i]]}
        return {"role": role, **record} if role else record

    def graph(parameters, stages, version):
        return {"schema": f"spiraltorch.nn.inference_plan.v{version}",
                "input_shape": [c["batch"], c["steps"], c["width"]],
                "parameters": parameters, "stages": stages}

    def linear(weight=0, bias=1, gelu=False):
        return {"kind": "linear", "weight": weight, "bias": bias, "gelu": gelu}

    def norm():
        return {"kind": "layer_norm", "gain": 0, "bias": 1, "epsilon": 1e-5}

    width, hidden, cols = c["width"], c["hidden"], c["causal_geometry"]["cols"]
    token = parameter("token_embedding", [256, width])
    position = parameter("position_embedding", [c["position_capacity"], width])
    projection = graph([parameter("geometry.projection.weight", [width, cols], "weight"),
                        parameter("geometry.projection.bias", [cols], "bias")], [linear()], 2)
    geometry = {"projection": projection, "curvature": float(c["causal_geometry"]["curvature"]),
                "raw_decay": parameter("geometry.raw_decay", [cols // 2])["values"],
                "raw_phase": parameter("geometry.raw_phase", [cols // 2])["values"],
                "raw_gains": [parameter(f"geometry.raw_gain.{i}", [c["heads"]])["values"]
                              for i in range(len(c["blocks"]))], "pair_metric": metric}
    mixture = c["causal_geometry"].get("mixture")
    if mixture is not None:
        del geometry["pair_metric"]
        geometry["mixture"] = {**mixture, "raw_logits": [
            parameter(f"geometry.mixture_logits.{i}", [c["heads"], len(mixture["metrics"])])["values"]
            for i in range(len(c["blocks"]))]}
    elif metric is None:
        del geometry["pair_metric"]
        geometry["head_pair_metrics"] = c["causal_geometry"]["head_pair_metrics"]
    blocks = []
    for i, block in enumerate(c["blocks"]):
        prefix = f"block.{i}."
        pre = graph([parameter(prefix + "pre.gain", [width], "gain"),
                     parameter(prefix + "pre.bias", [width], "bias")], [norm()], 3)
        qkv = graph([parameter(prefix + "qkv.weight", [width, 3 * width], "weight"),
                     parameter(prefix + "qkv.bias", [3 * width], "bias")], [linear()], 2)
        output = graph([parameter(prefix + "output.weight", [width, width], "weight"),
                        parameter(prefix + "output.bias", [width], "bias")], [linear()], 2)
        feed = [parameter(prefix + "feed.gain", [width], "gain"),
                parameter(prefix + "feed.bias", [width], "bias"),
                parameter(prefix + "feed.up_weight", [width, hidden], "weight"),
                parameter(prefix + "feed.up_bias", [hidden], "bias")]
        stages = [norm(), linear(2, 3, True)]
        if block["topos"]:
            feed.append(parameter(prefix + "feed.topos_gate", [hidden], "gate"))
            stages.append({"kind": "topos_resonator", "gate": 4, "coupling": .2, "iterations": 4,
                           "saturation": .12, "porosity": .3, "max_volume": c["batch"] * c["steps"] * hidden})
        stages.append(linear(len(feed), len(feed) + 1))
        feed += [parameter(prefix + "feed.down_weight", [hidden, width], "weight"),
                 parameter(prefix + "feed.down_bias", [width], "bias")]
        blocks.append({"pre": pre, "heads": c["heads"], "qkv": qkv, "output": output,
                       "feed_forward": graph(feed, stages, 5 if block["topos"] else 3)})
    head = graph([parameter("head.gain", [width], "gain"), parameter("head.bias", [width], "bias"),
                  parameter("head.weight", [width, 256], "weight"),
                  parameter("head.output_bias", [256], "bias")], [norm(), linear(2, 3)], 3)
    require(used == names, "checkpoint owner order")
    return {"schema": ("spiraltorch.nn.byte_decoder_checkpoint.v4" if mixture is not None else
                       "spiraltorch.nn.byte_decoder_checkpoint.v3" if metric is None else "spiraltorch.nn.byte_decoder_checkpoint.v2"), "update_rule": "stateless_sgd.v1",
            "window_state": "reset_positions_and_geometry.v1", "attempted_revision": str(revision),
            "model": {"token": token, "position": position, "geometry": geometry, "blocks": blocks, "head": head}}


def runtime_controls(check, count):
    causal = check["causality"]
    require(all(causal[k] is True for k in ("suffix_gradient_zero", "sensitivity_control")), "causality controls")
    # These are reported maxima without raw control outputs: require the fixed
    # absolute tolerance, not an invented relative denominator or attestation.
    for key in ("prefix_max_abs_error", "other_document_max_abs_error", "extension_max_abs_error"):
        bits(causal[key])
        require(0 <= causal[key] <= 3e-6, "causality error")
    require(all(check["tapes"][k] is True for k in (
        "atomic_rejection", "foreign_tape", "good_bad_good", "invalid_bias_preserves_tape", "recovery",
        "retained_gradients", "retained_output", "stale_gradient", "superseded_tape")), "tape controls")
    cp = check["checkpoint"]
    require(all(cp[k] is True for k in ("exact_logits_loss", "exact_parameter_gradients", "exact_parameters",
        "foreign_gradients", "foreign_tape", "immutable_capture", "nonzero_update_control", "rejected_update_resume")),
        "checkpoint controls")
    for key, expected in (("captured_revision", 2), ("restored_updates", 2), ("final_revision", 6), ("parameter_count", count)):
        require(type(cp[key]) is int and cp[key] == expected, "checkpoint control cursor")
    require(cp["large_revision"] == "9007199254740994"
            and type(cp["json_bytes"]) is int and cp["json_bytes"] > 0, "checkpoint control serialization")
    require(type(cp["first_losses"]) is list and len(cp["first_losses"]) == 2, "checkpoint control losses")
    for value in cp["first_losses"]:
        bits(value)


def vector(actual, expected, length):
    require(type(actual) is list and type(expected) is list
            and len(actual) == len(expected) == length, "vector shape")
    return compare(actual, expected)


def exact(actual, expected):
    if type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), "resume shape")
        for a, e in zip(actual, expected):
            exact(a, e)
    else:
        require(bits(actual) == bits(expected), "resume bits")


def fisher_wide_reference():
    """Float64 analytic VJP for the fixed symmetric two-row regression."""
    p = 1. / (1. + math.exp(-1.))
    r, s = math.sqrt(p), math.sqrt(1. - p)
    t = 2. * (r - s) ** 2
    z = t / 4.
    angle = math.asin(math.sqrt(z))
    distance = 16. * angle ** 2
    slope = 4. * angle / math.sqrt(z * (1. - z))
    seed = rates.struct.unpack("!f", bits(3e38))[0]
    root_gradient = -math.log(2.) * seed * 2. * slope * (r - s)
    require(abs(root_gradient) > 3.4028234663852886e38, "boundary control lost overflow trigger")
    dot = (r - s) * root_gradient
    a, b = .5 * r * (root_gradient - r * dot), .5 * s * (-root_gradient - s * dot)
    return {"coordinates": [a, b, b, a], "raw_gain": [-.5 * seed * distance]}


def mixture_weights(actual, expected, blocks):
    require(type(actual) is list and type(expected) is list and len(actual) == len(expected) == blocks,
            "mixture mass block coverage")
    maximum = 0.
    for a, e in zip(actual, expected):
        maximum = max(maximum, vector(a, e, 6))
        for head in range(2):
            for row in (a[head * 3:(head + 1) * 3], e[head * 3:(head + 1) * 3]):
                require(all(0. <= v <= 1. for v in row) and abs(math.fsum(row) - 1.) <= 1e-6,
                        "mixture masses must be normalized")
    return maximum


def mixture_wide_reference():
    """Analytic float64 boundaries added after the independent numerical review."""
    maximum = rates.struct.unpack("!f", bytes.fromhex("7f7fffff"))[0]
    mass = [math.exp(v - .7) for v in (-.3, .2, .7)]
    total = math.fsum(mass)
    return [
        {"name": "finite_final", "output": [1.5], "logits": [-maximum / 4., maximum / 4.],
         "components": [maximum / 2.] * 2},
        *[{"name": name, "output": [maximum, 1.5], "logits": [-.25, .25], "components": [.5] * 4}
          for name in ("position_centering", "batch_centering")],
        {"name": "identical_components", "output": [maximum] * 2, "logits": [0.] * 3,
         "components": [maximum * w / total for w in mass] * 2},
    ]


def mixture_wide_control(boundary):
    require(type(boundary) is dict and boundary.get("schema") == "spiraltorch.mixture_wide_pullback.v1"
            and boundary.get("nonfinite_final_rejected") is True, "missing mixture wide pullback control")
    checks = boundary.get("checks")
    require(type(checks) is list and len(checks) == 4, "mixture wide control coverage")
    result = []
    for actual, expected in zip(checks, mixture_wide_reference()):
        require(actual.get("name") == expected["name"], "mixture wide control identity")
        errors = {}
        for field in ("output", "components", "logits"):
            values = expected[field]
            vector(actual[field], values, len(values))
            if any(values):
                errors[field + "_relative_l2"] = relative(actual[field], values)[0]
            else:
                exact(actual[field], values)
                errors[field + "_max_abs_error"] = 0.
        result.append({"name": expected["name"], **errors})
    return {"checks": result, "nonfinite_final_rejected": True}


def verify(reference, report, *, kind="flat"):
    require(kind in METRICS or kind in ("headwise", "mixture"), "unsupported metric comparison")
    metric = METRICS.get(kind)
    require(reference["schema"] == f"spiraltorch.resident_byte_geometry_{kind}.torch_fixture.v1", "reference schema")
    require(reference["device"] == "cpu" and reference["dtype"] == "float32"
            and type(reference["threads"]) is int and reference["threads"] == 1, "reference execution")
    require(reference["tolerance"] == {"atol": 3e-6, "rtol": 5e-5, "geometry_relative_l2": .002}, "changed gates")
    require(report["schema"] == f"spiraltorch.resident_byte_geometry_{kind}.validation.v1"
            and report["passed"] is True, "runtime schema/status")
    require(len(reference["cases"]) == len(report["checks"]) == 2, "case coverage")
    summaries = []
    for index, (case, check) in enumerate(zip(reference["cases"], report["checks"])):
        c = case["config"]
        if kind == "mixture":
            require("pair_metric" not in c["causal_geometry"] and "head_pair_metrics" not in c["causal_geometry"]
                    and "metric" not in check and "head_pair_metrics" not in check
                    and type(c["heads"]) is int and c["heads"] == 2, "mixture routing identity")
            expected_mixture = {"metrics": MIXTURE_METRICS, "scales": MIXTURE_SCALES[:index + 1]}
            exact_record(c["causal_geometry"].get("mixture"), expected_mixture)
            exact_record(check.get("mixture"), expected_mixture)
        elif kind == "headwise":
            require("pair_metric" not in c["causal_geometry"] and "metric" not in check
                    and c["causal_geometry"].get("head_pair_metrics") == check.get("head_pair_metrics") == HEADWISE[index]
                    and type(c["heads"]) is int and c["heads"] == (4, 2)[index], "headwise routing identity")
        else:
            require("head_pair_metrics" not in c["causal_geometry"] and "head_pair_metrics" not in check
                    and c["causal_geometry"]["pair_metric"] == check["metric"] == metric, "metric identity")
        descriptors = case["parameters"]
        if kind != "mixture":
            require("mixture" not in c["causal_geometry"] and "mixture" not in check, "foreign mixture mode")
        count = ((24, 39) if kind == "mixture" else (23, 37))[index]
        require(len(descriptors) == count and type(check["parameter_count"]) is int
                and check["parameter_count"] == count, "parameter count")
        names, shapes = [d["name"] for d in descriptors], [d["shape"] for d in descriptors]
        require(check["name"] == case["name"] and check["parameter_names"] == names
                and check["parameter_shapes"] == shapes, "case/layout identity")
        for shape in shapes:
            require(shape and all(type(v) is int and v > 0 for v in shape), "parameter shape")
        lengths = [math.prod(s) for s in shapes]
        geometry_end = 8 + 2 * index if kind == "mixture" else 7 + index
        geometry = list(range(2, geometry_end))
        require([i for i, n in enumerate(names) if n.startswith("geometry.")] == geometry, "geometry coverage")
        errors = {"logit_max_abs_error": 0., "loss_max_abs_error": 0., "parameter_max_abs_error": 0.,
                  "initial_vjp_max_abs_error": 0., "geometry_relative_l2": 0.,
                  "embedding_gradient_max_abs_error": 0., "bias_gradient_max_abs_error": 0.}
        if kind in ("headwise", "mixture"):
            errors["head_gain_relative_l2"] = 0.
        if kind == "mixture":
            errors["router_relative_l2"] = 0.
            errors["mixture_weight_max_abs_error"] = mixture_weights(check["mixture_weights"], case["mixture_weights"], index + 1)
        errors["logit_max_abs_error"] = vector(check["output"], case["output"], c["batch"] * c["steps"] * 256)
        require(len(check["parameter_gradients"]) == len(case["parameter_gradients"]) == count, "initial VJP coverage")
        minimum_norm = math.inf
        for i, (a, e, length) in enumerate(zip(check["parameter_gradients"], case["parameter_gradients"], lengths)):
            errors["initial_vjp_max_abs_error"] = max(errors["initial_vjp_max_abs_error"], vector(a, e, length))
            if i in geometry:
                error, norm = relative(a, e)
                errors["geometry_relative_l2"] = max(errors["geometry_relative_l2"], error)
                minimum_norm = min(minimum_norm, norm)
            if kind in ("headwise", "mixture") and names[i].startswith("geometry.raw_gain."):
                for actual, expected in zip(a, e):
                    error, _ = relative([actual], [expected])
                    errors["head_gain_relative_l2"] = max(errors["head_gain_relative_l2"], error)
            if kind == "mixture" and names[i].startswith("geometry.mixture_logits."):
                require(shapes[i] == [2, 3], "router shape")
                for head in range(2):
                    error, _ = relative(a[head * 3:(head + 1) * 3], e[head * 3:(head + 1) * 3])
                    errors["router_relative_l2"] = max(errors["router_relative_l2"], error)
        embedding_len = c["batch"] * c["steps"] * c["width"]
        errors["embedding_gradient_max_abs_error"] = vector(check["embedding_output_gradient"],
                                                            case["embedding_output_gradient"], embedding_len)
        bias_lengths = [len(bias[k]) for bias in case["biases"] for k in ("z", "pair") if bias[k] is not None]
        require(len(check["bias_gradients"]) == len(case["bias_gradients"]) == len(bias_lengths), "bias VJP coverage")
        for a, e, length in zip(check["bias_gradients"], case["bias_gradients"], bias_lengths):
            errors["bias_gradient_max_abs_error"] = max(errors["bias_gradient_max_abs_error"], vector(a, e, length))
        controls = check["metric_controls"]
        require(controls["geometry_parameter_slots"] == [2, geometry_end]
                and controls["qk_scores_zero"] is c["causal_geometry"]["metric_only_scores"], "metric control identity")
        for name, threshold in (("off_output_separation", 1e-6), ("detached_embedding_gradient_separation", 1e-8)):
            bits(controls[name])
            require(controls[name] > threshold, "insensitive metric control")
        for name in ("output_contrast_relative_l2", "pullback_contrast_relative_l2"):
            bits(controls[name])
            require(0 <= controls[name] <= .002, "metric contrast")
        require(check["gradient_layouts"] is True, "gradient layouts")
        runtime_controls(check, count)
        learning, actual_learning = case["learning"], check["learning"]
        require(type(learning["steps"]) is int and type(actual_learning["steps"]) is int
                and learning["steps"] == actual_learning["steps"] == 16
                and learning["rate"] == .125 and "rates" not in learning
                and len(learning["trace"]) == len(actual_learning["trace"]) == 16, "step/rate coverage")
        for step, (expected, actual) in enumerate(zip(learning["trace"], actual_learning["trace"]), 1):
            require(type(actual["revision"]) is int and actual["revision"] == step, "revision sequence")
            errors["loss_max_abs_error"] = max(errors["loss_max_abs_error"], compare([actual["loss"]], [expected["loss"]]))
            require(len(actual["parameters"]) == len(expected["parameters"]) == count, "parameter coverage")
            for a, e, length in zip(actual["parameters"], expected["parameters"], lengths):
                errors["parameter_max_abs_error"] = max(errors["parameter_max_abs_error"], vector(a, e, length))
            require(len(actual["geometry_gradients"]) == len(expected["geometry_gradients"]) == len(geometry), "gradient coverage")
            for i, a, e in zip(geometry, actual["geometry_gradients"], expected["geometry_gradients"]):
                vector(a, e, lengths[i])
                error, norm = relative(a, e)
                errors["geometry_relative_l2"] = max(errors["geometry_relative_l2"], error)
                minimum_norm = min(minimum_norm, norm)
            errors["embedding_gradient_max_abs_error"] = max(errors["embedding_gradient_max_abs_error"],
                vector(actual["embedding_output_gradient"], expected["embedding_output_gradient"], embedding_len))
            if kind == "mixture":
                errors["mixture_weight_max_abs_error"] = max(errors["mixture_weight_max_abs_error"],
                    mixture_weights(actual["mixture_weights"], expected["mixture_weights"], index + 1))
                for i in geometry:
                    if names[i].startswith("geometry.mixture_logits."):
                        for head in range(2):
                            a = actual["geometry_gradients"][i - 2][head * 3:(head + 1) * 3]
                            e = expected["geometry_gradients"][i - 2][head * 3:(head + 1) * 3]
                            errors["router_relative_l2"] = max(errors["router_relative_l2"], relative(a, e)[0])
        if kind in ("headwise", "mixture"):
            # Do not let larger Poincare heads hide a disconnected Fisher head.
            # Apply the same relative gate to each head's full SGD VJP history.
            for i in geometry:
                if not names[i].startswith("geometry.raw_gain."):
                    continue
                for head in range(lengths[i]):
                    a = [t["geometry_gradients"][i - 2][head] for t in actual_learning["trace"]]
                    e = [t["geometry_gradients"][i - 2][head] for t in learning["trace"]]
                    error, _ = relative(a, e)
                    errors["head_gain_relative_l2"] = max(errors["head_gain_relative_l2"], error)
        if kind == "mixture":
            for i in geometry:
                if not names[i].startswith("geometry.mixture_logits."):
                    continue
                for coordinate in range(lengths[i]):
                    a = [t["geometry_gradients"][i - 2][coordinate] for t in actual_learning["trace"]]
                    e = [t["geometry_gradients"][i - 2][coordinate] for t in learning["trace"]]
                    errors["router_relative_l2"] = max(errors["router_relative_l2"], relative(a, e)[0])
        resume = check["resume_trajectory"]
        for name, value in (("split_revision", 7), ("final_revision", 16), ("verified_steps", 16)):
            require(type(resume[name]) is int and resume[name] == value, "resume cursor")
        require(all(resume[name] is True for name in ("exact_losses", "exact_parameters", "exact_geometry_gradients",
                                                     "exact_embedding_gradients", "fresh_owner")), "resume controls")
        if kind == "mixture":
            require(resume.get("exact_mixture_weights") is True, "resume mixture masses")
        for name, revision in (("checkpoint_json", 7), ("final_checkpoint_json", 16)):
            checkpoint = json.loads(resume[name], object_pairs_hook=unique_object)
            exact_record(checkpoint, expected_checkpoint(case, actual_learning["trace"][revision - 1]["parameters"], revision, metric))
        require(len(resume["trace"]) == 16, "resume step coverage")
        for a, e in zip(resume["trace"], actual_learning["trace"]):
            require(type(a["revision"]) is int and a["revision"] == e["revision"], "resume revision")
            for field in ("loss", "parameters", "geometry_gradients", "embedding_output_gradient"):
                exact(a[field], e[field])
            if kind == "mixture":
                exact(a["mixture_weights"], e["mixture_weights"])
        summaries.append({"name": case["name"], "metric": metric, "parameter_tensors": count,
                          "parameter_scalars": sum(lengths), "steps": 16, "resume_exact_steps": 16,
                          "minimum_reference_geometry_gradient_l2": minimum_norm, **errors})
        if kind == "headwise":
            del summaries[-1]["metric"]
            summaries[-1]["head_pair_metrics"] = HEADWISE[index]
        if kind == "mixture":
            del summaries[-1]["metric"]
            summaries[-1]["mixture"] = expected_mixture
            summaries[-1]["initial_mixture_weights"] = check["mixture_weights"]
            summaries[-1]["last_pre_update_mixture_weights"] = actual_learning["trace"][-1]["mixture_weights"]
    result = {"schema": f"spiraltorch.resident_byte_geometry_{kind}.comparison.v1", "passed": True,
            "cases": summaries, "scope": "Synthetic numerical parity and reported same-runtime resume; not quality, speed, or runtime attestation"}
    if kind == "headwise":
        guard = report.get("head_group_guard")
        require(type(guard) is dict and guard.get("schema") == "spiraltorch.byte_headwise_group_guard.v1"
                and type(guard.get("fault_head")) is int and guard["fault_head"] == 3
                and type(guard.get("parameter_tensors")) is int and guard["parameter_tensors"] == 23
                and all(guard.get(k) is True for k in ("finite_control", "all_gradients_guarded", "atomic_rejection")),
                "missing head group guard")
        bits(guard["baseline_max_abs_bias"])
        require(guard["baseline_max_abs_bias"] > 2., "insensitive overflow control")
        result["head_group_guard"] = guard
    if kind == "fisher_rao":
        boundary = report.get("wide_pullback_control")
        require(type(boundary) is dict and boundary.get("schema")
                == "spiraltorch.fisher_rao_wide_pullback.v1", "missing wide pullback control")
        result["wide_pullback_control"] = {
            key + "_relative_l2": relative(boundary[key], expected)[0]
            for key, expected in fisher_wide_reference().items()}
    if kind == "mixture":
        guard = report.get("mixture_guard")
        require(type(guard) is dict and guard.get("schema") == "spiraltorch.byte_mixture_guard.v1"
                and type(guard.get("fault_head")) is int and guard["fault_head"] == 1
                and type(guard.get("fault_component")) is int and guard["fault_component"] == 0
                and type(guard.get("parameter_tensors")) is int and guard["parameter_tensors"] == 24
                and all(guard.get(k) is True for k in ("finite_control", "zero_mass_control", "all_gradients_guarded", "atomic_rejection")),
                "missing mixture guard")
        bits(guard["baseline_max_abs_bias"])
        require(guard["baseline_max_abs_bias"] > 2., "insensitive mixture guard")
        result["mixture_guard"] = guard
        result["wide_pullback_control"] = mixture_wide_control(report.get("wide_pullback_control"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "report", "output"):
        parser.add_argument(name, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fisher-rao", action="store_true", help="require the explicit Fisher-Rao schemas and metric")
    mode.add_argument("--headwise", action="store_true", help="require the fixed headwise routing and v3 checkpoints")
    mode.add_argument("--mixture", action="store_true", help="require learned mixtures and v4 checkpoints")
    args = parser.parse_args()
    reference, report = args.reference.read_bytes(), args.report.read_bytes()
    result = verify(json.loads(reference, object_pairs_hook=unique_object), json.loads(report, object_pairs_hook=unique_object),
                    kind="mixture" if args.mixture else "headwise" if args.headwise else "fisher_rao" if args.fisher_rao else "flat")
    result["input_sha256"] = {"reference": hashlib.sha256(reference).hexdigest(), "report": hashlib.sha256(report).hexdigest()}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
