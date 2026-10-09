import copy
import importlib.util
import json
import math
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("resume", Path(__file__).with_name("verify_byte_corpus_resume.py"))
RESUME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESUME)


def model_fixture(geometry):
    parameters = []

    def parameter(shape, role=None):
        value = {"shape": shape, "values": [0.1] * math.prod(shape)}
        parameters.append(copy.deepcopy(value))
        if role:
            value["role"] = role
        return value

    def graph(specs, stages, version=2):
        return {"schema": f"spiraltorch.nn.inference_plan.v{version}",
                "input_shape": [1, 2, 2], "parameters": [parameter(shape, role) for role, shape in specs],
                "stages": stages}

    norm = {"kind": "layer_norm", "gain": 0, "bias": 1, "epsilon": 1e-5}
    linear = {"kind": "linear", "weight": 0, "bias": 1, "gelu": False}
    model = {"token": parameter([256, 2]), "position": parameter([2, 2]), "geometry": None}
    if geometry:
        projection = graph([("weight", [2, 2]), ("bias", [2])], [linear])
        model["geometry"] = {"projection": projection, "curvature": -0.75,
                              "raw_decay": parameter([1])["values"],
                              "raw_phase": parameter([1])["values"],
                              "raw_gains": [parameter([1])["values"]]}
    block = {"heads": 1,
             "pre": graph([("gain", [2]), ("bias", [2])], [norm], 3),
             "qkv": graph([("weight", [2, 6]), ("bias", [6])], [linear]),
             "output": graph([("weight", [2, 2]), ("bias", [2])], [linear]),
             "feed_forward": graph([("gain", [2]), ("bias", [2]), ("weight", [2, 2]),
                                    ("bias", [2]), ("weight", [2, 2]), ("bias", [2])],
                                   [norm, {"kind": "linear", "weight": 2, "bias": 3, "gelu": True},
                                    {"kind": "linear", "weight": 4, "bias": 5, "gelu": False}], 3)}
    model["blocks"] = [block]
    model["head"] = graph([("gain", [2]), ("bias", [2]), ("weight", [2, 256]), ("bias", [256])],
                          [norm, {"kind": "linear", "weight": 2, "bias": 3, "gelu": False}], 3)
    return {"schema": "spiraltorch.nn.byte_decoder_checkpoint.v1", "attempted_revision": "1",
            "update_rule": "stateless_sgd.v1", "window_state": "reset_positions_and_geometry.v1",
            "model": model}, parameters


def fixture():
    request = {"schema": "spiraltorch.byte_corpus.request.v1", "config": {"batch": 1, "steps": 2}, "train_batches": [0, 1],
               "checkpoint_every": 2, "validation_batches": [0], "cases": []}
    full = {"schema": "spiraltorch.byte_corpus.result.v1", "engine": "spiraltorch",
            "request_sha256": "hash", "adapter": "same", "cases": []}
    checkpoint = {"schema": "spiraltorch.byte_corpus.checkpoint.v1", "request_sha256": "hash",
                  "completed_updates": 1, "cases": []}
    for name, geometry in (("plain", False), ("geometry", True)):
        model, parameters = model_fixture(geometry)
        request["cases"].append({"name": name, "seed": 7, "geometry": geometry,
                                 "parameters": parameters})
        training = [{"revision": n, "ce": 2.0} for n in (1, 2)]
        validation = [{"revision": n, "batch_losses": [2.0], "mean_ce": 2.0,
                       "bits_per_byte": 2.88539, "target_bytes": 2} for n in (0, 2)]
        full["cases"].append({"name": name, "seed": 7, "geometry": geometry,
                              "parameter_tensors": len(parameters),
                              "parameter_scalars": sum(len(p["values"]) for p in parameters),
                              "training": training, "validation": validation,
                              "final_parameters": [p["values"] for p in parameters]})
        checkpoint["cases"].append({"name": name, "training": copy.deepcopy(training[:1]),
                                    "validation": [{"revision": 0, "batch_losses": [2.0]}],
                                    "model_json": json.dumps(model)})
    partial = copy.deepcopy(full)
    partial["schema"] = "spiraltorch.byte_corpus.partial.v1"
    for case in partial["cases"]:
        case["training"] = case["training"][:1]
        case["validation"] = case["validation"][:1]
    return [request, "hash", full, partial, copy.deepcopy(full), checkpoint]


def frozen_fixture():
    args = fixture()
    request = args[0]
    request["schema"] = "spiraltorch.byte_corpus.request.v2"
    for value in [request, *args[2:]]:
        value["cases"].append(copy.deepcopy(value["cases"][1]))
        value["cases"][2]["name"] = "frozen"
    for index, case in enumerate(request["cases"]):
        case["geometry_update"] = "frozen" if index == 2 else "train"
        for i, parameter in enumerate(case["parameters"]):
            if case["geometry"] and 2 <= i < 7:
                parameter["name"] = "geometry." + str(i)
            else:
                parameter["name"] = "other." + str(i - 5 if case["geometry"] and i >= 7 else i)
    for report in args[2:5]:
        report["schema"] = report["schema"].removesuffix("v1") + "v2"
        for spec, case in zip(request["cases"], report["cases"]):
            case["geometry_update"] = spec["geometry_update"]
            case["trainable_parameter_scalars"] = sum(len(p["values"]) for p in spec["parameters"]
                if spec["geometry_update"] != "frozen" or not p["name"].startswith("geometry."))
    args[5]["schema"] = "spiraltorch.byte_corpus.checkpoint.v2"
    return args


def metric_fixture():
    args = frozen_fixture()
    request = args[0]
    request["schema"] = RESUME.STUDY.REQUEST_V3
    for value in [request, *args[2:]]:
        value["cases"] += copy.deepcopy(value["cases"][1:3])
        value["cases"][3]["name"], value["cases"][4]["name"] = "flat", "flat_frozen"
    for i in range(1, 5):
        request["cases"][i]["pair_metric"] = RESUME.STUDY.POINCARE if i < 3 else RESUME.STUDY.FLAT
    for report in args[2:5]:
        report["schema"] = report["schema"].removesuffix("v2") + "v3"
        for case, spec in zip(report["cases"], request["cases"]):
            case["pair_metric"] = spec.get("pair_metric")
    args[5]["schema"] = "spiraltorch.byte_corpus.checkpoint.v3"
    for case in args[5]["cases"][3:]:
        model = json.loads(case["model_json"])
        model["schema"] = "spiraltorch.nn.byte_decoder_checkpoint.v2"
        model["model"]["geometry"]["pair_metric"] = RESUME.STUDY.FLAT
        case["model_json"] = json.dumps(model)
    return args


def calibrated_fixture():
    args = metric_fixture()
    request = args[0]
    request["schema"] = RESUME.STUDY.REQUEST_V4
    request["bias_calibration"] = dict(source_request_sha256="a" * 64, train_batch_indices=[0, 1], relative_tolerance=1e-5)
    for value in [request, *args[2:]]:
        value["cases"] += copy.deepcopy(value["cases"][3:5])
        value["cases"][5]["name"], value["cases"][6]["name"] = "matched", "matched_frozen"
    for i, case in enumerate(request["cases"]):
        case["bias_initialization"] = RESUME.STUDY.MATCHED if i >= 5 else RESUME.STUDY.ORIGINAL
        if case["geometry"]:
            case["parameters"][6]["name"] = "geometry.raw_gain.0"
        if i >= 5:
            case["parameters"][6]["values"] = [.25]
    for report in args[2:5]:
        report["schema"] = report["schema"].removesuffix("v3") + "v4"
        report["bias_calibration"] = copy.deepcopy(request["bias_calibration"])
        for i, (case, spec) in enumerate(zip(report["cases"], request["cases"])):
            case["bias_initialization"] = spec["bias_initialization"]
            if i >= 5:
                case["final_parameters"][6] = [.25]
    args[5]["schema"] = "spiraltorch.byte_corpus.checkpoint.v4"
    for case in args[5]["cases"][5:]:
        model = json.loads(case["model_json"])
        model["model"]["geometry"]["raw_gains"][0] = [.25]
        case["model_json"] = json.dumps(model)
    return args


class ResumeVerifierTests(unittest.TestCase):
    def test_calibrated_resume_binds_fitted_initial_gain_and_metadata(self):
        self.assertTrue(RESUME.verify(*calibrated_fixture())["passed"])
        for mutation in ("init", "metadata", "initial_gain", "checkpoint_gain", "schema"):
            args = calibrated_fixture()
            if mutation == "schema":
                args[5]["schema"] = "spiraltorch.byte_corpus.checkpoint.v3"
            elif mutation == "checkpoint_gain":
                model = json.loads(args[5]["cases"][6]["model_json"])
                model["model"]["geometry"]["raw_gains"][0] = [.1]
                args[5]["cases"][6]["model_json"] = json.dumps(model)
            else:
                for report in args[2:5]:
                    if mutation == "init":
                        report["cases"][6]["bias_initialization"] = RESUME.STUDY.ORIGINAL
                    elif mutation == "metadata":
                        report["bias_calibration"]["train_batch_indices"] = [1, 0]
                    else:
                        report["cases"][6]["final_parameters"][6] = [.1]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                RESUME.verify(*args)

    def test_metric_resume_selects_model_schema_per_arm(self):
        result = RESUME.verify(*metric_fixture())
        self.assertEqual(result["schema"], "spiraltorch.byte_corpus.resume_verification.v3")
        self.assertEqual([c["pair_metric"] for c in result["cases"]],
                         [None, RESUME.STUDY.POINCARE, RESUME.STUDY.POINCARE, RESUME.STUDY.FLAT, RESUME.STUDY.FLAT])

    def test_metric_resume_rejects_changed_or_missing_identity(self):
        for variant in ("schema", "flat_to_poincare", "poincare_to_flat", "report", "missing", "frozen_drift"):
            args = metric_fixture()
            if variant == "schema":
                args[5]["schema"] = "spiraltorch.byte_corpus.checkpoint.v2"
            elif variant in ("flat_to_poincare", "poincare_to_flat"):
                slot = 3 if variant == "flat_to_poincare" else 1
                model = json.loads(args[5]["cases"][slot]["model_json"])
                if slot == 3:
                    model["schema"] = "spiraltorch.nn.byte_decoder_checkpoint.v1"
                    model["model"]["geometry"].pop("pair_metric")
                else:
                    model["schema"] = "spiraltorch.nn.byte_decoder_checkpoint.v2"
                    model["model"]["geometry"]["pair_metric"] = RESUME.STUDY.FLAT
                args[5]["cases"][slot]["model_json"] = json.dumps(model)
            else:
                for report in args[2:5]:
                    if variant == "report":
                        report["cases"][3]["pair_metric"] = RESUME.STUDY.POINCARE
                    elif variant == "missing":
                        report["cases"][0].pop("pair_metric")
                    else:
                        report["cases"][4]["final_parameters"][2][0] += 1e-7
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                RESUME.verify(*args)

    def test_invalid_request_policy_cannot_pass_even_with_matching_artifacts(self):
        for variant in ("missing_arm", "duplicate_mode", "ordinary_frozen", "geometry_init", "core_init", "v1_policy"):
            args = frozen_fixture()
            cases = args[0]["cases"]
            if variant == "missing_arm":
                for value in [args[0], *args[2:]]:
                    value["cases"].pop()
            elif variant == "duplicate_mode":
                cases[2]["geometry_update"] = "train"
            elif variant == "ordinary_frozen":
                cases[0]["geometry_update"] = "frozen"
            elif variant in ("geometry_init", "core_init"):
                cases[2]["parameters"][2 if variant == "geometry_init" else 0]["values"][0] += .01
            else:
                args[0]["schema"] = "spiraltorch.byte_corpus.request.v1"
                for value in args[2:]:
                    value["schema"] = value["schema"].removesuffix("v2") + "v1"
            for report in args[2:5]:
                for spec, case in zip(cases, report["cases"]):
                    case["geometry_update"] = spec["geometry_update"]
                    case["trainable_parameter_scalars"] = RESUME.STUDY.trainable_scalars(spec)
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                RESUME.verify(*args)

    def test_negative_zero_literal_survives_request_and_nested_checkpoint(self):
        args = frozen_fixture()
        for case in args[0]["cases"][1:]:
            case["parameters"][2]["values"][0] = -0.0
        for report in args[2:5]:
            for case in report["cases"][1:]:
                case["final_parameters"][2][0] = -0.0
        for case in args[5]["cases"][1:]:
            model = json.loads(case["model_json"])
            model["model"]["geometry"]["projection"]["parameters"][0]["values"][0] = -0.0
            case["model_json"] = json.dumps(model).replace("-0.0", "-0")
        args[0] = RESUME.STUDY.decode_json(json.dumps(args[0]).replace("-0.0", "-0"))
        self.assertTrue(RESUME.verify(*args)["passed"])

    def test_duplicate_nested_checkpoint_keys_are_rejected(self):
        args = frozen_fixture()
        case = args[5]["cases"][0]
        case["model_json"] = case["model_json"].replace('"schema":', '"schema":"ignored", "schema":', 1)
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            RESUME.verify(*args)

    def test_frozen_resume_has_explicit_v2_policy(self):
        self.assertEqual(RESUME.verify(*frozen_fixture())["schema"], "spiraltorch.byte_corpus.resume_verification.v2")

    def test_frozen_resume_rejects_policy_schema_and_even_matched_weight_drift(self):
        for variant in ("policy", "schema", "weight"):
            args = frozen_fixture()
            if variant == "schema":
                args[5]["schema"] = "spiraltorch.byte_corpus.checkpoint.v1"
            elif variant == "policy":
                for report in args[2:5]:
                    report["cases"][2]["geometry_update"] = "train"
            else:
                for report in args[2:5]:
                    report["cases"][2]["final_parameters"][2][0] += .01
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                RESUME.verify(*args)

    def test_exact_resume_and_prefix(self):
        self.assertTrue(RESUME.verify(*fixture())["passed"])

    def test_changed_resumed_parameter_or_loss_is_rejected(self):
        for key in ("final_parameters", "training"):
            args = fixture()
            if key == "training":
                args[4]["cases"][0][key][1]["ce"] += 1e-8
            else:
                args[4]["cases"][0][key][0][0] += 1e-8
            with self.assertRaises(ValueError):
                RESUME.verify(*args)

    def test_wrong_pause_evaluation_or_saved_history_is_rejected(self):
        for variant in range(4):
            args = fixture()
            if variant == 0:
                args[3]["cases"][0]["validation"].append({"revision": 1})
            elif variant == 1:
                args[5]["cases"][0]["training"][0]["ce"] = 3.0
            elif variant == 2:
                args[5]["cases"][0]["validation"][0]["batch_losses"] = []
            else:
                args[5]["cases"].reverse()
            with self.assertRaises(ValueError):
                RESUME.verify(*args)

    def test_incomplete_or_nonfinite_baseline_is_rejected(self):
        for variant in range(3):
            args = fixture()
            if variant == 0:
                args[2]["cases"].pop()
            elif variant == 1:
                args[2]["cases"][0]["validation"][0]["mean_ce"] = float("inf")
            else:
                args[2]["cases"][0]["final_parameters"][0][0] = float("inf")
            args[4] = copy.deepcopy(args[2])
            with self.assertRaises(ValueError):
                RESUME.verify(*args)

    def test_checkpoint_model_envelope_shape_and_weights_are_checked(self):
        for variant in range(4):
            args = fixture()
            model = json.loads(args[5]["cases"][0]["model_json"])
            if variant == 0:
                del model["model"]
            elif variant == 1:
                model["update_rule"] = "adam"
            elif variant == 2:
                model["model"]["token"]["values"][0] += 0.01
            else:
                model["model"]["token"]["shape"] = [512, 1]
            args[5]["cases"][0]["model_json"] = json.dumps(model)
            with self.assertRaises(ValueError):
                RESUME.verify(*args)

    def test_boolean_integer_and_signed_zero_mutations_are_not_exact(self):
        for baseline, changed in ((1.0, True), (1.0, 1), (0.0, -0.0)):
            args = fixture()
            args[2]["cases"][0]["final_parameters"][0][0] = baseline
            args[4] = copy.deepcopy(args[2])
            args[4]["cases"][0]["final_parameters"][0][0] = changed
            with self.assertRaises(ValueError):
                RESUME.verify(*args)
        args = fixture()
        args[2]["cases"][0]["training"][0]["ce"] = 0.0
        args[4] = copy.deepcopy(args[2])
        args[3]["cases"][0]["training"][0]["ce"] = -0.0
        with self.assertRaises(ValueError):
            RESUME.verify(*args)


if __name__ == "__main__":
    unittest.main()
