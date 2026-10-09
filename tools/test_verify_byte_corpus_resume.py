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
    request = {"config": {"batch": 1, "steps": 2}, "train_batches": [0, 1],
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


class ResumeVerifierTests(unittest.TestCase):
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
