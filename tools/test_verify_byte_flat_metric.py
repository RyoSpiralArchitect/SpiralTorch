"""Fabricated observations test rejection gates, not GPU correctness."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("verify", Path(__file__).with_name("verify_byte_flat_metric.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inputs():
    reference = {"schema": "spiraltorch.resident_byte_geometry_flat.torch_fixture.v1", "device": "cpu",
                 "dtype": "float32", "threads": 1,
                 "tolerance": {"atol": 3e-6, "rtol": 5e-5, "geometry_relative_l2": .002}, "cases": []}
    report = {"schema": "spiraltorch.resident_byte_geometry_flat.validation.v1", "passed": True, "checks": []}
    for index, count in enumerate((23, 37)):
        layout = [("token_embedding", [256, 4]), ("position_embedding", [6, 4]),
                  ("geometry.projection.weight", [4, 4]), ("geometry.projection.bias", [4]),
                  ("geometry.raw_decay", [2]), ("geometry.raw_phase", [2])]
        layout += [(f"geometry.raw_gain.{i}", [2]) for i in range(index + 1)]
        for i in range(index + 1):
            block = [("pre.gain", [4]), ("pre.bias", [4]), ("qkv.weight", [4, 12]), ("qkv.bias", [12]),
                     ("output.weight", [4, 4]), ("output.bias", [4]), ("feed.gain", [4]), ("feed.bias", [4]),
                     ("feed.up_weight", [4, 6]), ("feed.up_bias", [6])]
            if i == 1:
                block.append(("feed.topos_gate", [6]))
            block += [("feed.down_weight", [6, 4]), ("feed.down_bias", [4])]
            layout += [(f"block.{i}." + name, shape) for name, shape in block]
        layout += [("head.gain", [4]), ("head.bias", [4]), ("head.weight", [4, 256]), ("head.output_bias", [256])]
        descriptors = [{"name": name, "shape": shape, "values": [.1] * math.prod(shape)} for name, shape in layout]
        trace = [{"loss": 1. / n, "parameters": [[.1 + n * .001] * len(d["values"]) for d in descriptors],
                  "geometry_gradients": [[1e-7] * len(d["values"]) for d in descriptors[2:7 + index]],
                  "embedding_output_gradient": [.2] * 32} for n in range(1, 17)]
        case = {"name": str(index), "config": {"batch": 2, "steps": 4, "width": 4,
                    "hidden": 6, "heads": 2, "position_capacity": 6,
                    "blocks": [{"topos": i == 1} for i in range(index + 1)],
                    "causal_geometry": {"pair_metric": module.METRIC, "metric_only_scores": index == 0,
                                        "cols": 4, "curvature": -.75}},
                "parameters": descriptors, "output": [.2] * 2048,
                "parameter_gradients": [[1e-7] * len(d["values"]) for d in descriptors],
                "embedding_output_gradient": [.2] * 32, "biases": [], "bias_gradients": [],
                "learning": {"rate": .125, "steps": 16, "trace": trace}}
        observation = [{"revision": n, **copy.deepcopy(t)} for n, t in enumerate(trace, 1)]
        resume = {"split_revision": 7, "final_revision": 16, "verified_steps": 16,
                  "exact_losses": True, "exact_parameters": True, "exact_geometry_gradients": True,
                  "exact_embedding_gradients": True, "fresh_owner": True, "trace": copy.deepcopy(observation)}
        for name, revision in (("checkpoint_json", 7), ("final_checkpoint_json", 16)):
            resume[name] = json.dumps(module.expected_checkpoint(case, observation[revision - 1]["parameters"], revision))
        check = {"name": str(index), "metric": module.METRIC, "parameter_count": count,
                 "parameter_names": [d["name"] for d in descriptors], "parameter_shapes": [d["shape"] for d in descriptors],
                 "output": case["output"], "parameter_gradients": case["parameter_gradients"],
                 "embedding_output_gradient": case["embedding_output_gradient"], "bias_gradients": [],
                 "metric_controls": {"geometry_parameter_slots": [2, 7 + index], "qk_scores_zero": index == 0,
                     "off_output_separation": 1e-4, "detached_embedding_gradient_separation": 1e-6,
                     "output_contrast_relative_l2": 0., "pullback_contrast_relative_l2": 0.},
                 "gradient_layouts": True, "learning": {"steps": 16, "trace": observation}, "resume_trajectory": resume,
                 "causality": {"suffix_gradient_zero": True, "sensitivity_control": True,
                     "prefix_max_abs_error": 0., "other_document_max_abs_error": 0., "extension_max_abs_error": 0.},
                 "tapes": {key: True for key in ("atomic_rejection", "foreign_tape", "good_bad_good",
                     "invalid_bias_preserves_tape", "recovery", "retained_gradients", "retained_output",
                     "stale_gradient", "superseded_tape")},
                 "checkpoint": {"captured_revision": 2, "restored_updates": 2, "final_revision": 6,
                     "parameter_count": count, "json_bytes": 1234, "first_losses": [1., .5],
                     "large_revision": "9007199254740994", "exact_logits_loss": True,
                     "exact_parameter_gradients": True, "exact_parameters": True, "foreign_gradients": True,
                     "foreign_tape": True, "immutable_capture": True, "nonzero_update_control": True,
                     "rejected_update_resume": True}}
        reference["cases"].append(case)
        report["checks"].append(check)
    return copy.deepcopy(reference), copy.deepcopy(report)


class FlatVerifierTests(unittest.TestCase):
    def test_positive(self):
        self.assertTrue(module.verify(*inputs())["passed"])

    def reject(self, change):
        reference, report = inputs()
        change(reference, report)
        with self.assertRaises((ValueError, KeyError)):
            module.verify(reference, report)

    def test_identity(self):
        for field, bad in (("metric", "poincare_squared.v1"), ("parameter_count", 22),
                           ("parameter_count", True), ("name", "other"), ("gradient_layouts", False)):
            with self.subTest(field=field, bad=bad):
                self.reject(lambda r, a: a["checks"][0].__setitem__(field, bad))

    def test_coverage(self):
        self.reject(lambda r, a: a["checks"].pop())
        for field in ("parameters", "geometry_gradients", "embedding_output_gradient"):
            self.reject(lambda r, a: a["checks"][1]["learning"]["trace"][5][field].pop())
        self.reject(lambda r, a: a["checks"][1]["parameter_shapes"][0].__setitem__(0, 2))
        self.reject(lambda r, a: a["checks"][1]["learning"]["trace"].pop())

    def test_severed_tiny_gradients(self):
        for bad in (0., -.5e-7, .5e-7):
            self.reject(lambda r, a: a["checks"][0]["learning"]["trace"][10]["geometry_gradients"][0].__setitem__(0, bad))
            self.reject(lambda r, a: a["checks"][0]["parameter_gradients"][2].__setitem__(0, bad))

    def test_bad_numeric(self):
        for bad in (True, float("nan"), float("inf"), 1e300):
            self.reject(lambda r, a: a["checks"][0]["output"].__setitem__(0, bad))
        self.reject(lambda r, a: a["checks"][0]["learning"]["trace"][0].__setitem__("revision", True))

    def test_exact_resume_not_tolerance(self):
        self.reject(lambda r, a: a["checks"][1]["resume_trajectory"]["trace"][8]["parameters"][0].__setitem__(0, .109 + 1e-7))
        self.reject(lambda r, a: a["checks"][0]["resume_trajectory"]["trace"].pop())
        self.reject(lambda r, a: a["checks"][0]["resume_trajectory"].__setitem__("fresh_owner", False))

    def test_checkpoint_metric_and_clock(self):
        for field, bad in (("schema", "spiraltorch.nn.byte_decoder_checkpoint.v1"), ("attempted_revision", "8")):
            def change(r, a):
                resume = a["checks"][0]["resume_trajectory"]
                cp = json.loads(resume["checkpoint_json"])
                cp[field] = bad
                resume["checkpoint_json"] = json.dumps(cp)
            self.reject(change)
        self.reject(lambda r, a: a["checks"][0]["resume_trajectory"].__setitem__(
            "checkpoint_json", a["checks"][0]["resume_trajectory"]["checkpoint_json"].replace(module.METRIC, "poincare_squared.v1")))

    def reject_checkpoint(self, mutate):
        for index in range(2):
            for field in ("checkpoint_json", "final_checkpoint_json"):
                with self.subTest(index=index, field=field):
                    def change(r, a):
                        resume = a["checks"][index]["resume_trajectory"]
                        cp = json.loads(resume[field])
                        mutate(cp)
                        resume[field] = json.dumps(cp)
                    self.reject(change)

    def test_checkpoint_complete_payload(self):
        for field in ("model", "update_rule", "window_state"):
            self.reject_checkpoint(lambda cp: cp.pop(field))
        for field in ("token", "position", "geometry", "blocks", "head"):
            self.reject_checkpoint(lambda cp: cp["model"].pop(field))
        for field in ("projection", "curvature", "raw_decay", "raw_phase", "raw_gains"):
            self.reject_checkpoint(lambda cp: cp["model"]["geometry"].pop(field))
        self.reject_checkpoint(lambda cp: cp["model"]["token"].pop("values"))
        self.reject_checkpoint(lambda cp: cp["model"]["head"]["parameters"].pop())
        self.reject_checkpoint(lambda cp: cp["model"]["blocks"].pop())

    def test_checkpoint_weights_bound_to_trace(self):
        self.reject_checkpoint(lambda cp: cp["model"]["token"]["values"].__setitem__(0, .107 + 1e-7))
        for bad in (True, float("inf"), float("nan"), 1e300):
            self.reject_checkpoint(lambda cp: cp["model"]["geometry"]["raw_decay"].__setitem__(0, bad))
        self.reject_checkpoint(lambda cp: cp["model"]["head"]["parameters"][0]["values"].__setitem__(0, .2))

    def test_checkpoint_topology_and_types(self):
        self.reject_checkpoint(lambda cp: cp.__setitem__("unknown", 0))
        self.reject_checkpoint(lambda cp: cp["model"]["geometry"].__setitem__("curvature", -.5))
        self.reject_checkpoint(lambda cp: cp["model"]["token"]["shape"].__setitem__(0, 255))
        self.reject_checkpoint(lambda cp: cp["model"]["blocks"][0].__setitem__("heads", True))
        self.reject_checkpoint(lambda cp: cp["model"]["geometry"]["projection"]["stages"][0].__setitem__("gelu", True))
        self.reject_checkpoint(lambda cp: cp["model"]["head"]["stages"][0].__setitem__("epsilon", 1e-3))
        self.reject_checkpoint(lambda cp: cp["model"]["head"]["parameters"][0].__setitem__("role", "weight"))
        self.reject(lambda r, a: a["checks"][0]["resume_trajectory"].__setitem__("checkpoint_json",
            a["checks"][0]["resume_trajectory"]["checkpoint_json"].replace('"attempted_revision": "7"',
                '"attempted_revision": "bad", "attempted_revision": "7"')))

    def test_runtime_control_failures_and_missing(self):
        _, report = inputs()
        for section in ("causality", "tapes", "checkpoint"):
            self.reject(lambda r, a: a["checks"][0].pop(section))
            for key, value in report["checks"][0][section].items():
                self.reject(lambda r, a: a["checks"][0][section].pop(key))
                if type(value) is bool:
                    for bad in (False, 1):
                        self.reject(lambda r, a: a["checks"][0][section].__setitem__(key, bad))
        for key in ("prefix_max_abs_error", "other_document_max_abs_error", "extension_max_abs_error"):
            for bad in (1., -1., True, float("nan")):
                self.reject(lambda r, a: a["checks"][0]["causality"].__setitem__(key, bad))
        self.reject(lambda r, a: a["checks"][0]["checkpoint"].__setitem__("large_revision", 9007199254740994))
        self.reject(lambda r, a: a["checks"][0]["checkpoint"].__setitem__("restored_updates", 0))

    def test_gate_and_rate(self):
        self.reject(lambda r, a: r["tolerance"].__setitem__("atol", 1.))
        self.reject(lambda r, a: r["cases"][0]["learning"].__setitem__("rate", .1))
        self.reject(lambda r, a: a["checks"][0]["metric_controls"].__setitem__("off_output_separation", 0.))


if __name__ == "__main__":
    unittest.main()
