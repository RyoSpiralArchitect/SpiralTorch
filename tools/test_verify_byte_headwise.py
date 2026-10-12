"""Fabricated records exercise identity/rejection gates, not GPU correctness."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("flat_tests", Path(__file__).with_name("test_verify_byte_flat_metric.py"))
flat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flat)
verify = flat.module


def inputs():
    reference, report = flat.inputs()
    reference["schema"] = "spiraltorch.resident_byte_geometry_headwise.torch_fixture.v1"
    report["schema"] = "spiraltorch.resident_byte_geometry_headwise.validation.v1"
    report["head_group_guard"] = {"schema": "spiraltorch.byte_headwise_group_guard.v1", "fault_head": 3,
        "parameter_tensors": 23, "finite_control": True, "all_gradients_guarded": True,
        "atomic_rejection": True, "baseline_max_abs_bias": 3.}
    for index, (case, check) in enumerate(zip(reference["cases"], report["checks"])):
        del case["config"]["causal_geometry"]["pair_metric"]
        del check["metric"]
        case["config"]["causal_geometry"]["head_pair_metrics"] = copy.deepcopy(verify.HEADWISE[index])
        check["head_pair_metrics"] = copy.deepcopy(verify.HEADWISE[index])
        if index == 0:
            case["config"]["heads"] = 4
            case["parameters"][6]["shape"] = [4]
            case["parameters"][6]["values"] *= 2
            check["parameter_shapes"][6] = [4]
            for record in (case, check):
                record["parameter_gradients"][6] *= 2
                for row in record["learning"]["trace"]:
                    row["parameters"][6] *= 2
                    row["geometry_gradients"][4] *= 2
            check["resume_trajectory"]["trace"] = copy.deepcopy(check["learning"]["trace"])
        for key, revision in (("checkpoint_json", 7), ("final_checkpoint_json", 16)):
            check["resume_trajectory"][key] = json.dumps(verify.expected_checkpoint(
                case, check["learning"]["trace"][revision - 1]["parameters"], revision, None))
    return reference, report


class HeadwiseVerifierTests(unittest.TestCase):
    def reject(self, mutate):
        reference, report = inputs()
        mutate(reference, report)
        with self.assertRaises((ValueError, KeyError)):
            verify.verify(reference, report, kind="headwise")

    def test_positive_and_distinct_mode(self):
        self.assertTrue(verify.verify(*inputs(), kind="headwise")["passed"])
        for args, kind in ((inputs(), "flat"), (inputs(), "fisher_rao"), (flat.inputs(), "headwise")):
            with self.assertRaises(ValueError):
                verify.verify(*args, kind=kind)

    def test_every_route_is_bound_in_reference_and_report(self):
        for index, routing in enumerate(verify.HEADWISE):
            for block, row in enumerate(routing):
                for head in range(len(row)):
                    for surface in ("reference", "report", "both"):
                        def mutate(r, a):
                            if surface in ("reference", "both"):
                                r["cases"][index]["config"]["causal_geometry"]["head_pair_metrics"][block][head] = "unknown"
                            if surface in ("report", "both"):
                                a["checks"][index]["head_pair_metrics"][block][head] = "unknown"
                        self.reject(mutate)

    def test_checkpoint_routes_and_schema_are_exact(self):
        for field in ("checkpoint_json", "final_checkpoint_json"):
            for change in (
                lambda cp: cp.__setitem__("schema", "spiraltorch.nn.byte_decoder_checkpoint.v2"),
                lambda cp: cp["model"]["geometry"].pop("head_pair_metrics"),
                lambda cp: cp["model"]["geometry"].__setitem__("pair_metric", verify.METRIC),
                lambda cp: cp["model"]["geometry"]["head_pair_metrics"][0].reverse(),
            ):
                def mutate(r, a):
                    resume = a["checks"][0]["resume_trajectory"]
                    cp = json.loads(resume[field])
                    change(cp)
                    resume[field] = json.dumps(cp)
                self.reject(mutate)

    def test_uniform_field_cannot_hide_headwise_routing(self):
        self.reject(lambda r, a: r["cases"][0]["config"]["causal_geometry"].__setitem__("pair_metric", verify.METRIC))
        self.reject(lambda r, a: a["checks"][0].__setitem__("metric", verify.METRIC))
        r, a = flat.inputs()
        r["cases"][0]["config"]["causal_geometry"]["head_pair_metrics"] = verify.HEADWISE[0]
        with self.assertRaises(ValueError):
            verify.verify(r, a)

    def test_shapes_and_gradient_gates_are_not_relaxed(self):
        for mutate in (
            lambda r, a: r["cases"][0]["config"].__setitem__("heads", 2),
            lambda r, a: a["checks"][0]["learning"]["trace"][3]["geometry_gradients"][4].__setitem__(2, 0.),
            lambda r, a: a["checks"][1]["resume_trajectory"]["trace"][8]["parameters"][0].__setitem__(0, .109 + 1e-7),
            lambda r, a: r["tolerance"].__setitem__("rtol", .1),
            lambda r, a: a["checks"].pop(),
        ):
            self.reject(mutate)

    def test_group_guard_is_required_and_sensitive(self):
        self.reject(lambda r, a: a.pop("head_group_guard"))
        for key in ("finite_control", "all_gradients_guarded", "atomic_rejection"):
            self.reject(lambda r, a: a["head_group_guard"].__setitem__(key, False))
        for bad in (2., True, float("nan")):
            self.reject(lambda r, a: a["head_group_guard"].__setitem__("baseline_max_abs_bias", bad))

    def test_large_heads_cannot_hide_a_disconnected_small_head(self):
        def mutate(r, a):
            for trace, value in ((r["cases"][0]["learning"]["trace"], 1e-8),
                                 (a["checks"][0]["learning"]["trace"], 0.)):
                for row in trace:
                    row["geometry_gradients"][4] = [1., 1., value, 1.]
            a["checks"][0]["resume_trajectory"]["trace"] = copy.deepcopy(a["checks"][0]["learning"]["trace"])
        self.reject(mutate)


if __name__ == "__main__":
    unittest.main()
