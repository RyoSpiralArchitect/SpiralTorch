import copy
import json
import unittest

from validate_module_projective_results import causality_controls, flags, numeric_case, paired_controls, resume_controls
from test_validate_resident_query_pullback import valid as valid_paired


def numeric_pair():
    gradients = [[.2], [.3], [.1]]
    trace = [{"loss": 1. - step * .01, "parameters": [[step * .01], [0.], [.1]],
              "geometry_gradients": [[.1]]} for step in range(1, 17)]
    case = {"name": "synthetic", "parameters": [{"name": n} for n in ["token", "position", "geometry"]],
            "output": [1.], "parameter_gradients": gradients, "learning": {"trace": trace}}
    actual = copy.deepcopy(case)
    actual["qualification_failures"] = []
    for step, row in enumerate(actual["learning"]["trace"], 1):
        row.update(revision=step, geometry_gradient_relative_l2=[0.])
    return case, actual


def resumed_case():
    trace = [{"revision": step, "loss": 1., "parameters": [[.1]], "geometry_gradients": [[.2]],
              "embedding_output_gradient": [.3]} for step in range(1, 17)]
    resume = {"exact_losses": True, "exact_parameters": True, "exact_geometry_gradients": True,
              "exact_embedding_gradients": True, "fresh_owner": True,
              "split_revision": 7, "final_revision": 16, "verified_steps": 16,
              "trace": copy.deepcopy(trace), "checkpoint_json": "private-checkpoint-body",
              "final_checkpoint_json": "private-final-checkpoint"}
    return {"learning": {"trace": trace}, "resume_trajectory": resume}


class ModulePublisherTests(unittest.TestCase):
    def test_arbitrary_adapter_payload_cannot_leave(self):
        report, fixture, stopped = valid_paired()
        report["schema"] = "spiraltorch.resident_byte_query_pullback.validation.v2"
        report["adapter"] = {"checkpoint": "private-body", "tensor": [1.,2.]}
        result = paired_controls(report, fixture, stopped)
        self.assertNotIn("adapter", result)
        self.assertNotIn("private-body", json.dumps(result))

    def test_causality_requires_bounded_raw_witnesses_and_consistent_errors(self):
        control = {"sensitivity_control": True, "suffix_gradient_zero": True, "numeric_witnesses": {}}
        for name, n in [("prefix",512), ("other_document",1024), ("extension",1024)]:
            control[name+"_max_abs_error"] = 0.
            control["numeric_witnesses"][name] = {"actual": [0.]*n, "expected": [0.]*n}
        self.assertNotIn("numeric_witnesses", causality_controls(control))
        for name in ["prefix", "other_document", "extension"]:
            bad = copy.deepcopy(control)
            bad[name+"_max_abs_error"] = 1e100
            with self.subTest(name=name), self.assertRaises(ValueError):
                causality_controls(bad)
            bad = copy.deepcopy(control)
            bad["numeric_witnesses"][name]["actual"][0] = .01
            bad[name+"_max_abs_error"] = .01
            with self.subTest(name=name), self.assertRaises(ValueError):
                causality_controls(bad)
            bad = copy.deepcopy(control)
            bad["numeric_witnesses"][name]["actual"].pop()
            with self.subTest(name=name), self.assertRaises(ValueError):
                causality_controls(bad)

    def test_all_numeric_steps_are_verified_and_private_extras_are_not_copied(self):
        case, actual = numeric_pair()
        actual["private_extra"] = "never-publish-this"
        result = numeric_case(case, actual)
        self.assertEqual(len(result["steps"]), 16)
        self.assertTrue(all(s["parameter_count"] == 3 for s in result["steps"]))
        self.assertNotIn("never-publish-this", json.dumps(result))
        self.assertTrue(all("parameters" not in s for s in result["steps"]))

    def test_truncated_families_and_false_relative_reports_are_rejected(self):
        case, original = numeric_pair()
        changes = [
            lambda r: r["parameter_gradients"].pop(),
            lambda r: r["learning"]["trace"].pop(),
            lambda r: r["learning"]["trace"][0]["parameters"].pop(),
            lambda r: r["learning"]["trace"][0]["geometry_gradients"].pop(),
            lambda r: r["learning"]["trace"][0].update(geometry_gradient_relative_l2=[.001]),
            lambda r: r["learning"]["trace"][0].update(revision=2),
        ]
        for i, change in enumerate(changes):
            actual = copy.deepcopy(original)
            change(actual)
            with self.subTest(i=i), self.assertRaises(ValueError):
                numeric_case(case, actual)

    def test_absolute_tolerance_does_not_accept_inert_tiny_geometry(self):
        case, actual = numeric_pair()
        case["learning"]["trace"][0]["geometry_gradients"] = [[1e-7]]
        actual["learning"]["trace"][0]["geometry_gradients"] = [[0.]]
        with self.assertRaisesRegex(ValueError, "inert"):
            numeric_case(case, actual)

    def test_failed_conditions_cannot_be_published_as_passes(self):
        case, actual = numeric_pair()
        actual["qualification_failures"] = [{"gate": "failed"}]
        with self.assertRaises(ValueError):
            numeric_case(case, actual)
        actual["qualification_failures"] = []
        actual["query_controls_error"] = "failed"
        with self.assertRaises(ValueError):
            numeric_case(case, actual)

    def test_resume_is_exact_and_only_checkpoint_hashes_leave(self):
        actual = resumed_case()
        result = resume_controls(actual)
        self.assertEqual(result["verified_steps"], 16)
        self.assertNotIn("private-checkpoint-body", json.dumps(result))
        self.assertNotIn("checkpoint_json", result)
        self.assertEqual(len(result["checkpoint_json_sha256"]), 64)
        actual["resume_trajectory"]["trace"][0]["parameters"][0][0] += 1e-10
        with self.assertRaisesRegex(ValueError, "resumed values differ"):
            resume_controls(actual)

    def test_missing_resume_step_or_mixture_is_rejected(self):
        actual = resumed_case()
        actual["resume_trajectory"]["trace"].pop()
        with self.assertRaises(ValueError):
            resume_controls(actual)
        actual = resumed_case()
        actual["learning"]["trace"][0]["mixture_weights"] = [[.5, .5]]
        with self.assertRaises((ValueError, KeyError)):
            resume_controls(actual)

    def test_truthy_is_not_true(self):
        for value in [1, "true", None, False]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                flags({"control": value}, ["control"])


if __name__ == "__main__":
    unittest.main()
