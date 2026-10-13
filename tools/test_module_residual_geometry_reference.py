"""Independent stopped VJPs and unchanged inputs for the v2 reference."""
import unittest

import torch

import generate_resident_byte_geometry_torch_fixture as geometry
import diagnose_resident_byte_projective as diagnostic


class ModuleGeometryReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.legacy = geometry.case(2, True, False, 1863, projective_mixture=True, query_mixture=True)
        cls.module = geometry.case(2, True, False, 1863, projective_mixture=True, query_mixture=True,
                                   module_residual=True)

    def test_initial_state_and_data_are_unchanged(self):
        for key in ("parameters", "config", "windows", "inputs", "targets", "biases", "cotangent"):
            with self.subTest(key=key):
                self.assertEqual(self.legacy[key], self.module[key])
        self.assertNotIn("router_detached_parameter_gradients", self.legacy["query_controls"])

    def test_full_and_stopped_vjps_match_independent_evaluator(self):
        case = self.module
        values = [p["values"] for p in case["parameters"]]
        for stopped in (False, True):
            actual = diagnostic.evaluate(case, values, torch.float32, detached_router=stopped,
                                         module_residual=True)
            expected = (case["query_controls"]["router_detached_parameter_gradients"]
                        if stopped else case["parameter_gradients"])
            self.assertTrue(torch.equal(actual["output"], diagnostic.tensor(case["output"])))
            self.assertEqual(len(actual["gradients"]), len(expected))
            for a, b in zip(actual["gradients"], expected, strict=True):
                self.assertTrue(torch.equal(a, diagnostic.tensor(b)))
            embedding = (case["query_controls"]["router_detached_embedding_gradient"]
                         if stopped else case["embedding_output_gradient"])
            self.assertTrue(torch.equal(actual["embedding"], diagnostic.tensor(embedding)))

    def test_legacy_default_evaluator_is_unchanged(self):
        case = self.legacy
        actual = diagnostic.evaluate(case, [p["values"] for p in case["parameters"]], torch.float32)
        self.assertTrue(torch.equal(actual["output"], diagnostic.tensor(case["output"])))
        for a, b in zip(actual["gradients"], case["parameter_gradients"], strict=True):
            self.assertTrue(torch.equal(a, diagnostic.tensor(b)))

    def test_same_runtime_contrast_is_nonzero_and_unrelated_routes_match(self):
        case = self.module
        stopped = case["query_controls"]
        delta = (torch.tensor(case["embedding_output_gradient"]) -
                 torch.tensor(stopped["router_detached_embedding_gradient"]))
        self.assertGreater(delta.norm().item(), 1e-8)
        for descriptor, full, stop in zip(case["parameters"], case["parameter_gradients"],
                                         stopped["router_detached_parameter_gradients"], strict=True):
            if not descriptor["name"].startswith(("token_embedding", "position_embedding",
                    "geometry.projection.", "geometry.raw_decay", "geometry.raw_phase")):
                self.assertEqual(full, stop)


if __name__ == "__main__":
    unittest.main()
