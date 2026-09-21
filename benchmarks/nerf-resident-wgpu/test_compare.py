"""Admission regressions run without torch; numerical control is a separate run."""
import copy
import importlib.util
import itertools
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("nerf_compare", Path(__file__).with_name("compare.py"))
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def fixture():
    return {
        "status": "passed", "weights": [0.] * 12, "bias": [1.] * 4,
        "guards": {name: True for name in control.GUARDS},
        "cases": [
            {"rays": rows, "samples": count, "varying": varying, "seed": seed,
             "mode": "Midpoint" if seed is None else "Stratified { seed: 17 }",
             "ray_inputs": [[0., 0., 0., 0., 0., 1., 0., 1.]] * rows,
             "rgba": [0.] * (rows * 4)}
            for rows, count, varying, seed in itertools.product(
                [1, 65, 256], [1, 8, 64], [False, True], [None, 17])
        ],
    }


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.report = fixture()

    def rejects(self):
        with self.assertRaises(ValueError):
            control.admit(self.report)

    def test_complete_grid(self):
        self.assertEqual(len(control.admit(self.report)), 36)

    def test_missing_or_duplicate_condition(self):
        self.report["cases"][0] = copy.deepcopy(self.report["cases"][1])
        self.rejects()
        self.report["cases"].pop()
        self.rejects()

    def test_wrong_guard_names_and_nonboolean_acceptance(self):
        self.report["guards"]["renamed"] = self.report["guards"].pop("retained_version")
        self.rejects()
        self.report = fixture()
        self.report["guards"]["retained_version"] = 1
        self.rejects()

    def test_shapes(self):
        for field in ["ray_inputs", "rgba"]:
            self.report = fixture()
            self.report["cases"][0][field].pop()
            self.rejects()
        self.report = fixture()
        self.report["weights"].pop()
        self.rejects()

    def test_nonfinite_or_unrepresentable_inputs_and_outputs(self):
        for value in [float("nan"), float("inf"), 1e40]:
            for location in ["ray", "rgba", "bias"]:
                self.report = fixture()
                payload = (self.report["cases"][0]["ray_inputs"][0] if location == "ray"
                           else self.report["cases"][0]["rgba"] if location == "rgba"
                           else self.report["bias"])
                payload[0] = value
                self.rejects()

    def test_reversed_bounds_and_mode_drift(self):
        self.report["cases"][0]["ray_inputs"][0][7] = -1.
        self.rejects()
        self.report = fixture()
        self.report["cases"][0]["mode"] = "Stratified { seed: 99 }"
        self.rejects()

    def test_f32_identity_preserves_signed_zero(self):
        self.assertNotEqual(control.f32_bytes([0.]), control.f32_bytes([-0.]))

    def test_failed_fixture(self):
        self.report["status"] = "error"
        self.rejects()


if __name__ == "__main__":
    unittest.main()
