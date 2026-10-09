"""Small fabricated traces test the verifier, never stand in for GPU evidence."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("verify", Path(__file__).with_name("verify_byte_parameter_rates.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inputs():
    reference = {"schema": "spiraltorch.resident_byte_geometry_frozen.torch_fixture.v1",
                 "device": "cpu", "dtype": "float32", "threads": 1,
                 "tolerance": {"atol": 3e-6, "rtol": 5e-5, "geometry_relative_l2": .002}, "cases": []}
    report = {"schema": "spiraltorch.resident_byte_parameter_rates.validation.v1", "passed": True, "checks": []}
    for index, count in enumerate((23, 37)):
        params = [{"name": ("geometry." if 2 <= i < 7 + index else "other.") + str(i),
                   "shape": [1], "values": [.1]} for i in range(count)]
        rates = [0. if p["name"].startswith("geometry.") else .125 for p in params]
        trace = [{"loss": 1. / step, "parameters": [[.1 if r == 0 else .1 + step * .01] for r in rates],
                  "geometry_gradients": [[1e-7] for _ in range(5 + index)],
                  "embedding_output_gradient": [.1, .2]} for step in range(1, 17)]
        case = {"name": str(index), "parameters": params, "learning": {"rates": rates, "steps": 16, "trace": trace}}
        check = {"name": str(index), "parameter_count": count, "parameter_names": [p["name"] for p in params],
                 "parameter_shapes": [p["shape"] for p in params], "geometry_slots": [2, 7 + index], "rates": rates,
                 "invalid_rates_preserve_tape": True, "frozen_parameter_bits": True, "embedding_parameters_learn": True,
                 "metric_controls": {"detached_embedding_gradient_separation": 1e-6, "pullback_contrast_relative_l2": .0001},
                 "trace": [{**copy.deepcopy(s), "revision": n} for n, s in enumerate(trace, 1)]}
        reference["cases"].append(case)
        report["checks"].append(check)
    return copy.deepcopy(reference), copy.deepcopy(report)


class VerificationTests(unittest.TestCase):
    def test_complete_trace(self):
        self.assertTrue(module.verify(*inputs())["passed"])

    def test_missing_case_or_step(self):
        for kind in ("case", "step"):
            ref, report = inputs()
            (report["checks"] if kind == "case" else report["checks"][1]["trace"]).pop()
            with self.assertRaises(ValueError):
                module.verify(ref, report)

    def test_frozen_change_below_absolute_tolerance(self):
        ref, report = inputs()
        report["checks"][0]["trace"][8]["parameters"][2][0] += 1e-7
        with self.assertRaisesRegex(ValueError, "frozen parameter bits"):
            module.verify(ref, report)

    def test_severed_or_wrong_tiny_derivative(self):
        for bad in (0., 0.5e-7, -1e-7):
            ref, report = inputs()
            report["checks"][0]["trace"][10]["geometry_gradients"][0][0] = bad
            with self.assertRaises(ValueError):
                module.verify(ref, report)

    def test_missing_tensors_or_gradient_values(self):
        for field in ("parameters", "geometry_gradients", "embedding_output_gradient"):
            ref, report = inputs()
            report["checks"][1]["trace"][-1][field].pop()
            with self.assertRaises(ValueError):
                module.verify(ref, report)

    def test_invalid_scalar(self):
        for bad in (True, float("nan"), float("inf"), 1e300):
            ref, report = inputs()
            report["checks"][1]["trace"][-1]["embedding_output_gradient"][0] = bad
            with self.assertRaises(ValueError):
                module.verify(ref, report)

    def test_boolean_revision(self):
        ref, report = inputs()
        report["checks"][0]["trace"][0]["revision"] = True
        with self.assertRaisesRegex(ValueError, "revision sequence"):
            module.verify(ref, report)

    def test_wrong_rates(self):
        ref, report = inputs()
        report["checks"][0]["rates"][2] = .125
        with self.assertRaisesRegex(ValueError, "rate policy"):
            module.verify(ref, report)

    def test_failed_runtime_control(self):
        ref, report = inputs()
        report["checks"][1]["invalid_rates_preserve_tape"] = False
        with self.assertRaisesRegex(ValueError, "runtime controls"):
            module.verify(ref, report)

    def test_changed_gates(self):
        ref, report = inputs()
        ref["tolerance"]["atol"] = 1.
        with self.assertRaisesRegex(ValueError, "changed gates"):
            module.verify(ref, report)


if __name__ == "__main__":
    unittest.main()
