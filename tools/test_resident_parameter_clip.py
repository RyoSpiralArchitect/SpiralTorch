"""Negative controls for the numerical comparison, without requiring a GPU."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("clip_check", Path(__file__).with_name("check_resident_parameter_clip.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.reference = dict(schema="spiraltorch.resident_parameter_clip.v1", passed=True, cases=[])
        for name, (raw, active, limit) in zip(module.NAMES, module.inputs(torch), strict=True):
            vectors = [torch.tensor(v, dtype=torch.float32) for v in raw]
            limit = torch.tensor(limit, dtype=torch.float32).item()
            norm = sum(float(x)**2 for v, on in zip(vectors, active) if on for x in v)**.5
            scale = min(1., limit/norm) if norm > 2**-23 else 1.
            if abs(torch.tensor(scale, dtype=torch.float32).item()-1) <= 2**-23:
                scale = 1.
            cls.reference["cases"].append(dict(name=name, inputs=[v.tolist() for v in vectors],
                active=active, limit=limit,
                clipped=[(v.double()*scale).float().tolist() if on else v.tolist() for v,on in zip(vectors,active)]))

    def test_complete_reference(self):
        self.assertEqual(len(module.check(self.reference)["cases"]), len(module.NAMES))

    def test_above_floor_cannot_be_noop(self):
        report = copy.deepcopy(self.reference)
        case = report["cases"][module.NAMES.index("floor_above_9")]
        case["clipped"] = copy.deepcopy(case["inputs"])
        with self.assertRaises(AssertionError):
            module.check(report)

    def test_no_tiny_zero_substitute(self):
        report = copy.deepcopy(self.reference)
        report["cases"][5]["clipped"] = [[0., 0.], [0.]]
        with self.assertRaises(AssertionError):
            module.check(report)

    def test_no_vacuous_wide_input(self):
        report = copy.deepcopy(self.reference)
        report["cases"][4]["inputs"] = [[0., 0.], [0.]]
        with self.assertRaises(ValueError):
            module.check(report)

    def test_missing_case(self):
        report = copy.deepcopy(self.reference)
        report["cases"].pop()
        with self.assertRaises(ValueError):
            module.check(report)

    def test_frozen_and_signed_zero_cannot_change(self):
        report = copy.deepcopy(self.reference)
        report["cases"][1]["clipped"][0][1] = 0.
        with self.assertRaises(ValueError):
            module.check(report)


if __name__ == "__main__":
    unittest.main()
