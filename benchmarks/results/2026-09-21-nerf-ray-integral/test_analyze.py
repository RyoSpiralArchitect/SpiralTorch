import copy
import importlib.util
import math
from pathlib import Path
import struct
import unittest

spec = importlib.util.spec_from_file_location("analyze", Path(__file__).with_name("analyze.py"))
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def reference():
    cases = []
    for batch, samples, varying in analyze.GRID:
        meta = {"batch": batch, "samples": samples, "varying": varying,
                "position_bands": 2, "direction_bands": 1, "learning_rate": f32(0.01),
                "origins": [0.0] * (batch * 3), "directions": [1.0] * (batch * 3),
                "targets": [0.0] * (batch * 3), "bounds": [0.0, f32(1.05)] * batch,
                "parameters": {"test": {"shape": [1, 1], "values": [0.5]}}}
        cases.append({"metadata": meta, "values": [0.1] * (batch * 3), "elapsed_ns": [10.0] * 9})
    constant = [{"samples": samples, "width": f32(width),
                 "actual": [f32(f32(c) * -math.expm1(-2 * f32(width))) for c in [0.4, 0.2, 0.1]]}
                for samples in [1, 8, 64] for width in [0.0, 1e-8, 1.0, 20.0]]
    return {"runtime": "native-rust-cpu", "warmups": 5, "intervals": 9, "repetitions": 4,
            "cases": cases, "contract": {"constant_cases": constant, "layout_equal": True,
            "training": {"before": cases[2]["metadata"],
                         "after": {"test": {"shape": [1, 1], "values": [0.5]}},
                         "loss": 0.1, "avg_transmittance": 0.2}}}


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.reference = reference()
        self.report = copy.deepcopy(self.reference)

    def test_complete_report(self):
        self.assertEqual(len(analyze.validate(self.report, self.reference)["cases"]), 18)

    def test_f64_json_noise_preserves_exact_f32_identity(self):
        self.report["cases"][0]["metadata"]["bounds"][1] -= 2e-16
        analyze.validate(self.report, self.reference)

    def test_one_f32_bit_change_is_rejected(self):
        self.report["cases"][0]["metadata"]["bounds"][1] += 2**-23
        with self.assertRaises(AssertionError):
            analyze.validate(self.report, self.reference)

    def test_missing_and_duplicate_conditions_are_rejected(self):
        for change in [lambda r: r["cases"].pop(), lambda r: r["cases"].__setitem__(1, r["cases"][0])]:
            report = copy.deepcopy(self.reference)
            change(report)
            with self.assertRaises(AssertionError):
                analyze.validate(report, self.reference)

    def test_nonfinite_and_wrong_outputs_are_rejected(self):
        for value in [math.nan, math.inf, -math.inf, 100.0]:
            self.report["cases"][0]["values"][0] = value
            with self.assertRaises(AssertionError):
                analyze.validate(self.report, self.reference)

    def test_invented_zero_width_opacity_is_rejected(self):
        self.report["contract"]["constant_cases"][0]["actual"][0] = 1e-9
        with self.assertRaises(AssertionError):
            analyze.validate(self.report, self.reference)

    def test_invalid_timing_is_rejected(self):
        self.report["cases"][0]["elapsed_ns"][0] = 0
        with self.assertRaises(AssertionError):
            analyze.validate(self.report, self.reference)

    def test_layout_and_training_failures_are_rejected(self):
        self.report["contract"]["layout_equal"] = False
        with self.assertRaises(AssertionError):
            analyze.validate(self.report, self.reference)
        self.report = copy.deepcopy(self.reference)
        self.report["contract"]["training"]["after"]["test"]["values"][0] += 0.1
        with self.assertRaises(AssertionError):
            analyze.validate(self.report, self.reference)


if __name__ == "__main__":
    unittest.main()
