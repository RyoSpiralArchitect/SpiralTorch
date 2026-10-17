"""Reject incomplete or re-labelled synthetic conversion evidence."""

import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("publish", Path(__file__).with_name("publish.py"))
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        attention = []
        for sign in (0, 0x80000000):
            for high in (0x40800000, 0x40800001):
                for tail in (-1, 0, 1):
                    bits = (high - (2 << 23) + int(tail > 0 or (tail == 0 and high & 1))) ^ sign
                    attention.append(dict(sign=sign, high=high, tail_sign=tail,
                                          expected_bits=bits, actual_bits=bits, passed=True))
        self.fixture = {"cases": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]}
        self.report = dict(schema="spiraltorch.wide_rounding.validation.v1",
                           fixture_sha256=publish.FIXTURE_SHA, case_count=2,
                           actual_bits=[0, 1], mismatch_indices=[], attention=attention, passed=True)

    def test_preserves_actual_failure(self):
        publish.validate_rounding(self.report, self.fixture)
        self.report.update(actual_bits=[0, 0], mismatch_indices=[1], passed=False)
        publish.validate_rounding(self.report, self.fixture)
        self.report["passed"] = True
        with self.assertRaises(ValueError):
            publish.validate_rounding(self.report, self.fixture)

    def test_rejects_missing_or_repeated_cases(self):
        for field, value in (("actual_bits", [0]), ("case_count", 1),
                             ("attention", self.report["attention"][:-1]),
                             ("attention", self.report["attention"][:1] * 12)):
            report = copy.deepcopy(self.report)
            report[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                publish.validate_rounding(report, self.fixture)

    def test_checks_expectation_and_failure_indices(self):
        for field in ("actual_bits", "expected_bits"):
            report = copy.deepcopy(self.report)
            report["attention"][0][field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                publish.validate_rounding(report, self.fixture)
        self.report["mismatch_indices"] = [0]
        with self.assertRaises(ValueError):
            publish.validate_rounding(self.report, self.fixture)


if __name__ == "__main__":
    unittest.main()
