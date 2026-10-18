"""No GPU/model required for exact-comparison diagnostic regressions."""
import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest

API = runpy.run_path(str(Path(__file__).with_name("byte_study_diagnostics.py")))


class DiagnosticTests(unittest.TestCase):
    def test_small_float_differences_remain_exact_failures(self):
        before = json.dumps({"cases": [{"validation": [5.501455783843994]}]})
        after = json.dumps({"cases": [{"validation": [5.501457691192627]}]})
        diff = API["exact_report_differences"](before, after)
        self.assertEqual(len(diff), 1)
        self.assertEqual(diff[0]["path"], "$.cases[0].validation[0]")
        self.assertNotEqual(diff[0]["expected"], diff[0]["actual"])
        self.assertEqual(API["exact_report_differences"](before, before), [])

    def test_structure_types_encoding_and_invalid_json_are_distinct(self):
        checks = (("{}", '{"extra":1}', "keys"), ("[]", "[1]", "length"),
                  ("true", "1", "type"), ('{"x":1}', '{ "x": 1 }', "serialization_only"),
                  ("invalid", "{}", "invalid_json"))
        for before, after, kind in checks:
            self.assertEqual(API["exact_report_differences"](before, after)[0]["kind"], kind)

    def test_difference_count_and_strings_are_bounded(self):
        diff = API["exact_report_differences"](json.dumps([0] * 100), json.dumps([1] * 100), limit=3)
        self.assertEqual(len(diff), 3)
        long = API["exact_report_differences"](json.dumps("a" * 1000), json.dumps("b" * 1000))
        self.assertEqual(len(long[0]["expected"]), 160)
        for limit in (0, 65, True):
            with self.assertRaises(ValueError):
                API["exact_report_differences"]("[]", "[]", limit=limit)

    def test_receipts_retain_original_bytes_and_never_overwrite(self):
        with tempfile.TemporaryDirectory() as name:
            report = API["retain_detached_failure"](name, b'{ "fixture": true }', '{"x":1}',
                                                     ['{"x":2}', '{"x":1}'], 0)
            target = Path(name) / "round-0"
            self.assertEqual((target / "request.json").read_bytes(), b'{ "fixture": true }')
            for filename, expected in report["files"].items():
                self.assertEqual(expected, "sha256:" + hashlib.sha256((target / filename).read_bytes()).hexdigest())
            self.assertEqual(report["comparison"], "exact_json_bytes_no_tolerance")
            self.assertEqual(report["differences"][1], [])
            self.assertEqual(json.loads((target / "summary.json").read_text()), report)
            with self.assertRaises(FileExistsError):
                API["retain_detached_failure"](name, b"{}", "{}", ["{}", "{}"], 0)


if __name__ == "__main__":
    unittest.main()
