import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("layer_norm_evidence", Path(__file__).with_name("evidence.py"))
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.report = {"cases": [dict(rows=r, cols=c, max_scaled_error=[0., 0., 0.],
            intervals=[dict(iteration=i, route=route, ms=1.) for i in range(18) for route in range(3)])
            for r, c in sorted(evidence.SHAPES)]}

    def test_complete_matrix(self):
        self.assertEqual(len(evidence.validate_intervals(self.report, (0, 1, 2))), 6)

    def test_missing_duplicate_and_nonfinite_evidence_rejected(self):
        for mutation in ("shape", "interval", "duplicate", "error_count", "bad_error", "bad_time"):
            report = copy.deepcopy(self.report)
            case = report["cases"][0]
            if mutation == "shape":
                report["cases"].pop()
            elif mutation == "interval":
                case["intervals"].pop()
            elif mutation == "duplicate":
                case["intervals"][0] = case["intervals"][1]
            elif mutation == "error_count":
                case["max_scaled_error"] = []
            elif mutation == "bad_error":
                case["max_scaled_error"][0] = float("nan")
            else:
                case["intervals"][0]["ms"] = float("inf")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                evidence.validate_intervals(report, (0, 1, 2))


if __name__ == "__main__":
    unittest.main()
