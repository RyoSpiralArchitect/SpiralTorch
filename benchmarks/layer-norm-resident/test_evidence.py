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

    def test_centered_browser_contract(self):
        report = dict(schema="spiraltorch.resident_layer_norm.browser.v2", status="passed",
                      page_errors=[], console_messages=[], cases=7, masks_per_case=8,
                      training_steps=400, intermediate_readbacks=0, scale_nullspace_cases=5,
                      epsilon_cancellation_cases=2, dynamic_range_variants=20, guard_checks=4,
                      adapter="AdapterInfo { backend: BrowserWebGpu }",
                      browser_adapter_probe=dict(is_fallback_adapter=False), first_loss=2., last_loss=1e-9)
        evidence.validate_centered_browser(report)
        for key, value in [("masks_per_case", 7), ("scale_nullspace_cases", 0),
                           ("dynamic_range_variants", 19), ("epsilon_cancellation_cases", 1),
                           ("last_loss", float("nan")), ("first_loss", float("inf")),
                           ("last_loss", -1.), ("guard_checks", 0),
                           ("intermediate_readbacks", 1), ("adapter", "CPU"),
                           ("browser_adapter_probe", dict(is_fallback_adapter=True))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                evidence.validate_centered_browser(dict(report, **{key: value}))


if __name__ == "__main__":
    unittest.main()
