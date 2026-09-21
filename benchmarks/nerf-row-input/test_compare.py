"""Synthetic protocol mutation gates, not measured performance evidence."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare
from test_contract import fixture


def source():
    report = fixture()
    report.update(schema=compare.SCHEMA, comparison=compare.COMPARISON)
    for case in report["cases"]:
        for interval in case["intervals"]:
            interval["route"] = {"staged": "packed", "direct": "rows"}[interval["route"]]
    return report


class ComparisonTests(unittest.TestCase):
    def test_adaptation_is_nonmutating_and_preserves_input_bytes(self):
        raw = source()
        before = copy.deepcopy(raw)
        adapted = compare.adapt(raw)
        self.assertEqual(raw, before)
        self.assertEqual(adapted["cases"][0]["intervals"][0]["route"], "separate")
        self.assertEqual(adapted["cases"][0]["ray_inputs"], raw["cases"][0]["ray_inputs"])

    def test_old_and_unknown_route_identities_fail_closed(self):
        for raw in [fixture(), {**source(), "comparison": compare.base.COMPARISON}]:
            with self.assertRaises((ValueError, KeyError)):
                compare.adapt(raw)
        for route in ["staged", "direct", "separate", "single", "mps", "unknown"]:
            raw = source()
            raw["cases"][0]["intervals"][0]["route"] = route
            with self.assertRaises(ValueError):
                compare.adapt(raw)

    def test_missing_duplicate_nonfinite_and_bad_output_are_rejected(self):
        for kind in ["missing", "duplicate", "nan", "bad_output"]:
            raw = source()
            if kind == "missing":
                raw["cases"].pop()
            elif kind == "duplicate":
                raw["cases"][0] = raw["cases"][1]
            elif kind == "nan":
                raw["cases"][0]["intervals"][0]["elapsed_ms"] = float("nan")
            else:
                raw["cases"][0]["last_outputs"][0][0] = 99.
            with self.assertRaises(ValueError):
                compare.adapt(raw)

    def test_summary_is_renamed_and_recomputes_every_interval(self):
        result = compare.analyze([source()] * 3, [source()] * 3, [fixture("torch")] * 3)
        compare.validate_summary(result)
        for old in ["native_staged", "browser_direct", "native_separate", "browser_single"]:
            self.assertNotIn(old, json.dumps(result))
        self.assertEqual(result["descriptive_summary"]["native"]["geomean_packed_over_rows"], 1.)
        for kind in ["median", "missing", "nan", "aggregate"]:
            bad = copy.deepcopy(result)
            if kind == "median":
                bad["cases"][0]["summary"][0]["median_interval_ms"]["native_packed"] = 99.
            elif kind == "missing":
                bad["cases"][0]["intervals"].pop()
            elif kind == "nan":
                bad["cases"][0]["max_scaled_errors"]["native_rows"] = float("nan")
            else:
                bad["descriptive_summary"]["native"]["geomean_packed_over_rows"] = 99.
            with self.assertRaises(ValueError):
                compare.validate_summary(bad)

    def test_input_drift_and_incomplete_rounds_are_not_hidden(self):
        native = [source() for _ in range(3)]
        browser = [source() for _ in range(3)]
        torch = [fixture("torch") for _ in range(3)]
        with self.assertRaises(ValueError):
            compare.analyze(native[:2], browser, torch)
        browser[2]["cases"][0]["ray_inputs"][0][0] = 2.
        with self.assertRaises(ValueError):
            compare.analyze(native, browser, torch)


if __name__ == "__main__":
    unittest.main()
