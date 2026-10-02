"""Lightweight fail-closed coverage checks; no torch or GPU import required."""
import copy
import unittest
from bench_attention_chain_vs_torch import validate_report, projection_options, require_projection


class ReportTests(unittest.TestCase):
    def test_projection_controls_are_explicit_and_confirmed(self):
        self.assertEqual(projection_options(["st=register16"], {"st":"binary"}), {"st":"register16"})
        for values in (["st"], ["other=scalar"], ["st=typo"], ["st=scalar", "st=scalar"]):
            with self.assertRaises(ValueError):
                projection_options(values, {"st":"binary"})
        for report in ({}, {"projection":"scalar"}):
            with self.assertRaises(ValueError):
                require_projection(report, "register16")
        require_projection({"projection":"register16"}, "register16")

    def setUp(self):
        self.fixture = {"scenarios": [{"name": "s", "cases": [{"name": "c"}]}]}
        self.report = dict(status="passed", samples_per_route=3, warmup=2, burst=4,
            cases=[dict(name="s/c", samples=[dict(block=b, route=r,
                forwards=4 if r == "resident" else 1, elapsed_ms=1., max_abs_error=0.)
                for b in range(3) for r in ("resident", "host_to_host")])])

    def validate(self, report):
        validate_report(report, self.fixture, 3, 2, 4)

    def test_complete(self):
        self.validate(self.report)

    def test_missing_duplicate_or_failed_coverage(self):
        for mutate in (lambda r: r["cases"].clear(),
                       lambda r: r["cases"].append(copy.deepcopy(r["cases"][0])),
                       lambda r: r.update(status="failed"),
                       lambda r: r["cases"][0]["samples"].pop(),
                       lambda r: r["cases"][0]["samples"].append(r["cases"][0]["samples"][0])):
            report = copy.deepcopy(self.report)
            mutate(report)
            with self.assertRaises(ValueError):
                self.validate(report)

    def test_invalid_samples_or_recipe(self):
        for key, value in (("elapsed_ms", float("nan")), ("elapsed_ms", 0),
                           ("max_abs_error", float("inf")), ("forwards", 0), ("route", "unknown")):
            report = copy.deepcopy(self.report)
            report["cases"][0]["samples"][0][key] = value
            with self.assertRaises(ValueError):
                self.validate(report)
        self.report["burst"] = 8
        with self.assertRaises(ValueError):
            self.validate(self.report)


if __name__ == "__main__":
    unittest.main()
