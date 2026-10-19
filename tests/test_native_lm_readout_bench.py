"""Report matching is strict; timing ratios do not excuse mismatched workloads."""
import copy
import json
from pathlib import Path
import runpy
import tempfile
import unittest

helper = runpy.run_path(str(Path(__file__).resolve().parents[1]/"tools/compare_native_lm_readout.py"))


def reports():
    cases = []
    for rows in (16, 64):
        for active in (1, 4, rows):
            cases.append({"id": f"rows{rows}-selected{active}", "context_rows": rows, "selected_rows": active,
                          "ids": [1]*(rows+1), "labels": [-100]*(rows+1-active)+[1]*active,
                          "loss": 2., "timings_ms": [1., 2., 3., 4., 5.], "median_ms": 3., "first_call_ms": 7.})
    base = {"schema": "spiraltorch.native_lm_readout_bench.v1", "source_sha256": {"weights": "a"*64},
            "script_sha256": "b"*64, "vocabulary": 3, "label_smoothing": .1,
            "platform": "fixture", "python": "fixture", "thread_env": {"OMP_NUM_THREADS": "1"},
            "boundary": "fixture", "order": [list(range(6))]*5, "cases": cases}
    return {name: {**copy.deepcopy(base), "engine": engine,
                   "runtime": {"optional_ml_imports_forbidden": True, "device": "cpu", "native_extension_sha256": "c"*64,
                               "torch": "fixture-torch", "transformers": "fixture-transformers", "torch_threads": 1}}
            for name, engine in helper["ENGINES"].items()}


class ReadoutComparisonTests(unittest.TestCase):
    def compare(self, data):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for name, value in data.items():
                (directory/name).mkdir()
                (directory/name/"report.json").write_text(json.dumps(value))
            return helper["compare"](directory)

    def test_valid_reports_retain_every_sample(self):
        result = self.compare(reports())
        self.assertEqual(len(result["cases"]), 6)
        self.assertEqual(result["cases"][0]["native_before_over_after"], 1.)
        for case in result["cases"]:
            self.assertEqual(len(case["measurements"]), 4)
            self.assertTrue(all(len(m["timings_ms"]) == 5 for m in case["measurements"].values()))

    def test_mismatch_inputs_missing_cases_and_loss_fail(self):
        mutations = [lambda r: r["source_sha256"].update(weights="other"),
                     lambda r: r["cases"].pop(), lambda r: r["cases"][0]["ids"].__setitem__(0, 2),
                     lambda r: r["cases"][0].update(loss=2.01),
                     lambda r: r["cases"][0].update(selected_rows=2)]
        for mutate in mutations:
            data = reports()
            mutate(data["native-after"])
            with self.assertRaises(ValueError):
                self.compare(data)

    def test_invalid_timings_and_missing_import_guard_fail(self):
        mutations = [lambda r: r["cases"][0].update(median_ms=99.),
                     lambda r: r["cases"][0]["timings_ms"].pop(),
                     lambda r: r["cases"][0]["timings_ms"].__setitem__(0, float("nan")),
                     lambda r: r["runtime"].update(optional_ml_imports_forbidden=False)]
        for mutate in mutations:
            data = reports()
            mutate(data["native-before"])
            with self.assertRaises(ValueError):
                self.compare(data)

    def test_mismatched_or_missing_reference_runtimes_fail(self):
        for key, value in (("torch", "other-version"), ("transformers", "other-version"),
                           ("torch_threads", 2), ("torch_threads", True), ("device", "mps")):
            data = reports()
            data["torch-selected"]["runtime"][key] = value
            with self.assertRaises(ValueError):
                self.compare(data)
        for key in ("torch", "transformers"):
            data = reports()
            for name in ("torch-full", "torch-selected"):
                data[name]["runtime"][key] = None
            with self.assertRaises(ValueError):
                self.compare(data)
        for key in ("torch", "transformers", "torch_threads"):
            data = reports()
            for name in ("torch-full", "torch-selected"):
                del data[name]["runtime"][key]
            with self.assertRaises(ValueError):
                self.compare(data)
        data = reports()
        for name in ("torch-full", "torch-selected"):
            data[name]["runtime"]["torch_threads"] = 2
        with self.assertRaises(ValueError):
            self.compare(data)


if __name__ == "__main__":
    unittest.main()
