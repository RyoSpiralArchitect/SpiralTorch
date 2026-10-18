"""Synthetic receipt tests; no model, framework or downloaded data required."""
import array
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native_gpt2_compare", ROOT / "benchmarks/results/2026-10-18-native-gpt2/compare.py")
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


class ProbeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = []
        for mode in ("native", "reference"):
            directory = Path(self.temp.name) / mode
            directory.mkdir()
            rows = []
            for case in ("meaning", "models", "spiraltorch"):
                for arm in ("base", "wave"):
                    data = array.array("f", [float(arm == "wave")] * 50257).tobytes()
                    name = case + "-" + arm + ".f32"
                    (directory / name).write_bytes(data)
                    rows.append({"case": case, "arm": arm, "prompt": case, "ids": [1], "shape": [1, 1, 50257],
                                 "nll": 1., "logits_file": name, "logits_sha256": hashlib.sha256(data).hexdigest(),
                                 "generated_ids": [1, 2], "generated_text": "synthetic fixture"})
            report = {"schema": "spiraltorch.native_gpt2_probe.v1", "mode": mode, "byteorder": sys.byteorder,
                      "source_sha256": {"model.safetensors": "synthetic"}, "steps": 1, "base_model_identity": "synthetic",
                      "geometry_artifact_id": "synthetic", "scope": "test_only", "optional_ml_imports_forbidden": True,
                      "backend": mode, "torch": "unused", "transformers": "unused", "rows": rows}
            path = directory / "report.json"
            path.write_text(json.dumps(report))
            self.paths.append(path)

    def alter_reference(self, change):
        report = json.loads(self.paths[1].read_text())
        change(report)
        self.paths[1].write_text(json.dumps(report))

    def test_matches_all_logits_and_intervention_differences(self):
        report = COMPARE.compare(*self.paths)
        self.assertEqual(len(report["rows"]), 6)
        self.assertTrue(all(row["logit_error"]["max_abs"] == 0. for row in report["rows"]))
        self.assertTrue(all(row["greedy_ids_equal"] for row in report["rows"]))
        self.assertTrue(all(row["native_wave_logit_delta"]["max_abs"] == 1. for row in report["interventions"]))

    def test_rejects_unmatched_source_and_partial_cases(self):
        original = self.paths[1].read_bytes()
        for mutate in (lambda r: r["source_sha256"].update({"model.safetensors": "different"}),
                       lambda r: r["rows"].pop(), lambda r: r["rows"].append(r["rows"][0]),
                       lambda r: r["rows"][0].update({"generated_ids": [9, 2]})):
            self.paths[1].write_bytes(original)
            self.alter_reference(mutate)
            with self.assertRaises(ValueError):
                COMPARE.compare(*self.paths)

    def test_rejects_tampered_logits_and_paths(self):
        path = self.paths[1].parent / "meaning-base.f32"
        path.write_bytes(b"\0")
        with self.assertRaisesRegex(ValueError, "length or digest"):
            COMPARE.compare(*self.paths)
        self.alter_reference(lambda r: r["rows"][0].update({"logits_file": "../foreign.f32"}))
        with self.assertRaisesRegex(ValueError, "sibling"):
            COMPARE.compare(*self.paths)

    def test_reports_greedy_disagreement_without_suppressing_it(self):
        self.alter_reference(lambda r: r["rows"][0].update({"generated_ids": [1, 3]}))
        report = COMPARE.compare(*self.paths)
        self.assertEqual(sum(not row["greedy_ids_equal"] for row in report["rows"]), 1)

    def test_rejects_changed_inputs_between_arms_even_when_runtimes_match(self):
        for path in self.paths:
            report = json.loads(path.read_text())
            report["rows"][1].update({"prompt": "different", "ids": [3], "generated_ids": [3, 2]})
            path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "intervention inputs differ"):
            COMPARE.compare(*self.paths)


if __name__ == "__main__":
    unittest.main()
