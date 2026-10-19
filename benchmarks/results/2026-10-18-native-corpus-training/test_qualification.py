"""Synthetic transport-only regression tests for the qualification comparator."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest

H = runpy.run_path(str(Path(__file__).resolve().with_name("qualify.py")))


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.args = SimpleNamespace(cohort=self.root, output=self.root / "comparison.json")
        self.write("data.json", {"synthetic": True})
        self.write("contract.json", {"source_sha256": H["source_hashes"](), "recipe": H["RECIPE"],
            "gates": H["GATES"], "model_sha256": H["MODEL"], "chapter_sha256": H["CHAPTER"],
            "base_model_identity": H["BASE"], "initial_geometry_artifact_id": H["ARTIFACT"],
            "reference_ce_dtype": "f64", "total_updates": 2, "resume_from_update": 2,
            "input_sha256": {"data.json": H["digest"](self.root / "data.json")}})
        for name in ("recipe.json", "initial.json"):
            self.write(name, {"synthetic": True})
        runtime = {"native_extension_sha256": "test-binary", "optional_ml_imports_forbidden": True,
                   "qualification_contract_sha256": H["digest"](self.root / "contract.json")}
        summary = {"total_updates": 2, "completed_updates": 2, "exhausted": True,
                   "data_id": "test-data", "initial": {"updates": 0}, "config": {}, "plan": {},
                   "base_model_identity": H["BASE"]}
        geometry = {"placements": [{"path": "test", "tensors": [
            {"name": "gate", "role": "parameter", "shape": [1], "values": [.1]}]}]}
        rows = [{"update": i+1, "epoch": 0, "window_start": i, "window_end": i+1,
                 "target_tokens": 3, "loss": 1.0} for i in range(2)]
        for name in ("native-full", "native-resumed", "reference"):
            (self.root / name).mkdir()
        for name in ("native-full", "native-resumed"):
            self.write(name + "/runtime.json", runtime)
            self.write(name + "/result.json", dict(summary, initial={"updates": 0 if name == "native-full" else 2}))
            self.write(name + "/checkpoint-000002.json", {"geometry_json": json.dumps(geometry)})
            (self.root / name / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in rows) if name == "native-full" else "")
        self.write("reference/final-geometry.json", geometry)
        self.write("reference/result.json", dict(runtime, cross_entropy_dtype="f64", rows=rows,
                                                 base_parameter_versions_unchanged=True))

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def change(self, name, key, value):
        record = json.loads((self.root / name).read_text())
        record[key] = value
        self.write(name, record)

    def compare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            H["compare"](self.args)

    def test_matching_fixture_is_separately_labeled(self):
        self.compare()
        report = json.loads(self.args.output.read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(report["comparison_kind"], "posthoc_f64_ce_control")

    def test_loss_precision_binary_or_runtime_mismatch_fails_closed(self):
        path = "reference/result.json"
        original = json.loads((self.root / path).read_text())
        for key, value in (("cross_entropy_dtype", "f32"), ("native_extension_sha256", "other"),
                           ("qualification_contract_sha256", "other"), ("base_parameter_versions_unchanged", False)):
            self.change(path, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.compare()
            self.assertFalse(self.args.output.exists())
            self.write(path, original)

    def test_parameter_failure_is_retained_without_relaxing_gate(self):
        record = json.loads((self.root / "reference/final-geometry.json").read_text())
        record["placements"][0]["tensors"][0]["values"] = [.2]
        self.write("reference/final-geometry.json", record)
        with self.assertRaisesRegex(ValueError, "comparison failed"):
            self.compare()
        self.assertFalse(json.loads(self.args.output.read_text())["passed"])

    def test_native_resume_runtime_mismatch_is_rejected(self):
        self.change("native-resumed/runtime.json", "native_extension_sha256", "other")
        with self.assertRaisesRegex(ValueError, "runtime"):
            self.compare()

    def test_reference_order_must_match(self):
        record = json.loads((self.root / "reference/result.json").read_text())
        record["rows"].reverse()
        self.write("reference/result.json", record)
        with self.assertRaisesRegex(ValueError, "chronology"):
            self.compare()


if __name__ == "__main__":
    unittest.main()
