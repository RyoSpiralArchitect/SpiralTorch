"""Synthetic validation of learning-probe receipts, without an ML framework."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/results/2026-10-18-native-gpt2-learning"


def module(name):
    spec = importlib.util.spec_from_file_location("native_learning_" + name, ROOT / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


COMPARE = module("compare")
PROBE = module("probe")


class LearningProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = []
        for mode in ("native", "reference"):
            directory = Path(self.temp.name) / mode
            directory.mkdir()
            state = {"base_model_identity": "synthetic", "policy": "test_only", "placements": [{
                "path": "site", "recipe": {"schema": "synthetic"}, "tensors": [
                    {"name": "gate", "shape": [1], "values": [.2], "role": "parameter", "requires_grad": True},
                    {"name": "buffer", "shape": [], "values": [1.], "role": "buffer", "requires_grad": False},
                ],
            }]}
            initial = directory / "initial.json"
            initial.write_text(json.dumps(state))
            identity = "sha256:" + COMPARE.digest(initial)
            rows = []
            for step in range(3):
                state["placements"][0]["tensors"][0]["values"] = [.2 - .001 * (step + 1)]
                saved = PROBE.save(directory / f"state-{step}.json", state)
                gradient = PROBE.save(directory / f"gradient-{step}.json", {"site": {"gate": [.1]}})
                rows.append({"step": step, "case": str(step), "text": "test", "ids": [1, 2],
                             "pre_nll": 1., "post_nll": .9, "state": saved, "gradient": gradient})
            report = {"schema": "spiraltorch.native_gpt2_learning_probe.v1", "mode": mode, "scope": "synthetic",
                      "source_sha256": {"model": "synthetic"}, "learning_rate": .01, "base_model_identity": "synthetic",
                      "initial_geometry_artifact_id": identity, "rows": rows, "optional_ml_imports_forbidden": True,
                      "artifact_continuation_exact": True, "base_probe_unchanged": True, "base_parameter_versions_unchanged": True,
                      "torch": "unused", "transformers": "unused"}
            path = directory / "report.json"
            path.write_text(json.dumps(report))
            self.paths.append(path)

    def test_complete_comparison(self):
        result = COMPARE.compare(*self.paths)
        self.assertEqual(len(result["rows"]), 3)
        self.assertEqual(result["rows"][0]["parameters"]["site"]["gate"]["gradient_error"]["max_abs"], 0.)

    def test_public_projection_keeps_cases_not_recoverable_operand_norms(self):
        private = COMPARE.compare(*self.paths)
        original = json.dumps(private)
        public = COMPARE.public_projection(private, "0" * 64)
        self.assertEqual(json.dumps(private), original)
        self.assertEqual(public["publication"]["private_comparison_sha256"], "0" * 64)
        self.assertEqual(public["publication"]["operand_norm_fields_removed"], 12)
        self.assertEqual(len(public["rows"]), 3)
        for raw, row in zip(private["rows"], public["rows"]):
            self.assertEqual({k: v for k, v in raw.items() if k != "parameters"},
                             {k: v for k, v in row.items() if k != "parameters"})
            parameter = row["parameters"]["site"]["gate"]
            for kind, metric in parameter.items():
                self.assertNotIn("native_l2", metric)
                self.assertNotIn("reference_l2", metric)
                self.assertEqual(metric["max_abs"], raw["parameters"]["site"]["gate"][kind]["max_abs"])
                self.assertEqual(metric["rms"], raw["parameters"]["site"]["gate"][kind]["rms"])
            self.assertTrue(parameter["gradient_error"]["native_nonzero"])
            self.assertTrue(parameter["gradient_error"]["reference_nonzero"])

    def test_preserves_exact_artifact_bytes(self):
        class Artifact:
            def to_json(self):
                return '{"value":0.2}'
        path = Path(self.temp.name) / "exact.json"
        result = PROBE.save_artifact(path, Artifact())
        self.assertEqual(path.read_bytes(), b'{"value":0.2}')
        self.assertEqual(result["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_rejects_different_inputs_and_missing_steps(self):
        original = self.paths[1].read_text()
        for change in (lambda r: r["rows"][0].update({"ids": [1, 3]}), lambda r: r["rows"].pop(),
                       lambda r: r.update({"learning_rate": .02}), lambda r: r["rows"][1].update({"step": 2})):
            report = json.loads(original)
            change(report)
            self.paths[1].write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                COMPARE.compare(*self.paths)

    def test_rejects_tampered_artifact(self):
        (self.paths[1].parent / "gradient-0.json").write_text('{}')
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            COMPARE.compare(*self.paths)

    def test_rejects_frozen_buffer_changes_even_with_matching_hash(self):
        path = self.paths[1].parent / "state-0.json"
        state = json.loads(path.read_text())
        state["placements"][0]["tensors"][1]["values"] = [2.]
        path.write_text(json.dumps(state))
        report = json.loads(self.paths[1].read_text())
        report["rows"][0]["state"]["sha256"] = COMPARE.digest(path)
        self.paths[1].write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "frozen parameter or buffer"):
            COMPARE.compare(*self.paths)


if __name__ == "__main__":
    unittest.main()
