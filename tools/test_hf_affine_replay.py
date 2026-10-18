"""Fast stdlib checks; no model import, training, downloads or credentials."""
import json
from pathlib import Path
import runpy
import tempfile
import unittest

REPLAY = Path(__file__).resolve().parents[1] / "benchmarks/results/2026-10-18-hf-affine-control/replay.py"
execute_cases = runpy.run_path(str(REPLAY))["execute_cases"]
CASES = [{"arm": arm, "seed": seed} for seed in (41, 43, 47) for arm in ("ordinary", "wave_gate")]


class ReplayTests(unittest.TestCase):
    def test_cpu_and_mps_companion_contract_rejects_device_mismatch(self):
        names = runpy.run_path(str(REPLAY.with_name("verify_continuation.py")))["companion_names"]
        self.assertNotIn("spiraltorch-mps-rng.pt", names([{"device": "cpu"}] * 3))
        self.assertIn("spiraltorch-mps-rng.pt", names([{"device": "mps"}] * 3))
        for results in ([{"device": "cpu"}, {"device": "mps"}], [{"device": "cuda"}], [{}]):
            with self.assertRaises(ValueError):
                names(results)

    def test_manifest_precedes_execution_and_all_cases_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            calls = []
            def execute(case):
                manifest = json.loads(path.read_text())
                self.assertEqual(len(manifest["runs"]), 6)
                self.assertEqual(manifest["runs"][len(calls)]["status"], "running")
                calls.append(case)
                return dict(case, value=1)
            records = execute_cases(path, CASES, execute, lambda records: self.assertEqual(len(records), 6))
            manifest = json.loads(path.read_text())
            self.assertEqual(calls, CASES)
            self.assertEqual(manifest["status"], "completed")
            self.assertTrue(all(r["status"] == "completed" for r in manifest["runs"]))
            self.assertEqual(records, [r["result"] for r in manifest["runs"]])

    def test_failure_and_cancellation_preserve_completed_and_unrun_cases(self):
        for error in (RuntimeError, KeyboardInterrupt):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "manifest.json"
                calls = []
                def execute(case):
                    calls.append(case)
                    if len(calls) == 2:
                        raise error("private text must not enter public manifest")
                    return dict(case, value=1)
                with self.assertRaises(error):
                    execute_cases(path, CASES, execute, lambda _: self.fail("must not qualify"))
                manifest = json.loads(path.read_text())
                self.assertEqual(len(calls), 2)
                self.assertEqual(manifest["status"], "failed")
                self.assertEqual(manifest["error_type"], error.__name__)
                self.assertEqual([r["status"] for r in manifest["runs"]],
                                 ["completed", "failed"] + ["not_started"] * 4)
                self.assertNotIn("private text", path.read_text())

    def test_cross_run_validation_failure_does_not_erase_successful_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            def validate(_):
                raise ValueError("dataset drift")
            with self.assertRaises(ValueError):
                execute_cases(path, CASES, lambda case: dict(case, value=1), validate)
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["error_type"], "ValueError")
            self.assertTrue(all(r["status"] == "completed" for r in manifest["runs"]))


if __name__ == "__main__":
    unittest.main()
