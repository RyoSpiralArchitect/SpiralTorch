"""CPU-only scheduling failures and a real owned timeout; never a GPU round."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import byte_release_schedule as schedule


class Schedule(unittest.TestCase):
    def simulate(self, defect=None, environment="local-metal"):
        clock = [0.0]
        budget = schedule.ENVIRONMENT_BUDGET_SECONDS[environment]
        calls, verified, deadlines = [], [], []
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "attempt"

            def command(index, directory):
                calls.append(index)
                return [sys.executable, "-I", "-B", "-c", "raise SystemExit(0)"]

            def run(cmd, *, timeout, stdout):
                deadlines.append(timeout)
                clock[0] += 100
                if defect == "timeout":
                    raise subprocess.TimeoutExpired(cmd, timeout)
                if defect == "cancel":
                    raise KeyboardInterrupt
                return subprocess.CompletedProcess(cmd, 1 if defect == "process" else 0)

            def verify(directory, index):
                verified.append(index)
                (directory / "synthetic-evidence.json").write_text('{"synthetic_only":true}')
                if defect == "missing_evidence":
                    raise FileNotFoundError("synthetic missing child evidence")
                if defect == "bool_evidence":
                    return True
                if defect == "expired_postlude":
                    clock[0] = budget + 1
                else:
                    clock[0] += 50
                return "b" * 64

            with patch.object(schedule, "owned_run", side_effect=run), patch.object(
                    schedule.time, "monotonic", side_effect=lambda: clock[0]):
                if defect:
                    with self.assertRaises(BaseException):
                        schedule.run_three_rounds(output, "a" * 64, environment, command, verify)
                else:
                    result = schedule.run_three_rounds(output, "a" * 64, environment, command, verify)
                    self.assertEqual(result, json.loads((output / "result.json").read_text()))
                    with self.assertRaises(FileExistsError):
                        schedule.run_three_rounds(output, "a" * 64, environment, command, verify)
            return json.loads((output / "result.json").read_text()), calls, verified, deadlines

    def test_three_processes_share_one_budget_without_release_promotion(self):
        for environment, budget in (("local-metal", 5400), ("hosted-metal", 14400)):
            with self.subTest(environment=environment):
                result, calls, verified, deadlines = self.simulate(environment=environment)
                self.assertEqual(calls, [0, 1, 2])
                self.assertEqual(verified, [0, 1, 2])
                self.assertEqual(deadlines, [budget, budget - 150, budget - 300])
                self.assertEqual(result["budget_seconds"], budget)
                self.assertEqual(result["status"], "verified_rounds")
                self.assertEqual(result["seconds"], 450)
                self.assertEqual([row["evidence_sha256"] for row in result["rounds"]], ["b" * 64] * 3)
                self.assertFalse(result["release_ready"])

    def test_declared_limits_keep_local_unchanged_and_two_candidates_bounded(self):
        self.assertEqual(schedule.ENVIRONMENT_BUDGET_SECONDS, {"local-metal": 5400, "hosted-metal": 14400})
        self.assertEqual(2 * sum(schedule.ENVIRONMENT_BUDGET_SECONDS.values()), 11 * 60 * 60)
        self.assertEqual(schedule.ROUNDS, 3)

    def test_every_first_failure_stops_later_rounds_and_is_retained(self):
        for environment in schedule.ENVIRONMENTS:
            for defect, status in (("timeout", "timeout"), ("cancel", "cancelled"),
                               ("process", "failed"), ("missing_evidence", "failed"),
                               ("bool_evidence", "failed"), ("expired_postlude", "timeout")):
                with self.subTest(defect=defect, environment=environment):
                    result, calls, verified, deadlines = self.simulate(defect, environment)
                    self.assertEqual(calls, [0])
                    self.assertEqual(len(deadlines), 1)
                    self.assertEqual(result["status"], status)
                    self.assertEqual(len(result["rounds"]), 1)
                    self.assertFalse(result["release_ready"])
                    if defect in ("timeout", "cancel", "process"):
                        self.assertEqual(verified, [])

    def test_invalid_admission_environment_and_command_never_spawn(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(schedule, "owned_run") as run:
            for digest, environment in (("a", "local-metal"), ("a" * 64, "cpu")):
                with self.assertRaises(ValueError):
                    schedule.run_three_rounds(Path(temporary) / environment, digest, environment,
                                              lambda *_: [], lambda *_: "b" * 64)
            with self.assertRaises(ValueError):
                schedule.run_three_rounds(Path(temporary) / "command", "a" * 64, "local-metal",
                                          lambda *_: [sys.executable, "-c", "pass"], lambda *_: "b" * 64)
            run.assert_not_called()

    def test_real_cpu_process_timeout_retains_partial_output_and_stops(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(schedule.ENVIRONMENT_BUDGET_SECONDS, {"local-metal": 2.0}):
            output = Path(temporary) / "timeout"
            calls = []

            def command(index, directory):
                calls.append(index)
                return [sys.executable, "-I", "-B", "-S", "-c",
                        "import time; print('synthetic-before-timeout', flush=True); time.sleep(10)"]

            with self.assertRaises(subprocess.TimeoutExpired):
                schedule.run_three_rounds(output, "a" * 64, "local-metal", command,
                                          lambda *_: self.fail("timed-out process cannot be verified"))
            self.assertEqual(calls, [0])
            self.assertEqual(json.loads((output / "result.json").read_text())["status"], "timeout")
            self.assertIn("synthetic-before-timeout", (output / "round-1/process.log").read_text())


if __name__ == "__main__":
    unittest.main()
