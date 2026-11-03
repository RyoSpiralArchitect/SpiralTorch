"""The bounded collector's scheduling/retention contracts, with no GPU imports."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_embedding_trio import ROLES, execute_once, exercise
from test_compare_embedding_bits import pair


class FakeStudy:
    def __init__(self, *, fail_serial=False, fail_map=False, change_other=False):
        self.fail_serial, self.fail_map, self.change_other = fail_serial, fail_map, change_other
        self.lock, self.join = threading.Lock(), threading.Barrier(2, timeout=5)
        self.calls, self.returns, self.reads = 0, 0, []

    def run_wgpu_with_embedding_probe(self, *selection):
        assert selection == ("geometry", 1, 2, 0)
        with self.lock:
            call = self.calls
            self.calls += 1
        if call:
            self.join.wait()
        outcome, bits = pair()
        outcome["report"]["adapter"] = 'AdapterInfo { device_type: IntegratedGpu, backend: Metal }'
        if self.fail_serial and not call:
            outcome["status"] = "error"
        if self.change_other and call == 1:
            outcome["report"]["cases"][1]["training"][0]["ce"] = 1
        owner = self

        class Observation:
            run_outcome_json = json.dumps(outcome)

            def read_bits_json(self):
                owner.reads.append(call)
                assert owner.returns == (3 if call else 1)
                if owner.fail_map and call:
                    raise ValueError("test map failed")
                return json.dumps(bits)
        with self.lock:
            self.returns += 1
        return Observation()


class Collector(unittest.TestCase):
    def test_reads_after_both_returns_and_detects_noncoordinate_difference(self):
        for diff in (False, True):
            study = FakeStudy(change_other=diff)
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                result = exercise(study, root)
                self.assertEqual(sorted(study.reads), [0, 1, 2])
                self.assertEqual(study.calls, 3)
                self.assertEqual(result["status"] == "difference", diff)
                self.assertTrue(all(not row["differences"] for row in result["coordinate"]["workers"]))
                self.assertTrue(all((root / role / "bits.json").exists() for role in ROLES))

    def test_serial_error_prevents_workers_and_saves_the_outcome(self):
        study = FakeStudy(fail_serial=True)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                exercise(study, root)
            self.assertEqual(study.calls, 1)
            self.assertTrue((root / "serial/run-outcome.json").exists())
            self.assertFalse((root / "worker0/entered.json").exists())

    def test_worker_map_error_preserves_all_reports_and_does_not_retry(self):
        study = FakeStudy(fail_map=True)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "map failed"):
                exercise(study, root)
            self.assertEqual(study.calls, 3)
            self.assertEqual(len(study.reads), 2)
            self.assertTrue((root / "full-reports.json").exists())
            self.assertTrue(all((root / role / "run-outcome.json").exists() for role in ROLES))

    def test_existing_attempt_is_never_launched_again(self):
        with tempfile.TemporaryDirectory() as temp, patch("run_embedding_trio.owned_run") as launch:
            with self.assertRaises(FileExistsError):
                execute_once(Path(temp), Path("unused.whl"), Path("unused.zip"))
            launch.assert_not_called()
