import importlib.util
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("runner", Path(__file__).with_name("run_byte_corpus_segments.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FakeStudy:
    total_updates = 4
    calls = []
    fail = False

    @classmethod
    def from_json(cls, raw):
        study = cls()
        study.request_sha256 = runner.digest(raw)
        return study

    def validate_segment(self, stop):
        if not 0 <= stop <= self.total_updates:
            raise ValueError("out of range")

    def advance_wgpu(self, stop, *, resume):
        self.calls.append(stop)
        if self.fail:
            raise RuntimeError("simulated GPU failure")
        checkpoint = SimpleNamespace(completed_updates=stop, request_sha256=self.request_sha256,
                                     to_json=lambda: json.dumps({"cursor": stop}))
        return SimpleNamespace(checkpoint=checkpoint, report_json='{"synthetic":true}')


class LaunchTests(unittest.TestCase):
    def setUp(self):
        FakeStudy.calls = []
        FakeStudy.fail = False
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.request = self.root / "request.json"
        self.request.write_bytes(b'{"synthetic":true}\n')
        self.output = self.root / "run"

    def test_exclusive_claim_before_gpu_and_private_outputs(self):
        receipts = runner.run(self.request, self.output, [0, 2, 4], study_type=FakeStudy)
        self.assertEqual(FakeStudy.calls, [0, 2, 4])
        self.assertEqual([r["stop_after"] for r in receipts], [0, 2, 4])
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o700)
        self.assertTrue(json.loads((self.output / "claim.json").read_bytes())["before_gpu_launch"])
        for path in self.output.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        with self.assertRaises(FileExistsError):
            runner.run(self.request, self.output, [4], study_type=FakeStudy)
        self.assertEqual(FakeStudy.calls, [0, 2, 4])

    def test_failure_keeps_claim_and_no_completion(self):
        FakeStudy.fail = True
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            runner.run(self.request, self.output, [4], study_type=FakeStudy)
        self.assertTrue((self.output / "claim.json").exists())
        self.assertTrue((self.output / "failed.json").exists())
        self.assertFalse((self.output / "completed.json").exists())
        with self.assertRaises(FileExistsError):
            runner.run(self.request, self.output, [4], study_type=FakeStudy)
        self.assertEqual(FakeStudy.calls, [4])

    def test_invalid_stops_do_not_claim_or_launch(self):
        for stops in ([], [2], [4, 4], [4, 2], [-1, 4], [True, 4], list(range(33))):
            with self.subTest(stops=stops), self.assertRaises(ValueError):
                runner.run(self.request, self.output, stops, study_type=FakeStudy)
        self.assertFalse(self.output.exists())
        self.assertEqual(FakeStudy.calls, [])


if __name__ == "__main__":
    unittest.main()
