"""Model-free supervisor safety tests; never cargo or GPU execution."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

spec = importlib.util.spec_from_file_location("supervisor", Path(__file__).with_name("run_case24_once.py"))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.request = self.root / "request.json"
        self.checkpoint = self.root / "first.json"
        self.request.write_text("{}")
        self.checkpoint.write_text("{}")
        self.output = self.root / "evidence"
        self.calls = []

    def fake_stage(self, name, command, root, output, env, deadline):
        self.calls.append(name)
        self.assertEqual(deadline, 5200)
        if name == "admission_test":
            (output / "admission_test.stdout.log").write_text("test result: ok. 2 passed; 0 failed; 0 ignored;")
        if name == "build":
            binary = Path(env["CARGO_TARGET_DIR"]) / "debug/examples" / s.EXAMPLE
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"fake, never executed")

    def execute(self, **kwargs):
        options = dict(stage_runner=self.fake_stage, admission=lambda *a: None,
                       wall=lambda: 1000, clock=lambda: 100)
        options.update(kwargs)
        return s.execute(self.root, self.request, self.checkpoint, self.output, 1000, **options)

    def test_one_invocation_and_repeat_directory_rejected(self):
        self.assertEqual(self.execute(), 0)
        self.assertEqual(self.calls.count("diagnostic"), 1)
        for stage in ("source_patch", "rustc_version", "cargo_version"):
            self.assertEqual(self.calls.count(stage), 1)
            self.assertLess(self.calls.index(stage), self.calls.index("build"))
        historical = json.loads((self.output / "attempt.json").read_text())["historical_reference"]
        self.assertEqual(historical["failed_run_id"], 41297797362)
        self.assertEqual(historical["failed_artifact_id"], 13085278754)
        self.assertEqual(historical["failed_wheel_native_sha256"],
                         "44dc3ae1ef75104108517d6030ed374ee6d6610fd19501cc111b4985f3e5aeee")
        with self.assertRaises(FileExistsError):
            self.execute()
        self.assertEqual(self.calls.count("diagnostic"), 1)
        self.assertEqual(json.loads((self.output / "attempt.json").read_text())["status"], "completed")

    def test_wrong_hash_stops_before_cargo_or_gpu(self):
        self.assertEqual(self.execute(admission=s.admit), 1)
        self.assertEqual(self.calls, [])
        self.assertIn("SHA256 mismatch", (self.output / "attempt.json").read_text())

    def test_build_failure_never_invokes_gpu(self):
        def failure(name, *args):
            if name == "build":
                self.calls.append(name)
                raise RuntimeError("simulated build failure")
            return self.fake_stage(name, *args)
        self.assertEqual(self.execute(stage_runner=failure), 1)
        self.assertNotIn("diagnostic", self.calls)
        self.assertEqual(json.loads((self.output / "attempt.json").read_text())["binary_invocations"], 0)

    def test_empty_test_filter_is_not_admitted(self):
        def empty(name, *args):
            self.fake_stage(name, *args)
            if name == "admission_test":
                (self.output / "admission_test.stdout.log").write_text("test result: ok. 0 passed; 0 failed;")
        self.assertEqual(self.execute(stage_runner=empty), 1)
        self.assertNotIn("build", self.calls)
        self.assertNotIn("diagnostic", self.calls)

    def test_expired_job_stops_before_admission_or_stage(self):
        admitted = []
        self.assertEqual(self.execute(wall=lambda: 6101, admission=lambda *a: admitted.append(1)), 1)
        self.assertEqual(admitted, [])
        self.assertEqual(self.calls, [])

    def test_deadline_prevents_process_spawn(self):
        self.output.mkdir()
        with self.assertRaises(TimeoutError):
            s.run_stage("expired", ["must-not-exist"], self.root, self.output, os.environ.copy(), 1, clock=lambda: 2)
        self.assertEqual(json.loads((self.output / "expired.json").read_text())["status"], "timeout")

    @unittest.skipUnless(os.name == "posix", "POSIX session-group termination")
    def test_timeout_kills_group_and_retains_logs(self):
        self.output.mkdir()
        marker = self.root / "escaped-child"
        script = ("import subprocess,sys,time\n"
                  "print('started',flush=True)\n"
                  "subprocess.Popen([sys.executable,'-c',"
                  f"{('import time,pathlib;time.sleep(1);pathlib.Path(' + repr(str(marker)) + ').write_text(chr(120))')!r}])\n"
                  "time.sleep(30)\n")
        with self.assertRaises(subprocess.TimeoutExpired):
            s.run_stage("timeout", [sys.executable, "-c", script], self.root, self.output,
                        os.environ.copy(), time.monotonic() + .3)
        time.sleep(1)
        self.assertFalse(marker.exists(), "descendant survived process-group kill")
        self.assertIn("started", (self.output / "timeout.stdout.log").read_text())
        self.assertEqual(json.loads((self.output / "timeout.json").read_text())["status"], "timeout")


if __name__ == "__main__":
    unittest.main()
