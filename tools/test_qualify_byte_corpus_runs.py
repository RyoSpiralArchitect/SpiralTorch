import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "byte_run_qualifier", Path(__file__).with_name("qualify_byte_corpus_runs.py"))
qualifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualifier)


class MetadataTests(unittest.TestCase):
    def test_legacy_omission_is_not_reinterpreted(self):
        self.assertEqual(qualifier.case_metadata(dict(seed=11, geometry=True)),
                         dict(seed=11, geometry=True))

    def test_explicit_policy_preserved_and_weights_not_published(self):
        for policy in ("train", "frozen"):
            case = dict(seed=11, geometry=True, geometry_update=policy,
                        name="private-case-label", parameters=["private-weights"])
            self.assertEqual(qualifier.case_metadata(case),
                             dict(seed=11, geometry=True, geometry_update=policy))

    def test_failed_requirement_is_not_a_pass(self):
        with self.assertRaisesRegex(ValueError, "failed gate"):
            qualifier.require(False, "failed gate")


class TimeoutTests(unittest.TestCase):
    def test_bounded_integer_limit(self):
        for value in (1, "600", 1800, "7200"):
            self.assertEqual(qualifier.timeout_seconds(value), int(value))
        for value in (0, -1, 7201, "nan", "1.5", 1.0, True, None):
            with self.subTest(value=value), self.assertRaises(qualifier.argparse.ArgumentTypeError):
                qualifier.timeout_seconds(value)

    def test_default_and_explicit_worker_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(qualifier.subprocess, "run") as run:
                run.return_value.returncode = 0
                qualifier.run_child(root, "default", ["worker.py"])
                self.assertEqual(run.call_args.kwargs["timeout"], 600)
                qualifier.run_child(root, "long", ["worker.py"], timeout=1800)
                self.assertEqual(run.call_args.kwargs["timeout"], 1800)

    def test_real_timeout_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(subprocess.TimeoutExpired):
                qualifier.run_child(Path(directory), "timeout",
                                    ["-c", "import time; time.sleep(10)"], timeout=1)
            self.assertTrue((Path(directory) / "timeout.log").is_file())

    def test_cli_rejects_invalid_limit_before_importing_runtime(self):
        result = subprocess.run([sys.executable, str(Path(qualifier.__file__)), "run",
                                 "absent.json", "absent-output", "--stop", "512",
                                 "--wheel", "absent.whl", "--timeout-seconds", "7201"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("timeout must be an integer", result.stderr)


if __name__ == "__main__":
    unittest.main()
