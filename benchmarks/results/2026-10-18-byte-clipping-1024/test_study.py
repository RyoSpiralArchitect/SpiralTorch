import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("long_clip_study", Path(__file__).with_name("study.py"))
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class StudyTests(unittest.TestCase):
    def test_wrap_changes_only_schema_and_clip(self):
        original = dict(schema="spiraltorch.byte_corpus.request.v9",
                        optimizer={"kind": "adamw_zero_rate_freeze.v1"},
                        cases=[{"values": [-0.0, 0.125]}])
        for policy in study.POLICIES.values():
            actual = study.derived(original, policy)
            self.assertEqual(actual.pop("gradient_clip"), policy)
            actual["schema"] = original["schema"]
            self.assertEqual(study.encoded(actual), study.encoded(original))
            actual["cases"][0]["values"].append(99)
            self.assertEqual(len(original["cases"][0]["values"]), 2)

    def test_other_versions_rejected(self):
        for version in (8, 10):
            with self.assertRaises(ValueError):
                study.derived({"schema": f"spiraltorch.byte_corpus.request.v{version}"},
                              study.POLICIES["disabled"])

    def test_transport_preserves_signed_zero_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            study.write(path, {"value": -0.0})
            self.assertIn(b"-0.0", study.encoded(study.read(path)))
            with self.assertRaises(FileExistsError):
                study.write(path, {})

    def test_export_rejects_raw_weights_even_when_nested(self):
        with self.assertRaises(ValueError):
            study.screen({"rows": [{"final_parameters": [[1.0]]}]})
        study.screen({"parameters": [{"name": "qkv.bias", "max_abs_error": 1e-6}]})

    def test_process_failure_and_timeout_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failed = study.execute(root, "failed", [sys.executable, "-I", "-c", "raise SystemExit(3)"], 10)
            self.assertEqual(failed["exit_code"], 3)
            self.assertFalse(failed["timed_out"])
            timeout = study.execute(root, "timeout", [sys.executable, "-I", "-c",
                                    "import time; time.sleep(10)"], 1)
            self.assertTrue(timeout["timed_out"])
            self.assertNotEqual(timeout["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
