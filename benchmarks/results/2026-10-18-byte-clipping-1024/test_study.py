import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

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
            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_bytes(b'{"value":1,"value":2}')
            with self.assertRaises(ValueError):
                study.read(duplicate)

    def test_export_rejects_raw_weights_even_when_nested(self):
        with self.assertRaises(ValueError):
            study.screen({"rows": [{"final_parameters": [[1.0]]}]})
        study.screen({"parameters": [{"name": "qkv.bias", "max_abs_error": 1e-6}]})

    @unittest.skipUnless(os.name == "posix", "POSIX process ownership")
    def test_process_failure_and_timeout_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failed = study.execute(root, "failed", [sys.executable, "-I", "-B", "-c", "raise SystemExit(3)"], 10)
            self.assertEqual(failed["exit_code"], 3)
            self.assertFalse(failed["timed_out"])
            timeout = study.execute(root, "timeout", [sys.executable, "-I", "-B", "-c",
                                    "import time; time.sleep(10)"], 1)
            self.assertTrue(timeout["timed_out"])
            self.assertNotEqual(timeout["exit_code"], 0)

    def test_partial_report_records_failure_then_next_condition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("disabled", "active"):
                study.write(root / (name + ".json"), {})
                study.write(root / (name + "-torch.json"), {})
                (root / name / "continuous").mkdir(parents=True)
            (root / "disabled/continuous/report-1024.json").write_bytes(b'{"cases":')
            failed = study.collect_condition(root, "disabled", {"exit_code": 0}, {"exit_code": 1})
            self.assertIn("comparison_error", failed)
            self.assertNotIn("comparison", failed)
            self.assertNotIn("recovery", failed)
            self.assertTrue((root / "disabled-execution.json").is_file())
            self.assertEqual(study.read(root / "disabled-result.json"), failed)
            study.screen(failed)
            with patch.object(study, "summarize", return_value={"strict_gate": {"passed": False}}):
                next_row = study.collect_condition(root, "active", {"exit_code": 0}, {"exit_code": 1})
            self.assertFalse(next_row["comparison"]["strict_gate"]["passed"])

    def test_zero_exit_with_invalid_recovery_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            row = study.collect_condition(Path(directory), "disabled", {"exit_code": 1}, {"exit_code": 0})
            self.assertIn("recovery_error", row)
            self.assertNotIn("recovery", row)
            self.assertEqual(row["comparison_not_run"], "reference_execution_failed")

    def test_historical_revalidation_binds_git_and_current_checker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in study.POLICIES:
                study.write(root / (name + ".json"), {})
            checker = root / "checker.py"
            checker.write_bytes(b"original")
            study.write(root / "started.json", dict(sources={"checker.py": study.sha(b"original")},
                requests={name: study.sha((root / (name + ".json")).read_bytes()) for name in study.POLICIES}))
            # Use the actual repo root for the harness path calculation, while
            # checking a mocked historical blob and current pinned file.
            path = str(checker.relative_to(root))
            original_read = Path.read_bytes

            def contents(p):
                return checker.read_bytes() if p == study.ROOT / path else original_read(p)

            def git(args, **_kwargs):
                return ("a" * 40).encode() if args[1] == "rev-parse" else b"original"

            with patch.object(study.subprocess, "check_output", side_effect=git), \
                    patch.object(study, "HISTORICAL_REVISION", "a" * 40), \
                    patch.object(study, "HISTORICAL_MANIFEST", study.sha((root / "started.json").read_bytes())), \
                    patch.object(Path, "read_bytes", contents):
                study.verify(root, "a" * 40)
                checker.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "checker or model drift"):
                    study.verify(root, "a" * 40)
            with self.assertRaisesRegex(ValueError, "exact commit"):
                study.verify(root, "HEAD")

    def test_missing_historical_inventory_rejects_before_git_or_source_exceptions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source_map in ({}, {"tools/qualify_byte_corpus_runs.py": "0" * 64}):
                (root / "started.json").write_bytes(study.encoded(dict(sources=source_map, requests={})))
                with patch.object(study.subprocess, "check_output") as git:
                    with self.assertRaisesRegex(ValueError, "pinned execution"):
                        study.verify(root, study.HISTORICAL_REVISION)
                    git.assert_not_called()

    def test_historical_commit_existence_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study.write(root / "started.json", dict(sources={}, requests={}))
            with patch.object(study, "HISTORICAL_MANIFEST", study.sha((root / "started.json").read_bytes())), \
                    patch.object(study.subprocess, "check_output", return_value=b"not-a-commit"):
                with self.assertRaisesRegex(ValueError, "does not exist"):
                    study.verify(root, study.HISTORICAL_REVISION)

    def test_historical_inventory_cannot_change_between_decode_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests = {}
            for name in study.POLICIES:
                study.write(root / (name + ".json"), {})
                requests[name] = study.sha((root / (name + ".json")).read_bytes())
            bad = study.encoded(dict(sources={}, requests=requests))
            good = study.encoded(dict(sources={"checker.py": study.sha(b"original")}, requests=requests))
            original_read = Path.read_bytes
            reads = 0

            def contents(p):
                nonlocal reads
                if p == root / "started.json":
                    reads += 1
                    return bad if reads == 1 else good
                return original_read(p)

            with patch.object(study, "HISTORICAL_MANIFEST", study.sha(good)), \
                    patch.object(Path, "read_bytes", contents), \
                    patch.object(study.subprocess, "check_output") as git:
                with self.assertRaisesRegex(ValueError, "pinned execution"):
                    study.verify(root, study.HISTORICAL_REVISION)
                git.assert_not_called()
                self.assertEqual(reads, 1)


if __name__ == "__main__":
    unittest.main()
