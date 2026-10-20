"""Stdlib-only tests for bounded, non-publishing GPU diagnosis orchestration."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_byte_study_replay as replay


def fake_module(*, fail_first=False, fail_warm=False, skip=False, expected_failure=False):
    module = types.ModuleType("_fake_replay")
    module.events = []

    def first(case):
        module.events.append("first")
        if skip:
            case.skipTest("unexpected GPU skip")
        if fail_first or expected_failure:
            case.fail("injected first failure")

    def prepared(case):
        module.events.append("prepared")
        if fail_warm and module.events.count("prepared") == 2:
            case.fail("injected warm-process failure")

    if expected_failure:
        first = unittest.expectedFailure(first)
    module.StudyGpu = type("StudyGpu", (unittest.TestCase,), {
        "__module__": module.__name__, "test_00_first": first,
        replay.PREPARED_TEST: prepared,
    })
    return module


class ReplayTests(unittest.TestCase):
    def exercise(self, module):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            record = {}
            code = replay.exercise(module, root, record)
            logs = [path.read_text() for path in sorted(root.rglob("tests.log"))]
            return code, record, logs

    def test_original_order_then_exactly_one_warm_phase(self):
        module = fake_module()
        code, record, _ = self.exercise(module)
        self.assertEqual(code, 0)
        self.assertEqual(module.events, ["first", "prepared", "prepared"])
        self.assertEqual([row["tests_run"] for row in record["phases"]], [2, 1])
        self.assertEqual(record["outcome"], "no_mismatch_observed")
        self.assertTrue(all(row["valid"] for row in record["phases"]))

    def test_first_failure_is_not_retried_or_followed_by_warm_phase(self):
        module = fake_module(fail_first=True)
        code, record, logs = self.exercise(module)
        self.assertEqual(code, 1)
        self.assertEqual(module.events, ["first"])
        self.assertEqual(len(record["phases"]), 1)
        self.assertEqual(record["outcome"], "test_failure")
        self.assertIn("injected first failure", logs[0])

    def test_warm_failure_preserves_the_successful_first_phase(self):
        module = fake_module(fail_warm=True)
        code, record, logs = self.exercise(module)
        self.assertEqual(code, 1)
        self.assertEqual(module.events, ["first", "prepared", "prepared"])
        self.assertTrue(record["phases"][0]["valid"])
        self.assertFalse(record["phases"][1]["valid"])
        self.assertIn("injected warm-process failure", logs[1])

    def test_gpu_skip_is_not_a_success(self):
        code, record, _ = self.exercise(fake_module(skip=True))
        self.assertEqual(code, 1)
        self.assertEqual(len(record["phases"][0]["unexpected_skips"]), 1)

    def test_only_cpu_artifact_only_skip_is_permitted(self):
        module = fake_module()

        def cpu_only(case):
            case.skipTest("requires CPU-only artifact")

        module.StudyPortable = type("StudyPortable", (unittest.TestCase,), {
            "__module__": module.__name__,
            "test_cpu_only_execution_is_not_a_fallback": cpu_only,
        })
        code, record, _ = self.exercise(module)
        self.assertEqual(code, 0)
        self.assertEqual(len(record["phases"][0]["skipped"]), 1)
        self.assertEqual(record["phases"][0]["unexpected_skips"], [])

    def test_expected_failure_does_not_hide_a_red_oracle(self):
        code, record, _ = self.exercise(fake_module(expected_failure=True))
        self.assertEqual(code, 1)
        self.assertEqual(len(record["phases"][0]["expected_failures"]), 1)

    def test_zero_tests_cannot_be_valid(self):
        self.assertFalse(replay.phase_summary(unittest.TestResult())["valid"])

    def test_diagnostic_directories_are_distinct_and_environment_is_restored(self):
        seen = []
        module = fake_module()

        def prepared(case):
            seen.append({kind: os.environ["SPIRALTORCH_" + kind + "_STUDY_DIAGNOSTICS"]
                         for kind in replay.DIAGNOSTICS})

        setattr(module.StudyGpu, replay.PREPARED_TEST, prepared)
        keys = {"SPIRALTORCH_" + kind + "_STUDY_DIAGNOSTICS": "prior"
                for kind in replay.DIAGNOSTICS}
        with mock.patch.dict(os.environ, keys):
            self.assertEqual(self.exercise(module)[0], 0)
            self.assertEqual({key: os.environ[key] for key in keys}, keys)
        self.assertEqual(len(seen), 2)
        for kind in replay.DIAGNOSTICS:
            self.assertNotEqual(seen[0][kind], seen[1][kind])
            self.assertTrue(seen[0][kind].endswith("/" + kind.lower()))

    def test_import_audit_distinguishes_import_from_optional_discovery(self):
        attempts = []
        guard = replay.import_guard(attempts)
        guard("importlib.util.find_spec", ("torch",))
        guard("import", ("json",))
        with self.assertRaisesRegex(RuntimeError, "foreign ML"):
            guard("import", ("torch.nn",))
        self.assertEqual(attempts, ["torch.nn"])

    def test_blocked_setup_probe_is_not_confused_with_runtime_delegation(self):
        replay.require_native_runtime(["torch"], 1)
        with self.assertRaisesRegex(RuntimeError, "foreign ML"):
            replay.require_native_runtime(["torch", "torch.nn"], 1)
        with mock.patch.dict(sys.modules, {"torch": types.ModuleType("torch")}):
            with self.assertRaisesRegex(RuntimeError, "foreign ML"):
                replay.require_native_runtime([], 0)

    def test_installed_extension_must_match_the_selected_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "spiraltorch.abi3.so"
            library.write_bytes(b"native fixture")
            wheel = root / "fixture.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("spiraltorch/spiraltorch.abi3.so", library.read_bytes())
            native = types.SimpleNamespace(__file__=str(library))
            self.assertEqual(replay.native_identity(wheel, native)["sha256"], replay.digest(library))
            library.write_bytes(b"wrong installed extension")
            with self.assertRaisesRegex(ValueError, "does not match"):
                replay.native_identity(wheel, native)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(replay, "owned_run") as run:
            root = Path(directory)
            marker = root / "marker"
            marker.write_text("original evidence")
            with self.assertRaises(FileExistsError):
                replay.supervise(root, root, root / "absent.whl", 1)
            self.assertEqual(marker.read_text(), "original evidence")
            run.assert_not_called()

    def test_first_failure_is_saved_before_metadata_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {"outcome": "test_failure", "release_ready": False}

            def metadata(_):
                self.assertEqual(json.loads((root / "worker-result.json").read_text()), record)
                raise RuntimeError("injected metadata failure")

            with mock.patch("byte_study_diagnostics.runtime_identity", side_effect=metadata):
                replay.finish_worker(root, record, types.SimpleNamespace(st=object()))
            self.assertEqual(json.loads((root / "worker-result.json").read_text()), record)
            self.assertEqual(json.loads((root / "runtime.json").read_text()),
                             {"status": "collection_failed", "error_type": "RuntimeError"})

    def supervise(self, run_effect, *, source=None, worker_record=None, worker_code=0):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            wheel = root / "fixture.whl"
            wheel.write_bytes(b"fixture")
            output = root / "result"
            identity = {"commit": source or replay.SOURCE_COMMITS[0], "tree": "f" * 40}

            def execute(*args, **kwargs):
                if worker_record is not None:
                    replay.save(output / "worker-result.json", worker_record)
                if run_effect:
                    raise run_effect
                return subprocess.CompletedProcess(args[0], worker_code)

            with mock.patch.object(replay, "git_identity", return_value=identity), \
                    mock.patch.object(replay, "owned_run", side_effect=execute) as run:
                try:
                    code = replay.supervise(output, root, wheel, 2)
                except (FileNotFoundError, ValueError, RuntimeError) as error:
                    code = type(error).__name__
            return code, json.loads((output / "result.json").read_text()), run.call_count

    def test_timeout_retains_terminal_evidence_without_retry(self):
        code, record, calls = self.supervise(subprocess.TimeoutExpired("worker", 2))
        self.assertEqual((code, calls), (124, 1))
        self.assertEqual(record["outcome"], "timeout")
        self.assertFalse(record["release_ready"])
        self.assertIn("manifest.json", record["files"])
        self.assertIn("process.log", record["files"])

    def test_metadata_timeout_cannot_replace_the_first_test_failure(self):
        code, record, calls = self.supervise(
            subprocess.TimeoutExpired("metadata", 2), worker_record={"outcome": "test_failure"})
        self.assertEqual((code, calls), (124, 1))
        self.assertEqual(record["outcome"], "test_failure")
        self.assertEqual(record["worker_outcome"], "test_failure")
        self.assertEqual(record["supervision_outcome"], "timeout")
        self.assertIn("worker-result.json", record["files"])

    def test_successful_tests_with_stalled_metadata_are_not_job_success(self):
        code, record, _ = self.supervise(
            subprocess.TimeoutExpired("metadata", 2),
            worker_record={"outcome": "no_mismatch_observed"})
        self.assertEqual(code, 124)
        self.assertEqual(record["outcome"], "timeout")
        self.assertEqual(record["worker_outcome"], "no_mismatch_observed")

    def test_missing_worker_record_is_not_success(self):
        code, record, calls = self.supervise(None)
        self.assertEqual((code, calls), ("FileNotFoundError", 1))
        self.assertEqual(record["outcome"], "supervision_error")
        self.assertNotEqual(record["returncode"], 0)

    def test_worker_failure_is_not_retried(self):
        code, record, calls = self.supervise(
            None, worker_record={"outcome": "test_failure"}, worker_code=1)
        self.assertEqual((code, calls), (1, 1))
        self.assertEqual(record["outcome"], "test_failure")

    def test_mismatched_success_record_is_rejected(self):
        code, record, calls = self.supervise(None, worker_record={"outcome": "incomplete"})
        self.assertEqual((code, calls), ("RuntimeError", 1))
        self.assertEqual(record["outcome"], "supervision_error")

    def test_success_does_not_qualify_a_release(self):
        code, record, calls = self.supervise(None, worker_record={"outcome": "no_mismatch_observed"})
        self.assertEqual((code, calls), (0, 1))
        self.assertEqual(record["outcome"], "no_mismatch_observed")
        self.assertFalse(record["release_ready"])

    def test_undeclared_source_never_launches_worker(self):
        code, record, calls = self.supervise(None, source="a" * 40)
        self.assertEqual((code, calls), ("ValueError", 0))
        self.assertEqual(record["outcome"], "supervision_error")

    def test_workflow_is_manual_bounded_pinned_and_non_publishing(self):
        workflow = (replay.ROOT / ".github/workflows/byte_study_replay.yml").read_text()
        self.assertIn("on:\n  workflow_dispatch:", workflow)
        for trigger in ("push:", "pull_request:", "schedule:"):
            self.assertNotIn(trigger, workflow)
        self.assertIn("permissions:\n  contents: read", workflow)
        for mutation in ("contents: write", "id-token:", "secrets.", "twine", "publish_pypi",
                         "continue-on-error", "gh release", "git push"):
            self.assertNotIn(mutation, workflow)
        for source in replay.SOURCE_COMMITS:
            self.assertEqual(workflow.count(source), 3 if source == replay.SOURCE_COMMITS[0] else 2)
        self.assertIn("runs-on: macos-26", workflow)
        self.assertIn('toolchain: "1.99.1"', workflow)
        self.assertIn('RUSTUP_TOOLCHAIN: "1.99.1"', workflow)
        self.assertIn('test "$(rustc --version | cut -d \' \' -f 2)" = "1.99.1"', workflow)
        self.assertIn("rustc --version --verbose", workflow)
        self.assertIn("cargo --version --verbose", workflow)
        self.assertIn('python-version: "3.12.10"', workflow)
        self.assertIn('"maturin==1.15.0"', workflow)
        self.assertIn("--no-deps", workflow)
        self.assertIn("timeout-minutes: 180", workflow)
        self.assertIn("--timeout-seconds 7200", workflow)
        self.assertIn("cancel-in-progress: false", workflow)
        self.assertIn("if: always()", workflow)
        self.assertIn("retention-days: 1", workflow)
        self.assertIn("github.run_attempt", workflow)
        self.assertEqual(workflow.count("persist-credentials: false"), 2)
        artifact = workflow.split("uses: actions/upload-artifact@v7", 1)[1]
        self.assertIn("path: ${{ runner.temp }}/byte-study-replay/", artifact)
        self.assertNotIn("replay-wheel", artifact)


if __name__ == "__main__":
    unittest.main()
