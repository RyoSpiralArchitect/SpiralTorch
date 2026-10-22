#!/usr/bin/env python3
"""Stubbed orchestration only: no native import, GPU execution or numeric claim."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "retained", Path(__file__).with_name("run_byte_study_retained.py"))
API = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(API)


def encoded(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def selection():
    names = [f"case-{i}" for i in range(25)]
    names[23] = API.CASE
    return {"request_sha256": API.REQUEST_SHA256, "case": API.CASE, "case_index": 23,
            "version": 9, "full": {"cases": [{"name": name} for name in names]},
            "payloads": {"request.json": b"synthetic envelope; not a Rust model"}}


def report(updates=2):
    kind = "partial" if updates == 1 else "result"
    return encoded({"schema": f"spiraltorch.byte_corpus.{kind}.v9",
                    "request_sha256": API.REQUEST_SHA256,
                    "cases": [{**case, "training": [
                        {"revision": i + 1, "ce": 5.0 - i} for i in range(updates)]}
                        for case in selection()["full"]["cases"]]})


def checkpoint(revision=1):
    return encoded({"schema": "spiraltorch.nn.byte_decoder_checkpoint.v6",
                    "attempted_revision": str(revision), "model": {"weight": [1.0, -0.0]},
                    "optimizer": {"m": [0.1], "v": [0.01], "steps": [revision]}})


def runtime_metadata():
    return {"schema": "spiraltorch.synthetic_gpu_runtime.v1",
            "native_library": {"sha256": "mock"},
            "adapter": {"name": "mock GPU", "backend": "Metal", "device_type": "VirtualGpu"}}


class Checkpoint:
    def __init__(self, raw):
        self.raw = raw

    def to_json(self):
        return self.raw


class Observation:
    def __init__(self, study):
        self.study = study
        self.full_report = report()
        self.boundary_raw = checkpoint()
        self.read_error = self.export_error = None
        self.outcome_override = None

    @property
    def run_outcome_json(self):
        if self.outcome_override is not None:
            return encoded(self.outcome_override)
        return encoded({"schema": "spiraltorch.byte_retained_boundary.v1",
                        "request_sha256": API.REQUEST_SHA256, "case": API.CASE,
                        "requested_capture_after_update": 1, "capture_mode": "retained",
                        "boundary_read_policy": "explicit_after_run_returns",
                        "boundary_captured": True, "release_ready": False,
                        "status": "completed", "all_cases_completed": True,
                        "report": json.loads(self.full_report)})

    @property
    def report_json(self):
        if self.export_error:
            raise self.export_error
        return self.full_report

    def read_boundary(self):
        self.study.events.append("read")
        self.study.test.assertEqual((self.study.output / "run-outcome.json").read_bytes(),
                                    self.run_outcome_json.encode())
        if self.full_report is not None and not self.export_error:
            self.study.test.assertEqual((self.study.output / "full-report.json").read_bytes(),
                                        self.full_report.encode())
        if self.read_error:
            raise self.read_error
        return Checkpoint(self.boundary_raw)


class Study:
    total_updates = 2
    request_sha256 = API.REQUEST_SHA256

    def __init__(self, test, output):
        self.test, self.output = test, output
        self.events = []
        self.loaded = None
        self.observed = Observation(self)
        self.fail_phase = None

    def validate_segment(self, stop=None):
        self.events.append(("validate", stop))

    def advance_wgpu(self, stop=None, *, resume=None):
        name = "partial" if stop == 1 else "resume"
        self.events.append(name)
        if self.fail_phase == name:
            raise RuntimeError("synthetic " + name + " failure")
        if name == "resume":
            self.test.assertIs(resume, self.loaded)
        updates = 1 if stop == 1 else 2
        return types.SimpleNamespace(report_json=report(updates),
                                     checkpoint=Checkpoint(checkpoint(updates)))

    def checkpoint_from_json(self, raw):
        self.events.append("load")
        self.test.assertEqual(raw, checkpoint().encode())
        self.loaded = Checkpoint(raw.decode())
        return self.loaded

    def run_wgpu_with_retained_boundary(self, saved, name):
        self.events.append("full")
        self.test.assertIs(saved, self.loaded)
        self.test.assertEqual(name, API.CASE)
        if self.fail_phase == "full":
            raise RuntimeError("synthetic full call failure")
        return self.observed

    def case_checkpoint(self, saved, name):
        self.events.append("reference")
        self.test.assertIs(saved, self.loaded)
        self.test.assertEqual(name, API.CASE)
        return saved


class Exercise(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.output = Path(temp.name)
        self.study = Study(self, self.output)
        self.record = {}

    def run_attempt(self):
        return API.exercise(self.study, selection(), self.output, self.record)

    def test_exact_order_same_attempt_reference_and_single_read(self):
        self.assertEqual(self.run_attempt(), 0)
        self.assertEqual(self.study.events, [("validate", 1), ("validate", None),
                                           "partial", "load", "resume", "full", "read", "reference"])
        self.assertEqual(self.record["native_calls"], ["partial_1", "resume_2", "retained_full_2"])
        self.assertTrue(self.record["report_comparison"]["exact_bytes_equal"])
        self.assertTrue(self.record["boundary_comparison"]["exact_bytes_equal"])

    def test_report_bytes_not_reconstructed_and_mismatch_does_not_skip_read(self):
        original = report().replace("5.0", "5.00000000")
        self.study.observed.full_report = original
        self.assertEqual(self.run_attempt(), 1)
        self.assertEqual((self.output / "full-report.json").read_bytes(), original.encode())
        self.assertEqual(self.record["outcome"], "mismatch_observed")
        self.assertEqual(self.study.events.count("read"), 1)
        self.assertEqual(self.record["report_comparison"]["differences"][0]["kind"], "serialization_only")

    def test_each_boundary_component_difference_is_reported_without_tolerance(self):
        for field, value in (("model", {"weight": [1.0, 0.0]}),
                             ("optimizer", {"m": [0.10000001], "v": [0.01], "steps": [1]})):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                study = Study(self, output)
                changed = json.loads(checkpoint())
                changed[field] = value
                study.observed.boundary_raw = encoded(changed)
                record = {}
                self.assertEqual(API.exercise(study, selection(), output, record), 1)
                comparison = record["boundary_comparison"]
                self.assertFalse(comparison["exact_bytes_equal"])
                self.assertFalse(comparison["decoded_components_equal"][field])
                self.assertEqual(study.events.count("read"), 1)

    def test_numeric_mismatch_survives_later_readback_failure(self):
        self.study.observed.full_report = report().replace('"ce":4.0', '"ce":4.5')
        self.study.observed.read_error = RuntimeError("synthetic map failure")
        self.assertEqual(self.run_attempt(), 1)
        self.assertEqual(self.record["outcome"], "mismatch_observed")
        self.assertIn("boundary_error", self.record)
        self.assertIn("report_comparison", self.record)
        self.assertTrue((self.output / "run-outcome.json").is_file())

    def test_other_case_mismatch_is_not_attributed_to_the_selected_boundary(self):
        value = json.loads(report())
        value["cases"][0]["training"][1]["ce"] = 99.0
        self.study.observed.full_report = encoded(value)
        self.assertEqual(self.run_attempt(), 1)
        self.assertEqual(self.record["outcome"], "mismatch_observed")
        self.assertTrue(self.record["boundary_comparison"]["exact_bytes_equal"])
        selected = self.record["selected_case_training"]
        self.assertEqual(selected["full"], selected["resumed"])
        self.assertEqual(selected["case_index"], 23)

    def test_readback_failure_without_numeric_mismatch_is_not_success(self):
        self.study.observed.read_error = RuntimeError("synthetic map failure")
        self.assertEqual(self.run_attempt(), 1)
        self.assertEqual(self.record["outcome"], "boundary_read_error")

    def test_failed_study_with_readable_old_state_is_not_update_acceptance(self):
        outcome = json.loads(self.study.observed.run_outcome_json)
        outcome.update(status="error", all_cases_completed=False, error="synthetic rejected update")
        del outcome["report"]
        self.study.observed.outcome_override = outcome
        self.study.observed.full_report = None
        self.assertEqual(self.run_attempt(), 1)
        self.assertEqual(self.record["outcome"], "study_error")
        self.assertTrue(self.record["boundary_comparison"]["exact_bytes_equal"])
        self.assertFalse((self.output / "full-report.json").exists())

    def test_failed_study_without_boundary_preserves_both_errors(self):
        outcome = json.loads(self.study.observed.run_outcome_json)
        outcome.update(status="error", all_cases_completed=False, boundary_captured=False,
                       error="synthetic early error")
        del outcome["report"]
        self.study.observed.outcome_override = outcome
        self.study.observed.full_report = None
        self.study.observed.read_error = RuntimeError("boundary unavailable")
        self.assertEqual(self.run_attempt(), 1)
        self.assertEqual(self.record["outcome"], "study_error")
        self.assertIn("boundary_error", self.record)

    def test_identity_or_export_error_keeps_raw_outcome_and_still_reads_once(self):
        for change in ("identity", "export"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                study = Study(self, output)
                if change == "identity":
                    outcome = json.loads(study.observed.run_outcome_json)
                    outcome["request_sha256"] = "foreign"
                    study.observed.outcome_override = outcome
                else:
                    study.observed.export_error = RuntimeError("synthetic export error")
                with self.assertRaises(ValueError):
                    API.exercise(study, selection(), output, {})
                self.assertTrue((output / "run-outcome.json").exists())
                self.assertEqual(study.events.count("read"), 1)

    def test_missing_case_or_revision_is_rejected_after_preservation(self):
        payload = json.loads(report())
        payload["cases"].pop()
        self.study.observed.full_report = encoded(payload)
        with self.assertRaisesRegex(ValueError, "case order"):
            self.run_attempt()
        self.assertTrue((self.output / "full-report.json").is_file())
        self.assertEqual(self.study.events.count("read"), 1)

    def test_partial_schema_and_training_cursor_are_not_completed_report_schema(self):
        self.assertEqual(API.validate_report(report(1), selection(), 1)["schema"],
                         "spiraltorch.byte_corpus.partial.v9")
        wrong = json.loads(report(1))
        wrong["schema"] = "spiraltorch.byte_corpus.result.v9"
        with self.assertRaisesRegex(ValueError, "identity"):
            API.validate_report(encoded(wrong), selection(), 1)
        for revision in (True, 0, 2):
            wrong = json.loads(report(1))
            wrong["cases"][0]["training"][0]["revision"] = revision
            with self.subTest(revision=revision), self.assertRaisesRegex(ValueError, "cursor"):
                API.validate_report(encoded(wrong), selection(), 1)
        for updates in (True, 0, 3):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                API.validate_report(report(1), selection(), updates)

    def test_partial_resume_or_full_error_does_not_retry_or_continue(self):
        for phase in ("partial", "resume", "full"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                study = Study(self, Path(tmp))
                study.fail_phase = phase
                with self.assertRaisesRegex(RuntimeError, phase):
                    API.exercise(study, selection(), Path(tmp), {})
                self.assertEqual(study.events.count(phase), 1)
                self.assertEqual(study.events[-1], phase)

    def test_nonfinite_duplicate_and_wrong_boundary_cursor_fail_closed(self):
        bad = ('{"a":1,"a":2}', '{"a":1e999}', checkpoint(2),
               '{"attempted_revision":"1"}')
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                API.compare_boundaries(checkpoint().encode(), raw.encode())


class Supervision(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output = self.root / "out"
        self.wheel = self.root / "fake.whl"
        self.wheel.write_bytes(b"stubbed supervisor test, not a wheel")
        identity = {"commit": API.SOURCE_COMMIT, "tree": "synthetic",
                    "tracked_changes": 0, "untracked_files": 0}
        patch = mock.patch.object(API.replay, "git_identity", return_value=identity)
        self.identity = patch.start()
        self.addCleanup(patch.stop)
        self.calls = []

    def completion(self, code=0):
        return {"schema": "spiraltorch.retained_study.worker.v1", "protocol": API.PROTOCOL,
                "release_ready": False, "returncode": code, "stage": "finished",
                "study_status": "completed", "outcome": "no_mismatch_observed",
                "native_execution_completed": True,
                "native_library": {"sha256": "mock"},
                "native_calls": ["partial_1", "resume_2", "retained_full_2"],
                "report_comparison": {"exact_bytes_equal": True,
                    "full_sha256": API.boundary.digest(report().encode()),
                    "resumed_sha256": API.boundary.digest(report().encode())},
                "boundary_comparison": {"exact_bytes_equal": True,
                    "reference_sha256": API.boundary.digest(checkpoint().encode()),
                    "observed_sha256": API.boundary.digest(checkpoint().encode())},
                "unexpected_child_spawns": 0}

    def runner(self, command, *, timeout, stdout, cwd):
        self.calls.append((command, timeout))
        self.assertEqual(timeout, 17)
        self.assertIn("--worker", command)
        self.assertIn("-I", command)
        self.assertIn("-B", command)
        self.assertNotIn("--execute-once", command)
        for name, raw in (("full-report.json", report()), ("resumed-report.json", report()),
                          ("reference-boundary.json", checkpoint()),
                          ("observed-boundary.json", checkpoint())):
            API.save_raw(self.output / name, raw)
        API.boundary.save(self.output / "worker-result.json", self.completion())
        API.boundary.save(self.output / "runtime.json", runtime_metadata())
        return subprocess.CompletedProcess(command, 0)

    def supervise(self, runner=None):
        with mock.patch.object(API, "owned_run", side_effect=runner or self.runner):
            return API.supervise(selection(), self.output, self.root, self.wheel, 17)

    def test_one_process_no_retries_and_existing_output_never_reused(self):
        self.assertEqual(self.supervise(), 0)
        self.assertEqual(len(self.calls), 1)
        result = json.loads((self.output / "result.json").read_bytes())
        self.assertFalse(result["release_ready"])
        self.assertEqual(result["attempt_limit"], 1)
        with self.assertRaises(FileExistsError):
            self.supervise()
        self.assertEqual(len(self.calls), 1)

    def test_zero_exit_with_incomplete_or_contradictory_record_fails_closed(self):
        changes = [lambda r: r.pop("boundary_comparison"),
                   lambda r: r.pop("native_execution_completed"),
                   lambda r: r.update(native_execution_completed=False),
                   lambda r: r.update(release_ready=True),
                   lambda r: r.update(execution_error={"type": "RuntimeError"}),
                   lambda r: r["native_calls"].append("retained_full_2")]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                self.output = self.root / f"out-{index}"
                def run(command, **kwargs):
                    row = self.completion()
                    change(row)
                    API.boundary.save(self.output / "worker-result.json", row)
                    return subprocess.CompletedProcess(command, 0)
                self.assertEqual(self.supervise(run), 1)
                result = json.loads((self.output / "result.json").read_bytes())
                self.assertEqual(result["outcome"], "invalid_worker_record")

    def test_timeout_preserves_prior_result_and_partial_bytes_without_retry(self):
        def run(command, **kwargs):
            self.calls.append(command)
            API.save_raw(self.output / "run-outcome.json", b'{"original":"retained"}')
            row = self.completion(1)
            row["outcome"] = "mismatch_observed"
            API.boundary.save(self.output / "worker-result.json", row)
            raise subprocess.TimeoutExpired(command, 17)
        self.assertEqual(self.supervise(run), 124)
        result = json.loads((self.output / "result.json").read_bytes())
        self.assertEqual(result["outcome"], "timeout")
        self.assertEqual(result["worker_outcome"], "mismatch_observed")
        self.assertIn("run-outcome.json", result["files"])
        self.assertEqual(len(self.calls), 1)

    def test_successful_worker_exit_cannot_hide_missing_or_altered_evidence(self):
        for mode in ("missing", "changed"):
            with self.subTest(mode=mode):
                self.output = self.root / mode
                def run(command, **kwargs):
                    completed = self.runner(command, **kwargs)
                    path = self.output / "observed-boundary.json"
                    if mode == "missing":
                        path.unlink()
                    else:
                        path.write_bytes(checkpoint(2).encode())
                    return completed
                self.assertEqual(self.supervise(run), 1)
                result = json.loads((self.output / "result.json").read_bytes())
                self.assertEqual(result["supervision_outcome"], "error")
                self.assertEqual(result["worker_outcome"], "no_mismatch_observed")

    def test_missing_or_failed_runtime_provenance_never_qualifies_success(self):
        for mode in ("failed", "missing", "native_mismatch", "cpu", "incomplete"):
            with self.subTest(mode=mode):
                self.output = self.root / mode
                def run(command, **kwargs):
                    self.runner(command, **kwargs)
                    path = self.output / "runtime.json"
                    value = runtime_metadata()
                    if mode == "missing":
                        path.unlink()
                    else:
                        if mode == "failed":
                            value = {"status": "collection_failed", "error_type": "RuntimeError"}
                        elif mode == "native_mismatch":
                            value["native_library"]["sha256"] = "different"
                        elif mode == "cpu":
                            value["adapter"]["device_type"] = "Cpu"
                        elif mode == "incomplete":
                            value = {}
                        path.write_text(encoded(value))
                    return subprocess.CompletedProcess(command, 1)
                self.assertEqual(self.supervise(run), 1)
                result = json.loads((self.output / "result.json").read_bytes())
                self.assertEqual(result["outcome"], "metadata_error")
                self.assertEqual(result["metadata_status"], "failed")
                self.assertEqual(result["worker_outcome"], "no_mismatch_observed")

    def test_zero_exit_cannot_hide_metadata_failure(self):
        def run(command, **kwargs):
            completed = self.runner(command, **kwargs)
            (self.output / "runtime.json").write_text('{"status":"collection_failed"}')
            return completed
        self.assertEqual(self.supervise(run), 1)
        result = json.loads((self.output / "result.json").read_bytes())
        self.assertEqual(result["metadata_status"], "failed")
        self.assertEqual(result["outcome"], "invalid_worker_record")

    def test_metadata_failure_does_not_replace_a_preceding_mismatch(self):
        def run(command, **kwargs):
            self.runner(command, **kwargs)
            value = self.completion(1)
            value["outcome"] = "mismatch_observed"
            value["report_comparison"]["exact_bytes_equal"] = False
            (self.output / "worker-result.json").write_text(encoded(value))
            (self.output / "runtime.json").write_text('{"status":"collection_failed"}')
            return subprocess.CompletedProcess(command, 1)
        self.assertEqual(self.supervise(run), 1)
        result = json.loads((self.output / "result.json").read_bytes())
        self.assertEqual(result["metadata_status"], "failed")
        self.assertEqual(result["outcome"], "mismatch_observed")
        self.assertEqual(result["worker_outcome"], "mismatch_observed")

    def test_source_or_deadline_rejection_prevents_spawn(self):
        for field, value in (("commit", "wrong"), ("tracked_changes", 1), ("untracked_files", 1)):
            with self.subTest(field=field):
                saved = copy.deepcopy(self.identity.return_value)
                self.identity.return_value[field] = value
                with mock.patch.object(API, "owned_run") as run, self.assertRaises(ValueError):
                    API.supervise(selection(), self.output, self.root, self.wheel, 17)
                run.assert_not_called()
                self.identity.return_value = saved
        for timeout in (True, 0, 7201):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                API.supervise(selection(), self.output, self.root, self.wheel, timeout)

    def test_worker_claim_is_exclusive_before_native_import(self):
        self.output.mkdir()
        manifest = {"schema": "spiraltorch.retained_study.manifest.v1", "protocol": API.PROTOCOL,
                    "harness_sha256": API.replay.digest(API.__file__), "request_sha256": API.REQUEST_SHA256,
                    "harness_files": API.harness_files(),
                    "case": API.CASE, "case_count": 25, "release_ready": False,
                    "source_checkout": {"commit": API.SOURCE_COMMIT}}
        API.boundary.save(self.output / "manifest.json", manifest)
        self.assertEqual(API.claim_worker(self.output), manifest)
        with mock.patch.object(API.importlib, "import_module") as imported, self.assertRaises(FileExistsError):
            API.worker(self.output, self.wheel)
        imported.assert_not_called()

    def test_changed_helper_is_rejected_before_claim_or_native_import(self):
        self.output.mkdir()
        manifest = {"schema": "spiraltorch.retained_study.manifest.v1", "protocol": API.PROTOCOL,
                    "harness_sha256": API.replay.digest(API.__file__), "harness_files": {},
                    "request_sha256": API.REQUEST_SHA256, "case": API.CASE, "case_count": 25,
                    "release_ready": False, "source_checkout": {"commit": API.SOURCE_COMMIT}}
        API.boundary.save(self.output / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "manifest identity"):
            API.claim_worker(self.output)
        self.assertFalse((self.output / "worker-start.json").exists())

    def test_default_cli_only_reads_fixed_evidence_without_native_or_spawn(self):
        with mock.patch.object(API, "select_fixed", return_value=selection()), \
                mock.patch.object(API, "supervise") as run, \
                mock.patch.object(API.importlib, "import_module") as imported, \
                contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(API.main(["--failure-root", str(self.root)]), 0)
        run.assert_not_called()
        imported.assert_not_called()
        value = json.loads(stream.getvalue())
        self.assertFalse(value["gpu_executed"])
        self.assertFalse(value["native_runtime_validated"])

    def test_fixed_selection_rejects_changed_input_case_or_case_count(self):
        changes = [lambda r: r.update(version=6), lambda r: r.update(request_sha256="other"),
                   lambda r: r.update(case="other"), lambda r: r.update(case_index=0),
                   lambda r: r["full"]["cases"].pop()]
        for change in changes:
            selected = selection()
            change(selected)
            with mock.patch.object(API.boundary, "select", return_value=(selected, None)), \
                    self.assertRaises(ValueError):
                API.select_fixed(self.root)


class Worker(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.output = Path(temp.name)
        self.wheel = self.output / "synthetic.whl"
        self.wheel.write_bytes(b"offline mocked wheel")
        manifest = {"schema": "spiraltorch.retained_study.manifest.v1", "protocol": API.PROTOCOL,
                    "harness_sha256": API.replay.digest(API.__file__), "harness_files": API.harness_files(),
                    "request_sha256": API.REQUEST_SHA256, "case": API.CASE, "case_count": 25,
                    "release_ready": False, "source_checkout": {"commit": API.SOURCE_COMMIT},
                    "wheel": {"sha256": API.replay.digest(self.wheel)}}
        API.boundary.save(self.output / "manifest.json", manifest)
        self.study = Study(self, self.output)
        self.native = types.SimpleNamespace()
        self.runtime_collections = 0
        self.factory = mock.Mock(return_value=self.study)
        self.st = types.SimpleNamespace(nn=types.SimpleNamespace(
            ByteCorpusStudy=types.SimpleNamespace(from_json=self.factory,
                run_wgpu_with_retained_boundary=Study.run_wgpu_with_retained_boundary),
            ByteCorpusBoundaryObservation=Observation))

    def run_worker(self, *, late_audit=False, postlude_error=False, collection_error=False):
        def imported(name):
            return {"spiraltorch": self.st, "spiraltorch.spiraltorch": self.native}[name]

        def runtime(_st):
            self.runtime_collections += 1
            self.assertTrue((self.output / "worker-result.json").is_file())
            if collection_error:
                raise RuntimeError("private synthetic collection failure")
            return runtime_metadata()

        payload = API._runtime_payload
        with mock.patch.object(API, "select_fixed", return_value=selection()), \
                mock.patch.object(API.replay, "require_foreign_free_environment"), \
                mock.patch.object(API.replay, "install_import_guards"), \
                mock.patch.object(API.sys, "addaudithook"), \
                mock.patch.object(API.importlib, "import_module", side_effect=imported), \
                mock.patch.object(API.replay, "native_identity", return_value={"sha256": "mock"}), \
                mock.patch.object(API.replay, "require_native_runtime", side_effect=(
                    [None, RuntimeError("synthetic late import violation")] if late_audit else None)), \
                mock.patch.object(API, "runtime_identity", side_effect=runtime), \
                mock.patch.object(API, "_runtime_payload", side_effect=(
                    RuntimeError("synthetic postlude error") if postlude_error else payload)), \
                mock.patch.dict(API.os.environ):
            return API.worker(self.output, self.wheel)

    def test_worker_uses_full_request_once_and_records_before_postlude(self):
        self.assertEqual(self.run_worker(), 0)
        self.factory.assert_called_once_with(selection()["payloads"]["request.json"])
        value = json.loads((self.output / "worker-result.json").read_bytes())
        self.assertEqual(value["outcome"], "no_mismatch_observed")
        self.assertFalse(value["release_ready"])
        self.assertTrue((self.output / "runtime.json").is_file())
        self.assertEqual(self.runtime_collections, 1)

    def test_old_wheel_surface_rejects_before_the_first_gpu_call(self):
        self.st.nn.ByteCorpusBoundaryObservation = None
        self.assertEqual(self.run_worker(), 1)
        self.factory.assert_not_called()
        self.assertEqual(self.study.events, [])
        value = json.loads((self.output / "worker-result.json").read_bytes())
        self.assertEqual(value["outcome"], "execution_error")
        self.assertEqual(self.runtime_collections, 0)
        self.assertFalse(value["native_execution_completed"])

    def test_cpu_validation_failure_skips_native_execution_and_metadata(self):
        self.study.validate_segment = mock.Mock(side_effect=ValueError("synthetic validation"))
        self.assertEqual(self.run_worker(), 1)
        self.assertEqual(self.study.events, [])
        self.assertEqual(self.runtime_collections, 0)

    def test_first_persistence_failure_still_records_completed_native_phase(self):
        with mock.patch.object(API, "save_raw", side_effect=OSError("synthetic write failure")):
            with self.assertRaisesRegex(OSError, "write failure"):
                self.run_worker()
        value = json.loads((self.output / "worker-result.json").read_bytes())
        self.assertTrue(value["native_execution_completed"])
        self.assertEqual(value["outcome"], "execution_error")
        self.assertEqual(value["stage"], "partial")
        self.assertEqual(self.runtime_collections, 1)

    def test_first_native_failure_does_not_initialize_runtime_for_metadata(self):
        self.study.fail_phase = "partial"
        self.assertEqual(self.run_worker(), 1)
        self.assertEqual(self.runtime_collections, 0)
        self.assertEqual(self.study.events[-1], "partial")
        value = json.loads((self.output / "runtime.json").read_bytes())
        self.assertEqual(value["status"], "not_collected")
        self.assertEqual(value["reason"], "no_completed_native_phase")

    def test_later_native_failure_collects_metadata_only_after_partial_completes(self):
        self.study.fail_phase = "resume"
        self.assertEqual(self.run_worker(), 1)
        self.assertEqual(self.runtime_collections, 1)
        value = json.loads((self.output / "worker-result.json").read_bytes())
        self.assertTrue(value["native_execution_completed"])
        self.assertEqual(value["stage"], "resume")

    def test_real_metadata_wrapper_failure_preserves_comparison_and_exits_nonzero(self):
        self.assertEqual(self.run_worker(collection_error=True), 1)
        value = json.loads((self.output / "worker-result.json").read_bytes())
        self.assertEqual(value["outcome"], "no_mismatch_observed")
        self.assertEqual(value["returncode"], 0)
        raw = (self.output / "runtime.json").read_text()
        self.assertEqual(json.loads(raw)["status"], "collection_failed")
        self.assertNotIn("private synthetic", raw)
        self.assertEqual(self.runtime_collections, 1)

    def test_real_metadata_failure_cannot_erase_a_numeric_mismatch(self):
        self.study.observed.full_report = report().replace('"ce":4.0', '"ce":4.5')
        self.assertEqual(self.run_worker(collection_error=True), 1)
        value = json.loads((self.output / "worker-result.json").read_bytes())
        self.assertEqual(value["outcome"], "mismatch_observed")
        self.assertEqual(value["returncode"], 1)
        self.assertEqual(self.study.events.count("full"), 1)
        self.assertEqual(self.study.events.count("read"), 1)

    def test_late_audit_failure_preserves_completed_comparison_but_fails_attempt(self):
        self.assertEqual(self.run_worker(late_audit=True), 1)
        value = json.loads((self.output / "worker-result.json").read_bytes())
        self.assertEqual(value["outcome"], "execution_error")
        self.assertEqual(value["comparison_outcome"], "no_mismatch_observed")
        self.assertEqual(value["returncode"], 1)
        self.assertTrue((self.output / "run-outcome.json").is_file())

    def test_postlude_failure_cannot_erase_preceding_numeric_mismatch(self):
        self.study.observed.full_report = report().replace('"ce":4.0', '"ce":4.5')
        with self.assertRaisesRegex(RuntimeError, "postlude"):
            self.run_worker(postlude_error=True)
        value = json.loads((self.output / "worker-result.json").read_bytes())
        self.assertEqual(value["outcome"], "mismatch_observed")
        self.assertEqual(value["returncode"], 1)
        self.assertEqual(self.study.events.count("full"), 1)
        self.assertEqual(self.study.events.count("read"), 1)


if __name__ == "__main__":
    unittest.main()
