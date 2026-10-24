"""Orchestration fixtures only. Never imports a native model or uses a GPU."""
import ast
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import struct
import tempfile
import types
import unittest
import zipfile
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "prelude_pair", Path(__file__).with_name("run_byte_study_prelude_pair.py"))
API = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(API)
P, R, B = API.prelude, API.retained, API.boundary


def encoded(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


REQUEST = encoded({"schema": "spiraltorch.byte_corpus.request.v9", "rate": .001}).encode()


def selection():
    names = [f"case-{i}" for i in range(25)]
    names[23] = R.CASE
    return {"request_sha256": R.REQUEST_SHA256, "case": R.CASE, "case_index": 23,
            "full": {"cases": [{"name": name} for name in names]},
            "payloads": {"request.json": REQUEST}}


def report(updates):
    return encoded({"schema": f"spiraltorch.byte_corpus.{'partial' if updates == 1 else 'result'}.v9",
                    "request_sha256": R.REQUEST_SHA256,
                    "cases": [{"name": row["name"], "training": [
                        {"revision": i + 1, "ce": 5.0 - i} for i in range(updates)]}
                        for row in selection()["full"]["cases"]]})


def model(updates=1):
    return encoded({"schema": "spiraltorch.nn.byte_decoder_checkpoint.v6",
                    "attempted_revision": str(updates),
                    "model": {"weight": [1.0, -0.0]}, "optimizer": {"m": [0.1], "steps": updates}})


def checkpoint(updates=1):
    return encoded({"schema": "spiraltorch.byte_corpus.checkpoint.v9",
                    "request_sha256": R.REQUEST_SHA256, "completed_updates": updates,
                    "cases": [{"name": row["name"], "model_json": model(updates),
                               "training": [{"revision": i + 1, "ce": 5.0 - i} for i in range(updates)]}
                              for row in selection()["full"]["cases"]]})


class Saved:
    def __init__(self, raw):
        self.raw = raw

    def to_json(self):
        return self.raw


class Study:
    total_updates = 2
    optimizer_recipe_json = "{}"

    def __init__(self, output):
        self.output, self.events = output, []
        self.request_sha256 = R.REQUEST_SHA256
        self.report = report(2)
        self.captured = model()
        self.fail = None
        self.loaded = None
        self.override = None

    def advance_wgpu(self, stop=None, *, resume=None):
        stage = "partial" if stop == 1 else "resume"
        self.events.append(stage)
        if self.fail == stage:
            raise RuntimeError("synthetic " + stage)
        if stage == "resume" and resume is not self.loaded:
            raise ValueError("not the same partial checkpoint")
        updates = 1 if stop == 1 else 2
        return types.SimpleNamespace(report_json=report(updates), checkpoint=Saved(checkpoint(updates)))

    def checkpoint_from_json(self, raw):
        self.events.append("load")
        if raw != checkpoint().encode():
            raise ValueError("partial bytes changed")
        self.loaded = Saved(raw.decode())
        return self.loaded

    def run_wgpu(self):
        self.events.append("ordinary")
        if self.fail == "full":
            raise RuntimeError("synthetic full")
        return self.report

    def run_wgpu_with_retained_boundary(self, saved, case):
        self.events.append("retained")
        if saved is not self.loaded or case != R.CASE:
            raise ValueError("wrong capture reference")
        if self.fail == "full":
            raise RuntimeError("synthetic full")
        return self

    @property
    def report_json(self):
        if self.fail == "export":
            raise RuntimeError("synthetic export")
        return self.report

    @property
    def run_outcome_json(self):
        value = {"schema": "spiraltorch.byte_retained_boundary.v1",
                 "request_sha256": R.REQUEST_SHA256, "case": R.CASE,
                 "requested_capture_after_update": 1, "capture_mode": "retained",
                 "boundary_read_policy": "explicit_after_run_returns", "boundary_captured": True,
                 "release_ready": False, "status": "completed", "all_cases_completed": True,
                 "report": json.loads(self.report)}
        value.update(self.override or {})
        return encoded(value)

    def read_boundary(self):
        self.events.append("read")
        if not (self.output / "run-outcome.json").exists():
            raise AssertionError("outcome must precede read")
        if self.fail != "export" and not (self.output / "full-report.json").exists():
            raise AssertionError("full report must precede read")
        if self.fail == "read":
            raise RuntimeError("synthetic read")
        return Saved(self.captured)

    def case_checkpoint(self, saved, name):
        self.events.append("reference")
        if saved is not self.loaded or name != R.CASE:
            raise ValueError("not this attempt")
        return Saved(model())


class Temporary(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.output = Path(temp.name)
        patcher = mock.patch.object(R, "REQUEST_SHA256", B.digest(REQUEST))
        patcher.start()
        self.addCleanup(patcher.stop)

    def target(self, mode):
        directory = self.output / mode
        directory.mkdir()
        return P.Target(mode, selection(), directory), Study(directory)

    def exercise_target(self, target, study):
        target.begin(study, REQUEST)
        target.partial(study)
        target.resume(study)
        return target.full(study)


class TargetTests(Temporary):
    def test_ordinary_calls_native_run_without_boundary_api(self):
        target, study = self.target("ordinary")
        self.assertEqual(self.exercise_target(target, study), report(2))
        self.assertEqual(study.events, ["partial", "load", "resume", "ordinary"])
        self.assertEqual(target.record["outcome"], "no_mismatch_observed")
        self.assertFalse((target.output / "observed-boundary.json").exists())

    def test_retained_reads_only_after_complete_report_and_uses_same_attempt(self):
        target, study = self.target("retained")
        self.exercise_target(target, study)
        self.assertEqual(study.events, ["partial", "load", "resume", "retained", "read", "reference"])
        self.assertEqual((target.output / "first-checkpoint.json").read_text(), checkpoint())
        self.assertEqual((target.output / "reference-boundary.json").read_text(), model())
        self.assertEqual(target.record["native_calls"], ["partial_1", "resume_2", "retained_full_2"])

    def test_changed_request_fails_before_any_target_gpu_call(self):
        target, study = self.target("ordinary")
        with self.assertRaisesRegex(ValueError, "regenerated"):
            target.begin(study, REQUEST + b" ")
        self.assertEqual(study.events, [])

    def test_target_cannot_be_reordered_or_repeated(self):
        target, study = self.target("ordinary")
        for call in (target.partial, target.resume, target.full):
            with self.assertRaises(ValueError):
                call(study)
        self.exercise_target(target, study)
        with self.assertRaises(ValueError):
            target.begin(study, REQUEST)
        with self.assertRaises(ValueError):
            target.full(study)
        self.assertEqual(study.events.count("ordinary"), 1)

    def test_first_and_resumed_bytes_survive_full_error(self):
        target, study = self.target("ordinary")
        study.fail = "full"
        with self.assertRaisesRegex(RuntimeError, "synthetic full"):
            self.exercise_target(target, study)
        self.assertEqual((target.output / "first-checkpoint.json").read_text(), checkpoint())
        self.assertEqual((target.output / "resumed-checkpoint.json").read_text(), checkpoint(2))
        self.assertEqual(json.loads((target.output / "target.json").read_text())["outcome"], "execution_error")

    def test_report_mismatch_is_retained_without_retry(self):
        for mode in P.MODES:
            with self.subTest(mode=mode):
                target, study = self.target(mode)
                study.report = report(2).replace('"ce":4.0', '"ce":4.5', 1)
                with self.assertRaises(P.TargetMismatch):
                    self.exercise_target(target, study)
                self.assertEqual(target.record["outcome"], "mismatch_observed")
                self.assertEqual((target.output / "full-report.json").read_text(), study.report)
                self.assertEqual(study.events.count(mode), 1)

    def test_signed_zero_boundary_difference_is_not_tolerated(self):
        target, study = self.target("retained")
        study.captured = model().replace("-0.0", "0.0")
        with self.assertRaises(P.TargetMismatch):
            self.exercise_target(target, study)
        self.assertTrue(target.record["report_comparison"]["exact_bytes_equal"])
        self.assertFalse(target.record["boundary_comparison"]["exact_bytes_equal"])

    def test_read_failure_does_not_erase_report_difference(self):
        target, study = self.target("retained")
        study.report = report(2).replace('"ce":4.0', '"ce":4.5', 1)
        study.fail = "read"
        with self.assertRaisesRegex(RuntimeError, "boundary observation"):
            self.exercise_target(target, study)
        self.assertEqual(target.record["outcome"], "boundary_read_error")
        self.assertEqual(target.record["comparison_outcome"], "mismatch_observed")
        self.assertFalse(target.record["report_comparison"]["exact_bytes_equal"])

    def test_export_failure_still_attempts_boundary_read(self):
        target, study = self.target("retained")
        study.fail = "export"
        with self.assertRaises(ValueError):
            self.exercise_target(target, study)
        self.assertIn("read", study.events)
        self.assertTrue((target.output / "observed-boundary.json").is_file())
        self.assertIn("report_export_error", target.record)
        self.assertEqual(target.record["outcome"], "execution_error")

    def test_partial_failure_cannot_reach_resume_or_full(self):
        target, study = self.target("ordinary")
        study.fail = "partial"
        target.begin(study, REQUEST)
        with self.assertRaises(RuntimeError):
            target.partial(study)
        for call in (target.resume, target.full):
            with self.assertRaises(ValueError):
                call(study)
        self.assertEqual(study.events, ["partial"])

    def test_inconsistent_outcome_does_not_become_target_success(self):
        target, study = self.target("retained")
        study.override = {"all_cases_completed": False}
        with self.assertRaises(ValueError):
            self.exercise_target(target, study)
        self.assertEqual(target.record["outcome"], "execution_error")


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.raw = (API.ROOT / P.TEST_PATH).read_bytes()

    def test_only_three_top_level_target_assignments_and_begin_are_changed(self):
        actual, lines = P.instrumented_method(self.raw)
        self.assertEqual(lines, [586, 587, 588])
        owner = next(n for n in ast.parse(self.raw).body if isinstance(n, ast.ClassDef) and n.name == "StudyGpu")
        original = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == P.METHODS[-1])
        restored = copy.deepcopy(actual.body[0])
        restored.body = [node for node in restored.body
                         if ast.dump(node) != ast.dump(ast.parse("self._prelude.begin(study, raw)").body[0])]
        for old, new in P.REPLACEMENTS:
            expected = ast.dump(ast.parse(new).body[0])
            indices = [i for i, node in enumerate(restored.body) if ast.dump(node) == expected]
            self.assertEqual(len(indices), 1)
            restored.body[indices[0]] = ast.parse(old).body[0]
        self.assertEqual(ast.dump(restored), ast.dump(original))

    def test_any_source_change_is_rejected_without_importing_it(self):
        with self.assertRaisesRegex(ValueError, "pinned source"):
            P.instrumented_method(self.raw + b"\n")

    def test_moved_assignment_rejected_even_with_a_rebound_fixture_hash(self):
        changed = self.raw.replace(b"        first = study.advance_wgpu(1)\n", b"        first = changed()\n")
        with mock.patch.object(P, "TEST_SHA256", B.digest(changed)):
            with self.assertRaisesRegex(ValueError, "assignment"):
                P.instrumented_method(changed)


class SuiteTests(Temporary):
    def suite(self, failure=None):
        events = []
        study = Study(self.output / "target")

        def predecessor(instance):
            events.append(instance._testMethodName)
            if instance._testMethodName == failure:
                with instance.subTest(stage="synthetic"):
                    instance.fail("synthetic early failure")

        def check(instance):
            events.append("postlude")
            if failure == "postlude":
                instance.fail("synthetic postlude failure")

        methods = {name: predecessor for name in P.METHODS[:-1]}
        methods["check"] = check
        cls = type("FakeStudyGpu", (unittest.TestCase,), methods)
        module = types.SimpleNamespace(StudyGpu=cls, __file__="synthetic.py", study=study, raw=REQUEST)
        tree = ast.parse(
            "def " + P.METHODS[-1] + "(self):\n"
            "    self._prelude.begin(study, raw)\n"
            "    self._prelude.partial(study)\n"
            "    self._prelude.resume(study)\n"
            "    self._prelude.full(study)\n"
            "    self.check()\n")
        return module, tree, events, study

    def run_suite(self, failure=None, mismatch=False):
        module, tree, events, study = self.suite(failure)
        if mismatch:
            study.report = report(2).replace('"ce":4.0', '"ce":4.5', 1)
        record = {}
        with mock.patch.object(P, "instrumented_method", return_value=(tree, [1, 2, 3])):
            code = P.exercise(module, b"synthetic", "ordinary", selection(), self.output, record)
        return code, record, events, study

    def test_six_methods_in_exact_order(self):
        code, record, events, _ = self.run_suite()
        self.assertEqual(code, 0)
        self.assertEqual(events, list(P.METHODS[:-1]) + ["postlude"])
        self.assertEqual([e["method"] for e in record["tests"]["events"]], list(P.METHODS))

    def test_failed_subtest_stops_before_later_methods_and_target(self):
        code, record, events, study = self.run_suite(P.METHODS[1])
        self.assertEqual(code, 1)
        self.assertEqual(events, list(P.METHODS[:2]))
        self.assertEqual(study.events, [])
        self.assertEqual(record["outcome"], "prelude_failure")

    def test_postlude_failure_is_not_relabelled_as_a_target_mismatch(self):
        code, record, _, _ = self.run_suite("postlude")
        self.assertEqual(code, 1)
        self.assertEqual(record["target_outcome"], "no_mismatch_observed")
        self.assertEqual(record["outcome"], "target_or_postlude_error")

    def test_target_mismatch_stops_before_postlude_but_keeps_prefix(self):
        code, record, events, _ = self.run_suite(mismatch=True)
        self.assertEqual(code, 1)
        self.assertEqual(events, list(P.METHODS[:-1]))
        self.assertEqual(record["outcome"], "mismatch_observed")

    def test_diagnostic_environment_is_restored_after_failure(self):
        key = "SPIRALTORCH_PREPARED_STUDY_DIAGNOSTICS"
        with mock.patch.dict(os.environ, {key: "previous-value"}):
            self.run_suite(P.METHODS[0])
            self.assertEqual(os.environ[key], "previous-value")

    def test_completed_native_segment_is_observed_even_on_first_assertion_failure(self):
        class NativeSegment:
            pass

        def failed():
            completed_segment = NativeSegment()
            raise AssertionError(type(completed_segment).__name__)

        result = P.LoggedResult(unittest.runner._WritelnDecorator(io.StringIO()), True, 2,
                                segment_type=NativeSegment, source_file=__file__)
        try:
            failed()
        except AssertionError:
            import sys
            result.observe_completed_segment(sys.exc_info())
        self.assertTrue(result.native_completion_observed)

    def test_first_test_failure_without_completion_is_unknown_not_false(self):
        code, record, _, _ = self.run_suite(P.METHODS[0])
        self.assertEqual(code, 1)
        self.assertIsNone(record["native_execution_completed"])


class QualificationTests(Temporary):
    def complete(self, mode, mismatch=False):
        arm = self.output / ("01-ordinary" if mode == "ordinary" else "02-retained")
        target_dir = arm / "target"
        target_dir.mkdir(parents=True)
        target, study = P.Target(mode, selection(), target_dir), Study(target_dir)
        if mismatch:
            study.report = report(2).replace('"ce":4.0', '"ce":4.5', 1)
        try:
            self.exercise_target(target, study)
        except P.TargetMismatch:
            pass
        return arm, target

    def worker_evidence(self, arm, mode, mismatch=False):
        count = 2 if mismatch else 4
        verdict = "mismatch_observed" if mismatch else "no_mismatch_observed"
        native = {"filename": "synthetic.so", "sha256": P.NATIVE_SHA256}
        events = [{"method": name, "outcome": "ok"} for name in P.METHODS]
        if mismatch:
            events[-1].update(outcome="error", exception_type="TargetMismatch")
        manifest = {"fresh_inference_sha256": "synthetic-source",
                    "harness_files": {Path(API.replay.__file__).name: "synthetic-harness"}}
        record = {"schema": "spiraltorch.prelude_pair.worker.v1", "protocol": API.PROTOCOL,
                  "mode": mode, "release_ready": False, "outcome": verdict,
                  "returncode": int(mismatch), "native_library": native,
                  "verified_python_files": 1, "setup_subprocesses": [],
                  "foreign_import_attempts": [], "blocked_setup_import_attempts": [],
                  "unexpected_child_spawns": 0, "native_execution_completed": True,
                  "tests": {"tests_run": 6, "events": events, "failures": [],
                            "errors": [P.METHODS[-1]] if mismatch else [],
                            "skipped": [], "unexpected_skips": [], "expected_failures": [],
                            "unexpected_successes": []}, "child_runs": []}
        for index in range(count):
            directory = arm / f"children/{index:04d}"
            directory.mkdir(parents=True)
            row = {"directory": str(directory.relative_to(arm)), "valid": True,
                   "returncode": 0, "spawn_observed": True}
            child = {"schema": "spiraltorch.hosted_byte_replay.child.v1",
                     "outcome": "no_foreign_ml_observed", "source_sha256": "synthetic-source",
                     "harness_sha256": "synthetic-harness", "native_library": native,
                     "release_ready": False, "unexpected_child_spawns": 0,
                     "foreign_import_attempts": [], "blocked_setup_import_attempts": [],
                     "setup_subprocesses": []}
            B.save(directory / "parent-result.json", row)
            B.save(directory / "child-result.json", child)
            record["child_runs"].append(row)
        B.save(arm / "manifest.json", manifest)
        B.save(arm / "worker-result.json", record)
        B.save(arm / "runtime.json", {"schema": "spiraltorch.synthetic_gpu_runtime.v1",
                                    "native_library": native,
                                    "adapter": {"name": "synthetic", "backend": "Metal",
                                                "device_type": "VirtualGpu"}})
        return record

    def test_recompute_both_verdicts_from_raw_reports(self):
        for mode in P.MODES:
            arm, _ = self.complete(mode)
            self.assertEqual(API.verify_target(arm, mode, selection()), "no_mismatch_observed")

    def test_mismatch_is_not_a_supervisor_parse_error(self):
        arm, _ = self.complete("retained", mismatch=True)
        self.assertEqual(API.verify_target(arm, "retained", selection()), "mismatch_observed")

    def test_changed_raw_report_rejected_despite_claimed_success(self):
        arm, _ = self.complete("ordinary")
        (arm / "target/full-report.json").write_text(report(2) + "\n")
        with self.assertRaisesRegex(ValueError, "saved report"):
            API.verify_target(arm, "ordinary", selection())

    def test_boundary_reference_must_be_from_this_attempt(self):
        arm, _ = self.complete("retained")
        path = arm / "target/first-checkpoint.json"
        value = json.loads(path.read_text())
        value["cases"][23]["model_json"] = model().replace("-0.0", "0.0")
        path.write_text(encoded(value))
        with self.assertRaisesRegex(ValueError, "this attempt"):
            API.verify_target(arm, "retained", selection())

    def test_ordinary_checkpoint_history_is_checked_too(self):
        arm, _ = self.complete("ordinary")
        path = arm / "target/first-checkpoint.json"
        value = json.loads(path.read_text())
        value["cases"][0]["training"][0]["ce"] = 42.0
        path.write_text(encoded(value))
        with self.assertRaisesRegex(ValueError, "checkpoint and report"):
            API.verify_target(arm, "ordinary", selection())

    def test_retained_outcome_identity_is_rechecked_by_supervisor(self):
        arm, _ = self.complete("retained")
        path = arm / "target/run-outcome.json"
        value = json.loads(path.read_text())
        value["case"] = "some-other-case"
        path.write_text(encoded(value))
        with self.assertRaisesRegex(ValueError, "outcome"):
            API.verify_target(arm, "retained", selection())

    def test_complete_arm_and_exact_mismatch_can_both_qualify(self):
        for mode, mismatch in (("ordinary", False), ("retained", True)):
            with self.subTest(mode=mode):
                arm, _ = self.complete(mode, mismatch)
                self.worker_evidence(arm, mode, mismatch)
                self.assertEqual(API.qualify_arm(arm, mode, selection(), int(mismatch)),
                                 "mismatch_observed" if mismatch else "no_mismatch_observed")

    def test_missing_child_record_cannot_be_hidden_by_parent_valid_flag(self):
        arm, _ = self.complete("ordinary")
        self.worker_evidence(arm, "ordinary")
        (arm / "children/0001/child-result.json").unlink()
        with self.assertRaises(FileNotFoundError):
            API.qualify_arm(arm, "ordinary", selection(), 0)

    def test_changed_child_native_is_rejected(self):
        arm, _ = self.complete("retained")
        self.worker_evidence(arm, "retained")
        path = arm / "children/0000/child-result.json"
        value = json.loads(path.read_text())
        value["native_library"]["sha256"] = "not-the-wheel"
        path.write_text(encoded(value))
        with self.assertRaisesRegex(ValueError, "child evidence"):
            API.qualify_arm(arm, "retained", selection(), 0)

    def test_metadata_failure_does_not_qualify_a_passing_target(self):
        arm, _ = self.complete("ordinary")
        self.worker_evidence(arm, "ordinary")
        (arm / "runtime.json").write_text('{"status":"metadata_error"}')
        with self.assertRaises(ValueError):
            API.qualify_arm(arm, "ordinary", selection(), 0)

    def test_worker_signal_or_changed_history_rejected(self):
        arm, _ = self.complete("ordinary")
        record = self.worker_evidence(arm, "ordinary")
        with self.assertRaises(ValueError):
            API.qualify_arm(arm, "ordinary", selection(), -9)
        record["tests"]["events"][0]["outcome"] = "skip"
        (arm / "worker-result.json").write_text(encoded(record))
        with self.assertRaises(ValueError):
            API.qualify_arm(arm, "ordinary", selection(), 0)


class IdentityTests(Temporary):
    def test_cross_serialization_training_uses_exact_f32_bits(self):
        left = [{"revision": 1, "ce": 0.1}]
        right = [{"revision": 1, "ce": 0.10000000149011612}]
        self.assertTrue(API.matching_training(left, right, 1))
        # A single f32 ULP remains a difference; this is not a tolerance.
        bits = struct.unpack("<I", struct.pack("<f", right[0]["ce"]))[0]
        right[0]["ce"] = struct.unpack("<f", struct.pack("<I", bits + 1))[0]
        self.assertFalse(API.matching_training(left, right, 1))

    def test_cross_serialization_never_accepts_invalid_types_or_signed_zero_drift(self):
        right = [{"revision": 1, "ce": 0.0}]
        for value in (-0.0, True, float("nan"), float("inf"), 1e39, 10 ** 500):
            with self.subTest(value=str(value)[:20]):
                self.assertFalse(API.matching_training([{"revision": 1, "ce": value}], right, 1))
        self.assertFalse(API.matching_training([{"revision": True, "ce": 0.0}], right, 1))
        self.assertFalse(API.matching_training([], right, 1))

    def test_report_value_itself_must_be_the_f64_representation_of_f32(self):
        self.assertFalse(API.matching_training([{"revision": 1, "ce": 0.1}],
                                              [{"revision": 1, "ce": 0.1}], 1))
    def test_wheel_digest_checked_before_unzipping(self):
        wheel = self.output / "not-the-wheel.whl"
        wheel.write_bytes(b"not a zip")
        with self.assertRaisesRegex(ValueError, "recovered hosted"):
            API.require_wheel(wheel)

    def test_native_digest_checked_even_when_wheel_digest_is_rebound_for_fixture(self):
        wheel = self.output / "synthetic.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("spiraltorch/spiraltorch.abi3.so", b"not a native extension")
        with mock.patch.object(P, "WHEEL_SHA256", API.replay.digest(wheel)):
            with self.assertRaisesRegex(ValueError, "native payload"):
                API.require_wheel(wheel)

    def test_installed_python_facade_must_match_archive(self):
        wheel = self.output / "synthetic.whl"
        directory = self.output / "spiraltorch"
        directory.mkdir()
        path = directory / "__init__.py"
        path.write_bytes(b"# same code\n")
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("spiraltorch/__init__.py", b"# same code\n")
        module = types.SimpleNamespace(__file__=str(path))
        self.assertEqual(API.installed_package(wheel, module), 1)
        path.write_bytes(b"# changed code\n")
        with self.assertRaisesRegex(ValueError, "facade"):
            API.installed_package(wheel, module)

    def test_timeout_bounds_rejected_before_identity_commands(self):
        for value in (0, -1, 5401, True, 1.5):
            with self.subTest(value=value), mock.patch.object(API.replay, "git_identity") as git:
                with self.assertRaises(ValueError):
                    API.preflight(self.output, self.output, self.output / "unused", value)
                git.assert_not_called()

    def test_worker_claim_is_single_use(self):
        campaign = self.output / "campaign"
        arm = campaign / "01-ordinary"
        arm.mkdir(parents=True)
        manifest = {"protocol": API.PROTOCOL, "source_root": str(self.output), "timeout_seconds": 3}
        B.save(campaign / "manifest.json", manifest)
        B.save(arm / "manifest.json", dict(manifest, mode="ordinary",
                                         campaign_sha256=API.replay.digest(campaign / "manifest.json")))
        with mock.patch.object(API, "preflight", return_value=(manifest, selection())):
            API.claim_worker(arm, Path("synthetic.whl"))
            with self.assertRaises(FileExistsError):
                API.claim_worker(arm, Path("synthetic.whl"))

    def test_changed_campaign_rejected_before_worker_import(self):
        campaign = self.output / "campaign"
        arm = campaign / "01-ordinary"
        arm.mkdir(parents=True)
        B.save(campaign / "manifest.json", {"protocol": API.PROTOCOL})
        B.save(arm / "manifest.json", {"mode": "ordinary", "protocol": "changed"})
        with mock.patch.object(API, "preflight") as preflight:
            with self.assertRaises(ValueError):
                API.claim_worker(arm, Path("synthetic.whl"))
            preflight.assert_not_called()


class WorkerTests(Temporary):
    def run_worker(self, *, metadata_error=False, completion=True, outcome="no_mismatch_observed",
                   child_error=False, foreign_attempt=False, wrong_native=False, bad_surface=False):
        module = types.SimpleNamespace(st=object(), FRESH_INFERENCE="synthetic inference",
                                       subprocess=object())
        original = module.subprocess
        manifest = {"source_root": str(API.ROOT), "mode": "ordinary",
                    "expected_children": {"no_mismatch_observed": 4, "mismatch_observed": 2}}
        native = {"filename": "synthetic.so",
                  "sha256": "wrong" if wrong_native else P.NATIVE_SHA256}
        metadata_calls = []
        children = mock.Mock()
        if child_error:
            children.require_complete.side_effect = ValueError("synthetic child audit failure")

        def exercise(_module, _source, _mode, _selected, _output, record):
            record.update(outcome=outcome, target_outcome=outcome, native_execution_completed=completion)
            if foreign_attempt:
                record["foreign_import_attempts"].append("torch")
            return 0 if outcome == "no_mismatch_observed" else 1

        def metadata(_st):
            metadata_calls.append(True)
            self.assertTrue((self.output / "worker-result.json").is_file())
            if metadata_error:
                raise RuntimeError("synthetic metadata failure")
            return {"schema": "spiraltorch.synthetic_gpu_runtime.v1", "native_library": native,
                    "adapter": {"name": "synthetic", "backend": "Metal", "device_type": "VirtualGpu"}}

        spec = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda _: None))
        with contextlib.ExitStack() as stack:
            for owner, name, options in (
                (API, "claim_worker", {"return_value": (manifest, selection())}),
                (API.replay, "require_foreign_free_environment", {}),
                (API.replay, "install_import_guards", {}),
                (API.sys, "addaudithook", {}),
                (API.importlib.util, "spec_from_file_location", {"return_value": spec}),
                (API.importlib.util, "module_from_spec", {"return_value": module}),
                (API.importlib, "import_module", {"return_value": object()}),
                (API.replay, "native_identity", {"return_value": native}),
                (API, "installed_package", {"return_value": 1}),
                (API.replay, "GuardedChildren", {"return_value": children}),
                (P, "exercise", {"side_effect": exercise}),
                (R, "runtime_identity", {"side_effect": metadata}),
                (R, "require_client_surface",
                 {"side_effect": ValueError("synthetic missing surface")} if bad_surface else {}),
            ):
                stack.enter_context(mock.patch.object(owner, name, **options))
            stack.enter_context(mock.patch.dict(os.environ))
            code = API.worker(self.output, Path("never-imported.whl"))
        self.assertIs(module.subprocess, original)
        return code, json.loads((self.output / "worker-result.json").read_text()), metadata_calls, children

    def test_worker_result_is_persisted_before_metadata(self):
        code, record, metadata, children = self.run_worker()
        self.assertEqual(code, 0)
        self.assertEqual(record["outcome"], "no_mismatch_observed")
        self.assertEqual(metadata, [True])
        children.require_complete.assert_called_once_with(4)

    def test_metadata_error_fails_process_without_rewriting_comparison(self):
        code, record, metadata, _ = self.run_worker(metadata_error=True)
        self.assertEqual(code, 1)
        self.assertEqual(record["outcome"], "no_mismatch_observed")
        self.assertEqual(record["returncode"], 0)
        self.assertEqual(metadata, [True])

    def test_unknown_completion_does_not_create_a_device_for_metadata(self):
        code, record, metadata, _ = self.run_worker(completion=None, outcome="prelude_failure")
        self.assertEqual(code, 1)
        self.assertIsNone(record["native_execution_completed"])
        self.assertEqual(metadata, [])
        runtime = json.loads((self.output / "runtime.json").read_text())
        self.assertEqual(runtime["reason"], "native_completion_unobserved")

    def test_first_test_failure_with_completed_segment_collects_metadata(self):
        code, record, metadata, _ = self.run_worker(completion=True, outcome="prelude_failure")
        self.assertEqual(code, 1)
        self.assertEqual(record["outcome"], "prelude_failure")
        self.assertEqual(metadata, [True])

    def test_child_audit_failure_blocks_qualification_without_erasing_target(self):
        code, record, _, children = self.run_worker(outcome="mismatch_observed", child_error=True)
        self.assertEqual(code, 1)
        self.assertEqual(record["outcome"], "execution_error")
        self.assertEqual(record["target_outcome"], "mismatch_observed")
        children.require_complete.assert_called_once_with(2)

    def test_foreign_import_during_exercise_cannot_pass(self):
        code, record, _, _ = self.run_worker(foreign_attempt=True)
        self.assertEqual(code, 1)
        self.assertEqual(record["outcome"], "execution_error")

    def test_wrong_native_fails_before_exercise_or_metadata(self):
        code, record, metadata, children = self.run_worker(wrong_native=True)
        self.assertEqual(code, 1)
        self.assertEqual(record["outcome"], "execution_error")
        self.assertEqual(metadata, [])
        children.require_complete.assert_not_called()

    def test_missing_surface_does_not_initialize_metadata_device(self):
        code, record, metadata, _ = self.run_worker(bad_surface=True)
        self.assertEqual(code, 1)
        self.assertFalse(record["native_execution_completed"])
        self.assertEqual(metadata, [])


class SupervisorTests(Temporary):
    def manifest(self):
        return {"harness_checkout": {"tracked_changes": 0, "untracked_files": 0},
                "timeout_seconds": 3}

    def launch(self, run, qualify):
        output = self.output / "campaign"
        with mock.patch.object(API.replay, "owned_run", side_effect=run) as owned:
            with mock.patch.object(API, "qualify_arm", side_effect=qualify):
                code = API.supervise(self.manifest(), selection(), output, Path("not-imported.whl"))
        return code, json.loads((output / "result.json").read_text()), owned

    def test_exactly_two_fresh_worker_commands_even_if_first_mismatches(self):
        code, result, owned = self.launch(
            [subprocess.CompletedProcess([], 1), subprocess.CompletedProcess([], 0)],
            ["mismatch_observed", "no_mismatch_observed"])
        self.assertEqual(code, 1)
        self.assertEqual(result["outcome"], "mismatch_observed")
        self.assertEqual(owned.call_count, 2)
        commands = [call.args[0] for call in owned.call_args_list]
        self.assertNotEqual(commands[0], commands[1])
        self.assertTrue(all("--worker" in command for command in commands))

    def test_preparation_or_audit_failure_does_not_start_second_arm(self):
        code, result, owned = self.launch([subprocess.CompletedProcess([], 1)],
                                           ValueError("prelude did not finish"))
        self.assertEqual(code, 1)
        self.assertEqual(result["outcome"], "stopped")
        self.assertEqual(owned.call_count, 1)

    def test_timeout_never_retries_or_starts_other_arm(self):
        code, result, owned = self.launch(subprocess.TimeoutExpired("synthetic", 3), [])
        self.assertEqual(code, 124)
        self.assertEqual(result["outcome"], "timeout")
        self.assertEqual(owned.call_count, 1)

    def test_second_arm_timeout_keeps_first_qualified_mismatch(self):
        code, result, owned = self.launch(
            [subprocess.CompletedProcess([], 1), subprocess.TimeoutExpired("synthetic", 3)],
            ["mismatch_observed"])
        self.assertEqual(code, 124)
        self.assertEqual(result["outcome"], "timeout")
        self.assertEqual(result["comparison_outcome"], "mismatch_observed")
        self.assertEqual(owned.call_count, 2)

    def test_metadata_failure_keeps_verified_target_mismatch_without_qualifying_arm(self):
        with mock.patch.object(API, "verify_target", return_value="mismatch_observed"):
            code, result, owned = self.launch([subprocess.CompletedProcess([], 1)],
                                               ValueError("missing runtime metadata"))
        self.assertEqual(code, 1)
        self.assertEqual(result["outcome"], "stopped")
        self.assertEqual(result["comparison_outcome"], "mismatch_observed")
        self.assertEqual(result["arms"][0]["outcome"], "incomplete")
        self.assertEqual(owned.call_count, 1)

    def test_hung_metadata_keeps_verified_target_mismatch_without_retry(self):
        with mock.patch.object(API, "verify_target", return_value="mismatch_observed"):
            code, result, owned = self.launch(subprocess.TimeoutExpired("synthetic", 3), [])
        self.assertEqual(code, 124)
        self.assertEqual(result["outcome"], "timeout")
        self.assertEqual(result["comparison_outcome"], "mismatch_observed")
        self.assertEqual(owned.call_count, 1)

    def test_existing_campaign_cannot_be_overwritten_or_resumed(self):
        output = self.output / "exists"
        output.mkdir()
        with mock.patch.object(API.replay, "owned_run") as owned:
            with self.assertRaises(FileExistsError):
                API.supervise(self.manifest(), selection(), output, Path("unused.whl"))
            owned.assert_not_called()

    def test_dirty_harness_cannot_execute(self):
        manifest = self.manifest()
        manifest["harness_checkout"]["untracked_files"] = 1
        with mock.patch.object(API.replay, "owned_run") as owned:
            with self.assertRaisesRegex(ValueError, "commit"):
                API.supervise(manifest, selection(), self.output / "never", Path("unused.whl"))
            owned.assert_not_called()

    def test_default_cli_is_offline_even_if_output_is_supplied(self):
        wheel = self.output / "fake.whl"
        wheel.touch()
        with mock.patch.object(API, "preflight", return_value=({}, selection())):
            with mock.patch.object(API, "supervise") as supervise:
                with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
                    self.assertEqual(API.main(["--wheel", str(wheel), "--failure-root", str(self.output),
                                               "--source-root", str(self.output),
                                               "--output", str(self.output / "never")]), 0)
                self.assertFalse(json.loads(out.getvalue())["gpu_executed"])
                supervise.assert_not_called()


if __name__ == "__main__":
    unittest.main()
