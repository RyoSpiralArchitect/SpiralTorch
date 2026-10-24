"""Pinned historical prelude with one explicitly instrumented v9 target."""
import ast
import copy
import os
import unittest

import run_byte_study_retained as retained

boundary, replay = retained.boundary, retained.replay
TEST_PATH = "bindings/st-py/tests/test_nn_byte_study.py"
TEST_SHA256 = "c0859450ba491e7a7fd6df14b4124a90d77c153a7c443fa106523d2f2707a06b"
WHEEL_SHA256 = "4930f6a32f1774984a0bf4ab407895c957d3fd29cad4c74fd6b8553b017e0854"
NATIVE_SHA256 = "57488cd19ee0c30065b3489b596286d0b8cb28ee77be4ecb7228b10393fcae26"
MODES = ("ordinary", "retained")
METHODS = (
    "test_adamw_optimizer_history_survives_study_resume",
    "test_clipped_contextual_mixtures_preserve_all_frozen_clocks",
    "test_clipped_learning_freeze_and_resume_are_one_rust_contract",
    "test_detached_calls_share_a_recipe_not_mutable_models",
    "test_full_run_and_split_resume_are_identical",
    "test_preparation_chain_uses_one_shared_runner",
)
REPLACEMENTS = (
    ("first = study.advance_wgpu(1)", "first = self._prelude.partial(study)"),
    ("resumed = study.advance_wgpu(resume=study.checkpoint_from_json(first.checkpoint.to_json()))",
     "resumed = self._prelude.resume(study)"),
    ("full_report = study.run_wgpu()", "full_report = self._prelude.full(study)"),
)


def instrumented_method(raw):
    """Inspect only; compile/import/native execution belong to the worker."""
    if boundary.digest(raw) != TEST_SHA256:
        raise ValueError("historical test file differs from the exact pinned source")
    tree = ast.parse(raw)
    owners = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StudyGpu"]
    if len(owners) != 1:
        raise ValueError("expected one StudyGpu class")
    methods = {n.name: n for n in owners[0].body if isinstance(n, ast.FunctionDef)}
    if tuple(sorted(n for n in methods if n.startswith("test_"))) != METHODS:
        raise ValueError("historical GPU method sequence differs")
    method = copy.deepcopy(methods[METHODS[-1]])
    changed = []
    for original, replacement in REPLACEMENTS:
        expected = ast.dump(ast.parse(original).body[0])
        indices = [i for i, node in enumerate(method.body) if ast.dump(node) == expected]
        if len(indices) != 1:
            raise ValueError("target assignment moved or became ambiguous")
        i = indices[0]
        changed.append(method.body[i].lineno)
        method.body[i] = ast.copy_location(ast.parse(replacement).body[0], method.body[i])
    first = next(i for i, node in enumerate(method.body) if node.lineno == changed[0])
    method.body.insert(first, ast.copy_location(
        ast.parse("self._prelude.begin(study, raw)").body[0], method.body[first]))
    return ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), changed


class TargetMismatch(Exception):
    """A fully recorded target mismatch, not a request to retry."""


class Target:
    def __init__(self, mode, selected, output):
        if mode not in MODES:
            raise ValueError("unknown arm")
        self.mode, self.selected, self.output = mode, selected, output
        self.record = {"schema": "spiraltorch.prelude_target.v1", "mode": mode,
                       "release_ready": False, "outcome": "not_reached",
                       "native_calls": [], "stage": "prelude"}
        self.first = self.resumed = self.loaded = None
        self.persisted = False

    def begin(self, study, raw):
        raw = raw.encode() if isinstance(raw, str) else raw
        if (type(raw) is not bytes or boundary.digest(raw) != retained.REQUEST_SHA256
                or raw != self.selected["payloads"]["request.json"]
                or study.request_sha256 != retained.REQUEST_SHA256 or study.total_updates != 2
                or self.record["outcome"] != "not_reached"):
            raise ValueError("regenerated v9 request differs; no v9 training permitted")
        retained.save_raw(self.output / "request.json", raw)
        self.record.update(outcome="execution_error", stage="ready")

    def partial(self, study):
        if self.record["stage"] != "ready":
            raise ValueError("partial call out of order")
        self.record.update(stage="partial", native_calls=["partial_1"])
        self.first = study.advance_wgpu(1)
        raw = retained.save_raw(self.output / "first-report.json", self.first.report_json)
        retained.save_raw(self.output / "first-checkpoint.json", self.first.checkpoint.to_json())
        retained.validate_report(raw, self.selected, 1)
        self.record["stage"] = "partial_complete"
        return self.first

    def resume(self, study):
        if self.record["stage"] != "partial_complete":
            raise ValueError("resume call out of order")
        self.record["stage"] = "resume"
        self.loaded = study.checkpoint_from_json(boundary.read(self.output / "first-checkpoint.json"))
        self.record["native_calls"].append("resume_2")
        self.resumed = study.advance_wgpu(resume=self.loaded)
        raw = retained.save_raw(self.output / "resumed-report.json", self.resumed.report_json)
        retained.save_raw(self.output / "resumed-checkpoint.json", self.resumed.checkpoint.to_json())
        retained.validate_report(raw, self.selected, 2)
        self.record["stage"] = "resume_complete"
        return self.resumed

    def retained_full(self, study):
        observed = study.run_wgpu_with_retained_boundary(self.loaded, retained.CASE)
        raw = retained.save_raw(self.output / "run-outcome.json", observed.run_outcome_json)
        outcome = boundary.decode(raw)
        if (outcome.get("schema") != "spiraltorch.byte_retained_boundary.v1"
                or outcome.get("request_sha256") != retained.REQUEST_SHA256
                or outcome.get("case") != retained.CASE
                or type(outcome.get("requested_capture_after_update")) is not int
                or outcome["requested_capture_after_update"] != 1
                or outcome.get("capture_mode") != "retained"
                or outcome.get("boundary_read_policy") != "explicit_after_run_returns"
                or type(outcome.get("boundary_captured")) is not bool
                or outcome.get("release_ready") is not False):
            raise ValueError("retained outcome contract differs")
        report = None
        try:
            report = observed.report_json
            if report is not None:
                retained.save_raw(self.output / "full-report.json", report)
        except Exception as error:
            self.record["report_export_error"] = retained.error_record(error)
        try:
            captured = retained.save_raw(self.output / "observed-boundary.json",
                                         observed.read_boundary().to_json())
            reference = retained.save_raw(self.output / "reference-boundary.json",
                                          study.case_checkpoint(self.loaded, retained.CASE).to_json())
            self.record["boundary_comparison"] = retained.compare_boundaries(reference, captured)
        except Exception as error:
            self.record["boundary_error"] = retained.error_record(error)
        if (outcome.get("status") != "completed" or outcome.get("all_cases_completed") is not True
                or outcome.get("boundary_captured") is not True or report is None
                or "report_export_error" in self.record):
            raise ValueError("retained execution did not complete all cases")
        full = boundary.read(self.output / "full-report.json")
        if boundary.canonical_case(boundary.decode(full)) != boundary.canonical_case(outcome.get("report")):
            raise ValueError("retained outcome and exact report disagree")
        return full

    def full(self, study):
        if self.record["stage"] != "resume_complete":
            raise ValueError("full call out of order")
        self.record["stage"] = "full"
        self.record["native_calls"].append(self.mode + "_full_2")
        try:
            full = (self.retained_full(study) if self.mode == "retained" else retained.save_raw(
                self.output / "full-report.json", study.run_wgpu()))
            retained.validate_report(full, self.selected, 2)
            resumed = boundary.read(self.output / "resumed-report.json")
            self.record["report_comparison"] = {
                "exact_bytes_equal": full == resumed,
                "differences": retained.exact_report_differences(resumed, full, limit=64),
                "full_sha256": boundary.digest(full), "resumed_sha256": boundary.digest(resumed),
            }
            if self.loaded.to_json().encode() != boundary.read(self.output / "first-checkpoint.json"):
                raise ValueError("same-attempt partial checkpoint changed")
            mismatch = (full != resumed or self.record.get("boundary_comparison", {}).get(
                "exact_bytes_equal") is False)
            # Keep numerical evidence separate when a later observation also fails.
            self.record["comparison_outcome"] = "mismatch_observed" if mismatch else "no_mismatch_observed"
            self.record["outcome"] = ("boundary_read_error" if "boundary_error" in self.record
                                      else self.record["comparison_outcome"])
            self.record["stage"] = "finished"
        except Exception as error:
            self.record.update(outcome="execution_error", error=retained.error_record(error))
            raise
        finally:
            self.finalize()
        if self.record["outcome"] == "boundary_read_error":
            raise RuntimeError("boundary observation failed; inspect preserved target evidence")
        if mismatch:
            raise TargetMismatch("exact target difference preserved; no retry")
        return full.decode()

    def finalize(self):
        if not self.persisted:
            boundary.save(self.output / "target.json", self.record)
            self.persisted = True


class LoggedResult(unittest.TextTestResult):
    def __init__(self, *args, segment_type=None, source_file=None):
        super().__init__(*args)
        self.events = []
        self.segment_type, self.source_file = segment_type, source_file
        self.native_completion_observed = False

    def observe_completed_segment(self, err):
        # Inspect only exception-frame identities, not native properties or values.
        # A returned native segment is evidence even if its subsequent assertion failed.
        trace = err[2]
        while trace is not None:
            frame = trace.tb_frame
            if (self.segment_type is not None and frame.f_code.co_filename == self.source_file
                    and any(type(value) is self.segment_type for value in frame.f_locals.values())):
                self.native_completion_observed = True
            trace = trace.tb_next

    def startTest(self, test):
        self.events.append({"method": test._testMethodName, "outcome": "started"})
        super().startTest(test)

    def addSuccess(self, test):
        self.events[-1]["outcome"] = "ok"
        self.native_completion_observed = True
        super().addSuccess(test)

    def addError(self, test, err):
        self.observe_completed_segment(err)
        self.events[-1].update(outcome="error", exception_type=err[0].__name__)
        super().addError(test, err)

    def addFailure(self, test, err):
        self.observe_completed_segment(err)
        self.events[-1].update(outcome="failure", exception_type=err[0].__name__)
        super().addFailure(test, err)

    def addSkip(self, test, reason):
        self.events[-1].update(outcome="skip", reason=reason)
        super().addSkip(test, reason)

    def addSubTest(self, test, subtest, err):
        if err is not None:
            self.observe_completed_segment(err)
            self.events[-1].update(outcome="subtest_failure", exception_type=err[0].__name__)
        super().addSubTest(test, subtest, err)


def exercise(module, raw_source, mode, selected, output, record):
    method, changes = instrumented_method(raw_source)
    namespace = dict(module.__dict__)
    exec(compile(method, str(module.__file__), "exec"), namespace)
    subclass = type("PreparedPrelude", (module.StudyGpu,), {METHODS[-1]: namespace[METHODS[-1]]})
    target_dir = output / "target"
    target_dir.mkdir()
    target = Target(mode, selected, target_dir)
    last = subclass(METHODS[-1])
    last._prelude = target
    suite = unittest.TestSuite([module.StudyGpu(name) for name in METHODS[:-1]] + [last])
    keys = ["SPIRALTORCH_" + kind + "_STUDY_DIAGNOSTICS" for kind in replay.DIAGNOSTICS]
    old = {key: os.environ.get(key) for key in keys}
    record["target_assignment_lines"] = changes
    try:
        for key, kind in zip(keys, replay.DIAGNOSTICS):
            os.environ[key] = str(output / "prelude-failures" / kind.lower())
        with (output / "tests.log").open("x") as stream:
            segment_type = getattr(getattr(getattr(module, "st", None), "nn", None),
                                   "ByteCorpusStudySegment", None)
            result = unittest.TextTestRunner(stream=stream, verbosity=2, failfast=True,
                                             resultclass=lambda *args: LoggedResult(
                                                 *args, segment_type=segment_type,
                                                 source_file=str(module.__file__))).run(suite)
        record["tests"] = {**replay.phase_summary(result), "events": result.events}
        record["target_outcome"] = target.record["outcome"]
        record["native_execution_completed"] = (
            True if result.native_completion_observed or target.first is not None else
            None if result.testsRun else False)
        expected = ([{"method": name, "outcome": "ok"} for name in METHODS[:-1]]
                    + [{"method": METHODS[-1], "outcome": "error", "exception_type": "TargetMismatch"}])
        if (result.events == expected and len(result.errors) == 1 and not result.failures
                and not result.skipped and target.record["outcome"] == "mismatch_observed"):
            record["outcome"] = "mismatch_observed"
        elif (result.wasSuccessful() and result.testsRun == 6 and not result.skipped
              and not result.expectedFailures and target.record["outcome"] == "no_mismatch_observed"):
            record["outcome"] = "no_mismatch_observed"
        else:
            record["outcome"] = ("prelude_failure" if target.record["outcome"] == "not_reached"
                                 else "target_or_postlude_error")
        return 0 if record["outcome"] == "no_mismatch_observed" else 1
    finally:
        target.finalize()
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
