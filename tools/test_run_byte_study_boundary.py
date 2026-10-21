#!/usr/bin/env python3
"""Orchestration tests use synthetic records and mocked Rust commands, not a GPU."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "boundary", Path(__file__).with_name("run_byte_study_boundary.py"))
API = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(API)
SOURCE_IDENTITY = API.source_identity


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def fixture(root, version=6, *, differing=True):
    """Envelope fixture only; the real Rust CLI owns model/checkpoint validation."""
    folder = root / f"version-{version}"
    folder.mkdir(parents=True)
    names = ["first", "second"]
    request = {"schema": f"spiraltorch.byte_corpus.request.v{version}",
               "train_batches": [[], []], "cases": [{"name": name} for name in names]}
    raw = encoded(request)
    identity = API.digest(raw)
    full = {"schema": f"spiraltorch.byte_corpus.result.v{version}", "request_sha256": identity,
            "cases": [{"name": name, "training": [1.0, 2.0]} for name in names]}
    resumed = json.loads(json.dumps(full))
    if differing:
        resumed["cases"][1]["training"][1] = 3.0
    first = {"request_sha256": identity, "completed_updates": 1,
             "cases": [{"name": name} for name in names]}
    payloads = {"request.json": raw, "first-checkpoint.json": encoded(first),
                "full-report.json": encoded(full), "resumed-report.json": encoded(resumed),
                "first-report.json": b"{}", "resumed-checkpoint.json": b"{}",
                "runtime.json": b'{"synthetic":true}'}
    for name, value in payloads.items():
        (folder / name).write_bytes(value)
    manifest = {"schema": "spiraltorch.synthetic_run_resume_diagnostic.v1",
                "comparison": "exact_report_bytes_no_tolerance",
                "files": {name: "sha256:" + API.digest(raw) for name, raw in payloads.items()}}
    (folder / "summary.json").write_bytes(encoded(manifest))
    return folder


def rewrite(folder, name, mutate):
    value = json.loads((folder / name).read_bytes())
    mutate(value)
    raw = encoded(value)
    (folder / name).write_bytes(raw)
    manifest = json.loads((folder / "summary.json").read_bytes())
    manifest["files"][name] = "sha256:" + API.digest(raw)
    (folder / "summary.json").write_bytes(encoded(manifest))


class Selection(unittest.TestCase):
    def test_selects_first_retained_version_then_first_differing_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, 9)
            fixture(root, 6)
            chosen, reason = API.select(root)
            self.assertIsNone(reason)
            self.assertEqual((chosen["version"], chosen["case_index"], chosen["case"]),
                             (6, 1, "second"))

    def test_does_not_hunt_later_failures_after_serialization_only_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = fixture(root, 6, differing=False)
            raw = b"\n" + (folder / "resumed-report.json").read_bytes() + b"\n"
            (folder / "resumed-report.json").write_bytes(raw)
            manifest = json.loads((folder / "summary.json").read_bytes())
            manifest["files"]["resumed-report.json"] = "sha256:" + API.digest(raw)
            (folder / "summary.json").write_bytes(encoded(manifest))
            fixture(root, 9)
            chosen, reason = API.select(root)
            self.assertIsNone(chosen)
            self.assertEqual(reason, "no_canonical_case_difference_in_first_failure")

    def test_case_selection_does_not_conflate_booleans_integers_floats_or_signed_zero(self):
        for left, right in ((True, 1), (1, 1.0), (0.0, -0.0)):
            with self.subTest(left=left, right=right), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                folder = fixture(root)
                rewrite(folder, "full-report.json",
                        lambda v: v["cases"][0].update(training=[left]))
                rewrite(folder, "resumed-report.json",
                        lambda v: v["cases"][0].update(training=[right]))
                selected, _ = API.select(root)
                self.assertEqual(selected["case_index"], 0)

    def test_hash_corruption_and_unknown_paths_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = fixture(root)
            (folder / "full-report.json").write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "digest differs"):
                API.select(root)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = fixture(root)
            manifest = json.loads((folder / "summary.json").read_bytes())
            manifest["files"]["../outside.json"] = "sha256:bad"
            (folder / "summary.json").write_bytes(encoded(manifest))
            with self.assertRaises(ValueError):
                API.select(root)

    def test_foreign_identity_order_or_cursor_is_rejected(self):
        variants = [
            ("first-checkpoint.json", lambda v: v.update(completed_updates=True)),
            ("first-checkpoint.json", lambda v: v.update(completed_updates=2)),
            ("first-checkpoint.json", lambda v: v.update(request_sha256="other")),
            ("full-report.json", lambda v: v.update(request_sha256="other")),
            ("resumed-report.json", lambda v: v["cases"].reverse()),
        ]
        for name, change in variants:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                folder = fixture(root)
                rewrite(folder, name, change)
                with self.assertRaises(ValueError):
                    API.select(root)

    def test_links_nonregular_files_and_oversize_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real"
            fixture(real)
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaises(ValueError):
                API.select(alias)
            with self.assertRaises(ValueError):
                API.read(real)
            pipe = root / "pipe"
            os.mkfifo(pipe)
            with self.assertRaises(ValueError):
                API.read(pipe)
            source = real / "version-6" / "request.json"
            alias.unlink()
            alias.symlink_to(source)
            with self.assertRaises(OSError):
                API.read(alias)
            with self.assertRaises(ValueError):
                API.read(source, 1)

    def test_strict_json_rejects_duplicate_members_and_nonfinite_values(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e999}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                API.decode(raw)


class Supervision(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.failures = self.root / "failures"
        self.folder = fixture(self.failures)
        self.output = self.root / "out"
        self.executable = self.root / "target/release/examples" / API.EXAMPLE
        self.executable.parent.mkdir(parents=True)
        self.executable.write_bytes(b"mock executable; never launched")
        self.calls = []
        patch = mock.patch.object(API, "ROOT", self.root)
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(API, "source_identity", return_value={"mock": True})
        patch.start()
        self.addCleanup(patch.stop)

    def runner(self, command, *, stdout, timeout, cwd):
        self.calls.append((command, timeout))
        if command[0] == "git":
            stdout.write(b"0123456789012345678901234567890123456789\n")
        elif command[0] == "rustc":
            stdout.write(b"rustc synthetic metadata, not a real build\n")
        elif command[0] == "cargo":
            self.assertIn("--locked", command)
            self.assertIn("--target-dir", command)
            stdout.write(encoded({"reason": "compiler-artifact",
                                  "target": {"name": API.EXAMPLE, "kind": ["example"]},
                                  "executable": str(self.executable)}) + b"\n")
        else:
            attempt = Path(command[-1])
            attempt.mkdir()
            chosen, _ = API.select(self.failures)
            API.save(attempt / "invocation.json", {
                "schema": "spiraltorch.byte_boundary_invocation.v1",
                "request_sha256": chosen["request_sha256"], "case": chosen["case"],
                "requested_capture_after_update": 1, "all_cases_requested": True,
                "boundary_read_policy": "after_run_returns",
                "executable_sha256": API.digest(self.executable.read_bytes())})
            (attempt / "request.json").write_bytes(chosen["payloads"]["request.json"])
            (attempt / "reference-first-checkpoint.json").write_bytes(
                chosen["payloads"]["first-checkpoint.json"])
            (attempt / "observed-boundary.json").write_bytes(b"{}")
            (attempt / "reference-boundary.json").write_bytes(b"{}")
            API.save(attempt / "runtime.json", {"adapter": "mock adapter, not GPU evidence"})
            API.save(attempt / "run-outcome.json", {
                "status": "completed", "all_cases_completed": True, "report": chosen["resumed"]})
            API.save(attempt / "summary.json", {
                "schema": "spiraltorch.byte_boundary_probe.v1",
                "release_ready": False, "boundary_equal_to_reference": True,
                "observed_boundary_sha256": API.digest(b"{}"),
                "reference_boundary_sha256": API.digest(b"{}"),
                "run_error": None, "boundary_error": None})
        return subprocess.CompletedProcess(command, 0)

    def run_probe(self, runner=None):
        with mock.patch.object(API, "owned_run", side_effect=runner or self.runner):
            return API.supervise(self.failures, self.output, 41, 17)

    def test_one_build_one_probe_with_preserved_originals_and_no_release_claim(self):
        before = {p.name: p.read_bytes() for p in self.folder.iterdir()}
        self.assertEqual(self.run_probe(), 0)
        self.assertEqual([seconds for _, seconds in self.calls], [41, 17])
        self.assertEqual(len(self.calls), 2)
        result = json.loads((self.output / "result.json").read_bytes())
        self.assertEqual(result["status"], "observed")
        self.assertIs(result["release_ready"], False)
        comparisons = result["observation"]["selected_case"]
        self.assertEqual(comparisons["vs_original_resumed"], [])
        self.assertNotEqual(comparisons["vs_original_failed_full"], [])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.folder.iterdir()})
        for name, raw in before.items():
            self.assertEqual((self.output / "original" / name).read_bytes(), raw)
        with self.assertRaises(FileExistsError):
            self.run_probe()
        self.assertEqual(len(self.calls), 2)

    def test_missing_failure_does_not_build(self):
        with mock.patch.object(API, "owned_run") as run:
            code = API.supervise(self.root / "absent", self.output)
        self.assertEqual(code, 0)
        run.assert_not_called()
        self.assertEqual(json.loads((self.output / "result.json").read_bytes())["status"],
                         "not_selected")

    def test_build_failure_prevents_probe(self):
        with mock.patch.object(API, "owned_run", return_value=subprocess.CompletedProcess([], 1)) as run:
            self.assertEqual(API.supervise(self.failures, self.output), 1)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(json.loads((self.output / "result.json").read_bytes())["status"],
                         "build_failed")

    def test_metadata_commands_have_owned_deadlines_and_timeout_stops_before_build(self):
        for timed_out in ("git", "rustc"):
            with self.subTest(stage=timed_out):
                self.output = self.root / f"metadata-{timed_out}"
                self.calls.clear()
                def timeout(command, **kwargs):
                    completed = self.runner(command, **kwargs)
                    if command[0] == timed_out:
                        raise subprocess.TimeoutExpired(command, 10)
                    return completed
                with mock.patch.object(API, "source_identity", side_effect=SOURCE_IDENTITY):
                    self.assertEqual(self.run_probe(timeout), 2)
                self.assertTrue(all(seconds == 10 for _, seconds in self.calls))
                self.assertNotIn("cargo", [command[0] for command, _ in self.calls])
                terminal = json.loads((self.output / f"source-{timed_out}-terminal.json").read_bytes())
                self.assertEqual(terminal["status"], "timeout")
                self.assertFalse((self.output / "probe-start.json").exists())

    def test_build_timeout_prevents_probe(self):
        def timeout(command, **kwargs):
            self.calls.append((command, kwargs["timeout"]))
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        self.assertEqual(self.run_probe(timeout), 2)
        self.assertEqual(len(self.calls), 1)
        terminal = json.loads((self.output / "build-terminal.json").read_bytes())
        self.assertEqual((terminal["status"], terminal["timeout_seconds"]), ("timeout", 41))
        self.assertFalse((self.output / "probe-start.json").exists())

    def test_launch_failure_is_retained_without_retry(self):
        for stage in ("build", "probe"):
            with self.subTest(stage=stage):
                self.output = self.root / f"launch-{stage}"
                self.calls.clear()
                def fail(command, **kwargs):
                    if (command[0] == "cargo") == (stage == "build"):
                        self.calls.append((command, kwargs["timeout"]))
                        raise OSError("injected launch failure")
                    return self.runner(command, **kwargs)
                self.assertEqual(self.run_probe(fail), 2)
                self.assertEqual(len(self.calls), 1 if stage == "build" else 2)
                terminal = json.loads((self.output / f"{stage}-terminal.json").read_bytes())
                self.assertEqual(terminal["error_type"], "OSError")
                self.assertTrue((self.output / "original/full-report.json").is_file())

    def test_missing_cargo_artifact_never_uses_a_stale_executable(self):
        with mock.patch.object(API, "owned_run", return_value=subprocess.CompletedProcess([], 0)) as run:
            self.assertEqual(API.supervise(self.failures, self.output), 2)
        self.assertEqual(run.call_count, 1)
        self.assertTrue(self.executable.is_file())

    def test_timeout_retains_primary_outcome_and_terminal_without_retry(self):
        def timeout(command, **kwargs):
            completed = self.runner(command, **kwargs)
            if command[0] != "cargo":
                raise subprocess.TimeoutExpired(command, 17)
            return completed
        self.assertEqual(self.run_probe(timeout), 2)
        self.assertEqual(len(self.calls), 2)
        self.assertTrue((self.output / "attempt/run-outcome.json").is_file())
        terminal = json.loads((self.output / "probe-terminal.json").read_bytes())
        self.assertEqual(terminal["status"], "timeout")
        self.assertEqual(json.loads((self.output / "result.json").read_bytes())["status"], "timeout")

    def test_nonzero_probe_exit_is_preserved(self):
        def failed(command, **kwargs):
            completed = self.runner(command, **kwargs)
            if command[0] != "cargo":
                completed.returncode = 1
            return completed
        self.assertEqual(self.run_probe(failed), 1)
        self.assertEqual(json.loads((self.output / "result.json").read_bytes())["probe_returncode"], 1)
        self.assertEqual(len(self.calls), 2)

    def test_readback_failure_preserves_completed_report_and_nonzero_exit(self):
        def failed_read(command, **kwargs):
            completed = self.runner(command, **kwargs)
            if command[0] != "cargo":
                attempt = Path(command[-1])
                path = attempt / "summary.json"
                value = json.loads(path.read_bytes())
                value.update(boundary_error="readback failed", boundary_equal_to_reference=None,
                             observed_boundary_sha256=None)
                path.write_bytes(encoded(value))
                (attempt / "observed-boundary.json").unlink()
                completed.returncode = 1
            return completed
        self.assertEqual(self.run_probe(failed_read), 1)
        result = json.loads((self.output / "result.json").read_bytes())
        self.assertEqual(result["status"], "probe_failed_with_evidence")
        self.assertEqual(result["observation"]["boundary"]["boundary_error"], "readback failed")
        self.assertIn("selected_case", result["observation"])
        self.assertEqual(len(self.calls), 2)

    def test_rust_study_error_with_valid_boundary_remains_a_failed_probe(self):
        def failed_study(command, **kwargs):
            completed = self.runner(command, **kwargs)
            if command[0] != "cargo":
                attempt = Path(command[-1])
                (attempt / "run-outcome.json").write_bytes(encoded({
                    "status": "error", "all_cases_completed": False,
                    "error": "injected Rust study error"}))
                path = attempt / "summary.json"
                value = json.loads(path.read_bytes())
                value["run_error"] = "injected Rust study error"
                path.write_bytes(encoded(value))
                completed.returncode = 1
            return completed
        self.assertEqual(self.run_probe(failed_study), 1)
        result = json.loads((self.output / "result.json").read_bytes())
        self.assertEqual(result["status"], "probe_failed_with_evidence")
        self.assertEqual(result["observation"]["boundary"]["run_error"], "injected Rust study error")
        self.assertNotIn("selected_case", result["observation"])
        self.assertEqual(len(self.calls), 2)

    def test_exit_zero_missing_or_inconsistent_completion_is_not_accepted(self):
        for change in ("remove", "false_completion", "foreign_request", "missing_boundary"):
            with self.subTest(change=change):
                self.output = self.root / change
                self.calls.clear()
                def invalid(command, **kwargs):
                    completed = self.runner(command, **kwargs)
                    if command[0] != "cargo":
                        path = Path(command[-1]) / "run-outcome.json"
                        if change == "missing_boundary":
                            (Path(command[-1]) / "observed-boundary.json").unlink()
                        elif change == "remove":
                            path.unlink()
                        else:
                            value = json.loads(path.read_bytes())
                            if change == "false_completion":
                                value["all_cases_completed"] = False
                            else:
                                value["report"]["request_sha256"] = "foreign"
                            path.write_bytes(encoded(value))
                    return completed
                self.assertEqual(self.run_probe(invalid), 2)
                self.assertEqual(len(self.calls), 2)

    def test_invalid_budget_or_evidence_does_not_launch(self):
        with self.assertRaises(ValueError):
            API.supervise(self.failures, self.output, True, 10)
        (self.folder / "request.json").write_bytes(b"corrupted")
        with mock.patch.object(API, "owned_run") as run:
            self.assertEqual(API.supervise(self.failures, self.output), 2)
        run.assert_not_called()


class WorkflowContract(unittest.TestCase):
    def test_diagnostic_is_after_successful_retention_and_cannot_clear_original_failure(self):
        workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml").read_text()
        gpu = workflow.split("  wgpu-macos:\n", 1)[1].split("\n  wasm-build:", 1)[0]
        original = gpu.index("- name: Public Python attention and byte-LM handles on a real GPU")
        retained = gpu.index("- name: Retain exact prepared-study replay failure diagnostics")
        observed = gpu.index("- name: Observe one retained byte-study first-update boundary")
        archived = gpu.index("- name: Retain separate byte-study boundary diagnostics")
        self.assertLess(original, retained)
        self.assertLess(retained, observed)
        self.assertLess(observed, archived)
        self.assertIn("id: byte_python", gpu[original:retained])
        self.assertIn("id: prepared_evidence", gpu[retained:observed])
        self.assertIn("if: failure()", gpu[retained:observed])
        step = gpu[observed:archived]
        self.assertIn("if: failure() && steps.byte_python.outcome == 'failure' "
                      "&& steps.prepared_evidence.outcome == 'success'", step)
        self.assertIn("timeout-minutes: 35", step)
        self.assertIn("--build-timeout-seconds 900 --probe-timeout-seconds 900", step)
        self.assertNotIn("continue-on-error", gpu[original:])
        self.assertNotIn("|| true", gpu[original:])
        self.assertEqual(gpu.count("python -I -B bindings/st-py/tests/test_nn_byte_study.py -v"), 1)
        self.assertIn("if: always() && steps.byte_boundary.outcome != 'skipped'", gpu[archived:])
        self.assertIn("retention-days: 1", gpu[archived:])
        self.assertIn("python3 -I -S -B tools/test_run_byte_study_boundary.py", workflow)


if __name__ == "__main__":
    unittest.main()
