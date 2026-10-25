"""CPU-only orchestration controls; all numeric fixtures below are fabricated."""
import ast
from contextlib import ExitStack
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_byte_release_qualification as runner
from test_byte_release_contract import fabricated_admission
from test_byte_release_long_resume import FakeStudy

c, e = runner.contract, runner.evidence


def runtime(native):
    return dict(schema="spiraltorch.synthetic_gpu_runtime.v1", system="Darwin", native_library=native,
                adapter=dict(backend="Metal", device_type="IntegratedGpu", name="synthetic only"))


def inference_source():
    tree = ast.parse((runner.ROOT / c.STUDY).read_text())
    return next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "FRESH_INFERENCE" for target in node.targets))


def emit_short(index):
    for label, count in c.CAPTURES[index].items():
        for _ in range(count):
            files = {"report.json": b'{"synthetic":true}', "checkpoint.json": b'{"synthetic":true}'}
            if label == "contextual_input":
                files = {"request.json": c.encoded({"cases": list(range(25))})}
            if label == "detached_round":
                files = {f"report-{i}.json": b'{"synthetic":true}' for i in range(2)}
            if label == "sgd_before":
                files = {"parameters.json": b"[[1.0]]"}
            e.retain(label, lambda: files)
    if index == 7:
        e.retain("sgd_retained_gradient", lambda: {"expected.bin": b"raw", "observed.bin": b"raw"})


def short_evidence(index, directory):
    with e.recording(directory):
        emit_short(index)


def fake_children(directory, record, admission, package):
    audit = dict(foreign_import_attempts=[], blocked_setup_import_attempts=[],
                 unexpected_child_spawns=0, setup_subprocesses=[])
    for index in range(4):
        location = directory / "children" / f"{index:04d}"
        location.mkdir(parents=True)
        row = dict(directory=f"children/{index:04d}", valid=True, spawn_observed=True, returncode=0,
                   child_outcome="no_foreign_ml_observed")
        child = dict(schema="spiraltorch.hosted_byte_replay.child.v1", outcome="no_foreign_ml_observed",
                     source_sha256=record["inference_source_sha256"], harness_sha256=record["harness_sha256"],
                     release_ready=False, native_library=record["native_library"],
                     qualification=dict(admission_sha256=c.digest(admission), package=package,
                                        runtime=runtime(record["native_library"])), **audit)
        for name, value in (("parent-result.json", row), ("child-result.json", child),
                            ("input.json", {"synthetic": True}), ("stdout.log", {"synthetic": True})):
            c.save(location / name, value)
        (location / "stderr.log").write_bytes(b"")
        record["child_runs"].append(row)


def fake_round(directory, admission, package):
    """A structural fixture, never evidence of native execution."""
    directory.mkdir()
    native = {"filename": "spiraltorch.abi3.so", "sha256": "d" * 64}
    source_sha = runner.replay.hashlib.sha256(inference_source().encode()).hexdigest()
    audit = dict(foreign_import_attempts=[], blocked_setup_import_attempts=[],
                 unexpected_child_spawns=0, setup_subprocesses=[])
    record = dict(schema="spiraltorch.byte_release.worker.v1", outcome="completed", release_ready=False,
                  admission_sha256=c.digest(admission), round=1, returncode=0, package=package,
                  native_library=native, harness_sha256=runner.artifacts.sha256(runner.replay.__file__),
                  tests=[], child_runs=[], inference_source_sha256=source_sha, **audit)
    c.save(directory / "runtime.json", runtime(native))
    for index, spec in enumerate(c.TESTS):
        location = directory / f"test-{index:02d}"
        location.mkdir()
        row = dict(test=list(spec), directory=location.name, valid=True, tests_run=1,
                   skipped=[], failures=[], errors=[], expected_failures=[], unexpected_successes=[])
        short_evidence(index, location / "retained")
        c.save(location / "result.json", row)
        record["tests"].append(row)
    runner.long_resume.exercise(FakeStudy().client(), directory / "test-09/long")
    fake_children(directory, record, admission, package)
    c.save(directory / "worker-result.json", record)
    return record


class PreflightOffline(unittest.TestCase):
    def test_preflight_rejects_every_identity_drift_before_execution(self):
        admission = fabricated_admission()
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            path = Path(temporary) / "admission.json"
            c.save(path, admission)
            stack.enter_context(patch.object(runner.sys, "platform", "darwin"))
            source = stack.enter_context(patch.object(c, "source_identity", return_value=admission["source"]))
            wheel = stack.enter_context(patch.object(runner.artifacts, "wheel_inventory", return_value=admission["wheels"]["macos"]))
            foreign = stack.enter_context(patch.object(runner.replay, "require_foreign_free_environment"))
            package = stack.enter_context(patch.object(runner.artifacts, "installed_identity", return_value={"synthetic": True}))
            stack.enter_context(patch.dict(runner.os.environ, {}, clear=True))
            self.assertEqual(runner.preflight(path, c.digest(admission), "unused"), (admission, {"synthetic": True}))
            with self.assertRaisesRegex(ValueError, "digest"):
                runner.preflight(path, "0" * 64, "unused")
            source.return_value = {}
            with self.assertRaisesRegex(ValueError, "source tree"):
                runner.preflight(path, c.digest(admission), "unused")
            source.return_value = admission["source"]
            wheel.return_value = {}
            with self.assertRaisesRegex(ValueError, "selected wheel"):
                runner.preflight(path, c.digest(admission), "unused")
            wheel.return_value = admission["wheels"]["macos"]
            foreign.side_effect = RuntimeError("foreign installed")
            with self.assertRaisesRegex(RuntimeError, "foreign"):
                runner.preflight(path, c.digest(admission), "unused")
            foreign.side_effect = None
            package.side_effect = ValueError("installed bytes differ")
            with self.assertRaisesRegex(ValueError, "installed bytes"):
                runner.preflight(path, c.digest(admission), "unused")
            package.side_effect = None
            for kind in runner.replay.DIAGNOSTICS:
                with patch.dict(runner.os.environ, {"SPIRALTORCH_" + kind + "_STUDY_DIAGNOSTICS": "enabled"}):
                    with self.assertRaisesRegex(ValueError, "diagnostics"):
                        runner.preflight(path, c.digest(admission), "unused")
            with patch.object(runner.sys, "platform", "linux"), self.assertRaisesRegex(ValueError, "Metal"):
                runner.preflight(path, c.digest(admission), "unused")

    def test_default_cli_only_runs_cpu_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "file"
            path.write_bytes(b"synthetic")
            with patch.object(runner, "preflight", return_value=({}, {})) as check, \
                    patch.object(runner, "supervise") as run, patch("builtins.print"):
                self.assertEqual(runner.main(["--admission", str(path), "--admission-sha256", "a" * 64,
                                              "--wheel", str(path)]), 0)
                check.assert_called_once()
                run.assert_not_called()

    def test_fixed_tests_exist_without_importing_the_native_package(self):
        for path, name, method in c.TESTS:
            tree = ast.parse((runner.ROOT / path).read_text())
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
            self.assertIn(method, [node.name for node in cls.body if isinstance(node, ast.FunctionDef)])
        tree = ast.parse((runner.ROOT / c.STUDY).read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "StudyGpu")
        self.assertEqual(sorted(node.name for node in cls.body if isinstance(node, ast.FunctionDef)
                                and node.name.startswith("test_")), [row[2] for row in c.TESTS[:6]])


class EvidenceOffline(unittest.TestCase):
    def test_complete_fabricated_round_and_each_missing_audit_rejects(self):
        admission, package = fabricated_admission(), {"synthetic": True}
        with tempfile.TemporaryDirectory() as temporary, patch.object(c, "source_identity", return_value=admission["source"]):
            root = Path(temporary) / "round"
            record = fake_round(root, admission, package)
            self.assertTrue(c.is_hash(runner.verify_round(root, 0, admission, package)))
            mutations = (lambda r: r["tests"].pop(), lambda r: r["child_runs"].pop(),
                         lambda r: r.update(package={}), lambda r: r.update(admission_sha256="0" * 64),
                         lambda r: r.pop("foreign_import_attempts"), lambda r: r.pop("blocked_setup_import_attempts"),
                         lambda r: r.update(setup_subprocesses=[["sh", "-c", "echo hidden"]]),
                         lambda r: r.update(inference_source_sha256="0" * 64),
                         lambda r: r.update(harness_sha256="0" * 64),
                         lambda r: r["tests"][0].update(skipped=["not executed"]))
            for index, mutate in enumerate(mutations):
                changed = copy.deepcopy(record)
                mutate(changed)
                (root / "worker-result.json").write_bytes(c.encoded(changed))
                with self.subTest(index=index), self.assertRaises(ValueError):
                    runner.verify_round(root, 0, admission, package)
            (root / "worker-result.json").write_bytes(c.encoded(record))
            child_path = root / "children/0000/child-result.json"
            child = c.read(child_path)
            for key in ("schema", "release_ready", "harness_sha256", "setup_subprocesses", "qualification"):
                changed = copy.deepcopy(child)
                changed.pop(key)
                child_path.write_bytes(c.encoded(changed))
                with self.subTest(child_key=key), self.assertRaises(ValueError):
                    runner.verify_round(root, 0, admission, package)

    def test_retention_is_complete_exact_and_not_just_a_success_flag(self):
        for defect in ("changed", "missing", "duplicate", "extra", "pair", "incomplete", "coverage"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "raw"
                with e.recording(root):
                    e.retain("input", lambda: {"expected.bin": b"a", "observed.bin": b"a"})
                index = c.read(root / "index.json")
                event = index["events"][0]
                path = root / event["directory"] / "observed.bin"
                if defect == "changed":
                    path.write_bytes(b"b")
                elif defect == "missing":
                    path.unlink()
                elif defect == "duplicate":
                    index["events"].append(copy.deepcopy(event))
                elif defect == "extra":
                    (root / "unknown.json").write_bytes(b"{}")
                elif defect == "pair":
                    path.write_bytes(b"b")
                    event["files"]["observed.bin"]["sha256"] = runner.artifacts.sha256(path)
                elif defect == "incomplete":
                    event["complete"] = False
                elif defect == "coverage":
                    event["label"] = "wrong"
                (root / "index.json").write_bytes(c.encoded(index))
                with self.assertRaises((ValueError, AssertionError, FileNotFoundError)):
                    runner.verify_retained(root, {"input": 1})

    def test_long_verifier_requires_all_boundaries_and_original_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "long"
            runner.long_resume.exercise(FakeStudy().client(), root)
            runner.verify_long(root)
            original = c.read(root / "result.json")
            for mutate in (lambda r: r["files"].pop("split-1/first-checkpoint.json"),
                           lambda r: r.update(updates=127), lambda r: r.update(splits=[64, 1]),
                           lambda r: r.update(request_sha256="0" * 64)):
                bad = copy.deepcopy(original)
                mutate(bad)
                (root / "result.json").write_bytes(c.encoded(bad))
                with self.assertRaises(ValueError):
                    runner.verify_long(root)
            (root / "result.json").write_bytes(c.encoded(original))
            (root / "full-report.json").write_bytes(b"{}")
            with self.assertRaises(ValueError):
                runner.verify_long(root)

    def test_runtime_never_accepts_cpu_or_missing_adapter(self):
        native = {"synthetic": True}
        self.assertTrue(runner.runtime_valid(runtime(native), native))
        for adapter in (None, {}, {"backend": "Metal", "device_type": "Cpu", "name": "CPU"},
                        {"backend": "Vulkan", "device_type": "DiscreteGpu", "name": "other"}):
            value = runtime(native)
            value["adapter"] = adapter
            self.assertFalse(runner.runtime_valid(value, native))


class WorkerOffline(unittest.TestCase):
    def test_fresh_child_checks_package_and_keeps_postlude_failures(self):
        for changed in (False, True):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                root = Path(temporary)
                admission, package = fabricated_admission(), {"synthetic": True}
                native = {"filename": "spiraltorch.abi3.so", "sha256": "d" * 64}
                stack.enter_context(patch.object(runner, "preflight", return_value=(admission, package)))
                stack.enter_context(patch.object(runner.replay, "install_import_guards"))
                stack.enter_context(patch.object(runner.replay, "require_native_runtime"))
                stack.enter_context(patch.object(runner.sys, "addaudithook"))
                stack.enter_context(patch.object(runner.importlib, "import_module"))
                stack.enter_context(patch.object(runner.replay, "native_identity", return_value=native))
                stack.enter_context(patch.object(runner, "runtime_identity", return_value=runtime(native)))
                stack.enter_context(patch.object(runner.artifacts, "installed_identity", return_value={} if changed else package))
                if changed:
                    with self.assertRaisesRegex(ValueError, "postlude differs"):
                        runner.child_worker(root, "unused", "pass", "unused", c.digest(admission))
                else:
                    runner.child_worker(root, "unused", "pass", "unused", c.digest(admission))
                result = c.read(root / "child-result.json")
                self.assertEqual(result["outcome"], "execution_error" if changed else "no_foreign_ml_observed")
                self.assertFalse(result["release_ready"])
                self.assertEqual(result["source_sha256"], runner.replay.hashlib.sha256(b"pass").hexdigest())

    def test_failed_child_streams_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = {}
            children = runner.QualifiedChildren(root, "unused", "pass", {}, record, "unused", "a" * 64, {})
            directory = root / "children/0000"
            directory.mkdir()
            (directory / "stderr.log").write_bytes(b"original failure")
            record["child_runs"].append({"directory": "children/0000", "valid": False})
            failed = SimpleNamespace(returncode=1, stdout="", stderr="original failure")
            with patch.object(runner.replay.GuardedChildren, "run", return_value=failed):
                self.assertIs(children.run([]), failed)
            self.assertEqual((directory / "stderr.log").read_bytes(), b"original failure")

    def test_complete_worker_and_three_round_supervisor_with_synthetic_native(self):
        admission, package = fabricated_admission(), {"synthetic": True}
        native = {"filename": "spiraltorch.abi3.so", "sha256": "d" * 64}
        executed, modules = [], {}
        def method(index):
            def test(_self):
                executed.append(index)
                emit_short(index)
                if index == 5:
                    children = modules[c.STUDY].subprocess
                    fake_children(children.output, children.record, admission, package)
                if index == 9:
                    runner.long_resume.exercise(FakeStudy().client(), runner.os.environ["SPIRALTORCH_BYTE_RELEASE_OUTPUT"])
            return test
        for index, (path, class_name, name) in enumerate(c.TESTS):
            if path not in modules:
                modules[path] = SimpleNamespace(st=object(), FRESH_INFERENCE=inference_source(), subprocess=None)
                setattr(modules[path], class_name, type(class_name, (unittest.TestCase,), {}))
            setattr(getattr(modules[path], class_name), name, method(index))
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            ledger = root / "ledger"
            c.initialize_ledger(ledger)
            wheel = root / "wheel.whl"
            wheel.write_bytes(b"synthetic only")
            original_sha = runner.artifacts.sha256
            stack.enter_context(patch.object(runner.artifacts, "sha256", side_effect=lambda path:
                                            admission["wheels"]["macos"]["sha256"] if Path(path) == wheel else original_sha(path)))
            stack.enter_context(patch.object(runner, "preflight", return_value=(admission, package)))
            stack.enter_context(patch.object(runner, "load_tests", return_value=modules))
            stack.enter_context(patch.object(runner.replay, "install_import_guards"))
            stack.enter_context(patch.object(runner.replay, "require_native_runtime"))
            stack.enter_context(patch.object(runner.sys, "addaudithook"))
            stack.enter_context(patch.object(runner.importlib, "import_module"))
            stack.enter_context(patch.object(runner.replay, "native_identity", return_value=native))
            stack.enter_context(patch.object(runner, "runtime_identity", return_value=runtime(native)))
            stack.enter_context(patch.object(runner.artifacts, "installed_identity", return_value=package))
            stack.enter_context(patch.object(c, "source_identity", return_value=admission["source"]))
            stack.enter_context(patch.dict(runner.os.environ, {}, clear=True))
            def invoke(command, **_kwargs):
                args = dict(zip(command[5::2], command[6::2]))
                code = runner.worker(args["--output"], args["--wheel"], args["--admission"],
                                     args["--admission-sha256"], int(args["--round"]))
                return SimpleNamespace(returncode=code)
            owned = stack.enter_context(patch.object(runner.schedule, "owned_run", side_effect=invoke))
            result = runner.supervise(ledger, "unused", c.digest(admission), wheel, "local-metal")
            self.assertEqual(result["status"], "verified_rounds")
            self.assertEqual(owned.call_count, 3)
            self.assertEqual(executed, list(range(10)) * 3)
            self.assertFalse(result["release_ready"])
            self.assertEqual(c.read(ledger / "candidate-1/local-metal.result.json")["status"], "verified_rounds")

    def test_preflight_failure_never_imports_native_or_starts_tests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c.save(root / "start.json", {"admission_sha256": "a" * 64, "round": 1})
            with patch.object(runner, "preflight", side_effect=ValueError("synthetic preflight")), \
                    patch.object(runner, "load_tests") as load, patch.object(runner, "runtime_identity") as probe:
                self.assertEqual(runner.worker(root, "unused", "unused", "a" * 64, 0), 1)
            load.assert_not_called()
            probe.assert_not_called()
            self.assertEqual(c.read(root / "worker-result.json")["error_type"], "ValueError")
            self.assertEqual(c.read(root / "numeric-result.json")["tests"], [])

    def test_first_numeric_failure_keeps_raw_evidence_even_if_postlude_fails(self):
        executed = []
        class Synthetic(unittest.TestCase):
            def test_first(self):
                executed.append(1)
                e.retain("failed", lambda: {"report.json": b'{"synthetic_failure":true}'})
                self.fail("synthetic mismatch")
            def test_second(self):
                executed.append(2)
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            c.save(root / "start.json", {"admission_sha256": "a" * 64, "round": 1})
            module = SimpleNamespace(StudyGpu=Synthetic, st=object(), FRESH_INFERENCE="pass", subprocess=None)
            stack.enter_context(patch.object(c, "TESTS", ((c.STUDY, "StudyGpu", "test_first"),
                                                         (c.STUDY, "StudyGpu", "test_second"))))
            stack.enter_context(patch.object(runner, "preflight", return_value=(fabricated_admission(), {})))
            stack.enter_context(patch.object(runner, "load_tests", return_value={c.STUDY: module}))
            stack.enter_context(patch.object(runner.replay, "install_import_guards"))
            stack.enter_context(patch.object(runner.replay, "require_native_runtime"))
            stack.enter_context(patch.object(runner.sys, "addaudithook"))
            stack.enter_context(patch.object(runner.importlib, "import_module"))
            stack.enter_context(patch.object(runner.replay, "native_identity", return_value={}))
            stack.enter_context(patch.object(runner, "runtime_identity", side_effect=ValueError("synthetic postlude")))
            stack.enter_context(patch.dict(runner.os.environ, {}, clear=True))
            self.assertEqual(runner.worker(root, "unused", "unused", "a" * 64, 0), 1)
            self.assertEqual(executed, [1])
            numeric, final = c.read(root / "numeric-result.json"), c.read(root / "worker-result.json")
            self.assertEqual(numeric["outcome"], "test_failure")
            self.assertEqual(final["outcome"], "test_failure")
            self.assertEqual(final["postlude_error"], "ValueError")
            self.assertTrue((root / "test-00/retained/index.json").is_file())
            self.assertFalse((root / "test-01").exists())

    def test_supervisor_claim_and_failure_are_retained_without_retry(self):
        admission = fabricated_admission()
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "ledger"
            c.initialize_ledger(ledger)
            with patch.object(runner, "preflight", return_value=(admission, {})), \
                    patch.object(runner.schedule, "run_three_rounds", side_effect=RuntimeError("synthetic failure")) as run:
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    runner.supervise(ledger, "unused", c.digest(admission), "unused", "local-metal")
                with self.assertRaises(FileExistsError):
                    runner.supervise(ledger, "unused", c.digest(admission), "unused", "local-metal")
                self.assertEqual(run.call_count, 1)
            self.assertEqual(c.read(ledger / "candidate-1/local-metal.result.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
