#!/usr/bin/env python3
"""Artifact-bound byte acceptance. Default is CPU preflight, never a GPU run."""
import argparse
import ast
from collections import Counter
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import byte_release_artifacts as artifacts
import byte_release_contract as contract
import byte_release_evidence as evidence
import byte_release_long_resume as long_resume
import byte_release_schedule as schedule
import run_byte_study_replay as replay
from byte_study_diagnostics import runtime_identity


def load_admission(path, expected):
    value = contract.validate_admission(contract.read(path))
    if not contract.is_hash(expected) or contract.digest(value) != expected:
        raise ValueError("admission digest differs")
    return value


def preflight(path, expected, wheel):
    admission = load_admission(path, expected)
    if sys.platform != "darwin":
        raise ValueError("the admitted runtime environments are macOS Metal")
    if contract.source_identity(ROOT) != admission["source"]:
        raise ValueError("source tree or qualification tooling changed")
    if artifacts.wheel_inventory(wheel) != admission["wheels"]["macos"]:
        raise ValueError("selected wheel differs from the frozen macOS artifact")
    replay.require_foreign_free_environment()
    package = artifacts.installed_identity(admission["wheels"]["macos"])
    for kind in replay.DIAGNOSTICS:
        if os.environ.get("SPIRALTORCH_" + kind + "_STUDY_DIAGNOSTICS"):
            raise ValueError("normal-path qualification must not enable numerical diagnostics")
    return admission, package


def runtime_valid(value, native):
    adapter = value.get("adapter", {}) if type(value) is dict else {}
    return (type(value) is dict and value.get("schema") == "spiraltorch.synthetic_gpu_runtime.v1"
            and value.get("system") == "Darwin" and value.get("native_library") == native
            and type(adapter) is dict and adapter.get("backend") == "Metal"
            and adapter.get("device_type") not in (None, "", "Cpu")
            and type(adapter.get("name")) is str and bool(adapter["name"]))


def audit_valid(record):
    return (type(record.get("foreign_import_attempts")) is list
            and type(record.get("blocked_setup_import_attempts")) is list
            and record["foreign_import_attempts"] == record["blocked_setup_import_attempts"]
            and all(type(name) is str and name.split(".")[0] in replay.FOREIGN
                    for name in record["foreign_import_attempts"])
            and type(record.get("unexpected_child_spawns")) is int and record["unexpected_child_spawns"] == 0
            and type(record.get("setup_subprocesses")) is list
            and all(type(command) is list and tuple(command) in replay.SETUP_COMMANDS
                    for command in record["setup_subprocesses"]))


def child_worker(output, wheel, source, admission_path, admission_sha256):
    output = Path(output)
    record = {"schema": "spiraltorch.hosted_byte_replay.child.v1", "release_ready": False,
              "outcome": "execution_error", "pid": os.getpid(),
              "harness_sha256": artifacts.sha256(replay.__file__),
              "source_sha256": replay.hashlib.sha256(source.encode()).hexdigest(),
              "foreign_import_attempts": [], "unexpected_child_spawns": 0}
    setup = replay.SetupSubprocesses(record)
    try:
        admission, package = preflight(admission_path, admission_sha256, wheel)
        contract.save(output / "child-start.json", {"admission_sha256": admission_sha256, "pid": os.getpid()})
        replay.install_import_guards(record["foreign_import_attempts"])
        sys.addaudithook(setup.audit)
        st = importlib.import_module("spiraltorch")
        record["native_library"] = replay.native_identity(wheel, importlib.import_module("spiraltorch.spiraltorch"))
        setup_count = len(record["foreign_import_attempts"])
        record["blocked_setup_import_attempts"] = list(record["foreign_import_attempts"])
        replay.require_native_runtime(record["foreign_import_attempts"], setup_count)
        if record["unexpected_child_spawns"]:
            raise ValueError("undeclared child setup process")
        setup.active = False

        def reject_spawn(event, _args):
            if event in replay.PROCESS_EVENTS:
                record["unexpected_child_spawns"] += 1
                raise RuntimeError("undeclared nested qualification process")
        sys.addaudithook(reject_spawn)
        namespace = {"__name__": "__main__"}
        exec(compile(source, "<qualified-fresh-inference>", "exec"), namespace)
        after = artifacts.installed_identity(admission["wheels"]["macos"])
        runtime = runtime_identity(st)
        replay.require_native_runtime(record["foreign_import_attempts"], setup_count)
        if package != after or record["unexpected_child_spawns"] or not runtime_valid(runtime, record["native_library"]):
            raise ValueError("child identity or runtime postlude differs")
        record.update(outcome="no_foreign_ml_observed", qualification={
            "admission_sha256": admission_sha256, "package": after, "runtime": runtime,
        })
    except BaseException as error:
        record["error_type"] = type(error).__name__
        raise
    finally:
        setup.active = False
        contract.save(output / "child-result.json", record)


class QualifiedChildren(replay.GuardedChildren):
    def __init__(self, output, wheel, source, native, record, admission_path, admission_sha256, package):
        super().__init__(output, wheel, source, native, record)
        self.admission_path, self.admission_sha256, self.package = admission_path, admission_sha256, package

    def before_child(self, directory, input_text):
        evidence.raw_save(directory / "input.json", input_text)

    def child_bootstrap(self, directory):
        return (f"import runpy\nrunpy.run_path({str(Path(__file__).resolve())!r})['child_worker']("
                f"{str(directory)!r}, {str(self.wheel)!r}, {self.source!r}, "
                f"{str(self.admission_path)!r}, {self.admission_sha256!r})\n")

    def additional_child_valid(self, child):
        qualified = child.get("qualification", {})
        return (qualified.get("admission_sha256") == self.admission_sha256
                and qualified.get("package") == self.package
                and runtime_valid(qualified.get("runtime"), self.native))

    def run(self, command, **kwargs):
        result = super().run(command, **kwargs)
        if result.returncode != 0:
            return result
        directory = self.output / self.record["child_runs"][-1]["directory"]
        # Failed streams are retained by the base guard. Successful streams are
        # retained here before the original test compares its inference output.
        evidence.raw_save(directory / "stdout.log", result.stdout)
        evidence.raw_save(directory / "stderr.log", result.stderr)
        return result


def load_tests():
    modules = {}
    for path, _, _ in contract.TESTS:
        if path not in modules:
            spec = importlib.util.spec_from_file_location("_release_" + Path(path).stem, ROOT / path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules[path] = module
    names = unittest.defaultTestLoader.getTestCaseNames(modules[contract.STUDY].StudyGpu)
    if names != [method for path, _, method in contract.TESTS if path == contract.STUDY]:
        raise ValueError("the original six-test study order changed")
    return modules


def worker(output, wheel, admission_path, admission_sha256, round_index):
    output = Path(output)
    start = contract.read(output / "start.json")
    if (start.get("admission_sha256") != admission_sha256 or type(round_index) is not int
            or not 0 <= round_index < 3 or start.get("round") != round_index + 1):
        raise ValueError("worker lacks the exact supervisor round claim")
    contract.save(output / "worker-start.json", {"pid": os.getpid(), "admission_sha256": admission_sha256})
    record = {"schema": "spiraltorch.byte_release.worker.v1", "admission_sha256": admission_sha256,
              "round": round_index + 1, "outcome": "execution_error", "release_ready": False,
              "harness_sha256": artifacts.sha256(replay.__file__), "tests": [],
              "foreign_import_attempts": [], "unexpected_child_spawns": 0}
    setup, children, modules, native_complete = replay.SetupSubprocesses(record), None, {}, False
    started = time.monotonic()
    code = 1
    try:
        admission, record["package"] = preflight(admission_path, admission_sha256, wheel)
        replay.install_import_guards(record["foreign_import_attempts"])
        sys.addaudithook(setup.audit)
        os.environ.update(SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS="1", SPIRALTORCH_STRICT_GPU="1",
                          SPIRALTORCH_RUN_BYTE_RELEASE_QUALIFICATION="1")
        modules = load_tests()
        st = modules[contract.STUDY].st
        record["native_library"] = replay.native_identity(wheel, importlib.import_module("spiraltorch.spiraltorch"))
        setup_count = len(record["foreign_import_attempts"])
        record["blocked_setup_import_attempts"] = list(record["foreign_import_attempts"])
        replay.require_native_runtime(record["foreign_import_attempts"], setup_count)
        if record["unexpected_child_spawns"]:
            raise ValueError("undeclared setup process")
        setup.active = False
        children = QualifiedChildren(output, wheel, st_source := modules[contract.STUDY].FRESH_INFERENCE,
                                     record["native_library"], record, admission_path, admission_sha256, record["package"])
        record["inference_source_sha256"] = replay.hashlib.sha256(st_source.encode()).hexdigest()
        sys.addaudithook(children.audit)
        modules[contract.STUDY].subprocess, children.active = children, True
        for index, (path, class_name, method) in enumerate(contract.TESTS):
            directory = output / f"test-{index:02d}"
            directory.mkdir()
            if path == contract.LONG:
                os.environ["SPIRALTORCH_BYTE_RELEASE_OUTPUT"] = str(directory / "long")
            test = getattr(modules[path], class_name)(method)
            with evidence.recording(directory / "retained") as retained, (directory / "tests.log").open("x") as log:
                result = unittest.TextTestRunner(stream=log, verbosity=2, failfast=True).run(unittest.TestSuite([test]))
            row = replay.phase_summary(result)
            row.update(test=list(contract.TESTS[index]), directory=directory.name)
            row["valid"] = row["valid"] and row["tests_run"] == 1 and not row["skipped"]
            record["tests"].append(row)
            contract.save(directory / "result.json", row)
            native_complete |= row["valid"] or any(
                any(name in event["files"] for name in ("report.json", "checkpoint.json")) for event in retained.events)
            if not row["valid"]:
                record["outcome"] = "test_failure"
                break
        else:
            children.require_complete(4)
            replay.require_native_runtime(record["foreign_import_attempts"], setup_count)
            record["outcome"], code = "tests_passed", 0
    except BaseException as error:
        record["error_type"] = type(error).__name__
        if record["outcome"] != "test_failure":
            record["outcome"] = "execution_error"
    finally:
        setup.active = False
        contract.save(output / "numeric-result.json", record)
        try:
            if native_complete:
                runtime = runtime_identity(modules[contract.STUDY].st)
                contract.save(output / "runtime.json", runtime)
                if not runtime_valid(runtime, record["native_library"]):
                    raise ValueError("postlude did not identify the admitted Metal GPU")
            if code == 0:
                if artifacts.sha256(wheel) != admission["wheels"]["macos"]["sha256"]:
                    raise ValueError("wheel changed during the round")
                if artifacts.installed_identity(admission["wheels"]["macos"]) != record["package"]:
                    raise ValueError("installed package changed during the round")
                replay.require_native_runtime(record["foreign_import_attempts"], setup_count)
                if record["unexpected_child_spawns"] or not native_complete:
                    raise ValueError("worker audit is incomplete")
                record["outcome"] = "completed"
        except BaseException as error:
            record["postlude_error"] = type(error).__name__
            if code == 0:
                record["outcome"] = "postlude_failure"
            code = 1
        finally:
            if children is not None:
                children.active = False
        record.update(returncode=code, seconds=time.monotonic() - started)
        contract.save(output / "worker-result.json", record)
    return code


def verify_retained(directory, required):
    index = contract.read(directory / "index.json")
    if index.get("schema") != "spiraltorch.byte_release.retention.v1" or index.get("release_ready") is not False:
        raise ValueError("retained evidence index differs")
    counts = Counter()
    by_label = {}
    locations, files = set(), {"index.json"}
    total = 0
    if type(index.get("events")) is not list or len(index["events"]) > evidence.MAX_EVENTS:
        raise ValueError("retention event count exceeds the fixed bound")
    for event in index["events"]:
        if (type(event) is not dict or event.get("complete") is not True
                or type(event.get("files")) is not dict or not 1 <= len(event["files"]) <= 256
                or not contract.relative_file(event.get("directory"))
                or event["directory"] in locations):
            raise ValueError("incomplete retained event")
        locations.add(event["directory"])
        location = directory / event["directory"]
        if location.is_symlink() or not location.resolve().is_relative_to(directory.resolve()):
            raise ValueError("retention event escapes its test directory")
        counts[event["label"]] += 1
        by_label.setdefault(event["label"], []).append(location)
        for name, descriptor in event["files"].items():
            if not contract.relative_file(name) or "/" in name or type(descriptor) is not dict:
                raise ValueError("invalid retained filename or descriptor")
            path = location / name
            if (type(descriptor.get("bytes")) is not int or not 0 <= descriptor["bytes"] <= evidence.MAX_FILE
                    or path.is_symlink() or not path.resolve().is_relative_to(location.resolve())
                    or path.stat().st_size != descriptor["bytes"] or artifacts.sha256(path) != descriptor["sha256"]):
                raise ValueError("retained bytes differ from their index")
            total += descriptor["bytes"]
            files.add(path.relative_to(directory).as_posix())
        for suffix in ("bin", "json"):
            left, right = "expected." + suffix, "observed." + suffix
            if left in event["files"] and right in event["files"]:
                long_resume.require_exact((location / left).read_bytes(), (location / right).read_bytes(), event["label"])
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
    allowed = set(required) | ({"sgd_retained_gradient"} if "sgd_before" in required else set())
    if (total > evidence.MAX_TOTAL or total != index["bytes"] or actual != files
            or set(counts) - allowed or any(counts[label] != count for label, count in required.items())):
        raise ValueError("retained evidence coverage differs from the fixed matrix")
    return by_label


def verify_short(position, events):
    def equal(left, right, name="report.json"):
        long_resume.require_exact((left / name).read_bytes(), (right / name).read_bytes(), name)

    pairs = {0: ("adamw_full", "adamw_resumed"), 1: ("contextual_full", "contextual_resumed"),
             2: ("learning_full", "learning_resumed")}
    if position in pairs:
        left, right = pairs[position]
        for a, b in zip(events[left], events[right]):
            equal(a, b)
            equal(a, b, "checkpoint.json")
    if position == 0:
        equal(events["adamw_full"][0], events["adamw_ordinary"][0])
    if position == 1:
        request = contract.read(events["contextual_input"][0] / "request.json")
        if len(request.get("cases", [])) != 25:
            raise ValueError("the complete 25-case contextual study was not retained")
    if position == 3:
        expected = (events["detached_expected"][0] / "report.json").read_bytes()
        for event in events["detached_round"]:
            for index in range(2):
                long_resume.require_exact(expected, (event / f"report-{index}.json").read_bytes(), "detached report")
    if position == 4:
        for label in ("full_resumed", "full_complete_noop", "full_repeated"):
            equal(events["full_ordinary"][0], events[label][0])
    if position == 5:
        for a, b in zip(events["prepared_full"], events["prepared_resumed"]):
            equal(a, b)
    if position == 7:
        before = contract.read(events["sgd_before"][0] / "parameters.json")
        if len(events.get("sgd_retained_gradient", [])) != len(before):
            raise ValueError("retained raw gradient coverage differs")


def verify_long(directory):
    result = contract.read(directory / "result.json")
    expected = {"request.json", "ordinary-report.json", "full-report.json", "full-checkpoint.json"}
    expected.update(f"split-{cursor}/{name}.json" for cursor in (1, 64)
                    for name in ("first-report", "first-checkpoint", "resumed-report", "resumed-checkpoint"))
    if (result.get("schema") != long_resume.PROFILE or result.get("status") != "passed"
            or result.get("stage") != "complete" or result.get("updates") != 128
            or result.get("request_sha256") != long_resume.REQUEST_SHA256
            or result.get("splits") != [1, 64] or result.get("release_ready") is not False
            or type(result.get("files")) is not dict or set(result["files"]) != expected):
        raise ValueError("long study is incomplete")
    for name, entry in result["files"].items():
        path = directory / name
        if (type(entry.get("bytes")) is not int or not 0 <= entry["bytes"] <= evidence.MAX_FILE
                or path.is_symlink() or not path.resolve().is_relative_to(directory.resolve())
                or path.stat().st_size != entry["bytes"] or artifacts.sha256(path) != entry["sha256"]):
            raise ValueError("long study evidence changed")
    raw = (directory / "request.json").read_bytes()
    ordinary = (directory / "ordinary-report.json").read_bytes()
    long_resume.validate_completed_report(ordinary, raw)
    for path in ("full-report.json", "split-1/resumed-report.json", "split-64/resumed-report.json"):
        long_resume.require_exact(ordinary, (directory / path).read_bytes(), path)
    for path in ("split-1/resumed-checkpoint.json", "split-64/resumed-checkpoint.json"):
        long_resume.require_exact((directory / "full-checkpoint.json").read_bytes(), (directory / path).read_bytes(), path)


def verify_round(directory, index, admission, package):
    if contract.source_identity(ROOT) != admission["source"]:
        raise ValueError("source changed during execution")
    record = contract.read(directory / "worker-result.json")
    native = {"filename": Path(admission["wheels"]["macos"]["native"]["path"]).name,
              "sha256": admission["wheels"]["macos"]["native"]["sha256"]}
    if (record.get("schema") != "spiraltorch.byte_release.worker.v1" or record.get("outcome") != "completed"
            or record.get("admission_sha256") != contract.digest(admission) or record.get("round") != index + 1
            or record.get("returncode") != 0 or record.get("release_ready") is not False
            or record.get("package") != package or record.get("native_library") != native
            or record.get("harness_sha256") != artifacts.sha256(replay.__file__) or not audit_valid(record)
            or len(record.get("tests", [])) != len(contract.TESTS)
            or not runtime_valid(contract.read(directory / "runtime.json"), native)):
        raise ValueError("round completion or runtime identity is incomplete")
    for position, (row, expected, required) in enumerate(zip(record["tests"], contract.TESTS, contract.CAPTURES)):
        if (row.get("test") != list(expected) or row.get("directory") != f"test-{position:02d}"
                or row.get("valid") is not True or row.get("tests_run") != 1
                or any(row.get(key) for key in ("skipped", "failures", "errors", "expected_failures", "unexpected_successes"))):
            raise ValueError("test matrix coverage differs")
        location = directory / row["directory"]
        if contract.read(location / "result.json") != row:
            raise ValueError("test result changed")
        events = verify_retained(location / "retained", required)
        verify_short(position, events)
    verify_long(directory / "test-09/long")
    if len(record.get("child_runs", [])) != 4:
        raise ValueError("four fresh inference audits are required")
    tree = ast.parse((ROOT / contract.STUDY).read_text())
    source = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "FRESH_INFERENCE" for target in node.targets))
    if record.get("inference_source_sha256") != replay.hashlib.sha256(source.encode()).hexdigest():
        raise ValueError("fresh inference source differs from the frozen test")
    for index, row in enumerate(record["child_runs"]):
        if (row.get("directory") != f"children/{index:04d}" or row.get("valid") is not True
                or row.get("spawn_observed") is not True or row.get("returncode") != 0
                or row.get("child_outcome") != "no_foreign_ml_observed"):
            raise ValueError("child coverage differs")
        child_dir = directory / row["directory"]
        child = contract.read(child_dir / "child-result.json")
        qualified = child.get("qualification", {})
        if (contract.read(child_dir / "parent-result.json") != row
                or child.get("schema") != "spiraltorch.hosted_byte_replay.child.v1"
                or child.get("release_ready") is not False
                or child.get("harness_sha256") != artifacts.sha256(replay.__file__)
                or child.get("outcome") != "no_foreign_ml_observed"
                or child.get("source_sha256") != record["inference_source_sha256"]
                or child.get("native_library") != native or not audit_valid(child)
                or qualified.get("admission_sha256") != contract.digest(admission)
                or qualified.get("package") != package or not runtime_valid(qualified.get("runtime"), native)):
            raise ValueError("fresh inference identity/audit differs")
        contract.read(child_dir / "input.json")
        contract.read(child_dir / "stdout.log")
        if (child_dir / "stderr.log").stat().st_size != 0:
            raise ValueError("unexpected inference stderr")
    if any(path.is_symlink() for path in directory.rglob("*")):
        raise ValueError("round evidence contains a symbolic link")
    files = {path.relative_to(directory).as_posix(): artifacts.sha256(path)
             for path in sorted(directory.rglob("*")) if path.is_file()}
    contract.save(directory / "verified-files.json", files)
    return contract.digest(files)


def supervise(ledger, admission_path, expected, wheel, environment):
    admission, package = preflight(admission_path, expected, wheel)
    directory = contract.admit(ledger, admission)
    output = contract.claim_environment(directory, environment)
    admitted_path = directory / "admission.json"

    def command(index, location):
        return [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker",
                "--admission", str(admitted_path), "--admission-sha256", expected, "--wheel", str(wheel),
                "--output", str(location), "--round", str(index)]
    try:
        return schedule.run_three_rounds(output, expected, environment, command,
                                         lambda location, index: verify_round(location, index, admission, package))
    finally:
        if (output / "result.json").is_file():
            contract.finish_environment(directory, environment, contract.read(output / "result.json"))
        else:
            contract.finish_environment(directory, environment, {
                "schema": "spiraltorch.byte_release.environment_schedule.v1", "status": "failed",
                "admission_sha256": expected, "environment": environment, "release_ready": False,
            })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--environment", choices=schedule.ENVIRONMENTS)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--round", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    admission, wheel = args.admission.resolve(strict=True), args.wheel.resolve(strict=True)
    if args.worker:
        if args.output is None or args.round is None:
            parser.error("worker requires a claimed round")
        return worker(args.output.resolve(strict=True), wheel, admission, args.admission_sha256, args.round)
    if not args.execute:
        _, package = preflight(admission, args.admission_sha256, wheel)
        print(json.dumps({"preflight": "passed", "package": package, "release_ready": False}))
        return 0
    if args.ledger is None or args.environment is None:
        parser.error("execution requires the canonical ledger and named environment")
    result = supervise(args.ledger.resolve(strict=True), admission, args.admission_sha256, wheel, args.environment)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
