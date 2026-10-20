#!/usr/bin/env python3
"""Bounded, non-publishing replay of the original synthetic GPU study tests."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from owned_process import run as owned_run

SOURCE_COMMITS = (
    "7ad0c100e9b5beddb224182b728e648d376ac8b8",
    "073254cf5dce4074a930ff29aef6e2e48aa7fce3",
)
FOREIGN = ("torch", "numpy", "transformers", "peft", "accelerate", "safetensors")
DIAGNOSTICS = ("DETACHED", "RESUME", "PREPARED")
PREPARED_TEST = "test_preparation_chain_uses_one_shared_runner"
CPU_ONLY_TEST = "StudyPortable.test_cpu_only_execution_is_not_a_fallback"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def git_identity(root):
    values = subprocess.check_output(
        ["git", "rev-parse", "HEAD", "HEAD^{tree}"], cwd=root, text=True
    ).splitlines()
    changed = subprocess.check_output(["git", "diff", "--name-only", "-z", "HEAD", "--"], cwd=root)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root)
    return {"commit": values[0], "tree": values[1],
            "tracked_changes": changed.count(b"\0"), "untracked_files": untracked.count(b"\0")}


def native_identity(wheel, native):
    library = Path(native.__file__)
    with zipfile.ZipFile(wheel) as archive:
        entries = [name for name in archive.namelist()
                   if Path(name).name.startswith("spiraltorch.")
                   and name.endswith((".so", ".pyd"))]
        if len(entries) != 1:
            raise ValueError("expected exactly one SpiralTorch native extension in wheel")
        expected = hashlib.sha256(archive.read(entries[0])).hexdigest()
    actual = digest(library)
    if actual != expected:
        raise ValueError("installed native extension does not match the selected wheel")
    return {"filename": library.name, "sha256": actual}


def import_guard(attempts):
    def audit(event, args):
        if event == "import" and args and args[0].split(".")[0] in FOREIGN:
            attempts.append(args[0])
            raise RuntimeError("foreign ML import prohibited in native replay: " + args[0])
    return audit


def require_native_runtime(attempts, setup_count):
    if attempts[setup_count:] or any(name.split(".")[0] in FOREIGN for name in sys.modules):
        raise RuntimeError("foreign ML import observed during replay")


def phase_summary(result):
    unexpected_skips = [
        {"test": test.id(), "reason": reason}
        for test, reason in result.skipped if not test.id().endswith("." + CPU_ONLY_TEST)
    ]
    return {
        "tests_run": result.testsRun,
        "failures": [test.id() for test, _ in result.failures],
        "errors": [test.id() for test, _ in result.errors],
        "expected_failures": [test.id() for test, _ in result.expectedFailures],
        "unexpected_successes": [test.id() for test in result.unexpectedSuccesses],
        "skipped": [{"test": test.id(), "reason": reason} for test, reason in result.skipped],
        "unexpected_skips": unexpected_skips,
        "valid": (result.wasSuccessful() and result.testsRun > 0
                  and not unexpected_skips and not result.expectedFailures),
    }


def exercise(module, output, record):
    """No adapter probe, tensor snapshots, or checkpoint calls are added here."""
    phases = (
        ("00-original-study-suite", lambda: unittest.defaultTestLoader.loadTestsFromModule(module)),
        ("01-same-process-prepared", lambda: unittest.TestSuite([module.StudyGpu(PREPARED_TEST)])),
    )
    record["phases"] = []
    keys = ["SPIRALTORCH_" + kind + "_STUDY_DIAGNOSTICS" for kind in DIAGNOSTICS]
    prior = {key: os.environ.get(key) for key in keys}
    try:
        for name, suite_factory in phases:
            directory = output / name
            directory.mkdir()
            for kind, key in zip(DIAGNOSTICS, keys):
                os.environ[key] = str(directory / kind.lower())
            row = {"name": name, "status": "started"}
            record["phases"].append(row)
            suite = suite_factory()
            planned = suite.countTestCases()
            started = time.monotonic()
            with (directory / "tests.log").open("x", encoding="utf-8") as stream:
                # Preserve the original ordering up to the first failure. Never
                # continue a failed phase or retry it looking for a green result.
                result = unittest.TextTestRunner(stream=stream, verbosity=2, failfast=True).run(suite)
            row.update(phase_summary(result), planned_tests=planned,
                       seconds=time.monotonic() - started, status="completed")
            row["valid"] = row["valid"] and row["tests_run"] == planned
            save(directory / "result.json", row)
            print(json.dumps(row), flush=True)
            if not row["valid"]:
                record["outcome"] = "test_failure"
                return 1
        record["outcome"] = "no_mismatch_observed"
        return 0
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def finish_worker(output, record, module):
    # Persist the first numerical/execution outcome BEFORE device metadata.
    # A stalled or cancelled metadata postlude must not erase that outcome.
    save(output / "worker-result.json", record)
    if module is not None and hasattr(module, "st"):
        from byte_study_diagnostics import _runtime_payload, runtime_identity
        save(output / "runtime.json",
             json.loads(_runtime_payload(lambda: runtime_identity(module.st))))


def worker(output, wheel):
    record = {"schema": "spiraltorch.hosted_byte_replay.worker.v1",
              "release_ready": False, "outcome": "incomplete", "phases": [],
              "foreign_import_attempts": []}
    started = time.monotonic()
    module = None
    try:
        if any(name.split(".")[0] in FOREIGN for name in sys.modules):
            raise RuntimeError("foreign ML module already loaded")
        if any(importlib.util.find_spec(name) is not None for name in FOREIGN):
            raise RuntimeError("replay requires an environment without foreign ML packages")
        sys.addaudithook(import_guard(record["foreign_import_attempts"]))
        os.environ.update(SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS="1", SPIRALTORCH_STRICT_GPU="1")
        test_path = ROOT / "bindings/st-py/tests/test_nn_byte_study.py"
        spec = importlib.util.spec_from_file_location("_byte_study_replay_target", test_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        record["native_library"] = native_identity(
            wheel, importlib.import_module("spiraltorch.spiraltorch"))
        # The historical elliptic facade probes optional Torch at import time.
        # The audit hook blocks that probe before any Torch code can execute.
        # Preserve it separately; attempts during the actual tests still fail.
        setup_count = len(record["foreign_import_attempts"])
        record["blocked_setup_import_attempts"] = list(record["foreign_import_attempts"])
        require_native_runtime(record["foreign_import_attempts"], setup_count)
        code = exercise(module, output, record)
        require_native_runtime(record["foreign_import_attempts"], setup_count)
        return code
    except BaseException as error:
        if record["outcome"] != "test_failure":
            record["outcome"] = "execution_error"
        record["error_type"] = type(error).__name__
        raise
    finally:
        record["seconds"] = time.monotonic() - started
        finish_worker(output, record, module)


def first_outcome_after_timeout(output, record):
    record["outcome"] = "timeout"
    try:
        first = json.loads((output / "worker-result.json").read_text())
        record["worker_outcome"] = first["outcome"]
        if first["outcome"] in ("test_failure", "execution_error"):
            record["outcome"] = first["outcome"]
    except (OSError, ValueError, KeyError, TypeError):
        record["worker_record_status"] = "unavailable_or_incomplete"
    record["supervision_outcome"] = "timeout"


def supervise(output, source, wheel, timeout):
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    record = {"schema": "spiraltorch.hosted_byte_replay.v1", "release_ready": False,
              "outcome": "incomplete", "attempts": 1}
    started = time.monotonic()
    code = 1
    try:
        identity = git_identity(source)
        if identity["commit"] not in SOURCE_COMMITS:
            raise ValueError("source checkout is not one of the two declared revisions")
        manifest = {
            "schema": "spiraltorch.hosted_byte_replay.manifest.v1",
            "source_checkout": identity, "harness_checkout": git_identity(ROOT),
            "wheel": {"filename": wheel.name, "sha256": digest(wheel)},
            "harness_sha256": digest(__file__),
            "test_sha256": digest(ROOT / "bindings/st-py/tests/test_nn_byte_study.py"),
            "python": sys.version, "system": platform.system(), "release": platform.release(),
            "machine": platform.machine(),
            "runner_image": {key: os.environ.get(key) for key in ("ImageOS", "ImageVersion")},
            "rustup_toolchain": os.environ.get("RUSTUP_TOOLCHAIN"),
            "timeout_seconds": timeout, "release_ready": False,
            "protocol": ["original_study_suite_once", "same_process_prepared_test_once"],
            "stop_on_first_failure": True, "rust_ci_prelude_replayed": False,
            "historical_driver_or_gpu_identity_guaranteed": False,
        }
        save(output / "manifest.json", manifest)
        command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                   "--worker", "--output", str(output), "--wheel", str(wheel)]
        with (output / "process.log").open("x", encoding="utf-8") as stream:
            result = owned_run(command, timeout=timeout, stdout=stream, cwd=ROOT)
        code = result.returncode
        record["worker_returncode"] = code
        record["supervision_outcome"] = "completed"
        worker_result = json.loads((output / "worker-result.json").read_text())
        record["outcome"] = worker_result["outcome"]
        if code == 0 and worker_result["outcome"] != "no_mismatch_observed":
            raise RuntimeError("worker success does not match its retained result")
    except subprocess.TimeoutExpired:
        code = 124
        first_outcome_after_timeout(output, record)
    except BaseException as error:
        code = 1
        record.update(outcome="supervision_error", supervision_outcome="error",
                      error_type=type(error).__name__)
        raise
    finally:
        record["seconds"] = time.monotonic() - started
        record["returncode"] = code
        record["files"] = {
            str(path.relative_to(output)): {"sha256": digest(path), "bytes": path.stat().st_size}
            for path in sorted(output.rglob("*")) if path.is_file()
        }
        save(output / "result.json", record)
        print(json.dumps(record), flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.timeout_seconds <= 7200:
        parser.error("timeout must be in [1,7200] seconds")
    if not args.worker and args.source_root is None:
        parser.error("--source-root is required")
    output, wheel = args.output.resolve(), args.wheel.resolve(strict=True)
    if args.worker:
        if not (output / "manifest.json").is_file():
            parser.error("worker requires the supervisor's fresh manifest")
        return worker(output, wheel)
    return supervise(output, args.source_root.resolve(strict=True), wheel, args.timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
