#!/usr/bin/env python3
"""Offline by default; two explicit historical-prelude arms, never a retry."""
import argparse
import ast
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import struct
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import byte_study_prelude as prelude  # noqa: E402

retained = prelude.retained
boundary, replay = retained.boundary, retained.replay
PROTOCOL = "same_wheel_prelude_ordinary_then_retained.v1"
MAX_ARM_SECONDS = 5400


def harness_files():
    result = retained.harness_files()
    for path in (Path(__file__), Path(prelude.__file__)):
        result[path.name] = replay.digest(path)
    return result


def inference_source_hash(source):
    raw = boundary.read(source / prelude.TEST_PATH)
    if boundary.digest(raw) != prelude.TEST_SHA256:
        raise ValueError("historical test changed")
    matches = [node.value for node in ast.parse(raw).body if isinstance(node, ast.Assign)
               and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
               and node.targets[0].id == "FRESH_INFERENCE"]
    if len(matches) != 1:
        raise ValueError("fresh inference source is not unique")
    value = ast.literal_eval(matches[0])
    if not isinstance(value, str):
        raise ValueError("fresh inference must be a literal string")
    return boundary.digest(value.encode())


def require_wheel(wheel):
    if replay.digest(wheel) != prelude.WHEEL_SHA256:
        raise ValueError("requires the recovered hosted wheel; do not rebuild")
    with zipfile.ZipFile(wheel) as archive:
        natives = [name for name in archive.namelist() if name.endswith(".so")
                   and Path(name).name.startswith("spiraltorch.")]
        if len(natives) != 1 or boundary.digest(archive.read(natives[0])) != prelude.NATIVE_SHA256:
            raise ValueError("wheel native payload differs")


def preflight(failures, source, wheel, timeout):
    if type(timeout) is not int or not 1 <= timeout <= MAX_ARM_SECONDS:
        raise ValueError("each arm deadline must be in [1,5400] seconds")
    identity = replay.git_identity(source)
    if (identity["commit"] != retained.SOURCE_COMMIT or identity["tracked_changes"]
            or identity["untracked_files"]):
        raise ValueError("native source must be the fixed clean diagnostic revision")
    method, lines = prelude.instrumented_method(boundary.read(source / prelude.TEST_PATH))
    del method
    require_wheel(wheel)
    selected = retained.select_fixed(failures)
    manifest = {
        "schema": "spiraltorch.prelude_pair.manifest.v1", "protocol": PROTOCOL,
        "release_ready": False, "source_root": str(source), "source_checkout": identity,
        "harness_checkout": replay.git_identity(ROOT), "harness_files": harness_files(),
        "test_sha256": prelude.TEST_SHA256, "target_assignment_lines": lines,
        "fresh_inference_sha256": inference_source_hash(source),
        "wheel": {"filename": wheel.name, "sha256": prelude.WHEEL_SHA256,
                  "native_sha256": prelude.NATIVE_SHA256},
        "original_files": {name: boundary.digest(raw) for name, raw in selected["payloads"].items()},
        "request_sha256": retained.REQUEST_SHA256, "case": retained.CASE, "case_count": 25,
        "modes": list(prelude.MODES), "attempts_per_arm": 1, "timeout_seconds": timeout,
        "methods": list(prelude.METHODS), "python": sys.version,
        "runner_image": {key: os.environ.get(key) for key in ("ImageOS", "ImageVersion")},
        "expected_children": {"no_mismatch_observed": 4, "mismatch_observed": 2},
        "scope": "Fresh Python processes, shared hosted machine. Exact native wheel from run "
                 "40076826426; not the historical failing binary. Order/driver/retention effects "
                 "remain confounds. Common target recording changes host work.",
    }
    return manifest, selected


def claim_worker(output, wheel):
    manifest = boundary.decode(boundary.read(output / "manifest.json"))
    campaign_raw = boundary.read(output.parent / "manifest.json")
    campaign = boundary.decode(campaign_raw)
    mode = manifest.get("mode")
    if mode not in prelude.MODES or output.name != f"{prelude.MODES.index(mode) + 1:02d}-{mode}":
        raise ValueError("worker arm does not match its declared directory")
    expected = dict(campaign, mode=mode, campaign_sha256=boundary.digest(campaign_raw))
    if manifest != expected or campaign.get("protocol") != PROTOCOL:
        raise ValueError("worker manifest does not match its campaign")
    fresh, selected = preflight(output.parent / "original",
                                Path(manifest["source_root"]), wheel, manifest["timeout_seconds"])
    # Process-specific interpreter/runner strings are evidence, not an execution permission.
    for key in fresh:
        if key not in ("python", "runner_image") and fresh[key] != campaign.get(key):
            raise ValueError("preflight identity changed: " + key)
    boundary.save(output / "worker-start.json", {"pid": os.getpid(), "started_unix": time.time(),
                                                "manifest_sha256": replay.digest(output / "manifest.json")})
    return manifest, selected


def installed_package(wheel, package):
    """Bind the Python facade too, without importing its optional dependencies."""
    directory = Path(package.__file__).resolve().parent
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist()
                 if name.startswith("spiraltorch/") and name.endswith(".py")]
        if not names:
            raise ValueError("wheel has no Python facade")
        for name in names:
            relative = Path(name).relative_to("spiraltorch")
            if ".." in relative.parts or boundary.read(directory / relative) != archive.read(name):
                raise ValueError("installed Python facade differs from the admitted wheel")
    return len(names)


def matching_training(checkpoint, report, updates):
    """Direct f32 serialization and Value's f64 spelling must encode the same f32."""
    if not isinstance(checkpoint, list) or not isinstance(report, list):
        return False
    if len(checkpoint) != updates or len(report) != updates:
        return False
    for index, (left, right) in enumerate(zip(checkpoint, report), 1):
        for point in (left, right):
            if (not isinstance(point, dict) or set(point) != {"revision", "ce"}
                    or type(point["revision"]) is not int or point["revision"] != index
                    or type(point["ce"]) not in (int, float)):
                return False
        try:
            if not all(math.isfinite(point["ce"]) for point in (left, right)):
                return False
            packed = struct.pack("<f", right["ce"])
            if (struct.unpack("<f", packed)[0] != right["ce"]
                    or struct.pack("<f", left["ce"]) != packed):
                return False
        except (OverflowError, struct.error):
            return False
    return True


def worker(output, wheel):
    manifest, selected = claim_worker(output, wheel)
    record = {"schema": "spiraltorch.prelude_pair.worker.v1", "protocol": PROTOCOL,
              "mode": manifest["mode"], "release_ready": False, "outcome": "execution_error",
              "foreign_import_attempts": [], "native_execution_completed": False, "pid": os.getpid(),
              "harness_sha256": replay.digest(replay.__file__)}
    setup = replay.SetupSubprocesses(record)
    module = None
    code = 1
    started = time.monotonic()
    try:
        replay.require_foreign_free_environment()
        replay.install_import_guards(record["foreign_import_attempts"])
        sys.addaudithook(setup.audit)
        os.environ.update(SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS="1", SPIRALTORCH_STRICT_GPU="1")
        test_path = Path(manifest["source_root"]) / prelude.TEST_PATH
        spec = importlib.util.spec_from_file_location("_pinned_prelude_target", test_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        record["native_library"] = replay.native_identity(
            wheel, importlib.import_module("spiraltorch.spiraltorch"))
        if record["native_library"]["sha256"] != prelude.NATIVE_SHA256:
            raise ValueError("installed native differs from the recovered hosted binary")
        record["verified_python_files"] = installed_package(wheel, module.st)
        setup_count = len(record["foreign_import_attempts"])
        record["blocked_setup_import_attempts"] = list(record["foreign_import_attempts"])
        replay.require_native_runtime(record["foreign_import_attempts"], setup_count)
        if record["unexpected_child_spawns"]:
            raise ValueError("undeclared setup subprocess")
        retained.require_client_surface(module.st)
        setup.active = False
        children = replay.GuardedChildren(output, wheel, module.FRESH_INFERENCE,
                                         record["native_library"], record)
        sys.addaudithook(children.audit)
        original_subprocess = module.subprocess
        module.subprocess, children.active = children, True
        try:
            code = prelude.exercise(module, boundary.read(test_path), manifest["mode"],
                                    selected, output, record)
            replay.require_native_runtime(record["foreign_import_attempts"], setup_count)
            if record["outcome"] in manifest["expected_children"]:
                children.require_complete(manifest["expected_children"][record["outcome"]])
            elif record["unexpected_child_spawns"]:
                raise ValueError("undeclared diagnostic process")
        finally:
            module.subprocess, children.active = original_subprocess, False
    except Exception as error:
        record.update(outcome="execution_error", execution_error=retained.error_record(error))
        code = 1
    finally:
        setup.active = False
        record.update(returncode=code, seconds=time.monotonic() - started)
        boundary.save(output / "worker-result.json", record)
    if record["native_execution_completed"]:
        retained.save_raw(output / "runtime.json",
                          retained._runtime_payload(lambda: retained.runtime_identity(module.st)))
    else:
        boundary.save(output / "runtime.json", {
            "status": "not_collected",
            "reason": ("native_completion_unobserved" if record["native_execution_completed"] is None
                       else "no_completed_native_phase")})
    try:
        if retained.runtime_status(output, record) != "collected":
            return 1
    except (OSError, ValueError):
        return 1
    return code


def verify_target(output, mode, selected):
    target = output / "target"
    record = boundary.decode(boundary.read(target / "target.json"))
    if (record.get("schema") != "spiraltorch.prelude_target.v1" or record.get("mode") != mode
            or record.get("release_ready") is not False or record.get("stage") != "finished"
            or record.get("native_calls") != ["partial_1", "resume_2", mode + "_full_2"]
            or any(key in record for key in ("error", "boundary_error", "report_export_error"))
            or boundary.read(target / "request.json") != selected["payloads"]["request.json"]):
        raise ValueError("incomplete or inconsistent target evidence")
    first = boundary.read(target / "first-report.json")
    resumed = boundary.read(target / "resumed-report.json")
    full = boundary.read(target / "full-report.json")
    for raw, updates in ((first, 1), (resumed, 2), (full, 2)):
        retained.validate_report(raw, selected, updates)
    checkpoints = []
    for name, report_raw, updates in (("first", first, 1), ("resumed", resumed, 2)):
        checkpoint = boundary.decode(boundary.read(target / f"{name}-checkpoint.json"))
        report_value = boundary.decode(report_raw)
        if (checkpoint.get("schema") != "spiraltorch.byte_corpus.checkpoint.v9"
                or checkpoint.get("request_sha256") != retained.REQUEST_SHA256
                or type(checkpoint.get("completed_updates")) is not int
                or checkpoint["completed_updates"] != updates
                or [c["name"] for c in checkpoint["cases"]]
                != [c["name"] for c in selected["full"]["cases"]]
                or any(not matching_training(c["training"], r["training"], updates)
                       for c, r in zip(checkpoint["cases"], report_value["cases"]))):
            raise ValueError("same-attempt checkpoint and report disagree")
        checkpoints.append(checkpoint)
    report = record.get("report_comparison", {})
    if (report.get("exact_bytes_equal") is not (full == resumed)
            or report.get("full_sha256") != boundary.digest(full)
            or report.get("resumed_sha256") != boundary.digest(resumed)
            or report.get("differences") != retained.exact_report_differences(resumed, full, limit=64)):
        raise ValueError("saved report bytes do not support the target verdict")
    mismatch = full != resumed
    if mode == "retained":
        reference = boundary.read(target / "reference-boundary.json")
        observed = boundary.read(target / "observed-boundary.json")
        if checkpoints[0]["cases"][23]["model_json"].encode() != reference:
            raise ValueError("boundary reference is not this attempt's selected partial checkpoint")
        comparison = retained.compare_boundaries(reference, observed)
        if comparison != record.get("boundary_comparison"):
            raise ValueError("saved boundary bytes do not support the target verdict")
        outcome = boundary.decode(boundary.read(target / "run-outcome.json"))
        if (outcome.get("schema") != "spiraltorch.byte_retained_boundary.v1"
                or outcome.get("request_sha256") != retained.REQUEST_SHA256
                or outcome.get("case") != retained.CASE
                or type(outcome.get("requested_capture_after_update")) is not int
                or outcome["requested_capture_after_update"] != 1
                or outcome.get("capture_mode") != "retained"
                or outcome.get("boundary_read_policy") != "explicit_after_run_returns"
                or outcome.get("release_ready") is not False
                or outcome.get("status") != "completed" or outcome.get("all_cases_completed") is not True
                or outcome.get("boundary_captured") is not True
                or boundary.canonical_case(outcome.get("report"))
                != boundary.canonical_case(boundary.decode(full))):
            raise ValueError("retained outcome does not support completion")
        mismatch |= not comparison["exact_bytes_equal"]
    verdict = "mismatch_observed" if mismatch else "no_mismatch_observed"
    if record.get("outcome") != verdict or record.get("comparison_outcome") != verdict:
        raise ValueError("recomputed target verdict differs")
    return verdict


def verify_children(output, record, count):
    manifest = boundary.decode(boundary.read(output / "manifest.json"))
    children = record.get("child_runs", [])
    if len(children) != count:
        raise ValueError("wrong child count")
    for index, row in enumerate(children):
        directory = f"children/{index:04d}"
        if row.get("directory") != directory or row.get("valid") is not True:
            raise ValueError("child order or recorded qualification differs")
        path = output / directory
        parent = boundary.decode(boundary.read(path / "parent-result.json"))
        child = boundary.decode(boundary.read(path / "child-result.json"))
        if (parent != row or row.get("returncode") != 0 or row.get("spawn_observed") is not True
                or child.get("schema") != "spiraltorch.hosted_byte_replay.child.v1"
                or child.get("outcome") != "no_foreign_ml_observed"
                or child.get("source_sha256") != manifest["fresh_inference_sha256"]
                or child.get("harness_sha256") != manifest["harness_files"][Path(replay.__file__).name]
                or child.get("native_library") != record["native_library"]
                or child.get("release_ready") is not False
                or child.get("unexpected_child_spawns") != 0
                or not isinstance(child.get("foreign_import_attempts"), list)
                or child.get("foreign_import_attempts") != child.get("blocked_setup_import_attempts")
                or not isinstance(child.get("setup_subprocesses"), list)
                or any(tuple(command) not in replay.SETUP_COMMANDS for command in child["setup_subprocesses"])):
            raise ValueError("child evidence does not support qualification")


def qualify_arm(output, mode, selected, code):
    record = boundary.decode(boundary.read(output / "worker-result.json"))
    verdict = verify_target(output, mode, selected)
    expected = [{"method": name, "outcome": "ok"} for name in prelude.METHODS]
    if verdict == "mismatch_observed":
        expected[-1].update(outcome="error", exception_type="TargetMismatch")
    count = 4 if verdict == "no_mismatch_observed" else 2
    tests = record.get("tests", {})
    if (record.get("schema") != "spiraltorch.prelude_pair.worker.v1"
            or record.get("protocol") != PROTOCOL or record.get("mode") != mode
            or record.get("release_ready") is not False or record.get("outcome") != verdict
            or record.get("returncode") != (0 if verdict == "no_mismatch_observed" else 1)
            or type(record.get("returncode")) is not int
            or type(code) is not int or code != record["returncode"]
            or record.get("native_library", {}).get("sha256") != prelude.NATIVE_SHA256
            or type(record.get("verified_python_files")) is not int or record["verified_python_files"] < 1
            or not isinstance(record.get("setup_subprocesses"), list)
            or any(tuple(command) not in replay.SETUP_COMMANDS for command in record["setup_subprocesses"])
            or record.get("foreign_import_attempts") != record.get("blocked_setup_import_attempts")
            or not isinstance(record.get("foreign_import_attempts"), list)
            or record.get("unexpected_child_spawns") != 0
            or "execution_error" in record or tests.get("tests_run") != 6
            or tests.get("events") != expected or tests.get("failures") != []
            or len(tests.get("errors", [])) != (0 if count == 4 else 1)
            or any(tests.get(key) != [] for key in
                   ("skipped", "unexpected_skips", "expected_failures", "unexpected_successes"))
            or len(record.get("child_runs", [])) != count
            or any(child.get("valid") is not True for child in record["child_runs"])
            or retained.runtime_status(output, record) != "collected"):
        raise ValueError("worker history, supervision or runtime provenance is incomplete")
    verify_children(output, record, count)
    return verdict


def supervise(manifest, selected, output, wheel):
    if any(manifest["harness_checkout"][key] for key in ("tracked_changes", "untracked_files")):
        raise ValueError("commit the qualified harness before executing it")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    result = {"schema": "spiraltorch.prelude_pair.result.v1", "protocol": PROTOCOL,
              "release_ready": False, "outcome": "incomplete", "arms": []}
    code = 1
    try:
        original = output / "original" / "version-9"
        original.mkdir(parents=True)
        for name, raw in selected["payloads"].items():
            retained.save_raw(original / name, raw)
        boundary.save(output / "manifest.json", manifest)
        campaign_hash = replay.digest(output / "manifest.json")
        for index, mode in enumerate(prelude.MODES, 1):
            arm = output / f"{index:02d}-{mode}"
            arm.mkdir()
            boundary.save(arm / "manifest.json", dict(manifest, mode=mode, campaign_sha256=campaign_hash))
            row = {"mode": mode, "attempt": 1, "outcome": "incomplete"}
            result["arms"].append(row)
            command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                       "--worker", "--output", str(arm), "--wheel", str(wheel)]
            try:
                with (arm / "process.log").open("xb") as log:
                    process = replay.owned_run(command, timeout=manifest["timeout_seconds"], stdout=log, cwd=ROOT)
                row["worker_returncode"] = process.returncode
                row["outcome"] = qualify_arm(arm, mode, selected, process.returncode)
            finally:
                # A completed target remains evidence even if metadata hangs/fails.
                # This is deliberately separate from qualifying the whole arm.
                try:
                    row["target_comparison_outcome"] = verify_target(arm, mode, selected)
                    row["target_comparison_scope"] = "saved target bytes only; not full arm qualification"
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                    row["target_comparison_error"] = retained.error_record(error)
                boundary.save(arm / "result.json", row)
        result["outcome"] = ("mismatch_observed" if any(row["outcome"] == "mismatch_observed"
                                                       for row in result["arms"]) else "no_mismatch_observed")
        code = 0 if result["outcome"] == "no_mismatch_observed" else 1
    except subprocess.TimeoutExpired:
        result.update(outcome="timeout", supervision_outcome="timeout")
        code = 124
    except Exception as error:
        result.update(outcome="stopped", supervision_outcome="error", error=retained.error_record(error))
    finally:
        result["returncode"] = code
        result["comparison_outcome"] = (
            "mismatch_observed" if any(row["outcome"] == "mismatch_observed"
                                      or row.get("target_comparison_outcome") == "mismatch_observed"
                                      for row in result["arms"])
            else "no_mismatch_observed" if result["outcome"] == "no_mismatch_observed" else "incomplete")
        result["comparison_scope"] = "verified target comparison; supervision/provenance failures remain failures"
        result["files"] = {str(path.relative_to(output)): {"sha256": replay.digest(path),
                                                        "bytes": path.stat().st_size}
                           for path in sorted(output.rglob("*")) if path.is_file()}
        boundary.save(output / "result.json", result)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=MAX_ARM_SECONDS)
    parser.add_argument("--execute-pair", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    wheel = args.wheel.resolve(strict=True)
    if args.worker:
        if args.output is None or args.execute_pair or args.failure_root or args.source_root:
            parser.error("worker requires only its supervisor output and wheel")
        return worker(args.output.resolve(strict=True), wheel)
    if args.failure_root is None or args.source_root is None:
        parser.error("preflight requires --failure-root and --source-root")
    manifest, selected = preflight(args.failure_root.resolve(strict=True),
                                   args.source_root.resolve(strict=True), wheel, args.timeout_seconds)
    if not args.execute_pair:
        print(json.dumps({"status": "offline_preflight", "gpu_executed": False,
                          "release_ready": False, "manifest": manifest}, indent=2))
        return 0
    if args.output is None:
        parser.error("execution requires a fresh --output directory")
    return supervise(manifest, selected, args.output.resolve(), wheel)


if __name__ == "__main__":
    raise SystemExit(main())
