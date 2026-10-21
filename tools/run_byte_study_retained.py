#!/usr/bin/env python3
"""One explicit same-process boundary diagnostic; default is offline preflight."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
# Isolated entry points explicitly opt in to sibling tooling, not site packages.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_byte_study_boundary as boundary  # noqa: E402
import run_byte_study_replay as replay  # noqa: E402
from byte_study_diagnostics import exact_report_differences, _runtime_payload, runtime_identity  # noqa: E402
from owned_process import run as owned_run  # noqa: E402

SOURCE_COMMIT = "e053c314d54be06b346e30e5ea1a065d7f3fdc90"
REQUEST_SHA256 = "6711703a49361d1d2d577912dbe86d47d4d42122273fee797fb27fd6abf8c52c"
CASE = "mixture-11-3-train-query-train"
PROTOCOL = "partial_1_then_resume_2_then_retained_full_2.v1"


def harness_files():
    return {path.name: replay.digest(path) for path in (
        Path(__file__), Path(boundary.__file__), Path(replay.__file__),
        Path(__file__).with_name("byte_study_diagnostics.py"),
        Path(__file__).with_name("owned_process.py"))}


def save_raw(path, raw):
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if type(raw) is not bytes or len(raw) > boundary.LIMIT:
        raise ValueError("expected bounded exact native bytes")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return raw


def error_record(error):
    return {"type": type(error).__name__, "message": str(error)[:2048]}


def select_fixed(failures):
    selected, reason = boundary.select(failures)
    if (reason is not None or selected is None or selected["version"] != 9
            or selected["request_sha256"] != REQUEST_SHA256
            or selected["case"] != CASE or selected["case_index"] != 23
            or len(selected["full"]["cases"]) != 25):
        raise ValueError("requires the fixed original v9 failure with all 25 cases")
    return selected


def validate_report(raw, selected, updates):
    if type(updates) is not int or updates not in (1, 2):
        raise ValueError("expected the partial-1 or complete-2 report")
    value = boundary.decode(raw)
    kind = "partial" if updates == 1 else "result"
    if (value.get("schema") != f"spiraltorch.byte_corpus.{kind}.v9"
            or value.get("request_sha256") != selected["request_sha256"]
            or [case["name"] for case in value.get("cases", [])]
            != [case["name"] for case in selected["full"]["cases"]]):
        raise ValueError("report identity or complete case order differs")
    for case in value["cases"]:
        points = case.get("training", [])
        if (len(points) != updates
                or any(type(p.get("revision")) is not int or p["revision"] != i + 1
                       or type(p.get("ce")) not in (int, float)
                       for i, p in enumerate(points))):
            raise ValueError("report training cursor differs")
    return value


def compare_boundaries(reference, observed):
    left, right = boundary.decode(reference), boundary.decode(observed)
    for value in (left, right):
        if (value.get("schema") != "spiraltorch.nn.byte_decoder_checkpoint.v6"
                or value.get("attempted_revision") != "1"
                or not isinstance(value.get("model"), dict)
                or not isinstance(value.get("optimizer"), dict)):
            raise ValueError("boundary must describe the complete AdamW attempted update 1")
    return {
        "exact_bytes_equal": reference == observed,
        "differences": exact_report_differences(reference, observed, limit=64),
        "decoded_components_equal": {
            name: boundary.canonical_case(left.get(name)) == boundary.canonical_case(right.get(name))
            for name in ("model", "optimizer", "attempted_revision")
        },
        "reference_sha256": boundary.digest(reference),
        "observed_sha256": boundary.digest(observed),
    }


def exercise(study, selected, output, record):
    """Only native APIs learn. Persist the run before any diagnostic readback."""
    if study.total_updates != 2 or study.request_sha256 != selected["request_sha256"]:
        raise ValueError("native study identity differs")
    study.validate_segment(1)
    study.validate_segment()
    record["stage"] = "partial"
    record["native_calls"] = ["partial_1"]
    first = study.advance_wgpu(1)
    first_report = save_raw(output / "first-report.json", first.report_json)
    first_bytes = save_raw(output / "first-checkpoint.json", first.checkpoint.to_json())
    validate_report(first_report, selected, 1)
    loaded = study.checkpoint_from_json(first_bytes)
    record["stage"] = "resume"
    record["native_calls"].append("resume_2")
    resumed = study.advance_wgpu(resume=loaded)
    resumed_report = save_raw(output / "resumed-report.json", resumed.report_json)
    save_raw(output / "resumed-checkpoint.json", resumed.checkpoint.to_json())
    resumed_value = validate_report(resumed_report, selected, 2)

    record["stage"] = "retained_full"
    record["native_calls"].append("retained_full_2")
    observed = study.run_wgpu_with_retained_boundary(loaded, selected["case"])
    outcome_raw = save_raw(output / "run-outcome.json", observed.run_outcome_json)
    full_report = None
    try:
        report = observed.report_json
        if report is not None:
            full_report = save_raw(output / "full-report.json", report)
    except Exception as error:
        record["report_export_error"] = error_record(error)
    record["stage"] = "boundary_read"
    captured = None
    try:
        captured = save_raw(output / "observed-boundary.json", observed.read_boundary().to_json())
    except Exception as error:
        record["boundary_error"] = error_record(error)

    record["stage"] = "comparison"
    outcome = boundary.decode(outcome_raw)
    if (outcome.get("schema") != "spiraltorch.byte_retained_boundary.v1"
            or outcome.get("request_sha256") != selected["request_sha256"]
            or outcome.get("case") != selected["case"]
            or type(outcome.get("requested_capture_after_update")) is not int
            or outcome["requested_capture_after_update"] != 1
            or outcome.get("capture_mode") != "retained"
            or outcome.get("boundary_read_policy") != "explicit_after_run_returns"
            or type(outcome.get("boundary_captured")) is not bool
            or outcome.get("release_ready") is not False):
        raise ValueError("retained outcome identity or capture policy differs")
    completed = outcome.get("status") == "completed"
    if completed:
        if (outcome.get("all_cases_completed") is not True
                or outcome.get("boundary_captured") is not True or full_report is None):
            raise ValueError("completed outcome lacks report or captured boundary")
        parsed = validate_report(full_report, selected, 2)
        if boundary.canonical_case(parsed) != boundary.canonical_case(outcome.get("report")):
            raise ValueError("exact report export disagrees with the retained outcome")
        record["report_comparison"] = {
            "exact_bytes_equal": full_report == resumed_report,
            "differences": exact_report_differences(resumed_report, full_report, limit=64),
            "full_sha256": boundary.digest(full_report),
            "resumed_sha256": boundary.digest(resumed_report),
        }
        index = selected["case_index"]
        record["selected_case_training"] = {
            "case": selected["case"], "case_index": index,
            "full": parsed["cases"][index]["training"],
            "resumed": resumed_value["cases"][index]["training"],
            "scope": "Selected case only; an aggregate report mismatch may concern other cases.",
        }
    elif (outcome.get("status") != "error" or outcome.get("all_cases_completed") is not False
          or not isinstance(outcome.get("error"), str) or not outcome["error"]
          or "report" in outcome or full_report is not None):
        raise ValueError("inconsistent failed study outcome")
    if captured is not None and not outcome["boundary_captured"]:
        raise ValueError("boundary read succeeded without a retained boundary")

    # Extraction is deliberately after the full run, never a GPU restoration.
    reference = save_raw(output / "reference-boundary.json",
                         study.case_checkpoint(loaded, selected["case"]).to_json())
    if first_bytes != loaded.to_json().encode("utf-8"):
        raise ValueError("same-attempt reference changed")
    if captured is not None:
        record["boundary_comparison"] = compare_boundaries(reference, captured)
    record["study_status"] = outcome["status"]
    mismatch = ((completed and not record["report_comparison"]["exact_bytes_equal"])
                or (captured is not None and not record["boundary_comparison"]["exact_bytes_equal"]))
    record["outcome"] = ("study_error" if not completed else "mismatch_observed" if mismatch
                         else "boundary_read_error" if captured is None else "no_mismatch_observed")
    record["comparison_outcome"] = record["outcome"]
    record["stage"] = "finished"
    return 0 if record["outcome"] == "no_mismatch_observed" else 1


def claim_worker(output):
    manifest = boundary.decode(boundary.read(output / "manifest.json"))
    if (manifest.get("schema") != "spiraltorch.retained_study.manifest.v1"
            or manifest.get("harness_sha256") != replay.digest(__file__)
            or manifest.get("harness_files") != harness_files()
            or manifest.get("protocol") != PROTOCOL
            or manifest.get("request_sha256") != REQUEST_SHA256
            or manifest.get("case") != CASE or manifest.get("case_count") != 25
            or manifest.get("release_ready") is not False
            or manifest.get("source_checkout", {}).get("commit") != SOURCE_COMMIT):
        raise ValueError("worker manifest identity differs")
    # Claim before import or GPU acquisition, including direct --worker calls.
    boundary.save(output / "worker-start.json", {"pid": os.getpid(), "started_unix": time.time(),
                                                "manifest_sha256": replay.digest(output / "manifest.json")})
    return manifest


def require_client_surface(st):
    observation = getattr(st.nn, "ByteCorpusBoundaryObservation", None)
    if (observation is None or not callable(getattr(observation, "read_boundary", None))
            or not hasattr(observation, "run_outcome_json") or not hasattr(observation, "report_json")
            or not callable(getattr(st.nn.ByteCorpusStudy, "run_wgpu_with_retained_boundary", None))):
        raise ValueError("selected wheel lacks the retained-boundary exact-report client")


def verify_success_files(output, record):
    for group, names in (
        ("report_comparison", (("full-report.json", "full_sha256"),
                               ("resumed-report.json", "resumed_sha256"))),
        ("boundary_comparison", (("reference-boundary.json", "reference_sha256"),
                                 ("observed-boundary.json", "observed_sha256"))),
    ):
        contents = [boundary.read(output / name) for name, _ in names]
        if (contents[0] != contents[1]
                or any(boundary.digest(raw) != record[group].get(key)
                       for raw, (_, key) in zip(contents, names))):
            raise ValueError("saved comparison bytes disagree with successful worker record")


def worker(output, wheel):
    manifest = claim_worker(output)
    record = {"schema": "spiraltorch.retained_study.worker.v1", "protocol": PROTOCOL,
              "release_ready": False, "outcome": "execution_error", "stage": "setup",
              "foreign_import_attempts": [], "pid": os.getpid()}
    setup = replay.SetupSubprocesses(record)
    st = None
    started = time.monotonic()
    code = 1
    try:
        selected = select_fixed(output / "original")
        replay.require_foreign_free_environment()
        replay.install_import_guards(record["foreign_import_attempts"])
        sys.addaudithook(setup.audit)
        os.environ.update(SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS="1", SPIRALTORCH_STRICT_GPU="1")
        st = importlib.import_module("spiraltorch")
        record["native_library"] = replay.native_identity(
            wheel, importlib.import_module("spiraltorch.spiraltorch"))
        if replay.digest(wheel) != manifest["wheel"]["sha256"]:
            raise ValueError("wheel changed after preflight")
        setup_count = len(record["foreign_import_attempts"])
        record["blocked_setup_import_attempts"] = list(record["foreign_import_attempts"])
        replay.require_native_runtime(record["foreign_import_attempts"], setup_count)
        if record["unexpected_child_spawns"]:
            raise ValueError("undeclared setup subprocess")
        setup.active = False

        def reject_spawn(event, _args):
            if event in replay.PROCESS_EVENTS:
                record["unexpected_child_spawns"] += 1
                raise RuntimeError("no subprocess is permitted during the diagnostic")

        sys.addaudithook(reject_spawn)
        require_client_surface(st)
        study = st.nn.ByteCorpusStudy.from_json(selected["payloads"]["request.json"])
        code = exercise(study, selected, output, record)
        replay.require_native_runtime(record["foreign_import_attempts"], setup_count)
        if record["unexpected_child_spawns"]:
            raise ValueError("undeclared diagnostic subprocess")
    except Exception as error:
        record["execution_error"] = error_record(error)
        record["outcome"] = "execution_error"
        code = 1
    finally:
        setup.active = False
        record["returncode"] = code
        record["seconds"] = time.monotonic() - started
        boundary.save(output / "worker-result.json", record)
    # Identity collection is a postlude, never a prerequisite for saving failure.
    if st is not None:
        save_raw(output / "runtime.json", _runtime_payload(lambda: runtime_identity(st)))
    return code


def supervise(selected, output, source, wheel, timeout):
    if type(timeout) is not int or not 1 <= timeout <= 7200:
        raise ValueError("deadline must be an integer in [1,7200]")
    identity = replay.git_identity(source)
    if (identity["commit"] != SOURCE_COMMIT or identity["tracked_changes"]
            or identity["untracked_files"]):
        raise ValueError("native source must be the fixed clean merged revision")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    record = {"schema": "spiraltorch.retained_study.supervisor.v1", "release_ready": False,
              "attempt_limit": 1, "outcome": "supervision_error"}
    code = 1
    try:
        original = output / "original" / "version-9"
        original.mkdir(parents=True)
        for name, raw in selected["payloads"].items():
            save_raw(original / name, raw)
        manifest = {"schema": "spiraltorch.retained_study.manifest.v1", "protocol": PROTOCOL,
                    "source_checkout": identity, "harness_checkout": replay.git_identity(ROOT),
                    "harness_sha256": replay.digest(__file__), "release_ready": False,
                    "harness_files": harness_files(),
                    "request_sha256": selected["request_sha256"], "case": selected["case"],
                    "case_count": 25, "all_cases_in_original_order": True,
                    "wheel": {"filename": wheel.name, "sha256": replay.digest(wheel)},
                    "timeout_seconds": timeout, "python": sys.version,
                    "runner_image": {key: os.environ.get(key) for key in ("ImageOS", "ImageVersion")},
                    "warmup": "none", "historical_process_prelude_replayed": False,
                    "historical_binary_or_device_identity_guaranteed": False,
                    "source_to_wheel_provenance": "must retain the pinned build log separately",
                    "scope": "Instrumented new process. Same-attempt partial is the primary "
                             "reference; historical checkpoints are not substituted."}
        boundary.save(output / "manifest.json", manifest)
        command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker",
                   "--output", str(output), "--wheel", str(wheel)]
        with (output / "process.log").open("xb") as log:
            result = owned_run(command, timeout=timeout, stdout=log, cwd=ROOT)
        record.update(supervision_outcome="completed", worker_returncode=result.returncode)
        first = boundary.decode(boundary.read(output / "worker-result.json"))
        record["worker_outcome"] = first.get("outcome")
        valid = (first.get("schema") == "spiraltorch.retained_study.worker.v1"
                 and first.get("protocol") == PROTOCOL and first.get("release_ready") is False
                 and type(first.get("returncode")) is int and first["returncode"] == result.returncode)
        if result.returncode == 0:
            valid = (valid and first.get("outcome") == "no_mismatch_observed"
                     and first.get("stage") == "finished" and first.get("study_status") == "completed"
                     and first.get("native_calls") == ["partial_1", "resume_2", "retained_full_2"]
                     and first.get("report_comparison", {}).get("exact_bytes_equal") is True
                     and first.get("boundary_comparison", {}).get("exact_bytes_equal") is True
                     and first.get("unexpected_child_spawns") == 0
                     and not any(key in first for key in
                                 ("execution_error", "report_export_error", "boundary_error")))
            if valid:
                verify_success_files(output, first)
        code = 0 if valid and result.returncode == 0 else 1
        record["outcome"] = first.get("outcome") if valid else "invalid_worker_record"
    except subprocess.TimeoutExpired:
        code = 124
        record.update(outcome="timeout", supervision_outcome="timeout")
        try:
            first = boundary.decode(boundary.read(output / "worker-result.json"))
            record["worker_outcome"] = first.get("outcome")
        except (OSError, ValueError):
            record["worker_record_status"] = "unavailable_or_incomplete"
    except Exception as error:
        record.update(supervision_outcome="error", error=error_record(error))
        code = 1
    finally:
        record["returncode"] = code
        record["files"] = {
            str(path.relative_to(output)): {"sha256": replay.digest(path), "bytes": path.stat().st_size}
            for path in sorted(output.rglob("*")) if path.is_file()
        }
        boundary.save(output / "result.json", record)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--execute-once", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.output is None or args.wheel is None or args.execute_once:
            parser.error("worker requires its supervisor output and wheel")
        return worker(args.output.resolve(), args.wheel.resolve(strict=True))
    if args.failure_root is None:
        parser.error("--failure-root is required")
    selected = select_fixed(args.failure_root)
    if not args.execute_once:
        print(json.dumps({"status": "offline_preflight", "release_ready": False,
                          "request_sha256": selected["request_sha256"], "case": selected["case"],
                          "case_count": 25, "gpu_executed": False, "native_runtime_validated": False}))
        return 0
    if any(value is None for value in (args.output, args.wheel, args.source_root)):
        parser.error("execution requires --output, --wheel and --source-root")
    return supervise(selected, args.output.resolve(), args.source_root.resolve(strict=True),
                     args.wheel.resolve(strict=True), args.timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
