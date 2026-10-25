#!/usr/bin/env python3
"""One post-failure, instrumented boundary attempt; never release qualification."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from byte_study_diagnostics import exact_report_differences
from owned_process import run as owned_run

LIMIT = 64 * 1024 * 1024
FILES = ("request.json", "full-report.json", "resumed-report.json",
         "first-report.json", "first-checkpoint.json", "resumed-checkpoint.json")
VERSIONS = tuple(range(4, 10))
EXAMPLE = "resident_byte_boundary_probe"


def read(path, limit=LIMIT):
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("expected a bounded regular file")
        stream = os.fdopen(descriptor, "rb")
        descriptor = None
        with stream:
            data = stream.read(limit + 1)
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(data) > limit:
        raise ValueError("file grew beyond its bound")
    return data


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON member")
            result[key] = value
        return result

    def invalid(_value):
        raise ValueError("nonfinite JSON number")

    def finite(raw_number):
        number = float(raw_number)
        if not math.isfinite(number):
            raise ValueError("nonfinite JSON number")
        return number
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical_case(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def select(directory):
    """Select the first retained version, then first differing case in request order."""
    if directory.is_symlink():
        raise ValueError("failure root must be a directory, not a symlink")
    if not directory.exists():
        return None, "no_prepared_failure"
    if not directory.is_dir():
        raise ValueError("failure root must be a directory")
    entries = list(directory.iterdir())
    allowed = {f"version-{version}" for version in VERSIONS}
    if any(path.name not in allowed or path.is_symlink() or not path.is_dir()
           for path in entries):
        raise ValueError("unexpected prepared failure entry")
    if not entries:
        return None, "no_prepared_failure"
    source = min(entries, key=lambda path: int(path.name.removeprefix("version-")))
    version = int(source.name.removeprefix("version-"))
    manifest_bytes = read(source / "summary.json", 64 * 1024)
    manifest = decode(manifest_bytes)
    if (manifest.get("schema") != "spiraltorch.synthetic_run_resume_diagnostic.v1"
            or manifest.get("comparison") != "exact_report_bytes_no_tolerance"
            or not isinstance(manifest.get("files"), dict)):
        raise ValueError("expected original uninterrupted/resumed failure manifest")
    names = set(manifest["files"])
    if names not in (set(FILES), set(FILES) | {"runtime.json"}):
        raise ValueError("failure manifest has missing or unexpected payloads")
    payloads = {}
    for name in sorted(names):
        raw = read(source / name)
        if manifest["files"][name] != "sha256:" + digest(raw):
            raise ValueError(f"original failure digest differs: {name}")
        payloads[name] = raw
    payloads["summary.json"] = manifest_bytes
    request = decode(payloads["request.json"])
    first = decode(payloads["first-checkpoint.json"])
    full = decode(payloads["full-report.json"])
    resumed = decode(payloads["resumed-report.json"])
    identity = digest(payloads["request.json"])
    if (request.get("schema") != f"spiraltorch.byte_corpus.request.v{version}"
            or first.get("request_sha256") != identity
            or type(first.get("completed_updates")) is not int
            or first["completed_updates"] != 1
            or len(request.get("train_batches", [])) != 2):
        raise ValueError("requires the original two-update request and update-1 checkpoint")
    cases = request.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 32:
        raise ValueError("expected 1..32 cases")
    case_names = [case["name"] for case in cases]
    if any(type(name) is not str or not name for name in case_names) \
            or len(set(case_names)) != len(case_names):
        raise ValueError("case identities must be unique nonempty strings")
    if [case["name"] for case in first["cases"]] != case_names:
        raise ValueError("checkpoint case identities differ")
    for report in (full, resumed):
        if (report.get("request_sha256") != identity
                or report.get("schema") != f"spiraltorch.byte_corpus.result.v{version}"
                or [case["name"] for case in report["cases"]] != case_names):
            raise ValueError("report identity, schema or case order differs")
    for index, (left, right) in enumerate(zip(full["cases"], resumed["cases"])):
        if canonical_case(left) != canonical_case(right):
            return {"version": version, "case": case_names[index], "case_index": index,
                    "request_sha256": identity, "payloads": payloads,
                    "full": full, "resumed": resumed}, None
    return None, "no_canonical_case_difference_in_first_failure"


def source_identity(output):
    for stage, command in (("source-git", ["git", "rev-parse", "HEAD"]),
                           ("source-rustc", ["rustc", "-vV"])):
        if command_once(command, output, stage, 10) != 0:
            raise ValueError(f"source metadata command failed: {stage}")
    return {"checkout_commit": read(output / "source-git.log", 64 * 1024).decode().strip(),
            "harness_sha256": digest(read(Path(__file__))),
            "rustc": read(output / "source-rustc.log", 64 * 1024).decode().strip(),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "github_sha": os.environ.get("GITHUB_SHA")}


def command_once(command, output, stage, timeout):
    terminal = {"command": command, "timeout_seconds": timeout}
    started = time.monotonic()
    try:
        with (output / f"{stage}.log").open("xb") as log:
            terminal["returncode"] = owned_run(command, cwd=ROOT, timeout=timeout,
                                               stdout=log).returncode
    except BaseException as error:
        terminal["error_type"] = type(error).__name__
        terminal["status"] = "timeout" if isinstance(error, subprocess.TimeoutExpired) else "error"
        raise
    finally:
        terminal["elapsed_seconds"] = time.monotonic() - started
        save(output / f"{stage}-terminal.json", terminal)
    return terminal["returncode"]


def built_executable(output):
    artifacts = []
    for line in read(output / "build.log").splitlines():
        try:
            message = decode(line)
        except (ValueError, UnicodeError):
            continue
        if not isinstance(message, dict):
            continue
        target = message.get("target", {})
        if (message.get("reason") == "compiler-artifact" and target.get("name") == EXAMPLE
                and "example" in target.get("kind", []) and message.get("executable")):
            artifacts.append(Path(message["executable"]))
    if len(artifacts) != 1 or not artifacts[0].is_absolute():
        raise ValueError("expected one Cargo-reported diagnostic executable")
    executable = artifacts[0]
    if not executable.resolve().is_relative_to((ROOT / "target").resolve()):
        raise ValueError("Cargo executable is outside the declared target directory")
    return executable


def analyze(attempt, selected, executable_hash, returncode):
    outcome = decode(read(attempt / "run-outcome.json"))
    summary = decode(read(attempt / "summary.json", 64 * 1024))
    invocation = decode(read(attempt / "invocation.json", 64 * 1024))
    if (summary.get("schema") != "spiraltorch.byte_boundary_probe.v1"
            or summary.get("release_ready") is not False
            or invocation.get("schema") != "spiraltorch.byte_boundary_invocation.v1"
            or invocation.get("request_sha256") != selected["request_sha256"]
            or invocation.get("case") != selected["case"]
            or type(invocation.get("requested_capture_after_update")) is not int
            or invocation["requested_capture_after_update"] != 1
            or invocation.get("all_cases_requested") is not True
            or invocation.get("boundary_read_policy") != "after_run_returns"
            or invocation.get("executable_sha256") != executable_hash):
        raise ValueError("diagnostic completion identity differs")
    if (read(attempt / "request.json") != selected["payloads"]["request.json"]
            or read(attempt / "reference-first-checkpoint.json")
            != selected["payloads"]["first-checkpoint.json"]):
        raise ValueError("probe inputs differ from retained failure")
    reference = read(attempt / "reference-boundary.json")
    if summary.get("reference_boundary_sha256") != digest(reference):
        raise ValueError("reference boundary digest differs")
    if summary.get("boundary_error") is None:
        observed_boundary = read(attempt / "observed-boundary.json")
        if (summary.get("observed_boundary_sha256") != digest(observed_boundary)
                or summary.get("boundary_equal_to_reference") is not (observed_boundary == reference)):
            raise ValueError("observed boundary or exact comparison differs")
    elif (type(summary["boundary_error"]) is not str
          or summary.get("boundary_equal_to_reference") is not None):
        raise ValueError("inconsistent boundary read failure")
    result = {"probe_returncode": returncode, "boundary": summary}
    if outcome.get("status") == "completed" and outcome.get("all_cases_completed") is True:
        if summary.get("run_error") is not None:
            raise ValueError("completed report carries a study error")
        if not isinstance(decode(read(attempt / "runtime.json", 64 * 1024)).get("adapter"), str):
            raise ValueError("missing completed-run adapter identity")
        report = outcome["report"]
        if (report.get("request_sha256") != selected["request_sha256"]
                or report.get("schema") != selected["full"]["schema"]
                or [c["name"] for c in report["cases"]]
                != [c["name"] for c in selected["full"]["cases"]]):
            raise ValueError("observed report identity or case order differs")
        index = selected["case_index"]
        observed = canonical_case(report["cases"][index])
        result["selected_case"] = {
            label: exact_report_differences(
                canonical_case(selected[key]["cases"][index]), observed)
            for label, key in (("vs_original_failed_full", "full"), ("vs_original_resumed", "resumed"))
        }
    elif (outcome.get("status") != "error" or outcome.get("all_cases_completed") is not False
          or not isinstance(outcome.get("error"), str)
          or summary.get("run_error") != outcome["error"]):
        raise ValueError("missing or inconsistent study outcome")
    if returncode == 0 and (outcome["status"] != "completed"
                           or summary.get("boundary_equal_to_reference") is not True
                           or summary.get("run_error") is not None
                           or summary.get("boundary_error") is not None):
        raise ValueError("exit zero without completed equal-boundary evidence")
    return result


def supervise(failures, output, build_timeout=900, probe_timeout=900):
    for value in (build_timeout, probe_timeout):
        if type(value) is not int or not 1 <= value <= 1800:
            raise ValueError("deadlines must be integers in [1,1800]")
    output.mkdir(parents=True, exist_ok=False)
    record = {"schema": "spiraltorch.hosted_boundary_attempt.v1", "release_ready": False,
              "attempt_limit": 1,
              "scope": "Separate instrumented process after preserved failure; not original-process "
                       "reproduction, root-cause proof, a retry-to-green or release qualification."}
    try:
        selected, reason = select(failures)
        if selected is None:
            record.update(status="not_selected", reason=reason)
            return 0
        original = output / "original"
        original.mkdir()
        for name, raw in selected["payloads"].items():
            with (original / name).open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        selection = {key: selected[key] for key in ("version", "case", "case_index", "request_sha256")}
        selection["source"] = source_identity(output)
        selection["original_sha256"] = {name: digest(raw)
                                        for name, raw in selected["payloads"].items()}
        save(output / "selection.json", selection)
        record["selection"] = selection
        build = ["cargo", "build", "--locked", "--release", "-p", "st-nn",
                 "--no-default-features", "--features", "wgpu", "--example", EXAMPLE,
                 "--target-dir", str(ROOT / "target"), "--message-format=json-render-diagnostics"]
        code = command_once(build, output, "build", build_timeout)
        if code != 0:
            record.update(status="build_failed", build_returncode=code)
            return 1
        executable = built_executable(output)
        # Only the hash is retained, never the executable or wheel.
        executable_hash = digest(read(executable, 512 * 1024 * 1024))
        record["executable_sha256"] = executable_hash
        save(output / "probe-start.json", {"selection": selection, "executable_sha256": executable_hash,
                                         "timeout_seconds": probe_timeout, "release_ready": False})
        command = [str(executable), str(original / "request.json"),
                   str(original / "first-checkpoint.json"), selected["case"], str(output / "attempt")]
        code = command_once(command, output, "probe", probe_timeout)
        record["probe_returncode"] = code
        if digest(read(executable, 512 * 1024 * 1024)) != executable_hash:
            raise ValueError("executable changed during the attempt")
        record["observation"] = analyze(output / "attempt", selected, executable_hash, code)
        record["status"] = "observed" if code == 0 else "probe_failed_with_evidence"
        return 0 if code == 0 else 1
    except Exception as error:
        record.update(status="timeout" if isinstance(error, subprocess.TimeoutExpired) else "error",
                      error_type=type(error).__name__, error=str(error))
        return 2
    except BaseException as error:
        record.update(status="interrupted", error_type=type(error).__name__)
        raise
    finally:
        save(output / "result.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--build-timeout-seconds", type=int, default=900)
    parser.add_argument("--probe-timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    return supervise(args.failures.absolute(), args.output.absolute(),
                     args.build_timeout_seconds, args.probe_timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
