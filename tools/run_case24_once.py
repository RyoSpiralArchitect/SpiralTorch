#!/usr/bin/env python3
"""One bounded, source-built macOS case-24 diagnostic; never retries a stage."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import re
import subprocess
import sys
import time

SOURCE_PARENT = "e08b8a527f64826a499ce129879a13fa0e6f3b33"
REQUEST_SHA = "5fbcf5e1f4737932a2a84111103635cf333c95fa7529975cd77acb56788d3dd1"
CHECKPOINT_SHA = "8aa9a429457eea03ba23c65e2c094afb228de2919cfbfcf05e26447343232d19"
CASE = "mixture-11-3-train-query-frozen"
BUDGET_SECONDS = 5100
EXAMPLE = "resident_case24_diagnostic"


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def admit(request, checkpoint):
    if sha(request) != REQUEST_SHA or sha(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("retained input SHA256 mismatch; no build or GPU execution admitted")
    payload = json.loads(request.read_text())
    if (len(payload["cases"]) != 25 or payload["cases"][24]["name"] != CASE
            or len(payload["train_batches"]) != 2):
        raise ValueError("only retained case 24 and two total updates are authorized")


def run_stage(name, command, root, output, env, deadline, clock=time.monotonic):
    receipt = {"stage": name, "command": [str(x) for x in command], "status": "pending"}
    path = output / (name + ".json")
    save(path, receipt)
    child = None
    try:
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError("job-wide deadline exhausted before spawn")
        receipt.update(status="running", remaining_seconds_at_spawn=remaining)
        save(path, receipt)
        with (output / (name + ".stdout.log")).open("wb") as stdout, (output / (name + ".stderr.log")).open("wb") as stderr:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=stdout, stderr=stderr,
                                     start_new_session=True)
            receipt["pid"] = child.pid
            save(path, receipt)
            code = child.wait(timeout=max(0, deadline - clock()))
        if clock() >= deadline:
            raise TimeoutError("job-wide deadline reached during stage")
        receipt.update(returncode=code, status="passed" if code == 0 else "failed")
        if code:
            raise RuntimeError(f"{name} exited {code}; no retry")
    except BaseException as exc:
        if child is not None:
            # Kill the session group even if the leader exited but left descendants.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        receipt.update(status="timeout" if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)) else "failed",
                       error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["remaining_seconds_at_finish"] = deadline - clock()
        save(path, receipt)
    return receipt


def execute(root, request, checkpoint, output, job_start_epoch, *,
            stage_runner=run_stage, admission=admit, wall=time.time, clock=time.monotonic):
    # Exclusive directory is the attempt lock, including failed or timed-out attempts.
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "spiraltorch.case24.diagnostic_attempt.v1", "case": CASE,
               "case_index": 24, "total_updates": 2, "binary_invocations": 0,
               "job_start_epoch": job_start_epoch, "budget_seconds": BUDGET_SECONDS,
               "source_parent": SOURCE_PARENT, "status": "admitting",
               "historical_reference": {
                   "failed_run_id": 41297797362,
                   "failed_artifact_id": 13085278754,
                   "failed_wheel_native_sha256": "44dc3ae1ef75104108517d6030ed374ee6d6610fd19501cc111b4985f3e5aeee",
               },
               "caveats": ["Rebuilt instrumented Rust source, not the historical Python wheel.",
                           "One binary contains continuous and historical-resume trajectories.",
                           "No retry, tolerance relaxation, broad sweep, or performance claim."]}
    save(output / "attempt.json", receipt)
    try:
        now = wall()
        if not math.isfinite(job_start_epoch) or job_start_epoch > now:
            raise ValueError("job start must be a finite non-future epoch")
        deadline = clock() + job_start_epoch + BUDGET_SECONDS - now
        if deadline <= clock():
            raise TimeoutError("job-wide deadline already exhausted")
        # Preserve exactly the admitted bytes and execute only those copies.
        retained = output / "inputs"
        retained.mkdir()
        copied_request, copied_checkpoint = retained / "request.json", retained / "first-checkpoint.json"
        shutil.copyfile(request, copied_request)
        shutil.copyfile(checkpoint, copied_checkpoint)
        request, checkpoint = copied_request, copied_checkpoint
        admission(request, checkpoint)
        receipt["inputs"] = {"request_sha256": sha(request), "first_checkpoint_sha256": sha(checkpoint)}
        env = os.environ.copy()
        # Fresh dedicated target directory: this cannot select an installed Python wheel.
        env["CARGO_TARGET_DIR"] = str(output / "target")
        for name, args in (("source_head", ["rev-parse", "HEAD"]),
                           ("source_tree", ["rev-parse", "HEAD^{tree}"]),
                           ("source_patch", ["diff", "--binary", SOURCE_PARENT, "HEAD"]),
                           ("source_diff", ["diff", "--binary", "HEAD"]),
                           ("source_status", ["status", "--porcelain=v1"]),
                           ("source_parent_present", ["merge-base", "--is-ancestor", SOURCE_PARENT, "HEAD"])):
            stage_runner(name, ["git", *args], root, output, env, deadline)
        stage_runner("rustc_version", ["rustc", "--version", "--verbose"], root, output, env, deadline)
        stage_runner("cargo_version", ["cargo", "--version", "--verbose"], root, output, env, deadline)
        cargo = ["cargo", "test", "--locked", "-p", "st-nn", "--no-default-features",
                 "--lib", "case24_diagnostic"]
        stage_runner("admission_test", cargo, root, output, env, deadline)
        admission_log = (output / "admission_test.stdout.log").read_text()
        if not re.search(r"test result: ok\. 2 passed; 0 failed; 0 ignored;", admission_log):
            raise RuntimeError("model-free admission must report exactly 2 passing tests, not an empty filter")
        stage_runner("build", ["cargo", "build", "--locked", "-p", "st-nn", "--no-default-features",
                               "--features", "wgpu", "--example", EXAMPLE], root, output, env, deadline)
        binary = output / "target" / "debug" / "examples" / EXAMPLE
        receipt["binary_sha256"] = sha(binary)
        receipt["binary_invocations"] = 1  # Reserve the sole attempt before spawn.
        receipt["status"] = "diagnostic_reserved"
        save(output / "attempt.json", receipt)
        stage_runner("diagnostic", [str(binary), str(request), str(checkpoint), str(output / "diagnostic")],
                     root, output, env, deadline)
        receipt["status"] = "completed"
        return 0
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        save(output / "attempt.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--first-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-start-epoch", type=float, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f"supervisor received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    root = Path(__file__).resolve().parents[1]
    return execute(root, args.request.resolve(), args.first_checkpoint.resolve(),
                   args.output.resolve(), args.job_start_epoch)


if __name__ == "__main__":
    sys.exit(main())
