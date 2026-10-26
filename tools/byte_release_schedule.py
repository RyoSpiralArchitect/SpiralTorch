"""Three-round ownership/deadline primitive, not candidate admission authority.

The release coordinator must bind the command factory and evidence verifier to
the frozen candidate before calling this function. There is deliberately no CLI
or generic success-by-exit-code mode. No native package is imported here.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from owned_process import run as owned_run

ROUNDS = 3
ENVIRONMENT_BUDGET_SECONDS = {"local-metal": 90 * 60, "hosted-metal": 4 * 60 * 60}
ENVIRONMENTS = tuple(ENVIRONMENT_BUDGET_SECONDS)


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def remaining(started, budget_seconds):
    available = budget_seconds - (time.monotonic() - started)
    if available <= 0:
        raise TimeoutError("shared environment deadline exhausted")
    return available


def run_three_rounds(output, admission_sha256, environment, command_factory, verify_round):
    """Sequential fresh processes, one decreasing budget, no continuation retry.

    `verify_round` must independently inspect the saved numerical/child/identity
    evidence. Its return is an evidence digest, not a truthy success flag. The
    enclosing candidate ledger must prevent a renamed output directory from
    becoming a new attempt; this lower-level primitive does not admit candidates.
    """
    if (type(admission_sha256) is not str or not re.fullmatch("[0-9a-f]{64}", admission_sha256)
            or environment not in ENVIRONMENTS):
        raise ValueError("a frozen admission digest and declared environment are required")
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    budget_seconds = ENVIRONMENT_BUDGET_SECONDS[environment]
    started = time.monotonic()
    record = {"schema": "spiraltorch.byte_release.environment_schedule.v1",
              "admission_sha256": admission_sha256, "environment": environment,
              "budget_seconds": budget_seconds, "planned_rounds": ROUNDS,
              "rounds": [], "status": "incomplete", "release_ready": False}
    try:
        save(output / "start.json", record)
        for index in range(ROUNDS):
            available = remaining(started, budget_seconds)
            directory = output / f"round-{index + 1}"
            directory.mkdir()
            command = command_factory(index, directory)
            if (type(command) is not list or len(command) < 4
                    or command[:3] != [sys.executable, "-I", "-B"]
                    or any(type(arg) is not str for arg in command)):
                raise ValueError("worker must be an explicit isolated fresh Python command")
            row = {"round": index + 1, "status": "started", "budget_remaining_seconds": available,
                   "command_sha256": hashlib.sha256(json.dumps(command).encode()).hexdigest()}
            record["rounds"].append(row)
            save(directory / "start.json", dict(row, admission_sha256=admission_sha256))
            with (directory / "process.log").open("x", encoding="utf-8") as stream:
                completed = owned_run(command, timeout=remaining(started, budget_seconds), stdout=stream)
            row["returncode"] = completed.returncode
            if completed.returncode != 0:
                row["status"] = "process_failed"
                raise RuntimeError("round failed; later rounds must not run")
            remaining(started, budget_seconds)
            evidence = verify_round(directory, index)
            if type(evidence) is not str or not re.fullmatch("[0-9a-f]{64}", evidence):
                raise ValueError("round lacks a verified evidence digest")
            remaining(started, budget_seconds)
            row.update(status="verified", evidence_sha256=evidence)
            save(directory / "result.json", row)
        record["status"] = "verified_rounds"
        return record
    except (TimeoutError, subprocess.TimeoutExpired) as error:
        record.update(status="timeout", error_type=type(error).__name__)
        raise
    except (KeyboardInterrupt, SystemExit) as error:
        record.update(status="cancelled", error_type=type(error).__name__)
        raise
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        record["seconds"] = time.monotonic() - started
        if record["status"] == "incomplete":
            record["status"] = "failed"
        if record["rounds"] and record["rounds"][-1]["status"] == "started":
            record["rounds"][-1]["status"] = record["status"]
        save(output / "result.json", record)
