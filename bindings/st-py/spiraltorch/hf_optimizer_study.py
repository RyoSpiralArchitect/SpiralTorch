"""Resumable orchestration for matched HF Z-Space optimizer studies."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .hf_optimizer_control import (
    compare_hf_zspace_optimizer_factorized_run_cards,
    write_hf_zspace_optimizer_factorized_ablation_report,
)

HF_ZSPACE_FACTORIZED_STUDY_SCHEMA = "spiraltorch.hf_zspace_factorized_study.v1"
HF_ZSPACE_FACTORIZED_STUDY_EVENT_SCHEMA = (
    "spiraltorch.hf_zspace_factorized_study_event.v1"
)
HF_ZSPACE_FACTORIZED_STUDY_SUMMARY_SCHEMA = (
    "spiraltorch.hf_zspace_factorized_study_summary.v1"
)
HF_ZSPACE_FACTORIZED_STUDY_ARMS = (
    "observe",
    "dose_matched_constant",
    "raw",
    "dose_normalized",
)
HF_ZSPACE_FACTORIZED_STUDY_PLAN_FILENAME = "study-plan.json"
HF_ZSPACE_FACTORIZED_STUDY_EVENTS_FILENAME = "study-events.jsonl"
HF_ZSPACE_FACTORIZED_STUDY_SUMMARY_FILENAME = "study-summary.json"
HF_ZSPACE_FACTORIZED_STUDY_REPORT_FILENAME = "factorized-report.json"

_MANAGED_BRIDGE_FLAGS = frozenset(
    {
        "--seed",
        "--output-dir",
        "--run-card",
        "--trainer-trace-jsonl",
        "--zspace-optimizer-control",
        "--zspace-optimizer-trace-jsonl",
        "--zspace-optimizer-trajectory-arm",
        "--zspace-optimizer-trajectory-json",
        "--zspace-optimizer-trajectory-out",
        "--min-free-disk-gb",
    }
)
_EVENT_RESERVED_FIELDS = frozenset(
    {
        "schema",
        "sequence",
        "recorded_at",
        "study_id",
        "event_type",
        "previous_event_id",
        "event_id",
    }
)


class HFZSpaceFactorizedStudyError(ValueError):
    """Raised when a factorized study cannot proceed without losing evidence."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_id(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HFZSpaceFactorizedStudyError(f"unreadable JSON artifact: {path}") from exc
    if not isinstance(payload, Mapping):
        raise HFZSpaceFactorizedStudyError(f"JSON artifact is not an object: {path}")
    return {str(key): value for key, value in payload.items()}


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_events(path: Path, *, study_id: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise HFZSpaceFactorizedStudyError(
                        f"study journal has a blank row at line {line_number}"
                    )
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise HFZSpaceFactorizedStudyError(
                        f"study journal row {line_number} is not an object"
                    )
                event = {str(key): value for key, value in payload.items()}
                if event.get("schema") != HF_ZSPACE_FACTORIZED_STUDY_EVENT_SCHEMA:
                    raise HFZSpaceFactorizedStudyError(
                        f"study journal row {line_number} has an unsupported schema"
                    )
                if event.get("study_id") != study_id:
                    raise HFZSpaceFactorizedStudyError(
                        f"study journal row {line_number} belongs to another study"
                    )
                if event.get("sequence") != len(events) + 1:
                    raise HFZSpaceFactorizedStudyError(
                        f"study journal sequence is broken at line {line_number}"
                    )
                previous_id = None if not events else events[-1].get("event_id")
                if event.get("previous_event_id") != previous_id:
                    raise HFZSpaceFactorizedStudyError(
                        f"study journal hash chain is broken at line {line_number}"
                    )
                event_id = event.get("event_id")
                identity_payload = {
                    key: value for key, value in event.items() if key != "event_id"
                }
                if event_id != _sha256_id(identity_payload):
                    raise HFZSpaceFactorizedStudyError(
                        f"study journal event identity is invalid at line {line_number}"
                    )
                events.append(event)
    except json.JSONDecodeError as exc:
        raise HFZSpaceFactorizedStudyError(
            f"study journal contains truncated or invalid JSON: {path}"
        ) from exc
    return events


def _append_event(
    path: Path,
    events: list[dict[str, Any]],
    *,
    study_id: str,
    event_type: str,
    details: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    if details and _EVENT_RESERVED_FIELDS.intersection(details):
        raise HFZSpaceFactorizedStudyError(
            "study event details cannot replace reserved envelope fields"
        )
    event: dict[str, Any] = {
        "schema": HF_ZSPACE_FACTORIZED_STUDY_EVENT_SCHEMA,
        "sequence": len(events) + 1,
        "recorded_at": _utc_now(),
        "study_id": study_id,
        "event_type": event_type,
        "previous_event_id": None if not events else events[-1]["event_id"],
    }
    if details:
        event.update(details)
    event["event_id"] = _sha256_id(event)
    encoded = _canonical_json_bytes(event) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab", buffering=0) as handle:
        handle.write(encoded)
        os.fsync(handle.fileno())
    events.append(event)
    return event


def _option_name(argument: str) -> str | None:
    if not argument.startswith("--"):
        return None
    return argument.split("=", 1)[0]


def _flag_value(arguments: Sequence[str], name: str) -> str | None:
    prefix = f"{name}="
    for index, argument in enumerate(arguments):
        if argument.startswith(prefix):
            return argument[len(prefix) :]
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
    return None


def _validate_base_args(arguments: Sequence[str]) -> int:
    if not arguments:
        raise HFZSpaceFactorizedStudyError("bridge arguments cannot be empty")
    managed = sorted(
        {
            name
            for argument in arguments
            if (name := _option_name(argument)) in _MANAGED_BRIDGE_FLAGS
        }
    )
    if managed:
        raise HFZSpaceFactorizedStudyError(
            "study-managed bridge flags cannot be supplied directly: "
            + ", ".join(managed)
        )
    if "--train" not in arguments:
        raise HFZSpaceFactorizedStudyError("factorized study requires --train")
    if "--eval-before-train" not in arguments:
        raise HFZSpaceFactorizedStudyError(
            "factorized study requires --eval-before-train"
        )
    if _flag_value(arguments, "--eval-after-train-policy") != "always":
        raise HFZSpaceFactorizedStudyError(
            "factorized study requires --eval-after-train-policy always"
        )
    raw_steps = _flag_value(arguments, "--max-steps")
    try:
        max_steps = int(raw_steps or "")
    except ValueError as exc:
        raise HFZSpaceFactorizedStudyError(
            "factorized study requires an integer --max-steps"
        ) from exc
    if max_steps <= 0:
        raise HFZSpaceFactorizedStudyError(
            "factorized study requires a positive --max-steps"
        )
    return max_steps


def _default_bridge_script() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "hf_finetune_bridge.py"


def _runtime_source_fingerprint() -> dict[str, object]:
    package_root = Path(__file__).resolve().parent
    files: list[dict[str, object]] = []
    for path in sorted(
        candidate for candidate in package_root.rglob("*") if candidate.is_file()
    ):
        if path.suffix not in {".py", ".pyi", ".json", ".so", ".dylib", ".pyd"}:
            continue
        files.append(
            {
                "relative_path": path.relative_to(package_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    payload = {
        "package": "spiraltorch",
        "file_count": len(files),
        "files": files,
    }
    return {**payload, "source_id": _sha256_id(payload)}


def _git_source_provenance(
    launch_cwd: Path,
    *,
    excluded_path: Path | None = None,
) -> dict[str, object]:
    def git(cwd: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(cwd), *arguments],
            capture_output=True,
            check=False,
            text=True,
            timeout=10.0,
        )

    try:
        root_result = git(launch_cwd, "rev-parse", "--show-toplevel")
        if root_result.returncode != 0:
            return {"available": False}
        root = Path(root_result.stdout.strip()).resolve()
        head = git(root, "rev-parse", "HEAD")
        branch = git(root, "branch", "--show-current")
        status_arguments = ["status", "--porcelain=v1", "--untracked-files=all"]
        excluded_paths: list[str] = []
        if excluded_path is not None:
            try:
                relative_exclusion = excluded_path.resolve().relative_to(root)
            except ValueError:
                pass
            else:
                if relative_exclusion != Path("."):
                    excluded_paths.append(relative_exclusion.as_posix())
                    status_arguments.extend(
                        ["--", ".", f":(exclude){relative_exclusion.as_posix()}"]
                    )
        status = git(root, *status_arguments)
    except (OSError, subprocess.SubprocessError):
        return {"available": False}
    if head.returncode != 0 or branch.returncode != 0 or status.returncode != 0:
        return {"available": False, "root": str(root)}
    status_rows = status.stdout.splitlines()
    return {
        "available": True,
        "root": str(root),
        "head": head.stdout.strip(),
        "branch": branch.stdout.strip(),
        "dirty": bool(status_rows),
        "status_row_count": len(status_rows),
        "status_id": _sha256_id(status_rows),
        "status_rows": status_rows,
        "excluded_paths": excluded_paths,
    }


def _run_label(seed: int, arm: str) -> str:
    return f"s{seed}-{arm}"


def _build_run_plan(
    *,
    study_id: str,
    study_dir: Path,
    seed: int,
    arm: str,
    python_executable: Path,
    bridge_script: Path,
    base_args: Sequence[str],
    min_free_disk_gb: float,
) -> dict[str, object]:
    label = _run_label(seed, arm)
    run_dir = study_dir / "runs" / f"s{seed}" / arm
    output_dir = run_dir / "output"
    run_card = run_dir / "run-card.json"
    trainer_trace = run_dir / "trainer-trace.jsonl"
    optimizer_trace = run_dir / "optimizer-trace.jsonl"
    log_path = run_dir / "run.log"
    trajectory_path = study_dir / "runs" / f"s{seed}" / "trajectory.json"
    command = [
        str(python_executable),
        str(bridge_script),
        *base_args,
        "--seed",
        str(seed),
        "--zspace-optimizer-control",
        "observe" if arm == "observe" else "apply",
    ]
    if arm != "observe":
        command.extend(["--zspace-optimizer-trajectory-arm", arm])
    command.extend(
        [
            "--output-dir",
            str(output_dir),
            "--run-card",
            str(run_card),
            "--trainer-trace-jsonl",
            str(trainer_trace),
            "--zspace-optimizer-trace-jsonl",
            str(optimizer_trace),
            "--min-free-disk-gb",
            format(min_free_disk_gb, ".17g"),
        ]
    )
    command.extend(
        [
            "--zspace-optimizer-trajectory-out"
            if arm == "observe"
            else "--zspace-optimizer-trajectory-json",
            str(trajectory_path),
        ]
    )
    run_identity = {
        "study_id": study_id,
        "seed": seed,
        "arm": arm,
        "command": command,
    }
    return {
        "run_id": _sha256_id(run_identity),
        "label": label,
        "seed": seed,
        "arm": arm,
        "command": command,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "run_card": str(run_card),
        "trainer_trace": str(trainer_trace),
        "optimizer_trace": str(optimizer_trace),
        "log": str(log_path),
        "trajectory": str(trajectory_path),
    }


def build_hf_zspace_optimizer_factorized_study_plan(
    *,
    study_dir: str | Path,
    seeds: Sequence[int],
    bridge_args: Sequence[str],
    bridge_script: str | Path | None = None,
    python_executable: str | Path | None = None,
    launch_cwd: str | Path | None = None,
    min_free_disk_gb: float = 5.0,
) -> dict[str, object]:
    """Build one immutable four-arm study plan without executing training."""

    resolved_study_dir = Path(study_dir).expanduser().resolve()
    resolved_bridge = (
        Path(bridge_script or _default_bridge_script()).expanduser().resolve()
    )
    resolved_python = Path(python_executable or sys.executable).expanduser().resolve()
    resolved_cwd = Path(launch_cwd or Path.cwd()).expanduser().resolve()
    if not resolved_bridge.is_file():
        raise HFZSpaceFactorizedStudyError(
            f"HF fine-tune bridge does not exist: {resolved_bridge}"
        )
    if not resolved_python.is_file():
        raise HFZSpaceFactorizedStudyError(
            f"Python executable does not exist: {resolved_python}"
        )
    normalized_args = tuple(str(argument) for argument in bridge_args)
    max_steps = _validate_base_args(normalized_args)
    normalized_seeds = tuple(sorted(set(int(seed) for seed in seeds)))
    if not normalized_seeds or len(normalized_seeds) != len(seeds):
        raise HFZSpaceFactorizedStudyError(
            "factorized study seeds must be non-empty and unique"
        )
    if any(seed < 0 for seed in normalized_seeds):
        raise HFZSpaceFactorizedStudyError(
            "factorized study seeds must be non-negative"
        )
    if not math.isfinite(min_free_disk_gb) or min_free_disk_gb < 0.0:
        raise HFZSpaceFactorizedStudyError(
            "min_free_disk_gb must be finite and non-negative"
        )
    runtime_source = _runtime_source_fingerprint()
    git_provenance = _git_source_provenance(
        resolved_cwd,
        excluded_path=resolved_study_dir,
    )
    scientific_spec = {
        "schema": HF_ZSPACE_FACTORIZED_STUDY_SCHEMA,
        "arms": list(HF_ZSPACE_FACTORIZED_STUDY_ARMS),
        "seeds": list(normalized_seeds),
        "max_steps": max_steps,
        "bridge_args": list(normalized_args),
        "bridge_sha256": _sha256_file(resolved_bridge),
        "launch_cwd": str(resolved_cwd),
        "runtime_source_id": runtime_source["source_id"],
        "git_head": git_provenance.get("head"),
        "git_status_id": git_provenance.get("status_id"),
    }
    study_id = _sha256_id(scientific_spec)
    runs = [
        _build_run_plan(
            study_id=study_id,
            study_dir=resolved_study_dir,
            seed=seed,
            arm=arm,
            python_executable=resolved_python,
            bridge_script=resolved_bridge,
            base_args=normalized_args,
            min_free_disk_gb=min_free_disk_gb,
        )
        for seed in normalized_seeds
        for arm in HF_ZSPACE_FACTORIZED_STUDY_ARMS
    ]
    return {
        "schema": HF_ZSPACE_FACTORIZED_STUDY_SCHEMA,
        "row_type": "hf_zspace_factorized_study_plan",
        "status": "planned",
        "study_id": study_id,
        "study_dir": str(resolved_study_dir),
        "scientific_spec": scientific_spec,
        "runtime_source_fingerprint": runtime_source,
        "git_source_provenance": git_provenance,
        "execution_policy": {
            "python_executable": str(resolved_python),
            "bridge_script": str(resolved_bridge),
            "min_free_disk_gb": min_free_disk_gb,
            "run_order": "seed_then_observe_constant_raw_normalized",
            "resume_policy": "verified_run_card_and_journal_receipt_only",
            "failure_policy": "preserve_and_fail_closed",
        },
        "artifacts": {
            "plan": str(resolved_study_dir / HF_ZSPACE_FACTORIZED_STUDY_PLAN_FILENAME),
            "events": str(
                resolved_study_dir / HF_ZSPACE_FACTORIZED_STUDY_EVENTS_FILENAME
            ),
            "summary": str(
                resolved_study_dir / HF_ZSPACE_FACTORIZED_STUDY_SUMMARY_FILENAME
            ),
            "factorized_report": str(
                resolved_study_dir / HF_ZSPACE_FACTORIZED_STUDY_REPORT_FILENAME
            ),
        },
        "run_count": len(runs),
        "runs": runs,
    }


def _persist_plan(plan: Mapping[str, object]) -> Path:
    study_dir = Path(str(plan["study_dir"]))
    path = study_dir / HF_ZSPACE_FACTORIZED_STUDY_PLAN_FILENAME
    if path.exists():
        existing = _read_json(path)
        if existing != plan:
            raise HFZSpaceFactorizedStudyError(
                "study directory already contains a different immutable plan"
            )
        return path
    _atomic_write_json(path, plan)
    return path


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def _study_lock(study_dir: Path, study_id: str) -> Iterator[None]:
    path = study_dir / ".study.lock"
    token = uuid.uuid4().hex
    payload = {
        "study_id": study_id,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "token": token,
        "acquired_at": _utc_now(),
    }
    study_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            current = _read_json(path)
            if current.get("hostname") != socket.gethostname():
                raise HFZSpaceFactorizedStudyError(
                    "factorized study lock belongs to another host and cannot "
                    "be proven stale"
                )
            active = isinstance(current.get("pid"), int) and _pid_is_running(
                int(current["pid"])
            )
            if active:
                raise HFZSpaceFactorizedStudyError(
                    f"factorized study is already running under pid {current['pid']}"
                )
            stale = study_dir / f".study.lock.stale-{int(time.time_ns())}"
            os.replace(path, stale)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        break
    else:
        raise HFZSpaceFactorizedStudyError("failed to acquire factorized study lock")
    try:
        yield
    finally:
        try:
            current = _read_json(path)
        except HFZSpaceFactorizedStudyError:
            current = {}
        if current.get("token") == token:
            path.unlink(missing_ok=True)


def _finite_eval_loss(card: Mapping[str, object], field: str) -> float:
    value = card.get(field)
    if not isinstance(value, Mapping) or value.get("status") != "ok":
        raise HFZSpaceFactorizedStudyError(f"run card has no ready {field}")
    loss = value.get("eval_loss")
    if isinstance(loss, bool) or not isinstance(loss, (int, float)):
        raise HFZSpaceFactorizedStudyError(f"run card has no finite {field} loss")
    numeric = float(loss)
    if not math.isfinite(numeric):
        raise HFZSpaceFactorizedStudyError(f"run card has no finite {field} loss")
    return numeric


def _ready_identity_id(
    card: Mapping[str, object],
    field: str,
    id_field: str,
) -> str:
    identity = card.get(field)
    if (
        not isinstance(identity, Mapping)
        or identity.get("status") != "ready"
        or identity.get("identity_verified") is not True
        or identity.get("path_independent") is not True
    ):
        raise HFZSpaceFactorizedStudyError(f"run card has no ready {field}")
    value = identity.get(id_field)
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise HFZSpaceFactorizedStudyError(f"run card has no valid {field} identity")
    return value


def _validate_completed_run_card(
    run: Mapping[str, object],
    card: Mapping[str, object],
) -> dict[str, object]:
    if card.get("row_type") not in {
        "hf_finetune_run_card",
        "hf_gpt2_finetune_run_card",
    }:
        raise HFZSpaceFactorizedStudyError("run card has an unsupported row_type")
    if card.get("failure_stage") is not None or card.get("failure_error") is not None:
        raise HFZSpaceFactorizedStudyError("run card records a training failure")
    if card.get("train_requested") is not True:
        raise HFZSpaceFactorizedStudyError("run card does not represent a training run")
    expected_command = [str(value) for value in run["command"]]  # type: ignore[index]
    launch_command = card.get("launch_command")
    if not isinstance(launch_command, Sequence) or isinstance(
        launch_command, (str, bytes)
    ):
        raise HFZSpaceFactorizedStudyError("run card has no launch command")
    observed_command = [str(value) for value in launch_command]
    if observed_command[: len(expected_command)] != expected_command:
        raise HFZSpaceFactorizedStudyError(
            "run card launch command does not match the study run"
        )
    identity = card.get("training_recipe_identity")
    if not isinstance(identity, Mapping) or identity.get("status") != "ready":
        raise HFZSpaceFactorizedStudyError(
            "run card has no ready training recipe identity"
        )
    identity_payload = identity.get("identity_payload")
    training_arguments = (
        identity_payload.get("training_arguments")
        if isinstance(identity_payload, Mapping)
        else None
    )
    if not isinstance(training_arguments, Mapping) or training_arguments.get(
        "seed"
    ) != run.get("seed"):
        raise HFZSpaceFactorizedStudyError("run card seed does not match the study")
    receipt = card.get("zspace_optimizer_control_receipt")
    if not isinstance(receipt, Mapping) or receipt.get("status") != "ready":
        raise HFZSpaceFactorizedStudyError(
            "run card has no ready Z-Space optimizer receipt"
        )
    expected_arm = str(run["arm"])
    expected_mode = "observe" if expected_arm == "observe" else "apply"
    receipt_arm = "raw" if expected_arm == "observe" else expected_arm
    if (
        receipt.get("mode") != expected_mode
        or receipt.get("trajectory_arm") != receipt_arm
    ):
        raise HFZSpaceFactorizedStudyError(
            "run card optimizer arm does not match the study"
        )
    blockers = receipt.get("evidence_blockers")
    if blockers not in (None, []):
        raise HFZSpaceFactorizedStudyError(
            "run card optimizer receipt contains evidence blockers"
        )
    if receipt.get("schedule_evidence_complete") is not True:
        raise HFZSpaceFactorizedStudyError(
            "run card optimizer schedule evidence is incomplete"
        )
    if receipt.get("trajectory_validated") is not True:
        raise HFZSpaceFactorizedStudyError("run card trajectory is not Rust-validated")
    planned = receipt.get("planned_update_count")
    realized = receipt.get("realized_update_count")
    trajectory_steps = receipt.get("trajectory_step_count")
    if not (
        isinstance(planned, int)
        and not isinstance(planned, bool)
        and planned > 0
        and realized == planned
        and trajectory_steps == planned
        and receipt.get("training_horizon_complete") is True
        and receipt.get("trajectory_horizon_complete") is True
    ):
        raise HFZSpaceFactorizedStudyError(
            "run card does not prove a complete optimizer horizon"
        )
    before_loss = _finite_eval_loss(card, "eval_before_train")
    after_loss = _finite_eval_loss(card, "eval_after_train")
    trajectory_id = receipt.get("trajectory_id")
    if not isinstance(trajectory_id, str) or not trajectory_id.startswith("sha256:"):
        raise HFZSpaceFactorizedStudyError("run card has no trajectory identity")
    trajectory_path = Path(str(run["trajectory"]))
    if not trajectory_path.is_file():
        raise HFZSpaceFactorizedStudyError(
            f"run card trajectory artifact is missing: {trajectory_path}"
        )
    trajectory = _read_json(trajectory_path)
    if trajectory.get("trajectory_id") != trajectory_id:
        raise HFZSpaceFactorizedStudyError(
            "run card trajectory identity does not match its artifact"
        )
    optimizer_trace = Path(str(run["optimizer_trace"]))
    if not optimizer_trace.is_file() or receipt.get("trace_sha256") != _sha256_file(
        optimizer_trace
    ):
        raise HFZSpaceFactorizedStudyError(
            "run card optimizer trace is missing or differs from its receipt"
        )
    trainer_trace = Path(str(run["trainer_trace"]))
    trainer_receipt = card.get("trainer_trace_segment_receipt")
    if (
        not trainer_trace.is_file()
        or not isinstance(trainer_receipt, Mapping)
        or trainer_receipt.get("status") != "ready"
        or trainer_receipt.get("ready") is not True
        or trainer_receipt.get("trace_sha256") != _sha256_file(trainer_trace)
    ):
        raise HFZSpaceFactorizedStudyError(
            "run card trainer trace is missing or differs from its receipt"
        )
    if not Path(str(run["output_dir"])).is_dir():
        raise HFZSpaceFactorizedStudyError("run output directory is missing")
    return {
        "before_eval_loss": before_loss,
        "after_eval_loss": after_loss,
        "trajectory_id": trajectory_id,
        "realized_update_count": realized,
        "execution_identity_id": _ready_identity_id(
            card,
            "finetune_execution_identity_after_model",
            "observed_identity_id",
        ),
        "runtime_identity_id": _ready_identity_id(
            card,
            "model_runtime_identity_after_model",
            "observed_identity_id",
        ),
        "training_input_id": _ready_identity_id(
            card,
            "training_input_identity_after_load",
            "observed_input_id",
        ),
    }


def _enforce_study_identity_anchor(
    anchor: dict[str, str],
    facts: Mapping[str, object],
) -> None:
    for field in (
        "execution_identity_id",
        "runtime_identity_id",
        "training_input_id",
    ):
        value = facts.get(field)
        if not isinstance(value, str):
            raise HFZSpaceFactorizedStudyError(
                f"completed run has no study identity field {field}"
            )
        previous = anchor.setdefault(field, value)
        if previous != value:
            raise HFZSpaceFactorizedStudyError(
                f"completed runs disagree on study identity field {field}"
            )


def _completed_event_by_run(
    events: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    completed: dict[str, Mapping[str, object]] = {}
    for event in events:
        if event.get("event_type") in {
            "run_completed",
            "run_recovered",
            "run_reused",
        }:
            run_id = event.get("run_id")
            if isinstance(run_id, str):
                completed[run_id] = event
    return completed


def _verify_reusable_run(
    run: Mapping[str, object],
    event: Mapping[str, object] | None,
) -> dict[str, object] | None:
    card_path = Path(str(run["run_card"]))
    if not card_path.is_file():
        if event is not None:
            raise HFZSpaceFactorizedStudyError(
                f"completed run card is missing: {card_path}"
            )
        return None
    card = _read_json(card_path)
    facts = _validate_completed_run_card(run, card)
    card_sha256 = _sha256_file(card_path)
    if event is not None:
        if event.get("command_id") != run.get("run_id"):
            raise HFZSpaceFactorizedStudyError(
                f"completed run command identity changed: {run['label']}"
            )
        if event.get("run_card_sha256") != card_sha256:
            raise HFZSpaceFactorizedStudyError(
                f"completed run card changed after receipt: {card_path}"
            )
    return {**facts, "run_card_sha256": card_sha256, "recovered": event is None}


def _run_has_artifacts(run: Mapping[str, object]) -> bool:
    fields = [
        "output_dir",
        "run_card",
        "trainer_trace",
        "optimizer_trace",
        "log",
    ]
    if run.get("arm") == "observe":
        fields.append("trajectory")
    return any(Path(str(run[field])).exists() for field in fields)


def _quarantine_run_artifacts(
    study_dir: Path,
    run: Mapping[str, object],
) -> Path:
    destination = study_dir / "quarantine" / f"{run['label']}-{time.time_ns()}"
    destination.mkdir(parents=True, exist_ok=False)
    fields = [
        "output_dir",
        "run_card",
        "trainer_trace",
        "optimizer_trace",
        "log",
    ]
    if run.get("arm") == "observe":
        fields.append("trajectory")
    for field in fields:
        source = Path(str(run[field]))
        if source.exists():
            shutil.move(str(source), destination / source.name)
    return destination


def _free_disk_gb(path: Path) -> float:
    existing = path
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    return shutil.disk_usage(existing).free / (1024**3)


def _execute_run(run: Mapping[str, object], *, cwd: Path) -> tuple[int, float]:
    command = [str(value) for value in run["command"]]  # type: ignore[index]
    log_path = Path(str(run["log"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("xb") as log:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(result.returncode), time.monotonic() - started


def _write_summary(
    path: Path,
    *,
    plan: Mapping[str, object],
    status: str,
    run_statuses: Mapping[str, Mapping[str, object]],
    events: Sequence[Mapping[str, object]],
    factorized_report: Mapping[str, object] | None = None,
    identity_anchor: Mapping[str, str] | None = None,
    error: str | None = None,
) -> dict[str, object]:
    total_runs = int(plan["run_count"])
    completed = sum(
        1
        for value in run_statuses.values()
        if value.get("status") in {"completed", "recovered", "reused"}
    )
    report_path = Path(
        str(plan["artifacts"]["factorized_report"])  # type: ignore[index]
    )
    summary: dict[str, object] = {
        "schema": HF_ZSPACE_FACTORIZED_STUDY_SUMMARY_SCHEMA,
        "row_type": "hf_zspace_factorized_study_summary",
        "recorded_at": _utc_now(),
        "status": status,
        "study_id": plan["study_id"],
        "study_dir": plan["study_dir"],
        "run_count": total_runs,
        "completed_run_count": completed,
        "remaining_run_count": total_runs - completed,
        "event_count": len(events),
        "last_event": None if not events else events[-1].get("event_type"),
        "free_disk_gb": _free_disk_gb(Path(str(plan["study_dir"]))),
        "run_statuses": dict(run_statuses),
        "identity_anchor": dict(identity_anchor or {}),
        "factorized_report": str(report_path),
        "factorized_report_sha256": (
            _sha256_file(report_path) if report_path.is_file() else None
        ),
        "factorized_status": (
            None if factorized_report is None else factorized_report.get("status")
        ),
        "error": error,
    }
    _atomic_write_json(path, summary)
    return summary


def run_hf_zspace_optimizer_factorized_study(
    *,
    study_dir: str | Path,
    seeds: Sequence[int],
    bridge_args: Sequence[str],
    bridge_script: str | Path | None = None,
    python_executable: str | Path | None = None,
    launch_cwd: str | Path | None = None,
    min_free_disk_gb: float = 5.0,
    execute: bool = False,
    retry_failed: bool = False,
) -> dict[str, object]:
    """Plan or execute a fail-closed, resumable factorized optimizer study."""

    plan = build_hf_zspace_optimizer_factorized_study_plan(
        study_dir=study_dir,
        seeds=seeds,
        bridge_args=bridge_args,
        bridge_script=bridge_script,
        python_executable=python_executable,
        launch_cwd=launch_cwd,
        min_free_disk_gb=min_free_disk_gb,
    )
    root = Path(str(plan["study_dir"]))
    artifacts = plan["artifacts"]
    assert isinstance(artifacts, Mapping)
    event_path = Path(str(artifacts["events"]))
    summary_path = Path(str(artifacts["summary"]))
    study_id = str(plan["study_id"])
    launch_directory = Path(
        str(plan["scientific_spec"]["launch_cwd"])  # type: ignore[index]
    )
    min_free = float(
        plan["execution_policy"]["min_free_disk_gb"]  # type: ignore[index]
    )
    run_statuses: dict[str, dict[str, object]] = {}
    factorized_report: dict[str, object] | None = None
    identity_anchor: dict[str, str] = {}
    with _study_lock(root, study_id):
        _persist_plan(plan)
        events = _load_events(event_path, study_id=study_id)
        if not execute:
            if summary_path.is_file():
                summary = _read_json(summary_path)
                if summary.get("study_id") != study_id:
                    raise HFZSpaceFactorizedStudyError(
                        "study summary belongs to another immutable plan"
                    )
                return summary
            return _write_summary(
                summary_path,
                plan=plan,
                status="planned",
                run_statuses={},
                events=events,
            )
        _append_event(
            event_path,
            events,
            study_id=study_id,
            event_type="study_started" if not events else "study_resumed",
            details={"pid": os.getpid(), "hostname": socket.gethostname()},
        )
        completed_events = _completed_event_by_run(events)
        try:
            for raw_run in plan["runs"]:  # type: ignore[index]
                assert isinstance(raw_run, Mapping)
                run = {str(key): value for key, value in raw_run.items()}
                label = str(run["label"])
                run_id = str(run["run_id"])
                reusable = _verify_reusable_run(run, completed_events.get(run_id))
                if reusable is not None:
                    _enforce_study_identity_anchor(identity_anchor, reusable)
                    status = "recovered" if reusable["recovered"] else "reused"
                    run_statuses[label] = {"status": status, **reusable}
                    event = _append_event(
                        event_path,
                        events,
                        study_id=study_id,
                        event_type=(
                            "run_recovered" if reusable["recovered"] else "run_reused"
                        ),
                        details={
                            "run_id": run_id,
                            "command_id": run_id,
                            "label": label,
                            "seed": run["seed"],
                            "arm": run["arm"],
                            "run_card_sha256": reusable["run_card_sha256"],
                            "trajectory_id": reusable["trajectory_id"],
                            "execution_identity_id": reusable["execution_identity_id"],
                            "runtime_identity_id": reusable["runtime_identity_id"],
                            "training_input_id": reusable["training_input_id"],
                        },
                    )
                    completed_events[run_id] = event
                    _write_summary(
                        summary_path,
                        plan=plan,
                        status="running",
                        run_statuses=run_statuses,
                        events=events,
                        identity_anchor=identity_anchor,
                    )
                    continue
                if _run_has_artifacts(run):
                    if not retry_failed:
                        raise HFZSpaceFactorizedStudyError(
                            f"unverified artifacts block retry for {label}; "
                            "use retry_failed=True to preserve them in quarantine"
                        )
                    destination = _quarantine_run_artifacts(root, run)
                    _append_event(
                        event_path,
                        events,
                        study_id=study_id,
                        event_type="run_quarantined",
                        details={
                            "run_id": run_id,
                            "label": label,
                            "destination": str(destination),
                        },
                    )
                free_before = _free_disk_gb(root)
                if free_before < min_free:
                    raise HFZSpaceFactorizedStudyError(
                        f"free disk {free_before:.3f} GiB is below the "
                        f"{min_free:.3f} GiB study guard before {label}"
                    )
                _append_event(
                    event_path,
                    events,
                    study_id=study_id,
                    event_type="run_started",
                    details={
                        "run_id": run_id,
                        "command_id": run_id,
                        "label": label,
                        "seed": run["seed"],
                        "arm": run["arm"],
                        "command": run["command"],
                        "free_disk_gb_before": free_before,
                    },
                )
                returncode, duration = _execute_run(run, cwd=launch_directory)
                free_after = _free_disk_gb(root)
                if returncode != 0:
                    _append_event(
                        event_path,
                        events,
                        study_id=study_id,
                        event_type="run_failed",
                        details={
                            "run_id": run_id,
                            "command_id": run_id,
                            "label": label,
                            "returncode": returncode,
                            "duration_seconds": duration,
                            "free_disk_gb_after": free_after,
                            "log": run["log"],
                        },
                    )
                    run_statuses[label] = {
                        "status": "failed",
                        "returncode": returncode,
                        "log": run["log"],
                    }
                    return _write_summary(
                        summary_path,
                        plan=plan,
                        status="failed",
                        run_statuses=run_statuses,
                        events=events,
                        identity_anchor=identity_anchor,
                        error=f"{label} exited with return code {returncode}",
                    )
                card_path = Path(str(run["run_card"]))
                if not card_path.is_file():
                    raise HFZSpaceFactorizedStudyError(
                        f"successful child did not write its run card: {card_path}"
                    )
                card = _read_json(card_path)
                facts = _validate_completed_run_card(run, card)
                _enforce_study_identity_anchor(identity_anchor, facts)
                card_sha256 = _sha256_file(card_path)
                event = _append_event(
                    event_path,
                    events,
                    study_id=study_id,
                    event_type="run_completed",
                    details={
                        "run_id": run_id,
                        "command_id": run_id,
                        "label": label,
                        "seed": run["seed"],
                        "arm": run["arm"],
                        "returncode": returncode,
                        "duration_seconds": duration,
                        "free_disk_gb_after": free_after,
                        "run_card": str(card_path),
                        "run_card_sha256": card_sha256,
                        **facts,
                    },
                )
                completed_events[run_id] = event
                run_statuses[label] = {
                    "status": "completed",
                    "duration_seconds": duration,
                    "run_card_sha256": card_sha256,
                    **facts,
                }
                _write_summary(
                    summary_path,
                    plan=plan,
                    status="running",
                    run_statuses=run_statuses,
                    events=events,
                    identity_anchor=identity_anchor,
                )

            cards = [Path(str(run["run_card"])) for run in plan["runs"]]  # type: ignore[index]
            factorized_report = compare_hf_zspace_optimizer_factorized_run_cards(cards)
            report_path = Path(str(artifacts["factorized_report"]))
            write_hf_zspace_optimizer_factorized_ablation_report(
                factorized_report,
                report_path,
            )
            final_status = (
                "ready" if factorized_report.get("status") == "ready" else "blocked"
            )
            _append_event(
                event_path,
                events,
                study_id=study_id,
                event_type="study_completed",
                details={
                    "status": final_status,
                    "factorized_report": str(report_path),
                    "factorized_report_sha256": _sha256_file(report_path),
                    "matched_seed_count": factorized_report.get("matched_seed_count"),
                    "error_count": factorized_report.get("error_count"),
                },
            )
            return _write_summary(
                summary_path,
                plan=plan,
                status=final_status,
                run_statuses=run_statuses,
                events=events,
                factorized_report=factorized_report,
                identity_anchor=identity_anchor,
            )
        except Exception as exc:
            _append_event(
                event_path,
                events,
                study_id=study_id,
                event_type="study_failed",
                details={
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
            )
            _write_summary(
                summary_path,
                plan=plan,
                status="failed",
                run_statuses=run_statuses,
                events=events,
                factorized_report=factorized_report,
                identity_anchor=identity_anchor,
                error=str(exc),
            )
            raise


__all__ = [
    "HF_ZSPACE_FACTORIZED_STUDY_ARMS",
    "HF_ZSPACE_FACTORIZED_STUDY_EVENT_SCHEMA",
    "HF_ZSPACE_FACTORIZED_STUDY_REPORT_FILENAME",
    "HF_ZSPACE_FACTORIZED_STUDY_SCHEMA",
    "HF_ZSPACE_FACTORIZED_STUDY_SUMMARY_SCHEMA",
    "HFZSpaceFactorizedStudyError",
    "build_hf_zspace_optimizer_factorized_study_plan",
    "run_hf_zspace_optimizer_factorized_study",
]
