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
from typing import Any, Callable, Iterator, Mapping, Sequence

from .hf_optimizer_control import (
    HF_ZSPACE_FACTORIZED_ABLATION_SCHEMA,
    compare_hf_zspace_optimizer_factorized_run_cards,
    compare_hf_zspace_optimizer_feedback_run_cards,
    compare_hf_zspace_optimizer_polarity_run_cards,
    write_hf_zspace_optimizer_factorized_ablation_report,
    write_hf_zspace_optimizer_feedback_ablation_report,
    write_hf_zspace_optimizer_polarity_ablation_report,
)
from .zspace_optimizer import (
    ZSPACE_OPTIMIZER_FEEDBACK_CONTRACT_VERSION,
    ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_OWNER,
    zspace_optimizer_feedback_init,
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
HF_ZSPACE_FACTORIZED_GAIN_RESPONSE_SCHEMA = (
    "spiraltorch.hf_zspace_factorized_gain_response.v1"
)
HF_ZSPACE_FACTORIZED_GAIN_RESPONSE_FILENAME = "gain-response.json"
HF_ZSPACE_FEEDBACK_STUDY_SCHEMA = "spiraltorch.hf_zspace_feedback_study.v1"
HF_ZSPACE_FEEDBACK_STUDY_EVENT_SCHEMA = "spiraltorch.hf_zspace_feedback_study_event.v1"
HF_ZSPACE_FEEDBACK_STUDY_SUMMARY_SCHEMA = (
    "spiraltorch.hf_zspace_feedback_study_summary.v1"
)
HF_ZSPACE_FEEDBACK_STUDY_ARMS = (
    "observe",
    "raw_unguarded",
    "raw_loss_guard",
)
HF_ZSPACE_FEEDBACK_STUDY_PLAN_FILENAME = "feedback-study-plan.json"
HF_ZSPACE_FEEDBACK_STUDY_EVENTS_FILENAME = "feedback-study-events.jsonl"
HF_ZSPACE_FEEDBACK_STUDY_SUMMARY_FILENAME = "feedback-study-summary.json"
HF_ZSPACE_FEEDBACK_STUDY_REPORT_FILENAME = "feedback-report.json"
HF_ZSPACE_POLARITY_STUDY_SCHEMA = "spiraltorch.hf_zspace_polarity_study.v1"
HF_ZSPACE_POLARITY_STUDY_EVENT_SCHEMA = "spiraltorch.hf_zspace_polarity_study_event.v1"
HF_ZSPACE_POLARITY_STUDY_SUMMARY_SCHEMA = (
    "spiraltorch.hf_zspace_polarity_study_summary.v1"
)
HF_ZSPACE_POLARITY_STUDY_ARMS = (
    "observe",
    "dose_normalized",
    "dose_preserving_complement",
)
HF_ZSPACE_POLARITY_STUDY_PLAN_FILENAME = "polarity-study-plan.json"
HF_ZSPACE_POLARITY_STUDY_EVENTS_FILENAME = "polarity-study-events.jsonl"
HF_ZSPACE_POLARITY_STUDY_SUMMARY_FILENAME = "polarity-study-summary.json"
HF_ZSPACE_POLARITY_STUDY_REPORT_FILENAME = "polarity-report.json"

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
        "--validate-args-only",
    }
)
_FEEDBACK_CONFIG_FLAGS = {
    "loss_ema_alpha": "--zspace-optimizer-feedback-loss-ema-alpha",
    "relative_delta_ema_alpha": (
        "--zspace-optimizer-feedback-relative-delta-ema-alpha"
    ),
    "loss_floor": "--zspace-optimizer-feedback-loss-floor",
    "regression_threshold": "--zspace-optimizer-feedback-regression-threshold",
    "halt_threshold": "--zspace-optimizer-feedback-halt-threshold",
    "recovery_threshold": "--zspace-optimizer-feedback-recovery-threshold",
    "attenuation_rate": "--zspace-optimizer-feedback-attenuation-rate",
    "recovery_rate": "--zspace-optimizer-feedback-recovery-rate",
    "halt_regression_streak": ("--zspace-optimizer-feedback-halt-regression-streak"),
    "resume_improvement_streak": (
        "--zspace-optimizer-feedback-resume-improvement-streak"
    ),
    "warmup_observations": "--zspace-optimizer-feedback-warmup-observations",
    "max_stale_updates": "--zspace-optimizer-feedback-max-stale-updates",
    "maximum_gate": "--zspace-optimizer-feedback-maximum-gate",
}
_FEEDBACK_MANAGED_BRIDGE_FLAGS = frozenset(
    {
        *_MANAGED_BRIDGE_FLAGS,
        "--logging-steps",
        "--require-eval-dataset",
        "--zspace-optimizer-feedback",
        *_FEEDBACK_CONFIG_FLAGS.values(),
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
_CONTROL_GAIN_FLAG = "--zspace-optimizer-control-gain"
_GAIN_RESPONSE_CONTRASTS = (
    "dose_effect",
    "dose_normalized_shape_effect",
    "raw_total_effect",
    "shape_effect_at_raw_dose",
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


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


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


def _load_events(
    path: Path,
    *,
    study_id: str,
    event_schema: str = HF_ZSPACE_FACTORIZED_STUDY_EVENT_SCHEMA,
) -> list[dict[str, Any]]:
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
                if event.get("schema") != event_schema:
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
    event_schema: str = HF_ZSPACE_FACTORIZED_STUDY_EVENT_SCHEMA,
) -> dict[str, Any]:
    if details and _EVENT_RESERVED_FIELDS.intersection(details):
        raise HFZSpaceFactorizedStudyError(
            "study event details cannot replace reserved envelope fields"
        )
    event: dict[str, Any] = {
        "schema": event_schema,
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
    package = sys.modules.get("spiraltorch")
    native = getattr(package, "_rs", None)
    native_file = getattr(native, "__file__", None)
    native_path = (
        Path(native_file).expanduser().resolve()
        if isinstance(native_file, str) and native_file
        else None
    )
    native_extension: dict[str, object]
    if native_path is not None and native_path.is_file():
        native_extension = {
            "status": "ready",
            "filename": native_path.name,
            "size_bytes": native_path.stat().st_size,
            "sha256": _sha256_file(native_path),
        }
    else:
        native_extension = {
            "status": "unavailable",
            "filename": None,
            "size_bytes": None,
            "sha256": None,
        }
    payload = {
        "package": "spiraltorch",
        "file_count": len(files),
        "files": files,
        "loaded_native_extension": native_extension,
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
        "expected_mode": "observe" if arm == "observe" else "apply",
        "expected_trajectory_arm": "raw" if arm == "observe" else arm,
        "expected_feedback_mode": "off",
        "trajectory_owner": arm == "observe",
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


def build_hf_zspace_optimizer_polarity_study_plan(
    *,
    study_dir: str | Path,
    seeds: Sequence[int],
    bridge_args: Sequence[str],
    bridge_script: str | Path | None = None,
    python_executable: str | Path | None = None,
    launch_cwd: str | Path | None = None,
    min_free_disk_gb: float = 5.0,
) -> dict[str, object]:
    """Build one immutable dose-matched trajectory-polarity study plan."""

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
            "polarity study seeds must be non-empty and unique"
        )
    if any(seed < 0 for seed in normalized_seeds):
        raise HFZSpaceFactorizedStudyError(
            "polarity study seeds must be non-negative"
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
        "schema": HF_ZSPACE_POLARITY_STUDY_SCHEMA,
        "arms": list(HF_ZSPACE_POLARITY_STUDY_ARMS),
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
        for arm in HF_ZSPACE_POLARITY_STUDY_ARMS
    ]
    return {
        "schema": HF_ZSPACE_POLARITY_STUDY_SCHEMA,
        "row_type": "hf_zspace_polarity_study_plan",
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
            "run_order": "seed_then_observe_normalized_complement",
            "resume_policy": "verified_run_card_and_journal_receipt_only",
            "failure_policy": "preserve_and_fail_closed",
        },
        "artifacts": {
            "plan": str(resolved_study_dir / HF_ZSPACE_POLARITY_STUDY_PLAN_FILENAME),
            "events": str(
                resolved_study_dir / HF_ZSPACE_POLARITY_STUDY_EVENTS_FILENAME
            ),
            "summary": str(
                resolved_study_dir / HF_ZSPACE_POLARITY_STUDY_SUMMARY_FILENAME
            ),
            "polarity_report": str(
                resolved_study_dir / HF_ZSPACE_POLARITY_STUDY_REPORT_FILENAME
            ),
        },
        "run_count": len(runs),
        "runs": runs,
    }


def _validate_feedback_base_args(arguments: Sequence[str]) -> int:
    managed = sorted(
        {
            name
            for argument in arguments
            if (name := _option_name(argument)) in _FEEDBACK_MANAGED_BRIDGE_FLAGS
        }
    )
    if managed:
        raise HFZSpaceFactorizedStudyError(
            "feedback study-managed bridge flags cannot be supplied directly: "
            + ", ".join(managed)
        )
    return _validate_base_args(arguments)


def _validate_feedback_bridge_arguments(
    *,
    bridge_script: Path,
    python_executable: Path,
    bridge_args: Sequence[str],
    launch_cwd: Path,
) -> dict[str, object]:
    default_bridge = _default_bridge_script().expanduser().resolve()
    if bridge_script != default_bridge:
        return {
            "status": "not_run_custom_bridge",
            "ready": None,
            "validator": None,
            "validated_argument_count": 0,
        }
    command = [
        str(python_executable),
        str(bridge_script),
        *bridge_args,
        "--validate-args-only",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=launch_cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HFZSpaceFactorizedStudyError(
            "default HF bridge argument validation could not run"
        ) from exc
    if result.returncode != 0:
        detail_lines = [
            line.strip()
            for line in (result.stderr or result.stdout).splitlines()
            if line.strip()
        ]
        detail = detail_lines[-1] if detail_lines else f"exit {result.returncode}"
        raise HFZSpaceFactorizedStudyError(
            f"default HF bridge rejected study arguments: {detail}"
        )
    if "hf_finetune_bridge_args_valid" not in result.stdout.splitlines():
        raise HFZSpaceFactorizedStudyError(
            "default HF bridge argument validation returned no success receipt"
        )
    return {
        "status": "ready",
        "ready": True,
        "validator": "default_bridge_parse_args",
        "validated_argument_count": len(bridge_args),
    }


def _resolved_feedback_config(
    requested: Mapping[str, object] | None,
) -> dict[str, object]:
    try:
        checkpoint = zspace_optimizer_feedback_init(dict(requested or {}))
    except Exception as exc:
        raise HFZSpaceFactorizedStudyError(
            "Rust could not validate the feedback study config"
        ) from exc
    if (
        checkpoint.get("contract_version") != ZSPACE_OPTIMIZER_FEEDBACK_CONTRACT_VERSION
        or checkpoint.get("semantic_owner") != ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_OWNER
        or checkpoint.get("semantic_backend") != "rust"
    ):
        raise HFZSpaceFactorizedStudyError(
            "feedback study config did not resolve through the Rust contract"
        )
    config = checkpoint.get("config")
    if not isinstance(config, Mapping) or set(config) != set(_FEEDBACK_CONFIG_FLAGS):
        raise HFZSpaceFactorizedStudyError(
            "Rust feedback config does not match the study CLI surface"
        )
    return {str(key): value for key, value in config.items()}


def _feedback_config_cli_args(config: Mapping[str, object]) -> list[str]:
    arguments: list[str] = []
    for field, flag in _FEEDBACK_CONFIG_FLAGS.items():
        value = config[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HFZSpaceFactorizedStudyError(
                f"Rust feedback config field {field} is not numeric"
            )
        arguments.extend([flag, format(value, ".17g")])
    return arguments


def _build_feedback_run_plan(
    *,
    study_id: str,
    study_dir: Path,
    seed: int,
    arm: str,
    python_executable: Path,
    bridge_script: Path,
    base_args: Sequence[str],
    feedback_config: Mapping[str, object],
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
    expected_mode = "observe" if arm == "observe" else "apply"
    expected_feedback_mode = "loss_guard" if arm == "raw_loss_guard" else "off"
    command = [
        str(python_executable),
        str(bridge_script),
        *base_args,
        "--seed",
        str(seed),
        "--logging-steps",
        "1",
        "--require-eval-dataset",
        "--zspace-optimizer-control",
        expected_mode,
        "--zspace-optimizer-feedback",
        expected_feedback_mode,
    ]
    if expected_mode == "apply":
        command.extend(["--zspace-optimizer-trajectory-arm", "raw"])
    if expected_feedback_mode == "loss_guard":
        command.extend(_feedback_config_cli_args(feedback_config))
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
            (
                "--zspace-optimizer-trajectory-out"
                if arm == "observe"
                else "--zspace-optimizer-trajectory-json"
            ),
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
        "expected_mode": expected_mode,
        "expected_trajectory_arm": "raw",
        "expected_feedback_mode": expected_feedback_mode,
        "expected_feedback_config_id": (
            _sha256_id(feedback_config)
            if expected_feedback_mode == "loss_guard"
            else None
        ),
        "trajectory_owner": arm == "observe",
        "command": command,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "run_card": str(run_card),
        "trainer_trace": str(trainer_trace),
        "optimizer_trace": str(optimizer_trace),
        "log": str(log_path),
        "trajectory": str(trajectory_path),
    }


def build_hf_zspace_optimizer_feedback_study_plan(
    *,
    study_dir: str | Path,
    seeds: Sequence[int],
    bridge_args: Sequence[str],
    feedback_config: Mapping[str, object] | None = None,
    bridge_script: str | Path | None = None,
    python_executable: str | Path | None = None,
    launch_cwd: str | Path | None = None,
    min_free_disk_gb: float = 5.0,
) -> dict[str, object]:
    """Build one immutable baseline/unguarded/guarded HF study plan."""

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
    max_steps = _validate_feedback_base_args(normalized_args)
    normalized_seeds = tuple(sorted(set(int(seed) for seed in seeds)))
    if not normalized_seeds or len(normalized_seeds) != len(seeds):
        raise HFZSpaceFactorizedStudyError(
            "feedback study seeds must be non-empty and unique"
        )
    if any(seed < 0 for seed in normalized_seeds):
        raise HFZSpaceFactorizedStudyError("feedback study seeds must be non-negative")
    if not math.isfinite(min_free_disk_gb) or min_free_disk_gb < 0.0:
        raise HFZSpaceFactorizedStudyError(
            "min_free_disk_gb must be finite and non-negative"
        )
    bridge_argument_validation = _validate_feedback_bridge_arguments(
        bridge_script=resolved_bridge,
        python_executable=resolved_python,
        bridge_args=normalized_args,
        launch_cwd=resolved_cwd,
    )
    resolved_feedback_config = _resolved_feedback_config(feedback_config)
    feedback_config_id = _sha256_id(resolved_feedback_config)
    runtime_source = _runtime_source_fingerprint()
    git_provenance = _git_source_provenance(
        resolved_cwd,
        excluded_path=resolved_study_dir,
    )
    scientific_spec = {
        "schema": HF_ZSPACE_FEEDBACK_STUDY_SCHEMA,
        "arms": list(HF_ZSPACE_FEEDBACK_STUDY_ARMS),
        "seeds": list(normalized_seeds),
        "max_steps": max_steps,
        "logging_steps": 1,
        "require_eval_dataset": True,
        "bridge_args": list(normalized_args),
        "bridge_argument_validation": bridge_argument_validation,
        "bridge_sha256": _sha256_file(resolved_bridge),
        "launch_cwd": str(resolved_cwd),
        "runtime_source_id": runtime_source["source_id"],
        "git_head": git_provenance.get("head"),
        "git_status_id": git_provenance.get("status_id"),
        "feedback_contract": ZSPACE_OPTIMIZER_FEEDBACK_CONTRACT_VERSION,
        "feedback_semantic_owner": ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_OWNER,
        "feedback_config": resolved_feedback_config,
        "feedback_config_id": feedback_config_id,
        "feedback_evidence_boundary": (
            "within_run_loss_guard_not_counterfactual_efficacy"
        ),
    }
    study_id = _sha256_id(scientific_spec)
    runs = [
        _build_feedback_run_plan(
            study_id=study_id,
            study_dir=resolved_study_dir,
            seed=seed,
            arm=arm,
            python_executable=resolved_python,
            bridge_script=resolved_bridge,
            base_args=normalized_args,
            feedback_config=resolved_feedback_config,
            min_free_disk_gb=min_free_disk_gb,
        )
        for seed in normalized_seeds
        for arm in HF_ZSPACE_FEEDBACK_STUDY_ARMS
    ]
    return {
        "schema": HF_ZSPACE_FEEDBACK_STUDY_SCHEMA,
        "row_type": "hf_zspace_feedback_study_plan",
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
            "run_order": "seed_then_observe_unguarded_guarded",
            "resume_policy": "verified_run_card_and_journal_receipt_only",
            "failure_policy": "preserve_and_fail_closed",
        },
        "artifacts": {
            "plan": str(resolved_study_dir / HF_ZSPACE_FEEDBACK_STUDY_PLAN_FILENAME),
            "events": str(
                resolved_study_dir / HF_ZSPACE_FEEDBACK_STUDY_EVENTS_FILENAME
            ),
            "summary": str(
                resolved_study_dir / HF_ZSPACE_FEEDBACK_STUDY_SUMMARY_FILENAME
            ),
            "feedback_report": str(
                resolved_study_dir / HF_ZSPACE_FEEDBACK_STUDY_REPORT_FILENAME
            ),
        },
        "run_count": len(runs),
        "runs": runs,
    }


def _persist_plan(plan: Mapping[str, object]) -> Path:
    study_dir = Path(str(plan["study_dir"]))
    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get("plan"), str):
        raise HFZSpaceFactorizedStudyError("study plan has no plan artifact path")
    path = Path(str(artifacts["plan"]))
    if path.parent != study_dir:
        raise HFZSpaceFactorizedStudyError(
            "study plan artifact must remain inside the study directory"
        )
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
    expected_mode = str(
        run.get(
            "expected_mode",
            "observe" if expected_arm == "observe" else "apply",
        )
    )
    receipt_arm = str(
        run.get(
            "expected_trajectory_arm",
            "raw" if expected_arm == "observe" else expected_arm,
        )
    )
    expected_feedback = str(run.get("expected_feedback_mode", "off"))
    if (
        receipt.get("mode") != expected_mode
        or receipt.get("trajectory_arm") != receipt_arm
        or receipt.get("feedback_mode", "off") != expected_feedback
    ):
        raise HFZSpaceFactorizedStudyError(
            "run card optimizer arm does not match the study"
        )
    receipt_recipe = receipt.get("recipe")
    feedback_config = (
        receipt_recipe.get("feedback_config")
        if isinstance(receipt_recipe, Mapping)
        else None
    )
    expected_feedback_config_id = run.get("expected_feedback_config_id")
    observed_feedback_config_id = (
        _sha256_id(dict(feedback_config))
        if isinstance(feedback_config, Mapping)
        else None
    )
    if expected_feedback == "off":
        if feedback_config not in (None, {}):
            raise HFZSpaceFactorizedStudyError(
                "feedback-off run card unexpectedly contains feedback config"
            )
    elif (
        not isinstance(expected_feedback_config_id, str)
        or observed_feedback_config_id != expected_feedback_config_id
    ):
        raise HFZSpaceFactorizedStudyError(
            "run card feedback config does not match the immutable study plan"
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
    trajectory_policy_id = None
    if receipt_arm == "dose_preserving_complement":
        trajectory_policy_id = receipt.get("trajectory_policy_id")
        policy_path_value = receipt.get("trajectory_policy_path")
        if (
            not isinstance(trajectory_policy_id, str)
            or not trajectory_policy_id.startswith("sha256:")
            or receipt.get("trajectory_policy_validated") is not True
            or receipt.get("trajectory_policy_source_trajectory_id") != trajectory_id
            or not isinstance(policy_path_value, str)
        ):
            raise HFZSpaceFactorizedStudyError(
                "run card has no complete Rust trajectory policy evidence"
            )
        policy_path = Path(policy_path_value).resolve()
        output_path = Path(str(run["output_dir"])).resolve()
        if not policy_path.is_file() or not _path_is_within(policy_path, output_path):
            raise HFZSpaceFactorizedStudyError(
                "run card trajectory policy artifact is missing or misplaced"
            )
        if receipt.get("trajectory_policy_sha256") != _sha256_file(policy_path):
            raise HFZSpaceFactorizedStudyError(
                "run card trajectory policy artifact differs from its receipt"
            )
        policy_size = receipt.get("trajectory_policy_size_bytes")
        if (
            isinstance(policy_size, bool)
            or not isinstance(policy_size, int)
            or policy_size != policy_path.stat().st_size
        ):
            raise HFZSpaceFactorizedStudyError(
                "run card trajectory policy size differs from its receipt"
            )
        policy = _read_json(policy_path)
        if (
            policy.get("policy_id") != trajectory_policy_id
            or policy.get("source_trajectory_id") != trajectory_id
            or policy.get("policy_validated") is not True
        ):
            raise HFZSpaceFactorizedStudyError(
                "run card trajectory policy identity does not match its artifact"
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
        "trajectory_policy_id": trajectory_policy_id,
        "feedback_config_id": observed_feedback_config_id,
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
    if run.get("trajectory_owner", run.get("arm") == "observe") is True:
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
    if run.get("trajectory_owner", run.get("arm") == "observe") is True:
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
    summary_schema: str = HF_ZSPACE_FACTORIZED_STUDY_SUMMARY_SCHEMA,
    summary_row_type: str = "hf_zspace_factorized_study_summary",
    report_artifact_key: str = "factorized_report",
    report_prefix: str = "factorized",
) -> dict[str, object]:
    total_runs = int(plan["run_count"])
    completed = sum(
        1
        for value in run_statuses.values()
        if value.get("status") in {"completed", "recovered", "reused"}
    )
    report_path = Path(
        str(plan["artifacts"][report_artifact_key])  # type: ignore[index]
    )
    summary: dict[str, object] = {
        "schema": summary_schema,
        "row_type": summary_row_type,
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
        f"{report_prefix}_report": str(report_path),
        f"{report_prefix}_report_sha256": (
            _sha256_file(report_path) if report_path.is_file() else None
        ),
        f"{report_prefix}_status": (
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
    return _run_hf_zspace_optimizer_study_plan(
        plan,
        execute=execute,
        retry_failed=retry_failed,
        event_schema=HF_ZSPACE_FACTORIZED_STUDY_EVENT_SCHEMA,
        summary_schema=HF_ZSPACE_FACTORIZED_STUDY_SUMMARY_SCHEMA,
        summary_row_type="hf_zspace_factorized_study_summary",
        report_artifact_key="factorized_report",
        report_prefix="factorized",
        compare_cards=compare_hf_zspace_optimizer_factorized_run_cards,
        write_report=write_hf_zspace_optimizer_factorized_ablation_report,
    )


def _run_hf_zspace_optimizer_study_plan(
    plan: Mapping[str, object],
    *,
    execute: bool,
    retry_failed: bool,
    event_schema: str,
    summary_schema: str,
    summary_row_type: str,
    report_artifact_key: str,
    report_prefix: str,
    compare_cards: Callable[[Sequence[Path]], dict[str, object]],
    write_report: Callable[[Mapping[str, object], str | Path], str],
) -> dict[str, object]:
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

    def write_summary(
        *,
        status: str,
        run_statuses: Mapping[str, Mapping[str, object]],
        events: Sequence[Mapping[str, object]],
        factorized_report: Mapping[str, object] | None = None,
        identity_anchor: Mapping[str, str] | None = None,
        error: str | None = None,
    ) -> dict[str, object]:
        return _write_summary(
            summary_path,
            plan=plan,
            status=status,
            run_statuses=run_statuses,
            events=events,
            factorized_report=factorized_report,
            identity_anchor=identity_anchor,
            error=error,
            summary_schema=summary_schema,
            summary_row_type=summary_row_type,
            report_artifact_key=report_artifact_key,
            report_prefix=report_prefix,
        )

    with _study_lock(root, study_id):
        _persist_plan(plan)
        events = _load_events(
            event_path,
            study_id=study_id,
            event_schema=event_schema,
        )
        if not execute:
            if summary_path.is_file():
                summary = _read_json(summary_path)
                if summary.get("study_id") != study_id:
                    raise HFZSpaceFactorizedStudyError(
                        "study summary belongs to another immutable plan"
                    )
                return summary
            return write_summary(
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
            event_schema=event_schema,
        )
        completed_events = _completed_event_by_run(events)
        try:
            for raw_run in plan["runs"]:  # type: ignore[index]
                assert isinstance(raw_run, Mapping)
                run = {str(key): value for key, value in raw_run.items()}
                label = str(run["label"])
                run_id = str(run["run_id"])
                completion_event = completed_events.get(run_id)
                reuse_error: HFZSpaceFactorizedStudyError | None = None
                try:
                    reusable = _verify_reusable_run(run, completion_event)
                except HFZSpaceFactorizedStudyError as error:
                    # A completion receipt is immutable evidence. Failed or
                    # interrupted attempts may instead be preserved and retried.
                    if completion_event is not None or not retry_failed:
                        raise
                    reusable = None
                    reuse_error = error
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
                        event_schema=event_schema,
                    )
                    completed_events[run_id] = event
                    write_summary(
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
                            "validation_error": (
                                None if reuse_error is None else str(reuse_error)
                            ),
                        },
                        event_schema=event_schema,
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
                    event_schema=event_schema,
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
                        event_schema=event_schema,
                    )
                    run_statuses[label] = {
                        "status": "failed",
                        "returncode": returncode,
                        "log": run["log"],
                    }
                    return write_summary(
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
                    event_schema=event_schema,
                )
                completed_events[run_id] = event
                run_statuses[label] = {
                    "status": "completed",
                    "duration_seconds": duration,
                    "run_card_sha256": card_sha256,
                    **facts,
                }
                write_summary(
                    status="running",
                    run_statuses=run_statuses,
                    events=events,
                    identity_anchor=identity_anchor,
                )

            cards = [Path(str(run["run_card"])) for run in plan["runs"]]  # type: ignore[index]
            factorized_report = compare_cards(cards)
            report_path = Path(str(artifacts[report_artifact_key]))
            write_report(factorized_report, report_path)
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
                    f"{report_prefix}_report": str(report_path),
                    f"{report_prefix}_report_sha256": _sha256_file(report_path),
                    "matched_seed_count": factorized_report.get("matched_seed_count"),
                    "error_count": factorized_report.get("error_count"),
                },
                event_schema=event_schema,
            )
            return write_summary(
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
                event_schema=event_schema,
            )
            write_summary(
                status="failed",
                run_statuses=run_statuses,
                events=events,
                factorized_report=factorized_report,
                identity_anchor=identity_anchor,
                error=str(exc),
            )
            raise


def run_hf_zspace_optimizer_feedback_study(
    *,
    study_dir: str | Path,
    seeds: Sequence[int],
    bridge_args: Sequence[str],
    feedback_config: Mapping[str, object] | None = None,
    bridge_script: str | Path | None = None,
    python_executable: str | Path | None = None,
    launch_cwd: str | Path | None = None,
    min_free_disk_gb: float = 5.0,
    execute: bool = False,
    retry_failed: bool = False,
) -> dict[str, object]:
    """Plan or execute a fail-closed, resumable feedback ablation study."""

    plan = build_hf_zspace_optimizer_feedback_study_plan(
        study_dir=study_dir,
        seeds=seeds,
        bridge_args=bridge_args,
        feedback_config=feedback_config,
        bridge_script=bridge_script,
        python_executable=python_executable,
        launch_cwd=launch_cwd,
        min_free_disk_gb=min_free_disk_gb,
    )
    return _run_hf_zspace_optimizer_study_plan(
        plan,
        execute=execute,
        retry_failed=retry_failed,
        event_schema=HF_ZSPACE_FEEDBACK_STUDY_EVENT_SCHEMA,
        summary_schema=HF_ZSPACE_FEEDBACK_STUDY_SUMMARY_SCHEMA,
        summary_row_type="hf_zspace_feedback_study_summary",
        report_artifact_key="feedback_report",
        report_prefix="feedback",
        compare_cards=compare_hf_zspace_optimizer_feedback_run_cards,
        write_report=write_hf_zspace_optimizer_feedback_ablation_report,
    )


def run_hf_zspace_optimizer_polarity_study(
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
    """Plan or execute a fail-closed, dose-matched polarity study."""

    plan = build_hf_zspace_optimizer_polarity_study_plan(
        study_dir=study_dir,
        seeds=seeds,
        bridge_args=bridge_args,
        bridge_script=bridge_script,
        python_executable=python_executable,
        launch_cwd=launch_cwd,
        min_free_disk_gb=min_free_disk_gb,
    )
    return _run_hf_zspace_optimizer_study_plan(
        plan,
        execute=execute,
        retry_failed=retry_failed,
        event_schema=HF_ZSPACE_POLARITY_STUDY_EVENT_SCHEMA,
        summary_schema=HF_ZSPACE_POLARITY_STUDY_SUMMARY_SCHEMA,
        summary_row_type="hf_zspace_polarity_study_summary",
        report_artifact_key="polarity_report",
        report_prefix="polarity",
        compare_cards=compare_hf_zspace_optimizer_polarity_run_cards,
        write_report=write_hf_zspace_optimizer_polarity_ablation_report,
    )


def _split_control_gain(arguments: Sequence[str]) -> tuple[float, list[str]]:
    gains: list[float] = []
    base_arguments: list[str] = []
    index = 0
    while index < len(arguments):
        argument = str(arguments[index])
        if argument == _CONTROL_GAIN_FLAG:
            if index + 1 >= len(arguments):
                raise HFZSpaceFactorizedStudyError(
                    f"{_CONTROL_GAIN_FLAG} has no value in a study plan"
                )
            raw_gain = str(arguments[index + 1])
            index += 2
        elif argument.startswith(f"{_CONTROL_GAIN_FLAG}="):
            raw_gain = argument.split("=", 1)[1]
            index += 1
        else:
            base_arguments.append(argument)
            index += 1
            continue
        try:
            gain = float(raw_gain)
        except ValueError as exc:
            raise HFZSpaceFactorizedStudyError(
                f"{_CONTROL_GAIN_FLAG} is not numeric in a study plan"
            ) from exc
        if not math.isfinite(gain) or not 0.0 <= gain <= 1.0:
            raise HFZSpaceFactorizedStudyError(
                f"{_CONTROL_GAIN_FLAG} must be finite and in [0, 1]"
            )
        gains.append(gain)
    if len(gains) > 1:
        raise HFZSpaceFactorizedStudyError(
            f"a study plan contains repeated {_CONTROL_GAIN_FLAG} options"
        )
    return (1.0 if not gains else gains[0]), base_arguments


def _finite_report_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HFZSpaceFactorizedStudyError(f"{field} is not a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise HFZSpaceFactorizedStudyError(f"{field} is not a finite number")
    return numeric


def _load_completed_gain_study(study_dir: str | Path) -> dict[str, object]:
    root = Path(study_dir).expanduser().resolve()
    plan = _read_json(root / HF_ZSPACE_FACTORIZED_STUDY_PLAN_FILENAME)
    if plan.get("schema") != HF_ZSPACE_FACTORIZED_STUDY_SCHEMA:
        raise HFZSpaceFactorizedStudyError(f"unsupported study plan in {root}")
    scientific_spec = plan.get("scientific_spec")
    if not isinstance(scientific_spec, Mapping):
        raise HFZSpaceFactorizedStudyError(f"study plan has no scientific spec: {root}")
    study_id = plan.get("study_id")
    if not isinstance(study_id, str) or study_id != _sha256_id(scientific_spec):
        raise HFZSpaceFactorizedStudyError(f"study plan identity is invalid: {root}")
    if Path(str(plan.get("study_dir"))).resolve() != root:
        raise HFZSpaceFactorizedStudyError(
            f"study plan directory is inconsistent: {root}"
        )
    summary = _read_json(root / HF_ZSPACE_FACTORIZED_STUDY_SUMMARY_FILENAME)
    if (
        summary.get("schema") != HF_ZSPACE_FACTORIZED_STUDY_SUMMARY_SCHEMA
        or summary.get("study_id") != study_id
        or summary.get("status") != "ready"
        or summary.get("completed_run_count") != plan.get("run_count")
        or summary.get("remaining_run_count") != 0
    ):
        raise HFZSpaceFactorizedStudyError(f"study is not completely ready: {root}")
    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise HFZSpaceFactorizedStudyError(f"study plan has no artifacts: {root}")
    event_path = Path(str(artifacts.get("events"))).resolve()
    report_path = Path(str(artifacts.get("factorized_report"))).resolve()
    if event_path != (root / HF_ZSPACE_FACTORIZED_STUDY_EVENTS_FILENAME).resolve():
        raise HFZSpaceFactorizedStudyError(
            f"study journal path is inconsistent: {root}"
        )
    if report_path != (root / HF_ZSPACE_FACTORIZED_STUDY_REPORT_FILENAME).resolve():
        raise HFZSpaceFactorizedStudyError(f"study report path is inconsistent: {root}")
    events = _load_events(event_path, study_id=study_id)
    completed_events = [
        event for event in events if event.get("event_type") == "study_completed"
    ]
    if not completed_events or completed_events[-1].get("status") != "ready":
        raise HFZSpaceFactorizedStudyError(
            f"study journal has no ready completion receipt: {root}"
        )
    if summary.get("event_count") != len(events):
        raise HFZSpaceFactorizedStudyError(
            f"study summary journal count drifted: {root}"
        )
    report_sha256 = _sha256_file(report_path)
    if (
        summary.get("factorized_report_sha256") != report_sha256
        or completed_events[-1].get("factorized_report_sha256") != report_sha256
    ):
        raise HFZSpaceFactorizedStudyError(f"study report receipt drifted: {root}")
    report = _read_json(report_path)
    if (
        report.get("schema") != HF_ZSPACE_FACTORIZED_ABLATION_SCHEMA
        or report.get("status") != "ready"
        or report.get("error_count") != 0
    ):
        raise HFZSpaceFactorizedStudyError(
            f"factorized report is not completely ready: {root}"
        )
    arguments = scientific_spec.get("bridge_args")
    if not isinstance(arguments, Sequence) or isinstance(arguments, (str, bytes)):
        raise HFZSpaceFactorizedStudyError(
            f"study bridge arguments are invalid: {root}"
        )
    gain, base_arguments = _split_control_gain([str(value) for value in arguments])
    canonical_spec = {str(key): value for key, value in scientific_spec.items()}
    canonical_spec["bridge_args"] = base_arguments
    identity_anchor = summary.get("identity_anchor")
    if not isinstance(identity_anchor, Mapping) or any(
        not isinstance(identity_anchor.get(field), str)
        for field in (
            "execution_identity_id",
            "runtime_identity_id",
            "training_input_id",
        )
    ):
        raise HFZSpaceFactorizedStudyError(
            f"study identity anchor is incomplete: {root}"
        )
    seed_rows = report.get("factorized_seeds")
    if not isinstance(seed_rows, Sequence) or isinstance(seed_rows, (str, bytes)):
        raise HFZSpaceFactorizedStudyError(f"factorized seed rows are missing: {root}")
    baselines: dict[int, tuple[float, float]] = {}
    trajectory_ids: set[str] = set()
    for raw_row in seed_rows:
        if not isinstance(raw_row, Mapping) or raw_row.get("status") != "ready":
            raise HFZSpaceFactorizedStudyError(
                f"factorized seed row is not ready: {root}"
            )
        seed = raw_row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed in baselines:
            raise HFZSpaceFactorizedStudyError(
                f"factorized seed identity is invalid: {root}"
            )
        before = raw_row.get("eval_before_losses")
        after = raw_row.get("eval_after_losses")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise HFZSpaceFactorizedStudyError(
                f"factorized baseline is missing: {root}"
            )
        baselines[seed] = (
            _finite_report_number(
                before.get("observe"), field=f"seed {seed} observe before loss"
            ),
            _finite_report_number(
                after.get("observe"), field=f"seed {seed} observe after loss"
            ),
        )
        trajectory_id = raw_row.get("trajectory_id")
        if not isinstance(trajectory_id, str) or not trajectory_id.startswith(
            "sha256:"
        ):
            raise HFZSpaceFactorizedStudyError(
                f"factorized seed trajectory identity is invalid: {root}"
            )
        trajectory_ids.add(trajectory_id)
    if report.get("matched_seed_count") != len(baselines):
        raise HFZSpaceFactorizedStudyError(f"factorized seed count drifted: {root}")
    reported_seeds = report.get("seeds")
    if (
        not isinstance(reported_seeds, Sequence)
        or isinstance(reported_seeds, (str, bytes))
        or list(reported_seeds) != list(baselines)
    ):
        raise HFZSpaceFactorizedStudyError(
            f"factorized seed ordering is inconsistent: {root}"
        )
    contrasts = report.get("contrasts")
    if not isinstance(contrasts, Mapping):
        raise HFZSpaceFactorizedStudyError(f"factorized contrasts are missing: {root}")
    return {
        "root": str(root),
        "gain": gain,
        "study_id": study_id,
        "report_sha256": report_sha256,
        "canonical_spec": canonical_spec,
        "identity_anchor": {str(key): value for key, value in identity_anchor.items()},
        "baselines": baselines,
        "seeds": list(baselines),
        "trajectory_ids": sorted(trajectory_ids),
        "contrasts": {str(key): value for key, value in contrasts.items()},
    }


def _linear_gain_fit(points: Sequence[tuple[float, float]]) -> dict[str, float]:
    count = len(points)
    mean_x = sum(point[0] for point in points) / count
    mean_y = sum(point[1] for point in points) / count
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator <= 0.0:
        raise HFZSpaceFactorizedStudyError("gain response has no gain variance")
    slope = (
        sum((gain - mean_x) * (value - mean_y) for gain, value in points) / denominator
    )
    intercept = mean_y - slope * mean_x
    residual = sum((value - (intercept + slope * gain)) ** 2 for gain, value in points)
    total = sum((value - mean_y) ** 2 for _, value in points)
    if total == 0.0:
        r_squared = 1.0 if residual <= 1.0e-30 else 0.0
    else:
        r_squared = 1.0 - residual / total
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
    }


def compare_hf_zspace_optimizer_factorized_gain_studies(
    study_dirs: Sequence[str | Path],
) -> dict[str, object]:
    """Compare completed factorized studies whose only scientific change is gain."""

    if len(study_dirs) < 3:
        raise HFZSpaceFactorizedStudyError(
            "gain response requires at least three completed study directories"
        )
    snapshots = sorted(
        (_load_completed_gain_study(path) for path in study_dirs),
        key=lambda value: float(value["gain"]),
    )
    gains = [float(snapshot["gain"]) for snapshot in snapshots]
    if len(set(gains)) != len(gains):
        raise HFZSpaceFactorizedStudyError("gain response study gains must be unique")
    reference_spec = snapshots[0]["canonical_spec"]
    reference_anchor = snapshots[0]["identity_anchor"]
    reference_baselines = snapshots[0]["baselines"]
    reference_seeds = snapshots[0]["seeds"]
    for snapshot in snapshots[1:]:
        if snapshot["canonical_spec"] != reference_spec:
            raise HFZSpaceFactorizedStudyError(
                "gain studies differ in scientific inputs beyond control gain"
            )
        if snapshot["identity_anchor"] != reference_anchor:
            raise HFZSpaceFactorizedStudyError(
                "gain studies disagree on execution, runtime, or training input identity"
            )
        if snapshot["baselines"] != reference_baselines:
            raise HFZSpaceFactorizedStudyError(
                "gain studies do not reproduce the observe baseline exactly"
            )
        if snapshot["seeds"] != reference_seeds:
            raise HFZSpaceFactorizedStudyError(
                "gain studies disagree on factorized seed ordering"
            )
    responses: dict[str, object] = {}
    for contrast_name in _GAIN_RESPONSE_CONTRASTS:
        points: list[dict[str, object]] = []
        fit_points: list[tuple[float, float]] = []
        reference_left: str | None = None
        reference_right: str | None = None
        for snapshot in snapshots:
            contrasts = snapshot["contrasts"]
            assert isinstance(contrasts, Mapping)
            contrast = contrasts.get(contrast_name)
            if (
                not isinstance(contrast, Mapping)
                or contrast.get("lower_is_better") is not True
            ):
                raise HFZSpaceFactorizedStudyError(
                    f"gain study has no valid {contrast_name} contrast"
                )
            left_arm = contrast.get("left_arm")
            right_arm = contrast.get("right_arm")
            if not isinstance(left_arm, str) or not isinstance(right_arm, str):
                raise HFZSpaceFactorizedStudyError(
                    f"gain study has invalid {contrast_name} arms"
                )
            reference_left = left_arm if reference_left is None else reference_left
            reference_right = right_arm if reference_right is None else reference_right
            if left_arm != reference_left or right_arm != reference_right:
                raise HFZSpaceFactorizedStudyError(
                    f"gain studies disagree on {contrast_name} arm semantics"
                )
            mean = _finite_report_number(
                contrast.get("mean"), field=f"{contrast_name} mean"
            )
            raw_values = contrast.get("values")
            if not isinstance(raw_values, Sequence) or isinstance(
                raw_values, (str, bytes)
            ):
                raise HFZSpaceFactorizedStudyError(
                    f"gain study has no {contrast_name} seed values"
                )
            values = [
                _finite_report_number(value, field=f"{contrast_name} seed value")
                for value in raw_values
            ]
            if len(values) != len(reference_baselines):
                raise HFZSpaceFactorizedStudyError(
                    f"gain study {contrast_name} seed count drifted"
                )
            observed_mean = sum(values) / len(values)
            if not math.isclose(mean, observed_mean, rel_tol=1.0e-12, abs_tol=1.0e-15):
                raise HFZSpaceFactorizedStudyError(
                    f"gain study {contrast_name} mean differs from its seed values"
                )
            gain = float(snapshot["gain"])
            fit_points.append((gain, mean))
            points.append(
                {
                    "gain": gain,
                    "mean": mean,
                    "values": values,
                    "bounded_trend_direction": contrast.get("bounded_trend_direction"),
                }
            )
        means = [point[1] for point in fit_points]
        responses[contrast_name] = {
            "left_arm": reference_left,
            "right_arm": reference_right,
            "lower_is_better": True,
            "points": points,
            "ordinary_least_squares": _linear_gain_fit(fit_points),
            "mean_is_monotonic_non_decreasing": all(
                left <= right for left, right in zip(means, means[1:])
            ),
            "mean_is_monotonic_non_increasing": all(
                left >= right for left, right in zip(means, means[1:])
            ),
            "right_arm_better_at_every_gain": all(mean > 0.0 for mean in means),
            "left_arm_better_at_every_gain": all(mean < 0.0 for mean in means),
        }
    harm_trend = all(
        isinstance(response, Mapping)
        and response.get("mean_is_monotonic_non_decreasing") is True
        and response.get("right_arm_better_at_every_gain") is True
        and isinstance(response.get("ordinary_least_squares"), Mapping)
        and float(response["ordinary_least_squares"]["slope"]) > 0.0
        and float(response["ordinary_least_squares"]["r_squared"]) >= 0.95
        for response in responses.values()
    )
    improvement_trend = all(
        isinstance(response, Mapping)
        and response.get("mean_is_monotonic_non_increasing") is True
        and response.get("left_arm_better_at_every_gain") is True
        and isinstance(response.get("ordinary_least_squares"), Mapping)
        and float(response["ordinary_least_squares"]["slope"]) < 0.0
        and float(response["ordinary_least_squares"]["r_squared"]) >= 0.95
        for response in responses.values()
    )
    baseline_rows = [
        {
            "seed": seed,
            "eval_before_loss": losses[0],
            "eval_after_loss": losses[1],
        }
        for seed, losses in sorted(reference_baselines.items())
    ]
    source_studies = [
        {
            "gain": snapshot["gain"],
            "study_id": snapshot["study_id"],
            "factorized_report_sha256": snapshot["report_sha256"],
            "trajectory_ids": snapshot["trajectory_ids"],
        }
        for snapshot in snapshots
    ]
    identity_payload = {
        "schema": HF_ZSPACE_FACTORIZED_GAIN_RESPONSE_SCHEMA,
        "shared_scientific_spec_id": _sha256_id(reference_spec),
        "shared_identity_anchor": reference_anchor,
        "source_studies": source_studies,
        "contrasts": responses,
    }
    return {
        **identity_payload,
        "row_type": "hf_zspace_factorized_gain_response",
        "status": "ready",
        "gain_response_id": _sha256_id(identity_payload),
        "gain_count": len(gains),
        "gains": gains,
        "matched_seed_count": len(reference_baselines),
        "seeds": sorted(reference_baselines),
        "observe_baseline_exact_match": True,
        "observe_baselines": baseline_rows,
        "bounded_gain_correlated_loss_degradation_observed": harm_trend,
        "bounded_gain_correlated_loss_improvement_observed": improvement_trend,
        "bounded_improvement_observed": improvement_trend,
        "efficacy_claim_ready": False,
        "evidence_scope": "single_model_single_corpus_multi_seed_gain_response",
        "evidence_boundary": (
            "the matched studies establish a gain-correlated validation-loss response "
            "for one GPT-2 LoRA recipe; they do not establish statistical significance, "
            "mechanistic causality beyond the controlled optimizer path, or generality"
        ),
        "efficacy_claim_requirements": (
            "a prespecified, adequately powered multi-model evaluation with held-out "
            "quality and stability metrics remains required"
        ),
    }


def write_hf_zspace_optimizer_factorized_gain_response_report(
    report: Mapping[str, object],
    path: str | Path,
) -> str:
    """Write one deterministic factorized gain-response report."""

    output = Path(path)
    _atomic_write_json(output, report)
    return str(output)


__all__ = [
    "HF_ZSPACE_FACTORIZED_GAIN_RESPONSE_FILENAME",
    "HF_ZSPACE_FACTORIZED_GAIN_RESPONSE_SCHEMA",
    "HF_ZSPACE_FACTORIZED_STUDY_ARMS",
    "HF_ZSPACE_FACTORIZED_STUDY_EVENT_SCHEMA",
    "HF_ZSPACE_FACTORIZED_STUDY_REPORT_FILENAME",
    "HF_ZSPACE_FACTORIZED_STUDY_SCHEMA",
    "HF_ZSPACE_FACTORIZED_STUDY_SUMMARY_SCHEMA",
    "HF_ZSPACE_POLARITY_STUDY_ARMS",
    "HF_ZSPACE_POLARITY_STUDY_EVENT_SCHEMA",
    "HF_ZSPACE_POLARITY_STUDY_REPORT_FILENAME",
    "HF_ZSPACE_POLARITY_STUDY_SCHEMA",
    "HF_ZSPACE_POLARITY_STUDY_SUMMARY_SCHEMA",
    "HFZSpaceFactorizedStudyError",
    "build_hf_zspace_optimizer_factorized_study_plan",
    "build_hf_zspace_optimizer_polarity_study_plan",
    "compare_hf_zspace_optimizer_factorized_gain_studies",
    "run_hf_zspace_optimizer_factorized_study",
    "run_hf_zspace_optimizer_polarity_study",
    "write_hf_zspace_optimizer_factorized_gain_response_report",
]
