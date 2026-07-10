"""Audited recovery operations for Hugging Face adapter executors."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from ._process import local_pid_alive
from .hf_adapter_executor import (
    HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME,
    HF_ADAPTER_CONTINUATION_EXECUTOR_OUTPUT_RESOLUTION_SCHEMA,
    _executor_lock,
    _executor_lock_owner_is_stale,
    _load_executor_lock,
    _write_state,
    load_hf_adapter_continuation_executor,
)

__all__ = [
    "HF_ADAPTER_CONTINUATION_EXECUTOR_OUTPUT_RESOLUTION_SCHEMA",
    "HF_ADAPTER_CONTINUATION_EXECUTOR_QUARANTINE_SUFFIX",
    "hf_adapter_continuation_executor_output_quarantine_report",
    "hf_adapter_continuation_executor_output_resolution_lines",
    "quarantine_hf_adapter_continuation_executor_output",
]


HF_ADAPTER_CONTINUATION_EXECUTOR_QUARANTINE_SUFFIX = ".executor-quarantine"

_RESOLVABLE_ATTEMPT_STATUSES = {
    "cancelled",
    "failed",
    "postflight_failed",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path(
    report_or_path: Mapping[str, object] | str | Path,
) -> Path:
    value = (
        report_or_path.get("state_path")
        if isinstance(report_or_path, Mapping)
        else report_or_path
    )
    if value is None:
        raise ValueError("executor state_path is required for output recovery")
    expanded = Path(str(value)).expanduser()
    if expanded.is_symlink():
        raise ValueError(f"executor state cannot be a symbolic link: {expanded}")
    path = expanded.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_reason(value: str) -> str:
    raw = str(value)
    if any(ord(character) < 32 or ord(character) == 127 for character in raw):
        raise ValueError("quarantine reason must not contain control characters")
    reason = raw.strip()
    if not reason:
        raise ValueError("quarantine reason must not be empty")
    if len(reason) > 512:
        raise ValueError("quarantine reason must be at most 512 characters")
    return reason


def _output_root(state: Mapping[str, object]) -> Path:
    value = state.get("output_root")
    if value is None:
        raise ValueError("executor state is missing output_root")
    expanded = Path(str(value)).expanduser()
    if expanded.is_symlink():
        raise ValueError(f"executor output_root cannot be a symlink: {expanded}")
    root = expanded.resolve()
    if not root.is_dir():
        raise ValueError(f"executor output_root is not a real directory: {root}")
    return root


def _attempt(
    state: Mapping[str, object],
    attempt_id: str,
) -> tuple[dict[str, object], int]:
    normalized = str(attempt_id).strip()
    if not normalized:
        raise ValueError("attempt_id must not be empty")
    matches = [
        (row, index)
        for index, row in enumerate(state.get("generations") or [])
        if isinstance(row, dict) and row.get("attempt_id") == normalized
    ]
    if not matches:
        raise ValueError(f"executor attempt was not found: {normalized}")
    if len(matches) != 1:
        raise ValueError(f"executor state contains duplicate attempt_id: {normalized}")
    return matches[0]


def _run_id(state: Mapping[str, object]) -> str:
    value = state.get("run_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("executor state is missing a valid run_id")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("executor run_id must not contain control characters")
    return value


def _tree_snapshot(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    file_count = 0
    directory_count = 0
    symlink_count = 0
    special_count = 0
    total_file_bytes = 0
    entries = [path, *sorted(path.rglob("*"), key=lambda item: item.as_posix())]
    for entry in entries:
        relative = "." if entry == path else entry.relative_to(path).as_posix()
        metadata = entry.lstat()
        mode = metadata.st_mode
        if stat.S_ISREG(mode):
            kind = "file"
            file_count += 1
            total_file_bytes += metadata.st_size
            target = ""
        elif stat.S_ISDIR(mode):
            kind = "directory"
            directory_count += 1
            target = ""
        elif stat.S_ISLNK(mode):
            kind = "symlink"
            symlink_count += 1
            target = os.readlink(entry)
        else:
            kind = "special"
            special_count += 1
            target = ""
        digest.update(
            json.dumps(
                [relative, kind, stat.S_IMODE(mode), metadata.st_size, target],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return {
        "digest_kind": "sha256_tree_metadata_v1",
        "metadata_sha256": digest.hexdigest(),
        "entry_count": len(entries),
        "file_count": file_count,
        "directory_count": directory_count,
        "symlink_count": symlink_count,
        "special_count": special_count,
        "total_file_bytes": total_file_bytes,
    }


_TREE_SNAPSHOT_COUNT_FIELDS = (
    "entry_count",
    "file_count",
    "directory_count",
    "symlink_count",
    "special_count",
    "total_file_bytes",
)


def _recorded_tree_snapshot(
    value: object,
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} tree snapshot is missing")
    snapshot = dict(value)
    digest = snapshot.get("metadata_sha256")
    if (
        snapshot.get("digest_kind") != "sha256_tree_metadata_v1"
        or not isinstance(digest, str)
        or len(digest) != 64
    ):
        raise RuntimeError(f"{label} tree snapshot is invalid")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise RuntimeError(f"{label} tree snapshot is invalid") from exc
    for field in _TREE_SNAPSHOT_COUNT_FIELDS:
        count = snapshot.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise RuntimeError(f"{label} tree snapshot is invalid")
    if snapshot["special_count"] != 0:
        raise RuntimeError(f"{label} contains unsupported special filesystem entries")
    if snapshot["directory_count"] < 1 or snapshot["entry_count"] != sum(
        snapshot[field]
        for field in (
            "file_count",
            "directory_count",
            "symlink_count",
            "special_count",
        )
    ):
        raise RuntimeError(f"{label} tree snapshot is invalid")
    return snapshot


def _safe_tree_snapshot(path: Path, *, label: str) -> dict[str, object]:
    return _recorded_tree_snapshot(_tree_snapshot(path), label=label)


def _tree_snapshots_match(
    observed: Mapping[str, object],
    recorded: Mapping[str, object],
) -> bool:
    return all(
        observed.get(field) == recorded.get(field)
        for field in (
            "digest_kind",
            "metadata_sha256",
            *_TREE_SNAPSHOT_COUNT_FIELDS,
        )
    )


def _quarantine_root(output_root: Path) -> Path:
    return output_root.with_name(
        f"{output_root.name}{HF_ADAPTER_CONTINUATION_EXECUTOR_QUARANTINE_SUFFIX}"
    )


def _destination_path(
    state: Mapping[str, object],
    attempt: Mapping[str, object],
    source: Path,
    quarantine_root: Path,
) -> Path:
    identity = hashlib.sha256(
        (f"{_run_id(state)}\0{attempt.get('attempt_id')}\0{source}").encode(
            "utf-8"
        )
    ).hexdigest()[:24]
    return quarantine_root / f"{source.name}-{identity}"


def _resolution_id(destination: Path) -> str:
    return (
        "executor-output-resolution-"
        + hashlib.sha256(str(destination).encode("utf-8")).hexdigest()[:24]
    )


def _claimed_by_later_promotion(
    state: Mapping[str, object],
    *,
    attempt_index: int,
    source: Path,
) -> bool:
    generations = state.get("generations") or []
    return any(
        isinstance(row, Mapping)
        and row.get("status") in {"promoted", "promoted_recovered"}
        and row.get("output_dir") is not None
        and Path(str(row.get("output_dir"))).expanduser().resolve() == source
        for row in generations[attempt_index + 1 :]
    )


def _resolution_plan(
    state: Mapping[str, object],
    *,
    state_path: Path,
    attempt_id: str,
) -> dict[str, object]:
    output_root = _output_root(state)
    attempt, attempt_index = _attempt(state, attempt_id)
    output_value = attempt.get("output_dir")
    if output_value is None:
        raise ValueError("executor attempt is missing output_dir")
    expanded_source = Path(str(output_value)).expanduser()
    if expanded_source.is_symlink():
        raise ValueError(
            f"executor attempt output cannot be a symlink: {expanded_source}"
        )
    source = expanded_source.resolve()
    if source == output_root or source.parent != output_root:
        raise ValueError("executor attempt output must be a direct output_root child")
    if state_path == source or source in state_path.parents:
        raise ValueError("executor state_path cannot be inside the attempt output")
    claimed_by_later_promotion = _claimed_by_later_promotion(
        state,
        attempt_index=attempt_index,
        source=source,
    )

    quarantine_root = _quarantine_root(output_root)
    if quarantine_root.is_symlink():
        raise RuntimeError(
            f"executor quarantine root cannot be a symlink: {quarantine_root}"
        )
    if quarantine_root.exists() and not quarantine_root.is_dir():
        raise RuntimeError(
            f"executor quarantine root is not a directory: {quarantine_root}"
        )
    if state_path == quarantine_root or quarantine_root in state_path.parents:
        raise ValueError("executor state_path cannot be inside the quarantine root")
    destination = _destination_path(state, attempt, source, quarantine_root)
    if destination.is_symlink():
        raise RuntimeError(
            f"executor quarantine destination cannot be a symlink: {destination}"
        )
    existing = attempt.get("output_resolution")
    if existing is not None and not isinstance(existing, Mapping):
        raise RuntimeError("executor attempt output resolution must be an object")
    if isinstance(existing, Mapping):
        recorded_snapshot = _recorded_tree_snapshot(
            existing.get("tree_snapshot"),
            label="recorded quarantine output",
        )
        if (
            existing.get("row_type")
            != "hf_adapter_continuation_executor_output_resolution"
            or existing.get("schema")
            != HF_ADAPTER_CONTINUATION_EXECUTOR_OUTPUT_RESOLUTION_SCHEMA
            or existing.get("resolution_id") != _resolution_id(destination)
            or existing.get("run_id") != _run_id(state)
            or existing.get("attempt_id") != attempt.get("attempt_id")
            or existing.get("attempt_status") != attempt.get("status")
            or existing.get("lineage_depth") != attempt.get("lineage_depth")
            or existing.get("source_path") != str(source)
            or existing.get("destination_path") != str(destination)
            or existing.get("quarantine_root") != str(quarantine_root)
        ):
            raise RuntimeError("executor attempt has inconsistent output resolution")
        if _validate_reason(str(existing.get("reason") or "")) != existing.get("reason"):
            raise RuntimeError("executor output resolution reason is inconsistent")
        if (
            source.exists() and not claimed_by_later_promotion
        ) or not destination.is_dir():
            raise RuntimeError("executor output resolution artifacts are inconsistent")
        destination_snapshot = _safe_tree_snapshot(
            destination,
            label="quarantined output",
        )
        if not _tree_snapshots_match(destination_snapshot, recorded_snapshot):
            raise RuntimeError(
                "quarantined output no longer matches its recorded tree metadata"
            )
        return {
            "row_type": "hf_adapter_continuation_executor_output_quarantine",
            "schema": HF_ADAPTER_CONTINUATION_EXECUTOR_OUTPUT_RESOLUTION_SCHEMA,
            "status": "already_quarantined",
            "ready": True,
            "action": "resume_executor",
            "state_path": str(state_path),
            "output_root": str(output_root),
            "attempt_id": attempt.get("attempt_id"),
            "attempt_status": attempt.get("status"),
            "source_path": str(source),
            "destination_path": str(destination),
            "quarantine_root": str(quarantine_root),
            "source_snapshot": None,
            "destination_snapshot": destination_snapshot,
            "source_reused_by_later_promotion": claimed_by_later_promotion,
            "existing_resolution": dict(existing),
            "pending_resolution": None,
        }

    if claimed_by_later_promotion:
        raise RuntimeError("executor attempt output is claimed by a later promotion")
    if attempt.get("status") not in _RESOLVABLE_ATTEMPT_STATUSES:
        raise RuntimeError(
            "executor attempt is not cancelled, failed, or postflight_failed"
        )
    pending_state = state.get("pending_output_resolution")
    pending_attempt = attempt.get("pending_output_resolution")
    if (pending_state is None) != (pending_attempt is None):
        raise RuntimeError("executor output quarantine intent is inconsistent")
    pending = None
    if isinstance(pending_state, Mapping) and isinstance(pending_attempt, Mapping):
        if dict(pending_state) != dict(pending_attempt):
            raise RuntimeError("executor output quarantine intents differ")
        pending = dict(pending_state)
        recorded_snapshot = _recorded_tree_snapshot(
            pending.get("tree_snapshot"),
            label="pending quarantine output",
        )
        if (
            pending.get("schema")
            != HF_ADAPTER_CONTINUATION_EXECUTOR_OUTPUT_RESOLUTION_SCHEMA
            or pending.get("row_type")
            != "hf_adapter_continuation_executor_output_resolution_intent"
            or pending.get("resolution_id") != _resolution_id(destination)
            or pending.get("run_id") != _run_id(state)
            or pending.get("attempt_id") != attempt.get("attempt_id")
            or pending.get("attempt_status") != attempt.get("status")
            or pending.get("lineage_depth") != attempt.get("lineage_depth")
            or pending.get("source_path") != str(source)
            or pending.get("destination_path") != str(destination)
            or pending.get("quarantine_root") != str(quarantine_root)
        ):
            raise RuntimeError("executor output quarantine intent is invalid")
        if _validate_reason(str(pending.get("reason") or "")) != pending.get("reason"):
            raise RuntimeError("executor output quarantine intent reason is inconsistent")
    elif pending_state is not None or pending_attempt is not None:
        raise RuntimeError("executor output quarantine intent must be an object")

    source_exists = source.exists()
    destination_exists = destination.exists()
    if source_exists and not source.is_dir():
        raise RuntimeError(f"executor attempt output is not a directory: {source}")
    if destination_exists and not destination.is_dir():
        raise RuntimeError(f"executor quarantine destination is invalid: {destination}")
    if source_exists and destination_exists:
        raise RuntimeError("executor source and quarantine destination both exist")
    if not source_exists and not destination_exists:
        raise FileNotFoundError(source)
    if not source_exists and destination_exists and pending is None:
        raise RuntimeError(
            "executor quarantine destination has no matching durable intent"
        )
    source_snapshot = (
        _safe_tree_snapshot(source, label="executor attempt output")
        if source_exists
        else None
    )
    destination_snapshot = (
        _safe_tree_snapshot(destination, label="pending quarantined output")
        if destination_exists
        else None
    )
    if (
        pending is not None
        and source_snapshot is not None
        and not _tree_snapshots_match(source_snapshot, recorded_snapshot)
    ):
        raise RuntimeError("executor output changed after quarantine intent")
    if (
        pending is not None
        and destination_snapshot is not None
        and not _tree_snapshots_match(destination_snapshot, recorded_snapshot)
    ):
        raise RuntimeError("quarantined output changed after quarantine intent")
    return {
        "row_type": "hf_adapter_continuation_executor_output_quarantine",
        "schema": HF_ADAPTER_CONTINUATION_EXECUTOR_OUTPUT_RESOLUTION_SCHEMA,
        "status": (
            "quarantine_ready"
            if pending is None
            else "quarantine_resume_ready"
            if source_exists
            else "quarantine_adoption_ready"
        ),
        "ready": True,
        "action": "quarantine_output",
        "state_path": str(state_path),
        "output_root": str(output_root),
        "attempt_id": attempt.get("attempt_id"),
        "attempt_status": attempt.get("status"),
        "lineage_depth": attempt.get("lineage_depth"),
        "source_path": str(source),
        "source_exists": source_exists,
        "destination_path": str(destination),
        "destination_exists": destination_exists,
        "quarantine_root": str(quarantine_root),
        "source_snapshot": source_snapshot,
        "destination_snapshot": destination_snapshot,
        "existing_resolution": None,
        "pending_resolution": pending,
    }


def _executor_lock_report(output_root: Path) -> dict[str, object]:
    path = output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
    try:
        owner = _load_executor_lock(path)
    except RuntimeError as exc:
        return {
            "path": str(path),
            "status": "invalid",
            "owner": None,
            "error": f"{exc.__class__.__name__}: {exc}",
        }
    if owner is None:
        return {"path": str(path), "status": "absent", "owner": None, "error": None}
    hostname = owner.get("hostname")
    alive = (
        local_pid_alive(owner.get("pid")) if hostname == socket.gethostname() else None
    )
    status_value = (
        "stale"
        if _executor_lock_owner_is_stale(owner)
        else "active"
        if hostname == socket.gethostname() and alive is True
        else "unverified"
    )
    return {
        "path": str(path),
        "status": status_value,
        "owner": owner,
        "error": None,
    }


def hf_adapter_continuation_executor_output_quarantine_report(
    report_or_path: Mapping[str, object] | str | Path,
    *,
    attempt_id: str,
) -> dict[str, object]:
    """Plan one non-destructive output quarantine without mutating state."""

    state_path = _state_path(report_or_path)
    state = load_hf_adapter_continuation_executor(state_path)
    output_root = _output_root(state)
    lock = _executor_lock_report(output_root)
    if lock.get("status") not in {"absent", "stale"}:
        attempt, _ = _attempt(state, attempt_id)
        return {
            "row_type": "hf_adapter_continuation_executor_output_quarantine",
            "schema": HF_ADAPTER_CONTINUATION_EXECUTOR_OUTPUT_RESOLUTION_SCHEMA,
            "status": "executor_locked",
            "ready": False,
            "action": "wait_for_executor_exit",
            "state_path": str(state_path),
            "output_root": str(output_root),
            "attempt_id": attempt.get("attempt_id"),
            "attempt_status": attempt.get("status"),
            "executor_lock": lock,
        }
    plan = _resolution_plan(
        state,
        state_path=state_path,
        attempt_id=attempt_id,
    )
    plan["executor_lock"] = lock
    return plan


def quarantine_hf_adapter_continuation_executor_output(
    report_or_path: Mapping[str, object] | str | Path,
    *,
    attempt_id: str,
    reason: str = "operator_quarantine",
) -> dict[str, object]:
    """Atomically quarantine one failed output and persist recovery evidence."""

    normalized_reason = _validate_reason(reason)
    state_path = _state_path(report_or_path)
    initial = load_hf_adapter_continuation_executor(state_path)
    output_root = _output_root(initial)
    with _executor_lock(output_root):
        state = load_hf_adapter_continuation_executor(state_path)
        if _output_root(state) != output_root:
            raise RuntimeError("executor output_root changed during quarantine")
        plan = _resolution_plan(
            state,
            state_path=state_path,
            attempt_id=attempt_id,
        )
        existing = plan.get("existing_resolution")
        if isinstance(existing, Mapping):
            report = dict(existing)
            report.update(
                {
                    "created": False,
                    "state_path": str(state_path),
                    "executor_status": state.get("status"),
                }
            )
            return report

        attempt, _ = _attempt(state, attempt_id)
        source = Path(str(plan["source_path"]))
        destination = Path(str(plan["destination_path"]))
        quarantine_root = Path(str(plan["quarantine_root"]))
        history = state.setdefault("output_resolution_history", [])
        if not isinstance(history, list):
            raise ValueError("executor output_resolution_history must be a list")
        pending = plan.get("pending_resolution")
        if isinstance(pending, Mapping):
            intent = dict(pending)
        else:
            snapshot = plan.get("source_snapshot")
            if not isinstance(snapshot, Mapping):
                raise RuntimeError("executor output snapshot is unavailable")
            resolution_id = _resolution_id(destination)
            intent = {
                "row_type": "hf_adapter_continuation_executor_output_resolution_intent",
                "schema": HF_ADAPTER_CONTINUATION_EXECUTOR_OUTPUT_RESOLUTION_SCHEMA,
                "resolution_id": resolution_id,
                "prepared_at": _now(),
                "reason": normalized_reason,
                "run_id": _run_id(state),
                "invocation_count": state.get("invocation_count"),
                "attempt_id": attempt.get("attempt_id"),
                "attempt_status": attempt.get("status"),
                "lineage_depth": attempt.get("lineage_depth"),
                "source_path": str(source),
                "destination_path": str(destination),
                "quarantine_root": str(quarantine_root),
                "tree_snapshot": dict(snapshot),
            }
            state["pending_output_resolution"] = dict(intent)
            attempt["pending_output_resolution"] = dict(intent)
            _write_state(state_path, state)

        adopted = not source.exists() and destination.is_dir()
        if not adopted:
            if source.is_symlink() or not source.is_dir():
                raise RuntimeError("executor output changed before quarantine move")
            quarantine_root.mkdir(mode=0o700, parents=False, exist_ok=True)
            os.chmod(quarantine_root, 0o700)
            source.replace(destination)
        snapshot = _safe_tree_snapshot(destination, label="quarantined output")
        expected_snapshot = _recorded_tree_snapshot(
            intent["tree_snapshot"],
            label="pending quarantine output",
        )
        if not _tree_snapshots_match(snapshot, expected_snapshot):
            if not source.exists() and destination.exists():
                destination.replace(source)
            state.pop("pending_output_resolution", None)
            attempt.pop("pending_output_resolution", None)
            _write_state(state_path, state)
            raise RuntimeError("quarantined output metadata changed during move")

        resolution = {
            "row_type": "hf_adapter_continuation_executor_output_resolution",
            "schema": HF_ADAPTER_CONTINUATION_EXECUTOR_OUTPUT_RESOLUTION_SCHEMA,
            "resolution_id": intent["resolution_id"],
            "resolved_at": _now(),
            "reason": intent["reason"],
            "run_id": intent["run_id"],
            "invocation_count": intent["invocation_count"],
            "attempt_id": intent["attempt_id"],
            "attempt_status": intent["attempt_status"],
            "lineage_depth": intent["lineage_depth"],
            "source_path": str(source),
            "destination_path": str(destination),
            "quarantine_root": str(quarantine_root),
            "tree_snapshot": snapshot,
            "adopted_after_interrupted_write": adopted,
        }
        attempt["output_resolution"] = dict(resolution)
        attempt.pop("pending_output_resolution", None)
        history.append(dict(resolution))
        state.pop("pending_output_resolution", None)
        state["last_output_resolution"] = dict(resolution)
        state["status"] = "output_quarantined"
        state["action"] = "resume_executor"
        state["reason"] = "failed_generation_output_quarantined"
        state.pop("unresolved_generation", None)
        state.pop("output_resolution_gate", None)
        _write_state(state_path, state)
        report = dict(resolution)
        report.update(
            {
                "created": True,
                "state_path": str(state_path),
                "executor_status": state.get("status"),
            }
        )
        return report


def hf_adapter_continuation_executor_output_resolution_lines(
    report: Mapping[str, object],
) -> list[str]:
    """Render compact operator output for quarantine plans and resolutions."""

    return [
        "hf_adapter_continuation_executor_output_resolution "
        f"status={report.get('status', report.get('executor_status'))} "
        f"ready={report.get('ready')} "
        f"created={report.get('created')} "
        f"attempt={report.get('attempt_id')} "
        f"attempt_status={report.get('attempt_status')} "
        f"adopted={report.get('adopted_after_interrupted_write')} "
        f"source={report.get('source_path')} "
        f"destination={report.get('destination_path')} "
        f"state={report.get('state_path')}"
    ]
