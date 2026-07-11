"""Resumable execution loops for promoted Hugging Face PEFT adapters."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ._process import local_pid_alive, local_process_group_alive
from .hf_adapter import (
    HF_ADAPTER_LINEAGE_FILENAME,
    hf_adapter_promotion_chain_report,
)
from .hf_ft import (
    hf_finetune_scale_up_command,
    hf_finetune_scale_up_preflight_report,
)

__all__ = [
    "HF_ADAPTER_CONTINUATION_EXECUTOR_FILENAME",
    "HF_ADAPTER_CONTINUATION_EXECUTOR_CONTROL_DIRNAME",
    "HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME",
    "HF_ADAPTER_CONTINUATION_EXECUTOR_LOG_DIRNAME",
    "HF_ADAPTER_CONTINUATION_EXECUTOR_INTERRUPTION_CLAIM_SCHEMA",
    "HF_ADAPTER_CONTINUATION_EXECUTOR_OUTPUT_RESOLUTION_SCHEMA",
    "HF_ADAPTER_CONTINUATION_EXECUTOR_SCHEMA",
    "HF_ADAPTER_CONTINUATION_EXECUTOR_STOP_REQUEST_SCHEMA",
    "hf_adapter_continuation_executor_lines",
    "hf_adapter_continuation_executor_stop_request_lines",
    "load_hf_adapter_continuation_executor",
    "load_hf_adapter_continuation_executor_stop_request",
    "request_hf_adapter_continuation_executor_stop",
    "run_hf_adapter_continuation_executor",
]


HF_ADAPTER_CONTINUATION_EXECUTOR_SCHEMA = (
    "spiraltorch.hf_adapter_continuation_executor.v1"
)
HF_ADAPTER_CONTINUATION_EXECUTOR_FILENAME = (
    "spiraltorch-hf-adapter-continuation-executor.json"
)
HF_ADAPTER_CONTINUATION_EXECUTOR_CONTROL_DIRNAME = "executor-control"
HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME = (
    ".spiraltorch-hf-adapter-continuation-executor.lock"
)
HF_ADAPTER_CONTINUATION_EXECUTOR_LOG_DIRNAME = "executor-logs"
HF_ADAPTER_CONTINUATION_EXECUTOR_STOP_REQUEST_SCHEMA = (
    "spiraltorch.hf_adapter_continuation_executor_stop_request.v1"
)
HF_ADAPTER_CONTINUATION_EXECUTOR_OUTPUT_RESOLUTION_SCHEMA = (
    "spiraltorch.hf_adapter_continuation_executor_output_resolution.v1"
)
HF_ADAPTER_CONTINUATION_EXECUTOR_INTERRUPTION_CLAIM_SCHEMA = (
    "spiraltorch.hf_adapter_continuation_executor_interruption_claim.v1"
)
_PROCESS_PROGRESS_INTERVAL_SECONDS = 5.0
_PROCESS_STOP_POLL_INTERVAL_SECONDS = 0.2
_PROCESS_STOP_GRACE_SECONDS = 5.0


CommandRunner = Callable[[Sequence[str]], object]
ProcessStarted = Callable[[int], None]
ProcessProgress = Callable[[int], None]
ProcessStopRequested = Callable[[Mapping[str, object]], None]
StopRequestLoader = Callable[[], Mapping[str, object] | None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_paths(
    sources: str | Path | Sequence[str | Path],
) -> list[Path]:
    values = [sources] if isinstance(sources, (str, Path)) else list(sources)
    if not values:
        raise ValueError("at least one adapter chain source is required")
    return [Path(value).expanduser().resolve() for value in values]


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_executor_lock(path: Path) -> dict[str, object] | None:
    if path.is_symlink():
        raise RuntimeError(f"executor lock cannot be a symbolic link: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"executor lock is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"executor lock must contain a JSON object: {path}")
    return dict(payload)


def _executor_lock_owner_is_stale(owner: Mapping[str, object]) -> bool:
    return bool(
        owner.get("row_type") == "hf_adapter_continuation_executor_lock"
        and isinstance(owner.get("lock_id"), str)
        and owner.get("hostname") == socket.gethostname()
        and local_pid_alive(owner.get("pid")) is False
    )


def _reap_stale_executor_lock(path: Path) -> bool:
    reaper = path.with_name(f"{path.name}.reap")
    try:
        reaper.mkdir(mode=0o700)
    except FileExistsError:
        return False
    try:
        current = _load_executor_lock(path)
        if current is None or not _executor_lock_owner_is_stale(current):
            return False
        path.unlink()
        return True
    finally:
        reaper.rmdir()


@contextmanager
def _executor_lock(output_root: Path) -> Iterator[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
    owner = {
        "row_type": "hf_adapter_continuation_executor_lock",
        "lock_id": f"executor-lock-{uuid.uuid4().hex}",
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "created_at": _now(),
    }
    encoded = (json.dumps(owner, ensure_ascii=True, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    acquired = False
    for _ in range(8):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            existing = _load_executor_lock(path)
            if existing is None:
                continue
            if _executor_lock_owner_is_stale(existing):
                if not _reap_stale_executor_lock(path):
                    time.sleep(0.05)
                continue
            raise RuntimeError(
                "executor output_root is locked; inspect the recorded owner before retrying: "
                f"{path}"
            )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        acquired = True
        break
    if not acquired:
        raise RuntimeError(f"could not acquire executor lock: {path}")
    try:
        yield path
    finally:
        try:
            current = _load_executor_lock(path)
        except RuntimeError:
            current = None
        if current is not None and current.get("lock_id") == owner["lock_id"]:
            path.unlink(missing_ok=True)


def load_hf_adapter_continuation_executor(
    value: str | Path,
) -> dict[str, object]:
    path = Path(value).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"executor state must contain a JSON object: {path}")
    if payload.get("schema") != HF_ADAPTER_CONTINUATION_EXECUTOR_SCHEMA:
        raise ValueError(
            "unsupported HF adapter continuation executor schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("row_type") != "hf_adapter_continuation_executor":
        raise ValueError(
            "unsupported HF adapter continuation executor row type: "
            f"{payload.get('row_type')}"
        )
    report = dict(payload)
    report["state_path"] = str(path.resolve())
    return report


def _stop_request_path(
    output_root: Path,
    *,
    run_id: str,
    invocation_count: int,
) -> Path:
    identity = hashlib.sha256(
        f"{run_id}\0{invocation_count}".encode("utf-8")
    ).hexdigest()[:24]
    return (
        output_root
        / HF_ADAPTER_CONTINUATION_EXECUTOR_CONTROL_DIRNAME
        / f"stop-{invocation_count:06d}-{identity}.json"
    )


def _state_stop_request_path(state: Mapping[str, object]) -> Path:
    output_root_value = state.get("output_root")
    run_id = state.get("run_id")
    invocation_count = state.get("invocation_count")
    if output_root_value is None or not isinstance(run_id, str) or not run_id:
        raise ValueError("executor state is missing stop-request identity")
    if (
        isinstance(invocation_count, bool)
        or not isinstance(invocation_count, int)
        or invocation_count <= 0
    ):
        raise ValueError("executor state has an invalid invocation_count")
    return _stop_request_path(
        Path(str(output_root_value)).expanduser().resolve(),
        run_id=run_id,
        invocation_count=invocation_count,
    )


def _exclusive_write_json(path: Path, payload: Mapping[str, object]) -> None:
    control_dir = path.parent
    if control_dir.is_symlink():
        raise RuntimeError(
            f"executor control directory cannot be a symlink: {control_dir}"
        )
    control_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def load_hf_adapter_continuation_executor_stop_request(
    value: str | Path,
) -> dict[str, object]:
    """Load and validate one durable executor stop request."""

    path = Path(value).expanduser()
    if path.is_symlink():
        raise ValueError(f"executor stop request cannot be a symlink: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"executor stop request must contain a JSON object: {path}")
    if payload.get("schema") != HF_ADAPTER_CONTINUATION_EXECUTOR_STOP_REQUEST_SCHEMA:
        raise ValueError(
            "unsupported HF adapter continuation executor stop-request schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("row_type") != "hf_adapter_continuation_executor_stop_request":
        raise ValueError(
            "unsupported HF adapter continuation executor stop-request row type: "
            f"{payload.get('row_type')}"
        )
    report = dict(payload)
    report["request_path"] = str(path.resolve())
    return report


def _matching_stop_request(
    path: Path,
    *,
    run_id: object,
    invocation_count: object,
) -> dict[str, object] | None:
    try:
        request = load_hf_adapter_continuation_executor_stop_request(path)
    except FileNotFoundError:
        return None
    if (
        request.get("run_id") != run_id
        or request.get("invocation_count") != invocation_count
    ):
        raise ValueError("executor stop request targets a different invocation")
    return request


def request_hf_adapter_continuation_executor_stop(
    report_or_path: Mapping[str, object] | str | Path,
    *,
    reason: str = "operator_requested",
) -> dict[str, object]:
    """Request a cooperative stop without signalling an unverified PID."""

    state = (
        dict(report_or_path)
        if isinstance(report_or_path, Mapping)
        else load_hf_adapter_continuation_executor(report_or_path)
    )
    raw_reason = str(reason)
    if any(ord(character) < 32 or ord(character) == 127 for character in raw_reason):
        raise ValueError("stop reason must not contain control characters")
    normalized_reason = raw_reason.strip()
    if not normalized_reason:
        raise ValueError("stop reason must not be empty")
    if len(normalized_reason) > 512:
        raise ValueError("stop reason must be at most 512 characters")
    path = _state_stop_request_path(state)
    existing = _matching_stop_request(
        path,
        run_id=state.get("run_id"),
        invocation_count=state.get("invocation_count"),
    )
    if existing is not None:
        report = dict(existing)
        report["created"] = False
        return report
    if state.get("status") not in {"auditing", "running"}:
        raise RuntimeError(
            "executor is not running; no cooperative stop request was written"
        )
    execution = state.get("execution")
    lock_value = execution.get("lock_path") if isinstance(execution, Mapping) else None
    lock_path = None if lock_value is None else Path(str(lock_value)).expanduser()
    lock_owner = None if lock_path is None else _load_executor_lock(lock_path)
    if lock_owner is None:
        raise RuntimeError(
            "executor single-writer lock is absent; inspect interrupted state instead"
        )
    if (
        lock_owner.get("row_type") != "hf_adapter_continuation_executor_lock"
        or not isinstance(lock_owner.get("lock_id"), str)
        or not lock_owner.get("lock_id")
        or not isinstance(lock_owner.get("hostname"), str)
        or not lock_owner.get("hostname")
        or isinstance(lock_owner.get("pid"), bool)
        or not isinstance(lock_owner.get("pid"), int)
        or int(lock_owner.get("pid")) <= 0
    ):
        raise RuntimeError("executor single-writer lock owner is invalid")
    if (
        lock_owner.get("hostname") == socket.gethostname()
        and local_pid_alive(lock_owner.get("pid")) is not True
    ):
        raise RuntimeError(
            "executor single-writer lock owner is not alive; inspect interrupted state"
        )
    running = [
        row
        for row in state.get("generations") or []
        if isinstance(row, Mapping) and row.get("status") == "running"
    ]
    if len(running) > 1:
        raise RuntimeError("executor state contains multiple running attempts")
    active = running[-1] if running else None
    payload = {
        "row_type": "hf_adapter_continuation_executor_stop_request",
        "schema": HF_ADAPTER_CONTINUATION_EXECUTOR_STOP_REQUEST_SCHEMA,
        "request_id": f"executor-stop-{uuid.uuid4().hex}",
        "requested_at": _now(),
        "requested_by_pid": os.getpid(),
        "requested_by_hostname": socket.gethostname(),
        "reason": normalized_reason,
        "run_id": state.get("run_id"),
        "invocation_count": state.get("invocation_count"),
        "attempt_id": None if active is None else active.get("attempt_id"),
        "attempt_pid": None if active is None else active.get("pid"),
        "attempt_hostname": None if active is None else active.get("hostname"),
        "state_path": state.get("state_path"),
    }
    created = True
    try:
        _exclusive_write_json(path, payload)
    except FileExistsError:
        existing = _matching_stop_request(
            path,
            run_id=state.get("run_id"),
            invocation_count=state.get("invocation_count"),
        )
        if existing is None:
            raise RuntimeError("executor stop request disappeared during inspection")
        payload = existing
        created = False
    report = dict(payload)
    report["request_path"] = str(path.resolve())
    report["created"] = created
    return report


def hf_adapter_continuation_executor_stop_request_lines(
    report_or_path: Mapping[str, object] | str | Path,
) -> list[str]:
    report = (
        dict(report_or_path)
        if isinstance(report_or_path, Mapping)
        else load_hf_adapter_continuation_executor_stop_request(report_or_path)
    )
    return [
        "hf_adapter_continuation_executor_stop_request "
        f"created={report.get('created')} "
        f"run={report.get('run_id')} "
        f"invocation={report.get('invocation_count')} "
        f"attempt={report.get('attempt_id')} "
        f"reason={report.get('reason')} "
        f"request={report.get('request_path')}"
    ]


def _write_state(path: Path, state: dict[str, object]) -> None:
    state["state_path"] = str(path)
    state["updated_at"] = _now()
    state["generation_attempt_count"] = len(state.get("generations") or [])
    _atomic_write_json(path, state)


def _new_state(
    *,
    source_paths: Sequence[Path],
    output_root: Path,
    state_path: Path,
) -> dict[str, object]:
    created_at = _now()
    return {
        "row_type": "hf_adapter_continuation_executor",
        "schema": HF_ADAPTER_CONTINUATION_EXECUTOR_SCHEMA,
        "status": "initializing",
        "action": "audit_chain",
        "created_at": created_at,
        "updated_at": created_at,
        "run_id": f"hf-adapter-executor-{uuid.uuid4().hex}",
        "source_paths": [str(path) for path in source_paths],
        "output_root": str(output_root),
        "state_path": str(state_path),
        "invocation_count": 0,
        "generation_attempt_count": 0,
        "promoted_generation_count": 0,
        "generations": [],
        "policy_history": [],
    }


def _state_for_invocation(
    *,
    source_paths: Sequence[Path],
    output_root: Path,
    state_path: Path,
    policy: Mapping[str, object],
    scale_up: Mapping[str, object],
    execution: Mapping[str, object],
    run: bool,
    max_generations: int,
) -> dict[str, object]:
    state = (
        load_hf_adapter_continuation_executor(state_path)
        if state_path.is_file()
        else _new_state(
            source_paths=source_paths,
            output_root=output_root,
            state_path=state_path,
        )
    )
    expected_sources = [str(path) for path in source_paths]
    if state.get("source_paths") != expected_sources:
        raise ValueError("executor state source_paths differ; use a new state artifact")
    if state.get("output_root") != str(output_root):
        raise ValueError("executor state output_root differs; use a new state artifact")
    previous_policy = state.get("policy")
    if isinstance(previous_policy, Mapping) and dict(previous_policy) != dict(policy):
        history = state.setdefault("policy_history", [])
        if isinstance(history, list):
            history.append(
                {
                    "changed_at": _now(),
                    "previous": dict(previous_policy),
                    "current": dict(policy),
                }
            )
    state["policy"] = dict(policy)
    state["scale_up"] = dict(scale_up)
    state["execution"] = dict(execution)
    state["mode"] = "run" if run else "plan"
    state["max_generations_per_invocation"] = max_generations
    previous_stop_request = state.pop("stop_request", None)
    if isinstance(previous_stop_request, Mapping):
        history = state.setdefault("stop_request_history", [])
        if isinstance(history, list):
            history.append(dict(previous_stop_request))
    state["invocation_count"] = int(state.get("invocation_count") or 0) + 1
    state["invocation_started_at"] = _now()
    state["status"] = "auditing"
    state["action"] = "audit_chain"
    state.pop("failure", None)
    state.pop("reason", None)
    state.pop("completed_at", None)
    return state


def _chain_sources(
    source_paths: Sequence[Path],
    output_root: Path,
) -> list[Path]:
    values = list(source_paths)
    if output_root.is_dir() and any(output_root.rglob(HF_ADAPTER_LINEAGE_FILENAME)):
        values.append(output_root)
    return values


def _policy_kwargs(policy: Mapping[str, object]) -> dict[str, object]:
    return {
        "max_lineage_depth": policy.get("max_lineage_depth"),
        "target_eval_loss": policy.get("target_eval_loss"),
        "min_eval_improvement": policy.get("min_eval_improvement"),
        "plateau_patience": policy.get("plateau_patience", 1),
    }


def _audit_chain(
    *,
    source_paths: Sequence[Path],
    output_root: Path,
    recursive: bool,
    allow_inferred_roots: bool,
    select_adapter_id: str | None,
    command_artifacts: Sequence[Mapping[str, object] | str | Path] | None,
    policy: Mapping[str, object],
) -> dict[str, object]:
    return hf_adapter_promotion_chain_report(
        _chain_sources(source_paths, output_root),
        recursive=recursive,
        allow_inferred_roots=allow_inferred_roots,
        select_adapter_id=select_adapter_id,
        command_artifacts=command_artifacts,
        **_policy_kwargs(policy),
    )


def _node_for_path(
    chain: Mapping[str, object],
    adapter_path: Path,
) -> dict[str, object] | None:
    expected = adapter_path.resolve()
    for raw_node in chain.get("nodes") or []:
        if not isinstance(raw_node, Mapping):
            continue
        raw_path = raw_node.get("adapter_path")
        if raw_path is None:
            continue
        if Path(str(raw_path)).expanduser().resolve() == expected:
            return dict(raw_node)
    return None


def _selected_transition(
    chain: Mapping[str, object],
    *,
    child_adapter_id: object | None = None,
) -> dict[str, object] | None:
    target_adapter_id = (
        chain.get("selected_adapter_id")
        if child_adapter_id is None
        else child_adapter_id
    )
    matches = [
        dict(raw_transition)
        for raw_transition in chain.get("transitions") or []
        if isinstance(raw_transition, Mapping)
        and raw_transition.get("selected_path") is True
        and raw_transition.get("child_adapter_id") == target_adapter_id
    ]
    return matches[0] if len(matches) == 1 else None


def _record_chain_state(
    state: dict[str, object],
    chain: Mapping[str, object],
) -> None:
    state["chain_report"] = dict(chain)
    state["selected_adapter_id"] = chain.get("selected_adapter_id")
    state["selected_adapter_path"] = chain.get("selected_adapter_path")
    state["selected_lineage_depth"] = chain.get("selected_lineage_depth")
    state["continuation_policy"] = chain.get("continuation_policy")
    state["transition_count"] = chain.get("transition_count")
    state["ready_transition_count"] = chain.get("ready_transition_count")
    state["selected_path_transition_count"] = chain.get(
        "selected_path_transition_count"
    )
    state["selected_path_transitions_ready"] = chain.get(
        "selected_path_transitions_ready"
    )
    state["selected_transition"] = _selected_transition(chain)


def _postflight_report(
    chain: Mapping[str, object],
    *,
    output_dir: Path,
    expected_parent_adapter_id: object,
    expected_lineage_depth: int,
) -> dict[str, object]:
    node = _node_for_path(chain, output_dir)
    transition = (
        None
        if node is None
        else _selected_transition(
            chain,
            child_adapter_id=node.get("adapter_id"),
        )
    )
    checks = [
        {
            "name": "output_node",
            "passed": node is not None,
            "observed": None if node is None else node.get("adapter_id"),
            "threshold": str(output_dir),
        },
        {
            "name": "chain_eligible",
            "passed": node is not None and node.get("chain_eligible") is True,
            "observed": None if node is None else node.get("chain_eligible"),
            "threshold": True,
        },
        {
            "name": "parent_adapter_id",
            "passed": (
                node is not None
                and node.get("parent_adapter_id") == expected_parent_adapter_id
            ),
            "observed": None if node is None else node.get("parent_adapter_id"),
            "threshold": expected_parent_adapter_id,
        },
        {
            "name": "lineage_depth",
            "passed": (
                node is not None and node.get("lineage_depth") == expected_lineage_depth
            ),
            "observed": None if node is None else node.get("lineage_depth"),
            "threshold": expected_lineage_depth,
        },
        {
            "name": "selected_tip",
            "passed": (
                node is not None
                and chain.get("selected_adapter_id") == node.get("adapter_id")
            ),
            "observed": chain.get("selected_adapter_id"),
            "threshold": None if node is None else node.get("adapter_id"),
        },
        {
            "name": "selected_transition",
            "passed": transition is not None,
            "observed": None
            if transition is None
            else transition.get("child_adapter_id"),
            "threshold": None if node is None else node.get("adapter_id"),
        },
        {
            "name": "transition_ready",
            "passed": (
                transition is not None
                and transition.get("transition_ready") is True
            ),
            "observed": None
            if transition is None
            else transition.get("transition_ready"),
            "threshold": True,
        },
        {
            "name": "transition_parent_adapter_id",
            "passed": (
                transition is not None
                and transition.get("parent_adapter_id")
                == expected_parent_adapter_id
            ),
            "observed": None
            if transition is None
            else transition.get("parent_adapter_id"),
            "threshold": expected_parent_adapter_id,
        },
        {
            "name": "transition_child_adapter_id",
            "passed": (
                transition is not None
                and node is not None
                and transition.get("child_adapter_id") == node.get("adapter_id")
            ),
            "observed": None
            if transition is None
            else transition.get("child_adapter_id"),
            "threshold": None if node is None else node.get("adapter_id"),
        },
        {
            "name": "transition_lineage_depth",
            "passed": (
                transition is not None
                and transition.get("child_lineage_depth")
                == expected_lineage_depth
                and transition.get("depth_step") == 1
            ),
            "observed": None
            if transition is None
            else {
                "child_lineage_depth": transition.get("child_lineage_depth"),
                "depth_step": transition.get("depth_step"),
            },
            "threshold": {
                "child_lineage_depth": expected_lineage_depth,
                "depth_step": 1,
            },
        },
        {
            "name": "transition_parent_fingerprint",
            "passed": (
                transition is not None
                and transition.get("parent_fingerprint_verified") is True
            ),
            "observed": None
            if transition is None
            else transition.get("parent_fingerprint_verified"),
            "threshold": True,
        },
        {
            "name": "transition_weight_change",
            "passed": (
                transition is not None
                and transition.get("weights_changed_from_parent") is True
            ),
            "observed": None
            if transition is None
            else transition.get("weights_changed_from_parent"),
            "threshold": True,
        },
    ]
    failed = [row["name"] for row in checks if row["passed"] is not True]
    return {
        "row_type": "hf_adapter_continuation_executor_postflight",
        "status": "ready" if not failed else "blocked",
        "ready": not failed,
        "output_dir": str(output_dir),
        "adapter_id": None if node is None else node.get("adapter_id"),
        "parent_adapter_id": None if node is None else node.get("parent_adapter_id"),
        "lineage_depth": None if node is None else node.get("lineage_depth"),
        "promotion_ready": None if node is None else node.get("promotion_ready"),
        "chain_status": chain.get("status"),
        "chain_selected_adapter_id": chain.get("selected_adapter_id"),
        "transition_status": None
        if transition is None
        else transition.get("status"),
        "transition_ready": None
        if transition is None
        else transition.get("transition_ready"),
        "transition": transition,
        "eval_handoff_delta": None
        if transition is None
        else transition.get("eval_handoff_delta"),
        "child_eval_improvement": None
        if transition is None
        else transition.get("child_eval_improvement"),
        "artifact_probe_process_status": None
        if transition is None
        else transition.get("artifact_probe_process_status"),
        "artifact_probe_process_pid": None
        if transition is None
        else transition.get("artifact_probe_process_pid"),
        "artifact_probe_process_exit_code": None
        if transition is None
        else transition.get("artifact_probe_process_exit_code"),
        "failed_checks": failed,
        "checks": checks,
    }


def _finish(
    state: dict[str, object],
    state_path: Path,
    *,
    status: str,
    action: str,
    reason: str,
) -> dict[str, object]:
    state["status"] = status
    state["action"] = action
    state["reason"] = reason
    state["completed_at"] = _now()
    state["promoted_generation_count"] = sum(
        isinstance(row, Mapping)
        and row.get("status") in {"promoted", "promoted_recovered"}
        for row in state.get("generations") or []
    )
    _write_state(state_path, state)
    return state


def _command_returncode(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    returncode = getattr(value, "returncode", None)
    if isinstance(returncode, int):
        return returncode
    raise TypeError("command runner must return an int or object with returncode")


def _terminate_owned_process(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float,
) -> None:
    if process.poll() is not None:
        return

    def send_signal(*, force: bool) -> None:
        if os.name == "posix":
            try:
                os.killpg(
                    process.pid,
                    signal.SIGKILL if force else signal.SIGTERM,
                )
                return
            except OSError:
                pass
        elif os.name == "nt":
            try:
                subprocess.run(
                    [
                        "taskkill",
                        "/PID",
                        str(process.pid),
                        "/T",
                        *(["/F"] if force else []),
                    ],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                pass
            if process.poll() is not None:
                return
        try:
            process.kill() if force else process.terminate()
        except OSError:
            pass

    def owned_scope_alive() -> bool:
        if os.name != "posix":
            return process.poll() is None
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return process.poll() is None
        return True

    send_signal(force=False)
    deadline = time.monotonic() + grace_seconds
    while owned_scope_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    if owned_scope_alive():
        send_signal(force=True)


def _execute_command(
    command: Sequence[str],
    *,
    command_runner: CommandRunner | None,
    command_cwd: str | Path | None,
    command_env: Mapping[str, str] | None,
    log_path: Path | None,
    tee_output: bool,
    process_started: ProcessStarted | None,
    process_progress: ProcessProgress | None,
    stop_request_loader: StopRequestLoader | None = None,
    process_stop_requested: ProcessStopRequested | None = None,
) -> int:
    if command_runner is not None:
        return _command_returncode(command_runner(command))
    if log_path is None:
        raise ValueError("log_path is required for subprocess execution")
    environment = None
    if command_env is not None:
        environment = dict(os.environ)
        environment.update({str(key): str(value) for key, value in command_env.items()})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_descriptor = os.open(
        log_path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    with os.fdopen(log_descriptor, "wb", buffering=0) as log_handle:
        header = (
            f"[spiraltorch-executor] started_at={_now()} "
            f"command={shlex.join([str(value) for value in command])}\n"
        ).encode("utf-8")
        log_handle.write(header)
        log_bytes = len(header)
        last_progress_at: float | None = None
        process_group_options: dict[str, object] = {}
        if os.name == "posix":
            process_group_options["start_new_session"] = True
        elif os.name == "nt":
            process_group_options["creationflags"] = getattr(
                subprocess,
                "CREATE_NEW_PROCESS_GROUP",
                0,
            )
        process = subprocess.Popen(
            list(command),
            cwd=None if command_cwd is None else str(command_cwd),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            **process_group_options,
        )
        stop_watcher_done = threading.Event()
        stop_watcher: threading.Thread | None = None
        try:
            if process_started is not None:
                process_started(process.pid)
            if process.stdout is None:
                raise RuntimeError("subprocess stdout pipe was not created")
            if stop_request_loader is not None:

                def watch_stop_request() -> None:
                    while not stop_watcher_done.is_set():
                        try:
                            request = stop_request_loader()
                        except (OSError, RuntimeError, ValueError):
                            request = None
                        if request is not None and process.poll() is None:
                            if process_stop_requested is not None:
                                try:
                                    process_stop_requested(dict(request))
                                except Exception:
                                    pass
                            _terminate_owned_process(
                                process,
                                grace_seconds=_PROCESS_STOP_GRACE_SECONDS,
                            )
                            return
                        stop_watcher_done.wait(_PROCESS_STOP_POLL_INTERVAL_SECONDS)

                stop_watcher = threading.Thread(
                    target=watch_stop_request,
                    name="spiraltorch-adapter-executor-stop",
                    daemon=True,
                )
                stop_watcher.start()
            binary_output = getattr(sys.stdout, "buffer", None) if tee_output else None
            text_output = sys.stdout if tee_output and binary_output is None else None
            tee_failed = False
            read_chunk = getattr(process.stdout, "read1", process.stdout.read)
            while chunk := read_chunk(64 * 1024):
                log_handle.write(chunk)
                log_bytes += len(chunk)
                now = time.monotonic()
                if process_progress is not None and (
                    last_progress_at is None
                    or now - last_progress_at >= _PROCESS_PROGRESS_INTERVAL_SECONDS
                ):
                    process_progress(log_bytes)
                    last_progress_at = now
                if not tee_failed and (
                    binary_output is not None or text_output is not None
                ):
                    try:
                        if binary_output is not None:
                            binary_output.write(chunk)
                            binary_output.flush()
                        elif text_output is not None:
                            text_output.write(chunk.decode("utf-8", errors="replace"))
                            text_output.flush()
                    except (BrokenPipeError, OSError, UnicodeError):
                        tee_failed = True
            returncode = int(process.wait())
            log_handle.write(
                (
                    f"\n[spiraltorch-executor] exited_at={_now()} "
                    f"returncode={returncode}\n"
                ).encode("utf-8")
            )
            return returncode
        except BaseException as exc:
            if process.poll() is None:
                _terminate_owned_process(
                    process,
                    grace_seconds=_PROCESS_STOP_GRACE_SECONDS,
                )
                process.wait()
            try:
                log_handle.write(
                    (
                        f"\n[spiraltorch-executor] aborted_at={_now()} "
                        f"exception={exc.__class__.__name__}\n"
                    ).encode("utf-8")
                )
            except OSError:
                pass
            raise
        finally:
            stop_watcher_done.set()
            if stop_watcher is not None:
                stop_watcher.join(timeout=1.0)


def _running_attempt_process_observation(attempt: Mapping[str, object]) -> str:
    hostname = attempt.get("hostname")
    if not isinstance(hostname, str) or not hostname:
        return "legacy_unverified"
    if hostname != socket.gethostname():
        return "remote_unverified"
    alive = local_pid_alive(attempt.get("pid"))
    if alive is True:
        return "alive"
    if alive is False:
        return "exited"
    return "local_unverified"


def _recover_running_attempts(
    state: dict[str, object],
    chain: Mapping[str, object],
    *,
    retry_interrupted: bool,
) -> str | None:
    generations = state.get("generations") or []
    for raw_attempt in generations:
        if not isinstance(raw_attempt, dict) or raw_attempt.get("status") != "running":
            continue
        process_observation = _running_attempt_process_observation(raw_attempt)
        raw_attempt["process_liveness_observed_at"] = _now()
        raw_attempt["process_liveness_observation"] = process_observation
        if process_observation == "alive":
            return "running attempt process is still alive; monitor it before recovery"
        if process_observation == "remote_unverified":
            return "running attempt belongs to another host; verify it before recovery"
        if process_observation in {"legacy_unverified", "local_unverified"}:
            return "running attempt process liveness is unverified; refusing automatic recovery"
        interruption_claim = _interruption_claim_report(raw_attempt)
        raw_attempt["interruption_claim_observation"] = interruption_claim
        if interruption_claim.get("ready") is not True:
            return (
                "running attempt process scope is unverified; refusing automatic "
                f"recovery: {interruption_claim.get('issue')}"
            )
        output_value = raw_attempt.get("output_dir")
        if output_value is None:
            return "running attempt is missing output_dir"
        output_dir = Path(str(output_value)).expanduser()
        postflight = _postflight_report(
            chain,
            output_dir=output_dir,
            expected_parent_adapter_id=raw_attempt.get("parent_adapter_id"),
            expected_lineage_depth=int(raw_attempt.get("lineage_depth") or -1),
        )
        if postflight.get("ready") is True:
            raw_attempt["status"] = "promoted_recovered"
            raw_attempt["status_before_interruption"] = "running"
            raw_attempt["interruption_claim"] = interruption_claim
            raw_attempt["postflight"] = postflight
            raw_attempt["adapter_id"] = postflight.get("adapter_id")
            raw_attempt["completed_at"] = _now()
            continue
        if retry_interrupted and not output_dir.exists():
            raw_attempt["status"] = "interrupted_retry"
            raw_attempt["status_before_interruption"] = "running"
            raw_attempt["interruption_claim"] = interruption_claim
            raw_attempt["postflight"] = postflight
            raw_attempt["completed_at"] = _now()
            continue
        raw_attempt["postflight"] = postflight
        return (
            "running attempt is unresolved; verify or remove its partial output "
            "before retrying"
        )
    return None


def _unresolved_failed_attempt_output(
    state: Mapping[str, object],
) -> dict[str, object] | None:
    generations = state.get("generations") or []
    for index, raw_attempt in enumerate(generations):
        if not isinstance(raw_attempt, Mapping) or raw_attempt.get("status") not in {
            "cancelled",
            "failed",
            "interrupted",
            "postflight_failed",
        }:
            continue
        output_value = raw_attempt.get("output_dir")
        if output_value is None:
            continue
        output_dir = Path(str(output_value)).expanduser().resolve()
        if not output_dir.exists():
            continue
        claimed_by_promotion = any(
            isinstance(later, Mapping)
            and later.get("status") in {"promoted", "promoted_recovered"}
            and later.get("output_dir") is not None
            and Path(str(later.get("output_dir"))).expanduser().resolve() == output_dir
            for later in generations[index + 1 :]
        )
        if not claimed_by_promotion:
            return {
                "attempt_id": raw_attempt.get("attempt_id"),
                "attempt_status": raw_attempt.get("status"),
                "output_dir": str(output_dir),
            }
    return None


def _interruption_claim_report(
    attempt: Mapping[str, object],
) -> dict[str, object]:
    current_hostname = socket.gethostname()
    recorded_hostname = attempt.get("hostname")
    pid = attempt.get("pid")
    runner_kind = attempt.get("runner_kind")
    process_group_isolated = attempt.get("process_group_isolated")
    stop_scope = attempt.get("stop_scope")
    same_host = (
        recorded_hostname == current_hostname
        if isinstance(recorded_hostname, str) and recorded_hostname
        else None
    )
    pid_alive = local_pid_alive(pid) if same_host is True else None
    process_group_alive = (
        local_process_group_alive(pid)
        if (
            pid_alive is False
            and runner_kind == "subprocess"
            and process_group_isolated is True
            and stop_scope == "process_group"
        )
        else None
    )
    if attempt.get("status") != "running":
        issue = "attempt_not_running"
        action = "inspect_attempt_status"
    elif runner_kind != "subprocess":
        issue = "runner_scope_unverified"
        action = "inspect_interrupted_runner"
    elif same_host is False:
        issue = "attempt_belongs_to_remote_host"
        action = "inspect_remote_process"
    elif same_host is not True:
        issue = "attempt_host_unverified"
        action = "inspect_unverified_process"
    elif pid_alive is True:
        issue = "attempt_process_alive"
        action = "wait_for_process_exit"
    elif pid_alive is not False:
        issue = "attempt_process_liveness_unverified"
        action = "inspect_unverified_process"
    elif process_group_isolated is not True or stop_scope != "process_group":
        issue = "attempt_process_scope_unverified"
        action = "inspect_interrupted_process_scope"
    elif process_group_alive is True:
        issue = "attempt_process_group_alive"
        action = "wait_for_process_group_exit"
    elif process_group_alive is not False:
        issue = "attempt_process_group_liveness_unverified"
        action = "inspect_interrupted_process_scope"
    else:
        issue = None
        action = "quarantine_interrupted_output"
    return {
        "row_type": "hf_adapter_continuation_executor_interruption_claim",
        "schema": HF_ADAPTER_CONTINUATION_EXECUTOR_INTERRUPTION_CLAIM_SCHEMA,
        "observed_at": _now(),
        "attempt_id": attempt.get("attempt_id"),
        "attempt_status": attempt.get("status"),
        "runner_kind": runner_kind,
        "process_group_isolated": process_group_isolated,
        "stop_scope": stop_scope,
        "recorded_hostname": recorded_hostname,
        "current_hostname": current_hostname,
        "same_host": same_host,
        "pid": pid,
        "pid_alive_observed": pid_alive,
        "process_group_id": pid if process_group_isolated is True else None,
        "process_group_alive_observed": process_group_alive,
        "ready": issue is None,
        "issue": issue,
        "action": action,
    }


def _interruption_claim_matches_attempt(
    value: object,
    *,
    attempt: Mapping[str, object],
) -> bool:
    if not isinstance(value, Mapping):
        return False
    pid = attempt.get("pid")
    hostname = attempt.get("hostname")
    return bool(
        value.get("row_type") == "hf_adapter_continuation_executor_interruption_claim"
        and value.get("schema")
        == HF_ADAPTER_CONTINUATION_EXECUTOR_INTERRUPTION_CLAIM_SCHEMA
        and value.get("attempt_id") == attempt.get("attempt_id")
        and value.get("attempt_status") == "running"
        and value.get("runner_kind") == attempt.get("runner_kind") == "subprocess"
        and value.get("process_group_isolated")
        is attempt.get("process_group_isolated")
        is True
        and value.get("stop_scope") == attempt.get("stop_scope") == "process_group"
        and isinstance(hostname, str)
        and bool(hostname)
        and value.get("recorded_hostname") == hostname
        and value.get("current_hostname") == hostname
        and value.get("same_host") is True
        and not isinstance(pid, bool)
        and isinstance(pid, int)
        and pid > 0
        and value.get("pid") == pid
        and value.get("pid_alive_observed") is False
        and value.get("process_group_id") == pid
        and value.get("process_group_alive_observed") is False
        and value.get("ready") is True
        and value.get("issue") is None
        and value.get("action") == "quarantine_interrupted_output"
        and isinstance(value.get("observed_at"), str)
        and bool(value.get("observed_at"))
    )


def _pending_output_resolution_gate(
    state: Mapping[str, object],
) -> dict[str, object] | None:
    pending = state.get("pending_output_resolution")
    pending_attempts = [
        row
        for row in state.get("generations") or []
        if isinstance(row, Mapping) and row.get("pending_output_resolution") is not None
    ]
    if pending is None and not pending_attempts:
        return None
    pending_attempt = pending_attempts[0] if len(pending_attempts) == 1 else None
    attempt_pending = (
        pending_attempt.get("pending_output_resolution")
        if isinstance(pending_attempt, Mapping)
        else None
    )
    required_strings = (
        "resolution_id",
        "run_id",
        "reason",
        "source_path",
        "destination_path",
        "quarantine_root",
    )
    interruption_claim = (
        pending.get("interruption_claim") if isinstance(pending, Mapping) else None
    )
    attempt_interruption_claim = (
        pending_attempt.get("interruption_claim")
        if isinstance(pending_attempt, Mapping)
        else None
    )
    if (
        isinstance(pending_attempt, Mapping)
        and pending_attempt.get("status") == "interrupted"
    ):
        interruption_claim_consistent = bool(
            isinstance(interruption_claim, Mapping)
            and isinstance(attempt_interruption_claim, Mapping)
            and dict(interruption_claim) == dict(attempt_interruption_claim)
            and _interruption_claim_matches_attempt(
                interruption_claim,
                attempt=pending_attempt,
            )
        )
    else:
        interruption_claim_consistent = (
            interruption_claim is None and attempt_interruption_claim is None
        )
    consistent = bool(
        isinstance(pending, Mapping)
        and isinstance(attempt_pending, Mapping)
        and dict(attempt_pending) == dict(pending)
        and pending.get("row_type")
        == "hf_adapter_continuation_executor_output_resolution_intent"
        and pending.get("schema")
        == HF_ADAPTER_CONTINUATION_EXECUTOR_OUTPUT_RESOLUTION_SCHEMA
        and pending.get("attempt_id") == pending_attempt.get("attempt_id")
        and pending.get("attempt_status") == pending_attempt.get("status")
        and pending.get("lineage_depth") == pending_attempt.get("lineage_depth")
        and all(
            isinstance(pending.get(field), str) and bool(pending.get(field))
            for field in required_strings
        )
        and isinstance(pending.get("tree_snapshot"), Mapping)
        and interruption_claim_consistent
    )
    return {
        "issue": (
            "output_quarantine_incomplete"
            if consistent
            else "output_quarantine_intent_invalid"
        ),
        "attempt_ids": [row.get("attempt_id") for row in pending_attempts],
    }


def _selection_for_audit(
    state: Mapping[str, object],
    requested_adapter_id: str | None,
) -> str | None:
    if requested_adapter_id is None:
        return None
    for attempt in state.get("generations") or []:
        if isinstance(attempt, Mapping) and attempt.get("status") in {
            "promoted",
            "promoted_recovered",
        }:
            return None
    return requested_adapter_id


def _validate_optional_int(
    name: str,
    value: int | None,
    *,
    minimum: int,
) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _validate_optional_float(
    name: str,
    value: float | None,
    *,
    minimum: float,
    allow_none: bool = True,
) -> None:
    if value is None:
        if allow_none:
            return
        raise ValueError(f"{name} must be finite and >= {minimum}")
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and >= {minimum}") from exc
    if not math.isfinite(parsed) or parsed < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")


def _validate_executor_arguments(
    *,
    max_generations: int,
    max_lineage_depth: int | None,
    target_eval_loss: float | None,
    min_eval_improvement: float | None,
    plateau_patience: int,
    max_steps: int | None,
    max_steps_multiplier: float | None,
    max_train_samples: int | None,
    max_train_samples_multiplier: float | None,
    max_eval_samples: int | None,
    max_eval_blocks: int | None,
    streaming_validation_samples: int | None,
    output_prefix: str,
) -> None:
    _validate_optional_int("max_generations", max_generations, minimum=1)
    _validate_optional_int("max_lineage_depth", max_lineage_depth, minimum=0)
    _validate_optional_float("target_eval_loss", target_eval_loss, minimum=0.0)
    _validate_optional_float(
        "min_eval_improvement",
        min_eval_improvement,
        minimum=0.0,
    )
    _validate_optional_int("plateau_patience", plateau_patience, minimum=1)
    _validate_optional_int("max_steps", max_steps, minimum=1)
    _validate_optional_float(
        "max_steps_multiplier",
        max_steps_multiplier,
        minimum=0.0,
    )
    if max_steps_multiplier is not None and float(max_steps_multiplier) == 0.0:
        raise ValueError("max_steps_multiplier must be finite and > 0")
    _validate_optional_int("max_train_samples", max_train_samples, minimum=0)
    _validate_optional_float(
        "max_train_samples_multiplier",
        max_train_samples_multiplier,
        minimum=0.0,
    )
    if (
        max_train_samples_multiplier is not None
        and float(max_train_samples_multiplier) == 0.0
    ):
        raise ValueError("max_train_samples_multiplier must be finite and > 0")
    _validate_optional_int("max_eval_samples", max_eval_samples, minimum=0)
    _validate_optional_int("max_eval_blocks", max_eval_blocks, minimum=0)
    _validate_optional_int(
        "streaming_validation_samples",
        streaming_validation_samples,
        minimum=0,
    )
    if not output_prefix or Path(output_prefix).name != output_prefix:
        raise ValueError("output_prefix must be one path-safe name")


def _run_hf_adapter_continuation_executor_unlocked(
    sources: str | Path | Sequence[str | Path],
    *,
    output_root: str | Path,
    state_path: str | Path | None = None,
    run: bool = False,
    max_generations: int = 1,
    retry_interrupted: bool = False,
    recursive: bool = True,
    allow_inferred_roots: bool = True,
    select_adapter_id: str | None = None,
    command_artifacts: Sequence[Mapping[str, object] | str | Path] | None = None,
    max_lineage_depth: int | None = None,
    target_eval_loss: float | None = None,
    min_eval_improvement: float | None = None,
    plateau_patience: int = 1,
    output_prefix: str = "generation",
    max_steps: int | None = None,
    max_steps_multiplier: float | None = 1.0,
    max_train_samples: int | None = None,
    max_train_samples_multiplier: float | None = 1.0,
    max_eval_samples: int | None = None,
    max_eval_blocks: int | None = None,
    streaming_validation_samples: int | None = None,
    command_runner: CommandRunner | None = None,
    command_cwd: str | Path | None = None,
    command_env: Mapping[str, str] | None = None,
    tee_output: bool = True,
) -> dict[str, object]:
    """Plan or run promoted adapter generations until policy or budget stops."""

    _validate_executor_arguments(
        max_generations=max_generations,
        max_lineage_depth=max_lineage_depth,
        target_eval_loss=target_eval_loss,
        min_eval_improvement=min_eval_improvement,
        plateau_patience=plateau_patience,
        max_steps=max_steps,
        max_steps_multiplier=max_steps_multiplier,
        max_train_samples=max_train_samples,
        max_train_samples_multiplier=max_train_samples_multiplier,
        max_eval_samples=max_eval_samples,
        max_eval_blocks=max_eval_blocks,
        streaming_validation_samples=streaming_validation_samples,
        output_prefix=output_prefix,
    )
    source_paths = _source_paths(sources)
    resolved_output_root = Path(output_root).expanduser().resolve()
    resolved_state_path = (
        Path(state_path).expanduser().resolve()
        if state_path is not None
        else resolved_output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_FILENAME
    )
    if resolved_state_path == resolved_output_root:
        raise ValueError("state_path must be a file below or outside output_root")
    if resolved_state_path == (
        resolved_output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
    ):
        raise ValueError("state_path cannot overwrite the executor lock")
    if resolved_state_path == (
        resolved_output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_LOG_DIRNAME
    ):
        raise ValueError("state_path cannot replace the executor log directory")
    control_dir = (
        resolved_output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_CONTROL_DIRNAME
    )
    if resolved_state_path == control_dir or control_dir in resolved_state_path.parents:
        raise ValueError("state_path cannot be inside the executor control directory")
    if resolved_state_path in source_paths:
        raise ValueError("state_path cannot overwrite an adapter chain source")
    policy = {
        "max_lineage_depth": max_lineage_depth,
        "target_eval_loss": target_eval_loss,
        "min_eval_improvement": min_eval_improvement,
        "plateau_patience": plateau_patience,
    }
    scale_up = {
        "max_steps": max_steps,
        "max_steps_multiplier": max_steps_multiplier,
        "max_train_samples": max_train_samples,
        "max_train_samples_multiplier": max_train_samples_multiplier,
        "max_eval_samples": max_eval_samples,
        "max_eval_blocks": max_eval_blocks,
        "streaming_validation_samples": streaming_validation_samples,
        "output_prefix": output_prefix,
    }
    resolved_command_cwd = (
        Path(command_cwd).expanduser().resolve()
        if command_cwd is not None
        else Path.cwd().resolve()
    )
    log_dir = resolved_output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_LOG_DIRNAME
    execution = {
        "runner_kind": "custom" if command_runner is not None else "subprocess",
        "command_cwd": str(resolved_command_cwd),
        "lock_path": str(
            resolved_output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
        ),
        "log_dir": None if command_runner is not None else str(log_dir),
        "state_progress_interval_seconds": _PROCESS_PROGRESS_INTERVAL_SECONDS,
        "process_group_isolated": command_runner is None,
        "stop_scope": "process_group" if command_runner is None else "runner_boundary",
        "tee_output": bool(tee_output),
    }
    state = _state_for_invocation(
        source_paths=source_paths,
        output_root=resolved_output_root,
        state_path=resolved_state_path,
        policy=policy,
        scale_up=scale_up,
        execution=execution,
        run=run,
        max_generations=max_generations,
    )
    stop_request_path = _state_stop_request_path(state)
    execution["stop_request_path"] = str(stop_request_path)
    execution["stop_poll_interval_seconds"] = _PROCESS_STOP_POLL_INTERVAL_SECONDS
    execution["stop_grace_seconds"] = _PROCESS_STOP_GRACE_SECONDS
    state["execution"] = dict(execution)
    _write_state(resolved_state_path, state)
    executed = 0

    while True:
        stop_request = _matching_stop_request(
            stop_request_path,
            run_id=state.get("run_id"),
            invocation_count=state.get("invocation_count"),
        )
        if stop_request is not None:
            state["stop_request"] = stop_request
            state.pop("pending_generation", None)
            return _finish(
                state,
                resolved_state_path,
                status="stopped",
                action="resume_executor",
                reason="stop_requested",
            )
        output_resolution_gate = _pending_output_resolution_gate(state)
        if output_resolution_gate is not None:
            state["output_resolution_gate"] = output_resolution_gate
            intent_invalid = (
                output_resolution_gate.get("issue")
                == "output_quarantine_intent_invalid"
            )
            return _finish(
                state,
                resolved_state_path,
                status="blocked",
                action=(
                    "inspect_output_quarantine_intent"
                    if intent_invalid
                    else "complete_output_quarantine"
                ),
                reason=str(output_resolution_gate.get("issue")),
            )
        state.pop("output_resolution_gate", None)
        unresolved_output = _unresolved_failed_attempt_output(state)
        if unresolved_output is not None:
            state["unresolved_generation"] = unresolved_output
            return _finish(
                state,
                resolved_state_path,
                status="blocked",
                action="resolve_failed_generation_output",
                reason="failed_generation_output_exists",
            )
        state.pop("unresolved_generation", None)
        audit_selection = _selection_for_audit(state, select_adapter_id)
        try:
            chain = _audit_chain(
                source_paths=source_paths,
                output_root=resolved_output_root,
                recursive=recursive,
                allow_inferred_roots=allow_inferred_roots,
                select_adapter_id=audit_selection,
                command_artifacts=command_artifacts,
                policy=policy,
            )
        except Exception as exc:
            state["failure"] = f"{exc.__class__.__name__}: {exc}"
            return _finish(
                state,
                resolved_state_path,
                status="failed",
                action="inspect_chain_audit_failure",
                reason="chain_audit_failed",
            )
        _record_chain_state(state, chain)
        state["generations_executed_this_invocation"] = executed

        running_attempts = [
            attempt
            for attempt in state.get("generations") or []
            if isinstance(attempt, Mapping) and attempt.get("status") == "running"
        ]
        recovery_chain = chain
        if running_attempts:
            try:
                # An explicit initial selection points at the old parent. Recovery
                # must instead prove that the generated adapter became the live tip.
                recovery_chain = _audit_chain(
                    source_paths=source_paths,
                    output_root=resolved_output_root,
                    recursive=recursive,
                    allow_inferred_roots=allow_inferred_roots,
                    select_adapter_id=None,
                    command_artifacts=command_artifacts,
                    policy=policy,
                )
            except Exception as exc:
                state["failure"] = f"{exc.__class__.__name__}: {exc}"
                return _finish(
                    state,
                    resolved_state_path,
                    status="failed",
                    action="inspect_interrupted_generation_audit",
                    reason="interrupted_generation_audit_failed",
                )
        recovery_error = _recover_running_attempts(
            state,
            recovery_chain,
            retry_interrupted=retry_interrupted,
        )
        if recovery_error is not None:
            return _finish(
                state,
                resolved_state_path,
                status="blocked",
                action="audit_interrupted_generation",
                reason=recovery_error,
            )
        if any(
            attempt.get("status") == "promoted_recovered"
            for attempt in running_attempts
        ):
            chain = recovery_chain
            _record_chain_state(state, chain)

        policy_status = chain.get("continuation_policy_status")
        if policy_status == "stop":
            state.pop("pending_generation", None)
            return _finish(
                state,
                resolved_state_path,
                status="stopped",
                action="stop_training",
                reason="continuation_policy_stop",
            )
        if chain.get("chain_ready") is not True:
            return _finish(
                state,
                resolved_state_path,
                status="blocked",
                action="resolve_chain",
                reason=str(chain.get("status") or "chain_not_ready"),
            )
        if policy_status != "continue":
            return _finish(
                state,
                resolved_state_path,
                status="blocked",
                action="collect_policy_evidence",
                reason=str(policy_status or "continuation_policy_not_ready"),
            )
        if chain.get("continuation_artifacts_ready") is not True:
            return _finish(
                state,
                resolved_state_path,
                status="blocked",
                action="recover_launch_command",
                reason="continuation_command_missing",
            )
        if run and executed >= max_generations:
            state.pop("pending_generation", None)
            return _finish(
                state,
                resolved_state_path,
                status="generation_limit_reached",
                action="resume_executor",
                reason="max_generations_per_invocation_reached",
            )

        selected_depth = int(chain.get("selected_lineage_depth") or 0)
        next_depth = selected_depth + 1
        output_dir = (
            resolved_output_root / f"{output_prefix}-{next_depth:03d}"
        ).resolve()
        try:
            command = hf_finetune_scale_up_command(
                chain,
                output_dir=output_dir,
                max_steps=max_steps,
                max_steps_multiplier=max_steps_multiplier,
                max_train_samples=max_train_samples,
                max_train_samples_multiplier=max_train_samples_multiplier,
                max_eval_samples=max_eval_samples,
                max_eval_blocks=max_eval_blocks,
                streaming_validation_samples=streaming_validation_samples,
                source_cwd=resolved_command_cwd,
            )
            preflight = hf_finetune_scale_up_preflight_report(command)
        except Exception as exc:
            state["failure"] = f"{exc.__class__.__name__}: {exc}"
            return _finish(
                state,
                resolved_state_path,
                status="failed",
                action="inspect_scale_up_planning_failure",
                reason="scale_up_planning_failed",
            )
        pending = {
            "status": "planned",
            "planned_at": _now(),
            "parent_adapter_id": chain.get("selected_adapter_id"),
            "parent_adapter_path": chain.get("selected_adapter_path"),
            "lineage_depth": next_depth,
            "output_dir": str(output_dir),
            "source_transition": _selected_transition(chain),
            "command_runtime": command.get("command_runtime"),
            "parent_identity_contract": command.get(
                "adapter_continuation_identity_contract"
            ),
            "adapter_input_identity": preflight.get("adapter_input_identity"),
            "training_input_identity_contract": command.get(
                "training_input_identity_contract"
            ),
            "training_input_identity": preflight.get("training_input_identity"),
            "runtime_input_identity_contract": command.get(
                "runtime_input_identity_contract"
            ),
            "runtime_input_expected_id": preflight.get(
                "runtime_input_expected_id"
            ),
            "command": command,
            "preflight": preflight,
        }
        state["pending_generation"] = pending
        _write_state(resolved_state_path, state)
        if output_dir.exists():
            return _finish(
                state,
                resolved_state_path,
                status="blocked",
                action="resolve_output_collision",
                reason="generation_output_exists",
            )
        if command.get("status") != "ok":
            return _finish(
                state,
                resolved_state_path,
                status="blocked",
                action="resolve_scale_up_command",
                reason=str(command.get("status") or "scale_up_not_ready"),
            )
        if preflight.get("ready") is not True:
            return _finish(
                state,
                resolved_state_path,
                status="blocked",
                action="resolve_preflight",
                reason="scale_up_preflight_not_ready",
            )
        launch_stop_request = _matching_stop_request(
            stop_request_path,
            run_id=state.get("run_id"),
            invocation_count=state.get("invocation_count"),
        )
        if launch_stop_request is not None:
            state["stop_request"] = launch_stop_request
            state.pop("pending_generation", None)
            return _finish(
                state,
                resolved_state_path,
                status="stopped",
                action="resume_executor",
                reason="stop_requested",
            )
        if not run:
            return _finish(
                state,
                resolved_state_path,
                status="ready",
                action="run_generation",
                reason="dry_run_ready",
            )

        command_values = command.get("command")
        if not isinstance(command_values, Sequence) or isinstance(
            command_values,
            (str, bytes),
        ):
            return _finish(
                state,
                resolved_state_path,
                status="blocked",
                action="resolve_scale_up_command",
                reason="scale_up_command_missing",
            )
        resolved_command = [str(item) for item in command_values]
        attempt_id = f"generation-attempt-{uuid.uuid4().hex}"
        attempt_log_path = (
            None
            if command_runner is not None
            else log_dir / f"{output_prefix}-{next_depth:03d}-{attempt_id[-12:]}.log"
        )
        attempt = {
            "attempt_id": attempt_id,
            "status": "running",
            "started_at": _now(),
            "runner_kind": execution["runner_kind"],
            "process_group_isolated": execution["process_group_isolated"],
            "stop_scope": execution["stop_scope"],
            "hostname": socket.gethostname(),
            "pid": None,
            "command_cwd": str(resolved_command_cwd),
            "log_path": None if attempt_log_path is None else str(attempt_log_path),
            "log_stream": (
                None if attempt_log_path is None else "combined_stdout_stderr"
            ),
            "tee_output": bool(tee_output),
            "parent_adapter_id": chain.get("selected_adapter_id"),
            "parent_adapter_path": chain.get("selected_adapter_path"),
            "lineage_depth": next_depth,
            "output_dir": str(output_dir),
            "source_transition": _selected_transition(chain),
            "command_runtime": command.get("command_runtime"),
            "parent_identity_contract": command.get(
                "adapter_continuation_identity_contract"
            ),
            "adapter_input_identity": preflight.get("adapter_input_identity"),
            "training_input_identity_contract": command.get(
                "training_input_identity_contract"
            ),
            "training_input_identity": preflight.get("training_input_identity"),
            "runtime_input_identity_contract": command.get(
                "runtime_input_identity_contract"
            ),
            "runtime_input_expected_id": preflight.get(
                "runtime_input_expected_id"
            ),
            "command": resolved_command,
            "command_display": shlex.join(resolved_command),
            "scale_up": command,
            "preflight": preflight,
        }
        generations = state.setdefault("generations", [])
        if not isinstance(generations, list):
            raise ValueError("executor generations state must be a list")
        generations.append(attempt)
        state["status"] = "running"
        state["action"] = "run_generation"
        state["pending_generation"] = attempt
        _write_state(resolved_state_path, state)

        started = time.monotonic()
        observed_stop_request: dict[str, object] | None = None

        def record_process_started(pid: int) -> None:
            attempt["pid"] = pid
            attempt["process_started_at"] = _now()
            _write_state(resolved_state_path, state)

        def record_process_progress(log_bytes: int) -> None:
            attempt["last_output_at"] = _now()
            attempt["log_bytes_observed"] = log_bytes
            _write_state(resolved_state_path, state)

        def load_current_stop_request() -> dict[str, object] | None:
            return _matching_stop_request(
                stop_request_path,
                run_id=state.get("run_id"),
                invocation_count=state.get("invocation_count"),
            )

        def record_process_stop_requested(request: Mapping[str, object]) -> None:
            nonlocal observed_stop_request
            observed_stop_request = dict(request)

        try:
            returncode = _execute_command(
                resolved_command,
                command_runner=command_runner,
                command_cwd=command_cwd,
                command_env=command_env,
                log_path=attempt_log_path,
                tee_output=tee_output,
                process_started=(
                    None if command_runner is not None else record_process_started
                ),
                process_progress=(
                    None if command_runner is not None else record_process_progress
                ),
                stop_request_loader=(
                    None if command_runner is not None else load_current_stop_request
                ),
                process_stop_requested=(
                    None
                    if command_runner is not None
                    else record_process_stop_requested
                ),
            )
        except Exception as exc:
            attempt["status"] = "failed"
            attempt["failure"] = f"{exc.__class__.__name__}: {exc}"
            attempt["completed_at"] = _now()
            attempt["duration_seconds"] = time.monotonic() - started
            return _finish(
                state,
                resolved_state_path,
                status="failed",
                action="inspect_generation_failure",
                reason="command_runner_failed",
            )
        attempt["returncode"] = returncode
        attempt["process_exited_at"] = _now()
        attempt["completed_at"] = _now()
        attempt["duration_seconds"] = time.monotonic() - started
        boundary_stop_request = load_current_stop_request()
        if (
            observed_stop_request is None
            and returncode != 0
            and boundary_stop_request is not None
        ):
            observed_stop_request = boundary_stop_request
        if observed_stop_request is not None:
            attempt["status"] = "cancelled"
            attempt["stop_request"] = observed_stop_request
            state["stop_request"] = observed_stop_request
            state.pop("pending_generation", None)
            return _finish(
                state,
                resolved_state_path,
                status="stopped",
                action="resume_executor",
                reason="stop_requested",
            )
        if returncode != 0:
            attempt["status"] = "failed"
            return _finish(
                state,
                resolved_state_path,
                status="failed",
                action="inspect_generation_failure",
                reason=f"generation_command_returncode_{returncode}",
            )

        try:
            post_chain = _audit_chain(
                source_paths=source_paths,
                output_root=resolved_output_root,
                recursive=recursive,
                allow_inferred_roots=allow_inferred_roots,
                select_adapter_id=None,
                command_artifacts=command_artifacts,
                policy=policy,
            )
        except Exception as exc:
            attempt["status"] = "postflight_failed"
            attempt["failure"] = f"{exc.__class__.__name__}: {exc}"
            return _finish(
                state,
                resolved_state_path,
                status="failed",
                action="inspect_generation_postflight",
                reason="postflight_chain_audit_failed",
            )
        postflight = _postflight_report(
            post_chain,
            output_dir=output_dir,
            expected_parent_adapter_id=chain.get("selected_adapter_id"),
            expected_lineage_depth=next_depth,
        )
        attempt["postflight"] = postflight
        state["chain_report"] = post_chain
        if postflight.get("ready") is not True:
            attempt["status"] = "postflight_failed"
            return _finish(
                state,
                resolved_state_path,
                status="failed",
                action="inspect_generation_postflight",
                reason="generated_adapter_not_promotion_ready",
            )
        attempt["status"] = "promoted"
        attempt["adapter_id"] = postflight.get("adapter_id")
        _record_chain_state(state, post_chain)
        state.pop("pending_generation", None)
        executed += 1
        state["generations_executed_this_invocation"] = executed
        state["promoted_generation_count"] = sum(
            isinstance(row, Mapping)
            and row.get("status") in {"promoted", "promoted_recovered"}
            for row in generations
        )
        _write_state(resolved_state_path, state)


def run_hf_adapter_continuation_executor(
    sources: str | Path | Sequence[str | Path],
    *,
    output_root: str | Path,
    state_path: str | Path | None = None,
    run: bool = False,
    max_generations: int = 1,
    retry_interrupted: bool = False,
    recursive: bool = True,
    allow_inferred_roots: bool = True,
    select_adapter_id: str | None = None,
    command_artifacts: Sequence[Mapping[str, object] | str | Path] | None = None,
    max_lineage_depth: int | None = None,
    target_eval_loss: float | None = None,
    min_eval_improvement: float | None = None,
    plateau_patience: int = 1,
    output_prefix: str = "generation",
    max_steps: int | None = None,
    max_steps_multiplier: float | None = 1.0,
    max_train_samples: int | None = None,
    max_train_samples_multiplier: float | None = 1.0,
    max_eval_samples: int | None = None,
    max_eval_blocks: int | None = None,
    streaming_validation_samples: int | None = None,
    command_runner: CommandRunner | None = None,
    command_cwd: str | Path | None = None,
    command_env: Mapping[str, str] | None = None,
    tee_output: bool = True,
) -> dict[str, object]:
    """Plan or run an adapter continuation executor under a single-writer lock."""

    resolved_output_root = Path(output_root).expanduser().resolve()
    with _executor_lock(resolved_output_root):
        return _run_hf_adapter_continuation_executor_unlocked(
            sources,
            output_root=resolved_output_root,
            state_path=state_path,
            run=run,
            max_generations=max_generations,
            retry_interrupted=retry_interrupted,
            recursive=recursive,
            allow_inferred_roots=allow_inferred_roots,
            select_adapter_id=select_adapter_id,
            command_artifacts=command_artifacts,
            max_lineage_depth=max_lineage_depth,
            target_eval_loss=target_eval_loss,
            min_eval_improvement=min_eval_improvement,
            plateau_patience=plateau_patience,
            output_prefix=output_prefix,
            max_steps=max_steps,
            max_steps_multiplier=max_steps_multiplier,
            max_train_samples=max_train_samples,
            max_train_samples_multiplier=max_train_samples_multiplier,
            max_eval_samples=max_eval_samples,
            max_eval_blocks=max_eval_blocks,
            streaming_validation_samples=streaming_validation_samples,
            command_runner=command_runner,
            command_cwd=command_cwd,
            command_env=command_env,
            tee_output=tee_output,
        )


def hf_adapter_continuation_executor_lines(
    report_or_path: Mapping[str, object] | str | Path,
) -> list[str]:
    report = (
        dict(report_or_path)
        if isinstance(report_or_path, Mapping)
        else load_hf_adapter_continuation_executor(report_or_path)
    )
    lines = [
        (
            "hf_adapter_continuation_executor "
            f"status={report.get('status')} "
            f"action={report.get('action')} "
            f"reason={report.get('reason')} "
            f"depth={report.get('selected_lineage_depth')} "
            f"attempts={report.get('generation_attempt_count')} "
            f"promoted={report.get('promoted_generation_count')} "
            f"transitions={report.get('ready_transition_count')}/"
            f"{report.get('transition_count')} "
            f"state={report.get('state_path')}"
        )
    ]
    selected_transition = report.get("selected_transition")
    if isinstance(selected_transition, Mapping):
        lines.append(
            "hf_adapter_continuation_executor_transition "
            f"status={selected_transition.get('status')} "
            f"depth={selected_transition.get('parent_lineage_depth')}"
            f"->{selected_transition.get('child_lineage_depth')} "
            f"parent={selected_transition.get('parent_adapter_id')} "
            f"child={selected_transition.get('child_adapter_id')} "
            f"eval_handoff_delta={selected_transition.get('eval_handoff_delta')} "
            f"eval_improvement={selected_transition.get('child_eval_improvement')} "
            f"input_identity={selected_transition.get('input_identity_ready')} "
            "training_input_identity="
            f"{selected_transition.get('training_input_identity_ready')} "
            "runtime_input_required="
            f"{selected_transition.get('runtime_input_identity_required')} "
            "runtime_input_identity="
            f"{selected_transition.get('runtime_input_identity_ready')} "
            "probe_process="
            f"{selected_transition.get('artifact_probe_process_status')} "
            f"probe_pid={selected_transition.get('artifact_probe_process_pid')}"
        )
    pending = report.get("pending_generation")
    if isinstance(pending, Mapping):
        command = pending.get("command")
        command_status = command.get("status") if isinstance(command, Mapping) else None
        command_runtime = pending.get("command_runtime")
        runtime_status = (
            command_runtime.get("status")
            if isinstance(command_runtime, Mapping)
            else None
        )
        preflight = pending.get("preflight")
        preflight_status = (
            preflight.get("status") if isinstance(preflight, Mapping) else None
        )
        input_identity = pending.get("adapter_input_identity")
        input_identity_status = (
            input_identity.get("status")
            if isinstance(input_identity, Mapping)
            else None
        )
        training_input_identity = pending.get("training_input_identity")
        training_input_identity_status = (
            training_input_identity.get("status")
            if isinstance(training_input_identity, Mapping)
            else None
        )
        runtime_input_contract = pending.get("runtime_input_identity_contract")
        runtime_input_contract_status = (
            runtime_input_contract.get("status")
            if isinstance(runtime_input_contract, Mapping)
            else None
        )
        lines.append(
            "hf_adapter_continuation_executor_pending "
            f"status={pending.get('status')} "
            f"depth={pending.get('lineage_depth')} "
            f"command={command_status} "
            f"runtime={runtime_status} "
            f"preflight={preflight_status} "
            f"input_identity={input_identity_status} "
            f"training_input_identity={training_input_identity_status} "
            f"runtime_input_contract={runtime_input_contract_status} "
            f"runtime_input_expected={pending.get('runtime_input_expected_id')} "
            f"output={pending.get('output_dir')}"
        )
    for raw_attempt in report.get("generations") or []:
        if not isinstance(raw_attempt, Mapping):
            continue
        postflight = raw_attempt.get("postflight")
        postflight_transition_status = (
            postflight.get("transition_status")
            if isinstance(postflight, Mapping)
            else None
        )
        postflight_handoff = (
            postflight.get("eval_handoff_delta")
            if isinstance(postflight, Mapping)
            else None
        )
        command_runtime = raw_attempt.get("command_runtime")
        runtime_status = (
            command_runtime.get("status")
            if isinstance(command_runtime, Mapping)
            else None
        )
        input_identity = raw_attempt.get("adapter_input_identity")
        input_identity_status = (
            input_identity.get("status")
            if isinstance(input_identity, Mapping)
            else None
        )
        training_input_identity = raw_attempt.get("training_input_identity")
        training_input_identity_status = (
            training_input_identity.get("status")
            if isinstance(training_input_identity, Mapping)
            else None
        )
        runtime_input_contract = raw_attempt.get("runtime_input_identity_contract")
        runtime_input_contract_status = (
            runtime_input_contract.get("status")
            if isinstance(runtime_input_contract, Mapping)
            else None
        )
        lines.append(
            "hf_adapter_continuation_executor_generation "
            f"status={raw_attempt.get('status')} "
            f"depth={raw_attempt.get('lineage_depth')} "
            f"returncode={raw_attempt.get('returncode')} "
            f"pid={raw_attempt.get('pid')} "
            f"host={raw_attempt.get('hostname')} "
            f"adapter={raw_attempt.get('adapter_id')} "
            f"runtime={runtime_status} "
            f"input_identity={input_identity_status} "
            f"training_input_identity={training_input_identity_status} "
            f"runtime_input_contract={runtime_input_contract_status} "
            f"runtime_input_expected={raw_attempt.get('runtime_input_expected_id')} "
            "postflight="
            f"{postflight.get('status') if isinstance(postflight, Mapping) else None} "
            f"transition={postflight_transition_status} "
            f"eval_handoff_delta={postflight_handoff} "
            f"output={raw_attempt.get('output_dir')} "
            f"log={raw_attempt.get('log_path')}"
        )
    return lines
