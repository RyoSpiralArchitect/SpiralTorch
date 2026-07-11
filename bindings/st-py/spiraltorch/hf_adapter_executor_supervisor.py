"""Bounded supervision for detached Hugging Face adapter executors."""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ._process import local_pid_alive
from .hf_adapter_executor import (
    HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME,
    _valid_generation_plan_id,
)
from .hf_adapter_executor_launch import (
    HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_LOCK_FILENAME,
    hf_adapter_continuation_executor_launch_status_report,
    hf_adapter_continuation_executor_resume_report,
    load_hf_adapter_continuation_executor_launch,
    resume_hf_adapter_continuation_executor,
)

__all__ = [
    "HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISION_SCHEMA",
    "HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_FILENAME",
    "HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LOCK_FILENAME",
    "HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_SCHEMA",
    "HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_STATUS_SCHEMA",
    "HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_STOP_REQUEST_SCHEMA",
    "hf_adapter_continuation_executor_supervision_lines",
    "hf_adapter_continuation_executor_supervision_report",
    "hf_adapter_continuation_executor_supervisor_lines",
    "hf_adapter_continuation_executor_supervisor_status_lines",
    "hf_adapter_continuation_executor_supervisor_status_report",
    "hf_adapter_continuation_executor_supervisor_stop_request_lines",
    "load_hf_adapter_continuation_executor_supervisor",
    "load_hf_adapter_continuation_executor_supervisor_stop_request",
    "request_hf_adapter_continuation_executor_supervisor_stop",
    "supervise_hf_adapter_continuation_executor",
]


HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISION_SCHEMA = (
    "spiraltorch.hf_adapter_continuation_executor_supervision.v1"
)
HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_SCHEMA = (
    "spiraltorch.hf_adapter_continuation_executor_supervisor.v1"
)
HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_STATUS_SCHEMA = (
    "spiraltorch.hf_adapter_continuation_executor_supervisor_status.v1"
)
HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_STOP_REQUEST_SCHEMA = (
    "spiraltorch.hf_adapter_continuation_executor_supervisor_stop_request.v1"
)
HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_FILENAME = (
    "spiraltorch-hf-adapter-continuation-executor-supervisor.json"
)
HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LOCK_FILENAME = (
    ".spiraltorch-hf-adapter-continuation-executor-supervisor.lock"
)
_SUPERVISOR_CONTROL_DIRNAME = (
    ".spiraltorch-hf-adapter-continuation-executor-supervisor-control"
)

_AUTOMATIC_RESUME_REASON = "max_generations_per_invocation_reached"
_SUPERVISOR_LOCK_RETRIES = 8
_SUPERVISOR_LOCK_RETRY_SECONDS = 0.05


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_symlink():
        raise RuntimeError(f"executor supervisor state cannot be a symlink: {path}")
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


def _exclusive_write_json(path: Path, payload: Mapping[str, object]) -> None:
    control_dir = path.parent
    if control_dir.is_symlink():
        raise RuntimeError(
            f"executor supervisor control directory cannot be a symlink: {control_dir}"
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


def _supervisor_stop_request_path(
    output_root: Path,
    *,
    supervisor_run_id: str,
    invocation_count: int,
) -> Path:
    identity = hashlib.sha256(
        f"{supervisor_run_id}\0{invocation_count}".encode("utf-8")
    ).hexdigest()[:24]
    return (
        output_root
        / _SUPERVISOR_CONTROL_DIRNAME
        / f"stop-{invocation_count:06d}-{identity}.json"
    )


def _state_supervisor_stop_request_path(
    state: Mapping[str, object],
    run: Mapping[str, object],
) -> Path:
    output_root_value = state.get("output_root")
    supervisor_run_id = run.get("supervisor_run_id")
    invocation_count = run.get("invocation_count")
    if output_root_value is None or not isinstance(supervisor_run_id, str):
        raise ValueError("executor supervisor state is missing stop-request identity")
    if not supervisor_run_id:
        raise ValueError("executor supervisor run ID is invalid")
    if (
        isinstance(invocation_count, bool)
        or not isinstance(invocation_count, int)
        or invocation_count <= 0
    ):
        raise ValueError("executor supervisor invocation_count is invalid")
    expected = _supervisor_stop_request_path(
        Path(str(output_root_value)).expanduser().resolve(),
        supervisor_run_id=supervisor_run_id,
        invocation_count=invocation_count,
    )
    recorded = run.get("stop_request_path")
    if recorded is None or Path(str(recorded)).expanduser().resolve() != expected:
        raise ValueError("executor supervisor stop-request path is inconsistent")
    return expected


def load_hf_adapter_continuation_executor_supervisor_stop_request(
    value: str | Path,
) -> dict[str, object]:
    """Load and validate one durable supervisor stop request."""

    path = Path(value).expanduser()
    if path.is_symlink():
        raise ValueError(f"executor supervisor stop request cannot be a symlink: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"executor supervisor stop request must contain an object: {path}"
        )
    if (
        payload.get("schema")
        != HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_STOP_REQUEST_SCHEMA
    ):
        raise ValueError(
            "unsupported HF adapter executor supervisor stop-request schema: "
            f"{payload.get('schema')}"
        )
    if (
        payload.get("row_type")
        != "hf_adapter_continuation_executor_supervisor_stop_request"
    ):
        raise ValueError(
            "unsupported HF adapter executor supervisor stop-request row type: "
            f"{payload.get('row_type')}"
        )
    required_strings = (
        "request_id",
        "requested_at",
        "requested_by_hostname",
        "reason",
        "supervisor_run_id",
        "supervisor_hostname",
        "supervisor_state_path",
        "launch_state_path",
    )
    if any(
        not isinstance(payload.get(field), str) or not payload.get(field)
        for field in required_strings
    ):
        raise ValueError("executor supervisor stop request identity is invalid")
    for field in ("requested_by_pid", "supervisor_pid", "invocation_count"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"executor supervisor stop request {field} is invalid"
            )
    _normalized_supervisor_stop_reason(payload.get("reason"))
    report = dict(payload)
    report["request_path"] = str(path.resolve())
    return report


def _matching_supervisor_stop_request(
    path: Path,
    *,
    expected_identity: Mapping[str, object],
) -> dict[str, object] | None:
    try:
        request = load_hf_adapter_continuation_executor_supervisor_stop_request(path)
    except FileNotFoundError:
        return None
    for field, expected in expected_identity.items():
        if request.get(field) != expected:
            raise ValueError(
                "executor supervisor stop request identity differs for " f"{field}"
            )
    return request


def _supervisor_stop_request_identity(
    state: Mapping[str, object],
    run: Mapping[str, object],
    state_path: Path,
) -> dict[str, object]:
    return {
        "supervisor_run_id": run.get("supervisor_run_id"),
        "invocation_count": run.get("invocation_count"),
        "supervisor_pid": run.get("pid"),
        "supervisor_hostname": run.get("hostname"),
        "supervisor_state_path": str(state_path),
        "launch_state_path": state.get("launch_state_path"),
    }


def _launch_state_path(
    report_or_path: Mapping[str, object] | str | Path,
) -> Path:
    value = (
        report_or_path.get("launch_state_path")
        if isinstance(report_or_path, Mapping)
        else report_or_path
    )
    if value is None:
        raise ValueError("executor launch state path is required for supervision")
    path = Path(str(value)).expanduser()
    if path.is_symlink():
        raise ValueError(f"executor launch state cannot be a symbolic link: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _validate_launch_history(launch_state: Mapping[str, object]) -> None:
    raw_launches = launch_state.get("launches")
    if not isinstance(raw_launches, list) or any(
        not isinstance(row, Mapping) for row in raw_launches
    ):
        raise RuntimeError(
            "executor supervision history contains an invalid launch row"
        )
    if launch_state.get("launch_count") != len(raw_launches):
        raise RuntimeError("executor supervision launch_count is inconsistent")
    launch_ids = [row.get("launch_id") for row in raw_launches]
    if any(not isinstance(value, str) or not value for value in launch_ids):
        raise RuntimeError("executor supervision history contains an invalid launch ID")
    if len(set(launch_ids)) != len(launch_ids):
        raise RuntimeError("executor supervision history contains duplicate launch IDs")
    latest_launch_id = launch_state.get("latest_launch_id")
    if latest_launch_id is not None and raw_launches:
        if latest_launch_id != launch_ids[-1]:
            raise RuntimeError("executor supervision latest launch ID is inconsistent")


def _supervision_values(
    status: str,
    action: str,
    issue: str | None,
    *,
    resume_plan: Mapping[str, object] | None = None,
    resume_plan_error: str | None = None,
) -> dict[str, object]:
    return {
        "status": status,
        "action": action,
        "issue": issue,
        "resume_plan": None if resume_plan is None else dict(resume_plan),
        "resume_plan_error": resume_plan_error,
        "generation_plan_required": (
            None if resume_plan is None else resume_plan.get("generation_plan_required")
        ),
        "generation_plan_status": (
            None if resume_plan is None else resume_plan.get("generation_plan_status")
        ),
        "generation_plan_id": (
            None if resume_plan is None else resume_plan.get("generation_plan_id")
        ),
        "generation_plan_rebound": (
            None if resume_plan is None else resume_plan.get("generation_plan_rebound")
        ),
    }


def _launcher_supervision_gate(
    launch_status: Mapping[str, object],
    launcher_status: str,
) -> dict[str, object] | None:
    if launch_status.get("executor_handoff_observation") == "legacy_unverified":
        return _supervision_values(
            "blocked",
            "inspect_executor_handoff",
            "executor_handoff_unverified",
        )
    if (
        launcher_status in {"starting", "running"}
        and launch_status.get("healthy") is True
    ):
        return _supervision_values("waiting", "wait_for_executor", None)
    if launcher_status == "remote_running":
        return _supervision_values(
            "blocked", "inspect_remote_launcher", "remote_launcher_unverified"
        )
    if launcher_status == "running_unverified":
        return _supervision_values(
            "blocked", "inspect_unverified_launcher", "launcher_owner_unverified"
        )
    if launcher_status not in {"completed", "recoverable"}:
        return _supervision_values(
            "blocked",
            str(launch_status.get("recommended_action") or "inspect_executor_launcher"),
            f"launcher_{launcher_status}",
        )
    if launch_status.get("healthy") is not True:
        return _supervision_values(
            "blocked",
            str(launch_status.get("recommended_action") or "inspect_launcher_health"),
            "launcher_unhealthy",
        )
    return None


def _automatic_resume_supervision(
    launch_state_path: Path,
) -> dict[str, object]:
    try:
        resume_plan = hf_adapter_continuation_executor_resume_report(launch_state_path)
    except (OSError, RuntimeError, ValueError) as exc:
        return _supervision_values(
            "blocked",
            "inspect_resume_contract",
            "resume_plan_invalid",
            resume_plan_error=f"{exc.__class__.__name__}: {exc}",
        )
    if resume_plan.get("ready") is True:
        return _supervision_values(
            "resume_ready", "resume_executor", None, resume_plan=resume_plan
        )
    if (
        resume_plan.get("issue") == "executor_lock_unavailable"
        and resume_plan.get("executor_lock_status") == "alive"
    ):
        return _supervision_values(
            "waiting", "wait_for_executor_exit", None, resume_plan=resume_plan
        )
    return _supervision_values(
        "blocked",
        str(resume_plan.get("action") or "inspect_executor_resume"),
        str(resume_plan.get("issue") or "resume_not_ready"),
        resume_plan=resume_plan,
    )


def _terminal_supervision_decision(
    *,
    launch_state_path: Path,
    launch_status: Mapping[str, object],
    launcher_status: str,
    executor_lifecycle: object,
    executor_action: object,
    executor_reason: object,
) -> dict[str, object]:
    if executor_action == "stop_training":
        if (
            executor_lifecycle == "stopped"
            and executor_reason == "continuation_policy_stop"
        ):
            return _supervision_values("completed", "stop_training", None)
        return _supervision_values(
            "blocked",
            "inspect_executor_state",
            "executor_terminal_contract_invalid",
        )
    if executor_reason == "stop_requested":
        if executor_action != "resume_executor" or executor_lifecycle != "stopped":
            return _supervision_values(
                "blocked",
                "inspect_executor_state",
                "executor_terminal_contract_invalid",
            )
        return _supervision_values(
            "paused", "operator_resume_required", "operator_stop_boundary"
        )
    if launcher_status == "recoverable" or executor_lifecycle == "output_quarantined":
        return _supervision_values(
            "paused", "operator_resume_required", "recovery_boundary"
        )
    if (
        executor_action == "resume_executor"
        and executor_reason == _AUTOMATIC_RESUME_REASON
    ):
        if executor_lifecycle != "generation_limit_reached":
            return _supervision_values(
                "blocked",
                "inspect_executor_state",
                "executor_terminal_contract_invalid",
            )
        return _automatic_resume_supervision(launch_state_path)
    if executor_action == "resume_executor":
        return _supervision_values(
            "paused", "operator_resume_required", "manual_resume_boundary"
        )
    return _supervision_values(
        "blocked",
        str(launch_status.get("recommended_action") or "inspect_executor_state"),
        f"launcher_{launcher_status}",
    )


def hf_adapter_continuation_executor_supervision_report(
    report_or_path: Mapping[str, object] | str | Path,
) -> dict[str, object]:
    """Return one read-only automatic-supervision decision."""

    launch_state_path = _launch_state_path(report_or_path)
    launch_state = load_hf_adapter_continuation_executor_launch(launch_state_path)
    _validate_launch_history(launch_state)
    launch_status = hf_adapter_continuation_executor_launch_status_report(
        launch_state_path
    )
    executor_status = launch_status.get("executor_status")
    executor_status = (
        dict(executor_status) if isinstance(executor_status, Mapping) else None
    )
    launcher_status = str(launch_status.get("status") or "unknown")
    executor_lifecycle = (
        None if executor_status is None else executor_status.get("status")
    )
    executor_action = (
        None if executor_status is None else executor_status.get("executor_action")
    )
    executor_reason = (
        None if executor_status is None else executor_status.get("executor_reason")
    )
    decision = _launcher_supervision_gate(launch_status, launcher_status)
    if decision is None:
        decision = _terminal_supervision_decision(
            launch_state_path=launch_state_path,
            launch_status=launch_status,
            launcher_status=launcher_status,
            executor_lifecycle=executor_lifecycle,
            executor_action=executor_action,
            executor_reason=executor_reason,
        )
    status = str(decision["status"])
    action = str(decision["action"])
    issue = decision["issue"]
    resume_plan = decision.get("resume_plan")
    resume_plan = dict(resume_plan) if isinstance(resume_plan, Mapping) else None
    resume_plan_error = decision.get("resume_plan_error")

    healthy = status in {"completed", "paused", "resume_ready", "waiting"}
    latest_launch = launch_status.get("latest_launch")
    return {
        "row_type": "hf_adapter_continuation_executor_supervision",
        "schema": HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISION_SCHEMA,
        "created_at": _now(),
        "status": status,
        "healthy": healthy,
        "action": action,
        "issue": issue,
        "automatic_resume_allowed": status == "resume_ready",
        "automatic_resume_reason": _AUTOMATIC_RESUME_REASON,
        "launch_state_path": str(launch_state_path),
        "output_root": launch_state.get("output_root"),
        "launcher_status": launcher_status,
        "launcher_healthy": launch_status.get("healthy"),
        "launcher_process_observation": launch_status.get(
            "launcher_process_observation"
        ),
        "source_launch_id": (
            latest_launch.get("launch_id")
            if isinstance(latest_launch, Mapping)
            else None
        ),
        "launch_count": launch_status.get("launch_count"),
        "executor_status": executor_lifecycle,
        "executor_action": executor_action,
        "executor_reason": executor_reason,
        "executor_healthy": launch_status.get("executor_healthy"),
        "executor_run_id": (
            None if executor_status is None else executor_status.get("run_id")
        ),
        "executor_invocation_count": (
            None if executor_status is None else executor_status.get("invocation_count")
        ),
        "resume_plan_status": (
            None if resume_plan is None else resume_plan.get("status")
        ),
        "resume_plan_issue": (
            None if resume_plan is None else resume_plan.get("issue")
        ),
        "resume_plan_error": resume_plan_error,
        "generation_plan_required": decision.get("generation_plan_required"),
        "generation_plan_status": decision.get("generation_plan_status"),
        "generation_plan_id": decision.get("generation_plan_id"),
        "generation_plan_rebound": decision.get("generation_plan_rebound"),
    }


def hf_adapter_continuation_executor_supervision_lines(
    report_or_path: Mapping[str, object] | str | Path,
) -> list[str]:
    report = (
        dict(report_or_path)
        if isinstance(report_or_path, Mapping)
        and report_or_path.get("row_type")
        == "hf_adapter_continuation_executor_supervision"
        else hf_adapter_continuation_executor_supervision_report(report_or_path)
    )
    return [
        "hf_adapter_continuation_executor_supervision "
        f"status={report.get('status')} "
        f"healthy={report.get('healthy')} "
        f"action={report.get('action')} "
        f"issue={report.get('issue')} "
        f"automatic_resume={report.get('automatic_resume_allowed')} "
        f"launch={report.get('source_launch_id')} "
        f"invocation={report.get('executor_invocation_count')} "
        f"reason={report.get('executor_reason')} "
        f"plan_id={report.get('generation_plan_id')} "
        f"launch_state={report.get('launch_state_path')}"
    ]


def load_hf_adapter_continuation_executor_supervisor(
    value: str | Path,
) -> dict[str, object]:
    """Load and validate one durable supervisor history."""

    path = Path(value).expanduser()
    if path.is_symlink():
        raise ValueError(f"executor supervisor state cannot be a symlink: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"executor supervisor state must contain an object: {path}")
    if payload.get("schema") != HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_SCHEMA:
        raise ValueError(
            "unsupported HF adapter executor supervisor schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("row_type") != "hf_adapter_continuation_executor_supervisor":
        raise ValueError(
            "unsupported HF adapter executor supervisor row type: "
            f"{payload.get('row_type')}"
        )
    if not isinstance(payload.get("runs"), list):
        raise ValueError("executor supervisor runs must be a list")
    _validated_supervisor_runs(payload)
    report = dict(payload)
    report["supervisor_state_path"] = str(path.resolve())
    return report


def _load_supervisor_lock(path: Path) -> dict[str, object] | None:
    if path.is_symlink():
        raise RuntimeError(f"executor supervisor lock cannot be a symlink: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"executor supervisor lock is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"executor supervisor lock must contain an object: {path}")
    return dict(payload)


def _supervisor_lock_owner_is_stale(owner: Mapping[str, object]) -> bool:
    return bool(
        owner.get("row_type") == "hf_adapter_continuation_executor_supervisor_lock"
        and isinstance(owner.get("lock_id"), str)
        and owner.get("hostname") == socket.gethostname()
        and local_pid_alive(owner.get("pid")) is False
    )


def _reap_stale_supervisor_lock(path: Path) -> bool:
    reaper = path.with_name(f"{path.name}.reap")
    try:
        reaper.mkdir(mode=0o700)
    except FileExistsError:
        return False
    try:
        current = _load_supervisor_lock(path)
        if current is None or not _supervisor_lock_owner_is_stale(current):
            return False
        path.unlink()
        return True
    finally:
        reaper.rmdir()


@contextmanager
def _supervisor_lock(output_root: Path, state_path: Path) -> Iterator[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LOCK_FILENAME
    owner = {
        "row_type": "hf_adapter_continuation_executor_supervisor_lock",
        "lock_id": f"executor-supervisor-lock-{uuid.uuid4().hex}",
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "created_at": _now(),
        "supervisor_state_path": str(state_path),
    }
    encoded = (json.dumps(owner, ensure_ascii=True, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    acquired = False
    for _ in range(_SUPERVISOR_LOCK_RETRIES):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            existing = _load_supervisor_lock(path)
            if existing is None:
                continue
            if _supervisor_lock_owner_is_stale(existing):
                if not _reap_stale_supervisor_lock(path):
                    time.sleep(_SUPERVISOR_LOCK_RETRY_SECONDS)
                continue
            raise RuntimeError(
                "executor supervision is already owned; inspect the recorded owner: "
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
        raise RuntimeError(f"could not acquire executor supervisor lock: {path}")
    try:
        yield path
    finally:
        try:
            current = _load_supervisor_lock(path)
        except RuntimeError:
            current = None
        if current is not None and current.get("lock_id") == owner["lock_id"]:
            path.unlink(missing_ok=True)


def _new_supervisor_state(
    *,
    launch_state_path: Path,
    output_root: Path,
    supervisor_state_path: Path,
) -> dict[str, object]:
    created_at = _now()
    return {
        "row_type": "hf_adapter_continuation_executor_supervisor",
        "schema": HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_SCHEMA,
        "created_at": created_at,
        "updated_at": created_at,
        "status": "initializing",
        "action": "observe_executor",
        "healthy": True,
        "launch_state_path": str(launch_state_path),
        "output_root": str(output_root),
        "supervisor_state_path": str(supervisor_state_path),
        "invocation_count": 0,
        "total_resumes_started": 0,
        "runs": [],
    }


def _validate_supervisor_counters(
    state: Mapping[str, object],
    *,
    runs: list[Mapping[str, object]],
) -> None:
    expected_runs = len(runs)
    run_count = state.get("run_count")
    if run_count is not None and (
        isinstance(run_count, bool)
        or not isinstance(run_count, int)
        or run_count != expected_runs
    ):
        raise ValueError("executor supervisor run_count is inconsistent")
    invocation_count = state.get("invocation_count")
    if (
        isinstance(invocation_count, bool)
        or not isinstance(invocation_count, int)
        or invocation_count != expected_runs
    ):
        raise ValueError("executor supervisor invocation_count is inconsistent")
    total_resumes = state.get("total_resumes_started")
    if (
        isinstance(total_resumes, bool)
        or not isinstance(total_resumes, int)
        or total_resumes < 0
    ):
        raise ValueError("executor supervisor total resume count is invalid")
    counted_resumes = 0
    for run in runs:
        resumes_started = run.get("resumes_started")
        resume_events = run.get("resume_events")
        transitions = run.get("transitions")
        if (
            isinstance(resumes_started, bool)
            or not isinstance(resumes_started, int)
            or resumes_started < 0
            or not isinstance(resume_events, list)
            or any(not isinstance(event, Mapping) for event in resume_events)
            or not isinstance(transitions, list)
            or any(not isinstance(row, Mapping) for row in transitions)
        ):
            raise ValueError("executor supervisor resume history is invalid")
        if resumes_started != len(resume_events):
            raise ValueError("executor supervisor resume event count is inconsistent")
        transition_plan_ids = {
            transition.get("generation_plan_id")
            for transition in transitions
            if transition.get("status") == "resume_ready"
        }
        for event in resume_events:
            generation_plan_fields_present = any(
                field in event
                for field in (
                    "generation_plan_id",
                    "generation_plan_status",
                    "generation_plan_rebound",
                )
            )
            if not generation_plan_fields_present:
                continue
            generation_plan_id = event.get("generation_plan_id")
            if (
                not _valid_generation_plan_id(generation_plan_id)
                or event.get("generation_plan_status") != "ready"
                or not isinstance(event.get("generation_plan_rebound"), bool)
                or generation_plan_id not in transition_plan_ids
            ):
                raise ValueError(
                    "executor supervisor generation plan history is invalid"
                )
        counted_resumes += resumes_started
    if total_resumes != counted_resumes:
        raise ValueError("executor supervisor total resume count is inconsistent")


def _validate_supervisor_run_ids(
    state: Mapping[str, object],
    runs: list[Mapping[str, object]],
) -> None:
    run_ids = [row.get("supervisor_run_id") for row in runs]
    if any(not isinstance(value, str) or not value for value in run_ids):
        raise ValueError("executor supervisor history contains an invalid run ID")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("executor supervisor history contains duplicate run IDs")
    if runs and state.get("latest_supervisor_run_id") != run_ids[-1]:
        raise ValueError("executor supervisor latest run ID is inconsistent")


def _validated_supervisor_runs(
    state: Mapping[str, object],
) -> list[Mapping[str, object]]:
    runs = state.get("runs")
    if not isinstance(runs, list) or any(not isinstance(row, Mapping) for row in runs):
        raise ValueError("executor supervisor runs are invalid")
    _validate_supervisor_counters(state, runs=runs)
    _validate_supervisor_run_ids(state, runs)
    return runs


def _close_interrupted_supervisor_run(run: Mapping[str, object]) -> None:
    if run.get("status") != "running":
        return
    previous_hostname = run.get("hostname")
    previous_pid = run.get("pid")
    if not isinstance(previous_hostname, str) or not previous_hostname:
        raise RuntimeError("previous supervisor process identity is invalid")
    if previous_hostname != socket.gethostname():
        raise RuntimeError("previous supervisor process is remote and unverified")
    if local_pid_alive(previous_pid) is not False:
        raise RuntimeError("previous supervisor process may still be alive")
    if not isinstance(run, dict):
        raise RuntimeError("previous supervisor run is not mutable")
    run["status"] = "interrupted"
    run["action"] = "supervisor_restarted"
    run["process_liveness_observation"] = "exited"
    run["process_liveness_observed_at"] = _now()
    run["completed_at"] = _now()


def _supervisor_state_for_invocation(
    *,
    launch_state_path: Path,
    output_root: Path,
    supervisor_state_path: Path,
    max_resumes: int,
    poll_interval_seconds: float,
    timeout_seconds: float,
    handoff_timeout_seconds: float,
) -> tuple[dict[str, object], dict[str, object]]:
    state = (
        load_hf_adapter_continuation_executor_supervisor(supervisor_state_path)
        if supervisor_state_path.is_file()
        else _new_supervisor_state(
            launch_state_path=launch_state_path,
            output_root=output_root,
            supervisor_state_path=supervisor_state_path,
        )
    )
    if state.get("launch_state_path") != str(launch_state_path):
        raise ValueError("supervisor launch_state_path differs; use a new state")
    if state.get("output_root") != str(output_root):
        raise ValueError("supervisor output_root differs; use a new state")
    runs = _validated_supervisor_runs(state)
    if runs:
        _close_interrupted_supervisor_run(runs[-1])
    previous_stop_request = state.pop("stop_request", None)
    if isinstance(previous_stop_request, Mapping):
        history = state.setdefault("stop_request_history", [])
        if isinstance(history, list):
            history.append(dict(previous_stop_request))
    invocation_count = int(state.get("invocation_count") or 0) + 1
    supervisor_run_id = f"executor-supervisor-{uuid.uuid4().hex}"
    stop_request_path = _supervisor_stop_request_path(
        output_root,
        supervisor_run_id=supervisor_run_id,
        invocation_count=invocation_count,
    )
    run = {
        "supervisor_run_id": supervisor_run_id,
        "invocation_count": invocation_count,
        "status": "running",
        "action": "observe_executor",
        "healthy": True,
        "started_at": _now(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "max_resumes": max_resumes,
        "poll_interval_seconds": poll_interval_seconds,
        "timeout_seconds": timeout_seconds,
        "handoff_timeout_seconds": handoff_timeout_seconds,
        "stop_request_path": str(stop_request_path),
        "resumes_started": 0,
        "transitions": [],
        "resume_events": [],
    }
    runs.append(run)
    state["invocation_count"] = invocation_count
    state["latest_supervisor_run_id"] = run["supervisor_run_id"]
    state["status"] = "running"
    state["action"] = "observe_executor"
    state["healthy"] = True
    state.pop("completed_at", None)
    state.pop("reason", None)
    return state, run


def _write_supervisor_state(path: Path, state: dict[str, object]) -> None:
    state["updated_at"] = _now()
    state["run_count"] = len(state.get("runs") or [])
    _atomic_write_json(path, state)


def _decision_signature(decision: Mapping[str, object]) -> tuple[object, ...]:
    return (
        decision.get("status"),
        decision.get("action"),
        decision.get("issue"),
        decision.get("source_launch_id"),
        decision.get("executor_invocation_count"),
        decision.get("executor_reason"),
        decision.get("generation_plan_id"),
    )


def _supervisor_result(state: Mapping[str, object]) -> dict[str, object]:
    report = dict(state)
    runs = report.get("runs") or []
    report["latest_run"] = dict(runs[-1]) if runs else None
    return report


def _finish_supervisor(
    state: dict[str, object],
    run: dict[str, object],
    state_path: Path,
    *,
    status: str,
    action: str,
    healthy: bool,
    reason: str,
) -> dict[str, object]:
    completed_at = _now()
    run.update(
        {
            "status": status,
            "action": action,
            "healthy": healthy,
            "reason": reason,
            "completed_at": completed_at,
        }
    )
    state.update(
        {
            "status": status,
            "action": action,
            "healthy": healthy,
            "reason": reason,
            "completed_at": completed_at,
        }
    )
    _write_supervisor_state(state_path, state)
    return _supervisor_result(state)


def _validate_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_seconds(name: str, value: object, *, allow_zero: bool) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}"
        )
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0.0 or (not allow_zero and parsed == 0.0):
        raise ValueError(
            f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}"
        )
    return parsed


def _record_supervision_decision(
    state: dict[str, object],
    run: dict[str, object],
    state_path: Path,
    decision: Mapping[str, object],
    *,
    elapsed: float,
    previous_signature: tuple[object, ...] | None,
) -> tuple[object, ...]:
    signature = _decision_signature(decision)
    if signature != previous_signature:
        transitions = run["transitions"]
        if not isinstance(transitions, list):
            raise RuntimeError("executor supervisor transitions are invalid")
        transitions.append(
            {
                "observed_at": _now(),
                "elapsed_seconds": elapsed,
                **dict(decision),
            }
        )
    run["last_observed_at"] = _now()
    run["elapsed_seconds"] = elapsed
    run["last_decision"] = dict(decision)
    _write_supervisor_state(state_path, state)
    return signature


def _resume_once(
    state: dict[str, object],
    run: dict[str, object],
    state_path: Path,
    launch_state_path: Path,
    *,
    resume_budget: int,
    handoff_timeout: float,
) -> dict[str, object] | None:
    resumes_started = int(run.get("resumes_started") or 0)
    if resumes_started >= resume_budget:
        return _finish_supervisor(
            state,
            run,
            state_path,
            status="resume_budget_reached",
            action="run_supervisor_again",
            healthy=True,
            reason="max_resumes_per_supervisor_reached",
        )
    try:
        resume = resume_hf_adapter_continuation_executor(
            launch_state_path,
            handoff_timeout_seconds=handoff_timeout,
        )
    except Exception as exc:
        run["resume_error"] = f"{exc.__class__.__name__}: {exc}"
        return _finish_supervisor(
            state,
            run,
            state_path,
            status="blocked",
            action="inspect_executor_resume",
            healthy=False,
            reason="executor_resume_failed",
        )
    if resume.get("ready") is not True or resume.get("created") is not True:
        run["resume_result"] = dict(resume)
        return _finish_supervisor(
            state,
            run,
            state_path,
            status="blocked",
            action="inspect_executor_resume",
            healthy=False,
            reason="executor_resume_not_created",
        )
    run["resumes_started"] = resumes_started + 1
    state["total_resumes_started"] = int(state.get("total_resumes_started") or 0) + 1
    resume_events = run["resume_events"]
    if not isinstance(resume_events, list):
        raise RuntimeError("executor supervisor resume events are invalid")
    resume_events.append(
        {
            "resumed_at": _now(),
            "source_launch_id": resume.get("source_launch_id"),
            "resumed_launch_id": resume.get("resumed_launch_id"),
            "source_executor_invocation_count": resume.get("executor_invocation_count"),
            "resumed_executor_invocation_count": resume.get(
                "resumed_executor_invocation_count"
            ),
            "request_status": resume.get("status"),
            "generation_plan_id": resume.get("generation_plan_id"),
            "generation_plan_status": resume.get("generation_plan_status"),
            "generation_plan_rebound": resume.get("generation_plan_rebound"),
        }
    )
    _write_supervisor_state(state_path, state)
    return None


def _finish_from_supervision_decision(
    state: dict[str, object],
    run: dict[str, object],
    state_path: Path,
    decision: Mapping[str, object],
) -> dict[str, object]:
    if decision.get("status") == "completed":
        return _finish_supervisor(
            state,
            run,
            state_path,
            status="completed",
            action=str(decision.get("action") or "stop_training"),
            healthy=True,
            reason="executor_policy_completed",
        )
    if decision.get("status") == "paused":
        return _finish_supervisor(
            state,
            run,
            state_path,
            status="paused",
            action=str(decision.get("action") or "operator_resume_required"),
            healthy=True,
            reason=str(decision.get("issue") or "manual_boundary"),
        )
    return _finish_supervisor(
        state,
        run,
        state_path,
        status="blocked",
        action=str(decision.get("action") or "inspect_executor_state"),
        healthy=False,
        reason=str(decision.get("issue") or "supervision_blocked"),
    )


def _finish_if_supervisor_stop_requested(
    state: dict[str, object],
    run: dict[str, object],
    state_path: Path,
) -> dict[str, object] | None:
    try:
        stop_path = _state_supervisor_stop_request_path(state, run)
    except ValueError as exc:
        run["stop_request_error"] = f"{exc.__class__.__name__}: {exc}"
        return _finish_supervisor(
            state,
            run,
            state_path,
            status="blocked",
            action="inspect_supervisor_state",
            healthy=False,
            reason="supervisor_stop_request_path_invalid",
        )
    try:
        request = _matching_supervisor_stop_request(
            stop_path,
            expected_identity=_supervisor_stop_request_identity(
                state,
                run,
                state_path,
            ),
        )
    except (OSError, ValueError) as exc:
        run["stop_request_error"] = f"{exc.__class__.__name__}: {exc}"
        return _finish_supervisor(
            state,
            run,
            state_path,
            status="blocked",
            action="inspect_supervisor_stop_request",
            healthy=False,
            reason="supervisor_stop_request_invalid",
        )
    if request is None:
        return None
    run["stop_request"] = request
    state["stop_request"] = request
    return _finish_supervisor(
        state,
        run,
        state_path,
        status="stopped",
        action="operator_stopped_supervisor",
        healthy=True,
        reason="stop_requested",
    )


def _run_supervisor_loop(
    state: dict[str, object],
    run: dict[str, object],
    state_path: Path,
    launch_state_path: Path,
    *,
    resume_budget: int,
    poll_interval: float,
    timeout: float,
    handoff_timeout: float,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> dict[str, object]:
    started = monotonic()
    previous_signature: tuple[object, ...] | None = None
    try:
        while True:
            elapsed = max(0.0, monotonic() - started)
            stop_result = _finish_if_supervisor_stop_requested(
                state,
                run,
                state_path,
            )
            if stop_result is not None:
                return stop_result
            try:
                decision = hf_adapter_continuation_executor_supervision_report(
                    launch_state_path
                )
            except Exception as exc:
                run["observation_error"] = f"{exc.__class__.__name__}: {exc}"
                return _finish_supervisor(
                    state,
                    run,
                    state_path,
                    status="blocked",
                    action="inspect_supervision_error",
                    healthy=False,
                    reason="supervision_report_failed",
                )
            previous_signature = _record_supervision_decision(
                state,
                run,
                state_path,
                decision,
                elapsed=elapsed,
                previous_signature=previous_signature,
            )
            decision_status = decision.get("status")
            if decision_status not in {"waiting", "resume_ready"}:
                return _finish_from_supervision_decision(
                    state,
                    run,
                    state_path,
                    decision,
                )
            if timeout > 0.0 and elapsed >= timeout:
                return _finish_supervisor(
                    state,
                    run,
                    state_path,
                    status="timed_out",
                    action="inspect_executor_status",
                    healthy=False,
                    reason="supervisor_timeout_reached",
                )
            if decision_status == "waiting":
                wait_seconds = poll_interval
                if timeout > 0.0:
                    wait_seconds = min(wait_seconds, max(0.0, timeout - elapsed))
                sleep(wait_seconds)
                continue
            result = _resume_once(
                state,
                run,
                state_path,
                launch_state_path,
                resume_budget=resume_budget,
                handoff_timeout=handoff_timeout,
            )
            if result is not None:
                return result
    except BaseException as exc:
        run["interruption"] = f"{exc.__class__.__name__}: {exc}"
        _finish_supervisor(
            state,
            run,
            state_path,
            status="interrupted",
            action="restart_supervisor",
            healthy=False,
            reason="supervisor_process_interrupted",
        )
        raise


def supervise_hf_adapter_continuation_executor(
    report_or_path: Mapping[str, object] | str | Path,
    *,
    max_resumes: int = 1,
    poll_interval_seconds: float = 5.0,
    timeout_seconds: float = 0.0,
    handoff_timeout_seconds: float = 5.0,
    supervisor_state_path: str | Path | None = None,
    _sleep: Callable[[float], None] = time.sleep,
    _monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Supervise exact detached replays until policy, safety, or budget stops."""

    resume_budget = _validate_positive_int("max_resumes", max_resumes)
    poll_interval = _validate_seconds(
        "poll_interval_seconds", poll_interval_seconds, allow_zero=False
    )
    timeout = _validate_seconds("timeout_seconds", timeout_seconds, allow_zero=True)
    handoff_timeout = _validate_seconds(
        "handoff_timeout_seconds", handoff_timeout_seconds, allow_zero=True
    )
    launch_state_path = _launch_state_path(report_or_path)
    launch_state = load_hf_adapter_continuation_executor_launch(launch_state_path)
    output_root_value = launch_state.get("output_root")
    if not isinstance(output_root_value, str) or not output_root_value:
        raise ValueError("executor launch output_root is missing")
    executor_state_path_value = launch_state.get("executor_state_path")
    if not isinstance(executor_state_path_value, str) or not executor_state_path_value:
        raise ValueError("executor launch state path is missing")
    output_root = Path(output_root_value).expanduser().resolve()
    raw_state_path = (
        Path(supervisor_state_path).expanduser()
        if supervisor_state_path is not None
        else output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_FILENAME
    )
    if raw_state_path.is_symlink():
        raise ValueError(
            f"executor supervisor state cannot be a symlink: {raw_state_path}"
        )
    state_path = raw_state_path.resolve()
    reserved = {
        launch_state_path,
        Path(executor_state_path_value).expanduser().resolve(),
        output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME,
        output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_LOCK_FILENAME,
        output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LOCK_FILENAME,
    }
    if state_path in reserved:
        raise ValueError(
            "supervisor_state_path cannot overwrite executor state or locks"
        )

    with _supervisor_lock(output_root, state_path):
        state, run = _supervisor_state_for_invocation(
            launch_state_path=launch_state_path,
            output_root=output_root,
            supervisor_state_path=state_path,
            max_resumes=resume_budget,
            poll_interval_seconds=poll_interval,
            timeout_seconds=timeout,
            handoff_timeout_seconds=handoff_timeout,
        )
        _write_supervisor_state(state_path, state)
        return _run_supervisor_loop(
            state,
            run,
            state_path,
            launch_state_path,
            resume_budget=resume_budget,
            poll_interval=poll_interval,
            timeout=timeout,
            handoff_timeout=handoff_timeout,
            sleep=_sleep,
            monotonic=_monotonic,
        )


def _supervisor_process_observation(run: Mapping[str, object]) -> str:
    hostname = run.get("hostname")
    if not isinstance(hostname, str) or not hostname:
        return "unverified"
    if hostname != socket.gethostname():
        return "remote_unverified"
    alive = local_pid_alive(run.get("pid"))
    if alive is True:
        return "alive"
    if alive is False:
        return "exited"
    return "unverified"


def _supervisor_lock_observation(output_root: Path) -> dict[str, object]:
    path = output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LOCK_FILENAME
    try:
        owner = _load_supervisor_lock(path)
    except RuntimeError as exc:
        return {
            "path": str(path),
            "status": "invalid",
            "owner": None,
            "error": f"{exc.__class__.__name__}: {exc}",
        }
    if owner is None:
        return {"path": str(path), "status": "absent", "owner": None, "error": None}
    if (
        owner.get("row_type")
        != "hf_adapter_continuation_executor_supervisor_lock"
        or not isinstance(owner.get("lock_id"), str)
        or not owner.get("lock_id")
        or not isinstance(owner.get("hostname"), str)
        or not owner.get("hostname")
        or isinstance(owner.get("pid"), bool)
        or not isinstance(owner.get("pid"), int)
        or int(owner.get("pid")) <= 0
    ):
        return {
            "path": str(path),
            "status": "invalid",
            "owner": owner,
            "error": "executor supervisor lock owner identity is invalid",
        }
    return {
        "path": str(path),
        "status": _supervisor_process_observation(owner),
        "owner": owner,
        "error": None,
    }


def _supervisor_lock_targets_state(
    observation: Mapping[str, object],
    state_path: Path,
) -> bool:
    owner = observation.get("owner")
    if not isinstance(owner, Mapping):
        return False
    recorded = owner.get("supervisor_state_path")
    return bool(
        recorded is not None
        and Path(str(recorded)).expanduser().resolve() == state_path
    )


def _supervisor_lock_matches_run(
    observation: Mapping[str, object],
    run: Mapping[str, object],
    state_path: Path,
) -> bool:
    owner = observation.get("owner")
    return bool(
        isinstance(owner, Mapping)
        and _supervisor_lock_targets_state(observation, state_path)
        and owner.get("hostname") == run.get("hostname")
        and owner.get("pid") == run.get("pid")
    )


def _supervisor_state_and_path(
    report_or_path: Mapping[str, object] | str | Path,
) -> tuple[dict[str, object], Path]:
    state = (
        dict(report_or_path)
        if isinstance(report_or_path, Mapping)
        and report_or_path.get("row_type")
        == "hf_adapter_continuation_executor_supervisor"
        else load_hf_adapter_continuation_executor_supervisor(report_or_path)
    )
    path_value = state.get("supervisor_state_path")
    if path_value is None:
        raise ValueError("executor supervisor state path is missing")
    path = Path(str(path_value)).expanduser()
    if path.is_symlink():
        raise ValueError(f"executor supervisor state cannot be a symlink: {path}")
    return state, path.resolve()


def _supervisor_terminal_status(
    latest_status: str,
    *,
    lock_status: object,
    lock_matches_run: bool,
    lock_targets_state: bool,
) -> str:
    if lock_matches_run and lock_status == "alive":
        return "stopping"
    if lock_targets_state and lock_status == "alive":
        return "starting"
    if lock_targets_state and lock_status == "remote_unverified":
        return "remote_running"
    if lock_targets_state and lock_status in {"unverified", "invalid"}:
        return "running_unverified"
    if lock_status == "alive":
        return "ownership_conflict"
    return latest_status


def _supervisor_recommended_action(
    status: str,
    latest: Mapping[str, object] | None,
) -> str:
    actions = {
        "completed": "inspect_supervisor_result",
        "interrupted": "restart_supervisor",
        "ownership_conflict": "inspect_supervisor_ownership",
        "paused": "inspect_manual_boundary",
        "remote_running": "inspect_remote_supervisor",
        "resume_budget_reached": "run_supervisor_again",
        "running": "monitor_supervisor",
        "running_unverified": "inspect_supervisor_ownership",
        "starting": "wait_for_supervisor",
        "stopped": "restart_supervisor_if_needed",
        "stopping": "wait_for_supervisor",
    }
    if status in actions:
        return actions[status]
    if latest is not None and isinstance(latest.get("action"), str):
        return str(latest["action"])
    return "inspect_supervisor_state"


def _observed_supervisor_stop_request(
    state: Mapping[str, object],
    latest: Mapping[str, object] | None,
    state_path: Path,
) -> tuple[dict[str, object] | None, str | None, Path | None]:
    if latest is None:
        return None, None, None
    try:
        path = _state_supervisor_stop_request_path(state, latest)
        request = _matching_supervisor_stop_request(
            path,
            expected_identity=_supervisor_stop_request_identity(
                state,
                latest,
                state_path,
            ),
        )
    except (OSError, ValueError) as exc:
        return None, f"{exc.__class__.__name__}: {exc}", None
    return request, None, path


def _observed_supervisor_status(
    latest: Mapping[str, object] | None,
    *,
    process_observation: str,
    lock_status: object,
    lock_matches_run: bool,
    lock_targets_state: bool,
) -> str:
    if latest is None:
        return "empty"
    latest_status = str(latest.get("status") or "unknown")
    if latest_status != "running":
        return _supervisor_terminal_status(
            latest_status,
            lock_status=lock_status,
            lock_matches_run=lock_matches_run,
            lock_targets_state=lock_targets_state,
        )
    if lock_matches_run and lock_status == "alive":
        return "running"
    if lock_matches_run and lock_status == "remote_unverified":
        return "remote_running"
    if process_observation == "exited" and lock_status in {"absent", "exited"}:
        return "interrupted"
    return "running_unverified"


def _supervisor_health(
    state: Mapping[str, object],
    latest: Mapping[str, object] | None,
    *,
    status: str,
    stop_request_error: str | None,
) -> tuple[bool, list[str]]:
    healthy_statuses = {
        "completed",
        "paused",
        "resume_budget_reached",
        "running",
        "starting",
        "stopped",
        "stopping",
    }
    healthy = status in healthy_statuses
    if latest is not None and status not in {"running", "starting", "stopping"}:
        healthy = healthy and latest.get("healthy") is True
    issues: list[str] = []
    if stop_request_error is not None:
        issues.append("stop_request_invalid")
        healthy = False
    if state.get("status") != (None if latest is None else latest.get("status")):
        issues.append("state_latest_status_mismatch")
        healthy = False
    if status in {"remote_running", "running_unverified", "ownership_conflict"}:
        issues.append(status)
    return healthy, issues


def hf_adapter_continuation_executor_supervisor_status_report(
    report_or_path: Mapping[str, object] | str | Path,
) -> dict[str, object]:
    """Observe durable supervisor state, lock ownership, and stop control."""

    state, state_path = _supervisor_state_and_path(report_or_path)
    runs = _validated_supervisor_runs(state)
    latest = dict(runs[-1]) if runs else None
    output_root_value = state.get("output_root")
    if output_root_value is None:
        raise ValueError("executor supervisor output_root is missing")
    output_root = Path(str(output_root_value)).expanduser().resolve()
    lock = _supervisor_lock_observation(output_root)
    process_observation = (
        "missing" if latest is None else _supervisor_process_observation(latest)
    )
    lock_matches_run = bool(
        latest is not None
        and _supervisor_lock_matches_run(lock, latest, state_path)
    )
    lock_targets_state = _supervisor_lock_targets_state(lock, state_path)
    stop_request, stop_request_error, stop_request_path = (
        _observed_supervisor_stop_request(state, latest, state_path)
    )
    status = _observed_supervisor_status(
        latest,
        process_observation=process_observation,
        lock_status=lock.get("status"),
        lock_matches_run=lock_matches_run,
        lock_targets_state=lock_targets_state,
    )
    healthy, health_issues = _supervisor_health(
        state,
        latest,
        status=status,
        stop_request_error=stop_request_error,
    )
    resume_events = latest.get("resume_events") if latest is not None else None
    latest_resume = (
        resume_events[-1]
        if isinstance(resume_events, list)
        and resume_events
        and isinstance(resume_events[-1], Mapping)
        else None
    )
    return {
        "row_type": "hf_adapter_continuation_executor_supervisor_status",
        "schema": HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_STATUS_SCHEMA,
        "created_at": _now(),
        "status": status,
        "healthy": healthy,
        "recommended_action": _supervisor_recommended_action(status, latest),
        "health_issues": health_issues,
        "supervisor_state_path": str(state_path),
        "output_root": str(output_root),
        "invocation_count": state.get("invocation_count"),
        "total_resumes_started": state.get("total_resumes_started"),
        "latest_run": latest,
        "supervisor_run_id": (
            None if latest is None else latest.get("supervisor_run_id")
        ),
        "supervisor_process_observation": process_observation,
        "supervisor_pid_alive_observed": (
            True
            if process_observation == "alive"
            else False
            if process_observation == "exited"
            else None
        ),
        "supervisor_lock_owner_verified": lock_matches_run,
        "supervisor_lock": lock,
        "stop_requested": stop_request is not None,
        "stop_request": stop_request,
        "stop_request_error": stop_request_error,
        "stop_request_path": (
            None if stop_request_path is None else str(stop_request_path)
        ),
        "latest_resume_generation_plan_id": (
            None if latest_resume is None else latest_resume.get("generation_plan_id")
        ),
        "latest_resume_generation_plan_status": (
            None
            if latest_resume is None
            else latest_resume.get("generation_plan_status")
        ),
    }


def _normalized_supervisor_stop_reason(reason: object) -> str:
    raw_reason = str(reason)
    if any(ord(character) < 32 or ord(character) == 127 for character in raw_reason):
        raise ValueError("supervisor stop reason must not contain control characters")
    normalized = raw_reason.strip()
    if not normalized:
        raise ValueError("supervisor stop reason must not be empty")
    if len(normalized) > 512:
        raise ValueError("supervisor stop reason must be at most 512 characters")
    return normalized


def request_hf_adapter_continuation_executor_supervisor_stop(
    report_or_path: Mapping[str, object] | str | Path,
    *,
    reason: str = "operator_requested",
) -> dict[str, object]:
    """Request a cooperative supervisor stop without signalling a PID."""

    state, state_path = _supervisor_state_and_path(report_or_path)
    runs = _validated_supervisor_runs(state)
    if not runs:
        raise RuntimeError("executor supervisor has no run to stop")
    latest = runs[-1]
    stop_path = _state_supervisor_stop_request_path(state, latest)
    existing = _matching_supervisor_stop_request(
        stop_path,
        expected_identity=_supervisor_stop_request_identity(
            state,
            latest,
            state_path,
        ),
    )
    if existing is not None:
        report = dict(existing)
        report["created"] = False
        return report
    status = hf_adapter_continuation_executor_supervisor_status_report(state)
    if (
        status.get("status") != "running"
        or status.get("supervisor_lock_owner_verified") is not True
    ):
        raise RuntimeError(
            "executor supervisor ownership is not live and verified; no stop request "
            "was written"
        )
    payload = {
        "row_type": "hf_adapter_continuation_executor_supervisor_stop_request",
        "schema": HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_STOP_REQUEST_SCHEMA,
        "request_id": f"executor-supervisor-stop-{uuid.uuid4().hex}",
        "requested_at": _now(),
        "requested_by_pid": os.getpid(),
        "requested_by_hostname": socket.gethostname(),
        "reason": _normalized_supervisor_stop_reason(reason),
        "supervisor_run_id": latest.get("supervisor_run_id"),
        "invocation_count": latest.get("invocation_count"),
        "supervisor_pid": latest.get("pid"),
        "supervisor_hostname": latest.get("hostname"),
        "supervisor_state_path": str(state_path),
        "launch_state_path": state.get("launch_state_path"),
    }
    created = True
    try:
        _exclusive_write_json(stop_path, payload)
    except FileExistsError:
        existing = _matching_supervisor_stop_request(
            stop_path,
            expected_identity=_supervisor_stop_request_identity(
                state,
                latest,
                state_path,
            ),
        )
        if existing is None:
            raise RuntimeError(
                "executor supervisor stop request disappeared during inspection"
            )
        payload = existing
        created = False
    report = dict(payload)
    report["request_path"] = str(stop_path)
    report["created"] = created
    return report


def hf_adapter_continuation_executor_supervisor_status_lines(
    report_or_path: Mapping[str, object] | str | Path,
) -> list[str]:
    report = (
        dict(report_or_path)
        if isinstance(report_or_path, Mapping)
        and report_or_path.get("row_type")
        == "hf_adapter_continuation_executor_supervisor_status"
        else hf_adapter_continuation_executor_supervisor_status_report(report_or_path)
    )
    return [
        "hf_adapter_continuation_executor_supervisor_status "
        f"status={report.get('status')} "
        f"healthy={report.get('healthy')} "
        f"action={report.get('recommended_action')} "
        f"run={report.get('supervisor_run_id')} "
        f"invocation={report.get('invocation_count')} "
        f"pid_alive={report.get('supervisor_pid_alive_observed')} "
        f"lock_verified={report.get('supervisor_lock_owner_verified')} "
        f"stop_requested={report.get('stop_requested')} "
        f"plan_id={report.get('latest_resume_generation_plan_id')} "
        f"state={report.get('supervisor_state_path')}"
    ]


def hf_adapter_continuation_executor_supervisor_stop_request_lines(
    report_or_path: Mapping[str, object] | str | Path,
) -> list[str]:
    report = (
        dict(report_or_path)
        if isinstance(report_or_path, Mapping)
        else load_hf_adapter_continuation_executor_supervisor_stop_request(
            report_or_path
        )
    )
    return [
        "hf_adapter_continuation_executor_supervisor_stop_request "
        f"created={report.get('created')} "
        f"run={report.get('supervisor_run_id')} "
        f"invocation={report.get('invocation_count')} "
        f"reason={report.get('reason')} "
        f"request={report.get('request_path')}"
    ]


def hf_adapter_continuation_executor_supervisor_lines(
    report_or_path: Mapping[str, object] | str | Path,
) -> list[str]:
    report = (
        dict(report_or_path)
        if isinstance(report_or_path, Mapping)
        and report_or_path.get("row_type")
        == "hf_adapter_continuation_executor_supervisor"
        else load_hf_adapter_continuation_executor_supervisor(report_or_path)
    )
    latest = report.get("latest_run")
    if not isinstance(latest, Mapping):
        runs = report.get("runs") or []
        latest = runs[-1] if runs and isinstance(runs[-1], Mapping) else {}
    resume_events = latest.get("resume_events") or []
    latest_resume = (
        resume_events[-1]
        if isinstance(resume_events, list)
        and resume_events
        and isinstance(resume_events[-1], Mapping)
        else None
    )
    return [
        "hf_adapter_continuation_executor_supervisor "
        f"status={report.get('status')} "
        f"healthy={report.get('healthy')} "
        f"action={report.get('action')} "
        f"reason={report.get('reason')} "
        f"invocation={report.get('invocation_count')} "
        f"resumes={latest.get('resumes_started')} "
        f"total_resumes={report.get('total_resumes_started')} "
        f"plan_id={None if latest_resume is None else latest_resume.get('generation_plan_id')} "
        f"state={report.get('supervisor_state_path')}"
    ]
