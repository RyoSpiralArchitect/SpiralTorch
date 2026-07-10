"""Integrated runtime control for HF adapter continuation executors."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path

from .hf_adapter_executor import load_hf_adapter_continuation_executor
from .hf_adapter_executor_launch import (
    hf_adapter_continuation_executor_launch_status_report,
    load_hf_adapter_continuation_executor_launch,
)
from .hf_adapter_executor_status import (
    hf_adapter_continuation_executor_status_report,
)
from .hf_adapter_executor_supervisor import (
    HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_FILENAME,
    hf_adapter_continuation_executor_supervision_report,
    hf_adapter_continuation_executor_supervisor_status_report,
    load_hf_adapter_continuation_executor_supervisor,
)
from .hf_adapter_executor_supervisor_launch import (
    HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_FILENAME,
    hf_adapter_continuation_executor_supervisor_launch_status_report,
    launch_hf_adapter_continuation_executor_supervisor,
    load_hf_adapter_continuation_executor_supervisor_launch,
)

__all__ = [
    "HF_ADAPTER_CONTINUATION_EXECUTOR_RUNTIME_RECONCILE_SCHEMA",
    "HF_ADAPTER_CONTINUATION_EXECUTOR_RUNTIME_SCHEMA",
    "hf_adapter_continuation_executor_runtime_lines",
    "hf_adapter_continuation_executor_runtime_reconcile_lines",
    "hf_adapter_continuation_executor_runtime_report",
    "reconcile_hf_adapter_continuation_executor_runtime",
]


HF_ADAPTER_CONTINUATION_EXECUTOR_RUNTIME_SCHEMA = (
    "spiraltorch.hf_adapter_continuation_executor_runtime.v1"
)
HF_ADAPTER_CONTINUATION_EXECUTOR_RUNTIME_RECONCILE_SCHEMA = (
    "spiraltorch.hf_adapter_continuation_executor_runtime_reconcile.v1"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _input_launch_state_path(
    report_or_path: Mapping[str, object] | str | Path,
) -> Path:
    value = (
        report_or_path.get("launch_state_path")
        if isinstance(report_or_path, Mapping)
        else report_or_path
    )
    if value is None:
        raise ValueError("executor launch state path is required")
    path = Path(str(value)).expanduser()
    if path.is_symlink():
        raise ValueError(f"executor launch state cannot be a symlink: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _resolved_artifact_path(
    value: str | Path | None,
    default: Path,
    *,
    name: str,
) -> Path:
    path = Path(value).expanduser() if value is not None else default
    if path.is_symlink():
        raise ValueError(f"{name} cannot be a symlink: {path}")
    return path.resolve()


def _path_matches(value: object, expected: Path) -> bool:
    if value is None:
        return False
    try:
        observed = Path(str(value)).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return False
    return observed == expected


def _optional_report(
    path: Path,
    report: Callable[[str | Path], dict[str, object]],
) -> tuple[dict[str, object] | None, str | None]:
    if path.is_symlink():
        return None, f"ValueError: runtime artifact cannot be a symlink: {path}"
    if not path.is_file():
        return None, None
    try:
        return report(path), None
    except (OSError, RuntimeError, ValueError) as exc:
        return None, f"{exc.__class__.__name__}: {exc}"


def _runtime_paths(
    launch_state_path: Path,
    launch_state: Mapping[str, object],
    *,
    supervisor_state_path: str | Path | None,
    supervisor_launch_state_path: str | Path | None,
) -> dict[str, Path]:
    output_root_value = launch_state.get("output_root")
    executor_state_value = launch_state.get("executor_state_path")
    if not isinstance(output_root_value, str) or not output_root_value:
        raise ValueError("executor launch output_root is missing")
    if not isinstance(executor_state_value, str) or not executor_state_value:
        raise ValueError("executor launch state path is missing")
    output_root = Path(output_root_value).expanduser().resolve()
    executor_state = Path(executor_state_value).expanduser().resolve()
    supervisor_launch = _resolved_artifact_path(
        supervisor_launch_state_path,
        output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_FILENAME,
        name="supervisor_launch_state_path",
    )
    recorded_supervisor_state = None
    if supervisor_launch_state_path is None and supervisor_launch.is_file():
        try:
            preview = load_hf_adapter_continuation_executor_supervisor_launch(
                supervisor_launch
            )
        except (OSError, ValueError):
            preview = None
        if isinstance(preview, Mapping):
            recorded_supervisor_state = preview.get("supervisor_state_path")
    supervisor_state = _resolved_artifact_path(
        supervisor_state_path,
        (
            Path(str(recorded_supervisor_state)).expanduser()
            if recorded_supervisor_state is not None
            else output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_FILENAME
        ),
        name="supervisor_state_path",
    )
    paths = {
        "executor_launch_state_path": launch_state_path,
        "executor_state_path": executor_state,
        "output_root": output_root,
        "supervisor_state_path": supervisor_state,
        "supervisor_launch_state_path": supervisor_launch,
    }
    artifact_paths = {
        launch_state_path,
        executor_state,
        supervisor_state,
        supervisor_launch,
    }
    if len(artifact_paths) != 4:
        raise ValueError("runtime artifact paths must be distinct")
    return paths


def _identity_issues(
    paths: Mapping[str, Path],
    launch_state: Mapping[str, object],
    executor_state: Mapping[str, object] | None,
    supervisor_state: Mapping[str, object] | None,
    supervisor_launch: Mapping[str, object] | None,
) -> list[str]:
    issues: list[str] = []
    expected_launch = paths["executor_launch_state_path"]
    expected_executor = paths["executor_state_path"]
    expected_output = paths["output_root"]
    expected_supervisor = paths["supervisor_state_path"]
    expected_supervisor_launch = paths["supervisor_launch_state_path"]
    _append_identity_issues(
        issues,
        launch_state,
        (
            ("executor_launch.launch_state_path", "launch_state_path", expected_launch),
            (
                "executor_launch.executor_state_path",
                "executor_state_path",
                expected_executor,
            ),
            ("executor_launch.output_root", "output_root", expected_output),
        ),
    )
    if executor_state is not None:
        _append_identity_issues(
            issues,
            executor_state,
            (
                ("executor.state_path", "state_path", expected_executor),
                ("executor.output_root", "output_root", expected_output),
            ),
        )
    if supervisor_state is not None:
        _append_identity_issues(
            issues,
            supervisor_state,
            (
                (
                    "supervisor.state_path",
                    "supervisor_state_path",
                    expected_supervisor,
                ),
                (
                    "supervisor.launch_state_path",
                    "launch_state_path",
                    expected_launch,
                ),
                ("supervisor.output_root", "output_root", expected_output),
            ),
        )
    if supervisor_launch is not None:
        _append_identity_issues(
            issues,
            supervisor_launch,
            (
                (
                    "supervisor_launch.state_path",
                    "supervisor_launch_state_path",
                    expected_supervisor_launch,
                ),
                (
                    "supervisor_launch.executor_launch_state_path",
                    "executor_launch_state_path",
                    expected_launch,
                ),
                (
                    "supervisor_launch.supervisor_state_path",
                    "supervisor_state_path",
                    expected_supervisor,
                ),
                ("supervisor_launch.output_root", "output_root", expected_output),
            ),
        )
    return issues


def _append_identity_issues(
    issues: list[str],
    payload: Mapping[str, object],
    checks: tuple[tuple[str, str, Path], ...],
) -> None:
    for issue, field, expected in checks:
        if not _path_matches(payload.get(field), expected):
            issues.append(f"{issue}_mismatch")


def _decision_with_layers(
    decision: Mapping[str, object],
    *,
    source_status: object,
    executor_launch_status: object,
    supervisor_status: object,
    supervisor_launch_status: object,
) -> dict[str, object]:
    return {
        **dict(decision),
        "source_status": source_status,
        "executor_launch_status": executor_launch_status,
        "supervisor_status": supervisor_status,
        "supervisor_launch_status": supervisor_launch_status,
    }


def _waiting_runtime_decision(
    source_status: object,
    *,
    supervisor_present: bool,
    supervisor_launch_present: bool,
    supervisor_status: object,
    supervisor_launch_status: object,
    runtime_healthy: bool,
) -> dict[str, object]:
    if not supervisor_present and not supervisor_launch_present:
        return {
            "status": (
                "unmanaged_running"
                if source_status == "waiting"
                else "unmanaged_resume_ready"
            ),
            "healthy": runtime_healthy,
            "managed": False,
            "operational_ready": False,
            "requires_operator": False,
            "recommended_action": "launch_supervisor",
            "reconcile_action": "launch_supervisor",
            "reconcile_requires_restart": False,
        }
    if not supervisor_present:
        return {
            "status": f"supervisor_launch_{supervisor_launch_status or 'unknown'}",
            "healthy": False,
            "managed": False,
            "operational_ready": False,
            "requires_operator": True,
            "recommended_action": ("restart_supervisor_with_explicit_operator_intent"),
            "reconcile_action": "restart_supervisor",
            "reconcile_requires_restart": True,
        }
    if supervisor_status in {
        "interrupted",
        "resume_budget_reached",
        "stopped",
        "timed_out",
    }:
        return {
            "status": f"supervisor_{supervisor_status}",
            "healthy": runtime_healthy
            and supervisor_status in {"resume_budget_reached", "stopped"},
            "managed": False,
            "operational_ready": False,
            "requires_operator": True,
            "recommended_action": ("restart_supervisor_with_explicit_operator_intent"),
            "reconcile_action": "restart_supervisor",
            "reconcile_requires_restart": True,
        }
    return {
        "status": f"control_{supervisor_status or 'unknown'}",
        "healthy": False,
        "managed": False,
        "operational_ready": False,
        "requires_operator": True,
        "recommended_action": "inspect_supervisor_terminal_state",
        "reconcile_action": "none",
        "reconcile_requires_restart": False,
    }


def _runtime_decision(
    *,
    supervision: Mapping[str, object] | None,
    executor_launch: Mapping[str, object] | None,
    supervisor: Mapping[str, object] | None,
    supervisor_launch: Mapping[str, object] | None,
    integrity_issues: list[str],
    health_issues: list[str],
) -> dict[str, object]:
    source_status = None if supervision is None else supervision.get("status")
    executor_launch_status = (
        None if executor_launch is None else executor_launch.get("status")
    )
    supervisor_status = None if supervisor is None else supervisor.get("status")
    supervisor_launch_status = (
        None if supervisor_launch is None else supervisor_launch.get("status")
    )
    source_healthy = bool(
        supervision is not None and supervision.get("healthy") is True
    )
    runtime_healthy = source_healthy and not health_issues
    live_control = supervisor_status in {"running", "starting", "stopping"}
    if integrity_issues:
        status = "invalid"
        managed = False
        operational_ready = False
        requires_operator = True
        reconcile_action = "none"
        reconcile_requires_restart = False
        recommended_action = "inspect_runtime_health_issues"
        runtime_healthy = False
    elif live_control:
        status = f"managed_{supervisor_status}"
        managed = True
        operational_ready = runtime_healthy
        requires_operator = not runtime_healthy
        reconcile_action = "none"
        reconcile_requires_restart = False
        recommended_action = (
            "inspect_runtime_health_issues"
            if not runtime_healthy
            else (
                "monitor_runtime"
                if supervisor_status == "running"
                else "wait_for_supervisor"
            )
        )
    elif supervisor_status in {
        "ownership_conflict",
        "remote_running",
        "running_unverified",
    } or supervisor_launch_status in {"remote_running", "running_unverified"}:
        status = "control_unverified"
        managed = False
        operational_ready = False
        requires_operator = True
        reconcile_action = "none"
        reconcile_requires_restart = False
        recommended_action = "inspect_supervisor_ownership"
        runtime_healthy = False
    elif source_status == "completed":
        status = "completed"
        managed = False
        operational_ready = runtime_healthy
        requires_operator = False
        reconcile_action = "none"
        reconcile_requires_restart = False
        recommended_action = "inspect_executor_result"
    elif source_status == "paused":
        status = "paused"
        managed = False
        operational_ready = runtime_healthy
        requires_operator = True
        reconcile_action = "none"
        reconcile_requires_restart = False
        recommended_action = str(supervision.get("action") or "inspect_manual_boundary")
    elif source_status == "blocked":
        status = "blocked"
        managed = False
        operational_ready = False
        requires_operator = True
        reconcile_action = "none"
        reconcile_requires_restart = False
        recommended_action = str(supervision.get("action") or "inspect_runtime_block")
        runtime_healthy = False
    elif source_status in {"waiting", "resume_ready"}:
        return _decision_with_layers(
            _waiting_runtime_decision(
                source_status,
                supervisor_present=supervisor is not None,
                supervisor_launch_present=supervisor_launch is not None,
                supervisor_status=supervisor_status,
                supervisor_launch_status=supervisor_launch_status,
                runtime_healthy=runtime_healthy,
            ),
            source_status=source_status,
            executor_launch_status=executor_launch_status,
            supervisor_status=supervisor_status,
            supervisor_launch_status=supervisor_launch_status,
        )
    else:
        status = str(source_status or executor_launch_status or "unknown")
        managed = False
        operational_ready = False
        requires_operator = True
        reconcile_action = "none"
        reconcile_requires_restart = False
        recommended_action = "inspect_runtime_state"
        runtime_healthy = False
    return _decision_with_layers(
        {
            "status": status,
            "healthy": runtime_healthy,
            "managed": managed,
            "operational_ready": operational_ready,
            "requires_operator": requires_operator,
            "recommended_action": recommended_action,
            "reconcile_action": reconcile_action,
            "reconcile_requires_restart": reconcile_requires_restart,
        },
        source_status=source_status,
        executor_launch_status=executor_launch_status,
        supervisor_status=supervisor_status,
        supervisor_launch_status=supervisor_launch_status,
    )


def hf_adapter_continuation_executor_runtime_report(
    report_or_path: Mapping[str, object] | str | Path,
    *,
    supervisor_state_path: str | Path | None = None,
    supervisor_launch_state_path: str | Path | None = None,
) -> dict[str, object]:
    """Observe the full executor and supervisor runtime without mutating it."""

    launch_state_path = _input_launch_state_path(report_or_path)
    launch_state = load_hf_adapter_continuation_executor_launch(launch_state_path)
    paths = _runtime_paths(
        launch_state_path,
        launch_state,
        supervisor_state_path=supervisor_state_path,
        supervisor_launch_state_path=supervisor_launch_state_path,
    )
    executor_state, executor_state_error = _optional_report(
        paths["executor_state_path"],
        load_hf_adapter_continuation_executor,
    )
    supervisor_state, supervisor_state_error = _optional_report(
        paths["supervisor_state_path"],
        load_hf_adapter_continuation_executor_supervisor,
    )
    supervisor_launch, supervisor_launch_error = _optional_report(
        paths["supervisor_launch_state_path"],
        load_hf_adapter_continuation_executor_supervisor_launch,
    )
    executor_status, executor_status_error = _optional_report(
        paths["executor_state_path"],
        hf_adapter_continuation_executor_status_report,
    )
    supervisor_status, supervisor_status_error = _optional_report(
        paths["supervisor_state_path"],
        hf_adapter_continuation_executor_supervisor_status_report,
    )
    supervisor_launch_status, supervisor_launch_status_error = _optional_report(
        paths["supervisor_launch_state_path"],
        hf_adapter_continuation_executor_supervisor_launch_status_report,
    )
    try:
        executor_launch_status = hf_adapter_continuation_executor_launch_status_report(
            launch_state_path
        )
        executor_launch_status_error = None
    except (OSError, RuntimeError, ValueError) as exc:
        executor_launch_status = None
        executor_launch_status_error = f"{exc.__class__.__name__}: {exc}"
    try:
        supervision = hf_adapter_continuation_executor_supervision_report(
            launch_state_path
        )
        supervision_error = None
    except (OSError, RuntimeError, ValueError) as exc:
        supervision = None
        supervision_error = f"{exc.__class__.__name__}: {exc}"

    identity_issues = _identity_issues(
        paths,
        launch_state,
        executor_state,
        supervisor_state,
        supervisor_launch,
    )
    errors = {
        "executor_state": executor_state_error,
        "executor_status": executor_status_error,
        "executor_launch_status": executor_launch_status_error,
        "supervision": supervision_error,
        "supervisor_state": supervisor_state_error,
        "supervisor_status": supervisor_status_error,
        "supervisor_launch_state": supervisor_launch_error,
        "supervisor_launch_status": supervisor_launch_status_error,
    }
    integrity_issues = list(identity_issues)
    integrity_issues.extend(
        f"{name}_invalid" for name, error in errors.items() if error is not None
    )
    health_issues = list(integrity_issues)
    for prefix, nested in (
        ("executor", executor_status),
        ("supervisor", supervisor_status),
    ):
        if isinstance(nested, Mapping):
            health_issues.extend(
                f"{prefix}:{issue}" for issue in nested.get("health_issues") or []
            )
    if (
        isinstance(supervisor_launch_status, Mapping)
        and supervisor_launch_status.get("status")
        in {"running", "starting", "stopping", "superseded"}
        and supervisor_launch_status.get("healthy") is not True
    ):
        health_issues.append(
            f"supervisor_launch:{supervisor_launch_status.get('status')}_unhealthy"
        )
    health_issues = list(dict.fromkeys(health_issues))
    decision = _runtime_decision(
        supervision=supervision,
        executor_launch=executor_launch_status,
        supervisor=supervisor_status,
        supervisor_launch=supervisor_launch_status,
        integrity_issues=integrity_issues,
        health_issues=health_issues,
    )
    return {
        "row_type": "hf_adapter_continuation_executor_runtime",
        "schema": HF_ADAPTER_CONTINUATION_EXECUTOR_RUNTIME_SCHEMA,
        "created_at": _now(),
        **decision,
        "identity_verified": not identity_issues,
        "integrity_verified": not integrity_issues,
        "integrity_issues": integrity_issues,
        "reconcile_safe": bool(
            not health_issues
            and decision["status"] not in {"blocked", "control_unverified", "invalid"}
        ),
        "identity_issues": identity_issues,
        "health_issue_count": len(health_issues),
        "health_issues": health_issues,
        "artifact_errors": errors,
        "executor_launch_state_path": str(paths["executor_launch_state_path"]),
        "executor_state_path": str(paths["executor_state_path"]),
        "output_root": str(paths["output_root"]),
        "supervisor_state_path": str(paths["supervisor_state_path"]),
        "supervisor_launch_state_path": str(paths["supervisor_launch_state_path"]),
        "executor_launch": executor_launch_status,
        "executor": executor_status,
        "supervision": supervision,
        "supervisor": supervisor_status,
        "supervisor_launch": supervisor_launch_status,
    }


def _validate_reconcile_parameters(
    *,
    max_resumes: object,
    poll_interval_seconds: object,
    timeout_seconds: object,
    handoff_timeout_seconds: object,
    launch_handoff_timeout_seconds: object,
) -> None:
    if (
        isinstance(max_resumes, bool)
        or not isinstance(max_resumes, int)
        or max_resumes <= 0
    ):
        raise ValueError("max_resumes must be a positive integer")
    for name, value, allow_zero in (
        ("poll_interval_seconds", poll_interval_seconds, False),
        ("timeout_seconds", timeout_seconds, True),
        ("handoff_timeout_seconds", handoff_timeout_seconds, True),
        ("launch_handoff_timeout_seconds", launch_handoff_timeout_seconds, True),
    ):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be finite")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite") from exc
        if (
            not math.isfinite(parsed)
            or parsed < 0.0
            or (not allow_zero and parsed == 0.0)
        ):
            raise ValueError(
                f"{name} must be finite and "
                f"{'non-negative' if allow_zero else 'positive'}"
            )


def reconcile_hf_adapter_continuation_executor_runtime(
    report_or_path: Mapping[str, object] | str | Path,
    *,
    restart_supervisor: bool = False,
    max_resumes: int = 1,
    poll_interval_seconds: float = 5.0,
    timeout_seconds: float = 0.0,
    handoff_timeout_seconds: float = 5.0,
    launch_handoff_timeout_seconds: float = 5.0,
    supervisor_state_path: str | Path | None = None,
    supervisor_launch_state_path: str | Path | None = None,
    command_cwd: str | Path | None = None,
) -> dict[str, object]:
    """Reconcile one runtime while preserving every explicit stop boundary."""

    if not isinstance(restart_supervisor, bool):
        raise ValueError("restart_supervisor must be a boolean")
    _validate_reconcile_parameters(
        max_resumes=max_resumes,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        handoff_timeout_seconds=handoff_timeout_seconds,
        launch_handoff_timeout_seconds=launch_handoff_timeout_seconds,
    )
    before = hf_adapter_continuation_executor_runtime_report(
        report_or_path,
        supervisor_state_path=supervisor_state_path,
        supervisor_launch_state_path=supervisor_launch_state_path,
    )
    action = before.get("reconcile_action")
    requires_restart = before.get("reconcile_requires_restart") is True
    launch = None
    if (
        before.get("integrity_verified") is not True
        or before.get("reconcile_safe") is not True
    ):
        request_status = "blocked"
        succeeded = False
        created = False
    elif action == "restart_supervisor" and not restart_supervisor:
        request_status = "operator_restart_required"
        succeeded = False
        created = False
    elif action in {"launch_supervisor", "restart_supervisor"}:
        launch = launch_hf_adapter_continuation_executor_supervisor(
            before["executor_launch_state_path"],
            max_resumes=max_resumes,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=timeout_seconds,
            handoff_timeout_seconds=handoff_timeout_seconds,
            launch_handoff_timeout_seconds=launch_handoff_timeout_seconds,
            supervisor_state_path=before["supervisor_state_path"],
            supervisor_launch_state_path=before["supervisor_launch_state_path"],
            command_cwd=command_cwd,
        )
        request_status = str(launch.get("request_status") or "unknown")
        created = launch.get("created") is True
        succeeded = request_status in {
            "already_running",
            "completed",
            "handed_off",
        }
    else:
        request_status = (
            "already_managed" if before.get("managed") is True else "no_action"
        )
        succeeded = bool(
            before.get("operational_ready") is True
            or before.get("status") in {"completed", "paused"}
        )
        created = False
    after = hf_adapter_continuation_executor_runtime_report(
        before["executor_launch_state_path"],
        supervisor_state_path=before["supervisor_state_path"],
        supervisor_launch_state_path=before["supervisor_launch_state_path"],
    )
    return {
        "row_type": "hf_adapter_continuation_executor_runtime_reconcile",
        "schema": HF_ADAPTER_CONTINUATION_EXECUTOR_RUNTIME_RECONCILE_SCHEMA,
        "created_at": _now(),
        "request_status": request_status,
        "succeeded": succeeded,
        "created": created,
        "restart_supervisor_requested": restart_supervisor,
        "restart_supervisor_required": requires_restart,
        "action": action,
        "before": before,
        "after": after,
        "supervisor_launch": launch,
    }


def hf_adapter_continuation_executor_runtime_lines(
    report_or_path: Mapping[str, object] | str | Path,
    *,
    supervisor_state_path: str | Path | None = None,
    supervisor_launch_state_path: str | Path | None = None,
) -> list[str]:
    report = (
        dict(report_or_path)
        if isinstance(report_or_path, Mapping)
        and report_or_path.get("row_type") == "hf_adapter_continuation_executor_runtime"
        else hf_adapter_continuation_executor_runtime_report(
            report_or_path,
            supervisor_state_path=supervisor_state_path,
            supervisor_launch_state_path=supervisor_launch_state_path,
        )
    )
    executor = report.get("executor")
    return [
        "hf_adapter_continuation_executor_runtime "
        f"status={report.get('status')} "
        f"healthy={report.get('healthy')} "
        f"ready={report.get('operational_ready')} "
        f"managed={report.get('managed')} "
        f"operator={report.get('requires_operator')} "
        f"identity={report.get('identity_verified')} "
        f"issues={report.get('health_issue_count')} "
        f"action={report.get('recommended_action')} "
        f"source={report.get('source_status')} "
        f"executor_launch={report.get('executor_launch_status')} "
        f"supervisor={report.get('supervisor_status')} "
        f"supervisor_launch={report.get('supervisor_launch_status')} "
        "transition_evidence="
        f"{executor.get('transition_evidence_status') if isinstance(executor, Mapping) else None} "
        f"state={report.get('executor_launch_state_path')}"
    ]


def hf_adapter_continuation_executor_runtime_reconcile_lines(
    report: Mapping[str, object],
) -> list[str]:
    before = report.get("before")
    after = report.get("after")
    return [
        "hf_adapter_continuation_executor_runtime_reconcile "
        f"request={report.get('request_status')} "
        f"succeeded={report.get('succeeded')} "
        f"created={report.get('created')} "
        f"action={report.get('action')} "
        f"restart_requested={report.get('restart_supervisor_requested')} "
        f"restart_required={report.get('restart_supervisor_required')} "
        f"before={before.get('status') if isinstance(before, Mapping) else None} "
        f"after={after.get('status') if isinstance(after, Mapping) else None} "
        "state="
        f"{after.get('executor_launch_state_path') if isinstance(after, Mapping) else None}"
    ]
