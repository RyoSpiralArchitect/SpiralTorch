"""Read-only live status for Hugging Face adapter continuation executors."""

from __future__ import annotations

import socket
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from ._process import local_pid_alive
from .hf_adapter_executor import (
    HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME,
    load_hf_adapter_continuation_executor,
)

__all__ = [
    "HF_ADAPTER_CONTINUATION_EXECUTOR_STATUS_SCHEMA",
    "hf_adapter_continuation_executor_status_lines",
    "hf_adapter_continuation_executor_status_report",
]


HF_ADAPTER_CONTINUATION_EXECUTOR_STATUS_SCHEMA = (
    "spiraltorch.hf_adapter_continuation_executor_status.v1"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _path_report(value: object, *, expect_directory: bool) -> dict[str, object]:
    if value is None:
        return {
            "path": None,
            "exists": False,
            "kind_ready": False,
            "size_bytes": None,
            "modified_at": None,
            "age_seconds": None,
        }
    path = Path(str(value)).expanduser()
    exists = path.exists()
    kind_ready = path.is_dir() if expect_directory else path.is_file()
    size_bytes = None
    modified_at = None
    age_seconds = None
    if exists:
        try:
            stat = path.stat()
        except OSError:
            pass
        else:
            size_bytes = stat.st_size if path.is_file() else None
            modified_at = datetime.fromtimestamp(
                stat.st_mtime,
                tz=timezone.utc,
            ).isoformat()
            age_seconds = max(
                0.0, datetime.now(timezone.utc).timestamp() - stat.st_mtime
            )
    return {
        "path": str(path.resolve(strict=False)),
        "exists": exists,
        "kind_ready": kind_ready,
        "size_bytes": size_bytes,
        "modified_at": modified_at,
        "age_seconds": age_seconds,
    }


def _attempt_summary(attempt: Mapping[str, object] | None) -> dict[str, object] | None:
    if attempt is None:
        return None
    return {
        "attempt_id": attempt.get("attempt_id"),
        "status": attempt.get("status"),
        "lineage_depth": attempt.get("lineage_depth"),
        "runner_kind": attempt.get("runner_kind"),
        "hostname": attempt.get("hostname"),
        "pid": attempt.get("pid"),
        "started_at": attempt.get("started_at"),
        "process_started_at": attempt.get("process_started_at"),
        "process_exited_at": attempt.get("process_exited_at"),
        "process_liveness_observed_at": attempt.get("process_liveness_observed_at"),
        "process_liveness_observation": attempt.get("process_liveness_observation"),
        "last_output_at": attempt.get("last_output_at"),
        "log_bytes_observed": attempt.get("log_bytes_observed"),
        "completed_at": attempt.get("completed_at"),
        "returncode": attempt.get("returncode"),
        "command_cwd": attempt.get("command_cwd"),
        "output_dir": attempt.get("output_dir"),
        "log_path": attempt.get("log_path"),
    }


def hf_adapter_continuation_executor_status_report(
    report_or_path: Mapping[str, object] | str | Path,
) -> dict[str, object]:
    """Observe executor/process/artifact state without mutating the executor."""

    state = (
        dict(report_or_path)
        if isinstance(report_or_path, Mapping)
        else load_hf_adapter_continuation_executor(report_or_path)
    )
    generations = [
        row for row in state.get("generations") or [] if isinstance(row, Mapping)
    ]
    running = [row for row in generations if row.get("status") == "running"]
    active = running[-1] if running else None
    latest = generations[-1] if generations else None
    current_hostname = socket.gethostname()
    attempt_hostname = None if active is None else active.get("hostname")
    same_host = (
        None
        if active is None or not isinstance(attempt_hostname, str)
        else attempt_hostname == current_hostname
    )
    pid = None if active is None else active.get("pid")
    pid_alive = local_pid_alive(pid) if same_host is True else None

    executor_status = str(state.get("status") or "unknown")
    if active is None and executor_status == "running":
        observed_status = "running_unverified"
    elif active is None:
        observed_status = executor_status
    elif same_host is False:
        observed_status = "remote_running"
    elif same_host is None or pid_alive is None:
        observed_status = "running_unverified"
    elif pid_alive:
        observed_status = "running"
    else:
        observed_status = "interrupted"

    if observed_status == "running":
        recommended_action = "monitor_running_process"
    elif observed_status == "remote_running":
        recommended_action = "inspect_remote_process"
    elif observed_status == "running_unverified":
        recommended_action = "inspect_unverified_process"
    elif observed_status == "interrupted":
        recommended_action = "recover_interrupted_generation"
    elif observed_status == "failed":
        recommended_action = "inspect_executor_failure"
    elif observed_status == "blocked":
        recommended_action = "resolve_executor_block"
    else:
        recommended_action = "none"

    lifecycle_healthy = observed_status in {
        "auditing",
        "generation_limit_reached",
        "ready",
        "running",
        "stopped",
    }
    updated_at = _timestamp(state.get("updated_at"))
    state_age_seconds = (
        None
        if updated_at is None
        else max(0.0, (datetime.now(timezone.utc) - updated_at).total_seconds())
    )
    observed_attempt = active or latest
    output = _path_report(
        None if observed_attempt is None else observed_attempt.get("output_dir"),
        expect_directory=True,
    )
    log = _path_report(
        None if observed_attempt is None else observed_attempt.get("log_path"),
        expect_directory=False,
    )
    output_root = state.get("output_root")
    execution = state.get("execution")
    lock_path = execution.get("lock_path") if isinstance(execution, Mapping) else None
    lock = _path_report(
        lock_path
        if lock_path is not None
        else (
            None
            if output_root is None
            else Path(str(output_root)) / HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
        ),
        expect_directory=False,
    )
    health_issues: list[str] = []
    if len(running) > 1:
        health_issues.append("multiple_running_attempts")
    if active is not None and lock.get("kind_ready") is not True:
        health_issues.append("single_writer_lock_missing")
    if (
        observed_attempt is not None
        and observed_attempt.get("runner_kind") == "subprocess"
        and log.get("kind_ready") is not True
    ):
        health_issues.append("executor_log_missing")
    if (
        observed_attempt is not None
        and observed_attempt.get("status") in {"promoted", "promoted_recovered"}
        and output.get("kind_ready") is not True
    ):
        health_issues.append("promoted_output_missing")
    healthy = lifecycle_healthy and not health_issues
    if health_issues and recommended_action in {"monitor_running_process", "none"}:
        recommended_action = "inspect_executor_health_issues"
    return {
        "row_type": "hf_adapter_continuation_executor_status",
        "schema": HF_ADAPTER_CONTINUATION_EXECUTOR_STATUS_SCHEMA,
        "created_at": _now(),
        "status": observed_status,
        "healthy": healthy,
        "health_issue_count": len(health_issues),
        "health_issues": health_issues,
        "recommended_action": recommended_action,
        "executor_status": executor_status,
        "executor_action": state.get("action"),
        "executor_reason": state.get("reason"),
        "state_path": state.get("state_path"),
        "state_updated_at": state.get("updated_at"),
        "state_age_seconds": state_age_seconds,
        "run_id": state.get("run_id"),
        "invocation_count": state.get("invocation_count"),
        "generation_attempt_count": len(generations),
        "promoted_generation_count": state.get("promoted_generation_count"),
        "selected_lineage_depth": state.get("selected_lineage_depth"),
        "active_attempt_count": len(running),
        "active_attempt": _attempt_summary(active),
        "latest_attempt": _attempt_summary(latest),
        "current_hostname": current_hostname,
        "attempt_hostname": attempt_hostname,
        "same_host": same_host,
        "pid": pid,
        "pid_alive_observed": pid_alive,
        "process_identity_verified": False,
        "output": output,
        "log": log,
        "lock": lock,
    }


def hf_adapter_continuation_executor_status_lines(
    report_or_path: Mapping[str, object] | str | Path,
) -> list[str]:
    report = (
        dict(report_or_path)
        if isinstance(report_or_path, Mapping)
        and report_or_path.get("row_type") == "hf_adapter_continuation_executor_status"
        else hf_adapter_continuation_executor_status_report(report_or_path)
    )
    lines = [
        (
            "hf_adapter_continuation_executor_status "
            f"status={report.get('status')} "
            f"healthy={report.get('healthy')} "
            f"health_issues={report.get('health_issue_count')} "
            f"executor={report.get('executor_status')} "
            f"action={report.get('recommended_action')} "
            f"depth={report.get('selected_lineage_depth')} "
            f"attempts={report.get('generation_attempt_count')} "
            f"promoted={report.get('promoted_generation_count')} "
            f"state_age_seconds={report.get('state_age_seconds')} "
            f"state={report.get('state_path')}"
        )
    ]
    attempt = report.get("active_attempt") or report.get("latest_attempt")
    if isinstance(attempt, Mapping):
        log = report.get("log")
        output = report.get("output")
        lock = report.get("lock")
        lines.append(
            "hf_adapter_continuation_executor_process "
            f"attempt={attempt.get('attempt_id')} "
            f"status={attempt.get('status')} "
            f"runner={attempt.get('runner_kind')} "
            f"host={attempt.get('hostname')} "
            f"pid={attempt.get('pid')} "
            f"pid_alive={report.get('pid_alive_observed')} "
            f"output_exists={output.get('exists') if isinstance(output, Mapping) else None} "
            f"log_exists={log.get('exists') if isinstance(log, Mapping) else None} "
            f"log_bytes={log.get('size_bytes') if isinstance(log, Mapping) else None} "
            f"log_age_seconds={log.get('age_seconds') if isinstance(log, Mapping) else None} "
            f"lock_exists={lock.get('exists') if isinstance(lock, Mapping) else None} "
            f"log={attempt.get('log_path')}"
        )
    return lines
