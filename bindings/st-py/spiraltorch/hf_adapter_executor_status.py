"""Read-only live status for Hugging Face adapter continuation executors."""

from __future__ import annotations

import json
import socket
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from ._process import local_pid_alive
from .hf_adapter_executor import (
    HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME,
    _interruption_claim_report,
    _pending_output_resolution_gate,
    _unresolved_failed_attempt_output,
    load_hf_adapter_continuation_executor,
    load_hf_adapter_continuation_executor_stop_request,
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


def _lock_owner_report(value: object) -> dict[str, object]:
    report = {
        "valid": False,
        "error": None,
        "lock_id": None,
        "pid": None,
        "hostname": None,
        "same_host": None,
        "pid_alive_observed": None,
    }
    if value is None:
        return report
    path = Path(str(value)).expanduser()
    if not path.is_file():
        return report
    if path.is_symlink():
        report["error"] = "executor lock cannot be a symbolic link"
        return report
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        report["error"] = f"{exc.__class__.__name__}: {exc}"
        return report
    if not isinstance(payload, Mapping):
        report["error"] = "executor lock must contain a JSON object"
        return report
    pid = payload.get("pid")
    hostname = payload.get("hostname")
    valid = bool(
        payload.get("row_type") == "hf_adapter_continuation_executor_lock"
        and isinstance(payload.get("lock_id"), str)
        and bool(payload.get("lock_id"))
        and not isinstance(pid, bool)
        and isinstance(pid, int)
        and pid > 0
        and isinstance(hostname, str)
        and hostname
    )
    same_host = hostname == socket.gethostname() if isinstance(hostname, str) else None
    report.update(
        {
            "valid": valid,
            "error": None if valid else "executor lock owner fields are invalid",
            "lock_id": payload.get("lock_id"),
            "pid": pid,
            "hostname": hostname,
            "same_host": same_host,
            "pid_alive_observed": (
                local_pid_alive(pid) if valid and same_host is True else None
            ),
        }
    )
    return report


def _attempt_summary(attempt: Mapping[str, object] | None) -> dict[str, object] | None:
    if attempt is None:
        return None
    postflight = attempt.get("postflight")
    return {
        "attempt_id": attempt.get("attempt_id"),
        "status": attempt.get("status"),
        "lineage_depth": attempt.get("lineage_depth"),
        "runner_kind": attempt.get("runner_kind"),
        "process_group_isolated": attempt.get("process_group_isolated"),
        "stop_scope": attempt.get("stop_scope"),
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
        "adapter_id": attempt.get("adapter_id"),
        "parent_adapter_id": attempt.get("parent_adapter_id"),
        "command_runtime": attempt.get("command_runtime"),
        "parent_identity_contract": attempt.get("parent_identity_contract"),
        "adapter_input_identity": attempt.get("adapter_input_identity"),
        "training_input_identity_contract": attempt.get(
            "training_input_identity_contract"
        ),
        "training_input_identity": attempt.get("training_input_identity"),
        "dataset_input_identity_contract": attempt.get(
            "dataset_input_identity_contract"
        ),
        "dataset_input_expected_id": attempt.get("dataset_input_expected_id"),
        "dataset_input_effective_revision": attempt.get(
            "dataset_input_effective_revision"
        ),
        "dataset_input_effective_name": attempt.get("dataset_input_effective_name"),
        "dataset_materialization_identity_contract": attempt.get(
            "dataset_materialization_identity_contract"
        ),
        "dataset_materialization_expected_id": attempt.get(
            "dataset_materialization_expected_id"
        ),
        "tokenized_dataset_identity_contract": attempt.get(
            "tokenized_dataset_identity_contract"
        ),
        "tokenized_dataset_expected_id": attempt.get(
            "tokenized_dataset_expected_id"
        ),
        "runtime_input_identity_contract": attempt.get(
            "runtime_input_identity_contract"
        ),
        "runtime_input_expected_id": attempt.get("runtime_input_expected_id"),
        "execution_input_identity_contract": attempt.get(
            "execution_input_identity_contract"
        ),
        "execution_input_expected_id": attempt.get(
            "execution_input_expected_id"
        ),
        "source_transition": attempt.get("source_transition"),
        "postflight_transition": (
            postflight.get("transition") if isinstance(postflight, Mapping) else None
        ),
        "postflight_transition_ready": (
            postflight.get("transition_ready")
            if isinstance(postflight, Mapping)
            else None
        ),
        "stop_request": attempt.get("stop_request"),
        "status_before_interruption": attempt.get("status_before_interruption"),
        "interruption_claim": attempt.get("interruption_claim"),
        "output_resolution": attempt.get("output_resolution"),
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
    selected_transition = state.get("selected_transition")
    selected_depth = state.get("selected_lineage_depth")
    if selected_depth in (None, 0):
        transition_evidence_status = "not_applicable"
    elif (
        isinstance(selected_transition, Mapping)
        and selected_transition.get("transition_ready") is True
        and state.get("selected_path_transitions_ready") is True
    ):
        transition_evidence_status = "ready"
    elif (
        selected_transition is None
        and state.get("transition_count") is None
        and state.get("selected_path_transitions_ready") is None
    ):
        transition_evidence_status = "legacy_missing"
    else:
        transition_evidence_status = "not_ready"
    execution = state.get("execution")
    output_root = state.get("output_root")
    lock_path = execution.get("lock_path") if isinstance(execution, Mapping) else None
    lock_value = (
        lock_path
        if lock_path is not None
        else (
            None
            if output_root is None
            else Path(str(output_root)) / HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
        )
    )
    lock = _path_report(lock_value, expect_directory=False)
    lock_owner = _lock_owner_report(lock_value)
    lock_owner_may_be_active = bool(
        lock_owner.get("valid") is True
        and (
            lock_owner.get("same_host") is False
            or lock_owner.get("pid_alive_observed") is True
        )
    )
    stop_request_path = (
        execution.get("stop_request_path") if isinstance(execution, Mapping) else None
    )
    stop_request_artifact = _path_report(
        stop_request_path,
        expect_directory=False,
    )
    stop_request = None
    stop_request_error = None
    stop_request_valid = False
    if stop_request_artifact.get("exists") is True and stop_request_path is not None:
        try:
            stop_request = load_hf_adapter_continuation_executor_stop_request(
                str(stop_request_path)
            )
        except (OSError, ValueError) as exc:
            stop_request_error = f"{exc.__class__.__name__}: {exc}"
        else:
            stop_request_valid = bool(
                stop_request.get("run_id") == state.get("run_id")
                and stop_request.get("invocation_count")
                == state.get("invocation_count")
            )
            if not stop_request_valid:
                stop_request_error = "stop request targets a different invocation"
    if (
        stop_request_valid
        and lock_owner_may_be_active
        and executor_status in {"auditing", "running"}
    ):
        observed_status = "stopping"
    elif active is None and executor_status == "running":
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

    if observed_status == "stopping":
        recommended_action = "wait_for_graceful_stop"
    elif observed_status == "running":
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
    elif observed_status == "output_quarantined":
        recommended_action = "resume_executor"
    elif (
        observed_status in {"generation_limit_reached", "stopped"}
        and state.get("action") == "resume_executor"
    ):
        recommended_action = "resume_executor"
    else:
        recommended_action = "none"

    lifecycle_healthy = observed_status in {
        "auditing",
        "generation_limit_reached",
        "ready",
        "running",
        "output_quarantined",
        "stopping",
        "stopped",
    }
    updated_at = _timestamp(state.get("updated_at"))
    state_age_seconds = (
        None
        if updated_at is None
        else max(0.0, (datetime.now(timezone.utc) - updated_at).total_seconds())
    )
    observed_attempt = active or latest
    unresolved_generation = _unresolved_failed_attempt_output(state)
    pending_output_resolution = state.get("pending_output_resolution")
    pending_output_resolution_gate = _pending_output_resolution_gate(state)
    output = _path_report(
        None if observed_attempt is None else observed_attempt.get("output_dir"),
        expect_directory=True,
    )
    log = _path_report(
        None if observed_attempt is None else observed_attempt.get("log_path"),
        expect_directory=False,
    )
    interruption_claim = None
    if observed_status == "interrupted" and active is not None:
        interruption_claim = _interruption_claim_report(active)
    elif observed_attempt is not None and isinstance(
        observed_attempt.get("interruption_claim"), Mapping
    ):
        interruption_claim = dict(observed_attempt["interruption_claim"])
    interruption_lock_ready = bool(
        lock.get("exists") is False
        or (
            lock_owner.get("valid") is True
            and lock_owner.get("same_host") is True
            and lock_owner.get("pid_alive_observed") is False
        )
    )
    health_issues: list[str] = []
    if transition_evidence_status == "not_ready":
        health_issues.append("transition_evidence_not_ready")
    if pending_output_resolution_gate is not None:
        health_issues.append(str(pending_output_resolution_gate["issue"]))
    if stop_request_artifact.get("exists") is True and not stop_request_valid:
        health_issues.append("stop_request_invalid")
    if len(running) > 1:
        health_issues.append("multiple_running_attempts")
    if active is not None and lock.get("kind_ready") is not True:
        health_issues.append("single_writer_lock_missing")
    if (
        executor_status in {"auditing", "running"}
        and lock.get("kind_ready") is True
        and lock_owner.get("valid") is not True
    ):
        health_issues.append("single_writer_lock_invalid")
    if (
        executor_status in {"auditing", "running"}
        and lock_owner.get("same_host") is True
        and lock_owner.get("pid_alive_observed") is False
    ):
        health_issues.append("single_writer_lock_stale")
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
    if unresolved_generation is not None:
        unresolved_status = unresolved_generation.get("attempt_status")
        health_issues.append(
            "cancelled_output_present"
            if unresolved_status == "cancelled"
            else "interrupted_output_present"
            if unresolved_status == "interrupted"
            else "failed_generation_output_present"
        )
    if (
        observed_status == "interrupted"
        and active is not None
        and output.get("exists") is True
    ):
        health_issues.append(
            "interrupted_output_present"
            if isinstance(interruption_claim, Mapping)
            and interruption_claim.get("ready") is True
            and interruption_lock_ready
            else "interrupted_output_lock_unavailable"
            if isinstance(interruption_claim, Mapping)
            and interruption_claim.get("ready") is True
            else "interrupted_output_scope_unverified"
        )
    health_issues = list(dict.fromkeys(health_issues))
    healthy = lifecycle_healthy and not health_issues
    if "output_quarantine_intent_invalid" in health_issues:
        recommended_action = "inspect_output_quarantine_intent"
    elif "output_quarantine_incomplete" in health_issues:
        recommended_action = "complete_output_quarantine"
    elif "interrupted_output_present" in health_issues:
        recommended_action = "quarantine_interrupted_output"
    elif "interrupted_output_lock_unavailable" in health_issues:
        recommended_action = (
            "wait_for_executor_exit"
            if lock_owner.get("valid") is True
            and lock_owner.get("same_host") is True
            and lock_owner.get("pid_alive_observed") is True
            else "inspect_executor_lock"
        )
    elif "interrupted_output_scope_unverified" in health_issues:
        recommended_action = (
            str(interruption_claim.get("action"))
            if isinstance(interruption_claim, Mapping)
            else "inspect_interrupted_process_scope"
        )
    elif "cancelled_output_present" in health_issues:
        recommended_action = "resolve_cancelled_output"
    elif "failed_generation_output_present" in health_issues:
        recommended_action = "resolve_failed_generation_output"
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
        "selected_adapter_id": state.get("selected_adapter_id"),
        "selected_lineage_depth": selected_depth,
        "transition_count": state.get("transition_count"),
        "ready_transition_count": state.get("ready_transition_count"),
        "selected_path_transition_count": state.get(
            "selected_path_transition_count"
        ),
        "selected_path_transitions_ready": state.get(
            "selected_path_transitions_ready"
        ),
        "transition_evidence_status": transition_evidence_status,
        "selected_transition": selected_transition,
        "active_attempt_count": len(running),
        "active_attempt": _attempt_summary(active),
        "latest_attempt": _attempt_summary(latest),
        "last_output_resolution": state.get("last_output_resolution"),
        "pending_output_resolution": pending_output_resolution,
        "pending_output_resolution_attempt_ids": (
            []
            if pending_output_resolution_gate is None
            else pending_output_resolution_gate["attempt_ids"]
        ),
        "unresolved_generation": unresolved_generation,
        "interruption_claim": interruption_claim,
        "interruption_lock_ready": interruption_lock_ready,
        "current_hostname": current_hostname,
        "attempt_hostname": attempt_hostname,
        "same_host": same_host,
        "pid": pid,
        "pid_alive_observed": pid_alive,
        "process_identity_verified": False,
        "stop_requested": stop_request_valid,
        "stop_request": stop_request,
        "stop_request_error": stop_request_error,
        "stop_request_artifact": stop_request_artifact,
        "output": output,
        "log": log,
        "lock": lock,
        "lock_owner": lock_owner,
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
    unresolved = report.get("unresolved_generation")
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
            f"transitions={report.get('ready_transition_count')}/"
            f"{report.get('transition_count')} "
            f"transition_evidence={report.get('transition_evidence_status')} "
            f"state_age_seconds={report.get('state_age_seconds')} "
            f"stop_requested={report.get('stop_requested')} "
            "unresolved_attempt="
            f"{unresolved.get('attempt_id') if isinstance(unresolved, Mapping) else None} "
            f"state={report.get('state_path')}"
        )
    ]
    transition = report.get("selected_transition")
    if isinstance(transition, Mapping):
        lines.append(
            "hf_adapter_continuation_executor_status_transition "
            f"status={transition.get('status')} "
            f"depth={transition.get('parent_lineage_depth')}"
            f"->{transition.get('child_lineage_depth')} "
            f"parent={transition.get('parent_adapter_id')} "
            f"child={transition.get('child_adapter_id')} "
            f"eval_handoff_delta={transition.get('eval_handoff_delta')} "
            f"eval_improvement={transition.get('child_eval_improvement')} "
            f"input_identity={transition.get('input_identity_ready')} "
            "training_input_identity="
            f"{transition.get('training_input_identity_ready')} "
            "dataset_input_required="
            f"{transition.get('dataset_input_identity_required')} "
            "dataset_input_identity="
            f"{transition.get('dataset_input_identity_ready')} "
            "dataset_materialization_required="
            f"{transition.get('dataset_materialization_identity_required')} "
            "dataset_materialization_identity="
            f"{transition.get('dataset_materialization_identity_ready')} "
            "dataset_materialization_reissued="
            f"{transition.get('dataset_materialization_reissued')} "
            "tokenized_dataset_required="
            f"{transition.get('tokenized_dataset_identity_required')} "
            "tokenized_dataset_identity="
            f"{transition.get('tokenized_dataset_identity_ready')} "
            "tokenized_dataset_reissued="
            f"{transition.get('tokenized_dataset_reissued')} "
            "runtime_input_required="
            f"{transition.get('runtime_input_identity_required')} "
            "runtime_input_identity="
            f"{transition.get('runtime_input_identity_ready')} "
            "execution_input_required="
            f"{transition.get('execution_input_identity_required')} "
            "execution_input_identity="
            f"{transition.get('execution_input_identity_ready')} "
            f"probe_process={transition.get('artifact_probe_process_status')} "
            f"probe_pid={transition.get('artifact_probe_process_pid')}"
        )
    attempt = report.get("active_attempt") or report.get("latest_attempt")
    if isinstance(attempt, Mapping):
        command_runtime = attempt.get("command_runtime")
        adapter_input_identity = attempt.get("adapter_input_identity")
        training_input_identity = attempt.get("training_input_identity")
        dataset_input_contract = attempt.get("dataset_input_identity_contract")
        dataset_materialization_contract = attempt.get(
            "dataset_materialization_identity_contract"
        )
        tokenized_dataset_contract = attempt.get(
            "tokenized_dataset_identity_contract"
        )
        runtime_input_contract = attempt.get("runtime_input_identity_contract")
        execution_input_contract = attempt.get(
            "execution_input_identity_contract"
        )
        log = report.get("log")
        output = report.get("output")
        lock = report.get("lock")
        lock_owner = report.get("lock_owner")
        lines.append(
            "hf_adapter_continuation_executor_process "
            f"attempt={attempt.get('attempt_id')} "
            f"status={attempt.get('status')} "
            f"runner={attempt.get('runner_kind')} "
            "runtime="
            f"{command_runtime.get('status') if isinstance(command_runtime, Mapping) else None} "
            "input_identity="
            f"{adapter_input_identity.get('status') if isinstance(adapter_input_identity, Mapping) else None} "
            "training_input_identity="
            f"{training_input_identity.get('status') if isinstance(training_input_identity, Mapping) else None} "
            "dataset_input_contract="
            f"{dataset_input_contract.get('status') if isinstance(dataset_input_contract, Mapping) else None} "
            f"dataset_input_expected={attempt.get('dataset_input_expected_id')} "
            "dataset_input_revision="
            f"{attempt.get('dataset_input_effective_revision')} "
            f"dataset_input_name={attempt.get('dataset_input_effective_name')} "
            "dataset_materialization_contract="
            f"{dataset_materialization_contract.get('status') if isinstance(dataset_materialization_contract, Mapping) else None} "
            "dataset_materialization_expected="
            f"{attempt.get('dataset_materialization_expected_id')} "
            "tokenized_dataset_contract="
            f"{tokenized_dataset_contract.get('status') if isinstance(tokenized_dataset_contract, Mapping) else None} "
            "tokenized_dataset_expected="
            f"{attempt.get('tokenized_dataset_expected_id')} "
            "runtime_input_contract="
            f"{runtime_input_contract.get('status') if isinstance(runtime_input_contract, Mapping) else None} "
            f"runtime_input_expected={attempt.get('runtime_input_expected_id')} "
            "execution_input_contract="
            f"{execution_input_contract.get('status') if isinstance(execution_input_contract, Mapping) else None} "
            "execution_input_expected="
            f"{attempt.get('execution_input_expected_id')} "
            f"stop_scope={attempt.get('stop_scope')} "
            f"host={attempt.get('hostname')} "
            f"pid={attempt.get('pid')} "
            f"pid_alive={report.get('pid_alive_observed')} "
            f"output_exists={output.get('exists') if isinstance(output, Mapping) else None} "
            f"log_exists={log.get('exists') if isinstance(log, Mapping) else None} "
            f"log_bytes={log.get('size_bytes') if isinstance(log, Mapping) else None} "
            f"log_age_seconds={log.get('age_seconds') if isinstance(log, Mapping) else None} "
            f"lock_exists={lock.get('exists') if isinstance(lock, Mapping) else None} "
            "lock_owner_alive="
            f"{lock_owner.get('pid_alive_observed') if isinstance(lock_owner, Mapping) else None} "
            "output_resolution="
            f"{attempt.get('output_resolution', {}).get('resolution_id') if isinstance(attempt.get('output_resolution'), Mapping) else None} "
            "interruption_ready="
            f"{report.get('interruption_claim', {}).get('ready') if isinstance(report.get('interruption_claim'), Mapping) else None} "
            "process_group_alive="
            f"{report.get('interruption_claim', {}).get('process_group_alive_observed') if isinstance(report.get('interruption_claim'), Mapping) else None} "
            f"log={attempt.get('log_path')}"
        )
    return lines
