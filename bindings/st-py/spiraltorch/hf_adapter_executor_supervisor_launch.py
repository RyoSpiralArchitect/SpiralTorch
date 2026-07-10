"""Detached launch control for the HF adapter executor supervisor."""

from __future__ import annotations

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
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ._process import local_pid_alive
from .hf_adapter_executor_launch import (
    load_hf_adapter_continuation_executor_launch,
)
from .hf_adapter_executor_supervisor import (
    HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_FILENAME,
    HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LOCK_FILENAME,
    hf_adapter_continuation_executor_supervision_report,
    hf_adapter_continuation_executor_supervisor_status_report,
)

__all__ = [
    "HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_FILENAME",
    "HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_LOCK_FILENAME",
    "HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_SCHEMA",
    "HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_STATUS_SCHEMA",
    "hf_adapter_continuation_executor_supervisor_launch_lines",
    "hf_adapter_continuation_executor_supervisor_launch_status_lines",
    "hf_adapter_continuation_executor_supervisor_launch_status_report",
    "launch_hf_adapter_continuation_executor_supervisor",
    "load_hf_adapter_continuation_executor_supervisor_launch",
]


HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_SCHEMA = (
    "spiraltorch.hf_adapter_continuation_executor_supervisor_launch.v1"
)
HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_STATUS_SCHEMA = (
    "spiraltorch.hf_adapter_continuation_executor_supervisor_launch_status.v1"
)
HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_FILENAME = (
    "spiraltorch-hf-adapter-continuation-executor-supervisor-launch.json"
)
HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_LOCK_FILENAME = (
    ".spiraltorch-hf-adapter-continuation-executor-supervisor-launch.lock"
)

_LAUNCH_LOCK_RETRIES = 8
_LAUNCH_LOCK_RETRY_SECONDS = 0.05
_HANDOFF_POLL_SECONDS = 0.05


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_symlink():
        raise RuntimeError(f"supervisor launch state cannot be a symlink: {path}")
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


def _process_observation(record: Mapping[str, object]) -> str:
    hostname = record.get("hostname")
    if not isinstance(hostname, str) or not hostname:
        return "unverified"
    if hostname != socket.gethostname():
        return "remote_unverified"
    alive = local_pid_alive(record.get("pid"))
    if alive is True:
        return "alive"
    if alive is False:
        return "exited"
    return "unverified"


def _load_launch_lock(path: Path) -> dict[str, object] | None:
    if path.is_symlink():
        raise RuntimeError(f"supervisor launch lock cannot be a symlink: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"supervisor launch lock is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"supervisor launch lock must contain an object: {path}")
    return dict(payload)


def _launch_lock_owner_is_stale(owner: Mapping[str, object]) -> bool:
    return bool(
        owner.get("row_type")
        == "hf_adapter_continuation_executor_supervisor_launch_lock"
        and isinstance(owner.get("lock_id"), str)
        and owner.get("hostname") == socket.gethostname()
        and local_pid_alive(owner.get("pid")) is False
    )


def _reap_stale_launch_lock(path: Path) -> bool:
    reaper = path.with_name(f"{path.name}.reap")
    try:
        reaper.mkdir(mode=0o700)
    except FileExistsError:
        return False
    try:
        current = _load_launch_lock(path)
        if current is None or not _launch_lock_owner_is_stale(current):
            return False
        path.unlink()
        return True
    finally:
        reaper.rmdir()


@contextmanager
def _launch_lock(output_root: Path, launch_state_path: Path) -> Iterator[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_LOCK_FILENAME
    owner = {
        "row_type": "hf_adapter_continuation_executor_supervisor_launch_lock",
        "lock_id": f"executor-supervisor-launch-lock-{uuid.uuid4().hex}",
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "created_at": _now(),
        "supervisor_launch_state_path": str(launch_state_path),
    }
    encoded = (json.dumps(owner, ensure_ascii=True, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    acquired = False
    for _ in range(_LAUNCH_LOCK_RETRIES):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            existing = _load_launch_lock(path)
            if existing is None:
                continue
            if _launch_lock_owner_is_stale(existing):
                if not _reap_stale_launch_lock(path):
                    time.sleep(_LAUNCH_LOCK_RETRY_SECONDS)
                continue
            raise RuntimeError(
                "executor supervisor detached launch is already owned: "
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
        raise RuntimeError(f"could not acquire supervisor launch lock: {path}")
    try:
        yield path
    finally:
        try:
            current = _load_launch_lock(path)
        except RuntimeError:
            current = None
        if current is not None and current.get("lock_id") == owner["lock_id"]:
            path.unlink(missing_ok=True)


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


def _validate_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


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


def _resolved_state_path(value: str | Path | None, default: Path, *, name: str) -> Path:
    raw = Path(value).expanduser() if value is not None else default
    if raw.is_symlink():
        raise ValueError(f"{name} cannot be a symlink: {raw}")
    return raw.resolve()


def _new_launch_state(
    *,
    executor_launch_state_path: Path,
    supervisor_state_path: Path,
    supervisor_launch_state_path: Path,
    output_root: Path,
) -> dict[str, object]:
    created_at = _now()
    return {
        "row_type": "hf_adapter_continuation_executor_supervisor_launches",
        "schema": HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_SCHEMA,
        "created_at": created_at,
        "updated_at": created_at,
        "status": "initializing",
        "executor_launch_state_path": str(executor_launch_state_path),
        "supervisor_state_path": str(supervisor_state_path),
        "supervisor_launch_state_path": str(supervisor_launch_state_path),
        "output_root": str(output_root),
        "launch_count": 0,
        "launches": [],
    }


def load_hf_adapter_continuation_executor_supervisor_launch(
    value: str | Path,
) -> dict[str, object]:
    """Load and validate detached supervisor launch history."""

    path = Path(value).expanduser()
    if path.is_symlink():
        raise ValueError(f"supervisor launch state cannot be a symlink: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"supervisor launch state must contain an object: {path}")
    if payload.get("schema") != HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_SCHEMA:
        raise ValueError(
            "unsupported HF adapter executor supervisor launch schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("row_type") != "hf_adapter_continuation_executor_supervisor_launches":
        raise ValueError(
            "unsupported HF adapter executor supervisor launch row type: "
            f"{payload.get('row_type')}"
        )
    launches = payload.get("launches")
    if not isinstance(launches, list) or any(
        not isinstance(row, Mapping) for row in launches
    ):
        raise ValueError("supervisor launch history contains an invalid launch row")
    if payload.get("launch_count") != len(launches):
        raise ValueError("supervisor launch_count is inconsistent")
    launch_ids = [row.get("launch_id") for row in launches]
    if any(not isinstance(value, str) or not value for value in launch_ids):
        raise ValueError("supervisor launch history contains an invalid launch ID")
    if len(set(launch_ids)) != len(launch_ids):
        raise ValueError("supervisor launch history contains duplicate launch IDs")
    if launches and payload.get("latest_launch_id") != launch_ids[-1]:
        raise ValueError("supervisor latest launch ID is inconsistent")
    report = dict(payload)
    report["supervisor_launch_state_path"] = str(path.resolve())
    return report


def _launch_state_for_request(
    *,
    executor_launch_state_path: Path,
    supervisor_state_path: Path,
    supervisor_launch_state_path: Path,
    output_root: Path,
) -> dict[str, object]:
    state = (
        load_hf_adapter_continuation_executor_supervisor_launch(
            supervisor_launch_state_path
        )
        if supervisor_launch_state_path.is_file()
        else _new_launch_state(
            executor_launch_state_path=executor_launch_state_path,
            supervisor_state_path=supervisor_state_path,
            supervisor_launch_state_path=supervisor_launch_state_path,
            output_root=output_root,
        )
    )
    expected = {
        "executor_launch_state_path": str(executor_launch_state_path),
        "supervisor_state_path": str(supervisor_state_path),
        "output_root": str(output_root),
    }
    for field, value in expected.items():
        if state.get(field) != value:
            raise ValueError(f"supervisor launch {field} differs; use a new state")
    return state


def _write_launch_state(path: Path, state: dict[str, object]) -> None:
    launches = state.get("launches") or []
    state["launch_count"] = len(launches)
    state["updated_at"] = _now()
    _atomic_write_json(path, state)


def _process_group_options() -> dict[str, object]:
    if os.name == "posix":
        return {"start_new_session": True}
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {}


def _supervisor_child_command(
    executor_launch_state_path: Path,
    *,
    max_resumes: int,
    poll_interval_seconds: float,
    timeout_seconds: float,
    handoff_timeout_seconds: float,
    supervisor_state_path: Path,
) -> list[str]:
    script = (
        "import sys; "
        "from spiraltorch.hf_cli import "
        "adapter_continuation_executor_supervise_main; "
        "raise SystemExit(adapter_continuation_executor_supervise_main(sys.argv[1:]))"
    )
    return [
        sys.executable,
        "-c",
        script,
        str(executor_launch_state_path),
        "--max-resumes",
        str(max_resumes),
        "--poll-interval-seconds",
        str(poll_interval_seconds),
        "--timeout-seconds",
        str(timeout_seconds),
        "--handoff-timeout-seconds",
        str(handoff_timeout_seconds),
        "--state",
        str(supervisor_state_path),
    ]


def _reap_in_background(process: subprocess.Popen[bytes]) -> None:
    threading.Thread(
        target=process.wait,
        name="spiraltorch-adapter-supervisor-launch-reaper",
        daemon=True,
    ).start()


def _supervisor_baseline(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    status = hf_adapter_continuation_executor_supervisor_status_report(path)
    latest = status.get("latest_run")
    return {
        "invocation_count": status.get("invocation_count"),
        "supervisor_run_id": (
            latest.get("supervisor_run_id")
            if isinstance(latest, Mapping)
            else None
        ),
    }


def _is_new_supervisor_run(
    status: Mapping[str, object],
    baseline: Mapping[str, object] | None,
    *,
    pid: int,
) -> bool:
    latest = status.get("latest_run")
    if not isinstance(latest, Mapping):
        return False
    invocation = latest.get("invocation_count")
    baseline_invocation = (
        0 if baseline is None else baseline.get("invocation_count")
    )
    if (
        isinstance(baseline_invocation, bool)
        or not isinstance(baseline_invocation, int)
        or baseline_invocation < 0
    ):
        return False
    return bool(
        latest.get("pid") == pid
        and latest.get("hostname") == socket.gethostname()
        and isinstance(invocation, int)
        and not isinstance(invocation, bool)
        and invocation > baseline_invocation
    )


def _request_report(
    state: Mapping[str, object],
    *,
    request_status: str,
    created: bool,
) -> dict[str, object]:
    report = dict(state)
    launches = report.get("launches") or []
    report["latest_launch"] = launches[-1] if launches else None
    report["request_status"] = request_status
    report["created"] = created
    return report


def _terminate_owned_process(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float = 2.0,
) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _existing_supervisor_gate(
    supervisor_state_path: Path,
) -> tuple[str | None, dict[str, object] | None]:
    if not supervisor_state_path.is_file():
        return None, None
    status = hf_adapter_continuation_executor_supervisor_status_report(
        supervisor_state_path
    )
    lifecycle = status.get("status")
    if lifecycle in {"running", "starting", "stopping"}:
        return "already_running", status
    if lifecycle in {
        "remote_running",
        "running_unverified",
        "ownership_conflict",
    }:
        return "supervisor_owner_unverified", status
    return None, status


def _latest_launch_gate(
    latest: dict[str, object] | None,
    supervisor_status: Mapping[str, object] | None,
) -> str | None:
    if latest is None:
        return None
    observation = _process_observation(latest)
    latest["process_liveness_observed_at"] = _now()
    latest["process_liveness_observation"] = observation
    if observation in {"remote_unverified", "unverified"}:
        return "supervisor_launch_owner_unverified"
    if observation == "alive":
        nested_run = (
            supervisor_status.get("latest_run")
            if isinstance(supervisor_status, Mapping)
            else None
        )
        verified_owner = bool(
            isinstance(nested_run, Mapping)
            and nested_run.get("pid") == latest.get("pid")
            and supervisor_status.get("supervisor_lock_owner_verified") is True
            and supervisor_status.get("status") in {"running", "starting", "stopping"}
        )
        if verified_owner:
            return "already_running"
        launch_finished = latest.get("status") in {
            "completed",
            "exited_observed",
            "handed_off",
            "launch_failed",
        }
        nested_terminal = bool(
            isinstance(supervisor_status, Mapping)
            and supervisor_status.get("status")
            not in {
                "running",
                "starting",
                "stopping",
                "remote_running",
                "running_unverified",
            }
        )
        if launch_finished and nested_terminal:
            latest["process_liveness_observation"] = (
                "pid_alive_without_supervisor_ownership"
            )
            latest["status"] = "exited_observed"
            latest["process_exited_observed_at"] = _now()
            return None
        return "supervisor_launch_owner_unverified"
    if latest.get("status") in {"handed_off", "handoff_timeout", "launched"}:
        latest["status"] = "exited_observed"
        latest["process_exited_observed_at"] = _now()
    return None


def _record_handoff(
    attempt: dict[str, object],
    status: Mapping[str, object],
) -> None:
    latest_run = status.get("latest_run")
    if not isinstance(latest_run, Mapping):
        return
    attempt["supervisor_run_id"] = latest_run.get("supervisor_run_id")
    attempt["supervisor_invocation_count"] = latest_run.get("invocation_count")
    attempt["supervisor_status_at_handoff"] = status.get("status")


def _validated_launch_parameters(
    *,
    max_resumes: int,
    poll_interval_seconds: float,
    timeout_seconds: float,
    handoff_timeout_seconds: float,
    launch_handoff_timeout_seconds: float,
) -> dict[str, object]:
    return {
        "max_resumes": _validate_positive_int("max_resumes", max_resumes),
        "poll_interval_seconds": _validate_seconds(
            "poll_interval_seconds", poll_interval_seconds, allow_zero=False
        ),
        "timeout_seconds": _validate_seconds(
            "timeout_seconds", timeout_seconds, allow_zero=True
        ),
        "handoff_timeout_seconds": _validate_seconds(
            "handoff_timeout_seconds", handoff_timeout_seconds, allow_zero=True
        ),
        "launch_handoff_timeout_seconds": _validate_seconds(
            "launch_handoff_timeout_seconds",
            launch_handoff_timeout_seconds,
            allow_zero=True,
        ),
    }


def _supervisor_launch_paths(
    report_or_path: Mapping[str, object] | str | Path,
    *,
    supervisor_state_path: str | Path | None,
    supervisor_launch_state_path: str | Path | None,
    command_cwd: str | Path | None,
) -> dict[str, Path]:
    executor_launch_state_path = _input_launch_state_path(report_or_path)
    executor_launch = load_hf_adapter_continuation_executor_launch(
        executor_launch_state_path
    )
    output_root_value = executor_launch.get("output_root")
    if not isinstance(output_root_value, str) or not output_root_value:
        raise ValueError("executor launch output_root is missing")
    output_root = Path(output_root_value).expanduser().resolve()
    state_path = _resolved_state_path(
        supervisor_state_path,
        output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_FILENAME,
        name="supervisor_state_path",
    )
    launch_state_path = _resolved_state_path(
        supervisor_launch_state_path,
        output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_FILENAME,
        name="supervisor_launch_state_path",
    )
    cwd = (
        Path(command_cwd).expanduser().resolve()
        if command_cwd is not None
        else Path.cwd().resolve()
    )
    if not cwd.is_dir():
        raise ValueError(f"supervisor launch command_cwd does not exist: {cwd}")
    reserved = {
        executor_launch_state_path,
        state_path,
        output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LOCK_FILENAME,
        output_root
        / HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_LOCK_FILENAME,
    }
    executor_state_value = executor_launch.get("executor_state_path")
    if executor_state_value is not None:
        reserved.add(Path(str(executor_state_value)).expanduser().resolve())
    if launch_state_path in reserved:
        raise ValueError("supervisor_launch_state_path would overwrite runtime state")
    return {
        "executor_launch_state_path": executor_launch_state_path,
        "output_root": output_root,
        "supervisor_state_path": state_path,
        "supervisor_launch_state_path": launch_state_path,
        "command_cwd": cwd,
    }


def _launchable_supervisor_state(
    *,
    executor_launch_state_path: Path,
    supervisor_state_path: Path,
    supervisor_launch_state_path: Path,
    output_root: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object] | None]:
    state = _launch_state_for_request(
        executor_launch_state_path=executor_launch_state_path,
        supervisor_state_path=supervisor_state_path,
        supervisor_launch_state_path=supervisor_launch_state_path,
        output_root=output_root,
    )
    launches = state.get("launches")
    if not isinstance(launches, list):
        raise ValueError("supervisor launch history launches must be a list")
    existing_gate, supervisor_status = _existing_supervisor_gate(
        supervisor_state_path
    )
    latest = launches[-1] if launches and isinstance(launches[-1], dict) else None
    gate = existing_gate or _latest_launch_gate(latest, supervisor_status)
    if gate is not None:
        state["supervisor_status"] = supervisor_status
        _write_launch_state(supervisor_launch_state_path, state)
        return state, {}, _request_report(
            state,
            request_status=gate,
            created=False,
        )
    supervision = hf_adapter_continuation_executor_supervision_report(
        executor_launch_state_path
    )
    if supervision.get("status") not in {"waiting", "resume_ready"}:
        raise RuntimeError(
            "executor supervision is not launchable: "
            f"{supervision.get('status')}: {supervision.get('issue')}"
        )
    return state, supervision, None


def _new_supervisor_launch_attempt(
    state: dict[str, object],
    supervision: Mapping[str, object],
    parameters: Mapping[str, object],
    paths: Mapping[str, Path],
) -> tuple[dict[str, object], list[str], Path, dict[str, object] | None]:
    supervisor_state_path = paths["supervisor_state_path"]
    output_root = paths["output_root"]
    baseline = _supervisor_baseline(supervisor_state_path)
    launch_id = f"executor-supervisor-launch-{uuid.uuid4().hex}"
    log_dir = output_root / "supervisor-logs"
    if log_dir.is_symlink():
        raise RuntimeError(f"supervisor log directory cannot be a symlink: {log_dir}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"supervisor-{launch_id[-16:]}.log"
    command = _supervisor_child_command(
        paths["executor_launch_state_path"],
        max_resumes=int(parameters["max_resumes"]),
        poll_interval_seconds=float(parameters["poll_interval_seconds"]),
        timeout_seconds=float(parameters["timeout_seconds"]),
        handoff_timeout_seconds=float(parameters["handoff_timeout_seconds"]),
        supervisor_state_path=supervisor_state_path,
    )
    attempt = {
        "launch_id": launch_id,
        "status": "launching",
        "started_at": _now(),
        "hostname": socket.gethostname(),
        "pid": None,
        "command_cwd": str(paths["command_cwd"]),
        "command": command,
        "command_display": shlex.join(command),
        "executor_launch_state_path": str(paths["executor_launch_state_path"]),
        "supervisor_state_path": str(supervisor_state_path),
        "supervisor_baseline": baseline,
        **dict(parameters),
        "source_supervision_status": supervision.get("status"),
        "source_executor_launch_id": supervision.get("source_launch_id"),
        "source_executor_invocation_count": supervision.get(
            "executor_invocation_count"
        ),
        "log_path": str(log_path),
        "log_stream": "combined_stdout_stderr",
        "process_group_isolated": os.name in {"posix", "nt"},
    }
    launches = state["launches"]
    if not isinstance(launches, list):
        raise RuntimeError("supervisor launch history became invalid")
    launches.append(attempt)
    state["status"] = "launching"
    state["latest_launch_id"] = launch_id
    return attempt, command, log_path, baseline


def _spawn_supervisor_process(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
) -> subprocess.Popen[bytes]:
    descriptor = os.open(log_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb", buffering=0) as log_handle:
        log_handle.write(
            (
                f"[spiraltorch-supervisor-launch] started_at={_now()} "
                f"command={shlex.join(command)}\n"
            ).encode("utf-8")
        )
        return subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            **_process_group_options(),
        )


def _wait_for_supervisor_handoff(
    process: subprocess.Popen[bytes],
    attempt: dict[str, object],
    state: dict[str, object],
    *,
    supervisor_state_path: Path,
    baseline: Mapping[str, object] | None,
    timeout_seconds: float,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    while True:
        nested_status = None
        if supervisor_state_path.is_file():
            try:
                nested_status = (
                    hf_adapter_continuation_executor_supervisor_status_report(
                        supervisor_state_path
                    )
                )
            except (OSError, ValueError):
                nested_status = None
        new_run = bool(
            nested_status is not None
            and _is_new_supervisor_run(nested_status, baseline, pid=process.pid)
        )
        if new_run and nested_status is not None:
            _record_handoff(attempt, nested_status)
            if (
                process.poll() is None
                and nested_status.get("status") == "running"
                and nested_status.get("supervisor_lock_owner_verified") is True
            ):
                attempt["handoff_at"] = _now()
                attempt["status"] = "handed_off"
                state["status"] = "handed_off"
                return "handed_off"
        returncode = process.poll()
        if returncode is not None:
            attempt["returncode_observed"] = int(returncode)
            attempt["completed_at"] = _now()
            healthy_terminal = bool(
                new_run
                and nested_status is not None
                and nested_status.get("healthy") is True
                and nested_status.get("status")
                not in {"running", "starting", "stopping"}
            )
            status = "completed" if returncode == 0 and healthy_terminal else "launch_failed"
            attempt["status"] = status
            state["status"] = status
            return status
        if time.monotonic() >= deadline:
            attempt["status"] = "handoff_timeout"
            attempt["handoff_timeout_at"] = _now()
            state["status"] = "handoff_timeout"
            return "handoff_timeout"
        time.sleep(_HANDOFF_POLL_SECONDS)


def launch_hf_adapter_continuation_executor_supervisor(
    report_or_path: Mapping[str, object] | str | Path,
    *,
    max_resumes: int = 1,
    poll_interval_seconds: float = 5.0,
    timeout_seconds: float = 0.0,
    handoff_timeout_seconds: float = 5.0,
    launch_handoff_timeout_seconds: float = 5.0,
    supervisor_state_path: str | Path | None = None,
    supervisor_launch_state_path: str | Path | None = None,
    command_cwd: str | Path | None = None,
) -> dict[str, object]:
    """Launch a bounded supervisor and persist verified handoff evidence."""

    parameters = _validated_launch_parameters(
        max_resumes=max_resumes,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        handoff_timeout_seconds=handoff_timeout_seconds,
        launch_handoff_timeout_seconds=launch_handoff_timeout_seconds,
    )
    paths = _supervisor_launch_paths(
        report_or_path,
        supervisor_state_path=supervisor_state_path,
        supervisor_launch_state_path=supervisor_launch_state_path,
        command_cwd=command_cwd,
    )
    launch_state_path = paths["supervisor_launch_state_path"]
    with _launch_lock(paths["output_root"], launch_state_path):
        state, supervision, early_report = _launchable_supervisor_state(
            executor_launch_state_path=paths["executor_launch_state_path"],
            supervisor_state_path=paths["supervisor_state_path"],
            supervisor_launch_state_path=launch_state_path,
            output_root=paths["output_root"],
        )
        if early_report is not None:
            return early_report
        attempt, command, log_path, baseline = _new_supervisor_launch_attempt(
            state,
            supervision,
            parameters,
            paths,
        )
        _write_launch_state(launch_state_path, state)
        try:
            process = _spawn_supervisor_process(
                command,
                cwd=paths["command_cwd"],
                log_path=log_path,
            )
        except BaseException as exc:
            attempt["status"] = "launch_failed"
            attempt["failure"] = f"{exc.__class__.__name__}: {exc}"
            attempt["completed_at"] = _now()
            state["status"] = "launch_failed"
            _write_launch_state(launch_state_path, state)
            raise
        attempt["pid"] = int(process.pid)
        attempt["process_started_at"] = _now()
        attempt["status"] = "launched"
        state["status"] = "launched"
        try:
            _write_launch_state(launch_state_path, state)
        except BaseException:
            _terminate_owned_process(process)
            process.wait()
            raise
        try:
            request_status = _wait_for_supervisor_handoff(
                process,
                attempt,
                state,
                supervisor_state_path=paths["supervisor_state_path"],
                baseline=baseline,
                timeout_seconds=float(parameters["launch_handoff_timeout_seconds"]),
            )
            _write_launch_state(launch_state_path, state)
        finally:
            if process.poll() is None:
                _reap_in_background(process)
        return _request_report(state, request_status=request_status, created=True)


def _launch_lock_observation(output_root: Path) -> dict[str, object]:
    path = output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_LOCK_FILENAME
    try:
        owner = _load_launch_lock(path)
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
        != "hf_adapter_continuation_executor_supervisor_launch_lock"
        or not isinstance(owner.get("lock_id"), str)
        or not owner.get("lock_id")
    ):
        return {
            "path": str(path),
            "status": "invalid",
            "owner": owner,
            "error": "supervisor launch lock owner identity is invalid",
        }
    return {
        "path": str(path),
        "status": _process_observation(owner),
        "owner": owner,
        "error": None,
    }


def _launch_handoff_observation(
    latest: Mapping[str, object],
    supervisor_status: Mapping[str, object] | None,
) -> str:
    if latest.get("status") == "launch_failed":
        return "failed"
    if supervisor_status is None:
        return "supervisor_state_missing"
    current = supervisor_status.get("latest_run")
    if not isinstance(current, Mapping):
        return "supervisor_run_missing"
    recorded_run_id = latest.get("supervisor_run_id")
    recorded_invocation = latest.get("supervisor_invocation_count")
    if recorded_run_id is not None or recorded_invocation is not None:
        return (
            "recorded_match"
            if recorded_run_id == current.get("supervisor_run_id")
            and recorded_invocation == current.get("invocation_count")
            else "recorded_mismatch"
        )
    baseline = latest.get("supervisor_baseline")
    if baseline is not None and not isinstance(baseline, Mapping):
        return "baseline_invalid"
    baseline_invocation = (
        0 if baseline is None else int(baseline.get("invocation_count") or 0)
    )
    current_invocation = current.get("invocation_count")
    if (
        isinstance(current_invocation, int)
        and not isinstance(current_invocation, bool)
        and current_invocation > baseline_invocation
        and current.get("pid") == latest.get("pid")
        and current.get("hostname") == latest.get("hostname")
    ):
        return "baseline_advanced"
    return "baseline_not_advanced"


def _launch_state_and_path(
    report_or_path: Mapping[str, object] | str | Path,
) -> tuple[dict[str, object], Path]:
    state = (
        dict(report_or_path)
        if isinstance(report_or_path, Mapping)
        and report_or_path.get("row_type")
        == "hf_adapter_continuation_executor_supervisor_launches"
        else load_hf_adapter_continuation_executor_supervisor_launch(report_or_path)
    )
    value = state.get("supervisor_launch_state_path")
    if value is None:
        raise ValueError("supervisor launch state path is missing")
    path = Path(str(value)).expanduser()
    if path.is_symlink():
        raise ValueError(f"supervisor launch state cannot be a symlink: {path}")
    return state, path.resolve()


def _nested_supervisor_status(
    state_path_value: object,
) -> tuple[dict[str, object] | None, str | None]:
    if state_path_value is None:
        return None, "supervisor state path is missing"
    path = Path(str(state_path_value)).expanduser()
    if not path.is_file():
        return None, None
    try:
        return hf_adapter_continuation_executor_supervisor_status_report(path), None
    except (OSError, ValueError) as exc:
        return None, f"{exc.__class__.__name__}: {exc}"


def _observed_launch_status(
    latest: Mapping[str, object] | None,
    *,
    process_observation: str,
    handoff_observation: str,
    supervisor_status: Mapping[str, object] | None,
    launch_lock_status: object,
) -> str:
    if latest is None:
        return "empty"
    if latest.get("status") == "launch_failed":
        return "launch_failed"
    nested_status = (
        None if supervisor_status is None else supervisor_status.get("status")
    )
    handoff_matches = handoff_observation in {"baseline_advanced", "recorded_match"}
    if handoff_matches and nested_status is not None:
        return str(nested_status)
    if (
        handoff_observation == "recorded_mismatch"
        and supervisor_status is not None
        and supervisor_status.get("supervisor_lock_owner_verified") is True
        and nested_status in {"running", "starting", "stopping"}
    ):
        return "superseded"
    if process_observation == "remote_unverified":
        return "remote_running"
    if (
        process_observation == "alive"
        and latest.get("status") in {"launching", "launched"}
        and launch_lock_status == "alive"
    ):
        return "starting"
    if process_observation in {"alive", "unverified"}:
        return "running_unverified"
    return "handoff_unverified"


def _launch_recommended_action(
    status: str,
    supervisor_status: Mapping[str, object] | None,
) -> str:
    actions = {
        "completed": "inspect_supervisor_result",
        "empty": "launch_supervisor",
        "handoff_unverified": "inspect_supervisor_handoff",
        "interrupted": "restart_supervisor",
        "launch_failed": "inspect_supervisor_log",
        "ownership_conflict": "inspect_supervisor_ownership",
        "paused": "inspect_manual_boundary",
        "remote_running": "inspect_remote_supervisor",
        "resume_budget_reached": "launch_supervisor_again",
        "running": "monitor_supervisor",
        "running_unverified": "inspect_supervisor_ownership",
        "starting": "wait_for_supervisor_handoff",
        "stopped": "restart_supervisor_if_needed",
        "stopping": "wait_for_supervisor_exit",
        "superseded": "inspect_current_supervisor",
        "timed_out": "inspect_supervisor_status",
    }
    if status in actions:
        return actions[status]
    if supervisor_status is not None:
        nested_action = supervisor_status.get("recommended_action")
        if isinstance(nested_action, str) and nested_action:
            return nested_action
    return "inspect_supervisor_launch"


def hf_adapter_continuation_executor_supervisor_launch_status_report(
    report_or_path: Mapping[str, object] | str | Path,
) -> dict[str, object]:
    """Observe one detached supervisor launch and its current durable run."""

    state, launch_state_path = _launch_state_and_path(report_or_path)
    launches = state.get("launches") or []
    latest = dict(launches[-1]) if launches else None
    process_observation = (
        "missing" if latest is None else _process_observation(latest)
    )
    supervisor_status, supervisor_status_error = _nested_supervisor_status(
        state.get("supervisor_state_path")
    )
    handoff_observation = (
        "missing"
        if latest is None
        else _launch_handoff_observation(latest, supervisor_status)
    )
    output_root_value = state.get("output_root")
    if output_root_value is None:
        raise ValueError("supervisor launch output_root is missing")
    output_root = Path(str(output_root_value)).expanduser().resolve()
    launch_lock = _launch_lock_observation(output_root)
    status = _observed_launch_status(
        latest,
        process_observation=process_observation,
        handoff_observation=handoff_observation,
        supervisor_status=supervisor_status,
        launch_lock_status=launch_lock.get("status"),
    )
    healthy_statuses = {
        "completed",
        "paused",
        "resume_budget_reached",
        "running",
        "starting",
        "stopped",
        "stopping",
        "superseded",
    }
    healthy = status in healthy_statuses
    if status not in {"starting", "superseded"} and supervisor_status is not None:
        healthy = healthy and supervisor_status.get("healthy") is True
    if supervisor_status_error is not None:
        healthy = False
    log_path = None if latest is None else latest.get("log_path")
    log = Path(str(log_path)).expanduser() if log_path is not None else None
    log_exists = bool(log is not None and log.is_file())
    if latest is not None and latest.get("status") != "launching" and not log_exists:
        healthy = False
    return {
        "row_type": "hf_adapter_continuation_executor_supervisor_launch_status",
        "schema": HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_STATUS_SCHEMA,
        "created_at": _now(),
        "status": status,
        "healthy": healthy,
        "recommended_action": _launch_recommended_action(status, supervisor_status),
        "supervisor_launch_state_path": str(launch_state_path),
        "executor_launch_state_path": state.get("executor_launch_state_path"),
        "supervisor_state_path": state.get("supervisor_state_path"),
        "output_root": str(output_root),
        "launch_count": len(launches),
        "latest_launch": latest,
        "launcher_process_observation": process_observation,
        "launcher_pid_alive_observed": (
            True
            if process_observation == "alive"
            else False
            if process_observation == "exited"
            else None
        ),
        "supervisor_handoff_observation": handoff_observation,
        "supervisor_handoff_established": handoff_observation
        in {"baseline_advanced", "recorded_match"},
        "supervisor_status": supervisor_status,
        "supervisor_status_error": supervisor_status_error,
        "launch_lock": launch_lock,
        "log_path": None if log is None else str(log.resolve(strict=False)),
        "log_exists": log_exists,
    }


def hf_adapter_continuation_executor_supervisor_launch_lines(
    report_or_path: Mapping[str, object] | str | Path,
) -> list[str]:
    report = (
        dict(report_or_path)
        if isinstance(report_or_path, Mapping)
        else load_hf_adapter_continuation_executor_supervisor_launch(report_or_path)
    )
    launches = report.get("launches") or []
    latest = report.get("latest_launch")
    if not isinstance(latest, Mapping):
        latest = launches[-1] if launches and isinstance(launches[-1], Mapping) else {}
    return [
        "hf_adapter_continuation_executor_supervisor_launch "
        f"request={report.get('request_status')} "
        f"created={report.get('created')} "
        f"status={report.get('status')} "
        f"launch={latest.get('launch_id')} "
        f"pid={latest.get('pid')} "
        f"run={latest.get('supervisor_run_id')} "
        f"invocation={latest.get('supervisor_invocation_count')} "
        f"launches={report.get('launch_count', len(launches))} "
        f"state={report.get('supervisor_launch_state_path')}"
    ]


def hf_adapter_continuation_executor_supervisor_launch_status_lines(
    report_or_path: Mapping[str, object] | str | Path,
) -> list[str]:
    report = (
        dict(report_or_path)
        if isinstance(report_or_path, Mapping)
        and report_or_path.get("row_type")
        == "hf_adapter_continuation_executor_supervisor_launch_status"
        else hf_adapter_continuation_executor_supervisor_launch_status_report(
            report_or_path
        )
    )
    return [
        "hf_adapter_continuation_executor_supervisor_launch_status "
        f"status={report.get('status')} "
        f"healthy={report.get('healthy')} "
        f"action={report.get('recommended_action')} "
        f"launches={report.get('launch_count')} "
        f"pid_alive={report.get('launcher_pid_alive_observed')} "
        f"handoff={report.get('supervisor_handoff_observation')} "
        f"state={report.get('supervisor_launch_state_path')}"
    ]
