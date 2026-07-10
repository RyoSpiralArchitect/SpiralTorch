"""Detached launch control for Hugging Face adapter continuation executors."""

from __future__ import annotations

import json
import math
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ._process import local_pid_alive
from .hf_adapter_executor import (
    HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME,
    _load_executor_lock,
    _terminate_owned_process,
    load_hf_adapter_continuation_executor,
)
from .hf_adapter_executor_status import (
    hf_adapter_continuation_executor_status_report,
)

__all__ = [
    "HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_FILENAME",
    "HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_LOCK_FILENAME",
    "HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_SCHEMA",
    "HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_STATUS_SCHEMA",
    "hf_adapter_continuation_executor_launch_lines",
    "hf_adapter_continuation_executor_launch_status_lines",
    "hf_adapter_continuation_executor_launch_status_report",
    "launch_hf_adapter_continuation_executor",
    "load_hf_adapter_continuation_executor_launch",
]


HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_SCHEMA = (
    "spiraltorch.hf_adapter_continuation_executor_launch.v1"
)
HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_FILENAME = (
    "spiraltorch-hf-adapter-continuation-executor-launch.json"
)
HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_LOCK_FILENAME = (
    ".spiraltorch-hf-adapter-continuation-executor-launch.lock"
)
HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_STATUS_SCHEMA = (
    "spiraltorch.hf_adapter_continuation_executor_launch_status.v1"
)

_LAUNCH_HANDOFF_POLL_SECONDS = 0.05
_LAUNCH_TERMINATION_GRACE_SECONDS = 2.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_symlink():
        raise RuntimeError(f"executor launch state cannot be a symbolic link: {path}")
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


def load_hf_adapter_continuation_executor_launch(
    value: str | Path,
) -> dict[str, object]:
    """Load and validate detached executor launch history."""

    path = Path(value).expanduser()
    if path.is_symlink():
        raise ValueError(f"executor launch state cannot be a symbolic link: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"executor launch state must contain a JSON object: {path}")
    if payload.get("schema") != HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_SCHEMA:
        raise ValueError(
            f"unsupported HF adapter executor launch schema: {payload.get('schema')}"
        )
    if payload.get("row_type") != "hf_adapter_continuation_executor_launches":
        raise ValueError(
            "unsupported HF adapter executor launch row type: "
            f"{payload.get('row_type')}"
        )
    launches = payload.get("launches")
    if not isinstance(launches, list):
        raise ValueError("executor launch state launches must be a list")
    report = dict(payload)
    report["launch_state_path"] = str(path.resolve())
    return report


def _load_launch_lock(path: Path) -> dict[str, object] | None:
    if path.is_symlink():
        raise RuntimeError(f"executor launch lock cannot be a symbolic link: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"executor launch lock is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"executor launch lock must contain a JSON object: {path}")
    return dict(payload)


def _launch_lock_owner_is_stale(owner: Mapping[str, object]) -> bool:
    return bool(
        owner.get("row_type") == "hf_adapter_continuation_executor_launch_lock"
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
def _launch_lock(output_root: Path) -> Iterator[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_LOCK_FILENAME
    owner = {
        "row_type": "hf_adapter_continuation_executor_launch_lock",
        "lock_id": f"executor-launch-lock-{uuid.uuid4().hex}",
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
            existing = _load_launch_lock(path)
            if existing is None:
                continue
            if _launch_lock_owner_is_stale(existing):
                if not _reap_stale_launch_lock(path):
                    time.sleep(0.05)
                continue
            raise RuntimeError(
                "executor detached launch is locked; inspect the recorded owner: "
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
        raise RuntimeError(f"could not acquire executor launch lock: {path}")
    try:
        yield path
    finally:
        try:
            current = _load_launch_lock(path)
        except RuntimeError:
            current = None
        if current is not None and current.get("lock_id") == owner["lock_id"]:
            path.unlink(missing_ok=True)


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


def _executor_lock_observation(output_root: Path) -> dict[str, object]:
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
    observation = _process_observation(owner)
    return {
        "path": str(path),
        "status": observation,
        "owner": owner,
        "error": None,
    }


def _new_launch_state(
    *,
    output_root: Path,
    executor_state_path: Path,
    launch_state_path: Path,
) -> dict[str, object]:
    created_at = _now()
    return {
        "row_type": "hf_adapter_continuation_executor_launches",
        "schema": HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_SCHEMA,
        "created_at": created_at,
        "updated_at": created_at,
        "status": "initializing",
        "output_root": str(output_root),
        "executor_state_path": str(executor_state_path),
        "launch_state_path": str(launch_state_path),
        "launch_count": 0,
        "launches": [],
    }


def _launch_state_for_request(
    *,
    output_root: Path,
    executor_state_path: Path,
    launch_state_path: Path,
) -> dict[str, object]:
    state = (
        load_hf_adapter_continuation_executor_launch(launch_state_path)
        if launch_state_path.is_file()
        else _new_launch_state(
            output_root=output_root,
            executor_state_path=executor_state_path,
            launch_state_path=launch_state_path,
        )
    )
    if state.get("output_root") != str(output_root):
        raise ValueError("executor launch output_root differs; use a new launch state")
    if state.get("executor_state_path") != str(executor_state_path):
        raise ValueError("executor state path differs; use a new launch state")
    state["launch_state_path"] = str(launch_state_path)
    return state


def _executor_child_command(executor_argv: Sequence[str]) -> list[str]:
    script = (
        "import sys; "
        "from spiraltorch.hf_cli import adapter_continuation_executor_main; "
        "raise SystemExit(adapter_continuation_executor_main(sys.argv[1:]))"
    )
    return [sys.executable, "-c", script, *[str(value) for value in executor_argv]]


def _process_group_options() -> dict[str, object]:
    if os.name == "posix":
        return {"start_new_session": True}
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {}


def _write_launch_state(path: Path, state: dict[str, object]) -> None:
    state["updated_at"] = _now()
    launches = state.get("launches") or []
    state["launch_count"] = len(launches)
    _atomic_write_json(path, state)


def _report_for_request(
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


def _reap_in_background(process: subprocess.Popen[bytes]) -> None:
    threading.Thread(
        target=process.wait,
        name="spiraltorch-adapter-executor-launch-reaper",
        daemon=True,
    ).start()


def _executor_invocation_baseline(path: Path) -> dict[str, object] | None:
    try:
        state = load_hf_adapter_continuation_executor(path)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"existing executor state is unreadable: {path}: {exc}"
        ) from exc
    run_id = state.get("run_id")
    invocation_count = state.get("invocation_count")
    if (
        not isinstance(run_id, str)
        or not run_id
        or isinstance(invocation_count, bool)
        or not isinstance(invocation_count, int)
        or invocation_count <= 0
    ):
        raise ValueError("existing executor state has invalid invocation identity")
    return {
        "run_id": run_id,
        "invocation_count": invocation_count,
        "updated_at": state.get("updated_at"),
        "status": state.get("status"),
    }


def _is_new_executor_invocation(
    state: Mapping[str, object],
    baseline: Mapping[str, object] | None,
) -> bool:
    run_id = state.get("run_id")
    invocation_count = state.get("invocation_count")
    if (
        not isinstance(run_id, str)
        or not run_id
        or isinstance(invocation_count, bool)
        or not isinstance(invocation_count, int)
        or invocation_count <= 0
    ):
        return False
    if baseline is None:
        return True
    return bool(
        run_id == baseline.get("run_id")
        and invocation_count == int(baseline.get("invocation_count") or 0) + 1
    )


def _executor_lock_matches_pid(
    observation: Mapping[str, object],
    pid: object,
) -> bool:
    owner = observation.get("owner")
    return bool(
        observation.get("status") == "alive"
        and isinstance(owner, Mapping)
        and owner.get("row_type") == "hf_adapter_continuation_executor_lock"
        and owner.get("hostname") == socket.gethostname()
        and owner.get("pid") == pid
    )


def _argv_contains_option(argv: Sequence[str], name: str) -> bool:
    return any(value == name or value.startswith(f"{name}=") for value in argv)


def launch_hf_adapter_continuation_executor(
    executor_argv: Sequence[str],
    *,
    output_root: str | Path,
    executor_state_path: str | Path,
    launch_state_path: str | Path | None = None,
    command_cwd: str | Path | None = None,
    handoff_timeout_seconds: float = 5.0,
) -> dict[str, object]:
    """Launch an executor in a detached process and persist handoff evidence."""

    argv = [str(value) for value in executor_argv]
    if not argv:
        raise ValueError("executor_argv must not be empty")
    try:
        handoff_timeout = float(handoff_timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "handoff_timeout_seconds must be finite and non-negative"
        ) from exc
    if not math.isfinite(handoff_timeout) or handoff_timeout < 0.0:
        raise ValueError("handoff_timeout_seconds must be finite and non-negative")
    if not _argv_contains_option(argv, "--run"):
        raise ValueError("detached executor launch requires --run")
    forbidden = next(
        (
            name
            for name in (
                "--detach",
                "--detach-handoff-timeout-seconds",
                "--launch-state",
            )
            if _argv_contains_option(argv, name)
        ),
        None,
    )
    if forbidden is not None:
        raise ValueError(f"executor_argv must not contain launcher option {forbidden}")
    resolved_output_root = Path(output_root).expanduser().resolve()
    resolved_executor_state_path = Path(executor_state_path).expanduser().resolve()
    resolved_launch_state_path = (
        Path(launch_state_path).expanduser().resolve()
        if launch_state_path is not None
        else resolved_output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_FILENAME
    )
    resolved_command_cwd = (
        Path(command_cwd).expanduser().resolve()
        if command_cwd is not None
        else Path.cwd().resolve()
    )
    if resolved_launch_state_path == resolved_executor_state_path:
        raise ValueError("launch_state_path cannot overwrite executor state")
    reserved_paths = {
        resolved_output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME,
        resolved_output_root / HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_LOCK_FILENAME,
    }
    if resolved_launch_state_path in reserved_paths:
        raise ValueError("launch_state_path cannot overwrite an executor lock")
    with _launch_lock(resolved_output_root):
        state = _launch_state_for_request(
            output_root=resolved_output_root,
            executor_state_path=resolved_executor_state_path,
            launch_state_path=resolved_launch_state_path,
        )
        launches = state.setdefault("launches", [])
        if not isinstance(launches, list):
            raise ValueError("executor launch state launches must be a list")
        executor_lock = _executor_lock_observation(resolved_output_root)
        latest = launches[-1] if launches and isinstance(launches[-1], dict) else None
        if latest is not None:
            observation = _process_observation(latest)
            latest["process_liveness_observed_at"] = _now()
            latest["process_liveness_observation"] = observation
            if observation == "alive":
                if _executor_lock_matches_pid(executor_lock, latest.get("pid")):
                    state["executor_lock"] = executor_lock
                    _write_launch_state(resolved_launch_state_path, state)
                    return _report_for_request(
                        state,
                        request_status="already_running",
                        created=False,
                    )
                lock_released = executor_lock.get("status") in {"absent", "exited"}
                launch_finished = latest.get("status") in {
                    "completed",
                    "exited_observed",
                    "handed_off",
                    "launch_failed",
                }
                if lock_released and launch_finished:
                    latest["process_liveness_observation"] = (
                        "pid_alive_without_executor_ownership"
                    )
                    latest["status"] = "exited_observed"
                    latest["process_exited_observed_at"] = _now()
                else:
                    state["executor_lock"] = executor_lock
                    _write_launch_state(resolved_launch_state_path, state)
                    return _report_for_request(
                        state,
                        request_status="launch_owner_unverified",
                        created=False,
                    )
            if observation in {"remote_unverified", "unverified"}:
                _write_launch_state(resolved_launch_state_path, state)
                return _report_for_request(
                    state,
                    request_status="launch_owner_unverified",
                    created=False,
                )
            if latest.get("status") in {"handed_off", "handoff_timeout", "launched"}:
                latest["status"] = "exited_observed"
                latest["process_exited_observed_at"] = _now()

        if executor_lock.get("status") in {
            "alive",
            "remote_unverified",
            "unverified",
            "invalid",
        }:
            state["executor_lock"] = executor_lock
            _write_launch_state(resolved_launch_state_path, state)
            return _report_for_request(
                state,
                request_status="executor_locked",
                created=False,
            )

        executor_baseline = _executor_invocation_baseline(resolved_executor_state_path)

        launch_id = f"executor-launch-{uuid.uuid4().hex}"
        log_dir = resolved_output_root / "executor-logs"
        if log_dir.is_symlink():
            raise RuntimeError(f"executor log directory cannot be a symlink: {log_dir}")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"launcher-{launch_id[-16:]}.log"
        command = _executor_child_command(argv)
        attempt = {
            "launch_id": launch_id,
            "status": "launching",
            "started_at": _now(),
            "hostname": socket.gethostname(),
            "pid": None,
            "command_cwd": str(resolved_command_cwd),
            "command": command,
            "command_display": shlex.join(command),
            "executor_argv": argv,
            "executor_state_path": str(resolved_executor_state_path),
            "executor_baseline": executor_baseline,
            "log_path": str(log_path),
            "log_stream": "combined_stdout_stderr",
            "process_group_isolated": os.name in {"posix", "nt"},
        }
        launches.append(attempt)
        state["status"] = "launching"
        state["latest_launch_id"] = launch_id
        state["executor_lock"] = executor_lock
        _write_launch_state(resolved_launch_state_path, state)

        process: subprocess.Popen[bytes] | None = None
        try:
            descriptor = os.open(
                log_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            with os.fdopen(descriptor, "wb", buffering=0) as log_handle:
                log_handle.write(
                    (
                        f"[spiraltorch-executor-launch] started_at={_now()} "
                        f"command={shlex.join(command)}\n"
                    ).encode("utf-8")
                )
                process = subprocess.Popen(
                    command,
                    cwd=str(resolved_command_cwd),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    **_process_group_options(),
                )
        except BaseException as exc:
            attempt["status"] = "launch_failed"
            attempt["failure"] = f"{exc.__class__.__name__}: {exc}"
            attempt["completed_at"] = _now()
            state["status"] = "launch_failed"
            _write_launch_state(resolved_launch_state_path, state)
            raise

        attempt["pid"] = int(process.pid)
        attempt["process_started_at"] = _now()
        attempt["status"] = "launched"
        state["status"] = "launched"
        try:
            _write_launch_state(resolved_launch_state_path, state)
        except BaseException:
            _terminate_owned_process(
                process,
                grace_seconds=_LAUNCH_TERMINATION_GRACE_SECONDS,
            )
            process.wait()
            raise

        deadline = time.monotonic() + handoff_timeout
        request_status = "handoff_timeout"
        while True:
            if resolved_executor_state_path.is_file():
                try:
                    executor_state = load_hf_adapter_continuation_executor(
                        resolved_executor_state_path
                    )
                except (OSError, ValueError):
                    executor_state = None
                if executor_state is not None and _is_new_executor_invocation(
                    executor_state,
                    executor_baseline,
                ):
                    attempt["executor_status_at_handoff"] = executor_state.get("status")
                    attempt["executor_run_id"] = executor_state.get("run_id")
                    attempt["executor_invocation_count"] = executor_state.get(
                        "invocation_count"
                    )
                    if process.poll() is None:
                        lock_observation = _executor_lock_observation(
                            resolved_output_root
                        )
                        if _executor_lock_matches_pid(
                            lock_observation,
                            process.pid,
                        ):
                            attempt["executor_lock_at_handoff"] = lock_observation
                            attempt["handoff_at"] = _now()
                            attempt["status"] = "handed_off"
                            state["status"] = "handed_off"
                            request_status = "handed_off"
                            break
                    else:
                        attempt["returncode_observed"] = int(process.returncode or 0)
                        attempt["handoff_at"] = _now()
                        attempt["completed_at"] = _now()
                        if process.returncode == 0:
                            attempt["status"] = "completed"
                            state["status"] = "completed"
                            request_status = "completed"
                        else:
                            attempt["status"] = "launch_failed"
                            state["status"] = "launch_failed"
                            request_status = "launch_failed"
                        break
            returncode = process.poll()
            if returncode is not None:
                attempt["status"] = "launch_failed"
                attempt["returncode_observed"] = int(returncode)
                attempt["completed_at"] = _now()
                state["status"] = "launch_failed"
                request_status = "launch_failed"
                break
            if time.monotonic() >= deadline:
                attempt["status"] = "handoff_timeout"
                attempt["handoff_timeout_at"] = _now()
                state["status"] = "handoff_timeout"
                break
            time.sleep(_LAUNCH_HANDOFF_POLL_SECONDS)

        _write_launch_state(resolved_launch_state_path, state)
        if process.poll() is None:
            _reap_in_background(process)
        return _report_for_request(
            state,
            request_status=request_status,
            created=True,
        )


def _path_report(value: object, *, expect_directory: bool = False) -> dict[str, object]:
    if value is None:
        return {"path": None, "exists": False, "kind_ready": False, "size_bytes": None}
    path = Path(str(value)).expanduser()
    exists = path.exists()
    size_bytes = None
    if exists and path.is_file():
        try:
            size_bytes = path.stat().st_size
        except OSError:
            pass
    return {
        "path": str(path.resolve(strict=False)),
        "exists": exists,
        "kind_ready": path.is_dir() if expect_directory else path.is_file(),
        "size_bytes": size_bytes,
    }


def hf_adapter_continuation_executor_launch_status_report(
    report_or_path: Mapping[str, object] | str | Path,
) -> dict[str, object]:
    """Observe detached launcher, executor handoff, and durable artifacts."""

    launch_state = (
        dict(report_or_path)
        if isinstance(report_or_path, Mapping)
        else load_hf_adapter_continuation_executor_launch(report_or_path)
    )
    launches = [
        row for row in launch_state.get("launches") or [] if isinstance(row, Mapping)
    ]
    latest = launches[-1] if launches else None
    observation = "missing" if latest is None else _process_observation(latest)
    executor_state_path = launch_state.get("executor_state_path")
    executor_artifact = _path_report(executor_state_path)
    executor_status = None
    executor_error = None
    if executor_artifact.get("kind_ready") is True and executor_state_path is not None:
        try:
            executor_status = hf_adapter_continuation_executor_status_report(
                str(executor_state_path)
            )
        except (OSError, ValueError) as exc:
            executor_error = f"{exc.__class__.__name__}: {exc}"
    executor_lifecycle = (
        executor_status.get("status") if isinstance(executor_status, Mapping) else None
    )
    executor_healthy = (
        executor_status.get("healthy") if isinstance(executor_status, Mapping) else None
    )
    output_root_value = launch_state.get("output_root")
    executor_lock = (
        _executor_lock_observation(Path(str(output_root_value)).expanduser())
        if output_root_value is not None
        else {"path": None, "status": "unverified", "owner": None, "error": None}
    )
    launcher_owns_executor_lock = bool(
        latest is not None
        and _executor_lock_matches_pid(executor_lock, latest.get("pid"))
    )
    if latest is None:
        status = "empty"
    elif observation == "alive":
        if executor_lifecycle in {"generation_limit_reached", "ready", "stopped"}:
            status = "completed" if executor_healthy is True else "executor_unhealthy"
        elif executor_lifecycle in {"blocked", "failed", "interrupted"}:
            status = f"executor_{executor_lifecycle}"
        elif latest.get("status") == "launch_failed":
            status = "launch_failed"
        elif executor_status is None:
            status = "starting"
        elif executor_healthy is not True:
            status = "executor_unhealthy"
        elif not launcher_owns_executor_lock:
            status = "running_unverified"
        else:
            status = "running"
    elif observation == "remote_unverified":
        status = "remote_running"
    elif observation == "unverified":
        status = "running_unverified"
    elif executor_lifecycle in {"generation_limit_reached", "ready", "stopped"}:
        status = "completed" if executor_healthy is True else "executor_unhealthy"
    elif executor_lifecycle in {"blocked", "failed", "interrupted"}:
        status = f"executor_{executor_lifecycle}"
    elif executor_lifecycle in {
        "auditing",
        "running",
        "running_unverified",
        "stopping",
    }:
        status = "interrupted"
    elif latest.get("status") == "launch_failed":
        status = "launch_failed"
    else:
        status = "exited"

    healthy = status in {"completed", "running", "starting"}
    if status == "starting":
        recommended_action = "wait_for_executor_handoff"
    elif status == "running":
        recommended_action = "inspect_executor_status"
    elif status == "completed":
        recommended_action = "inspect_executor_result"
    elif status == "remote_running":
        recommended_action = "inspect_remote_launcher"
    elif status == "running_unverified":
        recommended_action = "inspect_unverified_launcher"
    elif status == "executor_blocked":
        recommended_action = "resolve_executor_block"
    elif status in {
        "executor_failed",
        "executor_interrupted",
        "executor_unhealthy",
    }:
        recommended_action = "inspect_executor_health"
    else:
        recommended_action = "inspect_launcher_log"
    log = _path_report(None if latest is None else latest.get("log_path"))
    if latest is not None and log.get("kind_ready") is not True:
        healthy = False
        recommended_action = "inspect_missing_launcher_log"
    return {
        "row_type": "hf_adapter_continuation_executor_launch_status",
        "schema": HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_STATUS_SCHEMA,
        "created_at": _now(),
        "status": status,
        "healthy": healthy,
        "recommended_action": recommended_action,
        "launch_state_path": launch_state.get("launch_state_path"),
        "launch_count": len(launches),
        "latest_launch": None if latest is None else dict(latest),
        "launcher_process_observation": observation,
        "launcher_pid_alive_observed": (
            True
            if observation == "alive"
            else False
            if observation == "exited"
            else None
        ),
        "launcher_executor_lock_owner_verified": launcher_owns_executor_lock,
        "executor_lock": executor_lock,
        "log": log,
        "executor_state": executor_artifact,
        "executor_status": executor_status,
        "executor_healthy": executor_healthy,
        "executor_status_error": executor_error,
    }


def hf_adapter_continuation_executor_launch_lines(
    report_or_path: Mapping[str, object] | str | Path,
) -> list[str]:
    report = (
        dict(report_or_path)
        if isinstance(report_or_path, Mapping)
        else load_hf_adapter_continuation_executor_launch(report_or_path)
    )
    launches = report.get("launches") or []
    latest = report.get("latest_launch") or (launches[-1] if launches else None)
    lines = [
        "hf_adapter_continuation_executor_launch "
        f"status={report.get('request_status', report.get('status'))} "
        f"created={report.get('created')} "
        f"launches={report.get('launch_count', len(launches))} "
        f"state={report.get('executor_state_path')} "
        f"launch_state={report.get('launch_state_path')}"
    ]
    if isinstance(latest, Mapping):
        lines.append(
            "hf_adapter_continuation_executor_launcher "
            f"launch={latest.get('launch_id')} "
            f"status={latest.get('status')} "
            f"host={latest.get('hostname')} "
            f"pid={latest.get('pid')} "
            f"executor={latest.get('executor_status_at_handoff')} "
            f"log={latest.get('log_path')}"
        )
    return lines


def hf_adapter_continuation_executor_launch_status_lines(
    report_or_path: Mapping[str, object] | str | Path,
) -> list[str]:
    report = (
        dict(report_or_path)
        if isinstance(report_or_path, Mapping)
        and report_or_path.get("row_type")
        == "hf_adapter_continuation_executor_launch_status"
        else hf_adapter_continuation_executor_launch_status_report(report_or_path)
    )
    latest = report.get("latest_launch")
    log = report.get("log")
    executor = report.get("executor_status")
    return [
        "hf_adapter_continuation_executor_launch_status "
        f"status={report.get('status')} "
        f"healthy={report.get('healthy')} "
        f"action={report.get('recommended_action')} "
        f"launches={report.get('launch_count')} "
        f"pid={latest.get('pid') if isinstance(latest, Mapping) else None} "
        f"pid_alive={report.get('launcher_pid_alive_observed')} "
        "lock_owner_verified="
        f"{report.get('launcher_executor_lock_owner_verified')} "
        f"executor={executor.get('status') if isinstance(executor, Mapping) else None} "
        f"executor_healthy={report.get('executor_healthy')} "
        f"log_exists={log.get('exists') if isinstance(log, Mapping) else None} "
        f"launch_state={report.get('launch_state_path')}"
    ]
