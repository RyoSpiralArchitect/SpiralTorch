from __future__ import annotations

import json
import os
import signal
import socket
import threading
import time
from pathlib import Path

import pytest
import spiraltorch as st
from spiraltorch import hf_adapter_executor_supervisor as supervisor
from spiraltorch import hf_adapter_executor_supervisor_launch as supervisor_launch
from spiraltorch import hf_cli
from tests.test_hf_adapter_executor_supervisor import _write_launch_artifact


def _wait_for_launch_status(
    path: Path,
    expected: set[str],
    *,
    timeout_seconds: float = 5.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    report: dict[str, object] = {}
    while time.monotonic() < deadline:
        report = (
            supervisor_launch.hf_adapter_continuation_executor_supervisor_launch_status_report(
                path
            )
        )
        if report.get("status") in expected:
            return report
        time.sleep(0.02)
    raise AssertionError(f"supervisor launch did not reach {expected}: {report}")


def _terminate_verified_supervisor(output_root: Path, pid: object) -> None:
    lock_path = (
        output_root
        / supervisor.HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LOCK_FILENAME
    )
    try:
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return
    if (
        not isinstance(pid, int)
        or owner.get("pid") != pid
        or owner.get("hostname") != socket.gethostname()
    ):
        return
    if os.name == "posix":
        os.killpg(pid, signal.SIGTERM)
    else:
        os.kill(pid, signal.SIGTERM)


def test_detached_supervisor_launch_stop_and_restart_real_process(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    executor_launch_path, executor_state_path = _write_launch_artifact(
        tmp_path,
        executor_status="auditing",
        executor_action="audit_chain",
        executor_reason="executor_started",
        launcher_status="handed_off",
        launcher_alive=True,
    )
    output_root = executor_state_path.parent
    first_pid: object = None
    second_pid: object = None
    try:
        first_code = hf_cli.adapter_continuation_executor_supervise_main(
            [
                str(executor_launch_path),
                "--detach",
                "--max-resumes",
                "2",
                "--poll-interval-seconds",
                "0.02",
                "--timeout-seconds",
                "5",
                "--detach-handoff-timeout-seconds",
                "3",
            ]
        )
        first_output = capsys.readouterr().out
        supervisor_launch_state_path = (
            output_root
            / supervisor_launch.HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_FILENAME
        )
        first = (
            supervisor_launch.load_hf_adapter_continuation_executor_supervisor_launch(
                supervisor_launch_state_path
            )
        )
        first["latest_launch"] = first["launches"][-1]
        first_pid = first["latest_launch"]["pid"]
        running = (
            supervisor_launch.hf_adapter_continuation_executor_supervisor_launch_status_report(
                first["supervisor_launch_state_path"]
            )
        )
        duplicate = (
            supervisor_launch.launch_hf_adapter_continuation_executor_supervisor(
                executor_launch_path,
                max_resumes=2,
                poll_interval_seconds=0.02,
                timeout_seconds=5.0,
                launch_handoff_timeout_seconds=3.0,
            )
        )
        status_code = hf_cli.adapter_continuation_executor_supervisor_status_main(
            [str(first["supervisor_state_path"])]
        )
        status_output = capsys.readouterr().out
        launch_status_code = (
            hf_cli.adapter_continuation_executor_supervisor_launch_status_main(
                [str(first["supervisor_launch_state_path"])]
            )
        )
        launch_status_output = capsys.readouterr().out
        stop_code = hf_cli.adapter_continuation_executor_supervisor_stop_main(
            [
                str(first["supervisor_state_path"]),
                "--reason",
                "detached_test_stop",
            ]
        )
        stop_output = capsys.readouterr().out
        stop = supervisor.request_hf_adapter_continuation_executor_supervisor_stop(
            first["supervisor_state_path"]
        )
        stopped = _wait_for_launch_status(
            Path(first["supervisor_launch_state_path"]),
            {"stopped"},
        )

        second = supervisor_launch.launch_hf_adapter_continuation_executor_supervisor(
            executor_launch_path,
            max_resumes=2,
            poll_interval_seconds=0.02,
            timeout_seconds=5.0,
            launch_handoff_timeout_seconds=3.0,
        )
        second_pid = second["latest_launch"]["pid"]
        second_stop = (
            supervisor.request_hf_adapter_continuation_executor_supervisor_stop(
                second["supervisor_state_path"],
                reason="detached_test_restart_stop",
            )
        )
        second_stopped = _wait_for_launch_status(
            Path(second["supervisor_launch_state_path"]),
            {"stopped"},
        )
        persisted = (
            supervisor_launch.load_hf_adapter_continuation_executor_supervisor_launch(
                second["supervisor_launch_state_path"]
            )
        )
        supervisor_state = supervisor.load_hf_adapter_continuation_executor_supervisor(
            second["supervisor_state_path"]
        )

        assert first_code == 0
        assert "request=handed_off" in first_output
        assert first["status"] == "handed_off"
        assert running["status"] == "running"
        assert running["healthy"] is True
        assert running["supervisor_handoff_established"] is True
        assert duplicate["request_status"] == "already_running"
        assert duplicate["created"] is False
        assert status_code == 0
        assert "status=running" in status_output
        assert launch_status_code == 0
        assert "status=running" in launch_status_output
        assert stop_code == 0
        assert "created=True" in stop_output
        assert stop["created"] is False
        assert stop["reason"] == "detached_test_stop"
        assert stopped["healthy"] is True
        assert second["request_status"] == "handed_off"
        assert second["created"] is True
        assert second_stop["created"] is True
        assert second_stopped["healthy"] is True
        assert persisted["launch_count"] == 2
        assert supervisor_state["invocation_count"] == 2
        assert [row["status"] for row in supervisor_state["runs"]] == [
            "stopped",
            "stopped",
        ]
        assert not (
            output_root
            / supervisor.HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LOCK_FILENAME
        ).exists()
        assert not (
            output_root
            / supervisor_launch.HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_LOCK_FILENAME
        ).exists()
    finally:
        _terminate_verified_supervisor(output_root, second_pid)
        _terminate_verified_supervisor(output_root, first_pid)
        (
            output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
        ).unlink(missing_ok=True)


def test_supervisor_launch_rejects_non_actionable_and_unsafe_paths(
    tmp_path: Path,
) -> None:
    completed_launch, _ = _write_launch_artifact(
        tmp_path / "completed",
        executor_status="stopped",
        executor_action="stop_training",
        executor_reason="continuation_policy_stop",
    )
    with pytest.raises(RuntimeError, match="not launchable"):
        supervisor_launch.launch_hf_adapter_continuation_executor_supervisor(
            completed_launch
        )

    active_launch, active_state = _write_launch_artifact(
        tmp_path / "symlink",
        executor_status="auditing",
        executor_action="audit_chain",
        executor_reason="executor_started",
        launcher_status="handed_off",
        launcher_alive=True,
    )
    target = tmp_path / "launch-target.json"
    link = tmp_path / "launch-link.json"
    link.symlink_to(target)
    try:
        with pytest.raises(ValueError, match="cannot be a symlink"):
            supervisor_launch.launch_hf_adapter_continuation_executor_supervisor(
                active_launch,
                supervisor_launch_state_path=link,
            )
        assert not target.exists()
    finally:
        (
            active_state.parent / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
        ).unlink(missing_ok=True)


def test_detach_is_idempotent_before_rechecking_source_launchability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor_launch_path, executor_state_path = _write_launch_artifact(
        tmp_path,
        executor_status="auditing",
        executor_action="audit_chain",
        executor_reason="executor_started",
        launcher_status="handed_off",
        launcher_alive=True,
    )
    output_root = executor_state_path.parent
    supervisor_state_path = (
        output_root / supervisor.HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_FILENAME
    )
    errors: list[BaseException] = []

    def run_supervisor() -> None:
        try:
            supervisor.supervise_hf_adapter_continuation_executor(
                executor_launch_path,
                max_resumes=2,
                poll_interval_seconds=0.02,
                timeout_seconds=5.0,
            )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run_supervisor, daemon=True)
    worker.start()
    deadline = time.monotonic() + 3.0
    while not supervisor_state_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert supervisor_state_path.is_file()
    monkeypatch.setattr(
        supervisor_launch,
        "hf_adapter_continuation_executor_supervision_report",
        lambda _path: pytest.fail(
            "verified duplicate launch rechecked executor launchability"
        ),
    )
    try:
        duplicate = (
            supervisor_launch.launch_hf_adapter_continuation_executor_supervisor(
                executor_launch_path
            )
        )
        assert duplicate["request_status"] == "already_running"
        assert duplicate["created"] is False
    finally:
        supervisor.request_hf_adapter_continuation_executor_supervisor_stop(
            supervisor_state_path,
            reason="idempotency_test_complete",
        )
        worker.join(timeout=3.0)
        (
            output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
        ).unlink(missing_ok=True)
    assert errors == []
    assert not worker.is_alive()
