from __future__ import annotations

import json
import os
import signal
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
import spiraltorch as st
from spiraltorch import hf_adapter_executor_runtime as runtime
from spiraltorch import hf_adapter_executor_supervisor as supervisor
from spiraltorch import hf_adapter_executor_supervisor_launch as supervisor_launch
from spiraltorch import hf_cli
from tests.test_hf_adapter_executor_supervisor import _write_launch_artifact


def _wait_for_runtime_status(
    path: Path,
    expected: set[str],
    *,
    timeout_seconds: float = 5.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    report: dict[str, object] = {}
    while time.monotonic() < deadline:
        report = runtime.hf_adapter_continuation_executor_runtime_report(path)
        if report.get("status") in expected:
            return report
        time.sleep(0.02)
    raise AssertionError(f"runtime did not reach {expected}: {report}")


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


def test_runtime_report_unifies_unmanaged_and_terminal_layers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    active_launch, active_state = _write_launch_artifact(
        tmp_path / "active",
        executor_status="auditing",
        executor_action="audit_chain",
        executor_reason="executor_started",
        launcher_status="handed_off",
        launcher_alive=True,
    )
    try:
        active = runtime.hf_adapter_continuation_executor_runtime_report(active_launch)
        lines = runtime.hf_adapter_continuation_executor_runtime_lines(active)
        cli_code = hf_cli.adapter_continuation_executor_runtime_main(
            [str(active_launch)]
        )
        cli_output = capsys.readouterr().out

        assert active["status"] == "unmanaged_running"
        assert active["healthy"] is True
        assert active["operational_ready"] is False
        assert active["managed"] is False
        assert active["identity_verified"] is True
        assert active["reconcile_action"] == "launch_supervisor"
        assert lines[0].startswith("hf_adapter_continuation_executor_runtime ")
        assert cli_code == 0
        assert "status=unmanaged_running" in cli_output
        assert st.hf_adapter_continuation_executor_runtime_report is (
            runtime.hf_adapter_continuation_executor_runtime_report
        )
    finally:
        (
            active_state.parent / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
        ).unlink(missing_ok=True)

    resume_launch, _ = _write_launch_artifact(tmp_path / "resume")
    resume_ready = runtime.hf_adapter_continuation_executor_runtime_report(
        resume_launch
    )
    resume_lines = runtime.hf_adapter_continuation_executor_runtime_lines(
        resume_ready
    )

    assert resume_ready["status"] == "unmanaged_resume_ready"
    assert resume_ready["generation_plan_status"] == "ready"
    assert str(resume_ready["generation_plan_id"]).startswith("sha256:")
    assert f"plan_id={resume_ready['generation_plan_id']}" in resume_lines[0]

    terminal_launch, _ = _write_launch_artifact(
        tmp_path / "terminal",
        executor_status="stopped",
        executor_action="stop_training",
        executor_reason="continuation_policy_stop",
    )
    terminal = runtime.hf_adapter_continuation_executor_runtime_report(terminal_launch)
    reconciled = runtime.reconcile_hf_adapter_continuation_executor_runtime(
        terminal_launch
    )

    assert terminal["status"] == "completed"
    assert terminal["healthy"] is True
    assert terminal["operational_ready"] is True
    assert terminal["reconcile_action"] == "none"
    assert reconciled["request_status"] == "no_action"
    assert reconciled["succeeded"] is True
    assert reconciled["created"] is False

    blocked_launch, _ = _write_launch_artifact(
        tmp_path / "blocked",
        executor_status="blocked",
        executor_action="inspect_executor_state",
        executor_reason="executor_state_requires_operator",
    )
    blocked = runtime.hf_adapter_continuation_executor_runtime_report(blocked_launch)

    assert blocked["status"] == "blocked"
    assert blocked["healthy"] is False
    assert blocked["identity_verified"] is True
    assert blocked["integrity_verified"] is True
    assert blocked["reconcile_safe"] is False


def test_runtime_rejects_cross_layer_identity_mismatch(tmp_path: Path) -> None:
    launch_path, state_path = _write_launch_artifact(
        tmp_path,
        executor_status="auditing",
        executor_action="audit_chain",
        executor_reason="executor_started",
        launcher_status="handed_off",
        launcher_alive=True,
    )
    output_root = state_path.parent
    supervisor_state_path = (
        output_root / supervisor.HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_FILENAME
    )
    supervisor_launch_path = (
        output_root
        / supervisor_launch.HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_FILENAME
    )
    now = datetime.now(timezone.utc).isoformat()
    supervisor_launch_path.write_text(
        json.dumps(
            {
                "row_type": "hf_adapter_continuation_executor_supervisor_launches",
                "schema": supervisor_launch.HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_SCHEMA,
                "created_at": now,
                "updated_at": now,
                "status": "initializing",
                "executor_launch_state_path": str(
                    (tmp_path / "different-launch.json").resolve()
                ),
                "supervisor_state_path": str(supervisor_state_path.resolve()),
                "supervisor_launch_state_path": str(supervisor_launch_path.resolve()),
                "output_root": str(output_root.resolve()),
                "launch_count": 0,
                "launches": [],
            }
        ),
        encoding="utf-8",
    )
    try:
        report = runtime.hf_adapter_continuation_executor_runtime_report(launch_path)
        reconciled = runtime.reconcile_hf_adapter_continuation_executor_runtime(
            launch_path
        )

        assert report["status"] == "invalid"
        assert report["healthy"] is False
        assert report["identity_verified"] is False
        assert (
            "supervisor_launch.executor_launch_state_path_mismatch"
            in report["identity_issues"]
        )
        assert reconciled["request_status"] == "blocked"
        assert reconciled["succeeded"] is False
        assert reconciled["created"] is False
    finally:
        (output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME).unlink(
            missing_ok=True
        )


def test_runtime_reconcile_preserves_stop_boundary_and_explicit_restart(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    launch_path, state_path = _write_launch_artifact(
        tmp_path,
        executor_status="auditing",
        executor_action="audit_chain",
        executor_reason="executor_started",
        launcher_status="handed_off",
        launcher_alive=True,
    )
    output_root = state_path.parent
    pids: list[object] = []
    try:
        first = runtime.reconcile_hf_adapter_continuation_executor_runtime(
            launch_path,
            max_resumes=2,
            poll_interval_seconds=0.02,
            timeout_seconds=5.0,
            launch_handoff_timeout_seconds=3.0,
        )
        first_launch = first["supervisor_launch"]
        assert isinstance(first_launch, dict)
        pids.append(first_launch["latest_launch"]["pid"])
        duplicate_code = hf_cli.adapter_continuation_executor_runtime_main(
            [
                str(launch_path),
                "--reconcile",
                "--max-resumes",
                "2",
                "--poll-interval-seconds",
                "0.02",
                "--timeout-seconds",
                "5",
                "--launch-handoff-timeout-seconds",
                "3",
            ]
        )
        duplicate_output = capsys.readouterr().out
        managed = runtime.hf_adapter_continuation_executor_runtime_report(launch_path)

        assert first["request_status"] == "handed_off"
        assert first["succeeded"] is True
        assert first["created"] is True
        assert first["before"]["status"] == "unmanaged_running"
        assert first["after"]["status"] == "managed_running"
        assert duplicate_code == 0
        assert "request=already_managed" in duplicate_output
        assert managed["managed"] is True
        assert managed["operational_ready"] is True

        launch_history = (
            supervisor_launch.load_hf_adapter_continuation_executor_supervisor_launch(
                managed["supervisor_launch_state_path"]
            )
        )
        Path(launch_history["launches"][-1]["log_path"]).unlink()
        degraded = runtime.hf_adapter_continuation_executor_runtime_report(launch_path)

        assert degraded["status"] == "managed_running"
        assert degraded["managed"] is True
        assert degraded["healthy"] is False
        assert degraded["operational_ready"] is False
        assert degraded["requires_operator"] is True
        assert "supervisor_launch:running_unhealthy" in degraded["health_issues"]

        supervisor.request_hf_adapter_continuation_executor_supervisor_stop(
            managed["supervisor_state_path"],
            reason="runtime_boundary_test",
        )
        stopped = _wait_for_runtime_status(launch_path, {"supervisor_stopped"})
        boundary = runtime.reconcile_hf_adapter_continuation_executor_runtime(
            launch_path
        )

        assert stopped["healthy"] is True
        assert stopped["operational_ready"] is False
        assert stopped["requires_operator"] is True
        assert stopped["reconcile_requires_restart"] is True
        assert boundary["request_status"] == "operator_restart_required"
        assert boundary["succeeded"] is False
        assert boundary["created"] is False

        restarted = runtime.reconcile_hf_adapter_continuation_executor_runtime(
            launch_path,
            restart_supervisor=True,
            max_resumes=2,
            poll_interval_seconds=0.02,
            timeout_seconds=5.0,
            launch_handoff_timeout_seconds=3.0,
        )
        second_launch = restarted["supervisor_launch"]
        assert isinstance(second_launch, dict)
        pids.append(second_launch["latest_launch"]["pid"])
        assert restarted["request_status"] == "handed_off"
        assert restarted["succeeded"] is True
        assert restarted["created"] is True
        assert restarted["after"]["status"] == "managed_running"

        supervisor.request_hf_adapter_continuation_executor_supervisor_stop(
            restarted["after"]["supervisor_state_path"],
            reason="runtime_restart_test_complete",
        )
        final = _wait_for_runtime_status(launch_path, {"supervisor_stopped"})
        history = supervisor.load_hf_adapter_continuation_executor_supervisor(
            final["supervisor_state_path"]
        )
        executor_lock = output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME

        assert history["invocation_count"] == 2
        assert [row["status"] for row in history["runs"]] == [
            "stopped",
            "stopped",
        ]
        assert executor_lock.is_file()
        assert not (
            output_root
            / supervisor.HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LOCK_FILENAME
        ).exists()
        assert not (
            output_root
            / supervisor_launch.HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_LOCK_FILENAME
        ).exists()

        history_payload = json.loads(
            Path(final["supervisor_state_path"]).read_text(encoding="utf-8")
        )
        history_payload["status"] = "timed_out"
        history_payload["runs"][-1].update(
            {
                "status": "timed_out",
                "action": "inspect_executor_status",
                "healthy": False,
                "reason": "supervisor_timeout_reached",
            }
        )
        Path(final["supervisor_state_path"]).write_text(
            json.dumps(history_payload),
            encoding="utf-8",
        )
        timed_out = runtime.hf_adapter_continuation_executor_runtime_report(launch_path)
        timed_out_boundary = runtime.reconcile_hf_adapter_continuation_executor_runtime(
            launch_path
        )

        assert timed_out["status"] == "supervisor_timed_out"
        assert timed_out["healthy"] is False
        assert timed_out["reconcile_safe"] is True
        assert timed_out["reconcile_requires_restart"] is True
        assert timed_out_boundary["request_status"] == "operator_restart_required"
        assert timed_out_boundary["succeeded"] is False
    finally:
        for pid in reversed(pids):
            _terminate_verified_supervisor(output_root, pid)
        (output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME).unlink(
            missing_ok=True
        )
