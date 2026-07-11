from __future__ import annotations

import json
import os
import signal
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
import spiraltorch as st
from spiraltorch import hf_adapter_executor, hf_adapter_executor_launch
from spiraltorch import hf_adapter_executor_supervisor as supervisor
from spiraltorch import hf_cli


def _sealed_pending_generation(
    state_path: Path,
    *,
    lineage_depth: int,
) -> dict[str, object]:
    output_dir = state_path.parent / f"generation-{lineage_depth:03d}"
    pending: dict[str, object] = {
        "status": "planned",
        "parent_adapter_id": "sha256:" + "2" * 64,
        "parent_adapter_path": str(state_path.parent / "parent-adapter"),
        "lineage_depth": lineage_depth,
        "output_dir": str(output_dir),
        "source_transition": None,
        "command": {
            "command": [
                sys.executable,
                "-c",
                "raise SystemExit(0)",
                "--output-dir",
                str(output_dir),
            ]
        },
    }
    hf_adapter_executor._seal_generation_plan(
        pending,
        policy={"status": "continue"},
        scale_up={"max_steps": 1},
        max_generations=1,
        execution={
            "command_cwd": str(state_path.parent),
            "command_env_override_keys": [],
            "command_env_override_count": 0,
            "command_env_sha256": None,
        },
    )
    return pending


def _executor_state(
    path: Path,
    *,
    status: str = "generation_limit_reached",
    action: str = "resume_executor",
    reason: str = "max_generations_per_invocation_reached",
    invocation_count: int = 1,
) -> dict[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    state: dict[str, object] = {
        "row_type": "hf_adapter_continuation_executor",
        "schema": st.HF_ADAPTER_CONTINUATION_EXECUTOR_SCHEMA,
        "status": status,
        "action": action,
        "reason": reason,
        "created_at": now,
        "updated_at": now,
        "run_id": "supervisor-test-run",
        "source_paths": [],
        "output_root": str(path.parent.resolve()),
        "state_path": str(path.resolve()),
        "invocation_count": invocation_count,
        "generation_attempt_count": 0,
        "promoted_generation_count": 0,
        "generations": [],
        "execution": {
            "lock_path": str(
                path.parent / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
            )
        },
    }
    if (
        status == "generation_limit_reached"
        and action == "resume_executor"
        and reason == "max_generations_per_invocation_reached"
    ):
        state["pending_generation"] = _sealed_pending_generation(
            path,
            lineage_depth=invocation_count + 1,
        )
    return state


def _write_launch_artifact(
    tmp_path: Path,
    *,
    executor_status: str = "generation_limit_reached",
    executor_action: str = "resume_executor",
    executor_reason: str = "max_generations_per_invocation_reached",
    launcher_status: str = "completed",
    launcher_alive: bool = False,
) -> tuple[Path, Path]:
    output_root = tmp_path / "executor"
    output_root.mkdir(parents=True)
    state_path = output_root / "state.json"
    state = _executor_state(
        state_path,
        status=executor_status,
        action=executor_action,
        reason=executor_reason,
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")
    command_cwd = tmp_path.resolve()
    argv = [
        str((tmp_path / "source").resolve()),
        "--output-root",
        str(output_root.resolve()),
        "--state",
        str(state_path.resolve()),
        "--run",
        "--max-generations",
        "1",
    ]
    log_path = output_root / "launcher.log"
    log_path.write_text("launcher\n", encoding="utf-8")
    pid = os.getpid() if launcher_alive else 99_999_999
    latest = {
        "launch_id": "supervisor-source-launch",
        "status": launcher_status,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": pid,
        "command_cwd": str(command_cwd),
        "command": hf_adapter_executor_launch._executor_child_command(argv),
        "executor_argv": argv,
        "resume_contract": hf_adapter_executor_launch._resume_contract(
            argv,
            output_root=output_root.resolve(),
            executor_state_path=state_path.resolve(),
            command_cwd=command_cwd,
        ),
        "executor_state_path": str(state_path.resolve()),
        "executor_baseline": None,
        "executor_run_id": "supervisor-test-run",
        "executor_invocation_count": 1,
        "log_path": str(log_path.resolve()),
        "process_group_isolated": True,
    }
    launch_path = output_root / "launch.json"
    payload = {
        "row_type": "hf_adapter_continuation_executor_launches",
        "schema": st.HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": launcher_status,
        "output_root": str(output_root.resolve()),
        "executor_state_path": str(state_path.resolve()),
        "launch_state_path": str(launch_path.resolve()),
        "latest_launch_id": latest["launch_id"],
        "launch_count": 1,
        "launches": [latest],
    }
    launch_path.write_text(json.dumps(payload), encoding="utf-8")
    if launcher_alive:
        lock_path = output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
        lock_path.write_text(
            json.dumps(
                {
                    "row_type": "hf_adapter_continuation_executor_lock",
                    "lock_id": "supervisor-active-executor",
                    "pid": pid,
                    "hostname": socket.gethostname(),
                }
            ),
            encoding="utf-8",
        )
    return launch_path, state_path


@pytest.mark.parametrize(
    (
        "executor_status",
        "executor_action",
        "executor_reason",
        "expected_status",
        "expected_action",
    ),
    [
        (
            "generation_limit_reached",
            "resume_executor",
            "max_generations_per_invocation_reached",
            "resume_ready",
            "resume_executor",
        ),
        (
            "stopped",
            "resume_executor",
            "stop_requested",
            "paused",
            "operator_resume_required",
        ),
        (
            "stopped",
            "stop_training",
            "continuation_policy_stop",
            "completed",
            "stop_training",
        ),
        (
            "output_quarantined",
            "resume_executor",
            "interrupted_output_quarantined",
            "paused",
            "operator_resume_required",
        ),
    ],
)
def test_supervision_decision_honors_automatic_and_manual_boundaries(
    tmp_path: Path,
    executor_status: str,
    executor_action: str,
    executor_reason: str,
    expected_status: str,
    expected_action: str,
) -> None:
    launch_path, _ = _write_launch_artifact(
        tmp_path,
        executor_status=executor_status,
        executor_action=executor_action,
        executor_reason=executor_reason,
    )

    report = supervisor.hf_adapter_continuation_executor_supervision_report(launch_path)

    assert report["status"] == expected_status
    assert report["action"] == expected_action
    assert report["healthy"] is True
    assert report["automatic_resume_allowed"] is (expected_status == "resume_ready")
    if expected_status == "resume_ready":
        assert report["generation_plan_required"] is True
        assert report["generation_plan_status"] == "ready"
        assert str(report["generation_plan_id"]).startswith("sha256:")
        assert report["generation_plan_rebound"] is True


def test_supervision_does_not_mask_failed_latest_launch_with_old_policy_state(
    tmp_path: Path,
) -> None:
    launch_path, _ = _write_launch_artifact(
        tmp_path,
        executor_status="stopped",
        executor_action="stop_training",
        executor_reason="continuation_policy_stop",
        launcher_status="launch_failed",
    )

    report = supervisor.hf_adapter_continuation_executor_supervision_report(launch_path)

    assert report["status"] == "blocked"
    assert report["healthy"] is False
    assert report["issue"] == "launcher_launch_failed"

    malformed_path, _ = _write_launch_artifact(
        tmp_path / "malformed",
        executor_status="stopped",
        executor_action="stop_training",
        executor_reason="continuation_policy_stop",
    )
    malformed = json.loads(malformed_path.read_text(encoding="utf-8"))
    malformed["launches"].append("not-a-launch")
    malformed["launch_count"] = 2
    malformed_path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid launch row"):
        supervisor.hf_adapter_continuation_executor_supervision_report(malformed_path)


def test_supervision_blocks_legacy_unverified_handoff(tmp_path: Path) -> None:
    launch_path, _ = _write_launch_artifact(tmp_path)
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    latest = launch["launches"][-1]
    latest.pop("resume_contract")
    latest.pop("executor_baseline")
    latest.pop("executor_run_id")
    latest.pop("executor_invocation_count")
    launch_path.write_text(json.dumps(launch), encoding="utf-8")

    launcher = st.hf_adapter_continuation_executor_launch_status_report(launch_path)
    report = supervisor.hf_adapter_continuation_executor_supervision_report(launch_path)

    assert launcher["status"] == "completed"
    assert launcher["executor_handoff_observation"] == "legacy_unverified"
    assert report["status"] == "blocked"
    assert report["action"] == "inspect_executor_handoff"
    assert report["issue"] == "executor_handoff_unverified"
    assert report["automatic_resume_allowed"] is False


def test_supervision_waits_for_terminal_executor_lock_to_drain(
    tmp_path: Path,
) -> None:
    launch_path, state_path = _write_launch_artifact(tmp_path)
    lock_path = state_path.parent / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
    lock_path.write_text(
        json.dumps(
            {
                "row_type": "hf_adapter_continuation_executor_lock",
                "lock_id": "terminal-drain-lock",
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
            }
        ),
        encoding="utf-8",
    )
    try:
        report = supervisor.hf_adapter_continuation_executor_supervision_report(
            launch_path
        )
        assert report["status"] == "waiting"
        assert report["action"] == "wait_for_executor_exit"
        assert report["healthy"] is True
        assert report["automatic_resume_allowed"] is False
    finally:
        lock_path.unlink()


@pytest.mark.parametrize(
    ("status", "action", "reason"),
    [
        ("stopped", "resume_executor", "max_generations_per_invocation_reached"),
        ("stopped", "stop_training", "stop_requested"),
    ],
)
def test_supervision_rejects_inconsistent_terminal_contracts(
    tmp_path: Path,
    status: str,
    action: str,
    reason: str,
) -> None:
    launch_path, _ = _write_launch_artifact(
        tmp_path,
        executor_status=status,
        executor_action=action,
        executor_reason=reason,
    )

    report = supervisor.hf_adapter_continuation_executor_supervision_report(launch_path)

    assert report["status"] == "blocked"
    assert report["action"] == "inspect_executor_state"
    assert report["issue"] == "executor_terminal_contract_invalid"
    assert report["automatic_resume_allowed"] is False


def _append_terminal_launch(
    launch_path: Path,
    state_path: Path,
    *,
    action: str,
    reason: str,
) -> dict[str, object]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    source_pending = state.get("pending_generation")
    source_generation_plan_id = (
        source_pending.get("generation_plan_id")
        if isinstance(source_pending, dict)
        else None
    )
    state["invocation_count"] = int(state["invocation_count"]) + 1
    state["status"] = (
        "stopped" if action == "stop_training" else "generation_limit_reached"
    )
    state["action"] = action
    state["reason"] = reason
    if reason == "max_generations_per_invocation_reached":
        state["pending_generation"] = _sealed_pending_generation(
            state_path,
            lineage_depth=int(state["invocation_count"]) + 1,
        )
    else:
        state.pop("pending_generation", None)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    source = launch["launches"][-1]
    latest = dict(source)
    resumed_argv = hf_adapter_executor_launch._generation_plan_rebound_argv(
        source["executor_argv"],
        generation_plan_id=str(source_generation_plan_id),
    )
    latest.update(
        {
            "launch_id": f"supervisor-resumed-{state['invocation_count']}",
            "status": "completed",
            "pid": 99_999_998,
            "executor_baseline": {
                "run_id": state["run_id"],
                "invocation_count": int(state["invocation_count"]) - 1,
            },
            "executor_run_id": state["run_id"],
            "executor_invocation_count": state["invocation_count"],
            "executor_argv": resumed_argv,
            "command": hf_adapter_executor_launch._executor_child_command(resumed_argv),
            "resume_contract": hf_adapter_executor_launch._resume_contract(
                resumed_argv,
                output_root=state_path.parent.resolve(),
                executor_state_path=state_path.resolve(),
                command_cwd=Path(str(source["command_cwd"])),
            ),
            "resumed_from_generation_plan_id": source_generation_plan_id,
        }
    )
    launch["launches"].append(latest)
    launch["latest_launch_id"] = latest["launch_id"]
    launch["launch_count"] = len(launch["launches"])
    launch["status"] = "completed"
    launch_path.write_text(json.dumps(launch), encoding="utf-8")
    return {
        "row_type": "hf_adapter_continuation_executor_resume",
        "ready": True,
        "created": True,
        "status": "completed",
        "source_launch_id": source["launch_id"],
        "resumed_launch_id": latest["launch_id"],
        "executor_invocation_count": int(state["invocation_count"]) - 1,
        "resumed_executor_invocation_count": state["invocation_count"],
        "generation_plan_id": source_generation_plan_id,
        "generation_plan_status": "ready",
        "generation_plan_rebound": True,
    }


def test_supervisor_resumes_once_then_completes_and_persists_transitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch_path, state_path = _write_launch_artifact(tmp_path)

    def fake_resume(
        _path: Path,
        *,
        handoff_timeout_seconds: float,
    ) -> dict[str, object]:
        assert handoff_timeout_seconds == 2.5
        return _append_terminal_launch(
            launch_path,
            state_path,
            action="stop_training",
            reason="continuation_policy_stop",
        )

    monkeypatch.setattr(
        supervisor,
        "resume_hf_adapter_continuation_executor",
        fake_resume,
    )
    report = supervisor.supervise_hf_adapter_continuation_executor(
        launch_path,
        max_resumes=3,
        poll_interval_seconds=0.01,
        timeout_seconds=2.0,
        handoff_timeout_seconds=2.5,
        _sleep=lambda _seconds: None,
    )
    state = supervisor.load_hf_adapter_continuation_executor_supervisor(
        report["supervisor_state_path"]
    )
    latest = state["runs"][-1]

    assert report["status"] == "completed"
    assert report["healthy"] is True
    assert report["total_resumes_started"] == 1
    assert latest["resumes_started"] == 1
    assert [row["status"] for row in latest["transitions"]] == [
        "resume_ready",
        "completed",
    ]
    assert latest["resume_events"][0]["resumed_executor_invocation_count"] == 2
    assert (
        latest["resume_events"][0]["generation_plan_id"]
        == report["latest_run"]["transitions"][0]["generation_plan_id"]
    )
    assert latest["resume_events"][0]["generation_plan_rebound"] is True
    assert not (
        state_path.parent
        / supervisor.HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LOCK_FILENAME
    ).exists()
    status = supervisor.hf_adapter_continuation_executor_supervisor_status_report(
        report["supervisor_state_path"]
    )
    assert status["latest_resume_generation_plan_id"] == latest["resume_events"][0][
        "generation_plan_id"
    ]
    assert (
        f"plan_id={status['latest_resume_generation_plan_id']}"
        in supervisor.hf_adapter_continuation_executor_supervisor_status_lines(status)[
            0
        ]
    )

    corrupt = json.loads(Path(report["supervisor_state_path"]).read_text("utf-8"))
    corrupt["runs"][-1]["resume_events"][0]["generation_plan_id"] = "invalid"
    Path(report["supervisor_state_path"]).write_text(
        json.dumps(corrupt),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="generation plan history is invalid"):
        supervisor.load_hf_adapter_continuation_executor_supervisor(
            report["supervisor_state_path"]
        )


def test_supervisor_budget_timeout_and_live_owner_lock_are_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget_launch, budget_state = _write_launch_artifact(tmp_path / "budget")
    monkeypatch.setattr(
        supervisor,
        "resume_hf_adapter_continuation_executor",
        lambda _path, *, handoff_timeout_seconds: _append_terminal_launch(
            budget_launch,
            budget_state,
            action="resume_executor",
            reason="max_generations_per_invocation_reached",
        ),
    )
    budget = supervisor.supervise_hf_adapter_continuation_executor(
        budget_launch,
        max_resumes=1,
        poll_interval_seconds=0.01,
        timeout_seconds=2.0,
        _sleep=lambda _seconds: None,
    )
    assert budget["status"] == "resume_budget_reached"
    assert budget["healthy"] is True
    assert budget["latest_run"]["resumes_started"] == 1

    timeout_launch, timeout_state = _write_launch_artifact(
        tmp_path / "timeout",
        executor_status="auditing",
        executor_action="audit_chain",
        executor_reason="executor_started",
        launcher_status="handed_off",
        launcher_alive=True,
    )
    times = iter([0.0, 0.0, 0.6])
    timed_out = supervisor.supervise_hf_adapter_continuation_executor(
        timeout_launch,
        max_resumes=1,
        poll_interval_seconds=0.1,
        timeout_seconds=0.5,
        _sleep=lambda _seconds: None,
        _monotonic=lambda: next(times),
    )
    assert timed_out["status"] == "timed_out"
    assert timed_out["healthy"] is False
    (timeout_state.parent / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME).unlink()

    lock_launch, lock_state = _write_launch_artifact(tmp_path / "locked")
    supervisor_lock = (
        lock_state.parent
        / supervisor.HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LOCK_FILENAME
    )
    supervisor_lock.write_text(
        json.dumps(
            {
                "row_type": "hf_adapter_continuation_executor_supervisor_lock",
                "lock_id": "live-supervisor-lock",
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="already owned"):
        supervisor.supervise_hf_adapter_continuation_executor(lock_launch)
    supervisor_lock.unlink()


def test_supervisor_status_and_cooperative_stop_are_run_targeted(
    tmp_path: Path,
) -> None:
    launch_path, executor_state_path = _write_launch_artifact(
        tmp_path,
        executor_status="auditing",
        executor_action="audit_chain",
        executor_reason="executor_started",
        launcher_status="handed_off",
        launcher_alive=True,
    )
    results: list[dict[str, object]] = []
    errors: list[BaseException] = []

    def run_supervisor() -> None:
        try:
            results.append(
                supervisor.supervise_hf_adapter_continuation_executor(
                    launch_path,
                    max_resumes=2,
                    poll_interval_seconds=0.02,
                    timeout_seconds=5.0,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run_supervisor, daemon=True)
    worker.start()
    state_path = (
        executor_state_path.parent
        / supervisor.HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_FILENAME
    )
    deadline = time.monotonic() + 3.0
    while not state_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert state_path.is_file()

    running = supervisor.hf_adapter_continuation_executor_supervisor_status_report(
        state_path
    )
    requested = (
        supervisor.request_hf_adapter_continuation_executor_supervisor_stop(
            state_path,
            reason="test_operator_boundary",
        )
    )
    repeated = supervisor.request_hf_adapter_continuation_executor_supervisor_stop(
        state_path,
        reason="ignored_after_idempotent_create",
    )
    worker.join(timeout=3.0)
    terminal = supervisor.hf_adapter_continuation_executor_supervisor_status_report(
        state_path
    )

    try:
        assert errors == []
        assert not worker.is_alive()
        assert running["status"] == "running"
        assert running["healthy"] is True
        assert running["supervisor_lock_owner_verified"] is True
        assert requested["created"] is True
        assert requested["reason"] == "test_operator_boundary"
        assert repeated["created"] is False
        assert repeated["request_id"] == requested["request_id"]
        assert results[0]["status"] == "stopped"
        assert results[0]["reason"] == "stop_requested"
        assert terminal["status"] == "stopped"
        assert terminal["healthy"] is True
        assert terminal["stop_requested"] is True
        assert terminal["supervisor_lock_owner_verified"] is False
        assert (
            supervisor.hf_adapter_continuation_executor_supervisor_status_lines(
                terminal
            )[0]
            .startswith("hf_adapter_continuation_executor_supervisor_status ")
        )
        assert "created=True" in (
            supervisor.hf_adapter_continuation_executor_supervisor_stop_request_lines(
                requested
            )[0]
        )
        request_path = Path(requested["request_path"])
        original = json.loads(request_path.read_text(encoding="utf-8"))
        malformed = dict(original)
        malformed.pop("reason")
        request_path.write_text(json.dumps(malformed), encoding="utf-8")
        with pytest.raises(ValueError, match="identity is invalid"):
            supervisor.load_hf_adapter_continuation_executor_supervisor_stop_request(
                request_path
            )
        invalid = (
            supervisor.hf_adapter_continuation_executor_supervisor_status_report(
                state_path
            )
        )
        assert invalid["healthy"] is False
        assert "stop_request_invalid" in invalid["health_issues"]

        identity_mismatch = dict(original)
        identity_mismatch["supervisor_pid"] += 1
        request_path.write_text(json.dumps(identity_mismatch), encoding="utf-8")
        mismatched = (
            supervisor.hf_adapter_continuation_executor_supervisor_status_report(
                state_path
            )
        )
        assert mismatched["healthy"] is False
        assert "stop_request_invalid" in mismatched["health_issues"]
        assert "supervisor_pid" in str(mismatched["stop_request_error"])
    finally:
        (
            executor_state_path.parent
            / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
        ).unlink(missing_ok=True)


def test_supervisor_restart_closes_interrupted_run_and_rejects_corrupt_history(
    tmp_path: Path,
) -> None:
    launch_path, state_path = _write_launch_artifact(
        tmp_path / "restart",
        executor_status="stopped",
        executor_action="stop_training",
        executor_reason="continuation_policy_stop",
    )
    first = supervisor.supervise_hf_adapter_continuation_executor(launch_path)
    supervisor_state_path = Path(first["supervisor_state_path"])
    interrupted = json.loads(supervisor_state_path.read_text(encoding="utf-8"))
    interrupted["status"] = "running"
    interrupted["runs"][-1]["status"] = "running"
    supervisor_state_path.write_text(json.dumps(interrupted), encoding="utf-8")
    with pytest.raises(RuntimeError, match="may still be alive"):
        supervisor.supervise_hf_adapter_continuation_executor(launch_path)

    interrupted["runs"][-1]["hostname"] = "remote.example"
    supervisor_state_path.write_text(json.dumps(interrupted), encoding="utf-8")
    with pytest.raises(RuntimeError, match="remote and unverified"):
        supervisor.supervise_hf_adapter_continuation_executor(launch_path)

    interrupted["runs"][-1]["hostname"] = socket.gethostname()
    interrupted["runs"][-1]["pid"] = 99_999_997
    supervisor_state_path.write_text(json.dumps(interrupted), encoding="utf-8")

    stale_lock = (
        state_path.parent
        / supervisor.HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LOCK_FILENAME
    )
    stale_lock.write_text(
        json.dumps(
            {
                "row_type": "hf_adapter_continuation_executor_supervisor_lock",
                "lock_id": "stale-supervisor-lock",
                "pid": 99_999_997,
                "hostname": socket.gethostname(),
            }
        ),
        encoding="utf-8",
    )
    restarted = supervisor.supervise_hf_adapter_continuation_executor(launch_path)
    persisted = supervisor.load_hf_adapter_continuation_executor_supervisor(
        supervisor_state_path
    )

    assert restarted["status"] == "completed"
    assert persisted["invocation_count"] == 2
    assert persisted["runs"][0]["status"] == "interrupted"
    assert persisted["runs"][0]["action"] == "supervisor_restarted"
    assert persisted["runs"][1]["status"] == "completed"
    assert not stale_lock.exists()

    corrupt = json.loads(supervisor_state_path.read_text(encoding="utf-8"))
    corrupt["run_count"] = 99
    supervisor_state_path.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(ValueError, match="run_count is inconsistent"):
        supervisor.supervise_hf_adapter_continuation_executor(launch_path)

    corrupt["run_count"] = len(corrupt["runs"])
    corrupt["total_resumes_started"] = 99
    supervisor_state_path.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(ValueError, match="total resume count is inconsistent"):
        supervisor.supervise_hf_adapter_continuation_executor(launch_path)


def _terminal_executor_command(
    state_path: Path,
    lock_path: Path,
    *,
    invocation_count: int,
) -> list[str]:
    auditing = _executor_state(
        state_path,
        status="auditing",
        action="audit_chain",
        reason="executor_started",
        invocation_count=invocation_count,
    )
    terminal = dict(auditing)
    terminal.update(
        {
            "status": "generation_limit_reached",
            "action": "resume_executor",
            "reason": "max_generations_per_invocation_reached",
            "pending_generation": _sealed_pending_generation(
                state_path,
                lineage_depth=invocation_count + 1,
            ),
        }
    )
    script = """
import json
import os
import socket
import sys
import time
from pathlib import Path

auditing = json.loads(sys.argv[1])
terminal = json.loads(sys.argv[2])
state_path = Path(sys.argv[3])
lock_path = Path(sys.argv[4])

def write_state(payload):
    temporary = state_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload), encoding='utf-8')
    os.replace(temporary, state_path)

descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
    json.dump(
        {
            'row_type': 'hf_adapter_continuation_executor_lock',
            'lock_id': 'supervisor-real-process-lock',
            'pid': os.getpid(),
            'hostname': socket.gethostname(),
        },
        handle,
    )
try:
    write_state(auditing)
    time.sleep(0.15)
    write_state(terminal)
finally:
    lock_path.unlink(missing_ok=True)
"""
    return [
        sys.executable,
        "-S",
        "-u",
        "-c",
        script,
        json.dumps(auditing),
        json.dumps(terminal),
        str(state_path),
        str(lock_path),
    ]


def _terminate_process_group(pid: object, lock_path: Path) -> None:
    if (
        not isinstance(pid, int)
        or hf_adapter_executor_launch.local_pid_alive(pid) is not True
    ):
        return
    try:
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return
    if owner.get("pid") != pid or owner.get("hostname") != socket.gethostname():
        return
    if os.name == "posix":
        os.killpg(pid, signal.SIGTERM)
    else:
        os.kill(pid, signal.SIGTERM)


def test_supervisor_drives_a_real_detached_relaunch_to_next_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch_path, state_path = _write_launch_artifact(tmp_path)
    lock_path = state_path.parent / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
    monkeypatch.setattr(
        hf_adapter_executor_launch,
        "_executor_child_command",
        lambda _argv: _terminal_executor_command(
            state_path,
            lock_path,
            invocation_count=2,
        ),
    )

    latest_pid: object = None
    try:
        report = supervisor.supervise_hf_adapter_continuation_executor(
            launch_path,
            max_resumes=1,
            poll_interval_seconds=0.02,
            timeout_seconds=5.0,
            handoff_timeout_seconds=2.0,
        )
        launch = st.load_hf_adapter_continuation_executor_launch(launch_path)
        latest_pid = launch["launches"][-1].get("pid")
        executor = st.load_hf_adapter_continuation_executor(state_path)

        assert report["status"] == "resume_budget_reached"
        assert report["latest_run"]["resumes_started"] == 1
        assert (
            report["latest_run"]["resume_events"][0][
                "resumed_executor_invocation_count"
            ]
            == 2
        )
        assert launch["launch_count"] == 2
        assert executor["invocation_count"] == 2
        assert executor["status"] == "generation_limit_reached"
    finally:
        _terminate_process_group(latest_pid, lock_path)


def test_supervisor_validation_and_lines(tmp_path: Path) -> None:
    launch_path, _ = _write_launch_artifact(
        tmp_path,
        executor_status="stopped",
        executor_action="stop_training",
        executor_reason="continuation_policy_stop",
    )
    with pytest.raises(ValueError, match="max_resumes"):
        supervisor.supervise_hf_adapter_continuation_executor(
            launch_path,
            max_resumes=0,
        )
    with pytest.raises(ValueError, match="poll_interval_seconds"):
        supervisor.supervise_hf_adapter_continuation_executor(
            launch_path,
            poll_interval_seconds=0.0,
        )
    state_target = tmp_path / "supervisor-state-target.json"
    state_link = tmp_path / "supervisor-state-link.json"
    state_link.symlink_to(state_target)
    with pytest.raises(ValueError, match="cannot be a symlink"):
        supervisor.supervise_hf_adapter_continuation_executor(
            launch_path,
            supervisor_state_path=state_link,
        )
    assert not state_target.exists()

    decision = supervisor.hf_adapter_continuation_executor_supervision_report(
        launch_path
    )
    report = supervisor.supervise_hf_adapter_continuation_executor(launch_path)
    assert (
        "status=completed"
        in supervisor.hf_adapter_continuation_executor_supervision_lines(decision)[0]
    )
    assert (
        "status=completed"
        in supervisor.hf_adapter_continuation_executor_supervisor_lines(report)[0]
    )


def test_supervisor_cli_and_public_exports(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_launch, _ = _write_launch_artifact(tmp_path / "plan")
    plan_code = hf_cli.adapter_continuation_executor_supervise_main(
        [str(plan_launch), "--plan"]
    )
    plan_output = capsys.readouterr().out

    execute_launch, _ = _write_launch_artifact(
        tmp_path / "execute",
        executor_status="stopped",
        executor_action="stop_training",
        executor_reason="continuation_policy_stop",
    )
    custom_state = tmp_path / "custom-supervisor.json"
    execute_code = hf_cli.adapter_continuation_executor_supervise_main(
        [
            str(execute_launch),
            "--max-resumes",
            "3",
            "--poll-interval-seconds",
            "0.1",
            "--timeout-seconds",
            "4",
            "--handoff-timeout-seconds",
            "2",
            "--state",
            str(custom_state),
        ]
    )
    execute_output = capsys.readouterr().out
    persisted = st.load_hf_adapter_continuation_executor_supervisor(custom_state)

    assert plan_code == 0
    assert "status=resume_ready" in plan_output
    assert execute_code == 0
    assert "status=completed" in execute_output
    assert persisted["runs"][-1]["max_resumes"] == 3
    for name in (
        "hf_adapter_executor_supervisor",
        "hf_adapter_executor_supervisor_launch",
        "hf_adapter_continuation_executor_supervision_report",
        "hf_adapter_continuation_executor_supervision_lines",
        "supervise_hf_adapter_continuation_executor",
        "load_hf_adapter_continuation_executor_supervisor",
        "hf_adapter_continuation_executor_supervisor_lines",
        "hf_adapter_continuation_executor_supervisor_status_report",
        "hf_adapter_continuation_executor_supervisor_status_lines",
        "request_hf_adapter_continuation_executor_supervisor_stop",
        "load_hf_adapter_continuation_executor_supervisor_stop_request",
        "launch_hf_adapter_continuation_executor_supervisor",
        "load_hf_adapter_continuation_executor_supervisor_launch",
        "hf_adapter_continuation_executor_supervisor_launch_status_report",
        "HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_SCHEMA",
        "HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_LAUNCH_SCHEMA",
        "HF_ADAPTER_CONTINUATION_EXECUTOR_SUPERVISOR_STOP_REQUEST_SCHEMA",
    ):
        assert name in st.__all__
