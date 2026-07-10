from __future__ import annotations

import json
import os
import signal
import socket
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
import spiraltorch as st
from spiraltorch import hf_adapter_executor_recovery, hf_cli


def _write_cancelled_state(
    tmp_path: Path,
    *,
    attempt_id: str = "generation-attempt-recovery",
) -> tuple[Path, Path, Path]:
    output_root = tmp_path / "executor"
    output_root.mkdir()
    output_dir = output_root / "generation-007"
    nested = output_dir / "checkpoint-3"
    nested.mkdir(parents=True)
    (output_dir / "partial.txt").write_text("partial", encoding="utf-8")
    (nested / "weights.bin").write_bytes(b"weights")
    log_path = output_root / "executor-logs" / "generation-007.log"
    log_path.parent.mkdir()
    log_path.write_text("cancelled", encoding="utf-8")
    state_path = output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_FILENAME
    now = datetime.now(timezone.utc).isoformat()
    state = {
        "row_type": "hf_adapter_continuation_executor",
        "schema": st.HF_ADAPTER_CONTINUATION_EXECUTOR_SCHEMA,
        "status": "stopped",
        "action": "resume_executor",
        "reason": "stop_requested",
        "created_at": now,
        "updated_at": now,
        "run_id": "recovery-test-run",
        "source_paths": [],
        "output_root": str(output_root.resolve()),
        "state_path": str(state_path.resolve()),
        "invocation_count": 1,
        "generation_attempt_count": 1,
        "promoted_generation_count": 0,
        "selected_lineage_depth": 6,
        "generations": [
            {
                "attempt_id": attempt_id,
                "status": "cancelled",
                "runner_kind": "subprocess",
                "process_group_isolated": True,
                "stop_scope": "process_group",
                "hostname": socket.gethostname(),
                "pid": 99_999_999,
                "lineage_depth": 7,
                "output_dir": str(output_dir.resolve()),
                "log_path": str(log_path.resolve()),
                "returncode": -15,
            }
        ],
        "execution": {
            "lock_path": str(
                output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
            )
        },
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return state_path, output_root, output_dir


def _write_lock(path: Path, *, pid: int) -> None:
    path.write_text(
        json.dumps(
            {
                "row_type": "hf_adapter_continuation_executor_lock",
                "lock_id": "recovery-lock",
                "pid": pid,
                "hostname": socket.gethostname(),
            }
        ),
        encoding="utf-8",
    )


def _mark_attempt_running(
    state_path: Path,
    *,
    pid: int,
    hostname: str | None = None,
    runner_kind: str = "subprocess",
    process_group_isolated: bool = True,
    stop_scope: str = "process_group",
) -> None:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "running"
    state["action"] = "run_generation"
    state["reason"] = "executor_process_interrupted"
    attempt = state["generations"][0]
    attempt["status"] = "running"
    attempt["runner_kind"] = runner_kind
    attempt["process_group_isolated"] = process_group_isolated
    attempt["stop_scope"] = stop_scope
    attempt["hostname"] = hostname or socket.gethostname()
    attempt["pid"] = pid
    attempt.pop("returncode", None)
    state["pending_generation"] = dict(attempt)
    state_path.write_text(json.dumps(state), encoding="utf-8")


def _write_launch_state(
    path: Path,
    *,
    output_root: Path,
    executor_state_path: Path,
    log_path: Path,
) -> Path:
    now = datetime.now(timezone.utc).isoformat()
    path.write_text(
        json.dumps(
            {
                "row_type": "hf_adapter_continuation_executor_launches",
                "schema": st.HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_SCHEMA,
                "created_at": now,
                "updated_at": now,
                "status": "handed_off",
                "output_root": str(output_root.resolve()),
                "executor_state_path": str(executor_state_path.resolve()),
                "launch_state_path": str(path.resolve()),
                "launch_count": 1,
                "launches": [
                    {
                        "launch_id": "recovery-launch",
                        "status": "handed_off",
                        "hostname": socket.gethostname(),
                        "pid": 99_999_999,
                        "log_path": str(log_path.resolve()),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_quarantine_is_atomic_audited_idempotent_and_recoverable(
    tmp_path: Path,
) -> None:
    state_path, output_root, output_dir = _write_cancelled_state(tmp_path)
    plan = st.hf_adapter_continuation_executor_output_quarantine_report(
        state_path,
        attempt_id="generation-attempt-recovery",
    )

    assert plan["status"] == "quarantine_ready"
    assert plan["ready"] is True
    assert plan["source_snapshot"]["file_count"] == 2
    assert plan["source_snapshot"]["total_file_bytes"] == 14
    destination = Path(plan["destination_path"])
    quarantine_root = Path(plan["quarantine_root"])
    assert quarantine_root.parent == output_root.parent
    assert output_root not in quarantine_root.parents

    resolution = st.quarantine_hf_adapter_continuation_executor_output(
        state_path,
        attempt_id="generation-attempt-recovery",
        reason="integration recovery",
    )
    loaded = st.load_hf_adapter_continuation_executor(state_path)
    status_report = st.hf_adapter_continuation_executor_status_report(state_path)
    repeated = st.quarantine_hf_adapter_continuation_executor_output(
        state_path,
        attempt_id="generation-attempt-recovery",
        reason="ignored idempotent reason",
    )

    assert resolution["created"] is True
    assert resolution["reason"] == "integration recovery"
    assert (
        resolution["tree_snapshot"]["metadata_sha256"]
        == plan["source_snapshot"]["metadata_sha256"]
    )
    assert output_dir.exists() is False
    assert destination.joinpath("partial.txt").read_text(encoding="utf-8") == "partial"
    assert (
        destination.joinpath("checkpoint-3", "weights.bin").read_bytes() == b"weights"
    )
    assert loaded["status"] == "output_quarantined"
    assert loaded["action"] == "resume_executor"
    assert (
        loaded["last_output_resolution"]["resolution_id"] == resolution["resolution_id"]
    )
    assert loaded["generations"][0]["status"] == "cancelled"
    assert loaded["generations"][0]["output_resolution"]["destination_path"] == str(
        destination
    )
    assert len(loaded["output_resolution_history"]) == 1
    assert status_report["status"] == "output_quarantined"
    assert status_report["healthy"] is True
    assert status_report["recommended_action"] == "resume_executor"
    assert status_report["health_issues"] == []
    assert repeated["created"] is False
    assert repeated["resolution_id"] == resolution["resolution_id"]
    assert repeated["reason"] == "integration recovery"
    if os.name != "nt":
        assert stat.S_IMODE(quarantine_root.stat().st_mode) == 0o700

    destination.joinpath("unexpected.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no longer matches"):
        st.hf_adapter_continuation_executor_output_quarantine_report(
            state_path,
            attempt_id="generation-attempt-recovery",
        )
    destination.joinpath("unexpected.txt").unlink()

    launcher_log = tmp_path / "launcher.log"
    launcher_log.write_text("launch", encoding="utf-8")
    launch_state = _write_launch_state(
        tmp_path / "launch.json",
        output_root=output_root,
        executor_state_path=state_path,
        log_path=launcher_log,
    )
    launch_status = st.hf_adapter_continuation_executor_launch_status_report(
        launch_state
    )
    assert launch_status["status"] == "recoverable"
    assert launch_status["healthy"] is True
    assert launch_status["recommended_action"] == "resume_executor"

    output_dir.mkdir()
    output_dir.joinpath("promoted.txt").write_text("new generation", encoding="utf-8")
    resumed = json.loads(state_path.read_text(encoding="utf-8"))
    resumed["generations"].append(
        {
            "attempt_id": "later-promoted-attempt",
            "status": "promoted",
            "lineage_depth": 7,
            "output_dir": str(output_dir.resolve()),
        }
    )
    state_path.write_text(json.dumps(resumed), encoding="utf-8")
    reused = st.quarantine_hf_adapter_continuation_executor_output(
        state_path,
        attempt_id="generation-attempt-recovery",
        reason="must not affect later promotion",
    )

    assert reused["created"] is False
    assert reused["resolution_id"] == resolution["resolution_id"]
    assert output_dir.joinpath("promoted.txt").read_text(encoding="utf-8") == (
        "new generation"
    )


def test_quarantine_fails_closed_on_live_lock_and_reaps_stale_lock(
    tmp_path: Path,
) -> None:
    state_path, output_root, output_dir = _write_cancelled_state(tmp_path)
    lock_path = output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
    _write_lock(lock_path, pid=os.getpid())

    plan = st.hf_adapter_continuation_executor_output_quarantine_report(
        state_path,
        attempt_id="generation-attempt-recovery",
    )
    with pytest.raises(RuntimeError, match="locked"):
        st.quarantine_hf_adapter_continuation_executor_output(
            state_path,
            attempt_id="generation-attempt-recovery",
        )

    assert plan["status"] == "executor_locked"
    assert plan["ready"] is False
    assert output_dir.is_dir()
    lock_path.write_text(
        json.dumps(
            {
                "row_type": "invalid_executor_lock",
                "lock_id": "invalid-stale-owner",
                "pid": 99_999_999,
                "hostname": socket.gethostname(),
            }
        ),
        encoding="utf-8",
    )
    invalid_plan = st.hf_adapter_continuation_executor_output_quarantine_report(
        state_path,
        attempt_id="generation-attempt-recovery",
    )
    assert invalid_plan["status"] == "executor_locked"
    assert invalid_plan["executor_lock"]["status"] == "unverified"
    lock_path.unlink()
    _write_lock(lock_path, pid=99_999_999)

    recovered = st.quarantine_hf_adapter_continuation_executor_output(
        state_path,
        attempt_id="generation-attempt-recovery",
    )
    assert recovered["created"] is True
    assert not lock_path.exists()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_quarantine_claims_interrupted_subprocess_and_restores_health(
    tmp_path: Path,
) -> None:
    state_path, output_root, output_dir = _write_cancelled_state(tmp_path)
    _mark_attempt_running(state_path, pid=99_999_999)
    launcher_log = tmp_path / "interrupted-launcher.log"
    launcher_log.write_text("interrupted", encoding="utf-8")
    launch_state = _write_launch_state(
        tmp_path / "interrupted-launch.json",
        output_root=output_root,
        executor_state_path=state_path,
        log_path=launcher_log,
    )

    status_before = st.hf_adapter_continuation_executor_status_report(state_path)
    launch_before = st.hf_adapter_continuation_executor_launch_status_report(
        launch_state
    )
    plan = st.hf_adapter_continuation_executor_output_quarantine_report(
        state_path,
        attempt_id="generation-attempt-recovery",
    )
    resolution = st.quarantine_hf_adapter_continuation_executor_output(
        state_path,
        attempt_id="generation-attempt-recovery",
        reason="verified interrupted subprocess",
    )
    loaded = st.load_hf_adapter_continuation_executor(state_path)
    status_after = st.hf_adapter_continuation_executor_status_report(state_path)
    launch_after = st.hf_adapter_continuation_executor_launch_status_report(
        launch_state
    )
    attempt = loaded["generations"][0]

    assert status_before["status"] == "interrupted"
    assert status_before["recommended_action"] == "quarantine_interrupted_output"
    assert status_before["interruption_claim"]["ready"] is True
    assert launch_before["status"] == "executor_interrupted"
    assert launch_before["recommended_action"] == "quarantine_interrupted_output"
    assert plan["status"] == "quarantine_ready"
    assert plan["interruption_claim"]["pid_alive_observed"] is False
    assert plan["interruption_claim"]["process_group_alive_observed"] is False
    assert resolution["created"] is True
    assert resolution["attempt_status"] == "interrupted"
    assert resolution["interruption_claim"]["schema"] == (
        st.HF_ADAPTER_CONTINUATION_EXECUTOR_INTERRUPTION_CLAIM_SCHEMA
    )
    assert attempt["status"] == "interrupted"
    assert attempt["status_before_interruption"] == "running"
    assert attempt["interruption_claim"] == resolution["interruption_claim"]
    assert "pending_generation" not in loaded
    assert loaded["reason"] == "interrupted_generation_output_quarantined"
    assert not output_dir.exists()
    assert status_after["status"] == "output_quarantined"
    assert status_after["healthy"] is True
    assert status_after["recommended_action"] == "resume_executor"
    assert status_after["interruption_claim"] == resolution["interruption_claim"]
    assert launch_after["status"] == "recoverable"
    assert launch_after["recommended_action"] == "resume_executor"


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_quarantine_rejects_claim_that_no_longer_matches_attempt(
    tmp_path: Path,
) -> None:
    state_path, _, _ = _write_cancelled_state(tmp_path)
    _mark_attempt_running(state_path, pid=99_999_999)
    st.quarantine_hf_adapter_continuation_executor_output(
        state_path,
        attempt_id="generation-attempt-recovery",
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["generations"][0]["pid"] = 100_000_000
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(RuntimeError, match="interruption claim is invalid"):
        st.hf_adapter_continuation_executor_output_quarantine_report(
            state_path,
            attempt_id="generation-attempt-recovery",
        )


def test_quarantine_refuses_live_interrupted_attempt(tmp_path: Path) -> None:
    state_path, _, output_dir = _write_cancelled_state(tmp_path)
    _mark_attempt_running(state_path, pid=os.getpid())

    plan = st.hf_adapter_continuation_executor_output_quarantine_report(
        state_path,
        attempt_id="generation-attempt-recovery",
    )
    with pytest.raises(RuntimeError, match="attempt_process_alive"):
        st.quarantine_hf_adapter_continuation_executor_output(
            state_path,
            attempt_id="generation-attempt-recovery",
        )

    assert plan["status"] == "interruption_unverified"
    assert plan["ready"] is False
    assert plan["action"] == "wait_for_process_exit"
    assert plan["interruption_claim"]["issue"] == "attempt_process_alive"
    assert output_dir.is_dir()


def test_quarantine_treats_out_of_range_pid_as_unverified(tmp_path: Path) -> None:
    state_path, _, output_dir = _write_cancelled_state(tmp_path)
    _mark_attempt_running(state_path, pid=10**100)

    plan = st.hf_adapter_continuation_executor_output_quarantine_report(
        state_path,
        attempt_id="generation-attempt-recovery",
    )

    assert plan["status"] == "interruption_unverified"
    assert plan["ready"] is False
    assert plan["action"] == "inspect_unverified_process"
    assert plan["interruption_claim"]["issue"] == (
        "attempt_process_liveness_unverified"
    )
    assert output_dir.is_dir()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_interrupted_status_waits_for_independent_live_executor_lock(
    tmp_path: Path,
) -> None:
    state_path, output_root, output_dir = _write_cancelled_state(tmp_path)
    _mark_attempt_running(state_path, pid=99_999_999)
    lock_path = output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
    _write_lock(lock_path, pid=os.getpid())

    status = st.hf_adapter_continuation_executor_status_report(state_path)
    plan = st.hf_adapter_continuation_executor_output_quarantine_report(
        state_path,
        attempt_id="generation-attempt-recovery",
    )

    assert status["status"] == "interrupted"
    assert status["interruption_claim"]["ready"] is True
    assert status["interruption_lock_ready"] is False
    assert "interrupted_output_lock_unavailable" in status["health_issues"]
    assert status["recommended_action"] == "wait_for_executor_exit"
    assert plan["status"] == "executor_locked"
    assert plan["action"] == "wait_for_executor_exit"
    assert output_dir.is_dir()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_quarantine_refuses_surviving_interrupted_process_group(
    tmp_path: Path,
) -> None:
    child_script = "import time; time.sleep(60)"
    leader_script = (
        "import subprocess, sys; "
        f"child=subprocess.Popen([sys.executable, '-c', {child_script!r}], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        "print(child.pid, flush=True)"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", leader_script],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert leader.stdout is not None
    child_pid = int(leader.stdout.readline().strip())
    leader.wait(timeout=10)
    state_path, _, output_dir = _write_cancelled_state(tmp_path)
    _mark_attempt_running(state_path, pid=leader.pid)
    try:
        plan = st.hf_adapter_continuation_executor_output_quarantine_report(
            state_path,
            attempt_id="generation-attempt-recovery",
        )
        with pytest.raises(RuntimeError, match="attempt_process_group_alive"):
            st.quarantine_hf_adapter_continuation_executor_output(
                state_path,
                attempt_id="generation-attempt-recovery",
            )

        assert plan["status"] == "interruption_unverified"
        assert plan["action"] == "wait_for_process_group_exit"
        assert plan["interruption_claim"]["pid"] == leader.pid
        assert plan["interruption_claim"]["pid_alive_observed"] is False
        assert plan["interruption_claim"]["process_group_alive_observed"] is True
        assert output_dir.is_dir()
    finally:
        try:
            os.killpg(leader.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)


def test_quarantine_rejects_wrong_attempt_claimed_output_and_symlink(
    tmp_path: Path,
) -> None:
    state_path, _, output_dir = _write_cancelled_state(tmp_path)
    with pytest.raises(ValueError, match="was not found"):
        st.hf_adapter_continuation_executor_output_quarantine_report(
            state_path,
            attempt_id="wrong-attempt",
        )

    plan = st.hf_adapter_continuation_executor_output_quarantine_report(
        state_path,
        attempt_id="generation-attempt-recovery",
    )
    quarantine_root = Path(plan["quarantine_root"])
    destination = Path(plan["destination_path"])
    quarantine_root.mkdir(mode=0o700)
    output_dir.replace(destination)
    with pytest.raises(RuntimeError, match="no matching durable intent"):
        st.hf_adapter_continuation_executor_output_quarantine_report(
            state_path,
            attempt_id="generation-attempt-recovery",
        )

    destination.replace(output_dir)
    quarantine_root.rmdir()

    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["generations"].append(
        {
            "attempt_id": "later-promotion",
            "status": "promoted",
            "output_dir": str(output_dir.resolve()),
        }
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="claimed by a later promotion"):
        st.hf_adapter_continuation_executor_output_quarantine_report(
            state_path,
            attempt_id="generation-attempt-recovery",
        )

    state["generations"].pop()
    state_path.write_text(json.dumps(state), encoding="utf-8")
    external = tmp_path / "external"
    external.mkdir()
    for child in sorted(output_dir.rglob("*"), reverse=True):
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    output_dir.rmdir()
    output_dir.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="cannot be a symlink"):
        st.hf_adapter_continuation_executor_output_quarantine_report(
            state_path,
            attempt_id="generation-attempt-recovery",
        )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation is POSIX-only")
def test_quarantine_rejects_special_filesystem_entries(tmp_path: Path) -> None:
    state_path, _, output_dir = _write_cancelled_state(tmp_path)
    os.mkfifo(output_dir / "training-stream")

    with pytest.raises(RuntimeError, match="special filesystem entries"):
        st.hf_adapter_continuation_executor_output_quarantine_report(
            state_path,
            attempt_id="generation-attempt-recovery",
        )


def test_quarantine_rejects_state_artifact_inside_attempt_output(
    tmp_path: Path,
) -> None:
    state_path, _, output_dir = _write_cancelled_state(tmp_path)
    nested_state = output_dir / "nested-state.json"
    nested_state.write_text(state_path.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ValueError, match="state_path cannot be inside"):
        st.hf_adapter_continuation_executor_output_quarantine_report(
            nested_state,
            attempt_id="generation-attempt-recovery",
        )


def test_quarantine_adopts_move_left_by_interrupted_state_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path, _, output_dir = _write_cancelled_state(tmp_path)
    real_write_state = hf_adapter_executor_recovery._write_state
    write_count = 0

    def interrupt_final_write(path: Path, state: dict[str, object]) -> None:
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise RuntimeError("simulated final state write interruption")
        real_write_state(path, state)

    monkeypatch.setattr(
        hf_adapter_executor_recovery,
        "_write_state",
        interrupt_final_write,
    )
    with pytest.raises(RuntimeError, match="simulated final state write"):
        st.quarantine_hf_adapter_continuation_executor_output(
            state_path,
            attempt_id="generation-attempt-recovery",
            reason="original interrupted move",
        )
    assert not output_dir.exists()
    persisted = st.load_hf_adapter_continuation_executor(state_path)
    destination = Path(persisted["pending_output_resolution"]["destination_path"])
    assert destination.is_dir()
    pending_status = st.hf_adapter_continuation_executor_status_report(state_path)
    assert pending_status["healthy"] is False
    assert "output_quarantine_incomplete" in pending_status["health_issues"]
    assert pending_status["recommended_action"] == "complete_output_quarantine"

    monkeypatch.setattr(
        hf_adapter_executor_recovery,
        "_write_state",
        real_write_state,
    )

    adopted_plan = st.hf_adapter_continuation_executor_output_quarantine_report(
        state_path,
        attempt_id="generation-attempt-recovery",
    )
    resolution = st.quarantine_hf_adapter_continuation_executor_output(
        state_path,
        attempt_id="generation-attempt-recovery",
        reason="adopt interrupted move",
    )

    assert adopted_plan["status"] == "quarantine_adoption_ready"
    assert adopted_plan["ready"] is True
    assert resolution["created"] is True
    assert resolution["adopted_after_interrupted_write"] is True
    assert resolution["reason"] == "original interrupted move"
    assert resolution["tree_snapshot"]["file_count"] == 2


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_interrupted_quarantine_adopts_move_after_final_write_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path, _, output_dir = _write_cancelled_state(tmp_path)
    _mark_attempt_running(state_path, pid=99_999_999)
    real_write_state = hf_adapter_executor_recovery._write_state
    write_count = 0

    def interrupt_final_write(path: Path, state: dict[str, object]) -> None:
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise RuntimeError("simulated interrupted final state write")
        real_write_state(path, state)

    monkeypatch.setattr(
        hf_adapter_executor_recovery,
        "_write_state",
        interrupt_final_write,
    )
    with pytest.raises(RuntimeError, match="interrupted final state write"):
        st.quarantine_hf_adapter_continuation_executor_output(
            state_path,
            attempt_id="generation-attempt-recovery",
            reason="interrupted claim adoption",
        )
    persisted = st.load_hf_adapter_continuation_executor(state_path)
    attempt = persisted["generations"][0]
    destination = Path(persisted["pending_output_resolution"]["destination_path"])

    assert not output_dir.exists()
    assert destination.is_dir()
    assert attempt["status"] == "interrupted"
    assert attempt["interruption_claim"]["ready"] is True
    assert (
        persisted["pending_output_resolution"]["interruption_claim"]
        == (attempt["interruption_claim"])
    )

    monkeypatch.setattr(hf_adapter_executor_recovery, "_write_state", real_write_state)
    plan = st.hf_adapter_continuation_executor_output_quarantine_report(
        state_path,
        attempt_id="generation-attempt-recovery",
    )
    resolution = st.quarantine_hf_adapter_continuation_executor_output(
        state_path,
        attempt_id="generation-attempt-recovery",
        reason="ignored adoption reason",
    )

    assert plan["status"] == "quarantine_adoption_ready"
    assert resolution["created"] is True
    assert resolution["adopted_after_interrupted_write"] is True
    assert resolution["reason"] == "interrupted claim adoption"
    assert resolution["interruption_claim"] == attempt["interruption_claim"]


def test_quarantine_cli_supports_plan_execute_and_health_gate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state_path, _, _ = _write_cancelled_state(tmp_path)
    attempt_id = "generation-attempt-recovery"
    plan_code = hf_cli.adapter_continuation_executor_quarantine_main(
        [str(state_path), "--attempt-id", attempt_id, "--plan"]
    )
    plan_output = capsys.readouterr().out
    execute_code = hf_cli.adapter_continuation_executor_quarantine_main(
        [
            str(state_path),
            "--attempt-id",
            attempt_id,
            "--reason",
            "cli recovery",
        ]
    )
    execute_output = capsys.readouterr().out
    status_code = hf_cli.adapter_continuation_executor_status_main(
        [str(state_path), "--require-healthy"]
    )
    status_output = capsys.readouterr().out

    assert plan_code == 0
    assert "status=quarantine_ready" in plan_output
    assert "ready=True" in plan_output
    assert execute_code == 0
    assert "created=True" in execute_output
    assert status_code == 0
    assert "status=output_quarantined" in status_output
    for name in (
        "hf_adapter_executor_recovery",
        "hf_adapter_continuation_executor_output_quarantine_report",
        "hf_adapter_continuation_executor_output_resolution_lines",
        "quarantine_hf_adapter_continuation_executor_output",
        "HF_ADAPTER_CONTINUATION_EXECUTOR_OUTPUT_RESOLUTION_SCHEMA",
        "HF_ADAPTER_CONTINUATION_EXECUTOR_INTERRUPTION_CLAIM_SCHEMA",
    ):
        assert name in st.__all__
