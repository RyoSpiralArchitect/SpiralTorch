from __future__ import annotations

import json
import os
import socket
import stat
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
    ):
        assert name in st.__all__
