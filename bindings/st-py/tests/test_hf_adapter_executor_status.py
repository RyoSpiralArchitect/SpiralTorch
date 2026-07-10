from __future__ import annotations

import json
import os
import socket
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

import spiraltorch as st
from spiraltorch import hf_adapter_executor, hf_cli


def _write_running_state(
    path: Path,
    *,
    pid: int | None,
    hostname: str,
    output_dir: Path,
    log_path: Path,
) -> Path:
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "row_type": "hf_adapter_continuation_executor",
        "schema": st.HF_ADAPTER_CONTINUATION_EXECUTOR_SCHEMA,
        "status": "running",
        "action": "run_generation",
        "created_at": now,
        "updated_at": now,
        "run_id": "status-test",
        "source_paths": [],
        "output_root": str(path.parent),
        "state_path": str(path),
        "invocation_count": 1,
        "generation_attempt_count": 1,
        "promoted_generation_count": 0,
        "selected_lineage_depth": 1,
        "generations": [
            {
                "attempt_id": "status-attempt",
                "status": "running",
                "runner_kind": "subprocess",
                "hostname": hostname,
                "pid": pid,
                "started_at": now,
                "process_started_at": now,
                "lineage_depth": 2,
                "output_dir": str(output_dir),
                "log_path": str(log_path),
                "command_cwd": str(path.parent),
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_subprocess_runner_persists_combined_log_and_reports_pid(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "executor.log"
    started_pids: list[int] = []
    observed_log_bytes: list[int] = []
    live_reports: list[dict[str, object]] = []
    output_dir = tmp_path / "generation-002"
    output_dir.mkdir()

    def process_started(pid: int) -> None:
        os.kill(pid, 0)
        started_pids.append(pid)
        state_path = _write_running_state(
            tmp_path / "live-state.json",
            pid=pid,
            hostname=socket.gethostname(),
            output_dir=output_dir,
            log_path=log_path,
        )
        live_reports.append(
            st.hf_adapter_continuation_executor_status_report(state_path)
        )

    returncode = hf_adapter_executor._execute_command(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys; print('stdout-line'); print('stderr-line', file=sys.stderr)",
        ],
        command_runner=None,
        command_cwd=tmp_path,
        command_env={"SPIRALTORCH_EXECUTOR_TEST": "1"},
        log_path=log_path,
        tee_output=False,
        process_started=process_started,
        process_progress=observed_log_bytes.append,
    )
    log = log_path.read_text(encoding="utf-8")

    assert returncode == 0
    assert len(started_pids) == 1
    assert observed_log_bytes and observed_log_bytes[-1] > len("stdout-line")
    assert live_reports[0]["status"] == "running"
    assert live_reports[0]["pid_alive_observed"] is True
    assert live_reports[0]["healthy"] is False
    assert "single_writer_lock_missing" in live_reports[0]["health_issues"]
    assert "[spiraltorch-executor] started_at=" in log
    assert "stdout-line" in log
    assert "stderr-line" in log
    assert "returncode=0" in log
    if os.name != "nt":
        assert stat.S_IMODE(log_path.stat().st_mode) == 0o600


def test_status_reports_live_local_process_and_artifacts(
    tmp_path: Path,
    capsys,
) -> None:
    output_dir = tmp_path / "generation-002"
    output_dir.mkdir()
    log_path = tmp_path / "executor.log"
    log_path.write_text("live-log", encoding="utf-8")
    lock_path = tmp_path / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
    lock_path.write_text("{}", encoding="utf-8")
    state_path = _write_running_state(
        tmp_path / "state.json",
        pid=os.getpid(),
        hostname=socket.gethostname(),
        output_dir=output_dir,
        log_path=log_path,
    )

    report = st.hf_adapter_continuation_executor_status_report(state_path)
    code = hf_cli.adapter_continuation_executor_status_main(
        [str(state_path), "--require-healthy"]
    )
    output = capsys.readouterr().out

    assert report["status"] == "running"
    assert report["healthy"] is True
    assert report["same_host"] is True
    assert report["pid_alive_observed"] is True
    assert report["process_identity_verified"] is False
    assert report["output"]["kind_ready"] is True
    assert report["log"]["kind_ready"] is True
    assert report["log"]["size_bytes"] == len("live-log")
    assert report["lock"]["exists"] is True
    assert code == 0
    assert "status=running" in output
    assert "pid_alive=True" in output


def test_status_fails_health_gate_for_interrupted_or_remote_process(
    tmp_path: Path,
    capsys,
) -> None:
    state_path = _write_running_state(
        tmp_path / "state.json",
        pid=99_999_999,
        hostname=socket.gethostname(),
        output_dir=tmp_path / "generation-002",
        log_path=tmp_path / "executor.log",
    )

    interrupted = st.hf_adapter_continuation_executor_status_report(state_path)
    interrupted_code = hf_cli.adapter_continuation_executor_status_main(
        [str(state_path), "--require-healthy"]
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["generations"][0]["hostname"] = "remote.invalid"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    remote = st.hf_adapter_continuation_executor_status_report(state_path)
    remote_code = hf_cli.adapter_continuation_executor_status_main(
        [str(state_path), "--require-healthy"]
    )
    capsys.readouterr()

    assert interrupted["status"] == "interrupted"
    assert interrupted["healthy"] is False
    assert interrupted["recommended_action"] == "recover_interrupted_generation"
    assert interrupted_code == 1
    assert remote["status"] == "remote_running"
    assert remote["healthy"] is False
    assert remote["pid_alive_observed"] is None
    assert remote_code == 1


def test_status_uses_latest_attempt_for_completed_executor(tmp_path: Path) -> None:
    output_dir = tmp_path / "generation-002"
    output_dir.mkdir()
    log_path = tmp_path / "executor.log"
    log_path.write_text("complete", encoding="utf-8")
    state_path = _write_running_state(
        tmp_path / "state.json",
        pid=99_999_999,
        hostname=socket.gethostname(),
        output_dir=output_dir,
        log_path=log_path,
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "generation_limit_reached"
    state["generations"][0]["status"] = "promoted"
    state["generations"][0]["returncode"] = 0
    state["promoted_generation_count"] = 1
    state_path.write_text(json.dumps(state), encoding="utf-8")

    report = st.hf_adapter_continuation_executor_status_report(state_path)

    assert report["status"] == "generation_limit_reached"
    assert report["healthy"] is True
    assert report["active_attempt"] is None
    assert report["latest_attempt"]["status"] == "promoted"
    assert report["log"]["size_bytes"] == len("complete")
    assert "hf_adapter_executor_status" in st.__all__

    log_path.unlink()
    output_dir.rmdir()
    degraded = st.hf_adapter_continuation_executor_status_report(state_path)

    assert degraded["status"] == "generation_limit_reached"
    assert degraded["healthy"] is False
    assert set(degraded["health_issues"]) == {
        "executor_log_missing",
        "promoted_output_missing",
    }
