from __future__ import annotations

import json
import os
import signal
import socket
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pytest
import spiraltorch as st
from spiraltorch import hf_adapter_executor_launch, hf_cli


def _executor_state(
    path: Path,
    *,
    invocation_count: int,
    run_id: str = "launch-test-run",
) -> dict[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "row_type": "hf_adapter_continuation_executor",
        "schema": st.HF_ADAPTER_CONTINUATION_EXECUTOR_SCHEMA,
        "status": "auditing",
        "action": "audit_chain",
        "created_at": now,
        "updated_at": now,
        "run_id": run_id,
        "source_paths": [],
        "output_root": str(path.parent),
        "state_path": str(path),
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


def _write_executor_state(
    path: Path,
    *,
    invocation_count: int,
    run_id: str = "launch-test-run",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _executor_state(
                path,
                invocation_count=invocation_count,
                run_id=run_id,
            )
        ),
        encoding="utf-8",
    )
    return path


def _fake_executor_command(
    state_path: Path,
    lock_path: Path,
    *,
    invocation_count: int,
    write_state: bool = True,
    sleep_seconds: float = 30.0,
) -> list[str]:
    payload = _executor_state(
        state_path,
        invocation_count=invocation_count,
    )
    script = """
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path

payload = json.loads(sys.argv[1])
state_path = Path(sys.argv[2])
lock_path = Path(sys.argv[3])
write_state = sys.argv[4] == "1"
sleep_seconds = float(sys.argv[5])

lock_path.parent.mkdir(parents=True, exist_ok=True)
descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "row_type": "hf_adapter_continuation_executor_lock",
            "lock_id": "fake-executor-lock",
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
        },
        handle,
    )

if write_state:
    state_path.write_text(json.dumps(payload), encoding="utf-8")

def stop(*_args):
    lock_path.unlink(missing_ok=True)
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
try:
    time.sleep(sleep_seconds)
finally:
    lock_path.unlink(missing_ok=True)
"""
    return [
        sys.executable,
        "-S",
        "-u",
        "-c",
        script,
        json.dumps(payload),
        str(state_path),
        str(lock_path),
        "1" if write_state else "0",
        str(sleep_seconds),
    ]


def _terminate_launcher(pid: int) -> None:
    if hf_adapter_executor_launch.local_pid_alive(pid) is not True:
        return
    if os.name == "posix":
        os.killpg(pid, signal.SIGTERM)
    else:
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 5.0
    while (
        hf_adapter_executor_launch.local_pid_alive(pid) is True
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)


def _flag(command: Sequence[str], name: str) -> str:
    index = list(command).index(name)
    return str(command[index + 1])


def _write_launch_state(
    path: Path,
    *,
    output_root: Path,
    executor_state_path: Path,
    latest: dict[str, object],
) -> Path:
    now = datetime.now(timezone.utc).isoformat()
    path.write_text(
        json.dumps(
            {
                "row_type": "hf_adapter_continuation_executor_launches",
                "schema": st.HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_SCHEMA,
                "created_at": now,
                "updated_at": now,
                "status": latest["status"],
                "output_root": str(output_root.resolve()),
                "executor_state_path": str(executor_state_path.resolve()),
                "launch_state_path": str(path.resolve()),
                "launch_count": 1,
                "launches": [latest],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_replayable_launch_state(
    tmp_path: Path,
    *,
    executor_action: str = "resume_executor",
    include_contract: bool = True,
) -> tuple[Path, Path, list[str]]:
    output_root = tmp_path / "executor"
    output_root.mkdir(parents=True)
    state_path = output_root / "state.json"
    state = _executor_state(state_path, invocation_count=1)
    state["status"] = (
        "generation_limit_reached"
        if executor_action == "resume_executor"
        else "stopped"
    )
    state["action"] = executor_action
    state["reason"] = (
        "max_generations_per_invocation_reached"
        if executor_action == "resume_executor"
        else "continuation_policy_stop"
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")
    log_path = output_root / "launcher.log"
    log_path.write_text("completed\n", encoding="utf-8")
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
    latest = {
        "launch_id": "replay-source-launch",
        "status": "completed",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": 99_999_999,
        "command_cwd": str(command_cwd),
        "command": hf_adapter_executor_launch._executor_child_command(argv),
        "executor_argv": argv,
        "executor_state_path": str(state_path.resolve()),
        "executor_run_id": "launch-test-run",
        "executor_invocation_count": 1,
        "log_path": str(log_path.resolve()),
        "process_group_isolated": True,
    }
    if include_contract:
        latest["resume_contract"] = hf_adapter_executor_launch._resume_contract(
            argv,
            output_root=output_root.resolve(),
            executor_state_path=state_path.resolve(),
            command_cwd=command_cwd,
        )
    launch_state_path = _write_launch_state(
        output_root / "launch.json",
        output_root=output_root,
        executor_state_path=state_path,
        latest=latest,
    )
    return launch_state_path, state_path, argv


def test_detached_launch_handoffs_blocks_duplicate_and_reports_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_root = tmp_path / "executor"
    state_path = output_root / "state.json"
    lock_path = output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
    monkeypatch.setattr(
        hf_adapter_executor_launch,
        "_executor_child_command",
        lambda _argv: _fake_executor_command(
            state_path,
            lock_path,
            invocation_count=1,
        ),
    )

    first = st.launch_hf_adapter_continuation_executor(
        ["source", "--output-root", str(output_root), "--run"],
        output_root=output_root,
        executor_state_path=state_path,
        handoff_timeout_seconds=3.0,
    )
    latest = first["latest_launch"]
    pid = latest["pid"]
    try:
        duplicate = st.launch_hf_adapter_continuation_executor(
            ["source", "--output-root", str(output_root), "--run"],
            output_root=output_root,
            executor_state_path=state_path,
        )
        launch_state_path = Path(first["launch_state_path"])
        loaded = st.load_hf_adapter_continuation_executor_launch(launch_state_path)
        status_report = st.hf_adapter_continuation_executor_launch_status_report(
            launch_state_path
        )
        status_code = hf_cli.adapter_continuation_executor_launch_status_main(
            [str(launch_state_path), "--require-healthy"]
        )
        output = capsys.readouterr().out

        assert first["request_status"] == "handed_off"
        assert first["created"] is True
        assert latest["executor_baseline"] is None
        assert latest["executor_invocation_count"] == 1
        assert latest["executor_lock_at_handoff"]["owner"]["pid"] == pid
        assert latest["process_group_isolated"] is True
        assert duplicate["request_status"] == "already_running"
        assert duplicate["created"] is False
        assert duplicate["launch_count"] == 1
        assert loaded["launch_count"] == 1
        assert status_report["status"] == "running"
        assert status_report["healthy"] is True
        assert status_report["launcher_pid_alive_observed"] is True
        assert status_code == 0
        assert "status=running" in output
        if os.name == "posix":
            assert os.getpgid(pid) == pid
            assert stat.S_IMODE(Path(latest["log_path"]).stat().st_mode) == 0o600
            assert stat.S_IMODE(launch_state_path.stat().st_mode) == 0o600
    finally:
        _terminate_launcher(pid)

    interrupted = st.hf_adapter_continuation_executor_launch_status_report(
        first["launch_state_path"]
    )
    assert interrupted["status"] == "interrupted"
    assert interrupted["healthy"] is False


def test_stale_executor_state_does_not_count_as_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "executor"
    state_path = _write_executor_state(
        output_root / "state.json",
        invocation_count=1,
    )
    lock_path = output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
    monkeypatch.setattr(
        hf_adapter_executor_launch,
        "_executor_child_command",
        lambda _argv: _fake_executor_command(
            state_path,
            lock_path,
            invocation_count=1,
            write_state=False,
        ),
    )

    report = st.launch_hf_adapter_continuation_executor(
        ["source", "--output-root", str(output_root), "--run"],
        output_root=output_root,
        executor_state_path=state_path,
        handoff_timeout_seconds=0.15,
    )
    latest = report["latest_launch"]
    try:
        deadline = time.monotonic() + 3.0
        while not lock_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert lock_path.is_file()
        status = st.hf_adapter_continuation_executor_launch_status_report(
            report["launch_state_path"]
        )
        assert report["request_status"] == "handoff_timeout"
        assert latest["status"] == "handoff_timeout"
        assert latest["executor_baseline"]["invocation_count"] == 1
        assert "handoff_at" not in latest
        assert status["status"] == "starting"
        assert status["executor_handoff_established"] is False
        assert status["executor_handoff_observation"] == "baseline_not_advanced"
    finally:
        _terminate_launcher(latest["pid"])

    exited = st.hf_adapter_continuation_executor_launch_status_report(
        report["launch_state_path"]
    )
    assert exited["status"] == "handoff_unverified"
    assert exited["recommended_action"] == "inspect_executor_handoff"


def test_handoff_timeout_does_not_trust_live_pid_without_executor_lock(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "executor"
    output_root.mkdir()
    state_path = _write_executor_state(
        output_root / "state.json",
        invocation_count=1,
    )
    log_path = output_root / "launcher.log"
    log_path.write_text("handoff timed out\n", encoding="utf-8")
    launch_path = _write_launch_state(
        output_root / "launch.json",
        output_root=output_root,
        executor_state_path=state_path,
        latest={
            "launch_id": "timed-out-reused-pid",
            "status": "handoff_timeout",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "executor_baseline": {
                "run_id": "launch-test-run",
                "invocation_count": 1,
            },
            "log_path": str(log_path.resolve()),
        },
    )

    status = st.hf_adapter_continuation_executor_launch_status_report(launch_path)

    assert status["status"] == "running_unverified"
    assert status["healthy"] is False
    assert status["recommended_action"] == "inspect_unverified_launcher"
    assert status["launcher_pid_alive_observed"] is True
    assert status["launcher_executor_lock_owner_verified"] is False
    assert status["executor_handoff_observation"] == "baseline_not_advanced"


def test_launch_status_treats_pre_spawn_local_artifact_as_starting(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "executor"
    output_root.mkdir()
    state_path = output_root / "state.json"
    launch_lock_path = (
        output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_LOCK_FILENAME
    )
    launch_lock_path.write_text(
        json.dumps(
            {
                "row_type": "hf_adapter_continuation_executor_launch_lock",
                "lock_id": "pre-spawn-launch-lock",
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
            }
        ),
        encoding="utf-8",
    )
    launch_path = _write_launch_state(
        output_root / "launch.json",
        output_root=output_root,
        executor_state_path=state_path,
        latest={
            "launch_id": "launching-before-spawn",
            "status": "launching",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "pid": None,
            "log_path": str(output_root / "pending.log"),
            "executor_baseline": None,
        },
    )

    report = st.hf_adapter_continuation_executor_launch_status_report(launch_path)

    assert report["status"] == "starting"
    assert report["healthy"] is True
    assert report["recommended_action"] == "wait_for_executor_handoff"
    assert report["launcher_launch_lock_owner_verified"] is True

    launch_lock_path.unlink()
    stale = st.hf_adapter_continuation_executor_launch_status_report(launch_path)
    assert stale["status"] == "running_unverified"
    assert stale["healthy"] is False
    assert stale["recommended_action"] == "inspect_unverified_launcher"


def test_failed_launcher_can_be_replaced_without_losing_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "executor"
    state_path = output_root / "state.json"
    lock_path = output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
    monkeypatch.setattr(
        hf_adapter_executor_launch,
        "_executor_child_command",
        lambda _argv: [sys.executable, "-c", "raise SystemExit(7)"],
    )
    failed = st.launch_hf_adapter_continuation_executor(
        ["source", "--output-root", str(output_root), "--run"],
        output_root=output_root,
        executor_state_path=state_path,
        handoff_timeout_seconds=1.0,
    )
    failed_status = st.hf_adapter_continuation_executor_launch_status_report(
        failed["launch_state_path"]
    )

    monkeypatch.setattr(
        hf_adapter_executor_launch,
        "_executor_child_command",
        lambda _argv: _fake_executor_command(
            state_path,
            lock_path,
            invocation_count=1,
        ),
    )
    resumed = st.launch_hf_adapter_continuation_executor(
        ["source", "--output-root", str(output_root), "--run"],
        output_root=output_root,
        executor_state_path=state_path,
        handoff_timeout_seconds=3.0,
    )
    try:
        status = st.hf_adapter_continuation_executor_launch_status_report(
            resumed["launch_state_path"]
        )
        assert failed["request_status"] == "launch_failed"
        assert failed["latest_launch"]["returncode_observed"] == 7
        assert failed_status["status"] == "launch_failed"
        assert failed_status["executor_handoff_observation"] == "failed"
        assert resumed["request_status"] == "handed_off"
        assert resumed["launch_count"] == 2
        assert resumed["launches"][0]["status"] == "launch_failed"
        assert status["status"] == "running"
    finally:
        _terminate_launcher(resumed["latest_launch"]["pid"])


def test_pid_liveness_without_executor_lock_does_not_block_relaunch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "executor"
    output_root.mkdir()
    state_path = _write_executor_state(
        output_root / "state.json",
        invocation_count=1,
    )
    launch_state_path = output_root / "launch.json"
    _write_launch_state(
        launch_state_path,
        output_root=output_root,
        executor_state_path=state_path,
        latest={
            "launch_id": "old-launch",
            "status": "handed_off",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "log_path": str(output_root / "old.log"),
        },
    )
    lock_path = output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
    monkeypatch.setattr(
        hf_adapter_executor_launch,
        "_executor_child_command",
        lambda _argv: _fake_executor_command(
            state_path,
            lock_path,
            invocation_count=2,
        ),
    )

    report = st.launch_hf_adapter_continuation_executor(
        ["source", "--output-root", str(output_root), "--run"],
        output_root=output_root,
        executor_state_path=state_path,
        launch_state_path=launch_state_path,
        handoff_timeout_seconds=3.0,
    )
    try:
        assert report["request_status"] == "handed_off"
        assert report["launch_count"] == 2
        assert report["launches"][0]["status"] == "exited_observed"
        assert (
            report["launches"][0]["process_liveness_observation"]
            == "pid_alive_without_executor_ownership"
        )
        assert report["latest_launch"]["executor_invocation_count"] == 2
    finally:
        _terminate_launcher(report["latest_launch"]["pid"])


def test_launch_status_propagates_executor_health_failure(tmp_path: Path) -> None:
    output_root = tmp_path / "executor"
    output_root.mkdir()
    state_path = output_root / "state.json"
    output_dir = output_root / "generation-001"
    output_dir.mkdir()
    executor_log = output_root / "executor.log"
    executor_log.write_text("running", encoding="utf-8")
    state = _executor_state(state_path, invocation_count=1)
    state["status"] = "running"
    state["generations"] = [
        {
            "attempt_id": "active-attempt",
            "status": "running",
            "runner_kind": "subprocess",
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "lineage_depth": 1,
            "output_dir": str(output_dir),
            "log_path": str(executor_log),
            "command_cwd": str(tmp_path),
        }
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    launcher_log = output_root / "launcher.log"
    launcher_log.write_text("launch", encoding="utf-8")
    launch_state_path = _write_launch_state(
        output_root / "launch.json",
        output_root=output_root,
        executor_state_path=state_path,
        latest={
            "launch_id": "health-launch",
            "status": "handed_off",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "log_path": str(launcher_log),
        },
    )

    report = st.hf_adapter_continuation_executor_launch_status_report(launch_state_path)

    assert report["status"] == "executor_unhealthy"
    assert report["healthy"] is False
    assert report["executor_healthy"] is False
    assert report["recommended_action"] == "inspect_executor_health_issues"
    assert "single_writer_lock_missing" in report["executor_status"]["health_issues"]

    state["status"] = "stopped"
    state["reason"] = "stop_requested"
    state["generations"][0]["status"] = "cancelled"
    state["generations"][0]["pid"] = 99_999_999
    state_path.write_text(json.dumps(state), encoding="utf-8")
    launch_state = json.loads(launch_state_path.read_text(encoding="utf-8"))
    launch_state["launches"][0]["pid"] = 99_999_999
    launch_state_path.write_text(json.dumps(launch_state), encoding="utf-8")

    terminal = st.hf_adapter_continuation_executor_launch_status_report(
        launch_state_path
    )
    assert terminal["status"] == "executor_unhealthy"
    assert terminal["healthy"] is False
    assert "cancelled_output_present" in terminal["executor_status"]["health_issues"]

    state["status"] = "output_quarantined"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    quarantined_but_unresolved = (
        st.hf_adapter_continuation_executor_launch_status_report(launch_state_path)
    )

    assert quarantined_but_unresolved["status"] == "executor_unhealthy"
    assert quarantined_but_unresolved["healthy"] is False
    assert (
        quarantined_but_unresolved["recommended_action"]
        == "resolve_cancelled_output"
    )


def test_detach_cli_replays_executor_arguments_without_recursing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source"
    artifact = tmp_path / "launch.json"
    output_root = tmp_path / "executor"
    launch_state = tmp_path / "launch-state.json"
    captured: dict[str, object] = {}

    def fake_launch(argv: Sequence[str], **kwargs: object) -> dict[str, object]:
        captured["argv"] = list(argv)
        captured["kwargs"] = kwargs
        return {
            "row_type": "hf_adapter_continuation_executor_launches",
            "request_status": "handed_off",
            "status": "handed_off",
            "created": True,
            "launch_count": 1,
            "launches": [],
            "executor_state_path": str(kwargs["executor_state_path"]),
            "launch_state_path": str(launch_state),
        }

    monkeypatch.setattr(hf_cli, "launch_hf_adapter_continuation_executor", fake_launch)
    code = hf_cli.adapter_continuation_executor_main(
        [
            str(source),
            "--output-root",
            str(output_root),
            "--run",
            "--detach",
            "--launch-state",
            str(launch_state),
            "--detach-handoff-timeout-seconds",
            "9.5",
            "--no-tee-output",
            "--max-generations",
            "3",
            "--expected-plan-id",
            "sha256:" + "a" * 64,
            "--require-pending-plan",
            "--retry-interrupted",
            "--output-prefix",
            "round",
            "--command-artifact",
            str(artifact),
            "--select-adapter-id",
            "adapter-7",
            "--no-recursive",
            "--no-infer-roots",
            "--max-lineage-depth",
            "8",
            "--target-eval-loss",
            "1.5",
            "--min-eval-improvement",
            "0.01",
            "--plateau-patience",
            "2",
            "--max-steps",
            "100",
            "--max-steps-multiplier",
            "1.25",
            "--max-train-samples",
            "200",
            "--max-train-samples-multiplier",
            "1.5",
            "--max-eval-samples",
            "20",
            "--max-eval-blocks",
            "4",
            "--streaming-validation-samples",
            "6",
        ]
    )
    output = capsys.readouterr().out
    child = captured["argv"]
    kwargs = captured["kwargs"]

    assert code == 0
    assert child[0] == str(source.resolve())
    assert "--run" in child
    assert "--detach" not in child
    assert "--launch-state" not in child
    assert "--json" not in child
    assert "--require-pending-plan" in child
    assert _flag(child, "--expected-plan-id") == "sha256:" + "a" * 64
    assert _flag(child, "--state") == str(
        (output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_FILENAME).resolve()
    )
    assert _flag(child, "--command-artifact") == str(artifact.resolve())
    assert _flag(child, "--max-lineage-depth") == "8"
    assert _flag(child, "--streaming-validation-samples") == "6"
    assert kwargs["launch_state_path"] == launch_state
    assert kwargs["handoff_timeout_seconds"] == 9.5
    assert "status=handed_off" in output


def test_durable_resume_replays_detached_executor_and_advances_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "executor"
    state_path = output_root / "state.json"
    lock_path = output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
    invocation = {"value": 1}

    def fake_child(_argv: Sequence[str]) -> list[str]:
        return _fake_executor_command(
            state_path,
            lock_path,
            invocation_count=invocation["value"],
        )

    monkeypatch.setattr(
        hf_adapter_executor_launch,
        "_executor_child_command",
        fake_child,
    )
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
    first = st.launch_hf_adapter_continuation_executor(
        argv,
        output_root=output_root,
        executor_state_path=state_path,
        command_cwd=tmp_path,
        handoff_timeout_seconds=3.0,
    )
    first_pid = first["latest_launch"]["pid"]
    _terminate_launcher(first_pid)
    terminal = st.load_hf_adapter_continuation_executor(state_path)
    terminal["status"] = "generation_limit_reached"
    terminal["action"] = "resume_executor"
    terminal["reason"] = "max_generations_per_invocation_reached"
    state_path.write_text(json.dumps(terminal), encoding="utf-8")

    plan = st.hf_adapter_continuation_executor_resume_report(first["launch_state_path"])
    invocation["value"] = 2
    resumed = st.resume_hf_adapter_continuation_executor(
        first["launch_state_path"],
        handoff_timeout_seconds=3.0,
    )
    resumed_payload = json.loads(
        Path(first["launch_state_path"]).read_text(encoding="utf-8")
    )
    second_pid = resumed_payload["launches"][-1]["pid"]
    try:
        launch_state = st.load_hf_adapter_continuation_executor_launch(
            first["launch_state_path"]
        )
        latest = launch_state["launches"][-1]
        launch_status = st.hf_adapter_continuation_executor_launch_status_report(
            first["launch_state_path"]
        )

        assert plan["status"] == "resume_ready"
        assert plan["ready"] is True
        assert plan["resume_contract_source"] == "recorded"
        assert plan["executor_identity_source"] == "recorded_handoff"
        assert plan["executor_invocation_count"] == 1
        assert resumed["status"] == "handed_off"
        assert resumed["created"] is True
        assert resumed["source_launch_id"] == first["latest_launch"]["launch_id"]
        assert resumed["resumed_executor_invocation_count"] == 2
        assert launch_state["launch_count"] == 2
        assert latest["resumed_from_launch_id"] == first["latest_launch"]["launch_id"]
        assert (
            latest["resumed_from_contract_sha256"]
            == plan["resume_contract"]["contract_sha256"]
        )
        assert latest["resumed_from_executor_run_id"] == "launch-test-run"
        assert latest["resumed_from_executor_invocation_count"] == 1
        assert launch_status["status"] == "running"
    finally:
        if isinstance(second_pid, int):
            _terminate_launcher(second_pid)


def test_resume_plan_validates_legacy_contract_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    legacy_path, _, _ = _write_replayable_launch_state(
        tmp_path / "legacy",
        include_contract=False,
    )
    legacy = st.hf_adapter_continuation_executor_resume_report(legacy_path)
    assert legacy["ready"] is True
    assert legacy["resume_contract_source"] == "legacy_command_reconstructed"

    launch_path, _, _ = _write_replayable_launch_state(tmp_path / "tampered")
    payload = json.loads(launch_path.read_text(encoding="utf-8"))
    payload["launches"][0]["executor_argv"].extend(["--max-lineage-depth", "99"])
    launch_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="resume contract is invalid"):
        st.hf_adapter_continuation_executor_resume_report(launch_path)

    identity_path, identity_state_path, _ = _write_replayable_launch_state(
        tmp_path / "identity"
    )
    identity_state = json.loads(identity_state_path.read_text(encoding="utf-8"))
    identity_state["invocation_count"] = 2
    identity_state_path.write_text(json.dumps(identity_state), encoding="utf-8")
    identity = st.hf_adapter_continuation_executor_resume_report(identity_path)
    assert identity["ready"] is False
    assert identity["issue"] == "executor_identity_changed"
    assert identity["action"] == "inspect_executor_invocation"

    missing_handoff_path, _, _ = _write_replayable_launch_state(
        tmp_path / "missing-handoff"
    )
    missing_handoff_payload = json.loads(
        missing_handoff_path.read_text(encoding="utf-8")
    )
    missing_handoff_payload["launches"][0].pop("executor_run_id")
    missing_handoff_payload["launches"][0].pop("executor_invocation_count")
    missing_handoff_path.write_text(
        json.dumps(missing_handoff_payload), encoding="utf-8"
    )
    missing_handoff = st.hf_adapter_continuation_executor_resume_report(
        missing_handoff_path
    )
    assert missing_handoff["ready"] is False
    assert missing_handoff["launcher_status"] == "handoff_unverified"
    assert missing_handoff["executor_identity_source"] == "recorded_handoff_missing"

    baseline_path, _, _ = _write_replayable_launch_state(tmp_path / "baseline")
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_payload["launches"][0].pop("executor_run_id")
    baseline_payload["launches"][0].pop("executor_invocation_count")
    baseline_payload["launches"][0]["executor_baseline"] = None
    baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")
    baseline = st.hf_adapter_continuation_executor_resume_report(baseline_path)
    assert baseline["ready"] is True
    assert baseline["executor_identity_source"] == "baseline_reconstructed"

    missing_log_path, _, _ = _write_replayable_launch_state(tmp_path / "missing-log")
    missing_log_payload = json.loads(missing_log_path.read_text(encoding="utf-8"))
    Path(missing_log_payload["launches"][0]["log_path"]).unlink()
    missing_log = st.hf_adapter_continuation_executor_resume_report(missing_log_path)
    assert missing_log["ready"] is False
    assert missing_log["issue"] == "launcher_unhealthy"
    assert missing_log["action"] == "inspect_missing_launcher_log"

    malformed_path, _, _ = _write_replayable_launch_state(tmp_path / "malformed")
    malformed = json.loads(malformed_path.read_text(encoding="utf-8"))
    malformed["launches"].append("not-a-launch")
    malformed["launch_count"] = 2
    malformed_path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid launch row"):
        st.hf_adapter_continuation_executor_resume_report(malformed_path)


def test_resume_plan_honors_terminal_action_and_cas_source_launch(
    tmp_path: Path,
) -> None:
    stopped_path, _, _ = _write_replayable_launch_state(
        tmp_path / "stopped",
        executor_action="stop_training",
    )
    stopped = st.hf_adapter_continuation_executor_resume_report(stopped_path)
    assert stopped["ready"] is False
    assert stopped["issue"] == "executor_action_not_resumable"
    assert stopped["action"] == "stop_training"

    launch_path, state_path, argv = _write_replayable_launch_state(tmp_path / "cas")
    plan = st.hf_adapter_continuation_executor_resume_report(launch_path)
    payload = json.loads(launch_path.read_text(encoding="utf-8"))
    newer = dict(payload["launches"][0])
    newer["launch_id"] = "newer-launch"
    payload["launches"].append(newer)
    payload["launch_count"] = 2
    launch_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source launch changed"):
        st.launch_hf_adapter_continuation_executor(
            argv,
            output_root=state_path.parent,
            executor_state_path=state_path,
            launch_state_path=launch_path,
            command_cwd=tmp_path / "cas",
            _resume_expectation={
                "launch_id": plan["source_launch_id"],
                "contract_sha256": plan["resume_contract"]["contract_sha256"],
                "executor_run_id": plan["executor_run_id"],
                "executor_invocation_count": plan["executor_invocation_count"],
            },
        )

    invocation_path, invocation_state_path, invocation_argv = (
        _write_replayable_launch_state(tmp_path / "invocation-cas")
    )
    invocation_plan = st.hf_adapter_continuation_executor_resume_report(invocation_path)
    advanced = json.loads(invocation_state_path.read_text(encoding="utf-8"))
    advanced["invocation_count"] = 2
    invocation_state_path.write_text(json.dumps(advanced), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invocation changed"):
        st.launch_hf_adapter_continuation_executor(
            invocation_argv,
            output_root=invocation_state_path.parent,
            executor_state_path=invocation_state_path,
            launch_state_path=invocation_path,
            command_cwd=tmp_path / "invocation-cas",
            _resume_expectation={
                "launch_id": invocation_plan["source_launch_id"],
                "contract_sha256": invocation_plan["resume_contract"][
                    "contract_sha256"
                ],
                "executor_run_id": invocation_plan["executor_run_id"],
                "executor_invocation_count": invocation_plan[
                    "executor_invocation_count"
                ],
            },
        )


def test_resume_plan_rejects_remote_locked_and_missing_cwd_sources(
    tmp_path: Path,
) -> None:
    remote_path, _, _ = _write_replayable_launch_state(tmp_path / "remote")
    remote_payload = json.loads(remote_path.read_text(encoding="utf-8"))
    remote_payload["launches"][0]["hostname"] = "remote.example"
    remote_path.write_text(json.dumps(remote_payload), encoding="utf-8")
    remote = st.hf_adapter_continuation_executor_resume_report(remote_path)
    assert remote["ready"] is False
    assert remote["issue"] == "launcher_not_resumable"
    assert remote["action"] == "inspect_remote_launcher"

    locked_path, locked_state_path, _ = _write_replayable_launch_state(
        tmp_path / "locked"
    )
    lock_path = (
        locked_state_path.parent / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
    )
    lock_path.write_text(
        json.dumps(
            {
                "row_type": "hf_adapter_continuation_executor_lock",
                "lock_id": "active-resume-test-lock",
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
            }
        ),
        encoding="utf-8",
    )
    locked = st.hf_adapter_continuation_executor_resume_report(locked_path)
    assert locked["ready"] is False
    assert locked["issue"] == "executor_lock_unavailable"
    assert locked["action"] == "inspect_executor_lock"

    missing_cwd_path, missing_cwd_state_path, missing_cwd_argv = (
        _write_replayable_launch_state(tmp_path / "missing-cwd")
    )
    missing_cwd_payload = json.loads(missing_cwd_path.read_text(encoding="utf-8"))
    missing_cwd = (tmp_path / "removed-command-cwd").resolve()
    latest = missing_cwd_payload["launches"][0]
    latest["command_cwd"] = str(missing_cwd)
    latest["resume_contract"] = hf_adapter_executor_launch._resume_contract(
        missing_cwd_argv,
        output_root=missing_cwd_state_path.parent.resolve(),
        executor_state_path=missing_cwd_state_path.resolve(),
        command_cwd=missing_cwd,
    )
    missing_cwd_path.write_text(json.dumps(missing_cwd_payload), encoding="utf-8")
    missing = st.hf_adapter_continuation_executor_resume_report(missing_cwd_path)
    assert missing["ready"] is False
    assert missing["issue"] == "command_cwd_missing"
    assert missing["action"] == "restore_command_cwd"


def test_resume_cli_supports_read_only_plan_and_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    launch_path, _, _ = _write_replayable_launch_state(tmp_path)
    plan_code = hf_cli.adapter_continuation_executor_resume_main(
        [str(launch_path), "--plan"]
    )
    plan_output = capsys.readouterr().out
    captured: dict[str, object] = {}

    def fake_resume(path: Path, *, handoff_timeout_seconds: float) -> dict[str, object]:
        captured["path"] = path
        captured["handoff_timeout_seconds"] = handoff_timeout_seconds
        report = st.hf_adapter_continuation_executor_resume_report(path)
        report.update(
            {
                "status": "handed_off",
                "ready": True,
                "created": True,
                "resumed_launch_id": "resumed-launch",
                "resumed_executor_invocation_count": 2,
            }
        )
        return report

    monkeypatch.setattr(
        hf_cli,
        "resume_hf_adapter_continuation_executor",
        fake_resume,
    )
    resume_code = hf_cli.adapter_continuation_executor_resume_main(
        [str(launch_path), "--handoff-timeout-seconds", "9.5"]
    )
    resume_output = capsys.readouterr().out

    assert plan_code == 0
    assert "status=resume_ready" in plan_output
    assert resume_code == 0
    assert "status=handed_off" in resume_output
    assert captured["path"] == launch_path
    assert captured["handoff_timeout_seconds"] == 9.5


def test_launch_validation_and_public_surface(tmp_path: Path) -> None:
    output_root = tmp_path / "executor"
    state_path = output_root / "state.json"

    with pytest.raises(ValueError, match="finite and non-negative"):
        st.launch_hf_adapter_continuation_executor(
            ["source"],
            output_root=output_root,
            executor_state_path=state_path,
            handoff_timeout_seconds=float("nan"),
        )
    with pytest.raises(ValueError, match="cannot overwrite an executor lock"):
        st.launch_hf_adapter_continuation_executor(
            ["source", "--run"],
            output_root=output_root,
            executor_state_path=state_path,
            launch_state_path=(
                output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
            ),
        )
    with pytest.raises(ValueError, match="requires --run"):
        st.launch_hf_adapter_continuation_executor(
            ["source"],
            output_root=output_root,
            executor_state_path=state_path,
        )
    with pytest.raises(ValueError, match="launcher option --detach"):
        st.launch_hf_adapter_continuation_executor(
            ["source", "--run", "--detach"],
            output_root=output_root,
            executor_state_path=state_path,
        )
    with pytest.raises(SystemExit):
        hf_cli.adapter_continuation_executor_main(
            ["source", "--output-root", str(output_root), "--detach"]
        )

    for name in (
        "hf_adapter_executor_launch",
        "launch_hf_adapter_continuation_executor",
        "load_hf_adapter_continuation_executor_launch",
        "hf_adapter_continuation_executor_launch_lines",
        "hf_adapter_continuation_executor_launch_status_report",
        "hf_adapter_continuation_executor_launch_status_lines",
        "hf_adapter_continuation_executor_resume_report",
        "resume_hf_adapter_continuation_executor",
        "hf_adapter_continuation_executor_resume_lines",
        "HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_SCHEMA",
        "HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_STATUS_SCHEMA",
        "HF_ADAPTER_CONTINUATION_EXECUTOR_RESUME_SCHEMA",
    ):
        assert name in st.__all__
