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
        assert report["request_status"] == "handoff_timeout"
        assert latest["status"] == "handoff_timeout"
        assert latest["executor_baseline"]["invocation_count"] == 1
        assert "handoff_at" not in latest
    finally:
        _terminate_launcher(latest["pid"])


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
    assert report["recommended_action"] == "inspect_executor_health"
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
    assert _flag(child, "--state") == str(
        (output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_FILENAME).resolve()
    )
    assert _flag(child, "--command-artifact") == str(artifact.resolve())
    assert _flag(child, "--max-lineage-depth") == "8"
    assert _flag(child, "--streaming-validation-samples") == "6"
    assert kwargs["launch_state_path"] == launch_state
    assert kwargs["handoff_timeout_seconds"] == 9.5
    assert "status=handed_off" in output


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
        "HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_SCHEMA",
        "HF_ADAPTER_CONTINUATION_EXECUTOR_LAUNCH_STATUS_SCHEMA",
    ):
        assert name in st.__all__
