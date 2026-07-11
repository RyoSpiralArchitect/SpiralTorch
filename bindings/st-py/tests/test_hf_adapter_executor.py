from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Mapping, Sequence

import pytest
import spiraltorch as st
from spiraltorch import hf_cli


def _write_adapter(path: Path, weights: bytes) -> Path:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "org/base",
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 4,
                "lora_alpha": 8,
                "target_modules": ["q_proj"],
            }
        ),
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(weights)
    return path


def _flag(command: Sequence[str], name: str) -> str:
    index = list(command).index(name)
    return str(command[index + 1])


def _run_card(
    parent: Path,
    *,
    before: float,
    after: float,
    launch_command: Sequence[str],
) -> dict[str, object]:
    return {
        "row_type": "hf_finetune_run_card",
        "load_status": "ok",
        "failure_stage": None,
        "model_saved": True,
        "adapter_saved": True,
        "model_name": str(parent),
        "model_artifact_kind": "peft_adapter",
        "finetune_mode": "lora",
        "finetune_start_report": {
            "mode": "adapter_warm_start",
            "adapter_weights_source": str(parent),
            "weights_only_warm_start": True,
            "trainer_checkpoint_resume": False,
        },
        "model_prepare_report": {
            "mode": "lora",
            "adapter_attached": True,
            "parameter_report_after": {
                "parameter_count": 100,
                "trainable_parameter_count": 8,
                "frozen_parameter_count": 92,
                "trainable_parameter_ratio": 0.08,
            },
        },
        "trainer_metrics": {"train_loss": after},
        "eval_before_train": {"status": "ok", "eval_loss": before},
        "eval_after_train": {"status": "ok", "eval_loss": after},
        "launch_command": list(launch_command),
        "launch_command_display": " ".join(launch_command),
        "launch_command_source": "test",
    }


def _launch_command(parent: Path, output: Path) -> list[str]:
    bridge = Path(__file__).resolve().parents[1] / "examples" / "hf_finetune_bridge.py"
    return [
        sys.executable,
        str(bridge),
        "--model-name",
        str(parent),
        "--train",
        "--output-dir",
        str(output),
        "--run-card",
        str(output / "spiraltorch-hf-finetune-run-card.json"),
        "--trainer-trace-jsonl",
        str(output / "spiraltorch-hf-finetune-trainer-trace.jsonl"),
        "--finetune-mode",
        "lora",
        "--max-steps",
        "1",
        "--max-train-samples",
        "8",
        "--adapter-promotion-gate",
        "--eval-before-train",
    ]


def _write_promoted_adapter(
    output: Path,
    parent: Path,
    *,
    weights: bytes,
    before: float,
    after: float,
    launch_command: Sequence[str] | None = None,
    runtime_input_id: str | None = None,
    expected_runtime_input_id: str | None = None,
    execution_input_id: str | None = None,
    expected_execution_input_id: str | None = None,
    dataset_input_id: str | None = None,
    expected_dataset_input_id: str | None = None,
    dataset_materialization_id: str | None = None,
    expected_dataset_materialization_id: str | None = None,
    tokenized_dataset_id: str | None = None,
    expected_tokenized_dataset_id: str | None = None,
    dataset_name: str = "org/corpus",
    dataset_revision: str = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
) -> dict[str, object]:
    adapter = _write_adapter(output, weights)
    command = list(launch_command or _launch_command(parent, output))
    run_card_path = adapter / "spiraltorch-hf-finetune-run-card.json"
    card = _run_card(
        parent,
        before=before,
        after=after,
        launch_command=command,
    )
    if runtime_input_id is not None:
        for key, phase in (
            ("model_runtime_identity_pre_model", "pre_model_load"),
            ("model_runtime_identity_after_model", "after_model_load"),
        ):
            card[key] = {
                "status": "ready",
                "phase": phase,
                "expected_identity_id": (
                    runtime_input_id
                    if phase == "after_model_load"
                    and expected_runtime_input_id is None
                    else expected_runtime_input_id
                ),
                "observed_identity_id": runtime_input_id,
                "identity_verified": True,
            }
        card["model_runtime_identity_contract"] = {
            "status": (
                "enforced"
                if expected_runtime_input_id is not None
                else "adopted"
            ),
            "expected_identity_id": expected_runtime_input_id,
            "observed_identity_id": runtime_input_id,
            "identity_verified": True,
        }
    if execution_input_id is not None:
        for key, phase in (
            ("finetune_execution_identity_pre_model", "pre_model_load"),
            ("finetune_execution_identity_after_model", "after_model_load"),
        ):
            card[key] = {
                "status": "ready",
                "phase": phase,
                "expected_identity_id": (
                    execution_input_id
                    if phase == "after_model_load"
                    and expected_execution_input_id is None
                    else expected_execution_input_id
                ),
                "observed_identity_id": execution_input_id,
                "identity_verified": True,
            }
        card["finetune_execution_identity_contract"] = {
            "status": (
                "enforced"
                if expected_execution_input_id is not None
                else "adopted"
            ),
            "expected_identity_id": expected_execution_input_id,
            "observed_identity_id": execution_input_id,
            "identity_verified": True,
        }
    if dataset_input_id is not None:
        card["dataset_input_identity"] = {
            "status": "ready",
            "phase": "preflight",
            "expected_identity_id": expected_dataset_input_id,
            "observed_identity_id": dataset_input_id,
            "effective_dataset_name": dataset_name,
            "effective_revision": dataset_revision,
            "identity_verified": True,
        }
        card["dataset_input_identity_after_load"] = {
            "status": "ready",
            "phase": "after_load",
            "expected_identity_id": expected_dataset_input_id or dataset_input_id,
            "observed_identity_id": dataset_input_id,
            "effective_dataset_name": dataset_name,
            "effective_revision": dataset_revision,
            "identity_verified": True,
        }
        card["dataset_input_identity_contract"] = {
            "status": (
                "enforced"
                if expected_dataset_input_id is not None
                else "adopted"
            ),
            "expected_identity_id": expected_dataset_input_id or dataset_input_id,
            "observed_identity_id": dataset_input_id,
            "effective_dataset_name": dataset_name,
            "effective_revision": dataset_revision,
            "identity_verified": True,
            "fail_fast": True,
        }
        command.extend(
            [
                "--dataset-name",
                dataset_name,
                "--dataset-revision",
                dataset_revision,
                "--expected-dataset-input-id",
                expected_dataset_input_id or dataset_input_id,
            ]
        )
        card["launch_command"] = command
        card["launch_command_display"] = " ".join(command)
    if dataset_materialization_id is not None:
        card["dataset_materialization_identity"] = {
            "status": "ready",
            "phase": "after_selection",
            "expected_identity_id": expected_dataset_materialization_id,
            "observed_identity_id": dataset_materialization_id,
            "identity_verified": True,
            "materialized_rows_verified": True,
            "total_rows": 8,
            "total_utf8_bytes": 128,
        }
        card["dataset_materialization_identity_contract"] = {
            "status": (
                "enforced"
                if expected_dataset_materialization_id is not None
                else "adopted"
            ),
            "expected_identity_id": (
                expected_dataset_materialization_id
                or dataset_materialization_id
            ),
            "observed_identity_id": dataset_materialization_id,
            "identity_verified": True,
            "fail_fast": True,
        }
        command.extend(
            [
                "--expected-dataset-materialization-id",
                expected_dataset_materialization_id
                or dataset_materialization_id,
            ]
        )
        card["launch_command"] = command
        card["launch_command_display"] = " ".join(command)
    if tokenized_dataset_id is not None:
        card["tokenized_dataset_identity"] = {
            "status": "ready",
            "phase": "after_tokenization",
            "expected_identity_id": expected_tokenized_dataset_id,
            "observed_identity_id": tokenized_dataset_id,
            "identity_verified": True,
            "tokenized_rows_verified": True,
            "total_rows": 4,
            "total_input_tokens": 32,
        }
        card["tokenized_dataset_identity_contract"] = {
            "status": (
                "enforced"
                if expected_tokenized_dataset_id is not None
                else "adopted"
            ),
            "expected_identity_id": (
                expected_tokenized_dataset_id or tokenized_dataset_id
            ),
            "observed_identity_id": tokenized_dataset_id,
            "identity_verified": True,
            "fail_fast": True,
        }
        command.extend(
            [
                "--expected-tokenized-dataset-id",
                expected_tokenized_dataset_id or tokenized_dataset_id,
            ]
        )
        card["launch_command"] = command
        card["launch_command_display"] = " ".join(command)
    lineage = st.write_hf_adapter_lineage(
        adapter,
        parent_adapter=parent,
        run_card=card,
        run_card_path=run_card_path,
    )
    card["adapter_lineage"] = lineage
    promotion = st.write_hf_adapter_promotion(
        adapter,
        card,
        parent_adapter=parent,
    )
    card["adapter_promotion"] = promotion
    run_card_path.write_text(json.dumps(card), encoding="utf-8")
    return lineage


def _seed_chain(
    tmp_path: Path,
    *,
    improvement: float = 0.1,
    runtime_input_id: str | None = None,
    execution_input_id: str | None = None,
    dataset_input_id: str | None = None,
    dataset_materialization_id: str | None = None,
    tokenized_dataset_id: str | None = None,
) -> tuple[Path, Path]:
    root = _write_adapter(tmp_path / "root", b"root")
    st.write_hf_adapter_lineage(root)
    child = tmp_path / "child"
    _write_promoted_adapter(
        child,
        root,
        weights=b"child",
        before=1.0,
        after=1.0 - improvement,
        runtime_input_id=runtime_input_id,
        execution_input_id=execution_input_id,
        dataset_input_id=dataset_input_id,
        expected_dataset_input_id=dataset_input_id,
        dataset_materialization_id=dataset_materialization_id,
        expected_dataset_materialization_id=dataset_materialization_id,
        tokenized_dataset_id=tokenized_dataset_id,
        expected_tokenized_dataset_id=tokenized_dataset_id,
    )
    return root, child


class FakeFineTuneRunner:
    def __init__(
        self,
        improvements: Sequence[float],
        *,
        fail_returncode: int | None = None,
        weight_prefix: str = "generated",
    ) -> None:
        self.improvements = list(improvements)
        self.fail_returncode = fail_returncode
        self.weight_prefix = weight_prefix
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str]) -> int:
        values = [str(item) for item in command]
        self.commands.append(values)
        if self.fail_returncode is not None:
            return self.fail_returncode
        output = Path(_flag(values, "--output-dir"))
        parent = Path(_flag(values, "--model-name"))
        improvement = self.improvements[len(self.commands) - 1]
        before = 0.9 - (len(self.commands) - 1) * 0.1
        _write_promoted_adapter(
            output,
            parent,
            weights=f"{self.weight_prefix}-{len(self.commands)}".encode(),
            before=before,
            after=before - improvement,
            launch_command=values,
        )
        return 0


def test_executor_dry_run_writes_replayable_state_and_cli(
    tmp_path: Path,
    capsys,
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    state_path = output_root / "state.json"

    report = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        max_lineage_depth=3,
        max_steps=2,
        max_train_samples=16,
    )
    loaded = st.load_hf_adapter_continuation_executor(state_path)
    code = hf_cli.adapter_continuation_executor_main(
        [
            str(child),
            "--output-root",
            str(output_root),
            "--state",
            str(state_path),
            "--max-lineage-depth",
            "3",
            "--max-steps",
            "2",
            "--max-train-samples",
            "16",
        ]
    )
    output = capsys.readouterr().out

    assert report["status"] == "ready"
    assert report["action"] == "run_generation"
    assert report["pending_generation"]["lineage_depth"] == 2
    assert report["pending_generation"]["preflight"]["ready"] is True
    selected_transition = report["selected_transition"]
    assert report["transition_count"] == 1
    assert report["ready_transition_count"] == 1
    assert report["selected_path_transition_count"] == 1
    assert report["selected_path_transitions_ready"] is True
    assert selected_transition["status"] == "ready"
    assert selected_transition["child_adapter_id"] == report["selected_adapter_id"]
    assert selected_transition["child_lineage_depth"] == 1
    assert report["pending_generation"]["source_transition"] == selected_transition
    assert report["pending_generation"]["command"][
        "promotion_chain_selected_transition"
    ] == selected_transition
    assert report["pending_generation"]["preflight"][
        "promotion_chain_selected_transition"
    ] == selected_transition
    command_runtime = report["pending_generation"]["command_runtime"]
    assert command_runtime["status"] == "portable_module"
    assert command_runtime["source_kind"] == "python_bridge_script"
    assert report["pending_generation"]["command"]["command"][:3] == [
        sys.executable,
        "-m",
        "spiraltorch.hf_finetune_entrypoint",
    ]
    assert report["pending_generation"]["preflight"][
        "command_runtime_module_importable"
    ] is True
    identity_contract = report["pending_generation"]["parent_identity_contract"]
    input_identity = report["pending_generation"]["adapter_input_identity"]
    training_input_contract = report["pending_generation"][
        "training_input_identity_contract"
    ]
    training_input_identity = report["pending_generation"][
        "training_input_identity"
    ]
    assert identity_contract["status"] == "enforced"
    assert identity_contract["expected_parent_adapter_id"] == (
        report["selected_adapter_id"]
    )
    assert input_identity["status"] == "ready"
    assert input_identity["observed_adapter_id"] == report["selected_adapter_id"]
    assert training_input_contract["status"] == "not_applicable"
    assert training_input_identity["status"] == "not_applicable"
    resolved_command = report["pending_generation"]["command"]["command"]
    assert _flag(resolved_command, "--expected-parent-adapter-id") == (
        report["selected_adapter_id"]
    )
    assert report["generation_attempt_count"] == 0
    assert not (output_root / "generation-002").exists()
    assert loaded["status"] == "ready"
    assert loaded["invocation_count"] == 1
    assert code == 0
    assert "status=ready" in output
    assert "depth=2" in output
    assert "hf_adapter_executor" in st.__all__
    lines = st.hf_adapter_continuation_executor_lines(report)
    assert "status=ready" in lines[0]
    assert "transitions=1/1" in lines[0]
    assert any(
        line.startswith("hf_adapter_continuation_executor_transition status=ready")
        for line in lines
    )
    assert any("runtime=portable_module" in line for line in lines)
    assert any("input_identity=ready" in line for line in lines)


def test_executor_preserves_enforced_runtime_input_contract(tmp_path: Path) -> None:
    runtime_input_id = "sha256:" + "6" * 64
    _, child = _seed_chain(tmp_path, runtime_input_id=runtime_input_id)

    report = st.run_hf_adapter_continuation_executor(
        child,
        output_root=tmp_path / "executor-runs",
        max_lineage_depth=3,
        max_steps=2,
        max_train_samples=16,
    )
    pending = report["pending_generation"]
    contract = pending["runtime_input_identity_contract"]
    command = pending["command"]["command"]

    assert contract["status"] == "enforced"
    assert contract["expected_identity_id"] == runtime_input_id
    assert pending["runtime_input_expected_id"] == runtime_input_id
    assert _flag(command, "--expected-runtime-input-id") == runtime_input_id
    assert any(
        "runtime_input_contract=enforced" in line
        and f"runtime_input_expected={runtime_input_id}" in line
        for line in st.hf_adapter_continuation_executor_lines(report)
    )


def test_executor_preserves_enforced_dataset_input_contract(tmp_path: Path) -> None:
    dataset_input_id = "sha256:" + "4" * 64
    _, child = _seed_chain(tmp_path, dataset_input_id=dataset_input_id)

    report = st.run_hf_adapter_continuation_executor(
        child,
        output_root=tmp_path / "executor-runs",
        max_lineage_depth=3,
        max_steps=2,
        max_train_samples=16,
    )
    pending = report["pending_generation"]
    contract = pending["dataset_input_identity_contract"]
    command = pending["command"]["command"]

    assert contract["status"] == "enforced"
    assert contract["expected_identity_id"] == dataset_input_id
    assert pending["dataset_input_expected_id"] == dataset_input_id
    assert pending["dataset_input_effective_name"] == "org/corpus"
    assert pending["dataset_input_effective_revision"] == (
        "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
    )
    assert _flag(command, "--expected-dataset-input-id") == dataset_input_id
    assert _flag(command, "--dataset-name") == "org/corpus"
    assert any(
        "dataset_input_contract=enforced" in line
        and f"dataset_input_expected={dataset_input_id}" in line
        for line in st.hf_adapter_continuation_executor_lines(report)
    )


def test_executor_preserves_enforced_dataset_materialization_contract(
    tmp_path: Path,
) -> None:
    materialization_id = "sha256:" + "5" * 64
    _, child = _seed_chain(
        tmp_path,
        dataset_materialization_id=materialization_id,
    )

    report = st.run_hf_adapter_continuation_executor(
        child,
        output_root=tmp_path / "executor-runs",
        max_lineage_depth=3,
        max_steps=2,
        max_train_samples=8,
    )
    pending = report["pending_generation"]
    contract = pending["dataset_materialization_identity_contract"]
    command = pending["command"]["command"]

    assert contract["status"] == "enforced"
    assert contract["expected_identity_id"] == materialization_id
    assert pending["dataset_materialization_expected_id"] == materialization_id
    assert (
        _flag(command, "--expected-dataset-materialization-id")
        == materialization_id
    )
    assert any(
        "dataset_materialization_contract=enforced" in line
        and f"dataset_materialization_expected={materialization_id}" in line
        for line in st.hf_adapter_continuation_executor_lines(report)
    )


def test_executor_preserves_enforced_tokenized_dataset_contract(
    tmp_path: Path,
) -> None:
    tokenized_id = "sha256:" + "a" * 64
    _, child = _seed_chain(tmp_path, tokenized_dataset_id=tokenized_id)

    report = st.run_hf_adapter_continuation_executor(
        child,
        output_root=tmp_path / "executor-runs",
        max_lineage_depth=3,
        max_steps=2,
        max_train_samples=8,
    )
    pending = report["pending_generation"]
    contract = pending["tokenized_dataset_identity_contract"]
    command = pending["command"]["command"]

    assert contract["status"] == "enforced"
    assert contract["expected_identity_id"] == tokenized_id
    assert pending["tokenized_dataset_expected_id"] == tokenized_id
    assert _flag(command, "--expected-tokenized-dataset-id") == tokenized_id
    assert any(
        "tokenized_dataset_contract=enforced" in line
        and f"tokenized_dataset_expected={tokenized_id}" in line
        for line in st.hf_adapter_continuation_executor_lines(report)
    )


def test_executor_reissues_post_selection_identities_for_shape_change(
    tmp_path: Path,
) -> None:
    materialization_id = "sha256:" + "5" * 64
    tokenized_id = "sha256:" + "a" * 64
    _, child = _seed_chain(
        tmp_path,
        dataset_materialization_id=materialization_id,
        tokenized_dataset_id=tokenized_id,
    )

    report = st.run_hf_adapter_continuation_executor(
        child,
        output_root=tmp_path / "executor-runs",
        max_lineage_depth=3,
        max_steps=2,
        max_train_samples=16,
    )
    pending = report["pending_generation"]
    command_report = pending["command"]
    command = command_report["command"]
    materialization_contract = pending[
        "dataset_materialization_identity_contract"
    ]
    tokenized_contract = pending["tokenized_dataset_identity_contract"]

    assert command_report["dataset_shape_override_changed"] is True
    assert command_report["dataset_shape_changes"] == [
        {
            "flag": "--max-train-samples",
            "source_value": "8",
            "target_value": "16",
        }
    ]
    assert materialization_contract["status"] == "reissued"
    assert tokenized_contract["status"] == "reissued"
    assert pending["dataset_materialization_expected_id"] is None
    assert pending["tokenized_dataset_expected_id"] is None
    assert "--expected-dataset-materialization-id" not in command
    assert "--expected-tokenized-dataset-id" not in command
    assert any(
        "dataset_materialization_contract=reissued" in line
        and "tokenized_dataset_contract=reissued" in line
        for line in st.hf_adapter_continuation_executor_lines(report)
    )


def test_executor_preserves_enforced_execution_input_contract(
    tmp_path: Path,
) -> None:
    execution_input_id = "sha256:" + "7" * 64
    _, child = _seed_chain(tmp_path, execution_input_id=execution_input_id)

    report = st.run_hf_adapter_continuation_executor(
        child,
        output_root=tmp_path / "executor-runs",
        max_lineage_depth=3,
        max_steps=2,
        max_train_samples=16,
    )
    pending = report["pending_generation"]
    contract = pending["execution_input_identity_contract"]
    command = pending["command"]["command"]

    assert contract["status"] == "enforced"
    assert contract["expected_identity_id"] == execution_input_id
    assert pending["execution_input_expected_id"] == execution_input_id
    assert _flag(command, "--expected-execution-input-id") == execution_input_id
    assert any(
        "execution_input_contract=enforced" in line
        and f"execution_input_expected={execution_input_id}" in line
        for line in st.hf_adapter_continuation_executor_lines(report)
    )


def test_executor_postflight_requires_ready_selected_transition(
    tmp_path: Path,
) -> None:
    root, child = _seed_chain(tmp_path)
    chain = st.hf_adapter_promotion_chain_report(child)
    parent_id = st.hf_adapter_fingerprint(root)["adapter_id"]

    ready = st.hf_adapter_executor._postflight_report(
        chain,
        output_dir=child,
        expected_parent_adapter_id=parent_id,
        expected_lineage_depth=1,
    )
    missing_chain = json.loads(json.dumps(chain))
    missing_chain["transitions"] = []
    missing = st.hf_adapter_executor._postflight_report(
        missing_chain,
        output_dir=child,
        expected_parent_adapter_id=parent_id,
        expected_lineage_depth=1,
    )
    tampered_chain = json.loads(json.dumps(chain))
    tampered_chain["transitions"][0]["transition_ready"] = False
    tampered = st.hf_adapter_executor._postflight_report(
        tampered_chain,
        output_dir=child,
        expected_parent_adapter_id=parent_id,
        expected_lineage_depth=1,
    )

    assert ready["ready"] is True
    assert ready["transition_ready"] is True
    assert ready["transition"]["parent_adapter_id"] == parent_id
    assert ready["transition"]["child_adapter_id"] == chain["selected_adapter_id"]
    assert missing["ready"] is False
    assert {"selected_transition", "transition_ready"}.issubset(
        missing["failed_checks"]
    )
    assert tampered["ready"] is False
    assert "transition_ready" in tampered["failed_checks"]


def test_executor_single_writer_lock_blocks_live_owner_and_reaps_stale_owner(
    tmp_path: Path,
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    output_root.mkdir()
    lock_path = output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
    lock_path.write_text(
        json.dumps(
            {
                "row_type": "hf_adapter_continuation_executor_lock",
                "lock_id": "live-owner",
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="is locked"):
        st.run_hf_adapter_continuation_executor(
            child,
            output_root=output_root,
            max_lineage_depth=2,
        )

    assert lock_path.is_file()
    lock_path.write_text(
        json.dumps(
            {
                "row_type": "hf_adapter_continuation_executor_lock",
                "lock_id": "stale-owner",
                "pid": 99_999_999,
                "hostname": socket.gethostname(),
            }
        ),
        encoding="utf-8",
    )
    report = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        max_lineage_depth=2,
    )

    assert report["status"] == "ready"
    assert not lock_path.exists()


def test_executor_blocks_until_pending_output_quarantine_is_finalized(
    tmp_path: Path,
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    state_path = output_root / "state.json"
    st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        max_lineage_depth=3,
    )
    state = st.load_hf_adapter_continuation_executor(state_path)
    attempt_id = "pending-quarantine-attempt"
    output_dir = output_root / "generation-002"
    quarantine_root = output_root.with_name(f"{output_root.name}.executor-quarantine")
    intent = {
        "row_type": "hf_adapter_continuation_executor_output_resolution_intent",
        "schema": st.HF_ADAPTER_CONTINUATION_EXECUTOR_OUTPUT_RESOLUTION_SCHEMA,
        "resolution_id": "pending-move",
        "run_id": state["run_id"],
        "reason": "interrupted quarantine",
        "attempt_id": attempt_id,
        "attempt_status": "cancelled",
        "lineage_depth": 2,
        "source_path": str(output_dir),
        "destination_path": str(quarantine_root / "generation-002-pending"),
        "quarantine_root": str(quarantine_root),
        "tree_snapshot": {"metadata_sha256": "pending"},
    }
    state["pending_output_resolution"] = dict(intent)
    state["generations"].append(
        {
            "attempt_id": attempt_id,
            "status": "cancelled",
            "lineage_depth": 2,
            "output_dir": str(output_dir),
            "pending_output_resolution": dict(intent),
        }
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")
    runner = FakeFineTuneRunner([0.05])

    blocked = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        run=True,
        max_lineage_depth=3,
        command_runner=runner,
    )

    assert blocked["status"] == "blocked"
    assert blocked["action"] == "complete_output_quarantine"
    assert blocked["reason"] == "output_quarantine_incomplete"
    assert blocked["output_resolution_gate"]["attempt_ids"] == [
        "pending-quarantine-attempt"
    ]
    assert runner.commands == []
    assert not (output_root / "generation-002").exists()


def test_executor_runs_multiple_generations_until_depth_policy_stops(
    tmp_path: Path,
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    runner = FakeFineTuneRunner([0.08, 0.07])

    report = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        run=True,
        max_generations=5,
        max_lineage_depth=3,
        min_eval_improvement=0.01,
        plateau_patience=2,
        command_runner=runner,
    )

    assert report["status"] == "stopped"
    assert report["reason"] == "continuation_policy_stop"
    assert report["continuation_policy"]["stop_reason_codes"] == [
        "max_lineage_depth_reached"
    ]
    assert report["generations_executed_this_invocation"] == 2
    assert report["promoted_generation_count"] == 2
    assert [row["status"] for row in report["generations"]] == [
        "promoted",
        "promoted",
    ]
    assert [row["lineage_depth"] for row in report["generations"]] == [2, 3]
    assert all(row["postflight"]["ready"] for row in report["generations"])
    assert report["transition_count"] == 3
    assert report["ready_transition_count"] == 3
    assert report["selected_path_transition_count"] == 3
    assert report["selected_path_transitions_ready"] is True
    postflight_transitions = [
        row["postflight"]["transition"] for row in report["generations"]
    ]
    assert all(row["transition_ready"] for row in postflight_transitions)
    assert [row["parent_adapter_id"] for row in postflight_transitions] == [
        report["generations"][0]["parent_adapter_id"],
        report["generations"][1]["parent_adapter_id"],
    ]
    assert [row["child_adapter_id"] for row in postflight_transitions] == [
        report["generations"][0]["adapter_id"],
        report["generations"][1]["adapter_id"],
    ]
    assert [row["child_lineage_depth"] for row in postflight_transitions] == [2, 3]
    assert [row["eval_handoff_delta"] for row in postflight_transitions] == (
        pytest.approx([0.0, -0.02])
    )
    assert [row["child_eval_improvement"] for row in postflight_transitions] == (
        pytest.approx([0.08, 0.07])
    )
    assert report["selected_transition"] == postflight_transitions[-1]
    assert report["generations"][1]["source_transition"] == (
        postflight_transitions[0]
    )
    generation_lines = [
        line
        for line in st.hf_adapter_continuation_executor_lines(report)
        if line.startswith("hf_adapter_continuation_executor_generation ")
    ]
    assert all("transition=ready" in line for line in generation_lines)
    assert (output_root / "generation-002" / st.HF_ADAPTER_LINEAGE_FILENAME).is_file()
    assert (output_root / "generation-003" / st.HF_ADAPTER_PROMOTION_FILENAME).is_file()
    assert len(runner.commands) == 2


def test_executor_flushes_subprocess_pid_and_progress_into_attempt_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, child = _seed_chain(tmp_path)
    runner = FakeFineTuneRunner([0.05], weight_prefix="observed")
    output_root = tmp_path / "executor-runs"
    lock_path = output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME

    def observed_execute(command: Sequence[str], **kwargs: object) -> int:
        assert kwargs["command_runner"] is None
        assert lock_path.is_file()
        kwargs["process_started"](4242)
        kwargs["process_progress"](512)
        return runner(command)

    monkeypatch.setattr(
        st.hf_adapter_executor,
        "_execute_command",
        observed_execute,
    )
    report = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        run=True,
        max_lineage_depth=2,
    )
    attempt = report["generations"][0]

    assert report["status"] == "stopped"
    assert attempt["runner_kind"] == "subprocess"
    assert attempt["hostname"] == socket.gethostname()
    assert attempt["pid"] == 4242
    assert attempt["log_bytes_observed"] == 512
    assert attempt["last_output_at"]
    assert attempt["process_started_at"]
    assert attempt["process_exited_at"]
    assert attempt["log_path"].endswith(".log")
    assert not lock_path.exists()


def test_executor_cooperatively_stops_silent_subprocess_and_blocks_partial_output(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    state_path = output_root / "state.json"
    partial_output = output_root / "generation-002"
    results: list[dict[str, object]] = []
    failures: list[BaseException] = []

    def scale_up_command(
        _chain: object,
        *,
        output_dir: Path,
        **_kwargs: object,
    ) -> dict[str, object]:
        script = (
            "from pathlib import Path; import time; "
            f"p=Path({str(output_dir)!r}); p.mkdir(parents=True); "
            "(p/'partial.txt').write_text('incomplete', encoding='utf-8'); "
            "time.sleep(60)"
        )
        return {
            "status": "ok",
            "command": [sys.executable, "-u", "-c", script],
        }

    monkeypatch.setattr(
        st.hf_adapter_executor,
        "hf_finetune_scale_up_command",
        scale_up_command,
    )
    monkeypatch.setattr(
        st.hf_adapter_executor,
        "hf_finetune_scale_up_preflight_report",
        lambda _command: {"status": "ready", "ready": True},
    )

    def run_executor() -> None:
        try:
            results.append(
                st.run_hf_adapter_continuation_executor(
                    child,
                    output_root=output_root,
                    state_path=state_path,
                    run=True,
                    max_lineage_depth=2,
                    tee_output=False,
                )
            )
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=run_executor, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if state_path.is_file() and partial_output.is_dir():
            live = json.loads(state_path.read_text(encoding="utf-8"))
            generations = live.get("generations") or []
            if generations and generations[-1].get("pid"):
                break
        time.sleep(0.05)
    else:
        pytest.fail("executor subprocess did not reach a live partial-output state")

    stop_code = hf_cli.adapter_continuation_executor_stop_main(
        [str(state_path), "--reason", "integration stop"]
    )
    stop_output = capsys.readouterr().out
    thread.join(timeout=15.0)

    assert not thread.is_alive()
    assert failures == []
    assert stop_code == 0
    assert "reason=integration stop" in stop_output
    assert len(results) == 1
    stopped = results[0]
    attempt = stopped["generations"][0]
    assert stopped["status"] == "stopped"
    assert stopped["reason"] == "stop_requested"
    assert attempt["status"] == "cancelled"
    assert attempt["process_group_isolated"] is True
    assert attempt["stop_scope"] == "process_group"
    assert attempt["stop_request"]["reason"] == "integration stop"
    assert attempt["returncode"] != 0
    assert partial_output.joinpath("partial.txt").is_file()
    assert not (
        output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
    ).exists()

    status = st.hf_adapter_continuation_executor_status_report(state_path)
    repeated_stop = st.request_hf_adapter_continuation_executor_stop(
        state_path,
        reason="ignored after completion",
    )
    blocked = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        max_lineage_depth=2,
    )
    blocked_code = hf_cli.adapter_continuation_executor_main(
        [
            str(child),
            "--output-root",
            str(output_root),
            "--state",
            str(state_path),
            "--max-lineage-depth",
            "2",
        ]
    )
    capsys.readouterr()

    assert status["status"] == "stopped"
    assert status["healthy"] is False
    assert status["recommended_action"] == "resolve_cancelled_output"
    assert "cancelled_output_present" in status["health_issues"]
    assert repeated_stop["created"] is False
    assert repeated_stop["request_id"] == stopped["stop_request"]["request_id"]
    assert repeated_stop["reason"] == "integration stop"
    assert blocked["status"] == "blocked"
    assert blocked_code == 1
    assert blocked["action"] == "resolve_failed_generation_output"
    assert blocked["unresolved_generation"]["attempt_status"] == "cancelled"
    assert blocked["stop_request_history"][0]["reason"] == "integration stop"

    quarantine = st.quarantine_hf_adapter_continuation_executor_output(
        state_path,
        attempt_id=attempt["attempt_id"],
        reason="integration quarantine",
    )
    recovered_status = st.hf_adapter_continuation_executor_status_report(state_path)
    resumed_plan = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        max_lineage_depth=2,
    )

    assert quarantine["created"] is True
    assert quarantine["reason"] == "integration quarantine"
    assert not partial_output.exists()
    assert Path(quarantine["destination_path"]).joinpath("partial.txt").is_file()
    assert recovered_status["status"] == "output_quarantined"
    assert recovered_status["healthy"] is True
    assert recovered_status["recommended_action"] == "resume_executor"
    assert resumed_plan["status"] == "ready"
    assert resumed_plan["action"] == "run_generation"


def test_executor_honors_stop_at_generation_boundary_after_promotion(
    tmp_path: Path,
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    state_path = output_root / "state.json"
    runner = FakeFineTuneRunner([0.05], weight_prefix="boundary")
    stop_reports: list[dict[str, object]] = []

    def stop_after_success(command: Sequence[str]) -> int:
        returncode = runner(command)
        stop_reports.append(
            st.request_hf_adapter_continuation_executor_stop(
                state_path,
                reason="stop after promoted generation",
            )
        )
        return returncode

    report = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        run=True,
        max_generations=2,
        max_lineage_depth=3,
        command_runner=stop_after_success,
    )

    assert report["status"] == "stopped"
    assert report["reason"] == "stop_requested"
    assert report["generations_executed_this_invocation"] == 1
    assert report["promoted_generation_count"] == 1
    assert report["generations"][0]["status"] == "promoted"
    assert report["generations"][0]["lineage_depth"] == 2
    assert report["stop_request"]["reason"] == "stop after promoted generation"
    assert stop_reports[0]["created"] is True
    assert not (output_root / "generation-003").exists()


def test_executor_honors_stop_during_preflight_without_launching(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    state_path = output_root / "state.json"
    runner = FakeFineTuneRunner([0.05], weight_prefix="must-not-launch")

    def request_stop(_command: Mapping[str, object]) -> dict[str, object]:
        st.request_hf_adapter_continuation_executor_stop(
            state_path,
            reason="stop during preflight",
        )
        return {"status": "ready", "ready": True}

    monkeypatch.setattr(
        st.hf_adapter_executor,
        "hf_finetune_scale_up_preflight_report",
        request_stop,
    )
    report = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        run=True,
        max_lineage_depth=2,
        command_runner=runner,
    )

    assert report["status"] == "stopped"
    assert report["reason"] == "stop_requested"
    assert report["stop_request"]["reason"] == "stop during preflight"
    assert report["generation_attempt_count"] == 0
    assert runner.commands == []
    assert not (output_root / "generation-002").exists()


def test_executor_stops_after_new_generation_completes_plateau(
    tmp_path: Path,
) -> None:
    _, child = _seed_chain(tmp_path, improvement=0.01)
    runner = FakeFineTuneRunner([0.005])

    report = st.run_hf_adapter_continuation_executor(
        child,
        output_root=tmp_path / "executor-runs",
        run=True,
        max_generations=4,
        min_eval_improvement=0.02,
        plateau_patience=2,
        command_runner=runner,
    )

    assert report["status"] == "stopped"
    assert report["generations_executed_this_invocation"] == 1
    assert report["promoted_generation_count"] == 1
    assert report["continuation_policy"]["stop_reason_codes"] == [
        "eval_improvement_plateau"
    ]
    assert report["continuation_policy"]["consecutive_below_min_eval_improvement"] == 2


def test_executor_records_failure_and_resumes_same_generation(
    tmp_path: Path,
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    state_path = output_root / "state.json"
    failed_runner = FakeFineTuneRunner([0.05], fail_returncode=7)

    failed = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        run=True,
        max_lineage_depth=2,
        command_runner=failed_runner,
    )
    resumed_runner = FakeFineTuneRunner([0.05])
    resumed = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        run=True,
        max_lineage_depth=2,
        command_runner=resumed_runner,
    )

    assert failed["status"] == "failed"
    assert failed["reason"] == "generation_command_returncode_7"
    assert failed["generations"][0]["returncode"] == 7
    assert resumed["status"] == "stopped"
    assert resumed["invocation_count"] == 2
    assert [row["status"] for row in resumed["generations"]] == [
        "failed",
        "promoted",
    ]
    assert resumed["promoted_generation_count"] == 1
    assert resumed["generations"][1]["lineage_depth"] == 2
    assert len(resumed_runner.commands) == 1


def test_executor_blocks_partial_output_left_by_failed_command(
    tmp_path: Path,
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    state_path = output_root / "state.json"

    def partial_failure(command: Sequence[str]) -> int:
        output = Path(_flag(command, "--output-dir"))
        output.mkdir(parents=True)
        (output / "partial.txt").write_text("incomplete", encoding="utf-8")
        return 9

    failed = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        run=True,
        max_lineage_depth=2,
        command_runner=partial_failure,
    )
    blocked = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        run=True,
        max_lineage_depth=2,
        command_runner=FakeFineTuneRunner([0.05]),
    )
    partial_output = output_root / "generation-002"
    shutil.rmtree(partial_output)
    resumed_runner = FakeFineTuneRunner([0.05], weight_prefix="after-partial")
    resumed = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        run=True,
        max_lineage_depth=2,
        command_runner=resumed_runner,
    )

    assert failed["status"] == "failed"
    assert blocked["status"] == "blocked"
    assert blocked["action"] == "resolve_failed_generation_output"
    assert blocked["unresolved_generation"]["output_dir"] == str(partial_output)
    assert resumed["status"] == "stopped"
    assert [row["status"] for row in resumed["generations"]] == [
        "failed",
        "promoted",
    ]
    assert len(resumed_runner.commands) == 1


def test_executor_resumes_after_per_invocation_generation_limit(
    tmp_path: Path,
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    state_path = output_root / "state.json"
    first_runner = FakeFineTuneRunner([0.06])

    first = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        run=True,
        max_generations=1,
        max_lineage_depth=3,
        command_runner=first_runner,
    )
    second_runner = FakeFineTuneRunner([0.05], weight_prefix="resumed")
    second = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        run=True,
        max_generations=1,
        max_lineage_depth=3,
        command_runner=second_runner,
    )

    assert first["status"] == "generation_limit_reached"
    assert first["selected_lineage_depth"] == 2
    assert first["promoted_generation_count"] == 1
    assert second["status"] == "stopped"
    assert second["selected_lineage_depth"] == 3
    assert second["invocation_count"] == 2
    assert second["promoted_generation_count"] == 2
    assert [row["lineage_depth"] for row in second["generations"]] == [2, 3]
    assert len(first_runner.commands) == 1
    assert len(second_runner.commands) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_generations": 1.5},
        {"max_lineage_depth": -1},
        {"target_eval_loss": float("nan")},
        {"plateau_patience": 0},
        {"max_steps_multiplier": 0.0},
        {"output_prefix": "../generation"},
    ],
)
def test_executor_rejects_invalid_configuration_before_writing_state(
    tmp_path: Path,
    kwargs: dict[str, object],
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    state_path = output_root / "state.json"

    with pytest.raises(ValueError):
        st.run_hf_adapter_continuation_executor(
            child,
            output_root=output_root,
            state_path=state_path,
            **kwargs,
        )

    assert not state_path.exists()
    assert not (
        output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
    ).exists()


@pytest.mark.parametrize(
    "reserved_name",
    [
        st.HF_ADAPTER_CONTINUATION_EXECUTOR_CONTROL_DIRNAME,
        st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME,
        st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOG_DIRNAME,
    ],
)
def test_executor_rejects_reserved_state_paths(
    tmp_path: Path,
    reserved_name: str,
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    reserved_path = output_root / reserved_name

    with pytest.raises(ValueError, match="state_path"):
        st.run_hf_adapter_continuation_executor(
            child,
            output_root=output_root,
            state_path=reserved_path,
        )

    assert not reserved_path.exists()
    assert not (
        output_root / st.HF_ADAPTER_CONTINUATION_EXECUTOR_LOCK_FILENAME
    ).exists()


def test_executor_fails_closed_on_interrupted_attempt_then_explicitly_retries(
    tmp_path: Path,
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    state_path = output_root / "state.json"
    planned = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        max_lineage_depth=2,
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    pending = planned["pending_generation"]
    state["generations"] = [
        {
            "attempt_id": "interrupted-test",
            "status": "running",
            "runner_kind": "subprocess",
            "process_group_isolated": True,
            "stop_scope": "process_group",
            "hostname": socket.gethostname(),
            "pid": 99_999_999,
            "parent_adapter_id": pending["parent_adapter_id"],
            "lineage_depth": pending["lineage_depth"],
            "output_dir": pending["output_dir"],
        }
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")

    blocked = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        max_lineage_depth=2,
    )
    runner = FakeFineTuneRunner([0.05], weight_prefix="retried")
    retried = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        run=True,
        retry_interrupted=True,
        max_lineage_depth=2,
        command_runner=runner,
    )

    assert blocked["status"] == "blocked"
    assert blocked["action"] == "audit_interrupted_generation"
    assert "unresolved" in blocked["reason"]
    assert retried["status"] == "stopped"
    assert [row["status"] for row in retried["generations"]] == [
        "interrupted_retry",
        "promoted",
    ]
    assert retried["promoted_generation_count"] == 1
    assert retried["generations"][0]["interruption_claim"]["ready"] is True
    assert len(runner.commands) == 1


def test_executor_refuses_retry_while_recorded_child_pid_is_alive(
    tmp_path: Path,
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    state_path = output_root / "state.json"
    planned = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        max_lineage_depth=2,
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    pending = planned["pending_generation"]
    state["generations"] = [
        {
            "attempt_id": "still-running-test",
            "status": "running",
            "runner_kind": "subprocess",
            "process_group_isolated": True,
            "stop_scope": "process_group",
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "parent_adapter_id": pending["parent_adapter_id"],
            "lineage_depth": pending["lineage_depth"],
            "output_dir": pending["output_dir"],
        }
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    runner = FakeFineTuneRunner([0.05], weight_prefix="must-not-run")

    blocked = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        run=True,
        retry_interrupted=True,
        max_lineage_depth=2,
        command_runner=runner,
    )

    assert blocked["status"] == "blocked"
    assert blocked["action"] == "audit_interrupted_generation"
    assert "still alive" in blocked["reason"]
    assert blocked["generations"][0]["process_liveness_observation"] == "alive"
    assert runner.commands == []


def test_executor_refuses_retry_while_interrupted_process_group_is_alive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    state_path = output_root / "state.json"
    planned = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        max_lineage_depth=2,
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    pending = planned["pending_generation"]
    state["generations"] = [
        {
            "attempt_id": "surviving-group-test",
            "status": "running",
            "runner_kind": "subprocess",
            "process_group_isolated": True,
            "stop_scope": "process_group",
            "hostname": socket.gethostname(),
            "pid": 42_424_242,
            "parent_adapter_id": pending["parent_adapter_id"],
            "lineage_depth": pending["lineage_depth"],
            "output_dir": pending["output_dir"],
        }
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(st.hf_adapter_executor, "local_pid_alive", lambda _pid: False)
    monkeypatch.setattr(
        st.hf_adapter_executor,
        "local_process_group_alive",
        lambda _process_group_id: True,
    )
    runner = FakeFineTuneRunner([0.05], weight_prefix="must-not-run")

    blocked = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        run=True,
        retry_interrupted=True,
        max_lineage_depth=2,
        command_runner=runner,
    )

    claim = blocked["generations"][0]["interruption_claim_observation"]
    assert blocked["status"] == "blocked"
    assert blocked["action"] == "audit_interrupted_generation"
    assert "attempt_process_group_alive" in blocked["reason"]
    assert claim["pid_alive_observed"] is False
    assert claim["process_group_alive_observed"] is True
    assert claim["ready"] is False
    assert runner.commands == []


def test_executor_recovers_completed_output_after_explicit_selection_interrupt(
    tmp_path: Path,
) -> None:
    _, child = _seed_chain(tmp_path)
    output_root = tmp_path / "executor-runs"
    state_path = output_root / "state.json"
    chain = st.hf_adapter_promotion_chain_report(child)
    selected_adapter_id = chain["selected_adapter_id"]
    planned = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        select_adapter_id=selected_adapter_id,
        max_lineage_depth=2,
    )
    pending = planned["pending_generation"]
    output_dir = Path(pending["output_dir"])
    command = pending["command"]["command"]
    _write_promoted_adapter(
        output_dir,
        child,
        weights=b"recovered-explicit-selection",
        before=0.9,
        after=0.84,
        launch_command=command,
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["generations"] = [
        {
            "attempt_id": "interrupted-after-command",
            "status": "running",
            "runner_kind": "subprocess",
            "process_group_isolated": True,
            "stop_scope": "process_group",
            "hostname": socket.gethostname(),
            "pid": 99_999_999,
            "parent_adapter_id": pending["parent_adapter_id"],
            "lineage_depth": pending["lineage_depth"],
            "output_dir": pending["output_dir"],
        }
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")

    recovered = st.run_hf_adapter_continuation_executor(
        child,
        output_root=output_root,
        state_path=state_path,
        select_adapter_id=selected_adapter_id,
        max_lineage_depth=2,
    )

    assert recovered["status"] == "stopped"
    assert recovered["selected_lineage_depth"] == 2
    assert recovered["promoted_generation_count"] == 1
    assert recovered["generations"][0]["status"] == "promoted_recovered"
    assert recovered["generations"][0]["interruption_claim"]["ready"] is True
    assert recovered["generations"][0]["postflight"]["ready"] is True
    assert recovered["generations"][0]["postflight"]["transition_ready"] is True
    assert recovered["selected_transition"] == (
        recovered["generations"][0]["postflight"]["transition"]
    )
