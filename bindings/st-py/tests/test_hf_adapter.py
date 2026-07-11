from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import spiraltorch as st
from spiraltorch import hf_cli


def _write_adapter(path: Path, weights: bytes, *, base: str = "org/base") -> Path:
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": base,
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


def _run_card(parent: Path, *, before: float = 1.0, after: float = 0.9) -> dict:
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
        "trainer_metrics": {"train_loss": 0.8},
        "eval_before_train": {
            "status": "ok",
            "eval_loss": before,
            "eval_perplexity": 2.718,
        },
        "eval_after_train": {
            "status": "ok",
            "eval_loss": after,
            "eval_perplexity": 2.46,
        },
    }


def _artifact_probe(adapter: Path, *, new_token_count: int = 4) -> dict:
    return {
        "row_type": "hf_causal_lm_artifact_probe",
        "status": "ready",
        "report_path": str(adapter / "spiraltorch-hf-artifact-probe.json"),
        "artifact": {
            "artifact_kind": "peft_adapter",
            "artifact_source": str(adapter.resolve()),
            "adapter_loaded": True,
        },
        "device": "cpu",
        "worker_pid": 2002,
        "new_token_count": new_token_count,
        "generated_text_changed": True,
        "local_files_only": True,
        "generation": {"do_sample": False},
        "process_isolation": {
            "schema": "spiraltorch.hf_artifact_probe_process.v1",
            "status": "ready",
            "fresh_process": True,
            "runner_kind": "python_module",
            "worker_module": "spiraltorch.hf_artifact_probe_worker",
            "parent_pid": 1001,
            "pid": 2002,
            "reported_worker_pid": 2002,
            "worker_pid_matches": True,
            "exit_code": 0,
            "timed_out": False,
            "duration_seconds": 0.25,
        },
    }


def _adapter_launch_command(
    parent: Path,
    child: Path,
    run_card_path: Path,
    *,
    max_train_samples: int = 8,
    block_size: int = 128,
) -> list[str]:
    bridge = Path(__file__).resolve().parents[1] / "examples" / "hf_finetune_bridge.py"
    return [
        sys.executable,
        str(bridge),
        "--model-name",
        str(parent),
        "--train",
        "--output-dir",
        str(child),
        "--run-card",
        str(run_card_path),
        "--trainer-trace-jsonl",
        str(child / "spiraltorch-hf-finetune-trainer-trace.jsonl"),
        "--finetune-mode",
        "lora",
        "--max-steps",
        "1",
        "--max-train-samples",
        str(max_train_samples),
        "--block-size",
        str(block_size),
    ]


def _write_promoted_child(
    child: Path,
    parent: Path,
    weights: bytes,
    *,
    launch_command: bool = True,
    before: float = 1.0,
    after: float = 0.9,
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
    finetune_replay_id: str | None = None,
    expected_finetune_replay_id: str | None = None,
    trainer_trace_summary: dict[str, object] | None = None,
    max_train_samples: int = 8,
    block_size: int = 128,
    dataset_revision: str = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",
) -> tuple[Path, dict, dict]:
    adapter = _write_adapter(child, weights)
    run_card_path = adapter / "spiraltorch-hf-finetune-run-card.json"
    card = _run_card(parent, before=before, after=after)
    if trainer_trace_summary is not None:
        card["trainer_trace_summary"] = dict(trainer_trace_summary)
    if runtime_input_id is not None:
        card["model_runtime_identity_pre_model"] = {
            "row_type": "hf_causal_lm_runtime_identity",
            "status": "ready",
            "phase": "pre_model_load",
            "expected_identity_id": expected_runtime_input_id,
            "observed_identity_id": runtime_input_id,
            "identity_verified": True,
        }
        card["model_runtime_identity_after_model"] = {
            "row_type": "hf_causal_lm_runtime_identity",
            "status": "ready",
            "phase": "after_model_load",
            "expected_identity_id": (
                expected_runtime_input_id or runtime_input_id
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
            "fail_fast": expected_runtime_input_id is not None,
        }
    if execution_input_id is not None:
        card["finetune_execution_identity_pre_model"] = {
            "row_type": "hf_finetune_execution_identity",
            "status": "ready",
            "phase": "pre_model_load",
            "expected_identity_id": expected_execution_input_id,
            "observed_identity_id": execution_input_id,
            "identity_verified": True,
        }
        card["finetune_execution_identity_after_model"] = {
            "row_type": "hf_finetune_execution_identity",
            "status": "ready",
            "phase": "after_model_load",
            "expected_identity_id": (
                expected_execution_input_id or execution_input_id
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
            "fail_fast": True,
        }
    if dataset_input_id is not None:
        card["dataset_input_identity"] = {
            "row_type": "hf_dataset_input_identity",
            "status": "ready",
            "phase": "preflight",
            "expected_identity_id": expected_dataset_input_id,
            "observed_identity_id": dataset_input_id,
            "effective_revision": dataset_revision,
            "identity_verified": True,
        }
        card["dataset_input_identity_after_load"] = {
            "row_type": "hf_dataset_input_identity",
            "status": "ready",
            "phase": "after_load",
            "expected_identity_id": expected_dataset_input_id or dataset_input_id,
            "observed_identity_id": dataset_input_id,
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
            "effective_revision": dataset_revision,
            "identity_verified": True,
            "fail_fast": True,
        }
    if dataset_materialization_id is not None:
        card["dataset_materialization_identity"] = {
            "row_type": "hf_dataset_materialization_identity",
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
    if tokenized_dataset_id is not None:
        card["tokenized_dataset_identity"] = {
            "row_type": "hf_tokenized_dataset_identity",
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
    if finetune_replay_id is not None:
        card["finetune_replay_identity"] = {
            "row_type": "hf_finetune_replay_identity",
            "status": "ready",
            "phase": "before_trainer_init",
            "expected_identity_id": expected_finetune_replay_id,
            "observed_identity_id": finetune_replay_id,
            "identity_verified": True,
            "component_count": 8,
            "applicable_component_count": 7,
            "ready_component_count": 7,
        }
        card["finetune_replay_identity_contract"] = {
            "status": (
                "enforced"
                if expected_finetune_replay_id is not None
                else "adopted"
            ),
            "expected_identity_id": (
                expected_finetune_replay_id or finetune_replay_id
            ),
            "observed_identity_id": finetune_replay_id,
            "identity_verified": True,
            "fail_fast": True,
        }
    if launch_command:
        card["launch_command"] = _adapter_launch_command(
            parent,
            adapter,
            run_card_path,
            max_train_samples=max_train_samples,
            block_size=block_size,
        )
        if dataset_materialization_id is not None:
            card["launch_command"].extend(
                [
                    "--expected-dataset-materialization-id",
                    expected_dataset_materialization_id
                    or dataset_materialization_id,
                ]
            )
        if tokenized_dataset_id is not None:
            card["launch_command"].extend(
                [
                    "--expected-tokenized-dataset-id",
                    expected_tokenized_dataset_id or tokenized_dataset_id,
                ]
            )
        if finetune_replay_id is not None:
            card["launch_command"].extend(
                [
                    "--expected-finetune-replay-id",
                    expected_finetune_replay_id or finetune_replay_id,
                ]
            )
        card["launch_command_display"] = " ".join(card["launch_command"])
        card["launch_command_source"] = "test"
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
    return adapter, lineage, promotion


def test_adapter_fingerprint_is_path_independent_and_weight_sensitive(
    tmp_path: Path,
) -> None:
    first = _write_adapter(tmp_path / "first", b"adapter-a")
    copied = tmp_path / "copied"
    shutil.copytree(first, copied)

    first_report = st.hf_adapter_fingerprint(first)
    copied_report = st.hf_adapter_fingerprint(copied)
    (copied / "adapter_model.safetensors").write_bytes(b"adapter-b")
    changed_report = st.hf_adapter_fingerprint(copied)

    assert first_report["adapter_id"] == copied_report["adapter_id"]
    assert first_report["adapter_id"] != changed_report["adapter_id"]
    assert first_report["adapter_weight_file_count"] == 1
    assert first_report["base_model_name_or_path"] == "org/base"


def test_adapter_fingerprint_includes_shards_referenced_by_index(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "sharded"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "org/base",
                "peft_type": "LORA",
            }
        ),
        encoding="utf-8",
    )
    first_shard = adapter / "adapter_model-00001-of-00002.safetensors"
    second_shard = adapter / "adapter_model-00002-of-00002.safetensors"
    first_shard.write_bytes(b"first")
    second_shard.write_bytes(b"second")
    (adapter / "adapter_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.a": first_shard.name,
                    "layer.b": second_shard.name,
                }
            }
        ),
        encoding="utf-8",
    )

    before = st.hf_adapter_fingerprint(adapter)
    second_shard.write_bytes(b"changed")
    after = st.hf_adapter_fingerprint(adapter)

    assert before["adapter_weight_file_count"] == 3
    assert before["adapter_id"] != after["adapter_id"]


def test_adapter_input_identity_verifies_expected_lineage_and_detects_tamper(
    tmp_path: Path,
) -> None:
    adapter = _write_adapter(tmp_path / "adapter", b"adapter")
    lineage = st.write_hf_adapter_lineage(adapter)

    ready = st.hf_adapter_input_identity_report(
        adapter,
        expected_adapter_id=lineage["adapter_id"],
        expected_lineage_depth=0,
        expected_root_adapter_id=lineage["root_adapter_id"],
        require_lineage=True,
    )

    assert ready["status"] == "ready"
    assert ready["identity_verified"] is True
    assert ready["expected_identity_verified"] is True
    assert ready["lineage_fingerprint_verified"] is True
    assert ready["lineage_depth_verified"] is True
    assert ready["root_adapter_verified"] is True
    assert "verified=True" in st.hf_adapter_input_identity_lines(ready)[0]

    (adapter / "adapter_model.safetensors").write_bytes(b"tampered")
    blocked = st.hf_adapter_input_identity_report(
        adapter,
        expected_adapter_id=lineage["adapter_id"],
        expected_lineage_depth=0,
        expected_root_adapter_id=lineage["root_adapter_id"],
        require_lineage=True,
    )

    assert blocked["status"] == "blocked"
    assert blocked["identity_verified"] is False
    assert blocked["expected_identity_verified"] is False
    assert blocked["lineage_fingerprint_verified"] is False
    assert set(blocked["errors"]) == {
        "adapter fingerprint does not match expected adapter id",
        "adapter lineage fingerprint does not match adapter files",
    }

    with pytest.raises(ValueError, match="sha256"):
        st.hf_adapter_input_identity_report(
            adapter,
            expected_adapter_id="not-an-adapter-id",
        )


def test_adapter_lineage_chains_parent_and_run_card_identity(tmp_path: Path) -> None:
    parent = _write_adapter(tmp_path / "parent", b"parent")
    child = _write_adapter(tmp_path / "child", b"child")
    root = st.write_hf_adapter_lineage(parent)
    card = _run_card(parent)

    child_lineage = st.write_hf_adapter_lineage(
        child,
        parent_adapter=parent,
        run_card=card,
        run_card_path=tmp_path / "run-card.json",
    )
    loaded = st.load_hf_adapter_lineage(child)

    assert root["lineage_depth"] == 0
    assert child_lineage["lineage_depth"] == 1
    assert child_lineage["root_adapter_id"] == root["adapter_id"]
    assert child_lineage["parent_adapter_id"] == root["adapter_id"]
    assert child_lineage["parent_fingerprint_verified"] is True
    assert child_lineage["weights_changed_from_parent"] is True
    assert child_lineage["run_card_sha256"]
    assert loaded["adapter_id"] == child_lineage["adapter_id"]
    assert "depth=1" in st.hf_adapter_lineage_lines(loaded)[0]
    with pytest.raises(ValueError, match="parent directories must differ"):
        st.write_hf_adapter_lineage(parent, parent_adapter=parent)


def test_adapter_lineage_and_transition_require_declared_input_identity(
    tmp_path: Path,
) -> None:
    parent = _write_adapter(tmp_path / "parent", b"parent")
    parent_lineage = st.write_hf_adapter_lineage(parent)
    child = _write_adapter(tmp_path / "child", b"child")
    card = _run_card(parent)
    card["adapter_input_identity"] = st.hf_adapter_input_identity_report(
        parent,
        expected_adapter_id=parent_lineage["adapter_id"],
        expected_lineage_depth=0,
        expected_root_adapter_id=parent_lineage["root_adapter_id"],
        require_lineage=True,
        phase="preflight",
    )
    card["adapter_input_identity_after_load"] = (
        st.hf_adapter_input_identity_report(
            parent,
            expected_adapter_id=parent_lineage["adapter_id"],
            expected_lineage_depth=0,
            expected_root_adapter_id=parent_lineage["root_adapter_id"],
            require_lineage=True,
            phase="after_load",
        )
    )
    train_file = tmp_path / "train.txt"
    train_file.write_text("spiral training\n", encoding="utf-8")
    training_input_preflight = st.hf_finetune_input_identity_report(
        train_files=[train_file],
        phase="preflight",
    )
    card["training_input_identity"] = training_input_preflight
    card["training_input_identity_after_load"] = (
        st.hf_finetune_input_identity_report(
            train_files=[train_file],
            expected_input_id=training_input_preflight["observed_input_id"],
            phase="after_load",
        )
    )
    run_card_path = child / "spiraltorch-hf-finetune-run-card.json"
    lineage = st.write_hf_adapter_lineage(
        child,
        parent_adapter=parent,
        run_card=card,
        run_card_path=run_card_path,
    )
    card["adapter_lineage"] = lineage
    promotion = st.write_hf_adapter_promotion(
        child,
        card,
        parent_adapter=parent,
    )
    card["adapter_promotion"] = promotion
    run_card_path.write_text(json.dumps(card), encoding="utf-8")

    chain = st.hf_adapter_promotion_chain_report(tmp_path)
    transition = chain["transitions"][0]

    assert lineage["parent_input_identity_present"] is True
    assert lineage["parent_input_identity_verified"] is True
    assert lineage["parent_input_expected_adapter_id"] == parent_lineage["adapter_id"]
    assert transition["status"] == "ready"
    assert transition["input_identity_required"] is True
    assert transition["input_identity_ready"] is True
    assert transition["input_identity_preflight_status"] == "ready"
    assert transition["input_identity_after_load_status"] == "ready"
    assert lineage["training_input_identity_required"] is True
    assert lineage["training_input_identity_verified"] is True
    assert transition["training_input_identity_required"] is True
    assert transition["training_input_identity_ready"] is True
    assert transition["training_input_preflight_status"] == "ready"
    assert transition["training_input_after_load_status"] == "ready"
    assert transition["training_input_continuity_observed"] is False

    bad_card = _run_card(parent)
    bad_card["adapter_input_identity"] = dict(card["adapter_input_identity"])
    bad_card["adapter_input_identity"]["observed_adapter_id"] = (
        "sha256:" + "0" * 64
    )
    with pytest.raises(ValueError, match="input identity does not match parent"):
        st.write_hf_adapter_lineage(
            child,
            parent_adapter=parent,
            run_card=bad_card,
        )

    bad_training_card = json.loads(json.dumps(card))
    bad_training_card["training_input_identity_after_load"]["status"] = "blocked"
    with pytest.raises(ValueError, match="after-load identity is not ready"):
        st.write_hf_adapter_lineage(
            child,
            parent_adapter=parent,
            run_card=bad_training_card,
        )


def test_adapter_chain_adopts_then_requires_runtime_input_identity(
    tmp_path: Path,
) -> None:
    runtime_input_id = "sha256:" + "7" * 64
    root = _write_adapter(tmp_path / "root", b"root")
    st.write_hf_adapter_lineage(root)
    adopted, adopted_lineage, _ = _write_promoted_child(
        tmp_path / "adopted",
        root,
        b"adopted",
        runtime_input_id=runtime_input_id,
    )
    enforced, enforced_lineage, _ = _write_promoted_child(
        tmp_path / "enforced",
        adopted,
        b"enforced",
        runtime_input_id=runtime_input_id,
        expected_runtime_input_id=runtime_input_id,
    )
    _, dropped_lineage, _ = _write_promoted_child(
        tmp_path / "dropped",
        enforced,
        b"dropped",
        runtime_input_id=runtime_input_id,
    )

    chain = st.hf_adapter_promotion_chain_report(tmp_path)
    root_to_adopted, adopted_to_enforced, enforced_to_dropped = chain[
        "transitions"
    ]

    assert adopted_lineage["runtime_input_identity_required"] is False
    assert adopted_lineage["runtime_input_identity_verified"] is True
    assert adopted_lineage["runtime_input_observed_id"] == runtime_input_id
    assert root_to_adopted["runtime_input_identity_required"] is False
    assert root_to_adopted["runtime_input_identity_ready"] is True
    assert root_to_adopted["runtime_input_matches_parent"] is None
    assert root_to_adopted["status"] == "ready"

    assert enforced_lineage["runtime_input_identity_required"] is True
    assert enforced_lineage["runtime_input_expected_id"] == runtime_input_id
    assert adopted_to_enforced["runtime_input_identity_required"] is True
    assert adopted_to_enforced["runtime_input_identity_ready"] is True
    assert adopted_to_enforced["runtime_input_matches_parent"] is True
    assert adopted_to_enforced["status"] == "ready"

    assert dropped_lineage["runtime_input_identity_required"] is False
    assert enforced_to_dropped["runtime_input_identity_required"] is True
    assert enforced_to_dropped["runtime_input_identity_ready"] is False
    assert enforced_to_dropped["runtime_input_expected_id"] is None
    assert enforced_to_dropped["runtime_input_matches_parent"] is True
    assert enforced_to_dropped["lineage_ready"] is False
    assert enforced_to_dropped["status"] == "rejected"
    transition_lines = [
        line
        for line in st.hf_adapter_promotion_chain_lines(chain)
        if line.startswith("hf_adapter_promotion_chain_transition ")
    ]
    assert "runtime_input_ready=False" in transition_lines[-1]

    incomplete = _write_adapter(tmp_path / "incomplete", b"incomplete")
    incomplete_card = _run_card(enforced)
    incomplete_card["model_runtime_identity_after_model"] = {
        "status": "ready",
        "phase": "after_model_load",
        "expected_identity_id": runtime_input_id,
        "observed_identity_id": runtime_input_id,
        "identity_verified": True,
    }
    incomplete_card["model_runtime_identity_contract"] = {
        "status": "adopted",
        "expected_identity_id": None,
        "observed_identity_id": runtime_input_id,
        "identity_verified": True,
    }
    with pytest.raises(ValueError, match="pre-model identity is not ready"):
        st.write_hf_adapter_lineage(
            incomplete,
            parent_adapter=enforced,
            run_card=incomplete_card,
        )


def test_adapter_chain_adopts_then_requires_dataset_input_identity(
    tmp_path: Path,
) -> None:
    dataset_input_id = "sha256:" + "6" * 64
    root = _write_adapter(tmp_path / "root", b"root")
    st.write_hf_adapter_lineage(root)
    adopted, adopted_lineage, _ = _write_promoted_child(
        tmp_path / "adopted",
        root,
        b"adopted",
        dataset_input_id=dataset_input_id,
    )
    enforced, enforced_lineage, _ = _write_promoted_child(
        tmp_path / "enforced",
        adopted,
        b"enforced",
        dataset_input_id=dataset_input_id,
        expected_dataset_input_id=dataset_input_id,
    )
    _, dropped_lineage, _ = _write_promoted_child(
        tmp_path / "dropped",
        enforced,
        b"dropped",
    )

    chain = st.hf_adapter_promotion_chain_report(tmp_path)
    root_to_adopted, adopted_to_enforced, enforced_to_dropped = chain[
        "transitions"
    ]

    assert adopted_lineage["dataset_input_identity_verified"] is True
    assert adopted_lineage["dataset_input_observed_id"] == dataset_input_id
    assert root_to_adopted["dataset_input_identity_ready"] is True
    assert root_to_adopted["dataset_input_adopted"] is True
    assert root_to_adopted["status"] == "ready"

    assert enforced_lineage["dataset_input_identity_required"] is True
    assert enforced_lineage["dataset_input_expected_id"] == dataset_input_id
    assert adopted_to_enforced["dataset_input_identity_ready"] is True
    assert adopted_to_enforced["dataset_input_matches_parent"] is True
    assert adopted_to_enforced["status"] == "ready"

    assert dropped_lineage["dataset_input_identity_present"] is False
    assert enforced_to_dropped["dataset_input_identity_required"] is True
    assert enforced_to_dropped["dataset_input_identity_ready"] is False
    assert enforced_to_dropped["status"] == "rejected"
    transition_lines = [
        line
        for line in st.hf_adapter_promotion_chain_lines(chain)
        if line.startswith("hf_adapter_promotion_chain_transition ")
    ]
    assert "dataset_input_ready=False" in transition_lines[-1]


def test_adapter_chain_adopts_then_requires_dataset_materialization_identity(
    tmp_path: Path,
) -> None:
    materialization_id = "sha256:" + "7" * 64
    root = _write_adapter(tmp_path / "root", b"root")
    st.write_hf_adapter_lineage(root)
    adopted, adopted_lineage, _ = _write_promoted_child(
        tmp_path / "adopted",
        root,
        b"adopted",
        dataset_materialization_id=materialization_id,
    )
    enforced, enforced_lineage, _ = _write_promoted_child(
        tmp_path / "enforced",
        adopted,
        b"enforced",
        dataset_materialization_id=materialization_id,
        expected_dataset_materialization_id=materialization_id,
    )
    _, dropped_lineage, _ = _write_promoted_child(
        tmp_path / "dropped",
        enforced,
        b"dropped",
    )

    chain = st.hf_adapter_promotion_chain_report(tmp_path)
    root_to_adopted, adopted_to_enforced, enforced_to_dropped = chain[
        "transitions"
    ]

    assert adopted_lineage["dataset_materialization_identity_verified"] is True
    assert (
        adopted_lineage["dataset_materialization_observed_id"]
        == materialization_id
    )
    assert root_to_adopted["dataset_materialization_identity_ready"] is True
    assert root_to_adopted["dataset_materialization_adopted"] is True
    assert root_to_adopted["status"] == "ready"

    assert enforced_lineage["dataset_materialization_identity_required"] is True
    assert (
        enforced_lineage["dataset_materialization_expected_id"]
        == materialization_id
    )
    assert adopted_to_enforced["dataset_materialization_identity_ready"] is True
    assert adopted_to_enforced["dataset_materialization_matches_parent"] is True
    assert adopted_to_enforced["status"] == "ready"

    assert dropped_lineage["dataset_materialization_identity_present"] is False
    assert enforced_to_dropped["dataset_materialization_identity_required"] is True
    assert enforced_to_dropped["dataset_materialization_identity_ready"] is False
    assert enforced_to_dropped["status"] == "rejected"
    transition_lines = [
        line
        for line in st.hf_adapter_promotion_chain_lines(chain)
        if line.startswith("hf_adapter_promotion_chain_transition ")
    ]
    assert "dataset_materialization_ready=False" in transition_lines[-1]


def test_adapter_chain_adopts_then_requires_tokenized_dataset_identity(
    tmp_path: Path,
) -> None:
    tokenized_dataset_id = "sha256:" + "a" * 64
    root = _write_adapter(tmp_path / "root", b"root")
    st.write_hf_adapter_lineage(root)
    adopted, adopted_lineage, _ = _write_promoted_child(
        tmp_path / "adopted",
        root,
        b"adopted",
        tokenized_dataset_id=tokenized_dataset_id,
    )
    enforced, enforced_lineage, _ = _write_promoted_child(
        tmp_path / "enforced",
        adopted,
        b"enforced",
        tokenized_dataset_id=tokenized_dataset_id,
        expected_tokenized_dataset_id=tokenized_dataset_id,
    )
    _, dropped_lineage, _ = _write_promoted_child(
        tmp_path / "dropped",
        enforced,
        b"dropped",
    )

    chain = st.hf_adapter_promotion_chain_report(tmp_path)
    root_to_adopted, adopted_to_enforced, enforced_to_dropped = chain[
        "transitions"
    ]

    assert adopted_lineage["tokenized_dataset_identity_verified"] is True
    assert adopted_lineage["tokenized_dataset_observed_id"] == tokenized_dataset_id
    assert root_to_adopted["tokenized_dataset_identity_ready"] is True
    assert root_to_adopted["tokenized_dataset_adopted"] is True
    assert root_to_adopted["status"] == "ready"

    assert enforced_lineage["tokenized_dataset_identity_required"] is True
    assert enforced_lineage["tokenized_dataset_expected_id"] == tokenized_dataset_id
    assert adopted_to_enforced["tokenized_dataset_identity_ready"] is True
    assert adopted_to_enforced["tokenized_dataset_matches_parent"] is True
    assert adopted_to_enforced["status"] == "ready"

    assert dropped_lineage["tokenized_dataset_identity_present"] is False
    assert enforced_to_dropped["tokenized_dataset_identity_required"] is True
    assert enforced_to_dropped["tokenized_dataset_identity_ready"] is False
    assert enforced_to_dropped["status"] == "rejected"
    transition_lines = [
        line
        for line in st.hf_adapter_promotion_chain_lines(chain)
        if line.startswith("hf_adapter_promotion_chain_transition ")
    ]
    assert "tokenized_dataset_ready=False" in transition_lines[-1]


def test_adapter_chain_reissues_dataset_identities_for_explicit_shape_change(
    tmp_path: Path,
) -> None:
    parent_materialization_id = "sha256:" + "5" * 64
    child_materialization_id = "sha256:" + "6" * 64
    parent_tokenized_id = "sha256:" + "a" * 64
    child_tokenized_id = "sha256:" + "b" * 64
    root = _write_adapter(tmp_path / "root", b"root")
    st.write_hf_adapter_lineage(root)
    parent, _, _ = _write_promoted_child(
        tmp_path / "parent",
        root,
        b"parent",
        dataset_materialization_id=parent_materialization_id,
        tokenized_dataset_id=parent_tokenized_id,
        max_train_samples=8,
    )
    _, child_lineage, _ = _write_promoted_child(
        tmp_path / "child",
        parent,
        b"child",
        dataset_materialization_id=child_materialization_id,
        expected_dataset_materialization_id=child_materialization_id,
        tokenized_dataset_id=child_tokenized_id,
        expected_tokenized_dataset_id=child_tokenized_id,
        max_train_samples=16,
    )

    chain = st.hf_adapter_promotion_chain_report(tmp_path)
    transition = chain["transitions"][-1]

    assert child_lineage["dataset_materialization_observed_id"] == (
        child_materialization_id
    )
    assert child_lineage["tokenized_dataset_observed_id"] == child_tokenized_id
    assert transition["status"] == "ready"
    assert transition["dataset_shape_reissued"] is True
    assert transition["dataset_materialization_reissued"] is True
    assert transition["tokenized_dataset_reissued"] is True
    assert transition["dataset_materialization_matches_parent"] is False
    assert transition["tokenized_dataset_matches_parent"] is False
    assert transition["dataset_shape_changes"] == [
        {
            "flag": "--max-train-samples",
            "parent_value": "8",
            "child_value": "16",
        }
    ]


def test_adapter_chain_reissues_only_tokenized_identity_for_block_size_change(
    tmp_path: Path,
) -> None:
    materialization_id = "sha256:" + "5" * 64
    parent_tokenized_id = "sha256:" + "a" * 64
    child_tokenized_id = "sha256:" + "b" * 64
    root = _write_adapter(tmp_path / "root", b"root")
    st.write_hf_adapter_lineage(root)
    parent, _, _ = _write_promoted_child(
        tmp_path / "parent",
        root,
        b"parent",
        dataset_materialization_id=materialization_id,
        tokenized_dataset_id=parent_tokenized_id,
        block_size=128,
    )
    _write_promoted_child(
        tmp_path / "child",
        parent,
        b"child",
        dataset_materialization_id=materialization_id,
        expected_dataset_materialization_id=materialization_id,
        tokenized_dataset_id=child_tokenized_id,
        expected_tokenized_dataset_id=child_tokenized_id,
        block_size=256,
    )

    transition = st.hf_adapter_promotion_chain_report(tmp_path)["transitions"][-1]

    assert transition["status"] == "ready"
    assert transition["dataset_shape_reissued"] is True
    assert transition["dataset_materialization_shape_reissued"] is False
    assert transition["dataset_materialization_reissued"] is False
    assert transition["dataset_materialization_matches_parent"] is True
    assert transition["tokenized_dataset_shape_reissued"] is True
    assert transition["tokenized_dataset_reissued"] is True
    assert transition["tokenized_dataset_matches_parent"] is False
    assert transition["dataset_shape_changes"] == [
        {
            "flag": "--block-size",
            "parent_value": "128",
            "child_value": "256",
        }
    ]


def test_adapter_chain_adopts_then_requires_execution_input_identity(
    tmp_path: Path,
) -> None:
    execution_input_id = "sha256:" + "8" * 64
    root = _write_adapter(tmp_path / "root", b"root")
    st.write_hf_adapter_lineage(root)
    adopted, adopted_lineage, _ = _write_promoted_child(
        tmp_path / "adopted",
        root,
        b"adopted",
        execution_input_id=execution_input_id,
    )
    enforced, enforced_lineage, _ = _write_promoted_child(
        tmp_path / "enforced",
        adopted,
        b"enforced",
        execution_input_id=execution_input_id,
        expected_execution_input_id=execution_input_id,
    )
    _, dropped_lineage, _ = _write_promoted_child(
        tmp_path / "dropped",
        enforced,
        b"dropped",
        execution_input_id=execution_input_id,
    )

    chain = st.hf_adapter_promotion_chain_report(tmp_path)
    root_to_adopted, adopted_to_enforced, enforced_to_dropped = chain[
        "transitions"
    ]

    assert adopted_lineage["execution_input_identity_required"] is False
    assert adopted_lineage["execution_input_identity_verified"] is True
    assert adopted_lineage["execution_input_observed_id"] == execution_input_id
    assert root_to_adopted["execution_input_identity_required"] is False
    assert root_to_adopted["execution_input_identity_ready"] is True
    assert root_to_adopted["execution_input_adopted"] is True
    assert root_to_adopted["execution_input_matches_parent"] is None
    assert root_to_adopted["status"] == "ready"

    assert enforced_lineage["execution_input_identity_required"] is True
    assert enforced_lineage["execution_input_expected_id"] == execution_input_id
    assert adopted_to_enforced["execution_input_identity_required"] is True
    assert adopted_to_enforced["execution_input_identity_ready"] is True
    assert adopted_to_enforced["execution_input_matches_parent"] is True
    assert adopted_to_enforced["status"] == "ready"

    assert dropped_lineage["execution_input_identity_required"] is False
    assert enforced_to_dropped["execution_input_identity_required"] is True
    assert enforced_to_dropped["execution_input_identity_ready"] is False
    assert enforced_to_dropped["execution_input_expected_id"] is None
    assert enforced_to_dropped["execution_input_matches_parent"] is True
    assert enforced_to_dropped["lineage_ready"] is False
    assert enforced_to_dropped["status"] == "rejected"
    transition_lines = [
        line
        for line in st.hf_adapter_promotion_chain_lines(chain)
        if line.startswith("hf_adapter_promotion_chain_transition ")
    ]
    assert "execution_input_ready=False" in transition_lines[-1]


def test_adapter_chain_reissues_finetune_replay_identity_per_generation(
    tmp_path: Path,
) -> None:
    adopted_replay_id = "sha256:" + "9" * 64
    reissued_replay_id = "sha256:" + "a" * 64
    root = _write_adapter(tmp_path / "root", b"root")
    st.write_hf_adapter_lineage(root)
    adopted, adopted_lineage, adopted_promotion = _write_promoted_child(
        tmp_path / "adopted",
        root,
        b"adopted",
        finetune_replay_id=adopted_replay_id,
    )
    enforced, enforced_lineage, enforced_promotion = _write_promoted_child(
        tmp_path / "enforced",
        adopted,
        b"enforced",
        finetune_replay_id=reissued_replay_id,
        expected_finetune_replay_id=reissued_replay_id,
    )
    _, dropped_lineage, _ = _write_promoted_child(
        tmp_path / "dropped",
        enforced,
        b"dropped",
    )

    chain = st.hf_adapter_promotion_chain_report(tmp_path)
    root_to_adopted, adopted_to_enforced, enforced_to_dropped = chain[
        "transitions"
    ]

    assert adopted_lineage["finetune_replay_identity_required"] is True
    assert adopted_lineage["finetune_replay_identity_verified"] is True
    assert adopted_lineage["finetune_replay_observed_id"] == adopted_replay_id
    assert adopted_promotion["finetune_replay_identity_verified"] is True
    assert root_to_adopted["finetune_replay_identity_required"] is True
    assert root_to_adopted["finetune_replay_identity_ready"] is True
    assert root_to_adopted["finetune_replay_identity_adopted"] is True
    assert root_to_adopted["finetune_replay_identity_reissued"] is False

    assert enforced_lineage["finetune_replay_identity_contract_status"] == (
        "enforced"
    )
    assert enforced_promotion["finetune_replay_observed_id"] == reissued_replay_id
    assert adopted_to_enforced["finetune_replay_identity_required"] is True
    assert adopted_to_enforced["finetune_replay_identity_ready"] is True
    assert adopted_to_enforced["finetune_replay_identity_reissued"] is True
    assert adopted_to_enforced["finetune_replay_matches_parent"] is False

    assert dropped_lineage["finetune_replay_identity_present"] is False
    assert enforced_to_dropped["finetune_replay_identity_required"] is True
    assert enforced_to_dropped["finetune_replay_identity_ready"] is False
    assert enforced_to_dropped["lineage_ready"] is False
    transition_lines = [
        line
        for line in st.hf_adapter_promotion_chain_lines(chain)
        if line.startswith("hf_adapter_promotion_chain_transition ")
    ]
    assert "finetune_replay_reissued=True" in transition_lines[1]
    assert "finetune_replay_ready=False" in transition_lines[2]


def test_adapter_chain_rejects_reused_parent_finetune_replay_identity(
    tmp_path: Path,
) -> None:
    replay_id = "sha256:" + "9" * 64
    root = _write_adapter(tmp_path / "root", b"root")
    st.write_hf_adapter_lineage(root)
    adopted, _, _ = _write_promoted_child(
        tmp_path / "adopted",
        root,
        b"adopted",
        finetune_replay_id=replay_id,
    )
    _write_promoted_child(
        tmp_path / "reused",
        adopted,
        b"reused",
        finetune_replay_id=replay_id,
        expected_finetune_replay_id=replay_id,
    )

    chain = st.hf_adapter_promotion_chain_report(tmp_path)
    reused_transition = chain["transitions"][1]

    assert reused_transition["finetune_replay_identity_required"] is True
    assert reused_transition["finetune_replay_identity_reissued"] is False
    assert reused_transition["finetune_replay_matches_parent"] is True
    assert reused_transition["finetune_replay_identity_ready"] is False
    assert reused_transition["lineage_ready"] is False
    assert reused_transition["status"] == "rejected"


def test_adapter_lineage_rejects_mismatched_finetune_replay_contract(
    tmp_path: Path,
) -> None:
    root = _write_adapter(tmp_path / "root", b"root")
    st.write_hf_adapter_lineage(root)

    with pytest.raises(ValueError, match="fine-tune replay identity does not match"):
        _write_promoted_child(
            tmp_path / "child",
            root,
            b"child",
            finetune_replay_id="sha256:" + "c" * 64,
            expected_finetune_replay_id="sha256:" + "d" * 64,
        )


def test_adapter_lineage_rejects_finetune_replay_without_final_contract(
    tmp_path: Path,
) -> None:
    replay_id = "sha256:" + "c" * 64
    root = _write_adapter(tmp_path / "root", b"root")
    child = _write_adapter(tmp_path / "child", b"child")
    st.write_hf_adapter_lineage(root)
    card = _run_card(root)
    card["finetune_replay_identity"] = {
        "row_type": "hf_finetune_replay_identity",
        "status": "ready",
        "phase": "before_trainer_init",
        "expected_identity_id": None,
        "observed_identity_id": replay_id,
        "identity_verified": True,
        "component_count": 8,
        "applicable_component_count": 7,
        "ready_component_count": 7,
    }

    with pytest.raises(ValueError, match="fine-tune replay contract is not final"):
        st.write_hf_adapter_lineage(
            child,
            parent_adapter=root,
            run_card=card,
        )


def test_adapter_promotion_requires_lineage_weight_change_and_eval(
    tmp_path: Path,
) -> None:
    parent = _write_adapter(tmp_path / "parent", b"parent")
    child = _write_adapter(tmp_path / "child", b"child")
    card = _run_card(parent, before=1.0, after=0.9)
    lineage = st.write_hf_adapter_lineage(
        child,
        parent_adapter=parent,
        run_card=card,
    )
    card["adapter_lineage"] = lineage

    promoted = st.write_hf_adapter_promotion(
        child,
        card,
        parent_adapter=parent,
    )
    stricter = st.hf_adapter_promotion_report(
        child,
        card,
        parent_adapter=parent,
        max_eval_loss_regression=-0.2,
    )
    loaded = st.load_hf_adapter_promotion(child)

    assert promoted["status"] == "ready"
    assert promoted["promotion_ready"] is True
    assert promoted["eval_loss_regression"] == pytest.approx(-0.1)
    assert promoted["recommendation"] == "promote_candidate"
    assert loaded["candidate_adapter_id"] == promoted["candidate_adapter_id"]
    assert "ready=True" in st.hf_adapter_promotion_lines(loaded)[0]
    assert stricter["status"] == "blocked"
    assert "eval_loss_regression" in stricter["failed_checks"]

    (child / "adapter_model.safetensors").write_bytes(b"tampered")
    tampered = st.hf_adapter_promotion_report(
        child,
        card,
        parent_adapter=parent,
    )
    assert tampered["status"] == "blocked"
    assert "candidate_fingerprint" in tampered["failed_checks"]


def test_adapter_promotion_can_require_fresh_artifact_reload_and_generation(
    tmp_path: Path,
) -> None:
    parent = _write_adapter(tmp_path / "parent", b"parent")
    child = _write_adapter(tmp_path / "child", b"child")
    card = _run_card(parent)
    card["adapter_artifact_probe"] = _artifact_probe(child)
    lineage = st.write_hf_adapter_lineage(
        child,
        parent_adapter=parent,
        run_card=card,
    )
    card["adapter_lineage"] = lineage

    ready = st.write_hf_adapter_promotion(
        child,
        card,
        parent_adapter=parent,
        require_artifact_probe=True,
    )

    assert ready["status"] == "ready"
    assert ready["promotion_ready"] is True
    assert ready["require_artifact_probe"] is True
    assert ready["artifact_probe_status"] == "ready"
    assert ready["artifact_probe_candidate_matches"] is True
    assert ready["artifact_probe_new_token_count"] == 4
    assert ready["artifact_probe_local_files_only"] is True
    assert ready["artifact_probe_do_sample"] is False
    assert ready["artifact_probe_process_status"] == "ready"
    assert ready["artifact_probe_process_fresh"] is True
    assert ready["artifact_probe_process_parent_pid"] == 1001
    assert ready["artifact_probe_process_pid"] == 2002
    assert ready["artifact_probe_process_exit_code"] == 0
    assert ready["artifact_probe_process_timed_out"] is False
    assert not ready["failed_checks"]
    assert not ready["missing_checks"]

    run_card_path = child / "spiraltorch-hf-finetune-run-card.json"
    card["adapter_promotion"] = ready
    run_card_path.write_text(json.dumps(card), encoding="utf-8")
    chain = st.hf_adapter_promotion_chain_report(child)
    child_node = next(
        node for node in chain["nodes"] if node["adapter_path"] == str(child)
    )
    assert child_node["promotion_revalidated_ready"] is True
    assert child_node["artifact_probe_status"] == "ready"
    assert child_node["artifact_probe_new_token_count"] == 4
    assert child_node["artifact_probe_process_status"] == "ready"
    assert child_node["artifact_probe_process_pid"] == 2002


def test_adapter_promotion_chain_revalidates_gated_root_probe(
    tmp_path: Path,
) -> None:
    root = _write_adapter(tmp_path / "root", b"root")
    run_card_path = root / "spiraltorch-hf-finetune-run-card.json"
    card = _run_card(Path("org/base"))
    card.update(
        {
            "model_name": "org/base",
            "model_artifact_kind": "full_model",
            "finetune_start_report": {
                "mode": "new_adapter",
                "adapter_weights_source": None,
                "weights_only_warm_start": False,
                "trainer_checkpoint_resume": False,
            },
            "adapter_artifact_probe": _artifact_probe(root),
        }
    )
    lineage = st.write_hf_adapter_lineage(
        root,
        run_card=card,
        run_card_path=run_card_path,
    )
    card["adapter_lineage"] = lineage
    promotion = st.write_hf_adapter_promotion(
        root,
        card,
        require_artifact_probe=True,
    )
    card["adapter_promotion"] = promotion
    run_card_path.write_text(json.dumps(card), encoding="utf-8")

    ready = st.hf_adapter_promotion_chain_report(root)
    root_node = next(node for node in ready["nodes"] if node["lineage_depth"] == 0)

    assert root_node["status"] == "ready"
    assert root_node["promotion_revalidated_ready"] is True
    assert root_node["artifact_probe_status"] == "ready"

    promotion_path = root / st.HF_ADAPTER_PROMOTION_FILENAME
    tampered = json.loads(promotion_path.read_text(encoding="utf-8"))
    tampered["artifact_probe_new_token_count"] = 999
    promotion_path.write_text(json.dumps(tampered), encoding="utf-8")
    rejected = st.hf_adapter_promotion_chain_report(root)
    rejected_root = next(
        node for node in rejected["nodes"] if node["lineage_depth"] == 0
    )

    assert rejected_root["status"] == "rejected"
    assert any(
        issue["code"] == "promotion_revalidation_mismatch"
        for issue in rejected_root["issues"]
    )


def test_adapter_promotion_blocks_missing_or_mismatched_artifact_probe(
    tmp_path: Path,
) -> None:
    parent = _write_adapter(tmp_path / "parent", b"parent")
    missing_child = _write_adapter(tmp_path / "missing", b"missing")
    missing_card = _run_card(parent)
    missing_lineage = st.write_hf_adapter_lineage(
        missing_child,
        parent_adapter=parent,
        run_card=missing_card,
    )
    missing_card["adapter_lineage"] = missing_lineage

    missing = st.hf_adapter_promotion_report(
        missing_child,
        missing_card,
        parent_adapter=parent,
        require_artifact_probe=True,
    )

    assert missing["status"] == "needs_evidence"
    assert missing["recommendation"] == "run_artifact_reload_probe"
    assert set(missing["missing_checks"]) == {
        "artifact_reload",
        "artifact_generation",
        "artifact_process_isolation",
    }

    mismatched_child = _write_adapter(tmp_path / "mismatched", b"mismatched")
    mismatched_card = _run_card(parent)
    mismatched_card["adapter_artifact_probe"] = _artifact_probe(missing_child)
    mismatched_lineage = st.write_hf_adapter_lineage(
        mismatched_child,
        parent_adapter=parent,
        run_card=mismatched_card,
    )
    mismatched_card["adapter_lineage"] = mismatched_lineage
    mismatched = st.hf_adapter_promotion_report(
        mismatched_child,
        mismatched_card,
        parent_adapter=parent,
        require_artifact_probe=True,
    )

    assert mismatched["status"] == "blocked"
    assert "artifact_reload" in mismatched["failed_checks"]
    assert mismatched["artifact_probe_candidate_matches"] is False

    nonlocal_card = _run_card(parent)
    nonlocal_card["adapter_artifact_probe"] = _artifact_probe(mismatched_child)
    nonlocal_card["adapter_artifact_probe"]["local_files_only"] = False
    nonlocal_lineage = st.write_hf_adapter_lineage(
        mismatched_child,
        parent_adapter=parent,
        run_card=nonlocal_card,
    )
    nonlocal_card["adapter_lineage"] = nonlocal_lineage
    nonlocal_report = st.hf_adapter_promotion_report(
        mismatched_child,
        nonlocal_card,
        parent_adapter=parent,
        require_artifact_probe=True,
    )
    assert "artifact_reload" in nonlocal_report["failed_checks"]

    sampled_card = _run_card(parent)
    sampled_card["adapter_artifact_probe"] = _artifact_probe(mismatched_child)
    sampled_card["adapter_artifact_probe"]["generation"]["do_sample"] = True
    sampled_lineage = st.write_hf_adapter_lineage(
        mismatched_child,
        parent_adapter=parent,
        run_card=sampled_card,
    )
    sampled_card["adapter_lineage"] = sampled_lineage
    sampled_report = st.hf_adapter_promotion_report(
        mismatched_child,
        sampled_card,
        parent_adapter=parent,
        require_artifact_probe=True,
    )
    assert "artifact_generation" in sampled_report["failed_checks"]

    same_process_card = _run_card(parent)
    same_process_card["adapter_artifact_probe"] = _artifact_probe(mismatched_child)
    same_process_card["adapter_artifact_probe"]["process_isolation"]["pid"] = 1001
    same_process_card["adapter_artifact_probe"]["worker_pid"] = 1001
    same_process_lineage = st.write_hf_adapter_lineage(
        mismatched_child,
        parent_adapter=parent,
        run_card=same_process_card,
    )
    same_process_card["adapter_lineage"] = same_process_lineage
    same_process_report = st.hf_adapter_promotion_report(
        mismatched_child,
        same_process_card,
        parent_adapter=parent,
        require_artifact_probe=True,
    )
    assert "artifact_process_isolation" in same_process_report["failed_checks"]


def test_adapter_lineage_records_unverified_remote_parent(tmp_path: Path) -> None:
    child = _write_adapter(tmp_path / "child", b"child")
    card = _run_card(Path("org/remote-adapter"))
    lineage = st.write_hf_adapter_lineage(child, run_card=card)
    card["adapter_lineage"] = lineage

    report = st.hf_adapter_promotion_report(child, card)

    assert lineage["parent_adapter_reference"] == "org/remote-adapter"
    assert lineage["parent_fingerprint_verified"] is False
    assert lineage["lineage_depth"] is None
    assert report["status"] == "needs_evidence"
    assert "parent_fingerprint" in report["missing_checks"]


def test_adapter_promotion_needs_before_after_eval(tmp_path: Path) -> None:
    parent = _write_adapter(tmp_path / "parent", b"parent")
    child = _write_adapter(tmp_path / "child", b"child")
    card = _run_card(parent)
    card["eval_before_train"] = None
    card["eval_after_train"] = None
    lineage = st.write_hf_adapter_lineage(
        child,
        parent_adapter=parent,
        run_card=card,
    )
    card["adapter_lineage"] = lineage

    report = st.hf_adapter_promotion_report(
        child,
        card,
        parent_adapter=parent,
    )

    assert report["status"] == "needs_evidence"
    assert report["promotion_ready"] is False
    assert "eval_evidence" in report["missing_checks"]
    assert "eval_loss_regression" in report["missing_checks"]


def test_adapter_promotion_rejects_non_ready_lineage(tmp_path: Path) -> None:
    parent = _write_adapter(tmp_path / "parent", b"parent")
    child = _write_adapter(tmp_path / "child", b"child")
    card = _run_card(parent)
    lineage = st.write_hf_adapter_lineage(
        child,
        parent_adapter=parent,
        run_card=card,
    )
    lineage["status"] = "invalid"
    (child / st.HF_ADAPTER_LINEAGE_FILENAME).write_text(
        json.dumps(lineage),
        encoding="utf-8",
    )

    report = st.hf_adapter_promotion_report(
        child,
        card,
        parent_adapter=parent,
    )

    assert report["status"] == "blocked"
    assert "lineage_manifest" in report["failed_checks"]


def test_adapter_lineage_and_promotion_clis_write_auditable_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parent = _write_adapter(tmp_path / "parent", b"parent")
    child = _write_adapter(tmp_path / "child", b"child")
    card_path = tmp_path / "run-card.json"
    card = _run_card(parent)
    card["adapter_artifact_probe"] = _artifact_probe(child)
    card_path.write_text(
        json.dumps(card),
        encoding="utf-8",
    )

    lineage_code = hf_cli.adapter_lineage_main(
        [
            "--adapter",
            str(child),
            "--parent-adapter",
            str(parent),
            "--run-card",
            str(card_path),
        ]
    )
    lineage_output = capsys.readouterr().out
    promotion_code = hf_cli.adapter_promotion_main(
        [
            "--candidate",
            str(child),
            "--parent-adapter",
            str(parent),
            "--run-card",
            str(card_path),
            "--require-artifact-probe",
        ]
    )
    promotion_output = capsys.readouterr().out

    assert lineage_code == 0
    assert promotion_code == 0
    assert "depth=1" in lineage_output
    assert "ready=True" in promotion_output
    assert "artifact_probe=ready" in promotion_output
    assert "probe_process=ready" in promotion_output
    assert (child / st.HF_ADAPTER_LINEAGE_FILENAME).is_file()
    assert (child / st.HF_ADAPTER_PROMOTION_FILENAME).is_file()


def test_run_card_digest_survives_generic_row_type_normalization(
    tmp_path: Path,
) -> None:
    parent = _write_adapter(tmp_path / "parent", b"parent")
    child = _write_adapter(tmp_path / "child", b"child")
    legacy_card = _run_card(parent)
    legacy_card["row_type"] = "hf_gpt2_finetune_run_card"
    legacy_card["eval_before_train"]["row_type"] = "hf_gpt2_finetune_eval_report"
    legacy_card["eval_after_train"]["row_type"] = "hf_gpt2_finetune_eval_report"
    lineage = st.write_hf_adapter_lineage(
        child,
        parent_adapter=parent,
        run_card=legacy_card,
    )
    legacy_card["adapter_lineage"] = lineage
    generic_card_path = tmp_path / "generic-run-card.json"
    st.write_hf_finetune_run_card(legacy_card, generic_card_path)

    report = st.hf_adapter_promotion_report(
        child,
        generic_card_path,
        parent_adapter=parent,
    )

    assert report["promotion_ready"] is True
    assert "run_card_digest" not in report["failed_checks"]


def test_adapter_promotion_chain_selects_deepest_tip_for_scale_up(
    tmp_path: Path,
) -> None:
    root = _write_adapter(tmp_path / "root", b"root")
    root_lineage = st.write_hf_adapter_lineage(root)
    child, child_lineage, _ = _write_promoted_child(
        tmp_path / "child",
        root,
        b"child",
    )
    grandchild, grandchild_lineage, _ = _write_promoted_child(
        tmp_path / "grandchild",
        child,
        b"grandchild",
        before=0.9,
        after=0.8,
    )

    report = st.hf_adapter_promotion_chain_report(tmp_path)
    report_path = tmp_path / st.HF_ADAPTER_PROMOTION_CHAIN_FILENAME
    written = st.write_hf_adapter_promotion_chain(report, report_path)
    loaded = st.load_hf_adapter_promotion_chain(report_path)
    scale_up = st.hf_finetune_scale_up_command(
        report_path,
        output_dir=tmp_path / "great-grandchild",
        max_steps=4,
        max_train_samples=16,
    )
    direct_preflight = st.hf_finetune_scale_up_preflight_report(report)

    assert root_lineage["lineage_depth"] == 0
    assert child_lineage["lineage_depth"] == 1
    assert grandchild_lineage["lineage_depth"] == 2
    assert report["status"] == "ready"
    assert report["chain_ready"] is True
    assert report["continuation_ready"] is True
    assert report["node_count"] == 3
    assert report["eligible_node_count"] == 3
    assert report["selected_adapter_id"] == grandchild_lineage["adapter_id"]
    assert report["selected_adapter_path"] == str(grandchild.resolve())
    assert report["selected_path_adapter_ids"] == [
        root_lineage["adapter_id"],
        child_lineage["adapter_id"],
        grandchild_lineage["adapter_id"],
    ]
    assert report["transition_count"] == 2
    assert report["ready_transition_count"] == 2
    assert report["rejected_transition_count"] == 0
    assert report["selected_path_transition_count"] == 2
    assert report["selected_path_transitions_ready"] is True
    assert [row["status"] for row in report["transitions"]] == ["ready", "ready"]
    child_to_grandchild = report["transitions"][1]
    assert child_to_grandchild["row_type"] == (
        "hf_adapter_promotion_chain_transition"
    )
    assert child_to_grandchild["parent_adapter_id"] == child_lineage["adapter_id"]
    assert child_to_grandchild["child_adapter_id"] == grandchild_lineage["adapter_id"]
    assert child_to_grandchild["depth_step"] == 1
    assert child_to_grandchild["root_matches"] is True
    assert child_to_grandchild["base_model_matches"] is True
    assert child_to_grandchild["parent_fingerprint_verified"] is True
    assert child_to_grandchild["weights_changed_from_parent"] is True
    assert child_to_grandchild["eval_handoff_observed"] is True
    assert child_to_grandchild["eval_handoff_delta"] == pytest.approx(0.0)
    assert child_to_grandchild["child_eval_improvement"] == pytest.approx(0.1)
    assert child_to_grandchild["selected_path"] is True
    assert written["report_path"] == str(report_path.resolve())
    assert loaded["selected_adapter_id"] == grandchild_lineage["adapter_id"]
    report_lines = st.hf_adapter_promotion_chain_lines(report)
    transition_lines = [
        line
        for line in report_lines
        if line.startswith("hf_adapter_promotion_chain_transition ")
    ]
    assert "continuation_ready=True" in report_lines[0]
    assert "transitions=2" in report_lines[0]
    assert len(transition_lines) == 2
    assert "eval_handoff_delta=0.0" in transition_lines[1]
    assert scale_up["status"] == "ok"
    assert scale_up["adapter_continuation_applied"] is True
    assert scale_up["adapter_continuation_source"] == str(grandchild.resolve())
    assert (
        scale_up["adapter_continuation_source_adapter_id"]
        == grandchild_lineage["adapter_id"]
    )
    assert scale_up["adapter_continuation_expected_child_lineage_depth"] == 3
    identity_contract = scale_up["adapter_continuation_identity_contract"]
    assert identity_contract["status"] == "enforced"
    assert identity_contract["fail_fast"] is True
    assert (
        identity_contract["expected_parent_adapter_id"]
        == grandchild_lineage["adapter_id"]
    )
    assert identity_contract["expected_parent_lineage_depth"] == 2
    assert (
        identity_contract["expected_root_adapter_id"]
        == root_lineage["adapter_id"]
    )
    assert (
        scale_up["promotion_chain_selected_adapter_id"]
        == grandchild_lineage["adapter_id"]
    )
    assert scale_up["promotion_chain_source_path"] == str(report_path)
    assert scale_up["promotion_chain_transition_count"] == 2
    assert scale_up["promotion_chain_ready_transition_count"] == 2
    assert scale_up["promotion_chain_selected_path_transition_count"] == 2
    assert scale_up["promotion_chain_selected_path_transitions_ready"] is True
    assert scale_up["promotion_chain_selected_transition"] == child_to_grandchild
    assert scale_up["command"][scale_up["command"].index("--model-name") + 1] == str(
        grandchild.resolve()
    )
    assert scale_up["command"][
        scale_up["command"].index("--expected-parent-adapter-id") + 1
    ] == grandchild_lineage["adapter_id"]
    assert scale_up["command"][
        scale_up["command"].index("--expected-parent-lineage-depth") + 1
    ] == "2"
    assert scale_up["command"][
        scale_up["command"].index("--expected-root-adapter-id") + 1
    ] == root_lineage["adapter_id"]
    assert direct_preflight["status"] == "ready"
    assert direct_preflight["adapter_continuation_applied"] is True
    assert direct_preflight["adapter_continuation_source_adapter_id"] == (
        grandchild_lineage["adapter_id"]
    )
    assert direct_preflight["promotion_chain_selected_path_transitions_ready"] is True
    assert direct_preflight["promotion_chain_selected_transition"] == (
        child_to_grandchild
    )
    assert direct_preflight["adapter_input_identity"]["status"] == "ready"
    assert (
        direct_preflight["adapter_input_identity"]["observed_adapter_id"]
        == grandchild_lineage["adapter_id"]
    )
    legacy_chain = json.loads(json.dumps(report))
    for field in (
        "transitions",
        "transition_count",
        "ready_transition_count",
        "rejected_transition_count",
        "selected_path_transition_count",
        "selected_path_transitions_ready",
    ):
        legacy_chain.pop(field, None)
    legacy_scale_up = st.hf_finetune_scale_up_command(
        legacy_chain,
        output_dir=tmp_path / "legacy-great-grandchild",
    )
    assert legacy_scale_up["status"] == "ok"
    assert legacy_scale_up["promotion_chain_transition_count"] is None
    assert legacy_scale_up["promotion_chain_selected_transition"] is None
    unsupported = dict(report)
    unsupported["schema"] = "spiraltorch.hf_adapter_promotion_chain.v999"
    assert st.hf_finetune_scale_up_command(unsupported)["status"] == (
        "promotion_chain_unsupported_schema"
    )

    (grandchild / "adapter_model.safetensors").write_bytes(b"tampered")
    tampered_preflight = st.hf_finetune_scale_up_preflight_report(scale_up)
    assert tampered_preflight["status"] == "blocked"
    assert tampered_preflight["adapter_input_identity"]["status"] == "blocked"
    assert any(
        issue["field"] == "adapter_continuation_identity_contract"
        for issue in tampered_preflight["issues"]
    )


def test_adapter_promotion_chain_stops_before_blocked_generation(
    tmp_path: Path,
) -> None:
    root = _write_adapter(tmp_path / "root", b"root")
    st.write_hf_adapter_lineage(root)
    child, child_lineage, _ = _write_promoted_child(
        tmp_path / "child",
        root,
        b"child",
    )
    grandchild, _, _ = _write_promoted_child(
        tmp_path / "grandchild",
        child,
        b"grandchild",
    )
    promotion_path = grandchild / st.HF_ADAPTER_PROMOTION_FILENAME
    blocked = json.loads(promotion_path.read_text(encoding="utf-8"))
    blocked["status"] = "blocked"
    blocked["promotion_ready"] = False
    blocked["failed_checks"] = ["eval_loss_regression"]
    promotion_path.write_text(json.dumps(blocked), encoding="utf-8")

    report = st.hf_adapter_promotion_chain_report(tmp_path)

    assert report["status"] == "ready_with_rejections"
    assert report["selected_adapter_id"] == child_lineage["adapter_id"]
    assert report["selected_lineage_depth"] == 1
    assert report["rejected_node_count"] == 1
    assert report["transition_count"] == 2
    assert report["ready_transition_count"] == 1
    assert report["rejected_transition_count"] == 1
    assert report["selected_path_transition_count"] == 1
    assert report["selected_path_transitions_ready"] is True
    assert report["transitions"][1]["status"] == "rejected"
    assert report["transitions"][1]["selected_path"] is False
    grandchild_node = next(
        node for node in report["nodes"] if node["adapter_path"] == str(grandchild)
    )
    assert grandchild_node["status"] == "rejected"
    assert any(
        issue["code"] == "promotion_not_ready" for issue in grandchild_node["issues"]
    )


def test_adapter_continuation_policy_stops_plateau_and_blocks_scale_up(
    tmp_path: Path,
) -> None:
    root = _write_adapter(tmp_path / "root", b"root")
    st.write_hf_adapter_lineage(root)
    child, _, _ = _write_promoted_child(
        tmp_path / "child",
        root,
        b"child",
        before=1.0,
        after=0.99,
    )
    _, grandchild_lineage, _ = _write_promoted_child(
        tmp_path / "grandchild",
        child,
        b"grandchild",
        before=0.99,
        after=0.985,
    )

    report = st.hf_adapter_promotion_chain_report(
        tmp_path,
        min_eval_improvement=0.02,
        plateau_patience=2,
    )
    scale_up = st.hf_finetune_scale_up_command(report)
    preflight = st.hf_finetune_scale_up_preflight_report(report)
    policy_path = tmp_path / st.HF_ADAPTER_CONTINUATION_POLICY_FILENAME
    written = st.write_hf_adapter_continuation_policy(
        report,
        policy_path,
        min_eval_improvement=0.02,
        plateau_patience=2,
    )
    loaded = st.load_hf_adapter_continuation_policy(policy_path)

    assert report["selected_adapter_id"] == grandchild_lineage["adapter_id"]
    assert report["status"] == "stopped_by_policy"
    assert report["chain_ready"] is True
    assert report["continuation_artifacts_ready"] is True
    assert report["continuation_allowed"] is False
    assert report["continuation_ready"] is False
    assert report["continuation_stop_reason_codes"] == [
        "eval_improvement_plateau"
    ]
    policy = report["continuation_policy"]
    assert policy["status"] == "stop"
    assert policy["consecutive_below_min_eval_improvement"] == 2
    assert policy["selected_path_eval_improvement"] == pytest.approx(0.015)
    assert [row["eval_improvement"] for row in policy["observations"]] == (
        pytest.approx([0.01, 0.005])
    )
    assert scale_up["status"] == "promotion_chain_stopped_by_policy"
    assert scale_up["promotion_chain_continuation_stop_reason_codes"] == [
        "eval_improvement_plateau"
    ]
    assert preflight["status"] == "blocked"
    assert preflight["scale_up_status"] == "promotion_chain_stopped_by_policy"
    assert preflight["issues"][0]["field"] == "continuation_policy"
    assert preflight["promotion_chain_continuation_stop_reason_codes"] == [
        "eval_improvement_plateau"
    ]
    assert written["report_path"] == str(policy_path.resolve())
    assert loaded["stop_reason_codes"] == ["eval_improvement_plateau"]
    assert "status=stop" in st.hf_adapter_continuation_policy_lines(loaded)[0]
    assert "policy=stop" in st.hf_adapter_promotion_chain_lines(report)[0]


def test_adapter_continuation_policy_depth_target_missing_evidence_and_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _write_adapter(tmp_path / "root", b"root")
    st.write_hf_adapter_lineage(root)
    child, _, _ = _write_promoted_child(
        tmp_path / "child",
        root,
        b"child",
        before=1.0,
        after=0.9,
    )
    _, _, _ = _write_promoted_child(
        tmp_path / "grandchild",
        child,
        b"grandchild",
        before=0.9,
        after=0.8,
    )
    base = st.hf_adapter_promotion_chain_report(tmp_path)

    depth = st.hf_adapter_continuation_policy_report(base, max_lineage_depth=2)
    target = st.hf_adapter_continuation_policy_report(base, target_eval_loss=0.8)
    continuing = st.hf_adapter_continuation_policy_report(
        base,
        max_lineage_depth=3,
        target_eval_loss=0.7,
        min_eval_improvement=0.05,
        plateau_patience=2,
    )
    missing_chain = json.loads(json.dumps(base))
    missing_chain["nodes"][-1]["eval_after_loss"] = None
    missing = st.hf_adapter_continuation_policy_report(
        missing_chain,
        target_eval_loss=0.7,
        min_eval_improvement=0.05,
    )
    missing_nodes_chain = json.loads(json.dumps(base))
    missing_nodes_chain["nodes"] = []
    missing_nodes = st.hf_adapter_continuation_policy_report(
        missing_nodes_chain,
        min_eval_improvement=0.05,
        plateau_patience=2,
    )
    code = hf_cli.adapter_promotion_chain_main(
        [
            str(tmp_path),
            "--max-lineage-depth",
            "2",
            "--require-continuation-ready",
        ]
    )
    output = capsys.readouterr().out

    assert depth["status"] == "stop"
    assert depth["stop_reason_codes"] == ["max_lineage_depth_reached"]
    assert target["status"] == "stop"
    assert target["stop_reason_codes"] == ["target_eval_loss_reached"]
    assert continuing["status"] == "continue"
    assert continuing["continuation_allowed"] is True
    assert missing["status"] == "needs_evidence"
    assert missing["continuation_allowed"] is False
    assert missing["recommendation"] == "collect_eval_evidence"
    assert {row["field"] for row in missing["missing_evidence"]} == {
        "selected_eval_after_loss",
        "eval_improvement",
    }
    assert missing_nodes["status"] == "needs_evidence"
    assert {row["field"] for row in missing_nodes["missing_evidence"]} == {
        "selected_path_node",
        "selected_path_observations",
    }
    assert code == 1
    assert "status=stopped_by_policy" in output
    assert "code=max_lineage_depth_reached" in output


def test_adapter_continuation_policy_gates_selected_tip_geometry_telemetry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _write_adapter(tmp_path / "root", b"root")
    st.write_hf_adapter_lineage(root)
    _, child_lineage, promotion = _write_promoted_child(
        tmp_path / "child",
        root,
        b"child",
        trainer_trace_summary={
            "trace_training_telemetry_count": 4,
            "trace_mean_desire_stability": 0.72,
            "trace_max_psi_total": 0.55,
            "trace_last_inference_distortion_risk_score": 0.30,
        },
    )
    promotion_path = tmp_path / "child" / st.HF_ADAPTER_PROMOTION_FILENAME
    legacy_promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    for field in (
        "distortion_pressure_index",
        "trace_training_telemetry_count",
        "trace_mean_desire_stability",
        "trace_max_psi_total",
    ):
        legacy_promotion.pop(field)
    promotion_path.write_text(json.dumps(legacy_promotion), encoding="utf-8")

    passing = st.hf_adapter_promotion_chain_report(
        tmp_path,
        max_distortion_pressure_index=0.40,
        min_desire_stability=0.70,
        max_psi_total=0.60,
    )
    stopped = st.hf_adapter_continuation_policy_report(
        passing,
        max_distortion_pressure_index=0.20,
        min_desire_stability=0.80,
        max_psi_total=0.50,
    )
    missing_chain = json.loads(json.dumps(passing))
    selected = next(
        node
        for node in missing_chain["nodes"]
        if node["adapter_id"] == child_lineage["adapter_id"]
    )
    selected["distortion_pressure_index"] = -0.1
    selected["trace_mean_desire_stability"] = 1.1
    selected["trace_max_psi_total"] = 1.1
    missing = st.hf_adapter_continuation_policy_report(
        missing_chain,
        max_distortion_pressure_index=0.40,
        min_desire_stability=0.70,
        max_psi_total=0.60,
    )
    root_chain = st.hf_adapter_promotion_chain_report(
        root,
        max_distortion_pressure_index=0.40,
        min_desire_stability=0.70,
        max_psi_total=0.60,
    )
    code = hf_cli.adapter_promotion_chain_main(
        [
            str(tmp_path),
            "--max-distortion-pressure-index",
            "0.20",
            "--min-desire-stability",
            "0.80",
            "--max-psi-total",
            "0.50",
            "--require-continuation-ready",
        ]
    )
    output = capsys.readouterr().out

    assert promotion["distortion_pressure_index"] == pytest.approx(0.30)
    assert promotion["trace_training_telemetry_count"] == 4
    assert promotion["trace_mean_desire_stability"] == pytest.approx(0.72)
    assert promotion["trace_max_psi_total"] == pytest.approx(0.55)
    assert passing["continuation_policy_status"] == "continue"
    assert passing["continuation_ready"] is True
    assert passing["continuation_policy"]["geometry_gate_active"] is True
    assert passing["continuation_policy"][
        "selected_distortion_pressure_index"
    ] == pytest.approx(0.30)
    assert passing["transitions"][0][
        "child_trace_mean_desire_stability"
    ] == pytest.approx(0.72)
    assert stopped["status"] == "stop"
    assert stopped["stop_reason_codes"] == [
        "distortion_pressure_limit_exceeded",
        "desire_stability_below_minimum",
        "psi_total_limit_exceeded",
    ]
    assert missing["status"] == "needs_evidence"
    assert missing["recommendation"] == "collect_policy_evidence"
    assert {row["field"] for row in missing["missing_evidence"]} == {
        "selected_distortion_pressure_index",
        "selected_trace_mean_desire_stability",
        "selected_trace_max_psi_total",
    }
    assert root_chain["continuation_policy_status"] == "continue"
    assert code == 1
    assert "code=distortion_pressure_limit_exceeded" in output
    assert "code=desire_stability_below_minimum" in output
    assert "code=psi_total_limit_exceeded" in output


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_lineage_depth": -1},
        {"target_eval_loss": float("nan")},
        {"min_eval_improvement": -0.1},
        {"max_distortion_pressure_index": 1.1},
        {"min_desire_stability": -0.1},
        {"max_psi_total": -0.1},
        {"max_psi_total": 1.1},
        {"plateau_patience": 0},
    ],
)
def test_adapter_continuation_policy_rejects_invalid_thresholds(
    tmp_path: Path,
    kwargs: dict[str, object],
) -> None:
    root = _write_adapter(tmp_path / "root", b"root")
    st.write_hf_adapter_lineage(root)
    chain = st.hf_adapter_promotion_chain_report(root)

    with pytest.raises(ValueError):
        st.hf_adapter_continuation_policy_report(chain, **kwargs)


def test_adapter_promotion_chain_requires_explicit_selection_for_equal_forks(
    tmp_path: Path,
) -> None:
    root = _write_adapter(tmp_path / "root", b"root")
    st.write_hf_adapter_lineage(root)
    first, first_lineage, _ = _write_promoted_child(
        tmp_path / "first",
        root,
        b"first",
    )
    _, second_lineage, _ = _write_promoted_child(
        tmp_path / "second",
        root,
        b"second",
    )

    ambiguous = st.hf_adapter_promotion_chain_report(tmp_path)
    selected = st.hf_adapter_promotion_chain_report(
        tmp_path,
        select_adapter_id=first_lineage["adapter_id"],
    )

    assert ambiguous["status"] == "ambiguous"
    assert ambiguous["chain_ready"] is False
    assert ambiguous["continuation_ready"] is False
    assert ambiguous["fork_count"] == 1
    assert ambiguous["selection_status"] == "ambiguous_deepest_tips"
    assert selected["status"] == "ready"
    assert selected["selection_status"] == "explicit"
    assert selected["selected_adapter_path"] == str(first.resolve())
    assert selected["selected_adapter_id"] != second_lineage["adapter_id"]


def test_adapter_promotion_chain_recovers_legacy_launch_command_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _write_adapter(tmp_path / "root", b"root")
    st.write_hf_adapter_lineage(root)
    child, child_lineage, _ = _write_promoted_child(
        tmp_path / "child",
        root,
        b"child",
        launch_command=False,
    )
    run_card_path = child / "spiraltorch-hf-finetune-run-card.json"
    command_artifact = tmp_path / "scale-up-command.json"
    command_artifact.write_text(
        json.dumps(
            {
                "row_type": "hf_finetune_scale_up_command",
                "status": "ok",
                "run_returncode": 0,
                "command": _adapter_launch_command(root, child, run_card_path),
            }
        ),
        encoding="utf-8",
    )
    chain_path = tmp_path / "chain.json"

    code = hf_cli.adapter_promotion_chain_main(
        [
            str(tmp_path),
            "--command-artifact",
            str(command_artifact),
            "--out",
            str(chain_path),
            "--require-continuation-ready",
        ]
    )
    output = capsys.readouterr().out
    report = json.loads(chain_path.read_text(encoding="utf-8"))

    assert code == 0
    assert report["status"] == "ready"
    assert report["selected_adapter_id"] == child_lineage["adapter_id"]
    assert report["continuation_ready"] is True
    assert report["continuation_candidate"]["launch_command_source"] == (
        "command_artifact"
    )
    assert "continuation_ready=True" in output


def test_adapter_promotion_chain_infers_verified_pre_lineage_seed(
    tmp_path: Path,
) -> None:
    root = _write_adapter(tmp_path / "root", b"root")
    st.write_hf_adapter_lineage(root)
    child, child_lineage, _ = _write_promoted_child(
        tmp_path / "child",
        root,
        b"child",
    )
    (root / st.HF_ADAPTER_LINEAGE_FILENAME).unlink()

    inferred = st.hf_adapter_promotion_chain_report(child)
    strict = st.hf_adapter_promotion_chain_report(
        child,
        allow_inferred_roots=False,
    )

    assert inferred["status"] == "ready"
    assert inferred["root_count"] == 1
    assert inferred["eligible_node_count"] == 2
    assert inferred["selected_adapter_id"] == child_lineage["adapter_id"]
    root_node = next(node for node in inferred["nodes"] if node["lineage_depth"] == 0)
    assert root_node["lineage_status"] == "inferred_seed"
    assert root_node["validation_ready"] is True
    assert any(
        issue["code"] == "root_lineage_inferred" for issue in root_node["issues"]
    )
    assert strict["status"] == "blocked"
    assert strict["eligible_node_count"] == 0


def test_adapter_promotion_chain_revalidates_stored_promotion(
    tmp_path: Path,
) -> None:
    root = _write_adapter(tmp_path / "root", b"root")
    st.write_hf_adapter_lineage(root)
    child, _, _ = _write_promoted_child(
        tmp_path / "child",
        root,
        b"child",
    )
    promotion_path = child / st.HF_ADAPTER_PROMOTION_FILENAME
    tampered = json.loads(promotion_path.read_text(encoding="utf-8"))
    tampered["eval_loss_regression"] = 42.0
    promotion_path.write_text(json.dumps(tampered), encoding="utf-8")

    report = st.hf_adapter_promotion_chain_report(tmp_path)
    child_node = next(
        node for node in report["nodes"] if node["adapter_path"] == str(child)
    )

    assert child_node["status"] == "rejected"
    assert child_node["promotion_ready"] is True
    assert child_node["promotion_revalidated_ready"] is True
    assert any(
        issue["code"] == "promotion_revalidation_mismatch"
        for issue in child_node["issues"]
    )
