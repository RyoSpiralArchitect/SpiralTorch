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


def _adapter_launch_command(
    parent: Path,
    child: Path,
    run_card_path: Path,
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
        "8",
    ]


def _write_promoted_child(
    child: Path,
    parent: Path,
    weights: bytes,
    *,
    launch_command: bool = True,
) -> tuple[Path, dict, dict]:
    adapter = _write_adapter(child, weights)
    run_card_path = adapter / "spiraltorch-hf-finetune-run-card.json"
    card = _run_card(parent)
    if launch_command:
        card["launch_command"] = _adapter_launch_command(
            parent,
            adapter,
            run_card_path,
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
    card_path.write_text(
        json.dumps(_run_card(parent)),
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
        ]
    )
    promotion_output = capsys.readouterr().out

    assert lineage_code == 0
    assert promotion_code == 0
    assert "depth=1" in lineage_output
    assert "ready=True" in promotion_output
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
    assert written["report_path"] == str(report_path.resolve())
    assert loaded["selected_adapter_id"] == grandchild_lineage["adapter_id"]
    assert "continuation_ready=True" in st.hf_adapter_promotion_chain_lines(report)[0]
    assert scale_up["status"] == "ok"
    assert scale_up["adapter_continuation_applied"] is True
    assert scale_up["adapter_continuation_source"] == str(grandchild.resolve())
    assert (
        scale_up["adapter_continuation_source_adapter_id"]
        == grandchild_lineage["adapter_id"]
    )
    assert scale_up["adapter_continuation_expected_child_lineage_depth"] == 3
    assert (
        scale_up["promotion_chain_selected_adapter_id"]
        == grandchild_lineage["adapter_id"]
    )
    assert scale_up["promotion_chain_source_path"] == str(report_path)
    assert scale_up["command"][scale_up["command"].index("--model-name") + 1] == str(
        grandchild.resolve()
    )
    assert direct_preflight["status"] == "ready"
    assert direct_preflight["adapter_continuation_applied"] is True
    assert direct_preflight["adapter_continuation_source_adapter_id"] == (
        grandchild_lineage["adapter_id"]
    )
    unsupported = dict(report)
    unsupported["schema"] = "spiraltorch.hf_adapter_promotion_chain.v999"
    assert st.hf_finetune_scale_up_command(unsupported)["status"] == (
        "promotion_chain_unsupported_schema"
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
    grandchild_node = next(
        node for node in report["nodes"] if node["adapter_path"] == str(grandchild)
    )
    assert grandchild_node["status"] == "rejected"
    assert any(
        issue["code"] == "promotion_not_ready" for issue in grandchild_node["issues"]
    )


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
