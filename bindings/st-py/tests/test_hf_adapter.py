from __future__ import annotations

import json
import shutil
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
