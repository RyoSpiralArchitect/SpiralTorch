import shutil
from pathlib import Path

import pytest

import spiraltorch.hf_input_identity as hf_input_identity
from spiraltorch.hf_input_identity import (
    HF_FINETUNE_INPUT_IDENTITY_SCHEMA,
    hf_finetune_input_identity_lines,
    hf_finetune_input_identity_report,
)


def _input_tree(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True)
    config = root / "profiles.json"
    train = root / "train.txt"
    validation = root / "validation.txt"
    distortion = root / "distortion.json"
    checkpoint = root / "checkpoint-10"
    checkpoint.mkdir()
    config.write_text('{"profiles": {}}\n', encoding="utf-8")
    train.write_text("spiral training\n", encoding="utf-8")
    validation.write_text("spiral validation\n", encoding="utf-8")
    distortion.write_text('{"status": "ready"}\n', encoding="utf-8")
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 10}\n',
        encoding="utf-8",
    )
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer-state")
    return {
        "config": config,
        "train": train,
        "validation": validation,
        "distortion": distortion,
        "checkpoint": checkpoint,
    }


def _report(paths: dict[str, Path], **kwargs: object) -> dict[str, object]:
    return hf_finetune_input_identity_report(
        model_configs=paths["config"],
        train_files=[paths["train"]],
        validation_files=[paths["validation"]],
        inference_distortion_probe=paths["distortion"],
        resume_from_checkpoint=paths["checkpoint"],
        **kwargs,
    )


def test_finetune_input_identity_is_path_independent_and_detects_tamper(
    tmp_path: Path,
) -> None:
    source = _input_tree(tmp_path / "source")
    ready = _report(source, phase="plan")

    relocated_root = tmp_path / "relocated"
    shutil.copytree(tmp_path / "source", relocated_root)
    relocated = {
        "config": relocated_root / "profiles.json",
        "train": relocated_root / "train.txt",
        "validation": relocated_root / "validation.txt",
        "distortion": relocated_root / "distortion.json",
        "checkpoint": relocated_root / "checkpoint-10",
    }
    verified = _report(
        relocated,
        expected_input_id=ready["observed_input_id"],
        phase="preflight",
    )

    relocated["train"].write_text("tampered corpus\n", encoding="utf-8")
    blocked = _report(
        relocated,
        expected_input_id=ready["observed_input_id"],
        phase="preflight",
    )

    assert ready["schema"] == HF_FINETUNE_INPUT_IDENTITY_SCHEMA
    assert ready["status"] == "ready"
    assert ready["input_count"] == 5
    assert ready["file_count"] == 6
    assert ready["path_independent"] is True
    assert verified["status"] == "ready"
    assert verified["observed_input_id"] == ready["observed_input_id"]
    assert verified["expected_identity_verified"] is True
    assert blocked["status"] == "blocked"
    assert blocked["expected_identity_verified"] is False
    assert "fine-tune input fingerprint does not match expected input id" in blocked[
        "errors"
    ]
    assert "status=ready" in hf_finetune_input_identity_lines(verified)[0]


def test_finetune_input_identity_blocks_missing_and_invalid_expectation(
    tmp_path: Path,
) -> None:
    missing = hf_finetune_input_identity_report(
        train_files=[tmp_path / "missing.txt"]
    )
    empty = hf_finetune_input_identity_report()

    assert missing["status"] == "blocked"
    assert missing["observed_input_id"] is None
    assert missing["error_count"] == 1
    assert missing["identity_verified"] is False
    assert empty["status"] == "not_applicable"
    assert empty["identity_applicable"] is False
    assert empty["identity_verified"] is None
    expected_without_inputs = hf_finetune_input_identity_report(
        expected_input_id=f"sha256:{'0' * 64}"
    )
    assert expected_without_inputs["status"] == "blocked"
    assert expected_without_inputs["identity_verified"] is False
    with pytest.raises(ValueError, match="lowercase sha256"):
        hf_finetune_input_identity_report(expected_input_id="invalid")


def test_finetune_input_identity_detects_checkpoint_change_during_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _input_tree(tmp_path / "source")
    original = hf_input_identity._stable_file_identity

    def mutate_prior_file(path: Path):
        identity = original(path)
        if path.name == "trainer_state.json":
            (path.parent / "optimizer.pt").write_bytes(b"changed-after-hash")
        return identity

    monkeypatch.setattr(
        hf_input_identity,
        "_stable_file_identity",
        mutate_prior_file,
    )
    report = hf_finetune_input_identity_report(
        resume_from_checkpoint=paths["checkpoint"]
    )

    assert report["status"] == "blocked"
    assert report["identity_verified"] is False
    assert any("directory changed while hashing" in error for error in report["errors"])
