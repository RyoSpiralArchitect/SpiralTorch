from __future__ import annotations

import shutil
from pathlib import Path

from spiraltorch.hf_runtime_identity import (
    HF_CAUSAL_LM_RUNTIME_IDENTITY_SCHEMA,
    hf_causal_lm_runtime_identity_lines,
    hf_causal_lm_runtime_identity_report,
)


class _BackendTokenizer:
    def to_str(self) -> str:
        return '{"model":{"type":"BPE"}}'


class _Tokenizer:
    backend_tokenizer = _BackendTokenizer()
    special_tokens_map = {"eos_token": "<eos>"}
    model_input_names = ["input_ids", "attention_mask"]
    padding_side = "right"
    truncation_side = "right"

    def get_vocab(self) -> dict[str, int]:
        return {"<eos>": 0, "spiral": 1}

    def get_added_vocab(self) -> dict[str, int]:
        return {"<eos>": 0}


class _Config:
    def __init__(self, *, commit: str | None = None) -> None:
        self._commit_hash = commit

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": "gpt_neox",
            "hidden_size": 64,
            "_commit_hash": self._commit_hash,
            "_name_or_path": "/machine-specific/path",
            "transformers_version": "test",
        }


def _local_runtime(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        '{"model_type":"gpt_neox","hidden_size":64}\n',
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(b"base-weights")
    (root / "tokenizer.json").write_text(
        '{"model":{"type":"BPE"}}\n',
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        '{"eos_token":"<eos>"}\n',
        encoding="utf-8",
    )
    chat_templates = root / "chat_templates"
    chat_templates.mkdir()
    (chat_templates / "tool_use.jinja").write_text(
        "{{ messages | length }}\n",
        encoding="utf-8",
    )
    return root


def _report(
    source: str | Path,
    *,
    config: _Config | None = None,
    revision: str | None = None,
    expected_identity_id: str | None = None,
    phase: str = "pre_model_load",
) -> dict[str, object]:
    return hf_causal_lm_runtime_identity_report(
        base_model_source=source,
        base_model_revision=revision,
        tokenizer_source=source,
        tokenizer_source_kind="base_model",
        config=config or _Config(),
        tokenizer=_Tokenizer(),
        expected_identity_id=expected_identity_id,
        phase=phase,
    )


def test_runtime_identity_is_path_independent_and_detects_local_tamper(
    tmp_path: Path,
) -> None:
    source = _local_runtime(tmp_path / "source")
    ready = _report(source)
    relocated = tmp_path / "relocated"
    shutil.copytree(source, relocated)
    verified = _report(
        relocated,
        expected_identity_id=ready["observed_identity_id"],
        phase="after_model_load",
    )

    (relocated / "tokenizer.json").write_text(
        '{"model":{"type":"WordPiece"}}\n',
        encoding="utf-8",
    )
    blocked_tokenizer = _report(
        relocated,
        expected_identity_id=ready["observed_identity_id"],
    )
    shutil.copyfile(source / "tokenizer.json", relocated / "tokenizer.json")
    (relocated / "model.safetensors").write_bytes(b"different-base-weights")
    blocked_model = _report(
        relocated,
        expected_identity_id=ready["observed_identity_id"],
    )

    assert ready["schema"] == HF_CAUSAL_LM_RUNTIME_IDENTITY_SCHEMA
    assert ready["status"] == "ready"
    assert ready["path_independent"] is True
    assert ready["base_model"]["local_file_count"] == 2
    assert ready["tokenizer"]["local_file_count"] == 3
    assert verified["status"] == "ready"
    assert verified["expected_identity_verified"] is True
    assert verified["observed_identity_id"] == ready["observed_identity_id"]
    assert blocked_tokenizer["status"] == "blocked"
    assert blocked_model["status"] == "blocked"
    assert "status=ready" in hf_causal_lm_runtime_identity_lines(ready)[0]


def test_runtime_identity_pins_remote_commit_and_rejects_drift() -> None:
    commit = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
    ready = _report("org/model", config=_Config(commit=commit))
    verified = _report(
        "org/model",
        config=_Config(commit=commit),
        expected_identity_id=ready["observed_identity_id"],
    )
    explicitly_pinned = _report(
        "org/model",
        config=_Config(commit=commit),
        revision=commit,
    )
    drifted = _report(
        "org/model",
        config=_Config(commit="0" * 40),
        expected_identity_id=ready["observed_identity_id"],
    )

    assert ready["status"] == "ready"
    assert ready["base_model"]["source_kind"] == "remote"
    assert ready["base_model"]["observed_commit"] == commit
    assert verified["status"] == "ready"
    assert explicitly_pinned["observed_identity_id"] == ready["observed_identity_id"]
    assert drifted["status"] == "blocked"
    assert drifted["expected_identity_verified"] is False


def test_runtime_identity_keeps_unpinned_remote_source_incomplete() -> None:
    incomplete = _report("org/model")
    blocked = _report(
        "org/model",
        expected_identity_id=f"sha256:{'0' * 64}",
    )

    assert incomplete["status"] == "evidence_incomplete"
    assert incomplete["observed_identity_id"] is None
    assert incomplete["identity_verified"] is False
    assert blocked["status"] == "blocked"
    assert blocked["expected_identity_verified"] is False
