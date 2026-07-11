from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

import spiraltorch as st
from spiraltorch import hf_cli


class _Parameter:
    def __init__(self, size: int, *, requires_grad: bool = True) -> None:
        self._size = size
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._size


class _Model:
    def __init__(self, model_type: str = "qwen2") -> None:
        self.config = types.SimpleNamespace(model_type=model_type, use_cache=True)
        self.gradient_checkpointing = False
        self.input_require_grads = False
        self.parameters_by_name = [
            ("model.embed.weight", _Parameter(80)),
            ("model.layers.0.self_attn.q_proj.weight", _Parameter(20)),
        ]

    def named_modules(self):
        return iter(
            [
                ("", self),
                ("model.layers.0.self_attn.q_proj", object()),
                ("model.layers.0.self_attn.k_proj", object()),
                ("model.layers.0.self_attn.v_proj", object()),
                ("model.layers.0.self_attn.o_proj", object()),
            ]
        )

    def named_parameters(self):
        return iter(self.parameters_by_name)

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def enable_input_require_grads(self) -> None:
        self.input_require_grads = True


class _FakePeft:
    __version__ = "test-peft"
    TaskType = types.SimpleNamespace(CAUSAL_LM="CAUSAL_LM")
    last_config = None

    class LoraConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            _FakePeft.last_config = self

        def to_dict(self):
            return dict(self.kwargs)

    @staticmethod
    def get_peft_model(model, _config):
        for _name, parameter in model.parameters_by_name:
            parameter.requires_grad = False
        model.parameters_by_name.append(
            ("base_model.model.layers.0.self_attn.q_proj.lora_A.weight", _Parameter(8))
        )
        model.parameters_by_name.append(
            ("base_model.model.layers.0.self_attn.q_proj.lora_B.weight", _Parameter(8))
        )
        model.peft_config = {"default": _config}
        model.active_adapter = "default"
        return model


class _ArtifactConfig:
    model_type = "gpt2"
    use_cache = True

    def __init__(self, commit: str | None = None) -> None:
        self._commit_hash = commit

    def to_dict(self):
        return {"model_type": self.model_type, "n_embd": 8}


class _ArtifactBackendTokenizer:
    def to_str(self) -> str:
        return '{"model":{"type":"BPE"}}'


class _ArtifactTokenizer:
    backend_tokenizer = _ArtifactBackendTokenizer()
    special_tokens_map = {"eos_token": "<eos>"}
    model_input_names = ["input_ids", "attention_mask"]
    padding_side = "right"
    truncation_side = "right"

    def __init__(self, source: str) -> None:
        self.source = source

    def get_vocab(self):
        return {"<eos>": 0, "spiral": 1}

    def get_added_vocab(self):
        return {"<eos>": 0}

    def save_pretrained(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        (output / "tokenizer.json").write_text("{}\n", encoding="utf-8")


class _ArtifactModel(_Model):
    def __init__(self, source: str) -> None:
        super().__init__("gpt2")
        self.source = source
        self.adapter_source = None
        self.adapter_trainable = None
        self.merged = False
        self.safe_merge = None

    def merge_and_unload(self, *, safe_merge: bool = False):
        self.merged = True
        self.safe_merge = safe_merge
        return self

    def save_pretrained(
        self,
        output_dir: str | Path,
        *,
        safe_serialization: bool = True,
    ) -> None:
        output = Path(output_dir)
        (output / "config.json").write_text("{}\n", encoding="utf-8")
        filename = "model.safetensors" if safe_serialization else "pytorch_model.bin"
        (output / filename).write_bytes(b"model")


class _FakeTransformers:
    __version__ = "test-transformers"
    config_calls = []
    tokenizer_calls = []
    model_calls = []
    config_commit = None

    class AutoConfig:
        @staticmethod
        def from_pretrained(source, **kwargs):
            _FakeTransformers.config_calls.append((str(source), dict(kwargs)))
            return _ArtifactConfig(_FakeTransformers.config_commit)

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(source, **kwargs):
            _FakeTransformers.tokenizer_calls.append((str(source), dict(kwargs)))
            return _ArtifactTokenizer(str(source))

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(source, **kwargs):
            _FakeTransformers.model_calls.append((str(source), dict(kwargs)))
            return _ArtifactModel(str(source))


class _ArtifactPeft:
    __version__ = "test-artifact-peft"
    config_calls = []
    model_calls = []

    class PeftConfig:
        base_model_name_or_path = "org/remote-base"

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            _ArtifactPeft.config_calls.append((str(source), dict(kwargs)))
            return cls()

        def to_dict(self):
            return {"base_model_name_or_path": self.base_model_name_or_path}

    class PeftModel:
        @staticmethod
        def from_pretrained(model, source, *, is_trainable=False, **kwargs):
            _ArtifactPeft.model_calls.append(
                (str(source), bool(is_trainable), dict(kwargs))
            )
            model.adapter_source = str(source)
            model.adapter_trainable = bool(is_trainable)
            config = types.SimpleNamespace(
                to_dict=lambda: {
                    "base_model_name_or_path": model.source,
                    "peft_type": "LORA",
                    "r": 4,
                    "lora_alpha": 8,
                }
            )
            model.peft_config = {"default": config}
            model.active_adapter = "default"
            for _name, parameter in model.parameters_by_name:
                parameter.requires_grad = False
            model.parameters_by_name.extend(
                [
                    ("base_model.model.q_proj.lora_A.weight", _Parameter(4)),
                    ("base_model.model.q_proj.lora_B.weight", _Parameter(4)),
                ]
            )
            return model


@pytest.fixture(autouse=True)
def _reset_artifact_fakes() -> None:
    _FakeTransformers.config_calls.clear()
    _FakeTransformers.tokenizer_calls.clear()
    _FakeTransformers.model_calls.clear()
    _FakeTransformers.config_commit = None
    _ArtifactPeft.config_calls.clear()
    _ArtifactPeft.model_calls.clear()


def _write_adapter(path: Path, *, base_model: str = "org/base") -> Path:
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": base_model,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
            }
        ),
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(b"adapter")
    (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    return path


def test_peft_runtime_is_public_without_eager_peft_dependency() -> None:
    assert "hf_peft" in st.__all__
    assert "load_hf_causal_lm_artifact" in st.__all__
    assert "prepare_hf_finetune_model" in st.__all__
    assert callable(st.hf_finetune_adapter_config)


def test_artifact_report_detects_local_adapter_and_rejects_incomplete(
    tmp_path: Path,
) -> None:
    adapter = _write_adapter(tmp_path / "adapter")

    report = st.hf_causal_lm_artifact_report(adapter)

    assert report["status"] == "ready"
    assert report["artifact_kind"] == "peft_adapter"
    assert report["base_model_name_or_path"] == "org/base"
    assert report["tokenizer_source"] == str(adapter)
    assert report["adapter_weights_present"] is True
    assert "kind=peft_adapter" in st.hf_causal_lm_artifact_lines(report)[0]

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "org/base"}',
        encoding="utf-8",
    )
    invalid = st.hf_causal_lm_artifact_report(incomplete)
    assert invalid["status"] == "invalid"
    assert "weight" in " ".join(invalid["errors"])


def test_artifact_loader_reconstructs_local_trainable_adapter(tmp_path: Path) -> None:
    adapter = _write_adapter(tmp_path / "adapter")

    model, tokenizer, config, report = st.load_hf_causal_lm_artifact(
        adapter,
        is_trainable=True,
        transformers_module=_FakeTransformers,
        peft_module=_ArtifactPeft,
        loader_kwargs={"local_files_only": True, "trust_remote_code": False},
    )

    assert isinstance(config, _ArtifactConfig)
    assert tokenizer.source == str(adapter)
    assert model.source == "org/base"
    assert model.adapter_source == str(adapter)
    assert model.adapter_trainable is True
    assert report["adapter_loaded"] is True
    assert report["adapter_trainable"] is True
    assert report["active_adapter"] == "default"
    assert report["runtime_adapter_config"]["r"] == 4
    assert report["resolved_base_model_name_or_path"] == "org/base"
    assert report["resolved_tokenizer_source_kind"] == "adapter_artifact"
    summary = st.summarize_hf_causal_lm_artifact(report)
    assert summary["artifact_kind"] == "peft_adapter"
    assert summary["adapter_loaded"] is True
    assert summary["parameter_count"] == 108
    assert summary["trainable_parameter_count"] == 8
    assert _FakeTransformers.config_calls[0][0] == "org/base"
    assert _FakeTransformers.model_calls[0][0] == "org/base"
    assert _ArtifactPeft.model_calls[0][2] == {"local_files_only": True}


def test_artifact_loader_keeps_full_model_path_peft_free(tmp_path: Path) -> None:
    full_model = tmp_path / "full-model"
    full_model.mkdir()
    (full_model / "config.json").write_text("{}\n", encoding="utf-8")
    (full_model / "model.safetensors").write_bytes(b"model")
    (full_model / "tokenizer.json").write_text("{}\n", encoding="utf-8")

    model, tokenizer, _config, report = st.load_hf_causal_lm_artifact(
        full_model,
        transformers_module=_FakeTransformers,
        peft_module=types.SimpleNamespace(),
        loader_kwargs={"local_files_only": True},
    )

    assert model.source == str(full_model)
    assert tokenizer.source == str(full_model)
    assert report["artifact_kind"] == "full_model"
    assert report["adapter_loaded"] is False
    assert report["peft_version"] is None


def test_artifact_loader_resolves_remote_adapter_metadata() -> None:
    model, tokenizer, _config, report = st.load_hf_causal_lm_artifact(
        "org/remote-adapter",
        artifact_kind="peft-adapter",
        load_model=False,
        transformers_module=_FakeTransformers,
        peft_module=_ArtifactPeft,
        loader_kwargs={"local_files_only": True, "trust_remote_code": True},
    )

    assert model is None
    assert tokenizer.source == "org/remote-base"
    assert report["runtime_resolution_required"] is True
    assert report["resolved_base_model_name_or_path"] == "org/remote-base"
    assert report["runtime_adapter_config"] == {
        "base_model_name_or_path": "org/remote-base"
    }
    assert _ArtifactPeft.config_calls[0][1] == {"local_files_only": True}


def test_artifact_loader_pins_observed_commit_before_model_load() -> None:
    commit = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
    _FakeTransformers.config_commit = commit

    model, _tokenizer, _config, report = st.load_hf_causal_lm_artifact(
        "org/base",
        transformers_module=_FakeTransformers,
        loader_kwargs={"local_files_only": True},
        tokenizer_kwargs={"revision": "main"},
        model_kwargs={"revision": "main"},
    )

    assert model.source == "org/base"
    assert report["base_model_commit_pin_applied"] is True
    assert report["base_model_effective_revision"] == commit
    assert report["runtime_identity_pre_model"]["status"] == "ready"
    assert report["runtime_identity_after_model"]["status"] == "ready"
    assert (
        report["runtime_identity_pre_model"]["observed_identity_id"]
        == report["runtime_identity_after_model"]["observed_identity_id"]
    )
    assert _FakeTransformers.tokenizer_calls[0][1]["revision"] == commit
    assert _FakeTransformers.model_calls[0][1]["revision"] == commit


def test_artifact_loader_only_pins_valid_commits() -> None:
    _FakeTransformers.config_commit = "main"

    _model, _tokenizer, _config, report = st.load_hf_causal_lm_artifact(
        "org/base",
        transformers_module=_FakeTransformers,
    )

    assert report["base_model_commit_pin_applied"] is False
    assert report["resolved_base_model_commit"] is None
    assert report["runtime_identity_pre_model"]["status"] == "evidence_incomplete"
    assert "revision" not in _FakeTransformers.tokenizer_calls[0][1]
    assert "revision" not in _FakeTransformers.model_calls[0][1]


def test_artifact_loader_does_not_apply_base_commit_to_explicit_tokenizer() -> None:
    commit = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
    _FakeTransformers.config_commit = commit

    st.load_hf_causal_lm_artifact(
        "org/base",
        tokenizer_name_or_path="org/tokenizer",
        transformers_module=_FakeTransformers,
    )

    assert "revision" not in _FakeTransformers.tokenizer_calls[0][1]
    assert _FakeTransformers.model_calls[0][1]["revision"] == commit


def test_artifact_loader_blocks_commit_drift_before_model_weights() -> None:
    original_commit = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
    _FakeTransformers.config_commit = original_commit
    _model, _tokenizer, _config, original = st.load_hf_causal_lm_artifact(
        "org/base",
        transformers_module=_FakeTransformers,
    )
    expected_id = original["runtime_identity_after_model"]["observed_identity_id"]
    _FakeTransformers.config_calls.clear()
    _FakeTransformers.tokenizer_calls.clear()
    _FakeTransformers.model_calls.clear()
    _FakeTransformers.config_commit = "0" * 40

    with pytest.raises(st.HfCausalLmRuntimeIdentityError) as raised:
        st.load_hf_causal_lm_artifact(
            "org/base",
            transformers_module=_FakeTransformers,
            expected_runtime_identity_id=expected_id,
        )

    assert raised.value.report["status"] == "blocked"
    assert raised.value.report["expected_identity_verified"] is False
    assert _FakeTransformers.tokenizer_calls[0][1]["revision"] == "0" * 40
    assert _FakeTransformers.model_calls == []


def test_export_adapter_merges_atomically_into_full_model(tmp_path: Path) -> None:
    adapter = _write_adapter(tmp_path / "adapter")
    output = tmp_path / "merged"

    report = st.export_hf_merged_causal_lm(
        adapter,
        output,
        transformers_module=_FakeTransformers,
        peft_module=_ArtifactPeft,
        loader_kwargs={"local_files_only": True},
    )

    assert report["status"] == "exported"
    assert report["load_report"]["adapter_merged"] is True
    assert (output / "config.json").is_file()
    assert (output / "model.safetensors").is_file()
    assert (output / "tokenizer.json").is_file()
    assert (output / "spiraltorch-hf-merged-export.json").is_file()
    assert not (output / "adapter_config.json").exists()
    assert "adapter_merged=True" in st.hf_merged_causal_lm_export_lines(report)[0]

    with pytest.raises(ValueError, match="absent or empty"):
        st.export_hf_merged_causal_lm(
            adapter,
            output,
            transformers_module=_FakeTransformers,
            peft_module=_ArtifactPeft,
        )


def test_adapter_export_cli_inspects_without_loading_runtime(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter = _write_adapter(tmp_path / "adapter")

    code = hf_cli.adapter_export_main(["--adapter", str(adapter), "--inspect-only"])

    assert code == 0
    assert "kind=peft_adapter" in capsys.readouterr().out


def test_adapter_config_uses_family_defaults_and_validates_ranges() -> None:
    config = st.hf_finetune_adapter_config(
        mode="peft",
        model_family="qwen2",
        rank=8,
        alpha=16,
        dropout=0.1,
    )

    assert config["mode"] == "lora"
    assert config["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert config["target_modules_source"] == "model_family_default"
    with pytest.raises(ValueError, match="dropout"):
        st.hf_finetune_adapter_config(mode="lora", dropout=1.0)
    with pytest.raises(ValueError, match="rank"):
        st.hf_finetune_adapter_config(mode="lora", rank=0)


def test_lora_target_report_checks_real_module_suffixes() -> None:
    report = st.hf_finetune_lora_target_report(_Model(), model_family="qwen2")

    assert report["targets_verified"] is True
    assert report["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert report["matched_module_count"] == 4
    with pytest.raises(ValueError, match="none of the LoRA target modules"):
        st.hf_finetune_lora_target_report(
            _Model(),
            target_modules=["missing_projection"],
        )
    with pytest.raises(ValueError, match="did not all match"):
        st.hf_finetune_lora_target_report(
            _Model(),
            target_modules=["q_proj", "missing_projection"],
        )


def test_prepare_full_finetune_keeps_model_and_applies_checkpointing() -> None:
    model = _Model()
    prepared, report = st.prepare_hf_finetune_model(
        model,
        mode="full",
        gradient_checkpointing=True,
    )

    assert prepared is model
    assert report["adapter_attached"] is False
    assert report["parameter_report_after"]["trainable_parameter_count"] == 100
    assert report["gradient_checkpointing"]["enabled"] is True
    assert model.gradient_checkpointing is True
    assert model.input_require_grads is True
    assert model.config.use_cache is False


def test_prepare_lora_reports_trainable_adapter_and_resolved_targets() -> None:
    model = _Model()
    prepared, report = st.prepare_hf_finetune_model(
        model,
        mode="lora",
        model_family="qwen2",
        rank=8,
        alpha=16,
        dropout=0.1,
        gradient_checkpointing=True,
        peft_module=_FakePeft,
    )

    assert prepared is model
    assert report["adapter_attached"] is True
    assert report["peft_version"] == "test-peft"
    assert report["target_report"]["matched_module_count"] == 4
    assert report["parameter_report_before"]["trainable_parameter_count"] == 100
    assert report["parameter_report_after"]["trainable_parameter_count"] == 16
    assert report["parameter_report_after"][
        "trainable_parameter_ratio"
    ] == pytest.approx(16 / 116)
    assert _FakePeft.last_config.kwargs["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
    assert _FakePeft.last_config.kwargs["r"] == 8
    assert _FakePeft.last_config.kwargs["lora_alpha"] == 16.0


def test_prepare_preloaded_adapter_reuses_trainable_peft_model() -> None:
    model = _ArtifactModel("org/base")
    model = _ArtifactPeft.PeftModel.from_pretrained(
        model,
        "org/adapter",
        is_trainable=True,
    )

    prepared, report = st.prepare_hf_finetune_model(
        model,
        mode="lora",
        gradient_checkpointing=True,
        preloaded_adapter=True,
        peft_module=_ArtifactPeft,
    )

    assert prepared is model
    assert report["adapter_preloaded"] is True
    assert report["adapter_attached_now"] is False
    assert report["adapter_origin"] == "artifact"
    assert report["adapter_config_source"] == "loaded_artifact"
    assert report["adapter_config_applied"] is False
    assert report["active_adapter"] == "default"
    assert report["runtime_adapter_config"]["r"] == 4
    assert report["parameter_report_after"]["trainable_parameter_count"] == 8
    assert report["gradient_checkpointing"]["enabled"] is True

    with pytest.raises(ValueError, match="already contains a PEFT adapter"):
        st.prepare_hf_finetune_model(
            model,
            mode="lora",
            peft_module=_ArtifactPeft,
        )
    with pytest.raises(ValueError, match="full fine-tuning cannot reuse"):
        st.prepare_hf_finetune_model(
            model,
            mode="full",
            preloaded_adapter=True,
            peft_module=_ArtifactPeft,
        )


def test_prepare_preloaded_adapter_canonicalizes_unordered_runtime_config() -> None:
    class RuntimeConfig:
        def to_dict(self):
            return {
                "peft_type": "LORA",
                "target_modules": {"c_proj", "c_attn"},
                "nested": {"modules_to_save": frozenset({"lm_head", "embed"})},
            }

    model = _Model("gpt2")
    model.peft_config = {"default": RuntimeConfig()}
    model.active_adapter = "default"

    prepared, report = st.prepare_hf_finetune_model(
        model,
        mode="lora",
        preloaded_adapter=True,
        peft_module=_FakePeft,
    )

    assert prepared is model
    assert report["runtime_adapter_config"]["target_modules"] == [
        "c_attn",
        "c_proj",
    ]
    assert report["runtime_adapter_config"]["nested"]["modules_to_save"] == [
        "embed",
        "lm_head",
    ]
