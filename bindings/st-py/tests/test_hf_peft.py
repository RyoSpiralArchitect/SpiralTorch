from __future__ import annotations

import types

import pytest

import spiraltorch as st


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
        return model


def test_peft_runtime_is_public_without_eager_peft_dependency() -> None:
    assert "hf_peft" in st.__all__
    assert "prepare_hf_finetune_model" in st.__all__
    assert callable(st.hf_finetune_adapter_config)


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
