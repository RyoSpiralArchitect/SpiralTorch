from __future__ import annotations

from enum import Enum

from spiraltorch.hf_training_identity import (
    HF_FINETUNE_TRAINING_RECIPE_IDENTITY_SCHEMA,
    hf_finetune_training_recipe_identity_lines,
    hf_finetune_training_recipe_identity_report,
)


class _Strategy(Enum):
    STEPS = "steps"


class _Arguments:
    def __init__(self, **overrides: object) -> None:
        self.values: dict[str, object] = {
            "output_dir": "/tmp/first",
            "logging_dir": "/tmp/first/logs",
            "run_name": "first",
            "report_to": ["none"],
            "save_steps": 250,
            "learning_rate": 5.0e-5,
            "weight_decay": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1.0e-8,
            "max_grad_norm": 1.0,
            "optim": "adamw_torch",
            "lr_scheduler_type": "linear",
            "warmup_steps": 0,
            "max_steps": 1024,
            "num_train_epochs": 1.0,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "logging_strategy": "steps",
            "logging_steps": 25,
            "seed": 13,
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": False,
            "gradient_checkpointing": True,
            "eval_strategy": _Strategy.STEPS,
            "eval_steps": 128,
            "fp16": False,
            "bf16": False,
        }
        self.values.update(overrides)

    def to_dict(self) -> dict[str, object]:
        return dict(self.values)


def _model_prepare(**overrides: object) -> dict[str, object]:
    report: dict[str, object] = {
        "mode": "lora",
        "model_family": "gpt2",
        "adapter_attached": True,
        "adapter_attached_now": True,
        "adapter_preloaded": False,
        "adapter_origin": "new",
        "active_adapter": None,
        "adapter_config_source": "request",
        "adapter_config_applied": True,
        "adapter_config": {
            "mode": "lora",
            "rank": 16,
            "alpha": 32.0,
            "dropout": 0.05,
            "target_modules": ["c_attn", "c_proj"],
        },
        "target_report": {"target_modules": ["c_attn", "c_proj"]},
        "gradient_checkpointing": {
            "requested": True,
            "enabled": True,
            "enable_input_require_grads": True,
            "use_cache_before": True,
            "use_cache_after": False,
        },
        "parameter_report_after": {
            "parameter_count": 124_000_000,
            "trainable_parameter_count": 2_359_296,
            "frozen_parameter_count": 121_640_704,
            "trainable_parameter_ratio": 0.0190265806,
        },
    }
    report.update(overrides)
    return report


def _report(
    arguments: object | None = None,
    **overrides: object,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "model_prepare_report": _model_prepare(),
        "model_dtype_report": {
            "policy": "auto",
            "train_requested": True,
            "dtype_before": "torch.float16",
            "dtype_after": "torch.float32",
            "cast_status": "cast_float32",
        },
        "checkpoint_resume_report": {},
        "trainer_contract": {
            "trainer": "transformers.Trainer",
            "data_collator": "transformers.DataCollatorForLanguageModeling",
            "mlm": False,
            "trace_callback": {
                "enabled": True,
                "stop_on_nonfinite_loss": True,
                "loss_guard_threshold": 1.0e6,
            },
        },
    }
    kwargs.update(overrides)
    return hf_finetune_training_recipe_identity_report(
        arguments or _Arguments(),
        **kwargs,
    )


def test_training_recipe_identity_is_stable_across_operational_paths() -> None:
    first = _report()
    relocated = _report(
        _Arguments(
            output_dir="/different/output",
            logging_dir="/different/logs",
            run_name="relocated",
            report_to=["tensorboard"],
            save_steps=5,
        )
    )

    assert first["schema"] == HF_FINETUNE_TRAINING_RECIPE_IDENTITY_SCHEMA
    assert first["status"] == "ready"
    assert first["path_independent"] is True
    assert first["observed_identity_id"] == relocated["observed_identity_id"]
    payload = first["identity_payload"]
    assert isinstance(payload, dict)
    assert "output_dir" not in payload["training_arguments"]
    assert "save_steps" not in payload["training_arguments"]
    assert "eval_strategy" in payload["training_arguments"]
    assert "status=ready" in hf_finetune_training_recipe_identity_lines(first)[0]


def test_training_recipe_identity_detects_optimizer_schedule_and_seed_drift() -> None:
    baseline = _report()
    expected = str(baseline["observed_identity_id"])

    for arguments in (
        _Arguments(learning_rate=1.0e-5),
        _Arguments(weight_decay=0.1),
        _Arguments(lr_scheduler_type="cosine"),
        _Arguments(max_steps=2048),
        _Arguments(gradient_accumulation_steps=4),
        _Arguments(seed=29),
    ):
        drift = _report(arguments, expected_identity_id=expected)
        assert drift["status"] == "blocked"
        assert drift["expected_identity_verified"] is False


def test_training_recipe_identity_detects_model_dtype_and_resume_drift() -> None:
    baseline = _report()
    expected = str(baseline["observed_identity_id"])
    lora_config = dict(_model_prepare()["adapter_config"])
    lora_config["rank"] = 32
    lora_drift = _report(
        model_prepare_report=_model_prepare(adapter_config=lora_config),
        expected_identity_id=expected,
    )
    dtype_drift = _report(
        model_dtype_report={
            "policy": "native",
            "train_requested": True,
            "dtype_before": "torch.float16",
            "dtype_after": "torch.float16",
            "cast_status": "not_requested",
        },
        expected_identity_id=expected,
    )
    resume_drift = _report(
        checkpoint_resume_report={
            "trainer_state_present": True,
            "optimizer_state_present": True,
            "scheduler_state_present": True,
            "rng_state_present": True,
            "exact_state_available": True,
            "global_step": 1024,
            "saved_max_steps": 1024,
            "requested_max_steps": 2048,
            "extension_requested": True,
            "recommendation": "exact_trainer_resume",
        },
        expected_identity_id=expected,
    )

    assert lora_drift["status"] == "blocked"
    assert dtype_drift["status"] == "blocked"
    assert resume_drift["status"] == "blocked"


def test_training_recipe_identity_ignores_preloaded_adapter_source_path() -> None:
    runtime_config = {
        "base_model_name_or_path": "/models/first",
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 32.0,
        "lora_dropout": 0.05,
        "target_modules": ["c_attn", "c_proj"],
    }
    first = _report(
        model_prepare_report=_model_prepare(
            adapter_config_source="loaded_artifact",
            runtime_adapter_config=runtime_config,
        )
    )
    relocated_config = dict(runtime_config)
    relocated_config["base_model_name_or_path"] = "/relocated/base"
    relocated = _report(
        model_prepare_report=_model_prepare(
            adapter_config_source="loaded_artifact",
            runtime_adapter_config=relocated_config,
        )
    )
    rank_drift_config = dict(runtime_config)
    rank_drift_config["r"] = 32
    rank_drift = _report(
        model_prepare_report=_model_prepare(
            adapter_config_source="loaded_artifact",
            runtime_adapter_config=rank_drift_config,
        )
    )

    assert first["observed_identity_id"] == relocated["observed_identity_id"]
    assert first["observed_identity_id"] != rank_drift["observed_identity_id"]


def test_training_recipe_identity_ignores_unused_full_ft_lora_knobs() -> None:
    base_config = {
        "mode": "full",
        "enabled": False,
        "model_family": "gpt2",
        "rank": 16,
        "alpha": 32.0,
        "dropout": 0.05,
        "gradient_checkpointing": False,
    }
    baseline = _report(
        model_prepare_report=_model_prepare(
            mode="full",
            adapter_attached=False,
            adapter_config=base_config,
        )
    )
    unused_knob_config = dict(base_config, rank=128, alpha=256.0, dropout=0.4)
    unused_knob_drift = _report(
        model_prepare_report=_model_prepare(
            mode="full",
            adapter_attached=False,
            adapter_config=unused_knob_config,
        )
    )

    assert baseline["observed_identity_id"] == unused_knob_drift["observed_identity_id"]


def test_training_recipe_identity_canonicalizes_eval_strategy_alias() -> None:
    modern = _report(_Arguments(eval_strategy=_Strategy.STEPS))
    legacy_values = _Arguments().to_dict()
    legacy_values.pop("eval_strategy")
    legacy_values["evaluation_strategy"] = "steps"
    legacy = _report(legacy_values)

    assert modern["observed_identity_id"] == legacy["observed_identity_id"]


def test_training_recipe_identity_verifies_replay_and_rejects_invalid_values() -> None:
    baseline = _report()
    expected = str(baseline["observed_identity_id"])
    replay = _report(expected_identity_id=expected)
    invalid_value = _report(_Arguments(optim=object()))
    incomplete_model = _report(model_prepare_report={})

    assert replay["status"] == "ready"
    assert replay["expected_identity_verified"] is True
    assert invalid_value["status"] == "blocked"
    assert invalid_value["observed_identity_id"] is None
    assert "unsupported value type" in str(invalid_value["errors"])
    assert incomplete_model["status"] == "blocked"
    assert "must resolve to full or lora" in str(incomplete_model["errors"])
