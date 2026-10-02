from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate")


def test_alignment_masks_only_first_label_without_mutating_source():
    labels = torch.tensor([[1, 2, -100, 4], [5, 6, 7, 8]])
    original = labels.clone()
    inputs = labels.clone()
    base = {"input_ids": inputs, "labels": labels}
    aligned = st.HfCausalLabelAlignmentCollator(lambda _: base)([])
    torch.testing.assert_close(labels, original)
    torch.testing.assert_close(aligned["labels"][:, 1:], original[:, 1:])
    assert (aligned["labels"][:, 0] == -100).all()
    assert aligned["input_ids"] is inputs


@pytest.mark.parametrize(
    "labels", [None, [[1, 2]], torch.tensor([1, 2]), torch.tensor([[1]])]
)
def test_alignment_rejects_nontraining_label_shapes(labels):
    with pytest.raises((TypeError, ValueError)):
        st.HfCausalLabelAlignmentCollator(lambda _: {"labels": labels})([])


def _model(family, lora):
    torch.manual_seed(19)
    if family == "gpt2":
        model = transformers.GPT2LMHeadModel(
            transformers.GPT2Config(
                vocab_size=16,
                n_positions=16,
                n_embd=8,
                n_layer=1,
                n_head=1,
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
                use_cache=False,
            )
        )
    else:
        model = transformers.LlamaForCausalLM(
            transformers.LlamaConfig(
                vocab_size=16,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
                max_position_embeddings=16,
                use_cache=False,
                attention_dropout=0,
            )
        )
    if lora:
        peft = pytest.importorskip("peft")
        model = peft.get_peft_model(
            model,
            peft.LoraConfig(
                task_type="CAUSAL_LM",
                r=2,
                lora_alpha=4,
                lora_dropout=0,
                target_modules=["c_attn"] if family == "gpt2" else ["q_proj", "v_proj"],
                fan_in_fan_out=family == "gpt2",
            ),
        )
    return model


@pytest.mark.parametrize(
    "family,lora", [("gpt2", False), ("gpt2", True), ("llama", True)]
)
def test_zero_candidate_objective_matches_stock_trainer_updates_including_tail(
    tmp_path, family, lora
):
    model = _model(family, lora)
    # Five fixed-length examples give two complete accumulation groups and a tail.
    dataset = [
        {
            "input_ids": [1, 2, 3, 4, 5, 6],
            "labels": [1, 2, 3, 4, 5, 6],
            "attention_mask": [1] * 6,
        }
        for _ in range(5)
    ]
    aligned = st.HfCausalLabelAlignmentCollator(transformers.default_data_collator)
    recipe = st.hf_repetition_unlikelihood_recipe_contract(
        strength=0.1,
        ngram_order=3,
        context_window=16,
        max_candidates_per_position=8,
        objective_control={
            "normalization": "active_positions",
            "schedule": {"kind": "constant"},
        },
    )
    objective_collator = st.HfRepetitionUnlikelihoodCollator(
        aligned,
        strength=0.1,
        ngram_order=3,
        context_window=16,
        max_candidates_per_position=8,
    )

    def arguments(directory):
        return transformers.TrainingArguments(
            output_dir=str(directory),
            use_cpu=True,
            report_to=[],
            disable_tqdm=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            max_steps=3,
            optim="sgd",
            learning_rate=0.1,
            max_grad_norm=0,
            lr_scheduler_type="constant",
            save_strategy="no",
            logging_strategy="no",
            dataloader_pin_memory=False,
            seed=19,
            data_seed=19,
        )

    base = transformers.Trainer(
        model=copy.deepcopy(model),
        args=arguments(tmp_path / "base"),
        train_dataset=dataset,
        data_collator=aligned,
    )
    controlled = st.hf_repetition_unlikelihood_trainer_class(transformers.Trainer)(
        model=copy.deepcopy(model),
        args=arguments(tmp_path / "controlled"),
        train_dataset=dataset,
        data_collator=objective_collator,
        zspace_repetition_unlikelihood_recipe=recipe,
    )
    batches = [aligned([row]) for row in dataset[:2]]
    denominator = base._get_num_items_in_batch(batches, torch.device("cpu"))
    if denominator is not None:
        assert (
            int(denominator)
            == sum(int((b["labels"][:, 1:] != -100).sum()) for b in batches)
            == 10
        )
    base.train()
    controlled.train()
    assert base.state.global_step == controlled.state.global_step == 3
    assert controlled.zspace_repetition_unlikelihood_receipt()["candidate_count"] == 0
    assert any(
        not torch.equal(value, model.state_dict()[name])
        for name, value in base.model.state_dict().items()
    )
    for name, value in base.model.state_dict().items():
        torch.testing.assert_close(
            value, controlled.model.state_dict()[name], rtol=1e-6, atol=1e-7
        )


def test_bridge_alignment_is_opt_in_and_recorded_in_training_identity():
    path = Path(__file__).resolve().parents[1] / "examples/hf_gpt2_finetune_bridge.py"
    loader = importlib.util.spec_from_file_location("causal_alignment_bridge", path)
    bridge = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(bridge)
    legacy = bridge._training_recipe_trainer_contract(bridge.parse_args([]))
    aligned = bridge._training_recipe_trainer_contract(
        bridge.parse_args(["--causal-lm-mask-first-label"])
    )
    assert "causal_label_alignment" not in legacy["data_collator"]
    assert (
        aligned["data_collator"]["class"]
        == "spiraltorch.HfCausalLabelAlignmentCollator"
    )
    assert aligned["data_collator"]["causal_label_alignment"]["ignore_index"] == -100
