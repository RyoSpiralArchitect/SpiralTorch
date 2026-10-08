"""Real Trainer updates using a local random tiny model, not a quality benchmark."""

from __future__ import annotations

import copy
import json

import pytest

import spiraltorch as st

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate")


def _recipe(*, end_update=4, candidate_source="prior_continuation"):
    return st.hf_repetition_unlikelihood_recipe_contract(
        strength=0.2,
        ngram_order=3,
        context_window=16,
        max_candidates_per_position=8,
        candidate_source=candidate_source,
        objective_control={
            "normalization": "eligible_targets",
            "schedule": {
                "kind": "linear_decay",
                "start_update": 0,
                "end_update": end_update,
                "final_scale": 0.0,
            },
        },
    )


def _model(*, lora=False, family="gpt2"):
    torch.manual_seed(19)
    config = transformers.GPT2Config(
        vocab_size=8,
        n_positions=8,
        n_embd=8,
        n_layer=1,
        n_head=1,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
        bos_token_id=0,
        eos_token_id=7,
        pad_token_id=0,
    )
    model = transformers.GPT2LMHeadModel(config)
    if family == "llama":
        model = transformers.LlamaForCausalLM(
            transformers.LlamaConfig(
                vocab_size=8,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=8,
                attention_dropout=0.0,
                use_cache=False,
                bos_token_id=0,
                eos_token_id=7,
                pad_token_id=0,
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
                lora_dropout=0.0,
                target_modules=["c_attn"] if family == "gpt2" else ["q_proj", "v_proj"],
                fan_in_fan_out=family == "gpt2",
            ),
        )
    return model


def _dataset():
    return [
        {"input_ids": [1, 2, 3, 1, 2, 4], "attention_mask": [1] * 6, "labels": labels}
        for labels in [
            [1, 2, 3, 1, 2, 4],
            [-100] * 5 + [4],
            [-100, -100, 3, -100, -100, 4],
            [1, 2, 3, 1, 2, -100],
        ]
    ]


def _collator(candidate_source="prior_continuation"):
    return st.HfRepetitionUnlikelihoodCollator(
        transformers.default_data_collator,
        strength=0.2,
        ngram_order=3,
        context_window=16,
        max_candidates_per_position=8,
        candidate_source=candidate_source,
    )


class _Trainer(st.hf_repetition_unlikelihood_trainer_class(transformers.Trainer)):
    def _get_train_sampler(self, *args, **kwargs):
        return torch.utils.data.SequentialSampler(self.train_dataset)

    def compute_loss(self, *args, **kwargs):
        result = super().compute_loss(*args, **kwargs)
        if self.model.training:
            self.objective_slots.append(
                self.zspace_repetition_unlikelihood_receipt()["last_objective_control"]
            )
        return result


def _trainer(path, *, model=None, recipe=None, callbacks=None, steps=4, save=True):
    recipe = recipe or _recipe()
    args = transformers.TrainingArguments(
        output_dir=str(path),
        use_cpu=True,
        report_to=[],
        disable_tqdm=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=steps,
        optim="sgd",
        learning_rate=0.01,
        lr_scheduler_type="constant",
        max_grad_norm=0.0,
        weight_decay=0.0,
        save_strategy="steps" if save else "no",
        save_steps=2,
        logging_strategy="no",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        seed=37,
        data_seed=37,
    )
    trainer = _Trainer(
        model=model if model is not None else _model(),
        args=args,
        train_dataset=_dataset(),
        eval_dataset=_dataset(),
        data_collator=_collator(recipe["config"]["candidate_source"]),
        zspace_repetition_unlikelihood_recipe=recipe,
        callbacks=callbacks,
    )
    trainer.objective_slots = []
    return trainer


def test_real_accumulated_update_matches_mean_of_masked_microbatch_objectives(tmp_path):
    model = _model()
    oracle = copy.deepcopy(model)
    optimizer = torch.optim.SGD(oracle.parameters(), lr=0.01)
    oracle.train()
    for sample, eligible_count in zip(_dataset()[:2], [4, 1], strict=True):
        batch = transformers.default_data_collator([sample])
        output = oracle(**batch)
        # Candidate token 3 follows the repeated [1, 2] prefix at prediction 4.
        probability = output.logits[0, 4].float().softmax(-1)[3].clamp(max=1.0 - 1e-6)
        aux = -torch.log1p(-probability)
        ((output.loss + 0.2 / eligible_count * aux) / 2).backward()
    optimizer.step()
    trainer = _trainer(tmp_path, model=model, steps=1, save=False)
    trainer.train()
    assert [row["completed_update_slots"] for row in trainer.objective_slots] == [0, 0]
    assert [row["effective_strength"] for row in trainer.objective_slots] == [0.05, 0.2]
    for actual, expected in zip(model.parameters(), oracle.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


class _StopAfterTwo(transformers.TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 2:
            control.should_training_stop = True
        return control


@pytest.mark.parametrize(
    "family,lora", [("gpt2", False), ("gpt2", True), ("llama", True)]
)
def test_real_trainer_checkpoint_restores_objective_clock_and_all_weights(
    tmp_path, family, lora
):
    full = _trainer(tmp_path / "full", model=_model(lora=lora, family=family))
    initial = copy.deepcopy(full.model.state_dict())
    full.train()
    assert any(
        not torch.equal(value, initial[name])
        for name, value in full.model.state_dict().items()
    )
    if lora:
        for name, parameter in full.model.named_parameters():
            if not parameter.requires_grad:
                assert torch.equal(parameter, initial[name]), name
    prefix = _trainer(
        tmp_path / "split",
        model=_model(lora=lora, family=family),
        callbacks=[_StopAfterTwo()],
    )
    prefix.train()
    resumed = _trainer(tmp_path / "split", model=_model(lora=lora, family=family))
    resumed.train(resume_from_checkpoint=True)
    assert resumed.state.global_step == full.state.global_step == 4
    assert prefix.objective_slots + resumed.objective_slots == full.objective_slots
    assert [row["completed_update_slots"] for row in resumed.objective_slots] == [
        2,
        2,
        3,
        3,
    ]
    for name, value in full.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[name]), name
    # Evaluation remains the stock causal-LM loss with its original normalization.
    plain = transformers.Trainer(
        model=copy.deepcopy(full.model),
        args=full.args,
        eval_dataset=_dataset(),
        data_collator=transformers.default_data_collator,
    )
    assert full.evaluate()["eval_loss"] == pytest.approx(
        plain.evaluate()["eval_loss"], abs=1e-7
    )


@pytest.mark.parametrize(
    "mutation", ["recipe", "clock", "missing", "accumulation", "deprecated_alias"]
)
def test_resume_rejects_changed_recipe_or_incomplete_checkpoint_before_loading(
    tmp_path, mutation
):
    trainer = _trainer(tmp_path / "source", callbacks=[_StopAfterTwo()])
    trainer.train()
    checkpoint = tmp_path / "source" / "checkpoint-2"
    marker = checkpoint / "spiraltorch-repetition-objective.json"
    recipe = _recipe(end_update=8) if mutation == "recipe" else _recipe()
    if mutation == "clock":
        data = json.loads(marker.read_text())
        data["completed_update_slots"] = 1
        marker.write_text(json.dumps(data))
    elif mutation == "missing":
        marker.unlink()
    resumed = _trainer(tmp_path / "destination", recipe=recipe)
    if mutation == "accumulation":
        resumed.args.gradient_accumulation_steps = 4
    if mutation == "deprecated_alias":
        marker.unlink()
    before = copy.deepcopy(resumed.model.state_dict())
    with pytest.raises((ValueError, FileNotFoundError)):
        if mutation == "deprecated_alias":
            resumed.train(model_path=str(checkpoint))
        else:
            resumed.train(resume_from_checkpoint=str(checkpoint))
    for name, value in before.items():
        assert torch.equal(value, resumed.model.state_dict()[name]), name


def test_controlled_trainer_rejects_model_only_checkpoint_configuration(tmp_path):
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        save_only_model=True,
        report_to=[],
    )
    with pytest.raises(ValueError, match="full Trainer checkpoints"):
        _Trainer(
            model=_model(), args=args, zspace_repetition_unlikelihood_recipe=_recipe()
        )


def test_periodic_model_proposals_drive_controlled_lora_updates(tmp_path):
    trainer = _trainer(
        tmp_path,
        model=_model(lora=True),
        steps=2,
        save=False,
        recipe=_recipe(candidate_source="model_topk_periodic"),
    )
    trainer.train_dataset = [
        {
            "input_ids": [1, 2, 1, 2, 1, 4],
            "attention_mask": [1] * 6,
            "labels": [1, 2, 1, 2, 1, 4],
        }
        for _ in range(4)
    ]
    initial = copy.deepcopy(trainer.model.state_dict())
    trainer.train()
    receipt = trainer.zspace_repetition_unlikelihood_receipt()
    assert receipt["periodic_candidate_count"] > 0
    assert receipt["excluded_non_periodic_proposal_count"] > 0
    assert receipt["mean_weighted_auxiliary_loss"] > 0.0
    assert [row["completed_update_slots"] for row in trainer.objective_slots] == [
        0,
        0,
        1,
        1,
    ]
    assert any(
        not torch.equal(value, initial[name])
        for name, value in trainer.model.state_dict().items()
    )
