from __future__ import annotations

import math
import importlib.util
from pathlib import Path
from typing import Any

import pytest

import spiraltorch as st


torch = pytest.importorskip("torch")
BRIDGE = Path(__file__).resolve().parents[1] / "examples" / "hf_gpt2_finetune_bridge.py"


def _load_bridge() -> Any:
    spec = importlib.util.spec_from_file_location(
        "spiraltorch_hf_repetition_unlikelihood_bridge_test",
        BRIDGE,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load HF fine-tune bridge")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recipe(strength: float = 0.1) -> dict[str, object]:
    return st.hf_repetition_unlikelihood_recipe_contract(
        strength=strength,
        ngram_order=3,
        context_window=16,
        max_candidates_per_position=8,
    )


def _batch_plan(strength: float = 0.1) -> dict[str, object]:
    return {
        "plan_id": "sha256:" + "a" * 64,
        "request": {"config": dict(_recipe(strength)["config"])},
        "positions": [
            {
                "sequence_index": 0,
                "prediction_index": 4,
                "target_index": 5,
                "target_token_id": 4,
                "prefix_token_ids": [1, 2],
                "candidates": [
                    {
                        "token_id": 3,
                        "occurrence_count": 1,
                        "most_recent_distance": 3,
                    }
                ],
            }
        ],
        "aggregate": {"active_position_count": 1, "candidate_count": 1},
    }


def test_hf_repetition_unlikelihood_collator_plans_from_cpu_labels() -> None:
    observed: dict[str, Any] = {}

    def base_collator(_features: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "input_ids": torch.tensor([[1, 2, 3, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "labels": torch.tensor([[1, 2, -100, -100]]),
        }

    def planner(**kwargs: Any) -> dict[str, object]:
        observed.update(kwargs)
        return _batch_plan()

    collator = st.HfRepetitionUnlikelihoodCollator(
        base_collator,
        strength=0.1,
        ngram_order=3,
        context_window=16,
        max_candidates_per_position=8,
        planner=planner,
    )

    batch = collator([{"input_ids": [1, 2, 3, 0]}])

    assert observed["sequences"] == [
        {
            "token_ids": [1, 2, 3, 0],
            "token_mask": [True, True, True, False],
            "label_mask": [True, True, False, False],
        }
    ]
    metadata = batch[st.HF_REPETITION_UNLIKELIHOOD_BATCH_PLAN_KEY]
    assert isinstance(metadata, st.HfRepetitionUnlikelihoodBatchPlan)


class _BaseTrainer:
    def __init__(self, *, model: Any, **_kwargs: Any) -> None:
        self.model = model

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
    ) -> Any:
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        logits = torch.zeros(1, 6, 5)
        logits[0, 4, 3] = math.log(3.0)
        self.logits = torch.nn.Parameter(logits)

    def forward(self, **_inputs: Any) -> dict[str, Any]:
        return {
            "loss": self.logits.sum() * 0.0 + 2.0,
            "logits": self.logits,
        }


def _trainer(strength: float = 0.1) -> tuple[Any, _Model]:
    model = _Model()
    trainer_class = st.hf_repetition_unlikelihood_trainer_class(_BaseTrainer)
    trainer = trainer_class(
        model=model,
        zspace_repetition_unlikelihood_recipe=_recipe(strength),
    )
    return trainer, model


def _inputs(strength: float = 0.1) -> dict[str, object]:
    return {
        "labels": torch.tensor([[1, 2, 3, 1, 2, 4]]),
        st.HF_REPETITION_UNLIKELIHOOD_BATCH_PLAN_KEY: (
            st.HfRepetitionUnlikelihoodBatchPlan(_batch_plan(strength))
        ),
    }


def test_hf_trainer_adds_position_balanced_unlikelihood_during_training() -> None:
    trainer, model = _trainer()
    model.train()

    loss = trainer.compute_loss(model, _inputs())
    expected_auxiliary = -math.log(1.0 - 3.0 / 7.0)

    assert float(loss.detach()) == pytest.approx(2.0 + 0.1 * expected_auxiliary)
    assert trainer.model_accepts_loss_kwargs is False
    loss.backward()
    assert float(model.logits.grad[0, 4, 3]) > 0.0
    receipt = trainer.zspace_repetition_unlikelihood_receipt()
    assert receipt["status"] == "ready"
    assert receipt["training_batch_count"] == 1
    assert receipt["active_position_count"] == 1
    assert receipt["candidate_count"] == 1
    assert receipt["mean_auxiliary_loss"] == pytest.approx(expected_auxiliary)
    assert receipt["efficacy_claim_ready"] is False


def test_hf_trainer_keeps_evaluation_loss_as_plain_causal_lm_loss() -> None:
    trainer, model = _trainer()
    model.eval()

    loss = trainer.compute_loss(model, _inputs())

    assert float(loss.detach()) == pytest.approx(2.0)
    assert trainer.zspace_repetition_unlikelihood_receipt()["status"] == (
        "not_observed"
    )


def test_hf_trainer_rejects_a_plan_from_another_recipe() -> None:
    trainer, model = _trainer(strength=0.1)
    model.train()

    with pytest.raises(RuntimeError, match="does not match the training recipe"):
        trainer.compute_loss(model, _inputs(strength=0.2))


def test_hf_bridge_seals_the_objective_in_training_recipe_identity() -> None:
    bridge = _load_bridge()
    args = bridge.parse_args(
        [
            "--training-recipe-only",
            "--zspace-repetition-unlikelihood-strength",
            "0.1",
            "--zspace-repetition-unlikelihood-ngram-order",
            "3",
            "--zspace-repetition-unlikelihood-context-window",
            "128",
            "--zspace-repetition-unlikelihood-max-candidates",
            "8",
        ]
    )

    contract = bridge._training_recipe_trainer_contract(args)

    assert contract["trainer"] == "spiraltorch.HfRepetitionUnlikelihoodTrainer"
    assert contract["data_collator"]["class"] == (  # type: ignore[index]
        "spiraltorch.HfRepetitionUnlikelihoodCollator"
    )
    objective = contract["zspace_repetition_unlikelihood"]
    assert objective["enabled"] is True  # type: ignore[index]
    assert objective["config"]["strength"] == 0.1  # type: ignore[index]


def test_hf_bridge_preserves_the_stock_trainer_when_disabled() -> None:
    bridge = _load_bridge()
    args = bridge.parse_args(["--training-recipe-only"])

    contract = bridge._training_recipe_trainer_contract(args)

    assert contract["trainer"] == "transformers.Trainer"
    assert contract["data_collator"]["class"] == (  # type: ignore[index]
        "transformers.DataCollatorForLanguageModeling"
    )
    objective = contract["zspace_repetition_unlikelihood"]
    assert objective["enabled"] is False  # type: ignore[index]
    assert objective["data_collator"] is None  # type: ignore[index]
    assert objective["trainer"] is None  # type: ignore[index]


def test_hf_bridge_rejects_invalid_objective_configuration() -> None:
    bridge = _load_bridge()

    with pytest.raises(SystemExit):
        bridge.parse_args(
            [
                "--training-recipe-only",
                "--zspace-repetition-unlikelihood-strength",
                "-0.1",
            ]
        )
    with pytest.raises(SystemExit):
        bridge.parse_args(
            [
                "--zspace-repetition-unlikelihood-strength",
                "0.1",
            ]
        )


def test_hf_recipe_contract_rejects_noncanonical_numeric_types() -> None:
    with pytest.raises((TypeError, ValueError)):
        st.hf_repetition_unlikelihood_recipe_contract(
            strength=0.1,
            ngram_order=3.5,  # type: ignore[arg-type]
            context_window=16,
            max_candidates_per_position=8,
        )
