"""Hugging Face adapters for Rust-planned repetition-unlikelihood training."""

from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .repetition_unlikelihood import (
    ZSPACE_REPETITION_UNLIKELIHOOD_CONTRACT_VERSION,
    ZSPACE_REPETITION_UNLIKELIHOOD_DIFFERENTIATION_OWNER,
    ZSPACE_REPETITION_UNLIKELIHOOD_OBJECTIVE_RULE,
    ZSPACE_REPETITION_UNLIKELIHOOD_PROBABILITY_EPSILON,
    ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_BACKEND,
    ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_OWNER,
    validate_zspace_repetition_unlikelihood_plan,
    zspace_repetition_unlikelihood_plan,
)

HF_REPETITION_UNLIKELIHOOD_RECEIPT_SCHEMA = (
    "spiraltorch.hf_repetition_unlikelihood_receipt.v1"
)
HF_REPETITION_UNLIKELIHOOD_BATCH_PLAN_KEY = "_spiraltorch_repetition_unlikelihood_plan"

__all__ = [
    "HF_REPETITION_UNLIKELIHOOD_BATCH_PLAN_KEY",
    "HF_REPETITION_UNLIKELIHOOD_RECEIPT_SCHEMA",
    "HfRepetitionUnlikelihoodBatchPlan",
    "HfRepetitionUnlikelihoodCollator",
    "hf_repetition_unlikelihood_recipe_contract",
    "hf_repetition_unlikelihood_trainer_class",
]


@dataclass(frozen=True)
class HfRepetitionUnlikelihoodBatchPlan:
    """Opaque metadata kept off Trainer's recursive device-transfer path."""

    report: Mapping[str, Any]


def hf_repetition_unlikelihood_recipe_contract(
    *,
    strength: float,
    ngram_order: int,
    context_window: int,
    max_candidates_per_position: int,
) -> dict[str, object]:
    """Return the exact objective recipe embedded in training identity."""

    validation_plan = zspace_repetition_unlikelihood_plan(
        sequences=[
            {
                "token_ids": [0],
                "token_mask": [True],
                "label_mask": [True],
            }
        ],
        strength=strength,
        ngram_order=ngram_order,
        context_window=context_window,
        max_candidates_per_position=max_candidates_per_position,
    )
    request = validation_plan.get("request")
    config = request.get("config") if isinstance(request, Mapping) else None
    if not isinstance(config, Mapping):
        raise RuntimeError("Rust validation plan is missing its canonical config")
    canonical_config = dict(config)
    enabled = float(canonical_config["strength"]) > 0.0
    return {
        "schema": "spiraltorch.hf_repetition_unlikelihood_recipe.v1",
        "enabled": enabled,
        "semantic_owner": ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_OWNER,
        "semantic_backend": ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_BACKEND,
        "contract_version": ZSPACE_REPETITION_UNLIKELIHOOD_CONTRACT_VERSION,
        "differentiation_owner": ZSPACE_REPETITION_UNLIKELIHOOD_DIFFERENTIATION_OWNER,
        "objective_rule": ZSPACE_REPETITION_UNLIKELIHOOD_OBJECTIVE_RULE,
        "evaluation_loss": "causal_lm_loss_only",
        "gradient_accumulation_normalization": (
            "trainer_divides_the_combined_microbatch_mean" if enabled else None
        ),
        "model_accepts_loss_kwargs": False if enabled else None,
        "config": canonical_config,
        "data_collator": (
            "spiraltorch.HfRepetitionUnlikelihoodCollator" if enabled else None
        ),
        "trainer": ("spiraltorch.HfRepetitionUnlikelihoodTrainer" if enabled else None),
    }


class HfRepetitionUnlikelihoodCollator:
    """Decorate a causal-LM collator with one canonical Rust plan per batch."""

    def __init__(
        self,
        base_collator: Callable[[list[dict[str, Any]]], Mapping[str, Any]],
        *,
        strength: float,
        ngram_order: int,
        context_window: int,
        max_candidates_per_position: int,
        planner: Callable[..., dict[str, Any]] = zspace_repetition_unlikelihood_plan,
    ) -> None:
        self.base_collator = base_collator
        recipe = hf_repetition_unlikelihood_recipe_contract(
            strength=strength,
            ngram_order=ngram_order,
            context_window=context_window,
            max_candidates_per_position=max_candidates_per_position,
        )
        self.config = dict(recipe["config"])
        self._planner = planner

    @staticmethod
    def _rows(value: Any, *, label: str) -> list[list[Any]]:
        detached = value.detach() if callable(getattr(value, "detach", None)) else value
        cpu_value = (
            detached.cpu() if callable(getattr(detached, "cpu", None)) else detached
        )
        rows = (
            cpu_value.tolist()
            if callable(getattr(cpu_value, "tolist", None))
            else cpu_value
        )
        if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
            raise TypeError(f"{label} must be a rank-2 tensor-like value")
        return rows

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = dict(self.base_collator(features))
        token_rows = self._rows(batch.get("input_ids"), label="input_ids")
        label_rows = self._rows(batch.get("labels"), label="labels")
        attention = batch.get("attention_mask")
        attention_rows = (
            [[1] * len(row) for row in token_rows]
            if attention is None
            else self._rows(attention, label="attention_mask")
        )
        if not (
            len(token_rows) == len(label_rows) == len(attention_rows)
            and all(
                len(tokens) == len(labels) == len(mask)
                for tokens, labels, mask in zip(
                    token_rows, label_rows, attention_rows, strict=True
                )
            )
        ):
            raise ValueError(
                "causal-LM collator returned inconsistent batch dimensions"
            )
        sequences = [
            {
                "token_ids": [int(value) for value in tokens],
                "token_mask": [bool(value) for value in mask],
                "label_mask": [int(value) != -100 for value in labels],
            }
            for tokens, labels, mask in zip(
                token_rows, label_rows, attention_rows, strict=True
            )
        ]
        plan = self._planner(sequences=sequences, **self.config)
        batch[HF_REPETITION_UNLIKELIHOOD_BATCH_PLAN_KEY] = (
            HfRepetitionUnlikelihoodBatchPlan(plan)
        )
        return batch


class _RepetitionUnlikelihoodReceipt:
    def __init__(self, recipe: Mapping[str, object]) -> None:
        self.recipe = dict(recipe)
        self.training_batch_count = 0
        self.active_batch_count = 0
        self.active_position_count = 0
        self.candidate_count = 0
        self._plan_stream = hashlib.sha256()
        self._base_loss_sum: Any = None
        self._auxiliary_loss_sum: Any = None
        self._total_loss_sum: Any = None
        self._error: str | None = None

    def observe(
        self,
        plan: Mapping[str, Any],
        base_loss: Any,
        auxiliary_loss: Any,
        total_loss: Any,
    ) -> None:
        aggregate = plan.get("aggregate")
        if not isinstance(aggregate, Mapping):
            raise RuntimeError("Rust repetition-unlikelihood plan is missing aggregate")
        plan_id = plan.get("plan_id")
        if not isinstance(plan_id, str):
            raise RuntimeError("Rust repetition-unlikelihood plan is missing plan_id")
        active_positions = int(aggregate.get("active_position_count", 0))
        candidates = int(aggregate.get("candidate_count", 0))
        self.training_batch_count += 1
        self.active_batch_count += int(active_positions > 0)
        self.active_position_count += active_positions
        self.candidate_count += candidates
        self._plan_stream.update(plan_id.encode("ascii"))
        self._plan_stream.update(b"\n")
        for name, value in (
            ("_base_loss_sum", base_loss),
            ("_auxiliary_loss_sum", auxiliary_loss),
            ("_total_loss_sum", total_loss),
        ):
            detached = value.detach().float()
            current = getattr(self, name)
            setattr(self, name, detached if current is None else current + detached)

    def abort(self, error: BaseException) -> None:
        self._error = f"{error.__class__.__name__}: {error}"

    @staticmethod
    def _mean(value: Any, count: int) -> float | None:
        if value is None or count == 0:
            return None
        result = float((value / count).item())
        return result if math.isfinite(result) else None

    def report(self) -> dict[str, object]:
        count = self.training_batch_count
        strength = float(dict(self.recipe.get("config") or {}).get("strength", 0.0))
        auxiliary_mean = self._mean(self._auxiliary_loss_sum, count)
        return {
            "row_type": "hf_repetition_unlikelihood_receipt",
            "schema": HF_REPETITION_UNLIKELIHOOD_RECEIPT_SCHEMA,
            "status": "aborted"
            if self._error
            else "ready"
            if count
            else "not_observed",
            "semantic_owner": ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_OWNER,
            "semantic_backend": ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_BACKEND,
            "differentiation_owner": ZSPACE_REPETITION_UNLIKELIHOOD_DIFFERENTIATION_OWNER,
            "scope": "local_process_training_compute_loss_calls",
            "recipe": self.recipe,
            "training_batch_count": count,
            "active_batch_count": self.active_batch_count,
            "active_position_count": self.active_position_count,
            "candidate_count": self.candidate_count,
            "mean_candidates_per_active_position": (
                None
                if self.active_position_count == 0
                else self.candidate_count / self.active_position_count
            ),
            "plan_stream_id": (
                None if count == 0 else "sha256:" + self._plan_stream.hexdigest()
            ),
            "mean_base_training_loss": self._mean(self._base_loss_sum, count),
            "mean_auxiliary_loss": auxiliary_mean,
            "mean_weighted_auxiliary_loss": (
                None if auxiliary_mean is None else strength * auxiliary_mean
            ),
            "mean_total_training_loss": self._mean(self._total_loss_sum, count),
            "error": self._error,
            "efficacy_claim_ready": False,
            "evidence_boundary": (
                "receipt proves objective execution and observed loss terms, not "
                "reduced held-out generation loops or improved language quality"
            ),
        }


class _HfRepetitionUnlikelihoodTrainerMixin:
    _spiraltorch_base_compute_loss_accepts_num_items = False

    def __init__(
        self,
        *args: Any,
        zspace_repetition_unlikelihood_recipe: Mapping[str, object],
        **kwargs: Any,
    ) -> None:
        self._zspace_repetition_unlikelihood_recipe = dict(
            zspace_repetition_unlikelihood_recipe
        )
        self._zspace_repetition_unlikelihood_receipt = _RepetitionUnlikelihoodReceipt(
            self._zspace_repetition_unlikelihood_recipe
        )
        super().__init__(*args, **kwargs)
        # Otherwise Transformers token-normalizes only the model loss across an
        # accumulation group, multiplying this per-microbatch auxiliary term.
        self.model_accepts_loss_kwargs = False

    def _spiraltorch_base_compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        *,
        num_items_in_batch: Any,
    ) -> tuple[Any, Any]:
        kwargs: dict[str, Any] = {"return_outputs": True}
        if self._spiraltorch_base_compute_loss_accepts_num_items:
            kwargs["num_items_in_batch"] = num_items_in_batch
        return super().compute_loss(model, inputs, **kwargs)

    @staticmethod
    def _spiraltorch_auxiliary_loss(
        logits: Any,
        plan: Mapping[str, Any],
    ) -> Any:
        positions = plan.get("positions")
        if not isinstance(positions, list) or not positions:
            return logits.sum() * 0.0
        sequence_indices: list[int] = []
        prediction_indices: list[int] = []
        candidate_position_indices: list[int] = []
        candidate_token_ids: list[int] = []
        for position_index, position in enumerate(positions):
            if not isinstance(position, Mapping):
                raise RuntimeError("Rust plan returned a malformed active position")
            sequence_indices.append(int(position["sequence_index"]))
            prediction_indices.append(int(position["prediction_index"]))
            candidates = position.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise RuntimeError(
                    "Rust plan returned an active position without candidates"
                )
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    raise RuntimeError("Rust plan returned a malformed candidate")
                candidate_position_indices.append(position_index)
                candidate_token_ids.append(int(candidate["token_id"]))
        torch = __import__("torch")
        device = logits.device
        sequence_tensor = torch.tensor(
            sequence_indices, device=device, dtype=torch.long
        )
        prediction_tensor = torch.tensor(
            prediction_indices, device=device, dtype=torch.long
        )
        candidate_position_tensor = torch.tensor(
            candidate_position_indices, device=device, dtype=torch.long
        )
        candidate_token_tensor = torch.tensor(
            candidate_token_ids, device=device, dtype=torch.long
        )
        active_logits = logits[sequence_tensor, prediction_tensor].float()
        log_probabilities = torch.log_softmax(active_logits, dim=-1)
        candidate_probabilities = torch.exp(
            log_probabilities[candidate_position_tensor, candidate_token_tensor]
        )
        epsilon = ZSPACE_REPETITION_UNLIKELIHOOD_PROBABILITY_EPSILON
        candidate_losses = -torch.log1p(
            -candidate_probabilities.clamp(max=1.0 - epsilon)
        )
        position_sums = torch.zeros(
            len(positions), device=device, dtype=candidate_losses.dtype
        ).scatter_add_(0, candidate_position_tensor, candidate_losses)
        position_counts = torch.zeros_like(position_sums).scatter_add_(
            0,
            candidate_position_tensor,
            torch.ones_like(candidate_losses),
        )
        return (position_sums / position_counts).mean()

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        batch_plan = inputs.pop(HF_REPETITION_UNLIKELIHOOD_BATCH_PLAN_KEY, None)
        base_loss, outputs = self._spiraltorch_base_compute_loss(
            model,
            inputs,
            num_items_in_batch=num_items_in_batch,
        )
        if not model.training:
            return (base_loss, outputs) if return_outputs else base_loss
        if not isinstance(batch_plan, HfRepetitionUnlikelihoodBatchPlan):
            raise RuntimeError(
                "training batch is missing its Rust repetition-unlikelihood plan"
            )
        plan = validate_zspace_repetition_unlikelihood_plan(batch_plan.report)
        expected_config = self._zspace_repetition_unlikelihood_recipe.get("config")
        request = plan.get("request")
        observed_config = (
            request.get("config") if isinstance(request, Mapping) else None
        )
        if observed_config != expected_config:
            raise RuntimeError(
                "Rust repetition-unlikelihood plan does not match the training recipe"
            )
        logits = (
            outputs.get("logits") if isinstance(outputs, Mapping) else outputs.logits
        )
        auxiliary_loss = self._spiraltorch_auxiliary_loss(logits, plan)
        strength = float(dict(expected_config or {}).get("strength", 0.0))
        total_loss = base_loss + strength * auxiliary_loss
        self._zspace_repetition_unlikelihood_receipt.observe(
            plan,
            base_loss,
            auxiliary_loss,
            total_loss,
        )
        return (total_loss, outputs) if return_outputs else total_loss

    def zspace_repetition_unlikelihood_receipt(self) -> dict[str, object]:
        return self._zspace_repetition_unlikelihood_receipt.report()

    def abort_zspace_repetition_unlikelihood(self, error: BaseException) -> None:
        self._zspace_repetition_unlikelihood_receipt.abort(error)


def hf_repetition_unlikelihood_trainer_class(
    base_trainer_class: type[Any],
) -> type[Any]:
    """Compose the objective with the installed Transformers Trainer class."""

    parameters = inspect.signature(base_trainer_class.compute_loss).parameters
    trainer_class = type(
        "HfRepetitionUnlikelihoodTrainer",
        (_HfRepetitionUnlikelihoodTrainerMixin, base_trainer_class),
        {
            "_spiraltorch_base_compute_loss_accepts_num_items": (
                "num_items_in_batch" in parameters
            ),
            "__module__": __name__,
        },
    )
    return trainer_class
