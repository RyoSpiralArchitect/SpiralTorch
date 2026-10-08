"""Hugging Face adapters for Rust-planned repetition-unlikelihood training."""

from __future__ import annotations

import hashlib
import inspect
import copy
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from .repetition_objective import zspace_repetition_objective_control
from .repetition_unlikelihood import (
    ZSPACE_REPETITION_UNLIKELIHOOD_CONTRACT_VERSION,
    ZSPACE_REPETITION_UNLIKELIHOOD_DIFFERENTIATION_OWNER,
    ZSPACE_REPETITION_UNLIKELIHOOD_OBJECTIVE_RULE,
    ZSPACE_REPETITION_UNLIKELIHOOD_PROBABILITY_EPSILON,
    ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_OWNER,
    ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_RULE,
    ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_BACKEND,
    ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_OWNER,
    validate_zspace_repetition_unlikelihood_plan,
    zspace_repetition_unlikelihood_plan,
)

HF_REPETITION_UNLIKELIHOOD_RECEIPT_SCHEMA = (
    "spiraltorch.hf_repetition_unlikelihood_receipt.v3"
)
HF_REPETITION_UNLIKELIHOOD_BATCH_PLAN_KEY = "_spiraltorch_repetition_unlikelihood_plan"
_MODEL_TOPK_MAX_FLOAT_BYTES = 32 * 1024 * 1024
_MODEL_TOPK_SOURCE_KINDS = frozenset({"model_topk_history", "model_topk_periodic"})
_OBJECTIVE_CHECKPOINT = "spiraltorch-repetition-objective.json"
_OBJECTIVE_CHECKPOINT_SCHEMA = "spiraltorch.hf_repetition_objective_checkpoint.v1"

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

    report: Mapping[str, Any] | None
    sequences: tuple[Mapping[str, Any], ...] | None = None


def hf_repetition_unlikelihood_recipe_contract(
    *,
    strength: float,
    ngram_order: int,
    context_window: int,
    max_candidates_per_position: int,
    candidate_source: str | Mapping[str, object] = "prior_continuation",
    proposal_top_k: int = 8,
    objective_control: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return the exact objective recipe embedded in training identity."""

    source_kind = (
        candidate_source.get("kind")
        if isinstance(candidate_source, Mapping)
        else candidate_source
    )
    validation_sequence: dict[str, object] = {
        "token_ids": [0],
        "token_mask": [True],
        "label_mask": [True],
    }
    if source_kind in _MODEL_TOPK_SOURCE_KINDS:
        validation_sequence["proposal_token_ids"] = [[]]
    validation_plan = zspace_repetition_unlikelihood_plan(
        sequences=[validation_sequence],
        strength=strength,
        candidate_source=candidate_source,
        ngram_order=ngram_order,
        proposal_top_k=proposal_top_k,
        context_window=context_window,
        max_candidates_per_position=max_candidates_per_position,
    )
    request = validation_plan.get("request")
    config = request.get("config") if isinstance(request, Mapping) else None
    if not isinstance(config, Mapping):
        raise RuntimeError("Rust validation plan is missing its canonical config")
    canonical_config = dict(config)
    enabled = float(canonical_config["strength"]) > 0.0
    recipe = {
        "schema": "spiraltorch.hf_repetition_unlikelihood_recipe.v4",
        "enabled": enabled,
        "semantic_owner": ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_OWNER,
        "semantic_backend": ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_BACKEND,
        "contract_version": ZSPACE_REPETITION_UNLIKELIHOOD_CONTRACT_VERSION,
        "differentiation_owner": ZSPACE_REPETITION_UNLIKELIHOOD_DIFFERENTIATION_OWNER,
        "proposal_owner": ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_OWNER,
        "proposal_rule": ZSPACE_REPETITION_UNLIKELIHOOD_PROPOSAL_RULE,
        "objective_rule": ZSPACE_REPETITION_UNLIKELIHOOD_OBJECTIVE_RULE,
        "evaluation_loss": "causal_lm_loss_only",
        "evaluation_loss_normalization": (
            "preserve_base_trainer_loss_kwargs" if enabled else None
        ),
        "evaluation_num_items_in_batch": ("preserve_base_trainer" if enabled else None),
        "gradient_accumulation_normalization": (
            "trainer_divides_the_combined_microbatch_mean" if enabled else None
        ),
        "model_accepts_loss_kwargs": False if enabled else None,
        "proposal_materialization": (
            "after_model_forward_from_detached_logits"
            if enabled and source_kind in _MODEL_TOPK_SOURCE_KINDS
            else "in_data_collator" if enabled else None
        ),
        "config": canonical_config,
        "data_collator": (
            "spiraltorch.HfRepetitionUnlikelihoodCollator" if enabled else None
        ),
        "trainer": ("spiraltorch.HfRepetitionUnlikelihoodTrainer" if enabled else None),
    }
    if objective_control is not None:
        if not enabled:
            raise ValueError("objective control requires positive repetition strength")
        control = zspace_repetition_objective_control(
            config=dict(objective_control),
            base_strength=canonical_config["strength"],
            completed_update_slots=0,
            active_position_count=0,
            eligible_target_count=0,
        )
        recipe["schema"] = "spiraltorch.hf_repetition_unlikelihood_recipe.v5"
        recipe["objective_control"] = control["policy"]
        recipe["objective_rule"] = control["policy"]["objective_rule"]
    return recipe


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
        candidate_source: str | Mapping[str, object] = "prior_continuation",
        proposal_top_k: int = 8,
        planner: Callable[..., dict[str, Any]] = zspace_repetition_unlikelihood_plan,
    ) -> None:
        self.base_collator = base_collator
        recipe = hf_repetition_unlikelihood_recipe_contract(
            strength=strength,
            ngram_order=ngram_order,
            context_window=context_window,
            max_candidates_per_position=max_candidates_per_position,
            candidate_source=candidate_source,
            proposal_top_k=proposal_top_k,
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
        source = self.config.get("candidate_source")
        source_kind = source.get("kind") if isinstance(source, Mapping) else None
        if source_kind in _MODEL_TOPK_SOURCE_KINDS:
            metadata = HfRepetitionUnlikelihoodBatchPlan(
                report=None,
                sequences=tuple(sequences),
            )
        else:
            metadata = HfRepetitionUnlikelihoodBatchPlan(
                self._planner(sequences=sequences, **self.config)
            )
        batch[HF_REPETITION_UNLIKELIHOOD_BATCH_PLAN_KEY] = metadata
        return batch


class _RepetitionUnlikelihoodReceipt:
    def __init__(self, recipe: Mapping[str, object]) -> None:
        self.recipe = dict(recipe)
        self.training_batch_count = 0
        self.active_batch_count = 0
        self.active_position_count = 0
        self.candidate_count = 0
        self.eligible_target_count = 0
        self.proposal_count = 0
        self.excluded_target_proposal_count = 0
        self.excluded_out_of_history_proposal_count = 0
        self.excluded_non_periodic_proposal_count = 0
        self.periodic_candidate_count = 0
        self._plan_stream = hashlib.sha256()
        self._base_loss_sum: Any = None
        self._auxiliary_loss_sum: Any = None
        self._total_loss_sum: Any = None
        self._weighted_auxiliary_loss_sum: Any = None
        self._first_objective_slot: int | None = None
        self._last_objective_control: Mapping[str, Any] | None = None
        self._error: str | None = None

    def observe(
        self,
        plan: Mapping[str, Any],
        base_loss: Any,
        auxiliary_loss: Any,
        total_loss: Any,
        *,
        objective_control: Mapping[str, Any] | None = None,
        weighted_auxiliary_loss: Any = None,
    ) -> None:
        aggregate = plan.get("aggregate")
        if not isinstance(aggregate, Mapping):
            raise RuntimeError("Rust repetition-unlikelihood plan is missing aggregate")
        plan_id = plan.get("plan_id")
        if not isinstance(plan_id, str):
            raise RuntimeError("Rust repetition-unlikelihood plan is missing plan_id")
        active_positions = int(aggregate.get("active_position_count", 0))
        candidates = int(aggregate.get("candidate_count", 0))
        eligible_targets = int(aggregate.get("eligible_target_count", 0))
        self.training_batch_count += 1
        self.active_batch_count += int(active_positions > 0)
        self.active_position_count += active_positions
        self.candidate_count += candidates
        self.eligible_target_count += eligible_targets
        self.proposal_count += int(aggregate.get("proposal_count", 0))
        self.excluded_target_proposal_count += int(
            aggregate.get("excluded_target_proposal_count", 0)
        )
        self.excluded_out_of_history_proposal_count += int(
            aggregate.get("excluded_out_of_history_proposal_count", 0)
        )
        self.excluded_non_periodic_proposal_count += int(
            aggregate.get("excluded_non_periodic_proposal_count", 0)
        )
        self.periodic_candidate_count += int(
            aggregate.get("periodic_candidate_count", 0)
        )
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
        if objective_control is not None:
            weighted = weighted_auxiliary_loss.detach().float()
            current = self._weighted_auxiliary_loss_sum
            self._weighted_auxiliary_loss_sum = (
                weighted if current is None else current + weighted
            )
            if self._first_objective_slot is None:
                self._first_objective_slot = objective_control["completed_update_slots"]
            self._last_objective_control = objective_control

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
        report = {
            "row_type": "hf_repetition_unlikelihood_receipt",
            "schema": HF_REPETITION_UNLIKELIHOOD_RECEIPT_SCHEMA,
            "status": (
                "aborted" if self._error else "ready" if count else "not_observed"
            ),
            "semantic_owner": ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_OWNER,
            "semantic_backend": ZSPACE_REPETITION_UNLIKELIHOOD_SEMANTIC_BACKEND,
            "differentiation_owner": ZSPACE_REPETITION_UNLIKELIHOOD_DIFFERENTIATION_OWNER,
            "scope": "local_process_training_compute_loss_calls",
            "recipe": self.recipe,
            "training_batch_count": count,
            "active_batch_count": self.active_batch_count,
            "active_position_count": self.active_position_count,
            "candidate_count": self.candidate_count,
            "eligible_target_count": self.eligible_target_count,
            "active_position_ratio": (
                None
                if self.eligible_target_count == 0
                else self.active_position_count / self.eligible_target_count
            ),
            "proposal_count": self.proposal_count,
            "excluded_target_proposal_count": self.excluded_target_proposal_count,
            "excluded_out_of_history_proposal_count": (
                self.excluded_out_of_history_proposal_count
            ),
            "excluded_non_periodic_proposal_count": (
                self.excluded_non_periodic_proposal_count
            ),
            "periodic_candidate_count": self.periodic_candidate_count,
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
        if "objective_control" in self.recipe:
            report.update(
                {
                    "schema": "spiraltorch.hf_repetition_unlikelihood_receipt.v4",
                    "mean_weighted_auxiliary_loss": self._mean(
                        self._weighted_auxiliary_loss_sum, count
                    ),
                    "first_completed_update_slots": self._first_objective_slot,
                    "last_objective_control": self._last_objective_control,
                }
            )
        return report


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_checkpoint_json(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = handle.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
        raise ValueError("repetition objective checkpoint metadata exceeds 16 MiB")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("repetition objective checkpoint metadata must be an object")
    return value


class _HfRepetitionUnlikelihoodTrainerMixin:
    _spiraltorch_base_compute_loss_accepts_num_items = False

    def __init__(
        self,
        *args: Any,
        zspace_repetition_unlikelihood_recipe: Mapping[str, object],
        **kwargs: Any,
    ) -> None:
        self._zspace_repetition_unlikelihood_recipe = copy.deepcopy(
            dict(zspace_repetition_unlikelihood_recipe)
        )
        recipe = self._zspace_repetition_unlikelihood_recipe
        self._spiraltorch_objective_policy = recipe.get("objective_control")
        if self._spiraltorch_objective_policy is not None:
            config = dict(recipe["config"])
            canonical = hf_repetition_unlikelihood_recipe_contract(
                **config,
                ngram_order=3,
                objective_control=self._spiraltorch_objective_policy["config"],
            )
            if _canonical_json(recipe) != _canonical_json(canonical):
                raise ValueError(
                    "repetition objective recipe differs from the Rust contract"
                )
        self._zspace_repetition_unlikelihood_receipt = _RepetitionUnlikelihoodReceipt(
            self._zspace_repetition_unlikelihood_recipe
        )
        super().__init__(*args, **kwargs)
        if self._spiraltorch_objective_policy is not None and getattr(
            getattr(self, "args", None), "save_only_model", False
        ):
            raise ValueError(
                "objective control requires full Trainer checkpoints, not save_only_model"
            )
        self._spiraltorch_base_model_accepts_loss_kwargs = bool(
            getattr(self, "model_accepts_loss_kwargs", False)
        )
        # Otherwise Transformers token-normalizes only the model loss across an
        # accumulation group, multiplying this per-microbatch auxiliary term.
        self.model_accepts_loss_kwargs = False

    def train(
        self, resume_from_checkpoint: Any = None, *args: Any, **kwargs: Any
    ) -> Any:
        # Check before the base Trainer can load weights, including sharded paths.
        if "model_path" in kwargs:
            if resume_from_checkpoint not in (None, False):
                raise ValueError("specify only one checkpoint resume argument")
            resume_from_checkpoint = kwargs.pop("model_path")
        if resume_from_checkpoint is True:
            from transformers.trainer_utils import get_last_checkpoint

            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError("no Trainer checkpoint found for objective resume")
        if resume_from_checkpoint:
            root = Path(resume_from_checkpoint)
            marker = root / _OBJECTIVE_CHECKPOINT
            if self._spiraltorch_objective_policy is not None or marker.exists():
                saved = _read_checkpoint_json(marker)
                state = _read_checkpoint_json(root / "trainer_state.json")
                slot = state.get("global_step")
                if (
                    saved.get("schema") != _OBJECTIVE_CHECKPOINT_SCHEMA
                    or _canonical_json(saved.get("recipe"))
                    != _canonical_json(self._zspace_repetition_unlikelihood_recipe)
                    or type(slot) is not int
                    or slot < 0
                    or type(saved.get("completed_update_slots")) is not int
                    or saved["completed_update_slots"] != slot
                    or saved.get("training_partition")
                    != self._spiraltorch_training_partition()
                ):
                    raise ValueError(
                        "repetition objective checkpoint recipe or update clock mismatch"
                    )
                # Rust validates the portable clock range before model loading.
                self._spiraltorch_objective_control(slot, 0, 0)
        return super().train(resume_from_checkpoint, *args, **kwargs)

    def _spiraltorch_training_partition(self) -> dict[str, int]:
        return {
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
            "per_device_train_batch_size": self.args.per_device_train_batch_size,
            "world_size": self.args.world_size,
        }

    def _save_checkpoint(
        self, model: Any, trial: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if self._spiraltorch_objective_policy is not None and self.args.should_save:
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            slot = self.state.global_step
            self._spiraltorch_objective_control(slot, 0, 0)
            directory = (
                Path(self._get_output_dir(trial=trial))
                / f"{PREFIX_CHECKPOINT_DIR}-{slot}"
            )
            directory.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": _OBJECTIVE_CHECKPOINT_SCHEMA,
                "recipe": self._zspace_repetition_unlikelihood_recipe,
                "completed_update_slots": slot,
                "training_partition": self._spiraltorch_training_partition(),
            }
            # Write before the base checkpoint so its optional Hub upload includes
            # the guard. A missing/mismatched Trainer state cannot be resumed.
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=directory, delete=False
                ) as handle:
                    temporary = Path(handle.name)
                    handle.write(_canonical_json(payload) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, directory / _OBJECTIVE_CHECKPOINT)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return super()._save_checkpoint(model, trial, *args, **kwargs)

    def _spiraltorch_objective_control(
        self, slot: int, active: int, eligible: int
    ) -> dict[str, Any]:
        policy = self._spiraltorch_objective_policy
        if policy is None:
            raise ValueError("cannot resume a controlled objective without its policy")
        control = zspace_repetition_objective_control(
            config=policy["config"],
            base_strength=policy["base_strength"],
            completed_update_slots=slot,
            active_position_count=active,
            eligible_target_count=eligible,
        )
        if control["policy"] != policy:
            raise RuntimeError(
                "live repetition objective differs from the sealed Rust policy"
            )
        return control

    def _get_num_items_in_batch(self, batch_samples: Any, device: Any) -> Any:
        base_accepts = getattr(
            self, "_spiraltorch_base_model_accepts_loss_kwargs", False
        )
        model = getattr(self, "model", None)
        if model is None or model.training or not base_accepts:
            return super()._get_num_items_in_batch(batch_samples, device)

        self.model_accepts_loss_kwargs = True
        try:
            return super()._get_num_items_in_batch(batch_samples, device)
        finally:
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
        if model.training or not self._spiraltorch_base_model_accepts_loss_kwargs:
            return super().compute_loss(model, inputs, **kwargs)

        # Preserve the base Trainer's causal-LM normalization during evaluation.
        # Training still keeps this disabled so the auxiliary microbatch mean is
        # scaled together with the model loss under gradient accumulation.
        self.model_accepts_loss_kwargs = True
        try:
            return super().compute_loss(model, inputs, **kwargs)
        finally:
            self.model_accepts_loss_kwargs = False

    def _spiraltorch_materialize_plan(
        self,
        logits: Any,
        batch_plan: HfRepetitionUnlikelihoodBatchPlan,
    ) -> Mapping[str, Any]:
        if isinstance(batch_plan.report, Mapping):
            if batch_plan.sequences is not None:
                raise RuntimeError(
                    "Rust repetition-unlikelihood metadata mixes eager and deferred plans"
                )
            return batch_plan.report
        if batch_plan.report is not None or batch_plan.sequences is None:
            raise RuntimeError(
                "training batch has malformed deferred repetition metadata"
            )
        expected_config = self._zspace_repetition_unlikelihood_recipe.get("config")
        if not isinstance(expected_config, Mapping):
            raise RuntimeError("repetition-unlikelihood recipe is missing config")
        source = expected_config.get("candidate_source")
        source_kind = source.get("kind") if isinstance(source, Mapping) else None
        if source_kind not in _MODEL_TOPK_SOURCE_KINDS:
            raise RuntimeError("only model-top-k sources may defer repetition planning")
        proposal_top_k = int(source.get("proposal_top_k", 0))
        if getattr(logits, "ndim", None) != 3:
            raise RuntimeError("causal-LM logits must have rank 3")
        batch_size, sequence_width, vocabulary_size = map(int, logits.shape)
        if len(batch_plan.sequences) != batch_size:
            raise RuntimeError(
                "deferred repetition metadata does not match the logits batch"
            )
        if proposal_top_k > vocabulary_size:
            raise RuntimeError(
                "model vocabulary is smaller than repetition proposal_top_k"
            )

        materialized_sequences: list[dict[str, object]] = []
        sequence_indices: list[int] = []
        prediction_indices: list[int] = []
        target_coordinates: list[tuple[int, int]] = []
        for sequence_index, sequence in enumerate(batch_plan.sequences):
            token_ids = list(sequence.get("token_ids", []))
            token_mask = list(sequence.get("token_mask", []))
            label_mask = list(sequence.get("label_mask", []))
            if not (
                len(token_ids) == len(token_mask) == len(label_mask) == sequence_width
            ):
                raise RuntimeError(
                    "deferred repetition metadata does not match logits sequence width"
                )
            proposal_rows: list[list[int]] = [[] for _ in token_ids]
            materialized_sequences.append(
                {
                    "token_ids": token_ids,
                    "token_mask": token_mask,
                    "label_mask": label_mask,
                    "proposal_token_ids": proposal_rows,
                }
            )
            for target_index in range(1, len(token_ids)):
                if (
                    bool(label_mask[target_index])
                    and bool(token_mask[target_index])
                    and bool(token_mask[target_index - 1])
                ):
                    sequence_indices.append(sequence_index)
                    prediction_indices.append(target_index - 1)
                    target_coordinates.append((sequence_index, target_index))

        if target_coordinates:
            torch = __import__("torch")
            device = logits.device
            max_chunk_rows = max(
                1,
                _MODEL_TOPK_MAX_FLOAT_BYTES // (vocabulary_size * 4),
            )
            proposal_rows: list[list[int]] = []
            with torch.no_grad():
                for chunk_start in range(0, len(target_coordinates), max_chunk_rows):
                    chunk_end = min(
                        chunk_start + max_chunk_rows,
                        len(target_coordinates),
                    )
                    sequence_tensor = torch.tensor(
                        sequence_indices[chunk_start:chunk_end],
                        device=device,
                        dtype=torch.long,
                    )
                    prediction_tensor = torch.tensor(
                        prediction_indices[chunk_start:chunk_end],
                        device=device,
                        dtype=torch.long,
                    )
                    active_logits = (
                        logits[sequence_tensor, prediction_tensor].detach().float()
                    )
                    chunk_proposals = (
                        torch.topk(
                            active_logits,
                            k=proposal_top_k,
                            dim=-1,
                            largest=True,
                            sorted=True,
                        )
                        .indices.cpu()
                        .tolist()
                    )
                    proposal_rows.extend(chunk_proposals)
                    del active_logits
            for (sequence_index, target_index), proposals in zip(
                target_coordinates, proposal_rows, strict=True
            ):
                materialized_sequences[sequence_index]["proposal_token_ids"][  # type: ignore[index]
                    target_index
                ] = proposals

        return zspace_repetition_unlikelihood_plan(
            sequences=materialized_sequences,
            **dict(expected_config),
        )

    @staticmethod
    def _spiraltorch_auxiliary_loss(
        logits: Any,
        plan: Mapping[str, Any],
    ) -> Any:
        positions = plan.get("positions")
        if not isinstance(positions, list) or not positions:
            return logits[..., :0].sum()
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
        expected_config = self._zspace_repetition_unlikelihood_recipe.get("config")
        logits = (
            outputs.get("logits") if isinstance(outputs, Mapping) else outputs.logits
        )
        plan = validate_zspace_repetition_unlikelihood_plan(
            self._spiraltorch_materialize_plan(logits, batch_plan)
        )
        request = plan.get("request")
        observed_config = (
            request.get("config") if isinstance(request, Mapping) else None
        )
        if observed_config != expected_config:
            raise RuntimeError(
                "Rust repetition-unlikelihood plan does not match the training recipe"
            )
        auxiliary_loss = self._spiraltorch_auxiliary_loss(logits, plan)
        strength = float(dict(expected_config or {}).get("strength", 0.0))
        control = None
        if self._spiraltorch_objective_policy is not None:
            aggregate = plan["aggregate"]
            control = self._spiraltorch_objective_control(
                self.state.global_step,
                aggregate["active_position_count"],
                aggregate["eligible_target_count"],
            )
            strength = control["effective_strength"]
        weighted_auxiliary_loss = strength * auxiliary_loss
        total_loss = base_loss + weighted_auxiliary_loss
        self._zspace_repetition_unlikelihood_receipt.observe(
            plan,
            base_loss,
            auxiliary_loss,
            total_loss,
            objective_control=control,
            weighted_auxiliary_loss=weighted_auxiliary_loss,
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
