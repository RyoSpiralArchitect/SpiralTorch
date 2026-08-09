"""Rust-owned Z-space parameter control for external HF optimizers."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .zspace_optimizer import (
    ZSPACE_OPTIMIZER_FEEDBACK_CONTRACT_VERSION,
    ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_BACKEND,
    ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_OWNER,
    ZSPACE_PARAMETER_CONTROL_CONTRACT_VERSION,
    ZSPACE_PARAMETER_CONTROL_SEMANTIC_BACKEND,
    ZSPACE_PARAMETER_CONTROL_SEMANTIC_OWNER,
    ZSPACE_PARAMETER_TRAJECTORY_CONTRACT_VERSION,
    ZSPACE_PARAMETER_TRAJECTORY_SEMANTIC_BACKEND,
    ZSPACE_PARAMETER_TRAJECTORY_SEMANTIC_OWNER,
)

HF_ZSPACE_OPTIMIZER_CONTROL_SCHEMA = "spiraltorch.hf_zspace_optimizer_control.v1"
HF_ZSPACE_OPTIMIZER_RECEIPT_SCHEMA = "spiraltorch.hf_zspace_optimizer_receipt.v1"
HF_ZSPACE_OPTIMIZER_STATE_SCHEMA = "spiraltorch.hf_zspace_optimizer_state.v3"
HF_ZSPACE_OPTIMIZER_TRACE_SCHEMA = "spiraltorch.hf_zspace_optimizer_trace.v1"
HF_ZSPACE_MATCHED_ABLATION_SCHEMA = "spiraltorch.hf_zspace_matched_ablation.v1"
HF_ZSPACE_FACTORIZED_ABLATION_SCHEMA = "spiraltorch.hf_zspace_factorized_ablation.v1"
HF_ZSPACE_FEEDBACK_ABLATION_SCHEMA = "spiraltorch.hf_zspace_feedback_ablation.v1"
HF_ZSPACE_OPTIMIZER_STATE_FILENAME = "spiraltorch-hf-zspace-optimizer-state.json"
HF_ZSPACE_OPTIMIZER_TRACE_FILENAME = "spiraltorch-hf-zspace-optimizer-trace.jsonl"
HF_ZSPACE_OPTIMIZER_TRAJECTORY_FILENAME = (
    "spiraltorch-hf-zspace-optimizer-trajectory.json"
)
HF_ZSPACE_OPTIMIZER_MODES = ("off", "observe", "apply")
HF_ZSPACE_OPTIMIZER_FEEDBACK_MODES = ("off", "loss_guard")
HF_ZSPACE_OPTIMIZER_TRAJECTORY_ARMS = (
    "raw",
    "dose_matched_constant",
    "dose_normalized",
)
_HF_ZSPACE_OPTIMIZER_LEGACY_STATE_SCHEMA = "spiraltorch.hf_zspace_optimizer_state.v1"
_HF_ZSPACE_OPTIMIZER_V2_STATE_SCHEMA = "spiraltorch.hf_zspace_optimizer_state.v2"
_HF_ZSPACE_OPTIMIZER_V1_RECIPE_ADDITIONS = (
    "trajectory_arm",
    "trajectory_id",
    "trajectory_contract",
    "trajectory_semantic_owner",
    "trajectory_semantic_backend",
    "trajectory_required",
    "actuation_scale_source",
)
_HF_ZSPACE_OPTIMIZER_FEEDBACK_RECIPE_ADDITIONS = (
    "feedback_mode",
    "feedback_adaptive",
    "feedback_contract",
    "feedback_semantic_owner",
    "feedback_semantic_backend",
    "feedback_config",
    "feedback_control_target",
    "feedback_evidence_boundary",
)

__all__ = [
    "HF_ZSPACE_OPTIMIZER_CONTROL_SCHEMA",
    "HF_ZSPACE_FACTORIZED_ABLATION_SCHEMA",
    "HF_ZSPACE_FEEDBACK_ABLATION_SCHEMA",
    "HF_ZSPACE_MATCHED_ABLATION_SCHEMA",
    "HF_ZSPACE_OPTIMIZER_MODES",
    "HF_ZSPACE_OPTIMIZER_FEEDBACK_MODES",
    "HF_ZSPACE_OPTIMIZER_RECEIPT_SCHEMA",
    "HF_ZSPACE_OPTIMIZER_STATE_FILENAME",
    "HF_ZSPACE_OPTIMIZER_STATE_SCHEMA",
    "HF_ZSPACE_OPTIMIZER_TRACE_FILENAME",
    "HF_ZSPACE_OPTIMIZER_TRACE_SCHEMA",
    "HF_ZSPACE_OPTIMIZER_TRAJECTORY_ARMS",
    "HF_ZSPACE_OPTIMIZER_TRAJECTORY_FILENAME",
    "hf_zspace_optimizer_control_callback",
    "hf_zspace_optimizer_recipe_contract",
    "compare_hf_zspace_optimizer_run_cards",
    "compare_hf_zspace_optimizer_factorized_run_cards",
    "compare_hf_zspace_optimizer_feedback_run_cards",
    "write_hf_zspace_optimizer_matched_ablation_report",
    "write_hf_zspace_optimizer_factorized_ablation_report",
    "write_hf_zspace_optimizer_feedback_ablation_report",
]


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a finite number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a positive integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a positive integer") from exc
    if numeric <= 0 or numeric != value:
        raise ValueError(f"{label} must be a positive integer")
    return numeric


def _non_negative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a non-negative integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a non-negative integer") from exc
    if numeric < 0 or numeric != value:
        raise ValueError(f"{label} must be a non-negative integer")
    return numeric


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_id(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _trace_file_evidence(path: Path | None) -> tuple[str | None, int | None]:
    if path is None or not path.is_file():
        return None, None
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _trace_prefix_matches(path: Path, size: int, expected_sha256: str) -> bool:
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as handle:
        while remaining > 0:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                return False
            digest.update(chunk)
            remaining -= len(chunk)
    return "sha256:" + digest.hexdigest() == expected_sha256


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path | None, payload: Mapping[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                dict(payload),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )


def hf_zspace_optimizer_recipe_contract(
    *,
    mode: str = "off",
    z_dim: int = 6,
    curvature: float = -0.04,
    control_gain: float = 1.0,
    volume_per_step: int = 8,
    trajectory_arm: str = "raw",
    trajectory_id: str | None = None,
    feedback_mode: str = "off",
    feedback_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return the path-independent update recipe used by the HF callback."""

    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in HF_ZSPACE_OPTIMIZER_MODES:
        raise ValueError("mode must be one of " + ", ".join(HF_ZSPACE_OPTIMIZER_MODES))
    resolved_z_dim = _positive_int(z_dim, label="z_dim")
    resolved_curvature = _finite_float(curvature, label="curvature")
    if resolved_curvature >= 0.0:
        raise ValueError("curvature must be negative")
    resolved_gain = _finite_float(control_gain, label="control_gain")
    if not 0.0 <= resolved_gain <= 1.0:
        raise ValueError("control_gain must be in [0, 1]")
    resolved_volume_per_step = _positive_int(
        volume_per_step,
        label="volume_per_step",
    )
    resolved_arm = str(trajectory_arm).strip().lower()
    if resolved_arm not in HF_ZSPACE_OPTIMIZER_TRAJECTORY_ARMS:
        raise ValueError(
            "trajectory_arm must be one of "
            + ", ".join(HF_ZSPACE_OPTIMIZER_TRAJECTORY_ARMS)
        )
    if resolved_mode != "apply" and resolved_arm != "raw":
        raise ValueError("non-raw trajectory arms require apply mode")
    if trajectory_id is not None and not _is_sha256_id(trajectory_id):
        raise ValueError("trajectory_id must be a sha256 identity")
    if resolved_mode != "apply" and trajectory_id is not None:
        raise ValueError("only apply mode accepts an input parameter trajectory")
    if resolved_mode == "apply" and resolved_arm != "raw" and trajectory_id is None:
        raise ValueError(
            f"apply trajectory arm {resolved_arm!r} requires trajectory_id"
        )
    resolved_feedback_mode = str(feedback_mode).strip().lower()
    if resolved_feedback_mode not in HF_ZSPACE_OPTIMIZER_FEEDBACK_MODES:
        raise ValueError(
            "feedback_mode must be one of "
            + ", ".join(HF_ZSPACE_OPTIMIZER_FEEDBACK_MODES)
        )
    if resolved_feedback_mode != "off" and resolved_mode == "off":
        raise ValueError("optimizer feedback requires observe or apply mode")
    requested_feedback_config = dict(feedback_config or {})
    if resolved_feedback_mode == "off" and requested_feedback_config:
        raise ValueError("feedback_config requires an active feedback mode")
    resolved_feedback_config: dict[str, object] | None = None
    if resolved_feedback_mode == "loss_guard":
        st = importlib.import_module("spiraltorch")
        checkpoint = st.zspace_optimizer_feedback_init(requested_feedback_config)
        config_payload = checkpoint.get("config")
        if not isinstance(config_payload, Mapping):
            raise RuntimeError("Rust feedback checkpoint has no validated config")
        resolved_feedback_config = dict(config_payload)
    return {
        "schema": HF_ZSPACE_OPTIMIZER_CONTROL_SCHEMA,
        "mode": resolved_mode,
        "model_update_intervention": resolved_mode == "apply",
        "control_observation": resolved_mode in {"observe", "apply"},
        "z_dim": resolved_z_dim,
        "curvature": resolved_curvature,
        "control_gain": resolved_gain,
        "volume_per_step": resolved_volume_per_step,
        "trajectory_arm": resolved_arm,
        "trajectory_id": trajectory_id,
        "trajectory_contract": ZSPACE_PARAMETER_TRAJECTORY_CONTRACT_VERSION,
        "trajectory_semantic_owner": ZSPACE_PARAMETER_TRAJECTORY_SEMANTIC_OWNER,
        "trajectory_semantic_backend": ZSPACE_PARAMETER_TRAJECTORY_SEMANTIC_BACKEND,
        "trajectory_required": (resolved_mode == "apply" and resolved_arm != "raw"),
        "control_signal": (
            "topos_progress_geometry"
            if resolved_feedback_mode == "off"
            else "topos_progress_geometry_guarded_by_training_loss_feedback"
        ),
        "feedback_mode": resolved_feedback_mode,
        "feedback_adaptive": resolved_feedback_mode != "off",
        "feedback_contract": (
            None
            if resolved_feedback_config is None
            else ZSPACE_OPTIMIZER_FEEDBACK_CONTRACT_VERSION
        ),
        "feedback_semantic_owner": (
            None
            if resolved_feedback_config is None
            else ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_OWNER
        ),
        "feedback_semantic_backend": (
            None
            if resolved_feedback_config is None
            else ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_BACKEND
        ),
        "feedback_config": resolved_feedback_config,
        "feedback_control_target": (
            None
            if resolved_feedback_config is None
            else "proposed_learning_rate_scale_deviation_from_identity"
        ),
        "feedback_evidence_boundary": (
            None
            if resolved_feedback_config is None
            else "within_run_loss_guard_not_counterfactual_efficacy"
        ),
        "observation_mapping": (
            "hf_global_step_to_open_topos_depth_and_volume"
            if resolved_feedback_mode == "off"
            else (
                "hf_global_step_to_open_topos_depth_and_volume;"
                "hf_training_log_to_rust_training_projection"
            )
        ),
        "control_pipeline": [
            "st-tensor::pure::topos",
            "st-core::runtime::zspace_optimizer",
            *(
                []
                if resolved_feedback_config is None
                else ["st-core::runtime::zspace_optimizer_feedback"]
            ),
            "torch.optim.Optimizer.step",
        ],
        "parameter_control_contract": ZSPACE_PARAMETER_CONTROL_CONTRACT_VERSION,
        "parameter_control_semantic_owner": ZSPACE_PARAMETER_CONTROL_SEMANTIC_OWNER,
        "parameter_control_semantic_backend": ZSPACE_PARAMETER_CONTROL_SEMANTIC_BACKEND,
        "actuation_transport": "temporary_param_group_lr_scale",
        "actuation_scale_source": (
            (
                "rust_parameter_control"
                if resolved_arm == "raw"
                else f"rust_parameter_trajectory.steps.{resolved_arm}_scale"
            )
            if resolved_feedback_mode == "off"
            else "rust_optimizer_feedback(applied_scale)"
        ),
        "scheduler_isolation": "restore_nominal_lr_before_scheduler_step",
        "first_control_source_step": 0,
        "control_target": "next_optimizer_update",
        "resume_state_required": resolved_mode in {"observe", "apply"},
        "fail_closed": True,
    }


def _legacy_optimizer_recipe(
    recipe: Mapping[str, object],
    state_schema: str,
) -> dict[str, object] | None:
    if (
        recipe.get("trajectory_arm") != "raw"
        or recipe.get("trajectory_id") is not None
        or recipe.get("feedback_mode") != "off"
    ):
        return None
    legacy = dict(recipe)
    additions = list(_HF_ZSPACE_OPTIMIZER_FEEDBACK_RECIPE_ADDITIONS)
    if state_schema == _HF_ZSPACE_OPTIMIZER_LEGACY_STATE_SCHEMA:
        additions.extend(_HF_ZSPACE_OPTIMIZER_V1_RECIPE_ADDITIONS)
    for field in additions:
        legacy.pop(field, None)
    return legacy


def _hookable_optimizer(optimizer: object) -> object:
    current = optimizer
    seen: set[int] = set()
    hookable: list[object] = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if callable(getattr(current, "register_step_pre_hook", None)) and callable(
            getattr(current, "register_step_post_hook", None)
        ):
            hookable.append(current)
        inner = getattr(current, "optimizer", None)
        if inner is None or inner is current:
            break
        current = inner
    if hookable:
        return hookable[-1]
    raise RuntimeError(
        "Z-space optimizer apply mode requires a hookable torch optimizer"
    )


def _optimizer_learning_rates(optimizer: object) -> list[float]:
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)):
        raise RuntimeError("optimizer param_groups are unavailable")
    rates: list[float] = []
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping) or "lr" not in group:
            raise RuntimeError(f"optimizer param group {index} has no learning rate")
        rate = _finite_float(group["lr"], label=f"param_groups[{index}].lr")
        if rate < 0.0:
            raise ValueError(f"param_groups[{index}].lr must be non-negative")
        rates.append(rate)
    if not rates:
        raise RuntimeError("optimizer has no parameter groups")
    return rates


def _set_optimizer_learning_rates(optimizer: object, rates: Sequence[float]) -> None:
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, Sequence) or len(groups) != len(rates):
        raise RuntimeError("optimizer parameter groups changed during Z-space control")
    for group, rate in zip(groups, rates):
        group["lr"] = float(rate)


def _safe_step(value: object, *, label: str) -> int:
    return _non_negative_int(0 if value is None else value, label=label)


def _validated_parameter_trajectory(
    value: str | Path | Mapping[str, object],
) -> dict[str, object]:
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        path = Path(value)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Z-space parameter trajectory must contain a mapping")
        payload = dict(payload)
    st = importlib.import_module("spiraltorch")
    return st.validate_zspace_parameter_trajectory(payload)


def hf_zspace_optimizer_control_callback(
    *,
    mode: str = "off",
    z_dim: int = 6,
    curvature: float = -0.04,
    control_gain: float = 1.0,
    volume_per_step: int = 8,
    trajectory_arm: str = "raw",
    trajectory: str | Path | Mapping[str, object] | None = None,
    feedback_mode: str = "off",
    feedback_config: Mapping[str, object] | None = None,
    trajectory_output_path: str | Path | None = None,
    trace_path: str | Path | None = None,
    reset_trace: bool = True,
    resume_from_checkpoint: str | Path | None = None,
):
    """Create a callback that observes or applies Rust-owned LR controls.

    Apply mode changes learning rates only inside ``optimizer.step()`` and
    restores the scheduler-owned nominal values before the scheduler advances.
    """

    try:
        transformers = importlib.import_module("transformers")
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError(
            "hf_zspace_optimizer_control_callback requires transformers"
        ) from exc
    base_cls = getattr(transformers, "TrainerCallback", object)
    resolved_trajectory = (
        None if trajectory is None else _validated_parameter_trajectory(trajectory)
    )
    resolved_trajectory_id = (
        None
        if resolved_trajectory is None
        else str(resolved_trajectory["trajectory_id"])
    )
    recipe = hf_zspace_optimizer_recipe_contract(
        mode=mode,
        z_dim=z_dim,
        curvature=curvature,
        control_gain=control_gain,
        volume_per_step=volume_per_step,
        trajectory_arm=trajectory_arm,
        trajectory_id=resolved_trajectory_id,
        feedback_mode=feedback_mode,
        feedback_config=feedback_config,
    )
    resolved_mode = str(recipe["mode"])
    resolved_arm = str(recipe["trajectory_arm"])
    resolved_feedback_mode = str(recipe["feedback_mode"])
    resolved_trace = None if trace_path is None else Path(trace_path)
    resolved_trajectory_output = (
        None if trajectory_output_path is None else Path(trajectory_output_path)
    )
    resolved_resume = (
        None if resume_from_checkpoint is None else Path(resume_from_checkpoint)
    )

    class SpiralTorchHFZSpaceOptimizerCallback(base_cls):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            self.recipe = dict(recipe)
            self.mode = resolved_mode
            self.trajectory_arm = resolved_arm
            self.feedback_mode = resolved_feedback_mode
            recipe_feedback_config = self.recipe.get("feedback_config")
            self.feedback_config = (
                None if recipe_feedback_config is None else dict(recipe_feedback_config)
            )
            self.feedback_state: dict[str, object] | None = None
            self.feedback_observation_sequence: list[dict[str, object]] = []
            self.feedback_control_sequence: list[dict[str, object]] = []
            self.trajectory_report = (
                None
                if resolved_trajectory is None
                else _json_clone(resolved_trajectory)
            )
            self.trajectory_generated = False
            self.trajectory_path: str | None = None
            self.trajectory_output_path = resolved_trajectory_output
            self.trace_path = resolved_trace
            self.optimizer: object | None = None
            self.zspace_trainer: object | None = None
            self.pending: dict[str, object] | None = None
            self.active_nominal_rates: list[float] | None = None
            self.active_effective_rates: list[float] | None = None
            self.active_target_step: int | None = None
            self.active_consumed: dict[str, object] | None = None
            self.active_feedback_report: dict[str, object] | None = None
            self.expected_target_step: int | None = None
            self.last_consumed_target_step: int | None = None
            self.max_steps: int | None = None
            self.derived_sequence: list[dict[str, object]] = []
            self.consumed_sequence: list[dict[str, object]] = []
            self.schedule_sequence: list[dict[str, object]] = []
            self.schedule_prefix_missing_count = 0
            self.applied_update_count = 0
            self.non_identity_update_count = 0
            self.observed_update_count = 0
            self.restored_update_count = 0
            self.resume_state_loaded = False
            self.resume_state_path: str | None = None
            self.resume_state_id: str | None = None
            self.trace_parent_path: str | None = None
            self.trace_parent_sha256: str | None = None
            self.trace_parent_size_bytes: int | None = None
            self.trace_segmented_on_resume = False
            self.started = False
            self.finished = False
            self.failure: str | None = None
            self.pre_hook_handle: object | None = None
            self.post_hook_handle: object | None = None
            if self.trace_path is not None and reset_trace and resolved_resume is None:
                self.trace_path.parent.mkdir(parents=True, exist_ok=True)
                self.trace_path.write_text("", encoding="utf-8")
            if self.mode != "off":
                st = importlib.import_module("spiraltorch")
                self.zspace_trainer = st.ZSpaceTrainer(
                    z_dim=int(self.recipe["z_dim"]),
                    topos_control_gain=float(self.recipe["control_gain"]),
                )
                if self.feedback_mode != "off":
                    if self.feedback_config is None:
                        raise RuntimeError(
                            "active Z-space optimizer feedback has no Rust config"
                        )
                    checkpoint = st.zspace_optimizer_feedback_init(self.feedback_config)
                    checkpoint_config = checkpoint.get("config")
                    checkpoint_state = checkpoint.get("state")
                    if (
                        not isinstance(checkpoint_config, Mapping)
                        or dict(checkpoint_config) != self.feedback_config
                        or not isinstance(checkpoint_state, Mapping)
                    ):
                        raise RuntimeError(
                            "Rust feedback initialization changed the sealed recipe"
                        )
                    self.feedback_state = _json_clone(checkpoint_state)
                if resolved_resume is not None:
                    self._restore_resume_state(st, resolved_resume)
                    self._start_resume_trace_segment()

        def _trace(self, event: str, **payload: object) -> None:
            _append_jsonl(
                self.trace_path,
                {
                    "schema": HF_ZSPACE_OPTIMIZER_TRACE_SCHEMA,
                    "event": event,
                    "mode": self.mode,
                    **payload,
                },
            )

        def _restore_resume_state(self, st: object, path: Path) -> None:
            state_path = (
                path if path.is_file() else path / HF_ZSPACE_OPTIMIZER_STATE_FILENAME
            )
            if not state_path.is_file():
                raise RuntimeError(
                    "Z-space optimizer resume state is missing: " + str(state_path)
                )
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise RuntimeError("Z-space optimizer resume state must be a mapping")
            payload = dict(payload)
            state_id = payload.pop("state_id", None)
            if not _is_sha256_id(state_id) or state_id != _sha256_id(payload):
                raise RuntimeError("Z-space optimizer resume-state identity mismatch")
            self.resume_state_id = str(state_id)
            state_schema = payload.get("schema")
            if state_schema not in {
                HF_ZSPACE_OPTIMIZER_STATE_SCHEMA,
                _HF_ZSPACE_OPTIMIZER_V2_STATE_SCHEMA,
                _HF_ZSPACE_OPTIMIZER_LEGACY_STATE_SCHEMA,
            }:
                raise RuntimeError("unsupported Z-space optimizer resume-state schema")
            if (
                state_schema != HF_ZSPACE_OPTIMIZER_STATE_SCHEMA
                and self.feedback_mode != "off"
            ):
                raise RuntimeError(
                    "legacy Z-space optimizer state has no feedback lineage"
                )
            stored_recipe = payload.get("recipe")
            recipe_matches = stored_recipe == self.recipe
            if state_schema != HF_ZSPACE_OPTIMIZER_STATE_SCHEMA:
                recipe_matches = (
                    recipe_matches
                    or stored_recipe
                    == _legacy_optimizer_recipe(self.recipe, str(state_schema))
                )
            if not recipe_matches:
                raise RuntimeError("Z-space optimizer resume recipe does not match")
            trainer_state = payload.get("zspace_trainer_state")
            if not isinstance(trainer_state, Mapping) or self.zspace_trainer is None:
                raise RuntimeError("Z-space optimizer resume state is incomplete")
            self.zspace_trainer.load_state_dict(dict(trainer_state), strict=True)
            self.max_steps = _positive_int(payload.get("max_steps"), label="max_steps")
            self.resume_hf_step = _safe_step(
                payload.get("hf_global_step"),
                label="hf_global_step",
            )
            if self.feedback_mode != "off":
                feedback_state = payload.get("feedback_state")
                if self.feedback_config is None or not isinstance(
                    feedback_state, Mapping
                ):
                    raise RuntimeError(
                        "Z-space optimizer feedback resume state is incomplete"
                    )
                feedback_checkpoint = st.zspace_optimizer_feedback_restore(
                    config=self.feedback_config,
                    state=dict(feedback_state),
                )
                restored_feedback_state = feedback_checkpoint.get("state")
                if not isinstance(restored_feedback_state, Mapping):
                    raise RuntimeError(
                        "Rust feedback restore returned no validated state"
                    )
                feedback_control_step = _safe_step(
                    restored_feedback_state.get("control_step"),
                    label="feedback_state.control_step",
                )
                if feedback_control_step != self.resume_hf_step:
                    raise RuntimeError(
                        "Z-space optimizer feedback step does not match Trainer"
                    )
                self.feedback_state = _json_clone(restored_feedback_state)
                self.feedback_observation_sequence = _sequence_rows(
                    payload.get("feedback_observation_sequence"),
                    label="feedback_observation_sequence",
                )
                self.feedback_control_sequence = _sequence_rows(
                    payload.get("feedback_control_sequence"),
                    label="feedback_control_sequence",
                )
                self._validate_feedback_resume_lineage()
            elif state_schema == HF_ZSPACE_OPTIMIZER_STATE_SCHEMA:
                if payload.get("feedback_state") is not None or any(
                    payload.get(field)
                    for field in (
                        "feedback_observation_sequence",
                        "feedback_control_sequence",
                    )
                ):
                    raise RuntimeError(
                        "disabled Z-space optimizer feedback has unexpected state"
                    )
            self.derived_sequence = _sequence_rows(
                payload.get("derived_sequence"),
                label="derived_sequence",
            )
            self.consumed_sequence = _sequence_rows(
                payload.get("consumed_sequence"),
                label="consumed_sequence",
            )
            schedule_sequence = payload.get("schedule_sequence")
            if (
                schedule_sequence is None
                and state_schema == _HF_ZSPACE_OPTIMIZER_LEGACY_STATE_SCHEMA
            ):
                self.schedule_sequence = []
                self.schedule_prefix_missing_count = len(self.consumed_sequence)
            else:
                self.schedule_sequence = _sequence_rows(
                    schedule_sequence,
                    label="schedule_sequence",
                )
                self.schedule_prefix_missing_count = (
                    0
                    if state_schema == _HF_ZSPACE_OPTIMIZER_LEGACY_STATE_SCHEMA
                    else _non_negative_int(
                        payload.get("schedule_prefix_missing_count"),
                        label="schedule_prefix_missing_count",
                    )
                )
            if len(self.schedule_sequence) + self.schedule_prefix_missing_count != len(
                self.consumed_sequence
            ):
                raise RuntimeError(
                    "Z-space optimizer resume schedule coverage is inconsistent"
                )
            state_input_trajectory_id = payload.get("input_trajectory_id")
            if state_input_trajectory_id != self.recipe.get("trajectory_id"):
                raise RuntimeError(
                    "Z-space optimizer resume trajectory identity does not match"
                )
            self.applied_update_count = _safe_step(
                payload.get("applied_update_count"),
                label="applied_update_count",
            )
            self.non_identity_update_count = _safe_step(
                payload.get("non_identity_update_count"),
                label="non_identity_update_count",
            )
            self.observed_update_count = _safe_step(
                payload.get("observed_update_count"),
                label="observed_update_count",
            )
            self.restored_update_count = _safe_step(
                payload.get("restored_update_count"),
                label="restored_update_count",
            )
            self._restore_trace_parent(payload)
            pending_report = payload.get("pending_report")
            if pending_report is not None:
                if not isinstance(pending_report, Mapping):
                    raise RuntimeError("Z-space optimizer pending report is malformed")
                control = st.zspace_parameter_control(dict(pending_report))
                source_step = _safe_step(
                    payload.get("pending_hf_source_step"),
                    label="pending_hf_source_step",
                )
                target_step = _positive_int(
                    payload.get("pending_hf_target_step"),
                    label="pending_hf_target_step",
                )
                if target_step != self.resume_hf_step + 1:
                    raise RuntimeError(
                        "Z-space optimizer pending target is not resumable"
                    )
                self.pending = {
                    "hf_source_step": source_step,
                    "hf_target_step": target_step,
                    "report": _json_clone(pending_report),
                    "control": _json_clone(control),
                }
            self.resume_state_loaded = True
            self.resume_state_path = str(state_path)

        def _validate_feedback_resume_lineage(self) -> None:
            if self.feedback_state is None:
                raise RuntimeError("Z-space optimizer feedback state is unavailable")
            control_step = _safe_step(
                self.feedback_state.get("control_step"),
                label="feedback_state.control_step",
            )
            if len(self.feedback_control_sequence) != control_step:
                raise RuntimeError(
                    "Z-space optimizer feedback control lineage is incomplete"
                )
            for expected_step, row in enumerate(
                self.feedback_control_sequence,
                start=1,
            ):
                target_step = _positive_int(
                    row.get("hf_target_step"),
                    label="feedback_control_sequence.hf_target_step",
                )
                if target_step != expected_step:
                    raise RuntimeError(
                        "Z-space optimizer feedback control lineage is not contiguous"
                    )
                if (
                    row.get("feedback_contract")
                    != ZSPACE_OPTIMIZER_FEEDBACK_CONTRACT_VERSION
                    or row.get("feedback_semantic_owner")
                    != ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_OWNER
                ):
                    raise RuntimeError(
                        "Z-space optimizer feedback control provenance is invalid"
                    )
                _finite_float(
                    row.get("proposed_learning_rate_scale"),
                    label="feedback_control_sequence.proposed_learning_rate_scale",
                )
                _finite_float(
                    row.get("applied_learning_rate_scale"),
                    label="feedback_control_sequence.applied_learning_rate_scale",
                )
            observation_count = _safe_step(
                self.feedback_state.get("observation_count"),
                label="feedback_state.observation_count",
            )
            if len(self.feedback_observation_sequence) != observation_count:
                raise RuntimeError(
                    "Z-space optimizer feedback observation lineage is incomplete"
                )
            previous_step: int | None = None
            for row in self.feedback_observation_sequence:
                observation_step = _safe_step(
                    row.get("hf_global_step"),
                    label="feedback_observation_sequence.hf_global_step",
                )
                if (
                    observation_step > control_step
                    or previous_step is not None
                    and observation_step <= previous_step
                ):
                    raise RuntimeError(
                        "Z-space optimizer feedback observation lineage is not ordered"
                    )
                if (
                    row.get("feedback_contract")
                    != ZSPACE_OPTIMIZER_FEEDBACK_CONTRACT_VERSION
                    or row.get("feedback_semantic_owner")
                    != ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_OWNER
                ):
                    raise RuntimeError(
                        "Z-space optimizer feedback observation provenance is invalid"
                    )
                _finite_float(
                    row.get("loss"),
                    label="feedback_observation_sequence.loss",
                )
                previous_step = observation_step
            last_observation_step = self.feedback_state.get("last_observation_step")
            if previous_step != last_observation_step:
                raise RuntimeError(
                    "Z-space optimizer feedback observation tail does not match state"
                )

        def _restore_trace_parent(self, payload: Mapping[str, object]) -> None:
            parent_path_value = payload.get("trace_path")
            parent_sha256 = payload.get("trace_sha256")
            parent_size = payload.get("trace_size_bytes")
            if (
                parent_path_value is None
                and parent_sha256 is None
                and parent_size is None
            ):
                return
            if not isinstance(parent_path_value, str) or not _is_sha256_id(
                parent_sha256
            ):
                raise RuntimeError(
                    "Z-space optimizer resume trace lineage is malformed"
                )
            resolved_size = _non_negative_int(
                parent_size,
                label="trace_size_bytes",
            )
            candidates = [Path(parent_path_value)]
            if self.trace_path is not None and self.trace_path not in candidates:
                candidates.append(self.trace_path)
            matched_path: Path | None = None
            for candidate in candidates:
                if _trace_prefix_matches(
                    candidate,
                    resolved_size,
                    str(parent_sha256),
                ):
                    matched_path = candidate
                    break
            if matched_path is None:
                raise RuntimeError(
                    "Z-space optimizer resume trace does not contain the "
                    "checkpoint prefix"
                )
            self.trace_parent_path = str(matched_path)
            self.trace_parent_sha256 = str(parent_sha256)
            self.trace_parent_size_bytes = resolved_size

        def _start_resume_trace_segment(self) -> None:
            if self.trace_path is None:
                return
            requested = self.trace_path
            requested.parent.mkdir(parents=True, exist_ok=True)
            candidate = requested
            if candidate.exists():
                suffix = candidate.suffix or ".jsonl"
                stem = (
                    candidate.name[: -len(suffix)]
                    if candidate.suffix
                    else candidate.name
                )
                segment = _safe_step(
                    getattr(self, "resume_hf_step", 0),
                    label="resume_hf_step",
                )
                index = 1
                while True:
                    candidate = requested.with_name(
                        f"{stem}.resume-{segment}.{index}{suffix}"
                    )
                    if not candidate.exists():
                        break
                    index += 1
            candidate.write_text("", encoding="utf-8")
            self.trace_path = candidate
            self.trace_segmented_on_resume = True
            self._trace(
                "trace_segment_started",
                hf_global_step=getattr(self, "resume_hf_step", None),
                resume_state_id=self.resume_state_id,
                parent_trace_path=self.trace_parent_path,
                parent_trace_sha256=self.trace_parent_sha256,
                parent_trace_size_bytes=self.trace_parent_size_bytes,
            )

        def _derive(self, hf_source_step: int) -> None:
            if self.zspace_trainer is None or self.max_steps is None:
                raise RuntimeError("Z-space optimizer controller is not initialized")
            if hf_source_step >= self.max_steps:
                self.pending = None
                return
            st = importlib.import_module("spiraltorch")
            volume_per_step_value = int(self.recipe["volume_per_step"])
            partial = st.topos_control_partial(
                curvature=float(self.recipe["curvature"]),
                max_depth=self.max_steps,
                max_volume=max(
                    volume_per_step_value,
                    self.max_steps * volume_per_step_value,
                ),
                observed_depth=hf_source_step,
                visited_volume=hf_source_step * volume_per_step_value,
                gradient_dim=int(self.recipe["z_dim"]),
            )
            self.zspace_trainer.step_partial(partial)
            report = self.zspace_trainer.last_optimizer_report
            if not isinstance(report, Mapping):
                raise RuntimeError("Z-space trainer emitted no optimizer report")
            control = st.zspace_parameter_control(report)
            target_step = hf_source_step + 1
            sequence_row = {
                "hf_source_step": hf_source_step,
                "hf_target_step": target_step,
                "zspace_source_step": control["source_step"],
                "absolute_learning_rate_scale": control["absolute_learning_rate_scale"],
                "parameter_control_contract": control["contract_version"],
                "parameter_control_semantic_owner": control["semantic_owner"],
            }
            self.derived_sequence.append(sequence_row)
            self.pending = {
                "hf_source_step": hf_source_step,
                "hf_target_step": target_step,
                "report": _json_clone(report),
                "control": _json_clone(control),
            }
            self._trace(
                "control_derived",
                **sequence_row,
                control=_json_clone(control),
                topos_control=_json_clone(report.get("topos_control")),
            )

        def _planned_scale(
            self,
            *,
            target_step: int,
            raw_scale: float,
            nominal_rates: Sequence[float],
        ) -> float:
            if self.trajectory_report is None:
                if self.trajectory_arm != "raw":
                    raise RuntimeError(
                        "non-raw Z-space actuation has no validated trajectory"
                    )
                return raw_scale
            steps = self.trajectory_report.get("steps")
            if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
                raise RuntimeError("validated Z-space trajectory has no steps")
            index = target_step - 1
            if index < 0 or index >= len(steps):
                raise RuntimeError(
                    "Z-space trajectory does not cover the optimizer update"
                )
            row = steps[index]
            if not isinstance(row, Mapping) or row.get("index") != index:
                raise RuntimeError("Z-space trajectory step index is inconsistent")
            planned_raw = _finite_float(
                row.get("raw_learning_rate_scale"),
                label="trajectory.raw_learning_rate_scale",
            )
            if planned_raw != raw_scale:
                raise RuntimeError(
                    "live Rust control differs from the calibrated trajectory"
                )
            planned_rates = row.get("nominal_learning_rates")
            if not isinstance(planned_rates, Sequence) or isinstance(
                planned_rates, (str, bytes)
            ):
                raise RuntimeError(
                    "Z-space trajectory nominal learning rates are malformed"
                )
            if len(planned_rates) != len(nominal_rates) or any(
                _finite_float(planned, label="trajectory.nominal_learning_rate")
                != float(observed)
                for planned, observed in zip(planned_rates, nominal_rates)
            ):
                raise RuntimeError(
                    "scheduler nominal learning rates differ from the calibrated "
                    "trajectory"
                )
            field = {
                "raw": "raw_learning_rate_scale",
                "dose_matched_constant": "dose_matched_constant_scale",
                "dose_normalized": "dose_normalized_scale",
            }[self.trajectory_arm]
            return _finite_float(row.get(field), label=f"trajectory.{field}")

        def _consume_pending(
            self, *, nominal_rates: Sequence[float]
        ) -> tuple[dict[str, object], dict[str, object] | None]:
            pending = self.pending
            if not isinstance(pending, Mapping):
                raise RuntimeError("no Rust-validated control is pending")
            target_step = _positive_int(
                pending.get("hf_target_step"),
                label="hf_target_step",
            )
            if self.expected_target_step != target_step:
                raise RuntimeError(
                    "Z-space control target does not match the optimizer update"
                )
            control = pending.get("control")
            if not isinstance(control, Mapping):
                raise RuntimeError("pending Z-space parameter control is malformed")
            scale = _finite_float(
                control.get("absolute_learning_rate_scale"),
                label="absolute_learning_rate_scale",
            )
            planned_scale = self._planned_scale(
                target_step=target_step,
                raw_scale=scale,
                nominal_rates=nominal_rates,
            )
            feedback_scale = planned_scale
            feedback_report: dict[str, object] | None = None
            if self.feedback_mode != "off":
                if self.feedback_config is None or self.feedback_state is None:
                    raise RuntimeError(
                        "active Z-space optimizer feedback state is unavailable"
                    )
                st = importlib.import_module("spiraltorch")
                feedback_report = st.zspace_optimizer_feedback_control(
                    config=self.feedback_config,
                    state=self.feedback_state,
                    target_step=target_step,
                    proposed_learning_rate_scale=planned_scale,
                )
                feedback_scale = _finite_float(
                    feedback_report.get("applied_learning_rate_scale"),
                    label="feedback.applied_learning_rate_scale",
                )
            applied_scale = feedback_scale if self.mode == "apply" else 1.0
            effective_rates = [float(rate) * applied_scale for rate in nominal_rates]
            consumed = {
                "hf_source_step": pending["hf_source_step"],
                "hf_target_step": target_step,
                "zspace_source_step": control["source_step"],
                "absolute_learning_rate_scale": scale,
                "planned_learning_rate_scale": planned_scale,
                "feedback_learning_rate_scale": feedback_scale,
                "applied_learning_rate_scale": applied_scale,
                "trajectory_arm": self.trajectory_arm,
                "trajectory_id": (
                    None
                    if self.trajectory_report is None
                    else self.trajectory_report.get("trajectory_id")
                ),
                "parameter_control_contract": control["contract_version"],
                "parameter_control_semantic_owner": control["semantic_owner"],
                "feedback_mode": self.feedback_mode,
                "feedback_contract": (
                    None
                    if feedback_report is None
                    else feedback_report.get("contract_version")
                ),
                "feedback_semantic_owner": (
                    None
                    if feedback_report is None
                    else feedback_report.get("semantic_owner")
                ),
                "feedback_disposition": (
                    None
                    if feedback_report is None
                    else feedback_report.get("disposition")
                ),
                "feedback_gate": (
                    None
                    if feedback_report is None
                    else feedback_report.get("feedback_gate")
                ),
                "effective_feedback_gate": (
                    None
                    if feedback_report is None
                    else feedback_report.get("effective_feedback_gate")
                ),
                "feedback_observation_age_updates": (
                    None
                    if feedback_report is None
                    else feedback_report.get("feedback_observation_age_updates")
                ),
                "nominal_learning_rates": list(nominal_rates),
                "effective_learning_rates": effective_rates,
            }
            return consumed, feedback_report

        def _commit_active_consumed(self, *, applied: bool) -> dict[str, object]:
            consumed = self.active_consumed
            if consumed is None:
                raise RuntimeError("completed optimizer update has no active control")
            feedback_report = self.active_feedback_report
            if self.feedback_mode != "off":
                if not isinstance(feedback_report, Mapping):
                    raise RuntimeError(
                        "completed optimizer update has no Rust feedback transition"
                    )
                state_after = feedback_report.get("state_after")
                if not isinstance(state_after, Mapping):
                    raise RuntimeError("Rust feedback transition has no state_after")
                self.feedback_state = _json_clone(state_after)
                self.feedback_control_sequence.append(
                    {
                        "hf_target_step": consumed["hf_target_step"],
                        "proposed_learning_rate_scale": feedback_report[
                            "proposed_learning_rate_scale"
                        ],
                        "applied_learning_rate_scale": feedback_report[
                            "applied_learning_rate_scale"
                        ],
                        "disposition": feedback_report["disposition"],
                        "feedback_gate": feedback_report["feedback_gate"],
                        "effective_feedback_gate": feedback_report[
                            "effective_feedback_gate"
                        ],
                        "feedback_observation_age_updates": feedback_report[
                            "feedback_observation_age_updates"
                        ],
                        "feedback_contract": feedback_report["contract_version"],
                        "feedback_semantic_owner": feedback_report["semantic_owner"],
                    }
                )
            self._record_consumed(consumed)
            if applied:
                self.applied_update_count += 1
                if any(
                    not math.isclose(
                        float(nominal),
                        float(effective),
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                    for nominal, effective in zip(
                        consumed["nominal_learning_rates"],
                        consumed["effective_learning_rates"],
                    )
                ):
                    self.non_identity_update_count += 1
            else:
                self.observed_update_count += 1
            target_step = int(consumed["hf_target_step"])
            self.last_consumed_target_step = target_step
            self.pending = None
            self.active_consumed = None
            self.active_feedback_report = None
            return consumed

        def _record_consumed(self, consumed: Mapping[str, object]) -> None:
            self.consumed_sequence.append(
                {
                    key: consumed[key]
                    for key in (
                        "hf_source_step",
                        "hf_target_step",
                        "zspace_source_step",
                        "absolute_learning_rate_scale",
                        "planned_learning_rate_scale",
                        "feedback_learning_rate_scale",
                        "feedback_mode",
                        "feedback_contract",
                        "feedback_semantic_owner",
                        "feedback_disposition",
                        "feedback_gate",
                        "effective_feedback_gate",
                        "feedback_observation_age_updates",
                        "parameter_control_contract",
                        "parameter_control_semantic_owner",
                    )
                }
            )
            self.schedule_sequence.append(
                {
                    key: _json_clone(consumed[key])
                    for key in (
                        "hf_target_step",
                        "absolute_learning_rate_scale",
                        "planned_learning_rate_scale",
                        "feedback_learning_rate_scale",
                        "applied_learning_rate_scale",
                        "trajectory_arm",
                        "trajectory_id",
                        "nominal_learning_rates",
                        "effective_learning_rates",
                    )
                }
            )

        def _pre_step_hook(self, optimizer, args, kwargs):  # type: ignore[no-untyped-def]
            if (
                self.active_nominal_rates is not None
                or self.active_consumed is not None
            ):
                raise RuntimeError("nested optimizer.step is unsupported")
            nominal_rates = _optimizer_learning_rates(optimizer)
            consumed, feedback_report = self._consume_pending(
                nominal_rates=nominal_rates
            )
            effective_rates = list(consumed["effective_learning_rates"])
            _set_optimizer_learning_rates(optimizer, effective_rates)
            self.active_nominal_rates = nominal_rates
            self.active_effective_rates = effective_rates
            self.active_target_step = int(consumed["hf_target_step"])
            self.active_consumed = consumed
            self.active_feedback_report = feedback_report
            self._trace("optimizer_step_pre", **consumed)
            return None

        def _post_step_hook(self, optimizer, args, kwargs):  # type: ignore[no-untyped-def]
            if self.active_nominal_rates is None or self.active_target_step is None:
                raise RuntimeError("optimizer.step post-hook has no active control")
            nominal_rates = list(self.active_nominal_rates)
            effective_rates = _optimizer_learning_rates(optimizer)
            if self.active_effective_rates is None or any(
                not math.isclose(
                    observed,
                    expected,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                for observed, expected in zip(
                    effective_rates,
                    self.active_effective_rates,
                )
            ):
                raise RuntimeError(
                    "optimizer learning rates changed inside the controlled step"
                )
            _set_optimizer_learning_rates(optimizer, nominal_rates)
            consumed = self._commit_active_consumed(applied=True)
            self.restored_update_count += 1
            self.active_nominal_rates = None
            self.active_effective_rates = None
            self.active_target_step = None
            self._trace(
                "optimizer_step_post",
                **consumed,
                nominal_learning_rates_restored=True,
            )
            return None

        def _bind_optimizer(self, optimizer: object) -> None:
            if self.mode != "apply":
                self.optimizer = optimizer
                return
            self.optimizer = _hookable_optimizer(optimizer)
            self.pre_hook_handle = self.optimizer.register_step_pre_hook(
                self._pre_step_hook
            )
            self.post_hook_handle = self.optimizer.register_step_post_hook(
                self._post_step_hook
            )

        def _remove_hooks(self) -> None:
            for attribute in ("pre_hook_handle", "post_hook_handle"):
                handle = getattr(self, attribute)
                remove = getattr(handle, "remove", None)
                if callable(remove):
                    remove()
                setattr(self, attribute, None)

        def _restore_active_rates(self) -> None:
            if self.optimizer is not None and self.active_nominal_rates is not None:
                _set_optimizer_learning_rates(
                    self.optimizer,
                    self.active_nominal_rates,
                )
                self.active_nominal_rates = None
                self.active_effective_rates = None
                self.active_target_step = None
            self.active_consumed = None
            self.active_feedback_report = None

        def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            self.started = True
            if self.mode == "off":
                self._trace(
                    "train_begin",
                    hf_global_step=_safe_step(state.global_step, label="global_step"),
                )
                return control
            global_step = _safe_step(state.global_step, label="global_step")
            max_steps = _positive_int(state.max_steps, label="max_steps")
            if self.max_steps is not None and self.max_steps != max_steps:
                raise RuntimeError("Z-space optimizer resume max_steps does not match")
            self.max_steps = max_steps
            if self.trajectory_report is not None:
                planned_steps = _positive_int(
                    self.trajectory_report.get("step_count"),
                    label="trajectory.step_count",
                )
                if planned_steps != max_steps:
                    raise RuntimeError(
                        "Z-space trajectory horizon does not match Trainer max_steps"
                    )
            if self.resume_state_loaded:
                if self.resume_hf_step != global_step:
                    raise RuntimeError(
                        "Z-space optimizer resume step does not match Trainer"
                    )
            elif global_step != 0:
                raise RuntimeError(
                    "nonzero Trainer state requires a Z-space optimizer resume state"
                )
            if self.feedback_mode != "off":
                if self.feedback_state is None:
                    raise RuntimeError(
                        "Z-space optimizer feedback state is unavailable"
                    )
                feedback_step = _safe_step(
                    self.feedback_state.get("control_step"),
                    label="feedback_state.control_step",
                )
                if feedback_step != global_step:
                    raise RuntimeError(
                        "Z-space optimizer feedback step does not match Trainer"
                    )
            optimizer = kwargs.get("optimizer")
            if optimizer is None:
                raise RuntimeError(
                    "Trainer did not expose its optimizer to the callback"
                )
            self._bind_optimizer(optimizer)
            if self.pending is None:
                self._derive(global_step)
            self._trace(
                "train_begin",
                hf_global_step=global_step,
                max_steps=max_steps,
                resume_state_loaded=self.resume_state_loaded,
            )
            return control

        def on_step_begin(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            if self.mode == "off":
                return control
            global_step = _safe_step(state.global_step, label="global_step")
            self.expected_target_step = global_step + 1
            if self.mode == "observe":
                if self.active_consumed is not None:
                    raise RuntimeError(
                        "previous observed optimizer update is unresolved"
                    )
                optimizer = kwargs.get("optimizer", self.optimizer)
                hookable = _hookable_optimizer(optimizer)
                nominal_rates = _optimizer_learning_rates(hookable)
                consumed, feedback_report = self._consume_pending(
                    nominal_rates=nominal_rates
                )
                self.active_consumed = consumed
                self.active_feedback_report = feedback_report
            return control

        def on_step_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            if self.mode == "off":
                return control
            global_step = _safe_step(state.global_step, label="global_step")
            if self.mode == "observe":
                if (
                    self.active_consumed is None
                    or int(self.active_consumed["hf_target_step"]) != global_step
                ):
                    raise RuntimeError(
                        "observed optimizer update has no matching active control"
                    )
                consumed = self._commit_active_consumed(applied=False)
                self._trace("optimizer_step_observed", **consumed)
            if self.last_consumed_target_step != global_step:
                raise RuntimeError(
                    "Z-space control was not consumed by the completed optimizer update"
                )
            self.expected_target_step = None
            self._derive(global_step)
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[no-untyped-def]
            if self.feedback_mode == "off" or not isinstance(logs, Mapping):
                return control
            if "loss" not in logs:
                return control
            if self.feedback_config is None or self.feedback_state is None:
                raise RuntimeError("Z-space optimizer feedback state is unavailable")
            observation_step = _safe_step(
                state.global_step,
                label="feedback_observation.step",
            )
            observation: dict[str, object] = {
                "step": observation_step,
                "max_steps": self.max_steps,
                "loss": logs["loss"],
            }
            for field, fallback in (
                ("epoch", getattr(state, "epoch", None)),
                ("grad_norm", None),
                ("learning_rate", None),
            ):
                value = logs.get(field, fallback)
                if value is not None:
                    observation[field] = value
            st = importlib.import_module("spiraltorch")
            report = st.zspace_optimizer_feedback_observe(
                config=self.feedback_config,
                state=self.feedback_state,
                observation=observation,
            )
            state_after = report.get("state_after")
            if not isinstance(state_after, Mapping):
                raise RuntimeError("Rust feedback observation has no state_after")
            self.feedback_state = _json_clone(state_after)
            sequence_row = {
                "hf_global_step": observation_step,
                "loss": observation["loss"],
                "action": report["action"],
                "relative_loss_delta": report["relative_loss_delta"],
                "relative_loss_delta_ema": report["relative_loss_delta_ema"],
                "gate_before": report["gate_before"],
                "gate_after": report["gate_after"],
                "feedback_contract": report["contract_version"],
                "feedback_semantic_owner": report["semantic_owner"],
                "projection": _json_clone(report["projection"]),
            }
            self.feedback_observation_sequence.append(sequence_row)
            self._trace("feedback_observed", **sequence_row)
            return control

        def _finalize_trajectory(self, output_dir: str | Path) -> None:
            if self.max_steps is None:
                raise RuntimeError("Z-space optimizer horizon is unavailable")
            realized_steps = len(self.consumed_sequence)
            if realized_steps > self.max_steps:
                raise RuntimeError(
                    "Z-space optimizer consumed beyond the planned horizon"
                )
            if len(self.schedule_sequence) + self.schedule_prefix_missing_count != (
                realized_steps
            ):
                raise RuntimeError(
                    "Z-space optimizer schedule coverage is inconsistent"
                )
            if self.schedule_prefix_missing_count > 0:
                self._trace(
                    "trajectory_unavailable",
                    reason="legacy_state_has_no_historical_scheduler_rows",
                    planned_step_count=self.max_steps,
                    realized_step_count=realized_steps,
                    schedule_prefix_missing_count=self.schedule_prefix_missing_count,
                )
                return
            if realized_steps == 0:
                self._trace(
                    "trajectory_unavailable",
                    reason="no_realized_optimizer_updates",
                    planned_step_count=self.max_steps,
                    realized_step_count=0,
                )
                return
            st = importlib.import_module("spiraltorch")
            if self.trajectory_report is None:
                self.trajectory_report = st.zspace_parameter_trajectory(
                    raw_learning_rate_scales=[
                        float(row["absolute_learning_rate_scale"])
                        for row in self.schedule_sequence
                    ],
                    nominal_learning_rates=[
                        list(row["nominal_learning_rates"])
                        for row in self.schedule_sequence
                    ],
                )
                self.trajectory_generated = True
            else:
                self.trajectory_report = st.validate_zspace_parameter_trajectory(
                    self.trajectory_report
                )
            path = self.trajectory_output_path or (
                Path(output_dir) / HF_ZSPACE_OPTIMIZER_TRAJECTORY_FILENAME
            )
            _write_json(path, self.trajectory_report)
            self.trajectory_path = str(path)
            trajectory_steps = _positive_int(
                self.trajectory_report.get("step_count"),
                label="trajectory.step_count",
            )
            if trajectory_steps != realized_steps:
                self._trace(
                    "trajectory_partially_consumed",
                    trajectory_id=self.trajectory_report.get("trajectory_id"),
                    trajectory_arm=self.trajectory_arm,
                    trajectory_path=self.trajectory_path,
                    planned_step_count=self.max_steps,
                    realized_step_count=realized_steps,
                    trajectory_step_count=trajectory_steps,
                )
                return
            self._trace(
                "trajectory_finalized",
                trajectory_id=self.trajectory_report.get("trajectory_id"),
                trajectory_arm=self.trajectory_arm,
                trajectory_generated=self.trajectory_generated,
                trajectory_path=self.trajectory_path,
            )

        def _evidence_blockers(self) -> list[str]:
            if self.mode == "off" or self.max_steps is None:
                return []
            realized_steps = len(self.consumed_sequence)
            blockers: list[str] = []
            if realized_steps == 0:
                blockers.append("no_realized_optimizer_updates")
            if realized_steps != self.max_steps:
                blockers.append("trainer_stopped_before_planned_horizon")
            if self.trajectory_report is not None:
                trajectory_steps = _receipt_count(
                    self.trajectory_report.get("step_count")
                )
                if trajectory_steps != realized_steps:
                    blockers.append("input_trajectory_only_partially_consumed")
            if self.active_consumed is not None:
                blockers.append("optimizer_update_control_not_committed")
            if self.feedback_mode != "off":
                if self.feedback_state is None:
                    blockers.append("feedback_state_missing")
                elif (
                    _safe_step(
                        self.feedback_state.get("control_step"),
                        label="feedback_state.control_step",
                    )
                    != realized_steps
                ):
                    blockers.append("feedback_control_step_mismatch")
                if len(self.feedback_control_sequence) != realized_steps:
                    blockers.append("feedback_control_lineage_incomplete")
            return blockers

        def _state_payload(self, hf_global_step: int) -> dict[str, object]:
            if self.zspace_trainer is None or self.max_steps is None:
                raise RuntimeError("Z-space optimizer state is unavailable")
            if self.active_consumed is not None:
                raise RuntimeError(
                    "cannot checkpoint an unresolved optimizer update control"
                )
            pending = self.pending or {}
            trace_sha256, trace_size_bytes = _trace_file_evidence(self.trace_path)
            payload: dict[str, object] = {
                "schema": HF_ZSPACE_OPTIMIZER_STATE_SCHEMA,
                "recipe": dict(self.recipe),
                "hf_global_step": hf_global_step,
                "max_steps": self.max_steps,
                "zspace_trainer_state": self.zspace_trainer.state_dict(),
                "feedback_state": _json_clone(self.feedback_state),
                "feedback_observation_sequence": _json_clone(
                    self.feedback_observation_sequence
                ),
                "feedback_control_sequence": _json_clone(
                    self.feedback_control_sequence
                ),
                "pending_hf_source_step": pending.get("hf_source_step"),
                "pending_hf_target_step": pending.get("hf_target_step"),
                "pending_report": pending.get("report"),
                "derived_sequence": _json_clone(self.derived_sequence),
                "consumed_sequence": _json_clone(self.consumed_sequence),
                "schedule_sequence": _json_clone(self.schedule_sequence),
                "schedule_prefix_missing_count": self.schedule_prefix_missing_count,
                "input_trajectory_id": self.recipe.get("trajectory_id"),
                "trajectory_id": (
                    None
                    if self.trajectory_report is None
                    else self.trajectory_report.get("trajectory_id")
                ),
                "trajectory_path": self.trajectory_path,
                "applied_update_count": self.applied_update_count,
                "non_identity_update_count": self.non_identity_update_count,
                "observed_update_count": self.observed_update_count,
                "restored_update_count": self.restored_update_count,
                "trace_path": (
                    None if self.trace_path is None else str(self.trace_path)
                ),
                "trace_sha256": trace_sha256,
                "trace_size_bytes": trace_size_bytes,
            }
            payload["state_id"] = _sha256_id(payload)
            return payload

        def on_save(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            if self.mode == "off":
                return control
            global_step = _safe_step(state.global_step, label="global_step")
            path = (
                Path(args.output_dir)
                / f"checkpoint-{global_step}"
                / HF_ZSPACE_OPTIMIZER_STATE_FILENAME
            )
            _write_json(path, self._state_payload(global_step))
            self._trace("state_saved", hf_global_step=global_step, state_path=str(path))
            return control

        def on_train_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            global_step = _safe_step(state.global_step, label="global_step")
            if self.mode != "off":
                if self.mode == "apply" and (
                    self.applied_update_count != self.restored_update_count
                ):
                    raise RuntimeError(
                        "Z-space optimizer learning rates were not restored"
                    )
                self._finalize_trajectory(args.output_dir)
                _write_json(
                    Path(args.output_dir) / HF_ZSPACE_OPTIMIZER_STATE_FILENAME,
                    self._state_payload(global_step),
                )
            self._remove_hooks()
            self.finished = True
            self._trace("train_end", hf_global_step=global_step, receipt=self.receipt())
            return control

        def abort(self, error: object) -> None:
            self.failure = f"{error.__class__.__name__}: {error}"
            self._restore_active_rates()
            self._remove_hooks()
            self._trace("aborted", failure=self.failure)

        def receipt(self) -> dict[str, object]:
            raw_scales = [
                float(row["absolute_learning_rate_scale"])
                for row in self.consumed_sequence
            ]
            raw_control_sequence = [
                {
                    key: row[key]
                    for key in (
                        "hf_source_step",
                        "hf_target_step",
                        "zspace_source_step",
                        "absolute_learning_rate_scale",
                        "parameter_control_contract",
                        "parameter_control_semantic_owner",
                    )
                }
                for row in self.consumed_sequence
            ]
            feedback_scales = (
                []
                if self.feedback_mode == "off"
                else [
                    float(row["feedback_learning_rate_scale"])
                    for row in self.consumed_sequence
                ]
            )
            feedback_dispositions = [
                row["feedback_disposition"]
                for row in self.consumed_sequence
                if row.get("feedback_disposition") is not None
            ]
            feedback_disposition_counts = {
                disposition: feedback_dispositions.count(disposition)
                for disposition in sorted(set(feedback_dispositions))
            }
            schedule_evidence_complete = (
                self.schedule_prefix_missing_count == 0
                and len(self.schedule_sequence) == len(self.consumed_sequence)
            )
            if schedule_evidence_complete:
                applied_scales = [
                    float(row["applied_learning_rate_scale"])
                    for row in self.schedule_sequence
                ]
                nominal_sequence = [
                    {
                        "hf_target_step": row["hf_target_step"],
                        "absolute_learning_rate_scale": row[
                            "absolute_learning_rate_scale"
                        ],
                        "nominal_learning_rates": row["nominal_learning_rates"],
                    }
                    for row in self.schedule_sequence
                ]
                nominal_dose = sum(
                    sum(
                        _finite_float(rate, label="nominal_learning_rate")
                        for rate in row["nominal_learning_rates"]
                    )
                    for row in self.schedule_sequence
                )
                actuated_dose = sum(
                    sum(
                        _finite_float(rate, label="effective_learning_rate")
                        for rate in row["effective_learning_rates"]
                    )
                    for row in self.schedule_sequence
                )
                actuated_dose_ratio = (
                    actuated_dose / nominal_dose if nominal_dose > 0.0 else None
                )
                nominal_sequence_id = _sha256_id(nominal_sequence)
                actuated_sequence_id = _sha256_id(self.schedule_sequence)
            else:
                applied_scales = []
                nominal_dose = None
                actuated_dose = None
                actuated_dose_ratio = None
                nominal_sequence_id = None
                actuated_sequence_id = None
            if self.failure is not None:
                status = "blocked"
            elif self.mode == "off":
                status = "disabled"
            elif not self.started:
                status = "pending"
            elif self.mode == "apply" and (
                self.applied_update_count != self.restored_update_count
            ):
                status = "blocked"
            elif self.finished and self._evidence_blockers():
                status = "blocked"
            elif self.finished:
                status = "ready"
            else:
                status = "active"
            trace_sha256, trace_size_bytes = _trace_file_evidence(self.trace_path)
            if not schedule_evidence_complete:
                evidence_boundary = (
                    "legacy v1 resume preserved optimizer actuation, but the "
                    "checkpoint had no historical scheduler rows; trajectory and "
                    "integrated-dose evidence are unavailable"
                )
            elif self.feedback_mode == "off":
                evidence_boundary = (
                    "receipt proves control derivation and optimizer actuation, "
                    "including Rust-owned trajectory and nominal-LR matching, not "
                    "learning-quality improvement; this progress-derived signal "
                    "does not use gradients or loss feedback"
                )
            else:
                evidence_boundary = (
                    "receipt proves control derivation and optimizer actuation, "
                    "including Rust-owned trajectory, nominal-LR matching, and "
                    "checkpointed within-run loss-feedback gating, not "
                    "learning-quality improvement or counterfactual efficacy"
                )
            return {
                "schema": HF_ZSPACE_OPTIMIZER_RECEIPT_SCHEMA,
                "status": status,
                "mode": self.mode,
                "feedback_mode": self.feedback_mode,
                "trajectory_arm": self.trajectory_arm,
                "recipe": dict(self.recipe),
                "model_update_intervened": self.non_identity_update_count > 0,
                "control_observed": bool(self.consumed_sequence),
                "derived_control_count": len(self.derived_sequence),
                "consumed_control_count": len(self.consumed_sequence),
                "observed_update_count": self.observed_update_count,
                "applied_update_count": self.applied_update_count,
                "non_identity_update_count": self.non_identity_update_count,
                "restored_update_count": self.restored_update_count,
                "unused_pending_control": self.pending is not None,
                "planned_update_count": self.max_steps,
                "realized_update_count": len(self.consumed_sequence),
                "training_horizon_complete": (
                    None
                    if self.max_steps is None
                    else len(self.consumed_sequence) == self.max_steps
                ),
                "trajectory_horizon_complete": (
                    None
                    if self.trajectory_report is None
                    else self.trajectory_report.get("step_count")
                    == len(self.consumed_sequence)
                ),
                "evidence_blockers": self._evidence_blockers(),
                "scale_min": min(raw_scales) if raw_scales else None,
                "scale_max": max(raw_scales) if raw_scales else None,
                "scale_mean": (
                    sum(raw_scales) / len(raw_scales) if raw_scales else None
                ),
                "applied_scale_min": min(applied_scales) if applied_scales else None,
                "applied_scale_max": max(applied_scales) if applied_scales else None,
                "applied_scale_mean": (
                    sum(applied_scales) / len(applied_scales)
                    if applied_scales
                    else None
                ),
                "control_sequence_id": _sha256_id(raw_control_sequence),
                "guarded_control_sequence_id": _sha256_id(self.consumed_sequence),
                "feedback_control_count": len(self.feedback_control_sequence),
                "feedback_observation_count": len(self.feedback_observation_sequence),
                "feedback_control_sequence_id": (
                    None
                    if self.feedback_mode == "off"
                    else _sha256_id(self.feedback_control_sequence)
                ),
                "feedback_observation_sequence_id": (
                    None
                    if self.feedback_mode == "off"
                    else _sha256_id(self.feedback_observation_sequence)
                ),
                "feedback_disposition_counts": feedback_disposition_counts,
                "feedback_active_update_count": feedback_disposition_counts.get(
                    "active", 0
                ),
                "feedback_identity_update_count": sum(
                    math.isclose(scale, 1.0, rel_tol=1e-12, abs_tol=1e-15)
                    for scale in feedback_scales
                ),
                "feedback_stale_update_count": feedback_disposition_counts.get(
                    "stale", 0
                ),
                "feedback_halted_update_count": feedback_disposition_counts.get(
                    "halted", 0
                ),
                "feedback_scale_min": (
                    min(feedback_scales) if feedback_scales else None
                ),
                "feedback_scale_max": (
                    max(feedback_scales) if feedback_scales else None
                ),
                "feedback_scale_mean": (
                    sum(feedback_scales) / len(feedback_scales)
                    if feedback_scales
                    else None
                ),
                "feedback_gate": (
                    None
                    if self.feedback_state is None
                    else self.feedback_state.get("gate")
                ),
                "feedback_halted": (
                    None
                    if self.feedback_state is None
                    else self.feedback_state.get("halted")
                ),
                "feedback_contract": self.recipe.get("feedback_contract"),
                "feedback_semantic_owner": self.recipe.get("feedback_semantic_owner"),
                "feedback_evidence_boundary": self.recipe.get(
                    "feedback_evidence_boundary"
                ),
                "nominal_schedule_sequence_id": nominal_sequence_id,
                "actuated_schedule_sequence_id": actuated_sequence_id,
                "schedule_evidence_complete": schedule_evidence_complete,
                "schedule_prefix_missing_count": self.schedule_prefix_missing_count,
                "nominal_learning_rate_dose": nominal_dose,
                "actuated_learning_rate_dose": actuated_dose,
                "actuated_learning_rate_dose_ratio": actuated_dose_ratio,
                "trajectory_id": (
                    None
                    if self.trajectory_report is None
                    else self.trajectory_report.get("trajectory_id")
                ),
                "trajectory_contract": (
                    None
                    if self.trajectory_report is None
                    else self.trajectory_report.get("contract_version")
                ),
                "trajectory_validated": (
                    None
                    if self.trajectory_report is None
                    else self.trajectory_report.get("trajectory_validated")
                ),
                "trajectory_generated": self.trajectory_generated,
                "trajectory_path": self.trajectory_path,
                "trajectory_step_count": (
                    None
                    if self.trajectory_report is None
                    else self.trajectory_report.get("step_count")
                ),
                "trajectory_identity_relative_tolerance": (
                    None
                    if self.trajectory_report is None
                    else self.trajectory_report.get("identity_relative_tolerance")
                ),
                "trajectory_identity_absolute_tolerance": (
                    None
                    if self.trajectory_report is None
                    else self.trajectory_report.get("identity_absolute_tolerance")
                ),
                "trajectory_raw_non_identity_update_count": (
                    None
                    if self.trajectory_report is None
                    else self.trajectory_report.get("raw_non_identity_update_count")
                ),
                "trajectory_dose_matched_constant_non_identity_update_count": (
                    None
                    if self.trajectory_report is None
                    else self.trajectory_report.get(
                        "dose_matched_constant_non_identity_update_count"
                    )
                ),
                "trajectory_dose_normalized_non_identity_update_count": (
                    None
                    if self.trajectory_report is None
                    else self.trajectory_report.get(
                        "dose_normalized_non_identity_update_count"
                    )
                ),
                "trajectory_nominal_dose": (
                    None
                    if self.trajectory_report is None
                    else self.trajectory_report.get("nominal_dose")
                ),
                "trajectory_raw_dose": (
                    None
                    if self.trajectory_report is None
                    else self.trajectory_report.get("raw_dose")
                ),
                "trajectory_dose_normalized_dose": (
                    None
                    if self.trajectory_report is None
                    else self.trajectory_report.get("dose_normalized_dose")
                ),
                "trajectory_raw_dose_ratio": (
                    None
                    if self.trajectory_report is None
                    else self.trajectory_report.get("raw_dose_ratio")
                ),
                "trajectory_dose_normalized_dose_ratio": (
                    None
                    if self.trajectory_report is None
                    else self.trajectory_report.get("dose_normalized_dose_ratio")
                ),
                "resume_state_loaded": self.resume_state_loaded,
                "resume_state_path": self.resume_state_path,
                "resume_state_id": self.resume_state_id,
                "trace_path": None if self.trace_path is None else str(self.trace_path),
                "trace_sha256": trace_sha256,
                "trace_size_bytes": trace_size_bytes,
                "trace_segmented_on_resume": self.trace_segmented_on_resume,
                "trace_parent_path": self.trace_parent_path,
                "trace_parent_sha256": self.trace_parent_sha256,
                "trace_parent_size_bytes": self.trace_parent_size_bytes,
                "scheduler_nominal_lr_restored": (
                    self.mode != "apply"
                    or self.applied_update_count == self.restored_update_count
                ),
                "failure": self.failure,
                "efficacy_evaluated": False,
                "evidence_boundary": evidence_boundary,
            }

    return SpiralTorchHFZSpaceOptimizerCallback()


def _sequence_rows(value: object, *, label: str) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeError(f"Z-space optimizer {label} must be a sequence")
    rows: list[dict[str, object]] = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"Z-space optimizer {label}[{index}] is invalid")
        rows.append(dict(row))
    return rows


def _load_run_card(value: str | Path | Mapping[str, object]) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"run card must contain a mapping: {path}")
    card = dict(payload)
    card.setdefault("_comparison_source_path", str(path))
    return card


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _receipt_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _is_sha256_id(value: object) -> bool:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
    ):
        return False
    try:
        int(value[7:], 16)
    except ValueError:
        return False
    return True


def _strongest_identity(
    card: Mapping[str, object],
    *fields: str,
) -> str | None:
    for field in fields:
        identity = _mapping(card.get(field))
        for key in (
            "observed_identity_id",
            "observed_input_id",
            "runtime_identity_id",
            "identity_id",
            "content_identity_id",
        ):
            value = identity.get(key)
            if _is_sha256_id(value):
                return value
    return None


def _base_training_recipe(card: Mapping[str, object]) -> tuple[str | None, int | None]:
    identity = _mapping(card.get("training_recipe_identity"))
    payload = _json_clone(identity.get("identity_payload"))
    if not isinstance(payload, dict):
        return None, None
    trainer_contract = payload.get("trainer_contract")
    if not isinstance(trainer_contract, dict):
        return None, None
    trainer_contract.pop("zspace_optimizer_control", None)
    arguments = payload.get("training_arguments")
    seed = arguments.get("seed") if isinstance(arguments, Mapping) else None
    if isinstance(seed, bool) or not isinstance(seed, int):
        return None, None
    return _sha256_id(payload), seed


def _eval_loss(card: Mapping[str, object], field: str) -> float | None:
    report = _mapping(card.get(field))
    candidates: list[object] = [
        report.get("loss"),
        report.get("eval_loss"),
        report.get("effective_eval_loss"),
    ]
    metrics = report.get("metrics")
    if isinstance(metrics, Mapping):
        candidates.extend((metrics.get("eval_loss"), metrics.get("loss")))
    for value in candidates:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return None


def _card_ablation_facts(card: Mapping[str, object]) -> dict[str, object]:
    receipt = _mapping(card.get("zspace_optimizer_control_receipt"))
    recipe = _mapping(card.get("zspace_optimizer_control_recipe"))
    base_recipe_id, seed = _base_training_recipe(card)
    return {
        "source_path": card.get("_comparison_source_path"),
        "mode": receipt.get("mode", recipe.get("mode")),
        "trajectory_arm": receipt.get(
            "trajectory_arm", recipe.get("trajectory_arm", "raw")
        ),
        "feedback_mode": receipt.get(
            "feedback_mode", recipe.get("feedback_mode", "off")
        ),
        "receipt": receipt,
        "base_training_recipe_id": base_recipe_id,
        "seed": seed,
        "identities": {
            "training_input": _strongest_identity(
                card,
                "training_input_identity_after_load",
                "training_input_identity",
            ),
            "dataset_materialization": _strongest_identity(
                card,
                "dataset_materialization_identity",
            ),
            "tokenized_dataset": _strongest_identity(
                card,
                "tokenized_dataset_identity",
            ),
            "model_runtime": _strongest_identity(
                card,
                "model_runtime_identity_after_model",
                "model_runtime_identity_pre_model",
            ),
            "execution": _strongest_identity(
                card,
                "finetune_execution_identity_after_model",
                "finetune_execution_identity_pre_model",
            ),
        },
        "eval_before_loss": _eval_loss(card, "eval_before_train"),
        "eval_after_loss": _eval_loss(card, "eval_after_train"),
        "training_completed": (
            card.get("failure_stage") is None and card.get("model_saved") is True
        ),
    }


def _matched_seed_pair(
    seed: int,
    observe: Mapping[str, object],
    apply: Mapping[str, object],
) -> dict[str, object]:
    errors: list[str] = []
    observe_receipt = _mapping(observe.get("receipt"))
    apply_receipt = _mapping(apply.get("receipt"))
    if observe_receipt.get("schema") != HF_ZSPACE_OPTIMIZER_RECEIPT_SCHEMA:
        errors.append("observe receipt schema is unsupported")
    if apply_receipt.get("schema") != HF_ZSPACE_OPTIMIZER_RECEIPT_SCHEMA:
        errors.append("apply receipt schema is unsupported")
    if observe_receipt.get("status") != "ready":
        errors.append("observe receipt is not ready")
    if apply_receipt.get("status") != "ready":
        errors.append("apply receipt is not ready")
    if observe_receipt.get("mode") != "observe":
        errors.append("observe arm receipt has the wrong mode")
    if apply_receipt.get("mode") != "apply":
        errors.append("apply arm receipt has the wrong mode")
    if observe.get("trajectory_arm") != "raw":
        errors.append("two-arm comparator requires a raw observe arm")
    if apply.get("trajectory_arm") != "raw":
        errors.append(
            "two-arm comparator requires raw apply; use the factorized comparator "
            "for calibrated trajectory arms"
        )
    observe_applied_count = _receipt_count(observe_receipt.get("applied_update_count"))
    if observe_applied_count != 0:
        errors.append("observe arm changed model-update learning rates")
    observe_count = _receipt_count(observe_receipt.get("observed_update_count"))
    apply_count = _receipt_count(apply_receipt.get("applied_update_count"))
    restored_count = _receipt_count(apply_receipt.get("restored_update_count"))
    observe_consumed_count = _receipt_count(
        observe_receipt.get("consumed_control_count")
    )
    apply_consumed_count = _receipt_count(apply_receipt.get("consumed_control_count"))
    if observe_count is None or observe_count <= 0:
        errors.append("observe arm consumed no control updates")
    if apply_count is None or apply_count <= 0:
        errors.append("apply arm has no proven optimizer actuation")
    if observe_consumed_count is None or observe_consumed_count != observe_count:
        errors.append("observe receipt control counts are inconsistent")
    if apply_consumed_count is None or apply_consumed_count != apply_count:
        errors.append("apply receipt control counts are inconsistent")
    if restored_count is None or apply_count != restored_count:
        errors.append("apply arm did not restore every nominal learning rate")
    if apply_receipt.get("model_update_intervened") is not True:
        errors.append("apply arm has no non-identity model-update intervention")
    observe_sequence = observe_receipt.get("control_sequence_id")
    apply_sequence = apply_receipt.get("control_sequence_id")
    if not _is_sha256_id(observe_sequence) or observe_sequence != apply_sequence:
        errors.append("observe/apply Rust control sequences differ")

    observe_identities = _mapping(observe.get("identities"))
    apply_identities = _mapping(apply.get("identities"))
    identity_matches: dict[str, bool] = {}
    for key in (
        "training_input",
        "dataset_materialization",
        "tokenized_dataset",
        "model_runtime",
        "execution",
    ):
        left = observe_identities.get(key)
        right = apply_identities.get(key)
        matched = isinstance(left, str) and left == right
        identity_matches[key] = matched
        if not matched:
            errors.append(f"{key} identity is missing or mismatched")
    base_recipe_match = isinstance(
        observe.get("base_training_recipe_id"), str
    ) and observe.get("base_training_recipe_id") == apply.get("base_training_recipe_id")
    if not base_recipe_match:
        errors.append("non-intervention training recipe differs")
    if observe.get("training_completed") is not True:
        errors.append("observe arm did not complete training")
    if apply.get("training_completed") is not True:
        errors.append("apply arm did not complete training")

    observe_before = observe.get("eval_before_loss")
    apply_before = apply.get("eval_before_loss")
    observe_after = observe.get("eval_after_loss")
    apply_after = apply.get("eval_after_loss")
    losses = (observe_before, apply_before, observe_after, apply_after)
    if any(not isinstance(value, float) for value in losses):
        errors.append("matched before/after eval losses are incomplete")
        initial_eval_delta = None
        apply_minus_observe = None
        difference_in_differences = None
    else:
        initial_eval_delta = float(apply_before) - float(observe_before)
        if not math.isclose(
            float(apply_before),
            float(observe_before),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            errors.append("before-train eval anchors differ")
        apply_minus_observe = float(apply_after) - float(observe_after)
        difference_in_differences = (float(apply_after) - float(apply_before)) - (
            float(observe_after) - float(observe_before)
        )

    match_payload = {
        "seed": seed,
        "base_training_recipe_id": observe.get("base_training_recipe_id"),
        "identities": observe_identities,
        "control_sequence_id": observe_sequence,
    }
    return {
        "seed": seed,
        "status": "ready" if not errors else "blocked",
        "matched_pair_id": _sha256_id(match_payload),
        "identity_matches": identity_matches,
        "base_training_recipe_match": base_recipe_match,
        "control_sequence_match": observe_sequence == apply_sequence,
        "observe_source_path": observe.get("source_path"),
        "apply_source_path": apply.get("source_path"),
        "observe_eval_before_loss": observe_before,
        "apply_eval_before_loss": apply_before,
        "initial_eval_loss_delta": initial_eval_delta,
        "observe_eval_after_loss": observe_after,
        "apply_eval_after_loss": apply_after,
        "apply_minus_observe_eval_after_loss": apply_minus_observe,
        "difference_in_differences": difference_in_differences,
        "lower_is_better": True,
        "apply_win": (
            None
            if difference_in_differences is None
            else difference_in_differences < 0.0
        ),
        "error_count": len(errors),
        "errors": errors,
    }


def compare_hf_zspace_optimizer_run_cards(
    run_cards: Sequence[str | Path | Mapping[str, object]],
) -> dict[str, object]:
    """Compare seed-matched observe/apply runs without overstating efficacy."""

    facts = [_card_ablation_facts(_load_run_card(card)) for card in run_cards]
    errors: list[str] = []
    grouped: dict[int, dict[str, dict[str, object]]] = {}
    for index, fact in enumerate(facts):
        seed = fact.get("seed")
        mode = fact.get("mode")
        if not isinstance(seed, int):
            errors.append(f"run card {index} has no verified training seed")
            continue
        if mode not in {"observe", "apply"}:
            errors.append(f"run card {index} is not an observe/apply arm")
            continue
        arms = grouped.setdefault(seed, {})
        if mode in arms:
            errors.append(f"seed {seed} has duplicate {mode} arms")
            continue
        arms[str(mode)] = fact

    pairs: list[dict[str, object]] = []
    for seed, arms in sorted(grouped.items()):
        if set(arms) != {"observe", "apply"}:
            errors.append(f"seed {seed} does not have one observe and one apply arm")
            continue
        pairs.append(_matched_seed_pair(seed, arms["observe"], arms["apply"]))
    blocked_pairs = [pair for pair in pairs if pair.get("status") != "ready"]
    ready_deltas = [
        float(pair["difference_in_differences"])
        for pair in pairs
        if pair.get("status") == "ready"
        and isinstance(pair.get("difference_in_differences"), float)
    ]
    status = "ready" if pairs and not errors and not blocked_pairs else "blocked"
    pair_count = len(pairs)
    bounded_trend_ready = status == "ready" and pair_count >= 3
    if not ready_deltas:
        bounded_trend_direction = None
    elif all(delta < 0.0 for delta in ready_deltas):
        bounded_trend_direction = "apply_better"
    elif all(delta > 0.0 for delta in ready_deltas):
        bounded_trend_direction = "apply_worse"
    elif all(delta == 0.0 for delta in ready_deltas):
        bounded_trend_direction = "no_observed_difference"
    else:
        bounded_trend_direction = "mixed"
    evidence_scope = (
        "multi_seed_matched_ablation"
        if bounded_trend_ready
        else "single_or_two_seed_diagnostic"
        if status == "ready"
        else "non_comparable"
    )
    return {
        "schema": HF_ZSPACE_MATCHED_ABLATION_SCHEMA,
        "status": status,
        "run_card_count": len(facts),
        "matched_pair_count": pair_count,
        "ready_pair_count": pair_count - len(blocked_pairs),
        "seeds": [pair["seed"] for pair in pairs],
        "evidence_scope": evidence_scope,
        "difference_in_differences_mean": (
            sum(ready_deltas) / len(ready_deltas) if ready_deltas else None
        ),
        "apply_win_count": sum(pair.get("apply_win") is True for pair in pairs),
        "bounded_trend_ready": bounded_trend_ready,
        "bounded_trend_direction": bounded_trend_direction,
        "efficacy_claim_ready": (
            bounded_trend_ready and bounded_trend_direction == "apply_better"
        ),
        "evidence_boundary": (
            "single/two-seed results are diagnostics; three or more directionally "
            "consistent matched seeds support a bounded trend, not statistical "
            "significance or a general superiority claim"
        ),
        "pairs": pairs,
        "error_count": len(errors) + len(blocked_pairs),
        "errors": errors,
    }


def write_hf_zspace_optimizer_matched_ablation_report(
    report: Mapping[str, object],
    path: str | Path,
) -> str:
    """Write one matched-ablation report as canonical, readable JSON."""

    output = Path(path)
    _write_json(output, report)
    return str(output)


_FACTORIZED_ARMS = (
    "observe",
    "dose_matched_constant",
    "raw",
    "dose_normalized",
)
_FACTORIZED_CONTRASTS = {
    "dose_effect": ("dose_matched_constant", "observe"),
    "shape_effect_at_raw_dose": ("raw", "dose_matched_constant"),
    "dose_normalized_shape_effect": ("dose_normalized", "observe"),
    "raw_total_effect": ("raw", "observe"),
}


def _factorized_arm_key(fact: Mapping[str, object]) -> str | None:
    mode = fact.get("mode")
    arm = fact.get("trajectory_arm")
    if mode == "observe" and arm == "raw":
        return "observe"
    if mode == "apply" and arm in HF_ZSPACE_OPTIMIZER_TRAJECTORY_ARMS:
        return str(arm)
    return None


def _factorized_seed_report(
    seed: int,
    arms: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    errors: list[str] = []
    observe = arms["observe"]
    receipts = {arm: _mapping(fact.get("receipt")) for arm, fact in arms.items()}
    for arm in _FACTORIZED_ARMS:
        receipt = receipts[arm]
        expected_mode = "observe" if arm == "observe" else "apply"
        if receipt.get("schema") != HF_ZSPACE_OPTIMIZER_RECEIPT_SCHEMA:
            errors.append(f"{arm} receipt schema is unsupported")
        if receipt.get("status") != "ready":
            errors.append(f"{arm} receipt is not ready")
        if receipt.get("mode") != expected_mode:
            errors.append(f"{arm} receipt has the wrong mode")
        if receipt.get("trajectory_arm") != ("raw" if arm == "observe" else arm):
            errors.append(f"{arm} receipt has the wrong trajectory arm")
        consumed = _receipt_count(receipt.get("consumed_control_count"))
        steps = _receipt_count(receipt.get("trajectory_step_count"))
        if consumed is None or consumed <= 0 or steps != consumed:
            errors.append(f"{arm} receipt has incomplete trajectory coverage")
        if receipt.get("trajectory_validated") is not True:
            errors.append(f"{arm} trajectory was not Rust-validated")
        if not _is_sha256_id(receipt.get("actuated_schedule_sequence_id")):
            errors.append(f"{arm} actuated schedule identity is invalid")
        if arm == "observe":
            if _receipt_count(receipt.get("observed_update_count")) != consumed:
                errors.append("observe arm did not observe every update")
            if _receipt_count(receipt.get("applied_update_count")) != 0:
                errors.append("observe arm changed optimizer learning rates")
        else:
            applied = _receipt_count(receipt.get("applied_update_count"))
            restored = _receipt_count(receipt.get("restored_update_count"))
            if applied != consumed or restored != applied:
                errors.append(f"{arm} actuation or restoration count is incomplete")

    trajectory_ids = {receipt.get("trajectory_id") for receipt in receipts.values()}
    trajectory_id = next(iter(trajectory_ids)) if len(trajectory_ids) == 1 else None
    if not _is_sha256_id(trajectory_id):
        errors.append("factorized arms do not share one valid trajectory identity")
    control_ids = {receipt.get("control_sequence_id") for receipt in receipts.values()}
    control_id = next(iter(control_ids)) if len(control_ids) == 1 else None
    if not _is_sha256_id(control_id):
        errors.append("factorized arms do not share one Rust control sequence")
    nominal_ids = {
        receipt.get("nominal_schedule_sequence_id") for receipt in receipts.values()
    }
    nominal_id = next(iter(nominal_ids)) if len(nominal_ids) == 1 else None
    if not _is_sha256_id(nominal_id):
        errors.append("factorized arms do not share one nominal LR sequence")

    trajectory_fields: dict[str, float] = {}
    for field in (
        "trajectory_identity_relative_tolerance",
        "trajectory_identity_absolute_tolerance",
        "trajectory_nominal_dose",
        "trajectory_raw_dose",
        "trajectory_dose_normalized_dose",
        "trajectory_raw_dose_ratio",
        "trajectory_dose_normalized_dose_ratio",
    ):
        values = [receipt.get(field) for receipt in receipts.values()]
        value = values[0] if all(item == values[0] for item in values[1:]) else None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            errors.append(f"factorized arms do not share a valid {field}")
        else:
            trajectory_fields[field] = float(value)

    expected_non_identity_counts: dict[str, int] = {}
    for arm in _FACTORIZED_ARMS[1:]:
        field = f"trajectory_{arm}_non_identity_update_count"
        values = [receipt.get(field) for receipt in receipts.values()]
        value = values[0] if all(item == values[0] for item in values[1:]) else None
        count = _receipt_count(value)
        step_count = _receipt_count(receipts[arm].get("trajectory_step_count"))
        if count is None or step_count is None or count > step_count:
            errors.append(f"factorized arms do not share a valid {field}")
        else:
            expected_non_identity_counts[arm] = count

    nominal_dose = trajectory_fields.get("trajectory_nominal_dose")
    raw_dose = trajectory_fields.get("trajectory_raw_dose")
    normalized_dose = trajectory_fields.get("trajectory_dose_normalized_dose")
    raw_ratio = trajectory_fields.get("trajectory_raw_dose_ratio")
    normalized_ratio = trajectory_fields.get("trajectory_dose_normalized_dose_ratio")
    if normalized_ratio is not None and not math.isclose(
        normalized_ratio, 1.0, rel_tol=1e-10, abs_tol=0.0
    ):
        errors.append("dose-normalized trajectory does not match baseline dose")
    if (
        nominal_dose is not None
        and raw_dose is not None
        and raw_ratio is not None
        and not math.isclose(
            raw_dose / nominal_dose,
            raw_ratio,
            rel_tol=1e-10,
            abs_tol=0.0,
        )
    ):
        errors.append("raw trajectory dose and ratio are inconsistent")
    if (
        nominal_dose is not None
        and normalized_dose is not None
        and normalized_ratio is not None
        and not math.isclose(
            normalized_dose / nominal_dose,
            normalized_ratio,
            rel_tol=1e-10,
            abs_tol=0.0,
        )
    ):
        errors.append("dose-normalized trajectory dose and ratio are inconsistent")

    for arm in _FACTORIZED_ARMS[1:]:
        receipt = receipts[arm]
        expected_count = expected_non_identity_counts.get(arm)
        actual_count = _receipt_count(receipt.get("non_identity_update_count"))
        if expected_count is not None:
            if actual_count != expected_count:
                errors.append(
                    f"{arm} non-identity update count differs from the Rust trajectory"
                )
            if receipt.get("model_update_intervened") is not (expected_count > 0):
                errors.append(
                    f"{arm} model-update intervention flag differs from the Rust trajectory"
                )
        if arm == "raw" and expected_count == 0:
            errors.append(f"{arm} produced no non-identity model update")

    for arm, receipt in receipts.items():
        actual_nominal = receipt.get("nominal_learning_rate_dose")
        actual_actuated = receipt.get("actuated_learning_rate_dose")
        actual_ratio = receipt.get("actuated_learning_rate_dose_ratio")
        actual_values = (actual_nominal, actual_actuated, actual_ratio)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in actual_values
        ):
            errors.append(f"{arm} receipt has invalid measured optimizer dose")
            continue
        expected_actuated = {
            "observe": nominal_dose,
            "dose_matched_constant": raw_dose,
            "raw": raw_dose,
            "dose_normalized": normalized_dose,
        }[arm]
        if nominal_dose is None or expected_actuated is None:
            continue
        if not math.isclose(
            float(actual_nominal), nominal_dose, rel_tol=1e-10, abs_tol=0.0
        ):
            errors.append(f"{arm} measured nominal dose differs from the trajectory")
        if not math.isclose(
            float(actual_actuated),
            expected_actuated,
            rel_tol=1e-10,
            abs_tol=0.0,
        ):
            errors.append(
                f"{arm} measured optimizer dose differs from its trajectory arm"
            )
        if not math.isclose(
            float(actual_ratio),
            expected_actuated / nominal_dose,
            rel_tol=1e-10,
            abs_tol=0.0,
        ):
            errors.append(f"{arm} measured optimizer dose ratio is inconsistent")

    base_recipe_ids = {fact.get("base_training_recipe_id") for fact in arms.values()}
    base_recipe_id = next(iter(base_recipe_ids)) if len(base_recipe_ids) == 1 else None
    if not _is_sha256_id(base_recipe_id):
        errors.append("non-intervention training recipes differ")

    reference_identities = _mapping(observe.get("identities"))
    identity_matches: dict[str, bool] = {}
    for key in (
        "training_input",
        "dataset_materialization",
        "tokenized_dataset",
        "model_runtime",
        "execution",
    ):
        reference = reference_identities.get(key)
        matched = _is_sha256_id(reference) and all(
            _mapping(fact.get("identities")).get(key) == reference
            for fact in arms.values()
        )
        identity_matches[key] = matched
        if not matched:
            errors.append(f"{key} identity is missing or mismatched")

    for arm, fact in arms.items():
        if fact.get("training_completed") is not True:
            errors.append(f"{arm} arm did not complete training")

    before_losses = {arm: fact.get("eval_before_loss") for arm, fact in arms.items()}
    after_losses = {arm: fact.get("eval_after_loss") for arm, fact in arms.items()}
    if any(not isinstance(value, float) for value in before_losses.values()) or any(
        not isinstance(value, float) for value in after_losses.values()
    ):
        errors.append("factorized before/after eval losses are incomplete")
        changes: dict[str, float] = {}
    else:
        anchor = float(before_losses["observe"])
        if any(
            not math.isclose(float(value), anchor, rel_tol=1e-9, abs_tol=1e-9)
            for value in before_losses.values()
        ):
            errors.append("before-train eval anchors differ")
        changes = {
            arm: float(after_losses[arm]) - float(before_losses[arm])
            for arm in _FACTORIZED_ARMS
        }
    contrasts = {
        name: (None if not changes else changes[left] - changes[right])
        for name, (left, right) in _FACTORIZED_CONTRASTS.items()
    }
    match_payload = {
        "seed": seed,
        "base_training_recipe_id": base_recipe_id,
        "identities": reference_identities,
        "trajectory_id": trajectory_id,
        "control_sequence_id": control_id,
        "nominal_schedule_sequence_id": nominal_id,
    }
    return {
        "seed": seed,
        "status": "ready" if not errors else "blocked",
        "factorized_match_id": _sha256_id(match_payload),
        "base_training_recipe_id": base_recipe_id,
        "trajectory_id": trajectory_id,
        "control_sequence_id": control_id,
        "nominal_schedule_sequence_id": nominal_id,
        "trajectory_non_identity_update_counts": expected_non_identity_counts,
        "trajectory_identity_tolerances": {
            "relative": trajectory_fields.get("trajectory_identity_relative_tolerance"),
            "absolute": trajectory_fields.get("trajectory_identity_absolute_tolerance"),
        },
        "identity_matches": identity_matches,
        "eval_before_losses": before_losses,
        "eval_after_losses": after_losses,
        "eval_loss_changes": changes,
        "contrasts": contrasts,
        "lower_is_better": True,
        "source_paths": {arm: fact.get("source_path") for arm, fact in arms.items()},
        "error_count": len(errors),
        "errors": errors,
    }


def _contrast_summary(
    reports: Sequence[Mapping[str, object]],
    name: str,
    *,
    status: str,
) -> dict[str, object]:
    values = [
        float(_mapping(report.get("contrasts"))[name])
        for report in reports
        if report.get("status") == "ready"
        and isinstance(_mapping(report.get("contrasts")).get(name), float)
    ]
    bounded = status == "ready" and len(values) >= 3
    if not values:
        direction = None
    elif all(value < 0.0 for value in values):
        direction = "left_arm_better"
    elif all(value > 0.0 for value in values):
        direction = "right_arm_better"
    elif all(value == 0.0 for value in values):
        direction = "no_observed_difference"
    else:
        direction = "mixed"
    left, right = _FACTORIZED_CONTRASTS[name]
    return {
        "left_arm": left,
        "right_arm": right,
        "lower_is_better": True,
        "values": values,
        "mean": sum(values) / len(values) if values else None,
        "left_arm_win_count": sum(value < 0.0 for value in values),
        "bounded_trend_ready": bounded,
        "bounded_trend_direction": direction,
    }


def compare_hf_zspace_optimizer_factorized_run_cards(
    run_cards: Sequence[str | Path | Mapping[str, object]],
) -> dict[str, object]:
    """Compare matched baseline, dose, raw-shape, and normalized-shape arms."""

    facts = [_card_ablation_facts(_load_run_card(card)) for card in run_cards]
    errors: list[str] = []
    grouped: dict[int, dict[str, dict[str, object]]] = {}
    for index, fact in enumerate(facts):
        seed = fact.get("seed")
        arm = _factorized_arm_key(fact)
        if not isinstance(seed, int):
            errors.append(f"run card {index} has no verified training seed")
            continue
        if arm is None:
            errors.append(f"run card {index} is not a factorized optimizer arm")
            continue
        seed_arms = grouped.setdefault(seed, {})
        if arm in seed_arms:
            errors.append(f"seed {seed} has duplicate {arm} arms")
            continue
        seed_arms[arm] = fact

    reports: list[dict[str, object]] = []
    expected = set(_FACTORIZED_ARMS)
    for seed, arms in sorted(grouped.items()):
        if set(arms) != expected:
            missing = sorted(expected - set(arms))
            errors.append(f"seed {seed} is missing factorized arms: {missing}")
            continue
        reports.append(_factorized_seed_report(seed, arms))
    blocked = [report for report in reports if report.get("status") != "ready"]
    status = "ready" if reports and not errors and not blocked else "blocked"
    report_errors = [
        f"seed {report['seed']}: {error}"
        for report in blocked
        for error in report.get("errors", [])
    ]
    all_errors = [*errors, *report_errors]
    contrast_summaries = {
        name: _contrast_summary(reports, name, status=status)
        for name in _FACTORIZED_CONTRASTS
    }
    normalized_summary = contrast_summaries["dose_normalized_shape_effect"]
    bounded_improvement_observed = (
        normalized_summary["bounded_trend_ready"] is True
        and normalized_summary["bounded_trend_direction"] == "left_arm_better"
    )
    return {
        "schema": HF_ZSPACE_FACTORIZED_ABLATION_SCHEMA,
        "status": status,
        "run_card_count": len(facts),
        "matched_seed_count": len(reports),
        "ready_seed_count": len(reports) - len(blocked),
        "seeds": [report["seed"] for report in reports],
        "evidence_scope": (
            "multi_seed_factorized_ablation"
            if status == "ready" and len(reports) >= 3
            else (
                "single_or_two_seed_factorized_diagnostic"
                if status == "ready"
                else "non_comparable"
            )
        ),
        "contrasts": contrast_summaries,
        "bounded_improvement_observed": bounded_improvement_observed,
        "efficacy_claim_ready": False,
        "evidence_boundary": (
            "the four-arm design separates integrated LR dose from schedule shape; "
            "three directionally consistent matched seeds support a bounded trend, "
            "not statistical significance or general superiority"
        ),
        "efficacy_claim_requirements": (
            "a prespecified, adequately powered multi-model evaluation with held-out "
            "quality and stability metrics is not implemented by this comparator"
        ),
        "factorized_seeds": reports,
        "error_count": len(all_errors),
        "errors": all_errors,
    }


def write_hf_zspace_optimizer_factorized_ablation_report(
    report: Mapping[str, object],
    path: str | Path,
) -> str:
    """Write one factorized optimizer-ablation report as canonical JSON."""

    output = Path(path)
    _write_json(output, report)
    return str(output)


_FEEDBACK_ABLATION_ARMS = (
    "observe",
    "raw_unguarded",
    "raw_loss_guard",
)
_FEEDBACK_ABLATION_CONTRASTS = {
    "unguarded_total_effect": ("raw_unguarded", "observe"),
    "guarded_total_effect": ("raw_loss_guard", "observe"),
    "guard_benefit": ("raw_loss_guard", "raw_unguarded"),
}


def _feedback_arm_key(fact: Mapping[str, object]) -> str | None:
    mode = fact.get("mode")
    arm = fact.get("trajectory_arm")
    feedback_mode = fact.get("feedback_mode")
    if mode == "observe" and arm == "raw" and feedback_mode == "off":
        return "observe"
    if mode == "apply" and arm == "raw" and feedback_mode == "off":
        return "raw_unguarded"
    if mode == "apply" and arm == "raw" and feedback_mode == "loss_guard":
        return "raw_loss_guard"
    return None


def _shared_positive_receipt_number(
    receipts: Mapping[str, Mapping[str, object]],
    field: str,
    errors: list[str],
) -> float | None:
    values = [receipt.get(field) for receipt in receipts.values()]
    value = values[0] if all(item == values[0] for item in values[1:]) else None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        errors.append(f"feedback arms do not share a valid {field}")
        return None
    return float(value)


def _feedback_seed_report(
    seed: int,
    arms: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    errors: list[str] = []
    receipts = {arm: _mapping(fact.get("receipt")) for arm, fact in arms.items()}

    for arm in _FEEDBACK_ABLATION_ARMS:
        receipt = receipts[arm]
        expected_mode = "observe" if arm == "observe" else "apply"
        expected_feedback = "loss_guard" if arm == "raw_loss_guard" else "off"
        if receipt.get("schema") != HF_ZSPACE_OPTIMIZER_RECEIPT_SCHEMA:
            errors.append(f"{arm} receipt schema is unsupported")
        if receipt.get("status") != "ready":
            errors.append(f"{arm} receipt is not ready")
        if receipt.get("mode") != expected_mode:
            errors.append(f"{arm} receipt has the wrong mode")
        if receipt.get("trajectory_arm") != "raw":
            errors.append(f"{arm} receipt is not a raw trajectory arm")
        if receipt.get("feedback_mode") != expected_feedback:
            errors.append(f"{arm} receipt has the wrong feedback mode")
        if receipt.get("evidence_blockers") not in (None, []):
            errors.append(f"{arm} receipt contains evidence blockers")
        if receipt.get("schedule_evidence_complete") is not True:
            errors.append(f"{arm} scheduler evidence is incomplete")
        if receipt.get("trajectory_validated") is not True:
            errors.append(f"{arm} trajectory was not Rust-validated")

        consumed = _receipt_count(receipt.get("consumed_control_count"))
        planned = _receipt_count(receipt.get("planned_update_count"))
        realized = _receipt_count(receipt.get("realized_update_count"))
        trajectory_steps = _receipt_count(receipt.get("trajectory_step_count"))
        if (
            consumed is None
            or consumed <= 0
            or planned != consumed
            or realized != consumed
            or trajectory_steps != consumed
            or receipt.get("training_horizon_complete") is not True
            or receipt.get("trajectory_horizon_complete") is not True
        ):
            errors.append(f"{arm} receipt has incomplete horizon coverage")
        if not _is_sha256_id(receipt.get("actuated_schedule_sequence_id")):
            errors.append(f"{arm} actuated schedule identity is invalid")

        if arm == "observe":
            if _receipt_count(receipt.get("observed_update_count")) != consumed:
                errors.append("observe arm did not observe every optimizer update")
            if _receipt_count(receipt.get("applied_update_count")) != 0:
                errors.append("observe arm changed optimizer learning rates")
        else:
            applied = _receipt_count(receipt.get("applied_update_count"))
            restored = _receipt_count(receipt.get("restored_update_count"))
            if applied != consumed or restored != applied:
                errors.append(f"{arm} actuation or restoration count is incomplete")
            if receipt.get("scheduler_nominal_lr_restored") is not True:
                errors.append(f"{arm} did not restore nominal scheduler rates")

    trajectory_ids = {receipt.get("trajectory_id") for receipt in receipts.values()}
    trajectory_id = next(iter(trajectory_ids)) if len(trajectory_ids) == 1 else None
    if not _is_sha256_id(trajectory_id):
        errors.append("feedback arms do not share one valid trajectory identity")
    control_ids = {receipt.get("control_sequence_id") for receipt in receipts.values()}
    control_id = next(iter(control_ids)) if len(control_ids) == 1 else None
    if not _is_sha256_id(control_id):
        errors.append("feedback arms do not share one raw Rust control sequence")
    nominal_ids = {
        receipt.get("nominal_schedule_sequence_id") for receipt in receipts.values()
    }
    nominal_id = next(iter(nominal_ids)) if len(nominal_ids) == 1 else None
    if not _is_sha256_id(nominal_id):
        errors.append("feedback arms do not share one nominal LR sequence")

    unguarded_receipt = receipts["raw_unguarded"]
    guarded_receipt = receipts["raw_loss_guard"]
    for arm in ("observe", "raw_unguarded"):
        receipt = receipts[arm]
        if _receipt_count(receipt.get("feedback_control_count")) != 0:
            errors.append(f"{arm} unexpectedly contains feedback controls")
        if _receipt_count(receipt.get("feedback_observation_count")) != 0:
            errors.append(f"{arm} unexpectedly contains feedback observations")
        if any(
            receipt.get(field) is not None
            for field in (
                "feedback_contract",
                "feedback_semantic_owner",
                "feedback_evidence_boundary",
            )
        ):
            errors.append(f"{arm} unexpectedly claims feedback provenance")

    guarded_consumed = _receipt_count(guarded_receipt.get("consumed_control_count"))
    guarded_feedback_controls = _receipt_count(
        guarded_receipt.get("feedback_control_count")
    )
    guarded_feedback_observations = _receipt_count(
        guarded_receipt.get("feedback_observation_count")
    )
    if guarded_feedback_controls != guarded_consumed:
        errors.append("guarded arm feedback control lineage is incomplete")
    if guarded_feedback_observations != guarded_consumed:
        errors.append("guarded arm must observe one training loss per optimizer update")
    if not _is_sha256_id(guarded_receipt.get("feedback_control_sequence_id")):
        errors.append("guarded feedback control sequence identity is invalid")
    if not _is_sha256_id(guarded_receipt.get("feedback_observation_sequence_id")):
        errors.append("guarded feedback observation sequence identity is invalid")
    if (
        guarded_receipt.get("feedback_contract")
        != ZSPACE_OPTIMIZER_FEEDBACK_CONTRACT_VERSION
    ):
        errors.append("guarded arm does not use the Rust feedback contract")
    if (
        guarded_receipt.get("feedback_semantic_owner")
        != ZSPACE_OPTIMIZER_FEEDBACK_SEMANTIC_OWNER
    ):
        errors.append("guarded arm has the wrong feedback semantic owner")
    if (
        guarded_receipt.get("feedback_evidence_boundary")
        != "within_run_loss_guard_not_counterfactual_efficacy"
    ):
        errors.append("guarded arm has an invalid feedback evidence boundary")

    guarded_recipe = _mapping(guarded_receipt.get("recipe"))
    feedback_config = _mapping(guarded_recipe.get("feedback_config"))
    feedback_config_id = _sha256_id(feedback_config) if feedback_config else None
    warmup_observations = _receipt_count(feedback_config.get("warmup_observations"))
    if not _is_sha256_id(feedback_config_id):
        errors.append("guarded arm has no sealed feedback config")
    if warmup_observations is None:
        errors.append("guarded feedback warmup is invalid")

    disposition_counts = _mapping(guarded_receipt.get("feedback_disposition_counts"))
    validated_dispositions: dict[str, int] = {}
    for disposition, value in disposition_counts.items():
        count = _receipt_count(value)
        if count is None:
            errors.append(f"guarded feedback disposition {disposition} is invalid")
        else:
            validated_dispositions[disposition] = count
    if (
        guarded_consumed is None
        or sum(validated_dispositions.values()) != guarded_consumed
    ):
        errors.append("guarded feedback disposition counts are incomplete")
    active_count = _receipt_count(guarded_receipt.get("feedback_active_update_count"))
    stale_count = _receipt_count(guarded_receipt.get("feedback_stale_update_count"))
    identity_count = _receipt_count(
        guarded_receipt.get("feedback_identity_update_count")
    )
    if active_count is None or active_count <= 0:
        errors.append("guarded feedback gate never became active")
    if stale_count != 0:
        errors.append("guarded feedback used stale loss observations")
    if (
        identity_count is None
        or warmup_observations is None
        or identity_count < warmup_observations + 1
    ):
        errors.append("guarded feedback does not prove fail-closed warmup updates")
    if guarded_receipt.get("model_update_intervened") is not True:
        errors.append("guarded arm never applied a non-identity model update")
    if _receipt_count(guarded_receipt.get("non_identity_update_count")) in (None, 0):
        errors.append("guarded arm has no non-identity update count")
    if unguarded_receipt.get("model_update_intervened") is not True:
        errors.append("unguarded arm never applied a non-identity model update")
    if guarded_receipt.get("actuated_schedule_sequence_id") == unguarded_receipt.get(
        "actuated_schedule_sequence_id"
    ):
        errors.append("feedback did not change the actuated optimizer schedule")

    nominal_dose = _shared_positive_receipt_number(
        receipts, "trajectory_nominal_dose", errors
    )
    raw_dose = _shared_positive_receipt_number(receipts, "trajectory_raw_dose", errors)
    measured_doses: dict[str, dict[str, float]] = {}
    for arm, receipt in receipts.items():
        values: dict[str, float] = {}
        for field in (
            "nominal_learning_rate_dose",
            "actuated_learning_rate_dose",
            "actuated_learning_rate_dose_ratio",
        ):
            value = receipt.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                errors.append(f"{arm} receipt has invalid {field}")
            else:
                values[field] = float(value)
        measured_doses[arm] = values
        if nominal_dose is not None and "nominal_learning_rate_dose" in values:
            if not math.isclose(
                values["nominal_learning_rate_dose"],
                nominal_dose,
                rel_tol=1e-10,
                abs_tol=0.0,
            ):
                errors.append(f"{arm} nominal dose differs from the trajectory")
        if {
            "nominal_learning_rate_dose",
            "actuated_learning_rate_dose",
            "actuated_learning_rate_dose_ratio",
        }.issubset(values):
            expected_ratio = (
                values["actuated_learning_rate_dose"]
                / values["nominal_learning_rate_dose"]
            )
            if not math.isclose(
                values["actuated_learning_rate_dose_ratio"],
                expected_ratio,
                rel_tol=1e-10,
                abs_tol=0.0,
            ):
                errors.append(f"{arm} measured dose ratio is inconsistent")
    if nominal_dose is not None:
        observed_dose = measured_doses["observe"].get("actuated_learning_rate_dose")
        if observed_dose is not None and not math.isclose(
            observed_dose, nominal_dose, rel_tol=1e-10, abs_tol=0.0
        ):
            errors.append("observe arm changed the nominal optimizer dose")
    if raw_dose is not None:
        unguarded_dose = measured_doses["raw_unguarded"].get(
            "actuated_learning_rate_dose"
        )
        if unguarded_dose is not None and not math.isclose(
            unguarded_dose, raw_dose, rel_tol=1e-10, abs_tol=0.0
        ):
            errors.append("unguarded arm dose differs from the raw Rust trajectory")

    base_recipe_ids = {fact.get("base_training_recipe_id") for fact in arms.values()}
    base_recipe_id = next(iter(base_recipe_ids)) if len(base_recipe_ids) == 1 else None
    if not _is_sha256_id(base_recipe_id):
        errors.append("non-intervention training recipes differ")
    reference_identities = _mapping(arms["observe"].get("identities"))
    identity_matches: dict[str, bool] = {}
    for key in (
        "training_input",
        "dataset_materialization",
        "tokenized_dataset",
        "model_runtime",
        "execution",
    ):
        reference = reference_identities.get(key)
        matched = _is_sha256_id(reference) and all(
            _mapping(fact.get("identities")).get(key) == reference
            for fact in arms.values()
        )
        identity_matches[key] = matched
        if not matched:
            errors.append(f"{key} identity is missing or mismatched")
    for arm, fact in arms.items():
        if fact.get("training_completed") is not True:
            errors.append(f"{arm} arm did not complete training")

    before_losses = {arm: fact.get("eval_before_loss") for arm, fact in arms.items()}
    after_losses = {arm: fact.get("eval_after_loss") for arm, fact in arms.items()}
    if any(not isinstance(value, float) for value in before_losses.values()) or any(
        not isinstance(value, float) for value in after_losses.values()
    ):
        errors.append("feedback before/after eval losses are incomplete")
        changes: dict[str, float] = {}
    else:
        anchor = float(before_losses["observe"])
        if any(
            not math.isclose(float(value), anchor, rel_tol=1e-9, abs_tol=1e-9)
            for value in before_losses.values()
        ):
            errors.append("before-train eval anchors differ")
        changes = {
            arm: float(after_losses[arm]) - float(before_losses[arm])
            for arm in _FEEDBACK_ABLATION_ARMS
        }
    contrasts = {
        name: (None if not changes else changes[left] - changes[right])
        for name, (left, right) in _FEEDBACK_ABLATION_CONTRASTS.items()
    }
    unguarded_effect = contrasts.get("unguarded_total_effect")
    guard_benefit = contrasts.get("guard_benefit")
    guard_recovery_fraction = (
        -float(guard_benefit) / float(unguarded_effect)
        if isinstance(unguarded_effect, float)
        and unguarded_effect > 0.0
        and isinstance(guard_benefit, float)
        else None
    )
    guarded_dose = measured_doses["raw_loss_guard"].get("actuated_learning_rate_dose")
    if (
        nominal_dose is None
        or raw_dose is None
        or guarded_dose is None
        or math.isclose(raw_dose, nominal_dose, rel_tol=1e-12, abs_tol=1e-15)
    ):
        guarded_dose_fraction = None
    else:
        guarded_dose_fraction = (guarded_dose - nominal_dose) / (
            raw_dose - nominal_dose
        )
    match_payload = {
        "seed": seed,
        "base_training_recipe_id": base_recipe_id,
        "identities": reference_identities,
        "trajectory_id": trajectory_id,
        "control_sequence_id": control_id,
        "nominal_schedule_sequence_id": nominal_id,
        "feedback_config_id": feedback_config_id,
    }
    return {
        "seed": seed,
        "status": "ready" if not errors else "blocked",
        "feedback_match_id": _sha256_id(match_payload),
        "base_training_recipe_id": base_recipe_id,
        "trajectory_id": trajectory_id,
        "control_sequence_id": control_id,
        "nominal_schedule_sequence_id": nominal_id,
        "feedback_config_id": feedback_config_id,
        "feedback_config": feedback_config,
        "identity_matches": identity_matches,
        "eval_before_losses": before_losses,
        "eval_after_losses": after_losses,
        "eval_loss_changes": changes,
        "contrasts": contrasts,
        "guard_recovery_fraction_of_unguarded_harm": guard_recovery_fraction,
        "optimizer_doses": measured_doses,
        "guarded_dose_fraction_of_raw_deviation": guarded_dose_fraction,
        "feedback_diagnostics": {
            "control_count": guarded_feedback_controls,
            "observation_count": guarded_feedback_observations,
            "active_update_count": active_count,
            "identity_update_count": identity_count,
            "stale_update_count": stale_count,
            "disposition_counts": validated_dispositions,
            "final_gate": guarded_receipt.get("feedback_gate"),
            "halted": guarded_receipt.get("feedback_halted"),
        },
        "lower_is_better": True,
        "source_paths": {arm: fact.get("source_path") for arm, fact in arms.items()},
        "error_count": len(errors),
        "errors": errors,
    }


def _descriptive_summary(values: Sequence[float]) -> dict[str, object]:
    count = len(values)
    if count == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "population_standard_deviation": None,
            "sample_standard_deviation": None,
        }
    mean = sum(values) / count
    ordered = sorted(values)
    midpoint = count // 2
    median = (
        ordered[midpoint]
        if count % 2 == 1
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    )
    squared_residuals = sum((value - mean) ** 2 for value in values)
    return {
        "count": count,
        "mean": mean,
        "median": median,
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "population_standard_deviation": math.sqrt(squared_residuals / count),
        "sample_standard_deviation": (
            math.sqrt(squared_residuals / (count - 1)) if count > 1 else None
        ),
    }


def _feedback_contrast_summary(
    reports: Sequence[Mapping[str, object]],
    name: str,
    *,
    status: str,
) -> dict[str, object]:
    values = [
        float(_mapping(report.get("contrasts"))[name])
        for report in reports
        if report.get("status") == "ready"
        and isinstance(_mapping(report.get("contrasts")).get(name), float)
    ]
    if not values:
        direction = None
    elif all(value < 0.0 for value in values):
        direction = "left_arm_better"
    elif all(value > 0.0 for value in values):
        direction = "right_arm_better"
    elif all(value == 0.0 for value in values):
        direction = "no_observed_difference"
    else:
        direction = "mixed"
    left, right = _FEEDBACK_ABLATION_CONTRASTS[name]
    return {
        "left_arm": left,
        "right_arm": right,
        "lower_is_better": True,
        "values": values,
        **_descriptive_summary(values),
        "left_arm_win_count": sum(value < 0.0 for value in values),
        "bounded_trend_ready": status == "ready" and len(values) >= 3,
        "bounded_trend_direction": direction,
    }


def _feedback_harm_recovery_summary(
    reports: Sequence[Mapping[str, object]],
    *,
    status: str,
) -> dict[str, object]:
    ready_reports = [report for report in reports if report.get("status") == "ready"]
    values = [
        float(report["guard_recovery_fraction_of_unguarded_harm"])
        for report in ready_reports
        if isinstance(report.get("guard_recovery_fraction_of_unguarded_harm"), float)
    ]
    applicable = bool(ready_reports) and len(values) == len(ready_reports)
    unguarded_effects = [
        float(_mapping(report.get("contrasts"))["unguarded_total_effect"])
        for report in ready_reports
        if isinstance(
            _mapping(report.get("contrasts")).get("unguarded_total_effect"),
            float,
        )
    ]
    guarded_effects = [
        float(_mapping(report.get("contrasts"))["guarded_total_effect"])
        for report in ready_reports
        if isinstance(
            _mapping(report.get("contrasts")).get("guarded_total_effect"),
            float,
        )
    ]
    if (
        applicable
        and len(unguarded_effects) == len(ready_reports)
        and len(guarded_effects) == len(ready_reports)
    ):
        mean_unguarded = sum(unguarded_effects) / len(unguarded_effects)
        mean_guarded = sum(guarded_effects) / len(guarded_effects)
        fraction_of_mean = (mean_unguarded - mean_guarded) / mean_unguarded
    else:
        mean_unguarded = None
        mean_guarded = None
        fraction_of_mean = None
    return {
        "applicable": applicable,
        "applicability_rule": (
            "defined only when every ready seed has positive unguarded_total_effect"
        ),
        "values": values if applicable else [],
        **_descriptive_summary(values if applicable else []),
        "mean_unguarded_total_effect": mean_unguarded,
        "mean_guarded_total_effect": mean_guarded,
        "fraction_of_mean_unguarded_harm": fraction_of_mean,
        "all_seeds_reduced_harm": applicable and all(value > 0.0 for value in values),
        "bounded_trend_ready": status == "ready" and applicable and len(values) >= 3,
    }


def compare_hf_zspace_optimizer_feedback_run_cards(
    run_cards: Sequence[str | Path | Mapping[str, object]],
) -> dict[str, object]:
    """Compare one baseline and matched unguarded/guarded raw-control arms."""

    facts = [_card_ablation_facts(_load_run_card(card)) for card in run_cards]
    errors: list[str] = []
    grouped: dict[int, dict[str, dict[str, object]]] = {}
    for index, fact in enumerate(facts):
        seed = fact.get("seed")
        arm = _feedback_arm_key(fact)
        if not isinstance(seed, int):
            errors.append(f"run card {index} has no verified training seed")
            continue
        if arm is None:
            errors.append(f"run card {index} is not a feedback-ablation arm")
            continue
        seed_arms = grouped.setdefault(seed, {})
        if arm in seed_arms:
            errors.append(f"seed {seed} has duplicate {arm} arms")
            continue
        seed_arms[arm] = fact

    reports: list[dict[str, object]] = []
    expected = set(_FEEDBACK_ABLATION_ARMS)
    for seed, arms in sorted(grouped.items()):
        if set(arms) != expected:
            missing = sorted(expected - set(arms))
            errors.append(f"seed {seed} is missing feedback arms: {missing}")
            continue
        reports.append(_feedback_seed_report(seed, arms))
    ready_reports = [report for report in reports if report.get("status") == "ready"]
    blocked = [report for report in reports if report.get("status") != "ready"]
    config_ids = {report.get("feedback_config_id") for report in ready_reports}
    if len(config_ids) > 1:
        errors.append("matched seeds do not share one feedback config")
    status = "ready" if reports and not errors and not blocked else "blocked"
    report_errors = [
        f"seed {report['seed']}: {error}"
        for report in blocked
        for error in report.get("errors", [])
    ]
    all_errors = [*errors, *report_errors]
    contrast_summaries = {
        name: _feedback_contrast_summary(reports, name, status=status)
        for name in _FEEDBACK_ABLATION_CONTRASTS
    }
    guard_summary = contrast_summaries["guard_benefit"]
    absolute_summary = contrast_summaries["guarded_total_effect"]
    harm_recovery = _feedback_harm_recovery_summary(reports, status=status)
    return {
        "schema": HF_ZSPACE_FEEDBACK_ABLATION_SCHEMA,
        "status": status,
        "run_card_count": len(facts),
        "matched_seed_count": len(reports),
        "ready_seed_count": len(ready_reports),
        "seeds": [report["seed"] for report in reports],
        "feedback_config_id": (
            next(iter(config_ids)) if len(config_ids) == 1 else None
        ),
        "evidence_scope": (
            "single_model_single_corpus_multi_seed_feedback_ablation"
            if status == "ready" and len(reports) >= 3
            else (
                "single_or_two_seed_feedback_diagnostic"
                if status == "ready"
                else "non_comparable"
            )
        ),
        "contrasts": contrast_summaries,
        "guard_recovery_fraction_of_unguarded_harm": harm_recovery,
        "bounded_guard_benefit_observed": (
            guard_summary["bounded_trend_ready"] is True
            and guard_summary["bounded_trend_direction"] == "left_arm_better"
        ),
        "bounded_absolute_improvement_observed": (
            absolute_summary["bounded_trend_ready"] is True
            and absolute_summary["bounded_trend_direction"] == "left_arm_better"
        ),
        "efficacy_claim_ready": False,
        "evidence_boundary": (
            "the three-arm design compares one frozen Rust raw-control trajectory "
            "with and without within-run loss feedback against an observe baseline; "
            "three directionally consistent matched seeds support a bounded trend, "
            "not statistical significance or general superiority"
        ),
        "efficacy_claim_requirements": (
            "a prespecified, adequately powered multi-model and multi-corpus "
            "evaluation with held-out quality and stability metrics is not "
            "implemented by this comparator"
        ),
        "feedback_seeds": reports,
        "error_count": len(all_errors),
        "errors": all_errors,
    }


def write_hf_zspace_optimizer_feedback_ablation_report(
    report: Mapping[str, object],
    path: str | Path,
) -> str:
    """Write one guarded-vs-unguarded optimizer report as canonical JSON."""

    output = Path(path)
    _write_json(output, report)
    return str(output)
