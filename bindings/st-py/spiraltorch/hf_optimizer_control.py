"""Rust-owned Z-space parameter control for external HF optimizers."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HF_ZSPACE_OPTIMIZER_CONTROL_SCHEMA = "spiraltorch.hf_zspace_optimizer_control.v1"
HF_ZSPACE_OPTIMIZER_RECEIPT_SCHEMA = "spiraltorch.hf_zspace_optimizer_receipt.v1"
HF_ZSPACE_OPTIMIZER_STATE_SCHEMA = "spiraltorch.hf_zspace_optimizer_state.v1"
HF_ZSPACE_OPTIMIZER_TRACE_SCHEMA = "spiraltorch.hf_zspace_optimizer_trace.v1"
HF_ZSPACE_MATCHED_ABLATION_SCHEMA = "spiraltorch.hf_zspace_matched_ablation.v1"
HF_ZSPACE_OPTIMIZER_STATE_FILENAME = "spiraltorch-hf-zspace-optimizer-state.json"
HF_ZSPACE_OPTIMIZER_TRACE_FILENAME = "spiraltorch-hf-zspace-optimizer-trace.jsonl"
HF_ZSPACE_OPTIMIZER_MODES = ("off", "observe", "apply")

__all__ = [
    "HF_ZSPACE_OPTIMIZER_CONTROL_SCHEMA",
    "HF_ZSPACE_MATCHED_ABLATION_SCHEMA",
    "HF_ZSPACE_OPTIMIZER_MODES",
    "HF_ZSPACE_OPTIMIZER_RECEIPT_SCHEMA",
    "HF_ZSPACE_OPTIMIZER_STATE_FILENAME",
    "HF_ZSPACE_OPTIMIZER_STATE_SCHEMA",
    "HF_ZSPACE_OPTIMIZER_TRACE_FILENAME",
    "HF_ZSPACE_OPTIMIZER_TRACE_SCHEMA",
    "hf_zspace_optimizer_control_callback",
    "hf_zspace_optimizer_recipe_contract",
    "compare_hf_zspace_optimizer_run_cards",
    "write_hf_zspace_optimizer_matched_ablation_report",
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
    return {
        "schema": HF_ZSPACE_OPTIMIZER_CONTROL_SCHEMA,
        "mode": resolved_mode,
        "model_update_intervention": resolved_mode == "apply",
        "control_observation": resolved_mode in {"observe", "apply"},
        "z_dim": resolved_z_dim,
        "curvature": resolved_curvature,
        "control_gain": resolved_gain,
        "volume_per_step": resolved_volume_per_step,
        "control_signal": "topos_progress_geometry",
        "feedback_adaptive": False,
        "observation_mapping": "hf_global_step_to_open_topos_depth_and_volume",
        "control_pipeline": [
            "st-tensor::pure::topos",
            "st-core::runtime::zspace_optimizer",
            "torch.optim.Optimizer.step",
        ],
        "parameter_control_contract": "spiraltorch.zspace_parameter_control.v2",
        "parameter_control_semantic_owner": "st-core::runtime::zspace_optimizer",
        "parameter_control_semantic_backend": "rust",
        "actuation_transport": "temporary_param_group_lr_scale",
        "scheduler_isolation": "restore_nominal_lr_before_scheduler_step",
        "first_control_source_step": 0,
        "control_target": "next_optimizer_update",
        "resume_state_required": resolved_mode in {"observe", "apply"},
        "fail_closed": True,
    }


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


def hf_zspace_optimizer_control_callback(
    *,
    mode: str = "off",
    z_dim: int = 6,
    curvature: float = -0.04,
    control_gain: float = 1.0,
    volume_per_step: int = 8,
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
    recipe = hf_zspace_optimizer_recipe_contract(
        mode=mode,
        z_dim=z_dim,
        curvature=curvature,
        control_gain=control_gain,
        volume_per_step=volume_per_step,
    )
    resolved_mode = str(recipe["mode"])
    resolved_trace = None if trace_path is None else Path(trace_path)
    resolved_resume = (
        None if resume_from_checkpoint is None else Path(resume_from_checkpoint)
    )

    class SpiralTorchHFZSpaceOptimizerCallback(base_cls):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            self.recipe = dict(recipe)
            self.mode = resolved_mode
            self.trace_path = resolved_trace
            self.optimizer: object | None = None
            self.zspace_trainer: object | None = None
            self.pending: dict[str, object] | None = None
            self.active_nominal_rates: list[float] | None = None
            self.active_effective_rates: list[float] | None = None
            self.active_target_step: int | None = None
            self.expected_target_step: int | None = None
            self.last_consumed_target_step: int | None = None
            self.max_steps: int | None = None
            self.derived_sequence: list[dict[str, object]] = []
            self.consumed_sequence: list[dict[str, object]] = []
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
            if payload.get("schema") != HF_ZSPACE_OPTIMIZER_STATE_SCHEMA:
                raise RuntimeError("unsupported Z-space optimizer resume-state schema")
            if payload.get("recipe") != self.recipe:
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
            self.derived_sequence = _sequence_rows(
                payload.get("derived_sequence"),
                label="derived_sequence",
            )
            self.consumed_sequence = _sequence_rows(
                payload.get("consumed_sequence"),
                label="consumed_sequence",
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

        def _consume_pending(
            self, *, nominal_rates: Sequence[float]
        ) -> dict[str, object]:
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
            effective_rates = [float(rate) * scale for rate in nominal_rates]
            return {
                "hf_source_step": pending["hf_source_step"],
                "hf_target_step": target_step,
                "zspace_source_step": control["source_step"],
                "absolute_learning_rate_scale": scale,
                "parameter_control_contract": control["contract_version"],
                "parameter_control_semantic_owner": control["semantic_owner"],
                "nominal_learning_rates": list(nominal_rates),
                "effective_learning_rates": effective_rates,
            }

        def _pre_step_hook(self, optimizer, args, kwargs):  # type: ignore[no-untyped-def]
            if self.active_nominal_rates is not None:
                raise RuntimeError("nested optimizer.step is unsupported")
            nominal_rates = _optimizer_learning_rates(optimizer)
            consumed = self._consume_pending(nominal_rates=nominal_rates)
            effective_rates = list(consumed["effective_learning_rates"])
            _set_optimizer_learning_rates(optimizer, effective_rates)
            self.active_nominal_rates = nominal_rates
            self.active_effective_rates = effective_rates
            self.active_target_step = int(consumed["hf_target_step"])
            self._trace("optimizer_step_pre", **consumed)
            return None

        def _post_step_hook(self, optimizer, args, kwargs):  # type: ignore[no-untyped-def]
            if self.active_nominal_rates is None or self.active_target_step is None:
                raise RuntimeError("optimizer.step post-hook has no active control")
            nominal_rates = list(self.active_nominal_rates)
            _set_optimizer_learning_rates(optimizer, nominal_rates)
            consumed = self._consume_pending(nominal_rates=nominal_rates)
            target_step = self.active_target_step
            self.consumed_sequence.append(
                {
                    key: consumed[key]
                    for key in (
                        "hf_source_step",
                        "hf_target_step",
                        "zspace_source_step",
                        "absolute_learning_rate_scale",
                        "parameter_control_contract",
                        "parameter_control_semantic_owner",
                    )
                }
            )
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
            self.restored_update_count += 1
            self.last_consumed_target_step = target_step
            self.pending = None
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
            if self.resume_state_loaded:
                if self.resume_hf_step != global_step:
                    raise RuntimeError(
                        "Z-space optimizer resume step does not match Trainer"
                    )
            elif global_step != 0:
                raise RuntimeError(
                    "nonzero Trainer state requires a Z-space optimizer resume state"
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
                optimizer = kwargs.get("optimizer", self.optimizer)
                hookable = _hookable_optimizer(optimizer)
                nominal_rates = _optimizer_learning_rates(hookable)
                consumed = self._consume_pending(nominal_rates=nominal_rates)
                self.consumed_sequence.append(
                    {
                        key: consumed[key]
                        for key in (
                            "hf_source_step",
                            "hf_target_step",
                            "zspace_source_step",
                            "absolute_learning_rate_scale",
                            "parameter_control_contract",
                            "parameter_control_semantic_owner",
                        )
                    }
                )
                self.observed_update_count += 1
                self.last_consumed_target_step = int(consumed["hf_target_step"])
                self.pending = None
                self._trace("optimizer_step_observed", **consumed)
            return control

        def on_step_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            if self.mode == "off":
                return control
            global_step = _safe_step(state.global_step, label="global_step")
            if self.last_consumed_target_step != global_step:
                raise RuntimeError(
                    "Z-space control was not consumed by the completed optimizer update"
                )
            self.expected_target_step = None
            self._derive(global_step)
            return control

        def _state_payload(self, hf_global_step: int) -> dict[str, object]:
            if self.zspace_trainer is None or self.max_steps is None:
                raise RuntimeError("Z-space optimizer state is unavailable")
            pending = self.pending or {}
            trace_sha256, trace_size_bytes = _trace_file_evidence(self.trace_path)
            payload: dict[str, object] = {
                "schema": HF_ZSPACE_OPTIMIZER_STATE_SCHEMA,
                "recipe": dict(self.recipe),
                "hf_global_step": hf_global_step,
                "max_steps": self.max_steps,
                "zspace_trainer_state": self.zspace_trainer.state_dict(),
                "pending_hf_source_step": pending.get("hf_source_step"),
                "pending_hf_target_step": pending.get("hf_target_step"),
                "pending_report": pending.get("report"),
                "derived_sequence": _json_clone(self.derived_sequence),
                "consumed_sequence": _json_clone(self.consumed_sequence),
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
                if self.mode == "apply" and self.applied_update_count == 0:
                    raise RuntimeError(
                        "Z-space apply mode completed without a parameter update"
                    )
                if self.mode == "apply" and (
                    self.applied_update_count != self.restored_update_count
                ):
                    raise RuntimeError(
                        "Z-space optimizer learning rates were not restored"
                    )
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
            scales = [
                float(row["absolute_learning_rate_scale"])
                for row in self.consumed_sequence
            ]
            if self.failure is not None:
                status = "blocked"
            elif self.mode == "off":
                status = "disabled"
            elif not self.started:
                status = "pending"
            elif self.mode == "apply" and self.applied_update_count == 0:
                status = "blocked"
            elif self.mode == "apply" and (
                self.applied_update_count != self.restored_update_count
            ):
                status = "blocked"
            elif self.finished:
                status = "ready"
            else:
                status = "active"
            trace_sha256, trace_size_bytes = _trace_file_evidence(self.trace_path)
            return {
                "schema": HF_ZSPACE_OPTIMIZER_RECEIPT_SCHEMA,
                "status": status,
                "mode": self.mode,
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
                "scale_min": min(scales) if scales else None,
                "scale_max": max(scales) if scales else None,
                "scale_mean": sum(scales) / len(scales) if scales else None,
                "control_sequence_id": _sha256_id(self.consumed_sequence),
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
                "evidence_boundary": (
                    "receipt proves control derivation and optimizer actuation, "
                    "not learning-quality improvement; this v1 signal is derived "
                    "from training progress, not gradients or loss feedback"
                ),
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
