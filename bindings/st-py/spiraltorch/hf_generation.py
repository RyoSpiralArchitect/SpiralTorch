"""Hugging Face generation helpers backed by SpiralTorch Z-Space controls."""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "ZSpaceActivationProbeHook",
    "ZSpaceRepressionLogitsProcessor",
    "build_zspace_activation_probe_hook",
    "build_zspace_repression_logits_processor",
    "build_zspace_softmax_logits_processor",
    "compare_zspace_inference_distortion_probes",
    "zspace_inference_distortion_sweep_report_from_probes",
    "zspace_inference_distortion_probe_cli_args",
    "zspace_inference_distortion_sweep_cli_args",
    "zspace_generation_control_bridge_cli_args",
    "zspace_generation_control_processor_kwargs",
    "zspace_generation_control_sweep_cli_args",
    "load_zspace_inference_distortion_probe",
    "load_zspace_inference_distortion_sweep",
    "load_zspace_generation_control_sweep",
    "summarize_zspace_inference_distortion_probe",
    "summarize_zspace_inference_distortion_probe_lines",
    "summarize_zspace_inference_distortion_probe_comparison_lines",
    "summarize_zspace_inference_distortion_sweep",
    "summarize_zspace_inference_distortion_sweep_lines",
    "summarize_zspace_generation_control_run",
    "summarize_zspace_generation_control_sweep",
    "summarize_zspace_generation_control_sweep_lines",
    "zspace_inference_distortion_processor_kwargs",
]


ADJUST_MIN = 0.25
ADJUST_MAX = 4.0
EPSILON = 1.0e-12


def _finite_float(value: object, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _optional_finite_float(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, label=label)


def _positive_float(value: object, *, label: str) -> float:
    result = _finite_float(value, label=label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _non_negative_float(value: object, *, label: str) -> float:
    result = _finite_float(value, label=label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _positive_int(value: object, *, label: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _non_negative_int(value: object, *, label: str) -> int:
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _unit_interval_float(value: object, *, label: str) -> float:
    result = _finite_float(value, label=label)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{label} must be in [0.0, 1.0]")
    return result


def _safe_number(value: object) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return value
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _mapping_item(row: Mapping[str, object], key: str) -> dict[str, object]:
    value = row.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _row_list(value: Any) -> list[float]:
    data = value.tolist() if hasattr(value, "tolist") else value
    if not data:
        return []
    first = data[0]
    if isinstance(first, list):
        return [float(item) for item in first]
    return [float(item) for item in data]


def _tensor_row_to_ints(value: Any) -> list[int]:
    data = value.detach().cpu().tolist() if hasattr(value, "detach") else value
    if data and isinstance(data[0], list):
        data = data[0]
    return [int(item) for item in data]


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(p * math.log(max(p, EPSILON)) for p in probabilities if p > 0.0)


def _softmax(values: Sequence[float], *, scale: float) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp((value - max_value) * scale) for value in values]
    total = max(sum(exps), EPSILON)
    return [value / total for value in exps]


def _math_zspace_softmax(
    values: Sequence[float],
    *,
    curvature: float,
    temperature: float,
    entropy_target: float | None,
    entropy_tolerance: float,
    entropy_gain: float,
    min_temperature: float,
    max_temperature: float,
) -> tuple[list[float], dict[str, object]]:
    base_temperature = max(min_temperature, min(max_temperature, temperature))
    scale = math.sqrt(-curvature) / base_temperature
    base_probabilities = _softmax(values, scale=scale)
    entropy = _entropy(base_probabilities)
    effective_temperature = base_temperature
    adaptive = False
    if entropy_target is not None:
        delta = entropy_target - entropy
        if abs(delta) > entropy_tolerance:
            adjust = max(ADJUST_MIN, min(ADJUST_MAX, 1.0 + entropy_gain * delta))
            effective_temperature = max(
                min_temperature,
                min(max_temperature, base_temperature * adjust),
            )
            adaptive = True
    probabilities = _softmax(values, scale=math.sqrt(-curvature) / effective_temperature)
    return probabilities, {
        "backend": "math_zspace_softmax",
        "entropy": _entropy(probabilities),
        "temperature": effective_temperature,
        "adaptive_temperature": adaptive,
    }


class ZSpaceRepressionLogitsProcessor:
    """Transformers-compatible logits processor for Z-Space generation control.

    The processor first subtracts a dynamic repetition/repression field from the
    selected top-k logits, then converts those logits through SpiralTorch's
    ZSpaceSoftmax when the native wheel is available. Returning log
    probabilities lets greedy decoding observe repression-driven rank changes.
    """

    def __init__(
        self,
        *,
        top_k: int = 64,
        curvature: float = -0.04,
        temperature: float = 1.0,
        entropy_target: float | None = None,
        entropy_tolerance: float = 1.0e-4,
        entropy_gain: float = 0.5,
        min_temperature: float | None = None,
        max_temperature: float | None = None,
        repression_window: int = 32,
        repression_strength: float = 1.0,
        last_token_repression: float = 0.5,
        ngram_size: int = 0,
        ngram_window: int = 0,
        ngram_repression_strength: float = 0.0,
        ngram_decay: float = 1.0,
        mask_non_top_k: bool = True,
        use_native_zspace: bool = True,
    ) -> None:
        self.top_k = _positive_int(top_k, label="top_k")
        self.curvature = _finite_float(curvature, label="curvature")
        if self.curvature >= 0.0:
            raise ValueError("curvature must be negative")
        self.temperature = _positive_float(temperature, label="temperature")
        self.entropy_target = _optional_finite_float(
            entropy_target,
            label="entropy_target",
        )
        self.entropy_tolerance = _non_negative_float(
            entropy_tolerance,
            label="entropy_tolerance",
        )
        self.entropy_gain = _non_negative_float(entropy_gain, label="entropy_gain")
        default_min = max(self.temperature * 0.1, 1.0e-3)
        default_max = self.temperature * 10.0
        self.min_temperature = _positive_float(
            default_min if min_temperature is None else min_temperature,
            label="min_temperature",
        )
        self.max_temperature = _positive_float(
            default_max if max_temperature is None else max_temperature,
            label="max_temperature",
        )
        if self.min_temperature > self.max_temperature:
            raise ValueError("min_temperature must be <= max_temperature")
        self.repression_window = _non_negative_int(
            repression_window,
            label="repression_window",
        )
        self.repression_strength = _non_negative_float(
            repression_strength,
            label="repression_strength",
        )
        self.last_token_repression = _non_negative_float(
            last_token_repression,
            label="last_token_repression",
        )
        self.ngram_size = _non_negative_int(ngram_size, label="ngram_size")
        self.ngram_window = _non_negative_int(ngram_window, label="ngram_window")
        self.ngram_repression_strength = _non_negative_float(
            ngram_repression_strength,
            label="ngram_repression_strength",
        )
        self.ngram_decay = _unit_interval_float(ngram_decay, label="ngram_decay")
        self.mask_non_top_k = bool(mask_non_top_k)
        self.use_native_zspace = bool(use_native_zspace)
        self._native_layer: Any | None = None
        self._native_error: str | None = None
        self._reports: list[dict[str, object]] = []

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        torch = importlib.import_module("torch")
        if scores is None or len(getattr(scores, "shape", ())) != 2:
            return scores
        batch_size = int(scores.shape[0])
        vocab_size = int(scores.shape[-1])
        if vocab_size <= 0:
            return scores
        k = min(self.top_k, vocab_size)
        top_values, top_indices = torch.topk(scores, k=k, dim=-1)
        processed = (
            torch.full_like(scores, float("-inf"))
            if self.mask_non_top_k
            else scores.clone()
        )
        rows: list[dict[str, object]] = []
        for row in range(batch_size):
            row_values = [float(item) for item in top_values[row].detach().cpu().tolist()]
            row_indices = [int(item) for item in top_indices[row].detach().cpu().tolist()]
            input_row = input_ids[row] if input_ids is not None else []
            token_counts, last_token, recent_tokens = self._recent_repression_state(
                input_row
            )
            adjusted_values, repression = self._apply_repression(
                row_values,
                row_indices,
                token_counts,
                last_token,
                recent_tokens,
            )
            probabilities, zspace_report = self._zspace_probabilities(adjusted_values)
            log_probabilities = [
                math.log(max(float(probability), EPSILON))
                for probability in probabilities
            ]
            update = torch.tensor(
                log_probabilities,
                dtype=scores.dtype,
                device=scores.device,
            )
            processed[row].scatter_(0, top_indices[row], update)
            before_pos = max(range(len(row_values)), key=row_values.__getitem__)
            after_pos = max(
                range(len(log_probabilities)),
                key=log_probabilities.__getitem__,
            )
            rows.append(
                {
                    "row": row,
                    "top_k": k,
                    "before_top_token": row_indices[before_pos],
                    "after_top_token": row_indices[after_pos],
                    "top_token_changed": row_indices[before_pos]
                    != row_indices[after_pos],
                    "before_top_logit": row_values[before_pos],
                    "after_top_log_probability": log_probabilities[after_pos],
                    "repressed_token_count": repression["repressed_token_count"],
                    "max_repression": repression["max_repression"],
                    "ngram_repressed_token_count": repression[
                        "ngram_repressed_token_count"
                    ],
                    "max_ngram_repression": repression["max_ngram_repression"],
                    "entropy": zspace_report.get("entropy"),
                    "temperature": zspace_report.get("temperature"),
                    "adaptive_temperature": zspace_report.get("adaptive_temperature"),
                    "backend": zspace_report.get("backend"),
                    "native_error": zspace_report.get("native_error"),
                }
            )
        self._reports.append(
            {
                "row_type": "zspace_repression_logits_processor_report",
                "status": "ok",
                "batch_size": batch_size,
                "vocab_size": vocab_size,
                "top_k": k,
                "curvature": self.curvature,
                "temperature": self.temperature,
                "entropy_target": self.entropy_target,
                "entropy_gain": self.entropy_gain,
                "min_temperature": self.min_temperature,
                "max_temperature": self.max_temperature,
                "repression_window": self.repression_window,
                "repression_strength": self.repression_strength,
                "last_token_repression": self.last_token_repression,
                "ngram_size": self.ngram_size,
                "ngram_window": self.ngram_window,
                "ngram_repression_strength": self.ngram_repression_strength,
                "ngram_decay": self.ngram_decay,
                "mask_non_top_k": self.mask_non_top_k,
                "rows": rows,
            }
        )
        return processed

    def report(self, *, limit: int | None = None) -> dict[str, object]:
        all_reports = list(self._reports)
        if limit is None:
            row_reports = all_reports
        elif limit <= 0:
            row_reports = []
        else:
            row_reports = all_reports[-limit:]
        rows = [
            row
            for report in row_reports
            for row in report.get("rows", [])
            if isinstance(row, Mapping)
        ]
        aggregate_rows = [
            row
            for report in all_reports
            for row in report.get("rows", [])
            if isinstance(row, Mapping)
        ]
        backends = [
            str(row["backend"])
            for row in aggregate_rows
            if row.get("backend")
        ]
        changed = sum(1 for row in rows if row.get("top_token_changed") is True)
        aggregate_changed = sum(
            1 for row in aggregate_rows if row.get("top_token_changed") is True
        )
        temperatures = [
            float(row["temperature"])
            for row in aggregate_rows
            if isinstance(row.get("temperature"), (int, float))
        ]
        entropies = [
            float(row["entropy"])
            for row in aggregate_rows
            if isinstance(row.get("entropy"), (int, float))
        ]
        ngram_repressed = [
            int(row["ngram_repressed_token_count"])
            for row in aggregate_rows
            if isinstance(row.get("ngram_repressed_token_count"), (int, float))
        ]
        ngram_repressions = [
            float(row["max_ngram_repression"])
            for row in aggregate_rows
            if isinstance(row.get("max_ngram_repression"), (int, float))
        ]
        return {
            "row_type": "zspace_repression_generation_control",
            "status": "ok" if all_reports else "unused",
            "processor": "ZSpaceRepressionLogitsProcessor",
            "calls": len(all_reports),
            "reported_rows": len(rows),
            "backend": backends[0] if backends else None,
            "rows": rows,
            "top_token_changed_count": aggregate_changed,
            "reported_top_token_changed_count": changed,
            "temperature_min": min(temperatures) if temperatures else None,
            "temperature_max": max(temperatures) if temperatures else None,
            "entropy_min": min(entropies) if entropies else None,
            "entropy_max": max(entropies) if entropies else None,
            "ngram_repressed_token_total": (
                sum(ngram_repressed) if ngram_repressed else 0
            ),
            "max_ngram_repression": (
                max(ngram_repressions) if ngram_repressions else 0.0
            ),
            "native_error": self._native_error,
        }

    def reset_report(self) -> None:
        self._reports.clear()

    def _recent_repression_state(
        self,
        input_row: Any,
    ) -> tuple[dict[int, int], int | None, list[int]]:
        tokens = _tensor_row_to_ints(input_row)
        recent = tokens[-self.repression_window :] if self.repression_window > 0 else []
        counts: dict[int, int] = {}
        for token in recent:
            counts[token] = counts.get(token, 0) + 1
        ngram_window = self.ngram_window or self.repression_window
        ngram_recent = tokens[-ngram_window:] if ngram_window > 0 else []
        return counts, (recent[-1] if recent else None), ngram_recent

    def _apply_repression(
        self,
        values: Sequence[float],
        indices: Sequence[int],
        counts: Mapping[int, int],
        last_token: int | None,
        recent_tokens: Sequence[int],
    ) -> tuple[list[float], dict[str, object]]:
        if not values:
            return [], {
                "repressed_token_count": 0,
                "max_repression": 0.0,
                "ngram_repressed_token_count": 0,
                "max_ngram_repression": 0.0,
            }
        adjusted: list[float] = []
        penalties: list[float] = []
        ngram_penalties: list[float] = []
        for value, token in zip(values, indices):
            count = counts.get(int(token), 0)
            penalty = self.repression_strength * float(count)
            if last_token is not None and int(token) == int(last_token):
                penalty += self.last_token_repression
            ngram_penalty = self._ngram_repression_penalty(
                recent_tokens,
                int(token),
            )
            penalty += ngram_penalty
            adjusted.append(float(value) - penalty)
            penalties.append(penalty)
            ngram_penalties.append(ngram_penalty)
        return adjusted, {
            "repressed_token_count": sum(1 for penalty in penalties if penalty > 0.0),
            "max_repression": max(penalties) if penalties else 0.0,
            "ngram_repressed_token_count": sum(
                1 for penalty in ngram_penalties if penalty > 0.0
            ),
            "max_ngram_repression": (
                max(ngram_penalties) if ngram_penalties else 0.0
            ),
        }

    def _ngram_repression_penalty(
        self,
        recent_tokens: Sequence[int],
        candidate_token: int,
    ) -> float:
        if (
            self.ngram_size <= 1
            or self.ngram_repression_strength <= 0.0
            or len(recent_tokens) < self.ngram_size
        ):
            return 0.0
        prefix_size = self.ngram_size - 1
        prefix = tuple(int(token) for token in recent_tokens[-prefix_size:])
        candidate_ngram = (*prefix, int(candidate_token))
        weighted_matches = 0.0
        latest_start = len(recent_tokens) - self.ngram_size
        for start in range(0, latest_start + 1):
            ngram = tuple(
                int(token)
                for token in recent_tokens[start : start + self.ngram_size]
            )
            if ngram != candidate_ngram:
                continue
            distance = max(0, latest_start - start)
            weighted_matches += self.ngram_decay ** distance
        return self.ngram_repression_strength * weighted_matches

    def _zspace_probabilities(
        self,
        values: Sequence[float],
    ) -> tuple[list[float], dict[str, object]]:
        if self.use_native_zspace:
            native = self._native_probabilities(values)
            if native is not None:
                return native
        return _math_zspace_softmax(
            values,
            curvature=self.curvature,
            temperature=self.temperature,
            entropy_target=self.entropy_target,
            entropy_tolerance=self.entropy_tolerance,
            entropy_gain=self.entropy_gain,
            min_temperature=self.min_temperature,
            max_temperature=self.max_temperature,
        )

    def _native_probabilities(
        self,
        values: Sequence[float],
    ) -> tuple[list[float], dict[str, object]] | None:
        try:
            if self._native_layer is None:
                st = importlib.import_module("spiraltorch")
                nn = getattr(st, "nn", None)
                layer_type = getattr(nn, "ZSpaceSoftmax", None)
                tensor_type = getattr(st, "Tensor", None)
                if layer_type is None or tensor_type is None:
                    self._native_error = "spiraltorch.nn.ZSpaceSoftmax unavailable"
                    return None
                kwargs: dict[str, object] = {
                    "entropy_target": self.entropy_target,
                    "entropy_tolerance": self.entropy_tolerance,
                    "entropy_gain": self.entropy_gain,
                    "min_temperature": self.min_temperature,
                    "max_temperature": self.max_temperature,
                }
                self._native_layer = layer_type(
                    self.curvature,
                    self.temperature,
                    **kwargs,
                )
            st = importlib.import_module("spiraltorch")
            tensor = st.Tensor(1, len(values), list(values))
            output = self._native_layer(tensor)
            probabilities = _row_list(output)
            entropies = self._native_layer.last_entropies()
            temperatures = self._native_layer.last_temperatures()
            return probabilities, {
                "backend": "spiraltorch_zspace_softmax",
                "entropy": float(entropies[0]) if entropies else _entropy(probabilities),
                "temperature": (
                    float(temperatures[0]) if temperatures else self.temperature
                ),
                "adaptive_temperature": self.entropy_target is not None,
                "native_error": None,
            }
        except Exception as exc:  # pragma: no cover - defensive native fallback
            self._native_error = f"{exc.__class__.__name__}: {exc}"
            return None


def build_zspace_repression_logits_processor(
    **kwargs: object,
) -> ZSpaceRepressionLogitsProcessor:
    """Build a Transformers-compatible SpiralTorch generation processor."""

    return ZSpaceRepressionLogitsProcessor(**kwargs)


def build_zspace_softmax_logits_processor(
    **kwargs: object,
) -> ZSpaceRepressionLogitsProcessor:
    """Alias for callers that first reach for the ZSpaceSoftmax surface."""

    return build_zspace_repression_logits_processor(**kwargs)


def zspace_inference_distortion_processor_kwargs(
    adapter_or_config: Mapping[str, object] | None = None,
    **overrides: object,
) -> dict[str, object]:
    """Return local-HF logits-processor kwargs from a distortion adapter.

    ``zspace_inference_distortion_adapter`` returns a serializable payload that
    can drive both hosted API request controls and local HF logits processors.
    This helper extracts only the kwargs accepted by
    :class:`ZSpaceRepressionLogitsProcessor`, with explicit overrides winning.
    """

    source = dict(adapter_or_config or {})
    if isinstance(source.get("logits_processor_kwargs"), Mapping):
        source = dict(source["logits_processor_kwargs"])  # type: ignore[index]
    source.update(overrides)
    allowed = {
        "top_k",
        "curvature",
        "temperature",
        "entropy_target",
        "entropy_tolerance",
        "entropy_gain",
        "min_temperature",
        "max_temperature",
        "repression_window",
        "repression_strength",
        "last_token_repression",
        "ngram_size",
        "ngram_window",
        "ngram_repression_strength",
        "ngram_decay",
        "mask_non_top_k",
        "use_native_zspace",
    }
    return {key: value for key, value in source.items() if key in allowed}


def _selected_module_names(
    model: Any,
    *,
    module_names: Sequence[str] | None,
    name_contains: Sequence[str] | None,
    max_modules: int,
) -> list[tuple[str, Any]]:
    if max_modules <= 0:
        return []
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise TypeError("model must expose named_modules()")
    exact = {str(name) for name in (module_names or [])}
    contains = [str(part) for part in (name_contains or []) if str(part)]
    selected: list[tuple[str, Any]] = []
    for name, module in named_modules():
        if not name:
            continue
        if exact and name not in exact:
            continue
        if contains and not any(part in name for part in contains):
            continue
        if not exact and not contains:
            continue
        selected.append((str(name), module))
        if len(selected) >= max(0, int(max_modules)):
            break
    return selected


def _activation_summary(value: Any) -> dict[str, object]:
    tensor = value[0] if isinstance(value, (tuple, list)) and value else value
    shape = getattr(tensor, "shape", None)
    summary: dict[str, object] = {
        "shape": [int(dim) for dim in shape] if shape is not None else None,
    }
    try:
        detached = tensor.detach() if hasattr(tensor, "detach") else tensor
        numeric = detached.float() if hasattr(detached, "float") else detached
        abs_value = numeric.abs() if hasattr(numeric, "abs") else None
        square = numeric.pow(2) if hasattr(numeric, "pow") else None
        if abs_value is not None:
            summary["mean_abs"] = float(abs_value.mean().item())
            summary["max_abs"] = float(abs_value.max().item())
        if square is not None:
            summary["l2"] = float(square.sum().sqrt().item())
    except Exception as exc:  # pragma: no cover - defensive optional torch path.
        summary["error"] = f"{exc.__class__.__name__}: {exc}"
    return summary


def _intervene_activation(
    output: Any,
    *,
    scale: float,
    bias: float,
) -> Any:
    if scale == 1.0 and bias == 0.0:
        return output
    if isinstance(output, tuple) and output:
        first = _intervene_activation(output[0], scale=scale, bias=bias)
        return (first, *output[1:])
    if isinstance(output, list) and output:
        updated = list(output)
        updated[0] = _intervene_activation(updated[0], scale=scale, bias=bias)
        return updated
    adjusted = output
    try:
        if scale != 1.0:
            adjusted = adjusted * scale
        if bias != 0.0:
            adjusted = adjusted + bias
        return adjusted
    except Exception:
        return output


class ZSpaceActivationProbeHook:
    """Attach lightweight activation analysis/intervention hooks to HF models."""

    def __init__(
        self,
        *,
        module_names: Sequence[str] | None = None,
        name_contains: Sequence[str] | None = None,
        max_modules: int = 8,
        record_limit: int = 64,
        intervention_scale: float = 1.0,
        intervention_bias: float = 0.0,
        origin: str = "hf:activation_probe",
    ) -> None:
        self.module_names = [str(name) for name in (module_names or [])]
        self.name_contains = [str(part) for part in (name_contains or [])]
        self.max_modules = _non_negative_int(max_modules, label="max_modules")
        self.record_limit = _non_negative_int(record_limit, label="record_limit")
        self.intervention_scale = _finite_float(
            intervention_scale,
            label="intervention_scale",
        )
        self.intervention_bias = _finite_float(
            intervention_bias,
            label="intervention_bias",
        )
        self.origin = str(origin)
        self._handles: list[Any] = []
        self._events: list[dict[str, object]] = []
        self._attached_modules: list[str] = []

    def attach(self, model: Any) -> "ZSpaceActivationProbeHook":
        """Register hooks on selected modules and return ``self``."""

        self.close()
        selected = _selected_module_names(
            model,
            module_names=self.module_names,
            name_contains=self.name_contains,
            max_modules=self.max_modules,
        )
        for name, module in selected:
            register = getattr(module, "register_forward_hook", None)
            if not callable(register):
                continue
            handle = register(self._hook(name))
            self._handles.append(handle)
            self._attached_modules.append(name)
        return self

    def close(self) -> None:
        """Remove active hooks."""

        for handle in self._handles:
            remove = getattr(handle, "remove", None)
            if callable(remove):
                remove()
        self._handles.clear()
        self._attached_modules.clear()

    def reset_report(self) -> None:
        self._events.clear()

    def report(self, *, limit: int | None = None) -> dict[str, object]:
        if limit is None:
            events = list(self._events)
        elif limit <= 0:
            events = []
        else:
            events = list(self._events[-limit:])
        l2_values = [
            float(event["output_l2"])
            for event in self._events
            if isinstance(event.get("output_l2"), (int, float))
        ]
        return {
            "row_type": "zspace_activation_probe_hook_report",
            "status": "ok" if self._events else "unused",
            "origin": self.origin,
            "attached_modules": list(self._attached_modules),
            "event_count": len(self._events),
            "reported_event_count": len(events),
            "intervention_scale": self.intervention_scale,
            "intervention_bias": self.intervention_bias,
            "output_l2_min": min(l2_values) if l2_values else None,
            "output_l2_max": max(l2_values) if l2_values else None,
            "events": events,
        }

    def __enter__(self) -> "ZSpaceActivationProbeHook":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    def _hook(self, name: str):
        def _capture(_module: Any, _inputs: Any, output: Any) -> Any:
            summary = _activation_summary(output)
            event = {
                "module": name,
                "output_shape": summary.get("shape"),
                "output_l2": summary.get("l2"),
                "output_mean_abs": summary.get("mean_abs"),
                "output_max_abs": summary.get("max_abs"),
                "intervened": self.intervention_scale != 1.0
                or self.intervention_bias != 0.0,
            }
            if summary.get("error"):
                event["error"] = summary["error"]
            self._events.append(event)
            if self.record_limit and len(self._events) > self.record_limit:
                self._events = self._events[-self.record_limit :]
            return _intervene_activation(
                output,
                scale=self.intervention_scale,
                bias=self.intervention_bias,
            )

        return _capture


def build_zspace_activation_probe_hook(
    **kwargs: object,
) -> ZSpaceActivationProbeHook:
    """Build a local-HF activation analysis/intervention hook."""

    return ZSpaceActivationProbeHook(**kwargs)


def load_zspace_generation_control_sweep(path: str | Path) -> dict[str, object]:
    """Load one Z-Space generation-control sweep JSON artifact."""

    input_path = Path(path)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{input_path} invalid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{input_path} did not contain a JSON object")
    return dict(payload)


def load_zspace_inference_distortion_probe(path: str | Path) -> dict[str, object]:
    """Load one Z-Space inference-distortion probe JSON artifact."""

    input_path = Path(path)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{input_path} invalid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{input_path} did not contain a JSON object")
    return dict(payload)


def load_zspace_inference_distortion_sweep(path: str | Path) -> dict[str, object]:
    """Load a Z-Space inference-distortion sweep JSON artifact."""

    input_path = Path(path)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{input_path} invalid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{input_path} did not contain a JSON object")
    return dict(payload)


def _probe_payload(
    report_or_path: str | Path | Mapping[str, object],
) -> tuple[dict[str, object], str | None]:
    if isinstance(report_or_path, (str, Path)):
        return load_zspace_inference_distortion_probe(report_or_path), str(report_or_path)
    if isinstance(report_or_path, Mapping):
        report = dict(report_or_path)
        probe_path = report.get("probe_path")
        return report, str(probe_path) if probe_path is not None else None
    raise TypeError("inference-distortion probe must be a Mapping or path")


def _telemetry_number(
    row: Mapping[str, object],
    key: str,
) -> int | float | None:
    value = row.get(key)
    if isinstance(value, Mapping):
        return None
    return _safe_number(value)


def _first_number(*values: object) -> int | float | None:
    for value in values:
        number = _safe_number(value)
        if number is not None:
            return number
    return None


def _nested_mapping(
    row: Mapping[str, object],
    *keys: str,
) -> dict[str, object]:
    current: object = row
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return dict(current) if isinstance(current, Mapping) else {}


def _preview_text(value: object, *, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 1)]}..."


def _truthy(value: object) -> bool:
    return value is True or str(value).lower() in {"true", "1", "yes", "ok"}


def _score_number(value: object, *, scale: float = 1.0) -> float:
    number = _safe_number(value)
    if number is None:
        return 0.0
    return math.tanh(max(0.0, float(number)) / max(scale, EPSILON))


def summarize_zspace_inference_distortion_probe(
    report_or_path: str | Path | Mapping[str, object],
    *,
    preview_chars: int = 160,
) -> dict[str, object]:
    """Summarize a local-HF/API Z-Space inference-distortion probe."""

    report, source_path = _probe_payload(report_or_path)
    adapter = _nested_mapping(report, "adapter")
    local = _nested_mapping(report, "local_hf")
    api = _nested_mapping(report, "api")
    control = _nested_mapping(local, "generation_control")
    activation = _nested_mapping(local, "activation_report")
    request_filter = _nested_mapping(api, "request_filter")
    request = _nested_mapping(adapter, "request")
    logits_kwargs = _nested_mapping(adapter, "logits_processor_kwargs")
    adapter_context_telemetry = _nested_mapping(adapter, "context_partial", "telemetry")
    api_telemetry = _nested_mapping(api, "telemetry")
    telemetry = dict(adapter_context_telemetry)
    telemetry.update(api_telemetry)
    api_inference_telemetry = _nested_mapping(api, "inference", "telemetry", "payload")
    telemetry.update(api_inference_telemetry)
    baseline_text = local.get("baseline_text")
    distorted_text = local.get("distorted_text")
    summary = {
        "row_type": "zspace_inference_distortion_probe_summary",
        "probe_path": source_path,
        "prompt": report.get("prompt"),
        "distortion_energy": _safe_number(adapter.get("distortion_energy")),
        "desire_pressure": _telemetry_number(telemetry, "zspace.desire.pressure"),
        "desire_stability": _telemetry_number(telemetry, "zspace.desire.stability"),
        "psi_total": _telemetry_number(telemetry, "zspace.psi.total"),
        "coherence": _telemetry_number(telemetry, "zspace.coherence"),
        "request_temperature": _first_number(
            request.get("temperature"),
            telemetry.get("zspace.request.temperature"),
        ),
        "request_top_p": _first_number(
            request.get("top_p"),
            telemetry.get("zspace.request.top_p"),
        ),
        "request_frequency_penalty": _telemetry_number(
            telemetry,
            "zspace.request.frequency_penalty",
        ),
        "request_presence_penalty": _telemetry_number(
            telemetry,
            "zspace.request.presence_penalty",
        ),
        "logits_repression_strength": _first_number(
            logits_kwargs.get("repression_strength"),
            telemetry.get("zspace.logits.repression_strength"),
        ),
        "logits_ngram_repression_strength": _first_number(
            logits_kwargs.get("ngram_repression_strength"),
            telemetry.get("zspace.logits.ngram_repression_strength"),
        ),
        "local_status": local.get("status"),
        "local_changed": local.get("changed"),
        "local_model": local.get("model"),
        "local_baseline_method": local.get("baseline_method"),
        "local_distorted_method": local.get("distorted_method"),
        "local_baseline_fallback_error": local.get("baseline_fallback_error"),
        "local_distorted_fallback_error": local.get("distorted_fallback_error"),
        "local_baseline_preview": _preview_text(
            baseline_text,
            limit=max(0, int(preview_chars)),
        ),
        "local_distorted_preview": _preview_text(
            distorted_text,
            limit=max(0, int(preview_chars)),
        ),
        "generation_control_status": control.get("status"),
        "generation_control_backend": control.get("backend"),
        "generation_control_calls": _safe_number(control.get("calls")),
        "generation_control_top_token_changed_count": _safe_number(
            control.get("top_token_changed_count")
        ),
        "generation_control_ngram_repressed_token_total": _safe_number(
            control.get("ngram_repressed_token_total")
        ),
        "activation_status": activation.get("status"),
        "activation_event_count": _safe_number(activation.get("event_count")),
        "activation_reported_event_count": _safe_number(
            activation.get("reported_event_count")
        ),
        "activation_output_l2_min": _safe_number(activation.get("output_l2_min")),
        "activation_output_l2_max": _safe_number(activation.get("output_l2_max")),
        "api_provider": api.get("provider"),
        "api_model": api.get("model"),
        "api_finish_reason": api.get("finish_reason"),
        "api_request_dropped_key_count": _safe_number(
            request_filter.get("dropped_key_count")
        ),
        "api_request_dropped_keys": list(request_filter.get("dropped_keys", []))
        if isinstance(request_filter.get("dropped_keys"), list)
        else [],
        "api_request_retry_dropped_key_count": _safe_number(
            request_filter.get("retry_dropped_key_count")
        ),
        "api_request_retry_dropped_keys": list(
            request_filter.get("retry_dropped_keys", [])
        )
        if isinstance(request_filter.get("retry_dropped_keys"), list)
        else [],
        "api_request_sent_keys": list(request_filter.get("sent_keys", []))
        if isinstance(request_filter.get("sent_keys"), list)
        else [],
        "api_text_preview": _preview_text(api.get("text"), limit=max(0, int(preview_chars))),
        "api_total_tokens": _telemetry_number(telemetry, "api_llm.total_tokens"),
        "api_response_entropy_norm": _telemetry_number(
            telemetry,
            "api_llm.response_entropy_norm",
        ),
        "api_empty_text": _telemetry_number(telemetry, "api_llm.empty_text"),
    }
    summary["effect_score"] = _probe_effect_score(summary)
    summary["risk_score"] = _probe_risk_score(summary)
    return summary


def summarize_zspace_inference_distortion_probe_lines(
    report_or_path: str | Path | Mapping[str, object],
    *,
    preview_chars: int = 96,
) -> list[str]:
    """Render a compact text summary for an inference-distortion probe."""

    summary = summarize_zspace_inference_distortion_probe(
        report_or_path,
        preview_chars=preview_chars,
    )
    lines = [
        (
            "zspace_inference_distortion_probe "
            f"local={summary.get('local_status')} "
            f"changed={summary.get('local_changed')} "
            f"backend={summary.get('generation_control_backend')} "
            f"top_changes={summary.get('generation_control_top_token_changed_count')} "
            f"activation_events={summary.get('activation_event_count')} "
            f"api={summary.get('api_provider')} "
            f"api_dropped={summary.get('api_request_dropped_key_count')} "
            f"api_retry_dropped={summary.get('api_request_retry_dropped_key_count')} "
            f"effect={summary.get('effect_score')} "
            f"risk={summary.get('risk_score')} "
            f"energy={summary.get('distortion_energy')} "
            f"temp={summary.get('request_temperature')} "
            f"top_p={summary.get('request_top_p')}"
        )
    ]
    if summary.get("local_baseline_preview") or summary.get("local_distorted_preview"):
        lines.append(
            "zspace_inference_distortion_local "
            f"baseline={summary.get('local_baseline_preview')!r} "
            f"distorted={summary.get('local_distorted_preview')!r}"
        )
    if summary.get("api_text_preview"):
        lines.append(
            "zspace_inference_distortion_api "
            f"text={summary.get('api_text_preview')!r}"
        )
    return lines


def _iter_probe_inputs(
    probes: (
        Mapping[str, str | Path | Mapping[str, object]]
        | Sequence[str | Path | Mapping[str, object]]
        | str
        | Path
        | Mapping[str, object]
    ),
    *,
    labels: Sequence[str] | None,
) -> list[tuple[str | None, str | Path | Mapping[str, object]]]:
    if isinstance(probes, Mapping):
        if "row_type" in probes or "adapter" in probes or "local_hf" in probes:
            label = labels[0] if labels else None
            return [(label, probes)]
        return [(str(label), value) for label, value in probes.items()]
    if isinstance(probes, (str, Path)):
        label = labels[0] if labels else None
        return [(label, probes)]
    if isinstance(probes, (bytes, bytearray)):
        raise TypeError("probes must be paths, mappings, or sequences")
    try:
        values = list(probes)
    except TypeError as exc:
        raise TypeError("probes must be paths, mappings, or sequences") from exc
    result = []
    for index, value in enumerate(values):
        label = labels[index] if labels and index < len(labels) else None
        result.append((label, value))
    return result


def _default_probe_label(
    source: str | Path | Mapping[str, object],
    *,
    index: int,
) -> str:
    if isinstance(source, (str, Path)):
        path = Path(source)
        return path.stem or str(path) or f"probe_{index}"
    prompt = source.get("prompt") if isinstance(source, Mapping) else None
    if prompt:
        return str(prompt)[:48]
    return f"probe_{index}"


def _probe_effect_score(row: Mapping[str, object]) -> float:
    local_changed = 1.0 if _truthy(row.get("local_changed")) else 0.0
    top_changes = _score_number(
        row.get("generation_control_top_token_changed_count"),
        scale=8.0,
    )
    activation = _score_number(row.get("activation_event_count"), scale=64.0)
    api_non_empty = 1.0 - _score_number(row.get("api_empty_text"), scale=1.0)
    energy = _score_number(row.get("distortion_energy"), scale=1.0)
    return (
        0.35 * local_changed
        + 0.28 * top_changes
        + 0.17 * activation
        + 0.10 * api_non_empty
        + 0.10 * energy
    )


def _probe_risk_score(row: Mapping[str, object]) -> float:
    energy = _score_number(row.get("distortion_energy"), scale=1.0)
    top_changes = _score_number(
        row.get("generation_control_top_token_changed_count"),
        scale=24.0,
    )
    activation_l2 = _score_number(row.get("activation_output_l2_max"), scale=512.0)
    api_empty = _score_number(row.get("api_empty_text"), scale=1.0)
    return 0.45 * energy + 0.25 * top_changes + 0.20 * activation_l2 + 0.10 * api_empty


def _ranked_probe_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    top_n: int,
) -> list[dict[str, object]]:
    ranked = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            -float(row.get("effect_score") or 0.0),
            float(row.get("risk_score") or 0.0),
            str(row.get("label") or ""),
        ),
    )
    if top_n >= 0:
        ranked = ranked[:top_n]
    for index, row in enumerate(ranked, 1):
        row["rank"] = index
    return ranked


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def compare_zspace_inference_distortion_probes(
    probes: (
        Mapping[str, str | Path | Mapping[str, object]]
        | Sequence[str | Path | Mapping[str, object]]
        | str
        | Path
        | Mapping[str, object]
    ),
    *,
    labels: Sequence[str] | None = None,
    top_n: int = 5,
    preview_chars: int = 96,
) -> dict[str, object]:
    """Compare multiple Z-Space inference-distortion probe artifacts."""

    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    rows: list[dict[str, object]] = []
    for index, (label, source) in enumerate(
        _iter_probe_inputs(probes, labels=labels),
        start=1,
    ):
        summary = summarize_zspace_inference_distortion_probe(
            source,
            preview_chars=preview_chars,
        )
        summary["label"] = label or _default_probe_label(source, index=index)
        summary["effect_score"] = _probe_effect_score(summary)
        summary["risk_score"] = _probe_risk_score(summary)
        rows.append(summary)
    top_probes = _ranked_probe_rows(rows, top_n=top_n)
    best = top_probes[0] if top_probes else None
    changed_count = sum(1 for row in rows if _truthy(row.get("local_changed")))
    activation_observed_count = sum(
        1
        for row in rows
        if _safe_number(row.get("activation_event_count")) is not None
        and float(row.get("activation_event_count") or 0.0) > 0.0
    )
    top_change_values = [
        float(value)
        for row in rows
        if (
            value := _safe_number(
                row.get("generation_control_top_token_changed_count")
            )
        )
        is not None
    ]
    api_empty_values = [
        float(value)
        for row in rows
        if (value := _safe_number(row.get("api_empty_text"))) is not None
    ]
    api_retry_dropped_values = [
        float(value)
        for row in rows
        if (
            value := _safe_number(row.get("api_request_retry_dropped_key_count"))
        )
        is not None
    ]
    api_dropped_values = [
        float(value)
        for row in rows
        if (value := _safe_number(row.get("api_request_dropped_key_count")))
        is not None
    ]
    api_retry_dropped_keys = sorted(
        {
            key
            for row in rows
            for key in _string_list(row.get("api_request_retry_dropped_keys"))
        }
    )
    return {
        "row_type": "zspace_inference_distortion_probe_comparison",
        "probe_count": len(rows),
        "labels": [row.get("label") for row in rows],
        "local_changed_count": changed_count,
        "activation_observed_count": activation_observed_count,
        "max_top_token_changed_count": (
            max(top_change_values) if top_change_values else None
        ),
        "api_visible_text_count": sum(1 for value in api_empty_values if value <= 0.0),
        "api_empty_text_count": sum(1 for value in api_empty_values if value > 0.0),
        "api_retry_dropped_probe_count": sum(
            1 for value in api_retry_dropped_values if value > 0.0
        ),
        "api_retry_dropped_key_total": (
            sum(api_retry_dropped_values) if api_retry_dropped_values else None
        ),
        "api_retry_dropped_keys": api_retry_dropped_keys,
        "api_request_dropped_key_total": (
            sum(api_dropped_values) if api_dropped_values else None
        ),
        "recommended_probe": None if best is None else best.get("label"),
        "recommended_reason": (
            None
            if best is None
            else "highest_effect_score_lowest_risk_tiebreak"
        ),
        "best_effect_score": None if best is None else best.get("effect_score"),
        "best_risk_score": None if best is None else best.get("risk_score"),
        "top_probes": top_probes,
        "summaries": rows,
    }


def _distortion_probe_runtime(report: Mapping[str, object]) -> dict[str, object] | None:
    runtime = report.get("runtime")
    if isinstance(runtime, Mapping):
        return dict(runtime)
    api = _nested_mapping(report, "api")
    local = _nested_mapping(report, "local_hf")
    if not api and not local:
        return None
    return {
        "local_model": local.get("model"),
        "allow_remote": None,
        "trust_remote_code": None,
        "max_new_tokens": None,
        "activation_module_name": [],
        "activation_name_contains": [],
        "api_provider": api.get("provider"),
        "api_model": api.get("model"),
        "api_max_tokens": None,
    }


def _unique_distortion_probe_label(existing: set[str], raw: object, *, index: int) -> str:
    base = str(raw or f"probe-{index:03d}").strip() or f"probe-{index:03d}"
    label = base
    suffix = 2
    while label in existing:
        label = f"{base}-{suffix}"
        suffix += 1
    existing.add(label)
    return label


def _probe_path_for_source(source: str | Path | Mapping[str, object]) -> str | None:
    if isinstance(source, (str, Path)):
        return str(source)
    probe_path = source.get("probe_path") if isinstance(source, Mapping) else None
    return str(probe_path) if probe_path is not None else None


def zspace_inference_distortion_sweep_report_from_probes(
    probes: (
        Mapping[str, str | Path | Mapping[str, object]]
        | Sequence[str | Path | Mapping[str, object]]
        | str
        | Path
        | Mapping[str, object]
    ),
    *,
    labels: Sequence[str] | None = None,
    prompt: object | None = None,
    runtime: Mapping[str, object] | None = None,
    report_path: str | Path | None = None,
    top_n: int = 5,
    preview_chars: int = 96,
) -> dict[str, object]:
    """Promote saved inference-distortion probes into a sweep-shaped report."""

    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    inputs = _iter_probe_inputs(probes, labels=labels)
    runs: list[dict[str, object]] = []
    comparison_inputs: dict[str, str | Path | Mapping[str, object]] = {}
    names: set[str] = set()
    prompt_value = prompt
    runtime_value = dict(runtime) if isinstance(runtime, Mapping) else None
    for index, (label, source) in enumerate(inputs, start=1):
        probe_payload, source_path = _probe_payload(source)
        name = _unique_distortion_probe_label(
            names,
            label or probe_payload.get("name") or _default_probe_label(source, index=index),
            index=index,
        )
        if prompt_value is None and probe_payload.get("prompt") is not None:
            prompt_value = probe_payload.get("prompt")
        if runtime_value is None:
            runtime_value = _distortion_probe_runtime(probe_payload)
        summary_source: str | Path | Mapping[str, object]
        summary_source = source if isinstance(source, (str, Path)) else probe_payload
        summary = summarize_zspace_inference_distortion_probe(
            summary_source,
            preview_chars=preview_chars,
        )
        config = probe_payload.get("config")
        run: dict[str, object] = {
            "name": name,
            "index": index,
            "config": dict(config) if isinstance(config, Mapping) else {},
            "probe_path": source_path or _probe_path_for_source(source),
            "status": "reported",
            "summary": summary,
            "reported": True,
        }
        if source_path is None:
            run["probe_payload"] = probe_payload
        runs.append(run)
        comparison_inputs[name] = summary_source
    comparison = compare_zspace_inference_distortion_probes(
        comparison_inputs,
        top_n=top_n,
        preview_chars=preview_chars,
    )
    report: dict[str, object] = {
        "row_type": "zspace_inference_distortion_sweep",
        "status": "reported",
        "dry_run": False,
        "prompt": prompt_value,
        "runtime": runtime_value or {},
        "execution": {
            "resume_existing": False,
            "force": False,
            "report_only": True,
            "from_probe_count": len(runs),
        },
        "run_count": len(runs),
        "attempted_run_count": 0,
        "completed_run_count": len(runs),
        "failed_run_count": 0,
        "missing_run_count": 0,
        "stale_run_count": 0,
        "reused_run_count": 0,
        "reported_run_count": len(runs),
        "skipped_run_count": 0,
        "runs": runs,
        "comparison": comparison,
        "summary_lines": summarize_zspace_inference_distortion_probe_comparison_lines(
            comparison,
            top_n=top_n,
            preview_chars=preview_chars,
        ),
        "source_probe_paths": [
            str(path)
            for path in (
                run.get("probe_path")
                for run in runs
                if run.get("probe_path") is not None
            )
        ],
        "report_path": None if report_path is None else str(report_path),
    }
    report["summary"] = summarize_zspace_inference_distortion_sweep(
        report,
        top_n=top_n,
        preview_chars=preview_chars,
    )
    report["summary_lines"] = summarize_zspace_inference_distortion_sweep_lines(
        report,
        top_n=top_n,
        preview_chars=preview_chars,
    )
    return report


def summarize_zspace_inference_distortion_probe_comparison_lines(
    comparison_or_probes: (
        Mapping[str, object]
        | Mapping[str, str | Path | Mapping[str, object]]
        | Sequence[str | Path | Mapping[str, object]]
        | str
        | Path
    ),
    *,
    top_n: int = 3,
    preview_chars: int = 96,
) -> list[str]:
    """Render compact status lines for an inference-distortion comparison."""

    if (
        isinstance(comparison_or_probes, Mapping)
        and comparison_or_probes.get("row_type")
        == "zspace_inference_distortion_probe_comparison"
    ):
        comparison = dict(comparison_or_probes)
    else:
        comparison = compare_zspace_inference_distortion_probes(
            comparison_or_probes,
            top_n=top_n,
            preview_chars=preview_chars,
        )
    lines = [
        (
            "zspace_inference_distortion_compare "
            f"probes={comparison.get('probe_count')} "
            f"recommended={comparison.get('recommended_probe')} "
            f"effect={comparison.get('best_effect_score')} "
            f"risk={comparison.get('best_risk_score')} "
            f"changed={comparison.get('local_changed_count')} "
            f"activation={comparison.get('activation_observed_count')} "
            f"max_top_changes={comparison.get('max_top_token_changed_count')} "
            f"api_visible={comparison.get('api_visible_text_count')} "
            f"api_empty={comparison.get('api_empty_text_count')} "
            f"api_retry_dropped={comparison.get('api_retry_dropped_probe_count')}"
        )
    ]
    for row in comparison.get("top_probes", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "zspace_inference_distortion_top "
            f"rank={row.get('rank')} "
            f"label={row.get('label')} "
            f"effect={row.get('effect_score')} "
            f"risk={row.get('risk_score')} "
            f"changed={row.get('local_changed')} "
            f"top_changes={row.get('generation_control_top_token_changed_count')} "
            f"api={row.get('api_provider')} "
            f"api_empty={row.get('api_empty_text')} "
            f"api_retry_dropped={row.get('api_request_retry_dropped_key_count')} "
            f"energy={row.get('distortion_energy')}"
        )
    return lines


def _distortion_sweep_payload(
    report_or_path: str | Path | Mapping[str, object],
) -> tuple[dict[str, object], str | None]:
    if isinstance(report_or_path, (str, Path)):
        return load_zspace_inference_distortion_sweep(report_or_path), str(report_or_path)
    if isinstance(report_or_path, Mapping):
        report = dict(report_or_path)
        report_path = report.get("report_path")
        return report, str(report_path) if report_path is not None else None
    raise TypeError("inference-distortion sweep must be a Mapping or path")


def zspace_inference_distortion_probe_cli_args(
    config: Mapping[str, object] | None,
) -> list[str]:
    """Return focused single-probe CLI args for a distortion config."""

    if not config:
        return []
    flag_map = [
        ("desire_pressure", "--desire-pressure"),
        ("desire_stability", "--desire-stability"),
        ("psi_total", "--psi-total"),
        ("coherence", "--coherence"),
        ("distortion_strength", "--distortion-strength"),
        ("base_temperature", "--base-temperature"),
        ("base_top_p", "--base-top-p"),
    ]
    args: list[str] = []
    for key, flag in flag_map:
        if key in config and config[key] is not None:
            args.extend([flag, _cli_value(config[key])])
    if config.get("include_penalties") is True:
        args.append("--include-penalties")
    return args


def zspace_inference_distortion_sweep_cli_args(
    config: Mapping[str, object] | None,
) -> list[str]:
    """Return focused sweep CLI args for replaying one distortion config."""

    if not config:
        return []
    flag_map = [
        ("desire_pressure", "--desire-pressure-values"),
        ("desire_stability", "--desire-stability-values"),
        ("psi_total", "--psi-total-values"),
        ("coherence", "--coherence-values"),
        ("distortion_strength", "--distortion-strength-values"),
        ("base_temperature", "--base-temperature"),
        ("base_top_p", "--base-top-p"),
    ]
    args: list[str] = []
    for key, flag in flag_map:
        if key in config and config[key] is not None:
            args.extend([flag, _cli_value(config[key])])
    if config.get("include_penalties") is True:
        args.append("--include-penalties")
    return args


def _distortion_sweep_runs(report: Mapping[str, object]) -> list[dict[str, object]]:
    runs_value = report.get("runs")
    if not isinstance(runs_value, Sequence) or isinstance(runs_value, (str, bytes)):
        return []
    return [dict(row) for row in runs_value if isinstance(row, Mapping)]


def _distortion_success_probe_inputs(
    runs: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in runs:
        if row.get("status") not in {"ok", "reused", "reported"}:
            continue
        probe_path = row.get("probe_path")
        if probe_path is None:
            continue
        path = Path(str(probe_path))
        if path.is_file():
            result[str(row.get("name") or path.stem)] = str(path)
    return result


def _distortion_comparison_for_sweep(
    report: Mapping[str, object],
    runs: Sequence[Mapping[str, object]],
    *,
    top_n: int,
    preview_chars: int,
) -> dict[str, object]:
    probe_inputs = _distortion_success_probe_inputs(runs)
    if probe_inputs:
        return compare_zspace_inference_distortion_probes(
            probe_inputs,
            top_n=top_n,
            preview_chars=preview_chars,
        )
    comparison = report.get("comparison")
    if (
        isinstance(comparison, Mapping)
        and comparison.get("row_type")
        == "zspace_inference_distortion_probe_comparison"
    ):
        copied = dict(comparison)
        top = copied.get("top_probes")
        if isinstance(top, Sequence) and not isinstance(top, (str, bytes)):
            copied["top_probes"] = [
                dict(row)
                for row in list(top)[: max(0, int(top_n))]
                if isinstance(row, Mapping)
            ]
        return copied
    return compare_zspace_inference_distortion_probes({}, top_n=top_n)


def _distortion_selected_run(
    runs: Sequence[Mapping[str, object]],
    label: object,
) -> dict[str, object] | None:
    if label is None:
        return None
    text = str(label)
    for row in runs:
        if str(row.get("name")) == text:
            return dict(row)
    return None


def _distortion_config_from_summary(
    row: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if row is None:
        return None
    fields = {
        "desire_pressure": row.get("desire_pressure"),
        "desire_stability": row.get("desire_stability"),
        "psi_total": row.get("psi_total"),
        "coherence": row.get("coherence"),
        "base_temperature": row.get("request_temperature"),
        "base_top_p": row.get("request_top_p"),
    }
    return {key: value for key, value in fields.items() if value is not None}


def _distortion_recommendation_payload(
    selected_run: Mapping[str, object] | None,
    best: Mapping[str, object] | None,
) -> tuple[
    dict[str, object] | None,
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    config = None
    if selected_run is not None and isinstance(selected_run.get("config"), Mapping):
        config = dict(selected_run["config"])  # type: ignore[index]
    if config is None:
        config = _distortion_config_from_summary(best)
    probe_payload: dict[str, object] = {}
    inline_probe = None if selected_run is None else selected_run.get("probe_payload")
    if isinstance(inline_probe, Mapping):
        probe_payload = dict(inline_probe)
    probe_path = None if selected_run is None else selected_run.get("probe_path")
    if probe_path is None and best is not None:
        probe_path = best.get("probe_path")
    if not probe_payload and probe_path is not None and Path(str(probe_path)).is_file():
        try:
            probe_payload = load_zspace_inference_distortion_probe(str(probe_path))
        except Exception:
            probe_payload = {}
    adapter = probe_payload.get("adapter")
    adapter_mapping = dict(adapter) if isinstance(adapter, Mapping) else {}
    request = adapter_mapping.get("request")
    activation_hook = adapter_mapping.get("activation_hook")
    api = probe_payload.get("api")
    api_mapping = dict(api) if isinstance(api, Mapping) else {}
    request_filter = api_mapping.get("request_filter")
    request_filter_mapping = (
        dict(request_filter) if isinstance(request_filter, Mapping) else {}
    )
    if not request_filter_mapping and best is not None:
        dropped_keys = best.get("api_request_dropped_keys")
        sent_keys = best.get("api_request_sent_keys")
        dropped_count = best.get("api_request_dropped_key_count")
        retry_dropped_keys = best.get("api_request_retry_dropped_keys")
        retry_dropped_count = best.get("api_request_retry_dropped_key_count")
        if (
            isinstance(dropped_keys, list)
            or isinstance(sent_keys, list)
            or dropped_count is not None
            or isinstance(retry_dropped_keys, list)
            or retry_dropped_count is not None
        ):
            request_filter_mapping = {
                "dropped_key_count": dropped_count,
                "dropped_keys": list(dropped_keys)
                if isinstance(dropped_keys, list)
                else [],
                "sent_keys": list(sent_keys) if isinstance(sent_keys, list) else [],
                "retry_dropped_key_count": retry_dropped_count,
                "retry_dropped_keys": list(retry_dropped_keys)
                if isinstance(retry_dropped_keys, list)
                else [],
            }
    processor_kwargs = (
        zspace_inference_distortion_processor_kwargs(adapter_mapping)
        if adapter_mapping
        else {}
    )
    return (
        config,
        dict(request) if isinstance(request, Mapping) else {},
        processor_kwargs,
        dict(activation_hook) if isinstance(activation_hook, Mapping) else {},
        request_filter_mapping,
    )


def summarize_zspace_inference_distortion_sweep(
    report_or_path: str | Path | Mapping[str, object],
    *,
    top_n: int = 5,
    preview_chars: int = 96,
) -> dict[str, object]:
    """Summarize a Z-Space inference-distortion sweep artifact."""

    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    report, source_path = _distortion_sweep_payload(report_or_path)
    runs = _distortion_sweep_runs(report)
    comparison = _distortion_comparison_for_sweep(
        report,
        runs,
        top_n=top_n,
        preview_chars=preview_chars,
    )
    top_probes_value = comparison.get("top_probes")
    top_probes = (
        [dict(row) for row in top_probes_value if isinstance(row, Mapping)]
        if isinstance(top_probes_value, Sequence)
        and not isinstance(top_probes_value, (str, bytes))
        else []
    )
    best = top_probes[0] if top_probes else None
    recommended_probe = None if best is None else best.get("label")
    selected_run = _distortion_selected_run(runs, recommended_probe)
    (
        config,
        request,
        processor_kwargs,
        activation_hook,
        request_filter,
    ) = _distortion_recommendation_payload(
        selected_run,
        best,
    )
    return {
        "row_type": "zspace_inference_distortion_sweep_summary",
        "sweep_path": source_path,
        "status": report.get("status"),
        "dry_run": bool(report.get("dry_run")),
        "prompt": report.get("prompt"),
        "run_count": _safe_number(report.get("run_count")) or len(runs),
        "completed_run_count": sum(
            1 for row in runs if row.get("status") in {"ok", "reused", "reported"}
        ),
        "attempted_run_count": _safe_number(report.get("attempted_run_count")),
        "reused_run_count": _safe_number(report.get("reused_run_count")),
        "reported_run_count": _safe_number(report.get("reported_run_count")),
        "failed_run_count": _safe_number(report.get("failed_run_count")),
        "missing_run_count": _safe_number(report.get("missing_run_count")),
        "stale_run_count": _safe_number(report.get("stale_run_count")),
        "recommended_probe": recommended_probe,
        "recommendation_reason": comparison.get("recommended_reason"),
        "recommended_effect_score": None if best is None else best.get("effect_score"),
        "recommended_risk_score": None if best is None else best.get("risk_score"),
        "recommended_probe_path": (
            None if selected_run is None else selected_run.get("probe_path")
        ),
        "recommended_config": config,
        "recommended_request": request,
        "recommended_request_filter": request_filter,
        "recommended_api_request_dropped_key_count": _safe_number(
            request_filter.get("dropped_key_count")
        ),
        "recommended_api_request_dropped_keys": list(
            request_filter.get("dropped_keys", [])
        )
        if isinstance(request_filter.get("dropped_keys"), list)
        else [],
        "recommended_api_request_retry_dropped_key_count": _safe_number(
            request_filter.get("retry_dropped_key_count")
        ),
        "recommended_api_request_retry_dropped_keys": list(
            request_filter.get("retry_dropped_keys", [])
        )
        if isinstance(request_filter.get("retry_dropped_keys"), list)
        else [],
        "recommended_api_request_sent_keys": list(request_filter.get("sent_keys", []))
        if isinstance(request_filter.get("sent_keys"), list)
        else [],
        "recommended_processor_kwargs": processor_kwargs,
        "recommended_activation_hook": activation_hook,
        "recommended_probe_cli_args": zspace_inference_distortion_probe_cli_args(config),
        "recommended_sweep_cli_args": zspace_inference_distortion_sweep_cli_args(config),
        "recommended_cli_args": zspace_inference_distortion_sweep_cli_args(config),
        "comparison": comparison,
        "top_probes": top_probes,
    }


def summarize_zspace_inference_distortion_sweep_lines(
    report_or_path: str | Path | Mapping[str, object],
    *,
    top_n: int = 3,
    preview_chars: int = 96,
) -> list[str]:
    """Render compact status lines for an inference-distortion sweep."""

    summary = summarize_zspace_inference_distortion_sweep(
        report_or_path,
        top_n=top_n,
        preview_chars=preview_chars,
    )
    lines = [
        (
            "zspace_inference_distortion_sweep "
            f"status={summary.get('status')} "
            f"runs={summary.get('completed_run_count')}/{summary.get('run_count')} "
            f"recommend={summary.get('recommended_probe')} "
            f"effect={summary.get('recommended_effect_score')} "
            f"risk={summary.get('recommended_risk_score')} "
            f"stale={summary.get('stale_run_count')}"
        )
    ]
    for row in summary.get("top_probes", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "zspace_inference_distortion_sweep_top "
            f"rank={row.get('rank')} "
            f"label={row.get('label')} "
            f"effect={row.get('effect_score')} "
            f"risk={row.get('risk_score')} "
            f"changed={row.get('local_changed')} "
            f"top_changes={row.get('generation_control_top_token_changed_count')}"
        )
    return lines


def _sweep_payload(
    report_or_path: str | Path | Mapping[str, object],
) -> tuple[dict[str, object], str | None]:
    if isinstance(report_or_path, (str, Path)):
        return load_zspace_generation_control_sweep(report_or_path), str(report_or_path)
    if isinstance(report_or_path, Mapping):
        return dict(report_or_path), None
    raise TypeError("generation-control sweep must be a Mapping or path")


def _run_generation(row: Mapping[str, object]) -> dict[str, object]:
    return _mapping_item(row, "generation")


def _run_control(row: Mapping[str, object]) -> dict[str, object]:
    generation = _run_generation(row)
    return _mapping_item(generation, "generation_control")


def _run_repetition(row: Mapping[str, object]) -> dict[str, object]:
    return _mapping_item(row, "repetition")


def _run_config(row: Mapping[str, object]) -> dict[str, object]:
    return _mapping_item(row, "config")


def summarize_zspace_generation_control_run(
    row: Mapping[str, object],
    *,
    baseline_continuation_sha256: object = None,
    baseline_loop_score: object = None,
) -> dict[str, object]:
    """Flatten one generation-control sweep row for comparison."""

    generation = _run_generation(row)
    control = _run_control(row)
    repetition = _run_repetition(row)
    config = _run_config(row)
    continuation_hash = generation.get("generated_continuation_sha256")
    baseline_hash = (
        None
        if baseline_continuation_sha256 is None
        else str(baseline_continuation_sha256)
    )
    changed_from_baseline = (
        None
        if not continuation_hash or not baseline_hash
        else str(continuation_hash) != baseline_hash
    )
    loop_score = _safe_number(repetition.get("loop_score"))
    baseline_loop = _safe_number(baseline_loop_score)
    loop_delta = None
    loop_reduction_ratio = None
    if loop_score is not None and baseline_loop is not None:
        loop_delta = float(loop_score) - float(baseline_loop)
        if float(baseline_loop) > 0.0:
            loop_reduction_ratio = (float(baseline_loop) - float(loop_score)) / float(
                baseline_loop
            )
    return {
        "row_type": "zspace_generation_control_run_summary",
        "name": row.get("name"),
        "kind": row.get("kind"),
        "status": row.get("status"),
        "error": row.get("error"),
        "generation_status": generation.get("status"),
        "generation_method": generation.get("generation_method"),
        "continuation": generation.get("generated_continuation_text"),
        "continuation_sha256": continuation_hash,
        "continuation_char_count": _safe_number(
            generation.get("generated_continuation_char_count")
        ),
        "new_token_count": _safe_number(generation.get("new_token_count")),
        "changed_from_baseline": changed_from_baseline,
        "loop_score": loop_score,
        "baseline_loop_score": baseline_loop,
        "loop_score_delta_from_baseline": loop_delta,
        "loop_score_reduction_ratio": loop_reduction_ratio,
        "unique_word_ratio": _safe_number(repetition.get("unique_word_ratio")),
        "repeated_ngram_total": _safe_number(
            repetition.get("repeated_ngram_total")
        ),
        "max_ngram_repetition": _safe_number(
            repetition.get("max_ngram_repetition")
        ),
        "consecutive_repeated_tokens": _safe_number(
            repetition.get("consecutive_repeated_tokens")
        ),
        "control_status": control.get("status"),
        "control_backend": control.get("backend"),
        "control_calls": _safe_number(control.get("calls")),
        "control_reported_rows": _safe_number(control.get("reported_rows")),
        "control_top_token_changed_count": _safe_number(
            control.get("top_token_changed_count")
        ),
        "control_reported_top_token_changed_count": _safe_number(
            control.get("reported_top_token_changed_count")
        ),
        "control_temperature_min": _safe_number(control.get("temperature_min")),
        "control_temperature_max": _safe_number(control.get("temperature_max")),
        "control_entropy_min": _safe_number(control.get("entropy_min")),
        "control_entropy_max": _safe_number(control.get("entropy_max")),
        "control_ngram_repressed_token_total": _safe_number(
            control.get("ngram_repressed_token_total")
        ),
        "control_max_ngram_repression": _safe_number(
            control.get("max_ngram_repression")
        ),
        "control_native_error": control.get("native_error"),
        "config_top_k": _safe_number(config.get("top_k")),
        "config_curvature": _safe_number(config.get("curvature")),
        "config_temperature": _safe_number(config.get("temperature")),
        "config_entropy_target": _safe_number(config.get("entropy_target")),
        "config_entropy_tolerance": _safe_number(
            config.get("entropy_tolerance")
        ),
        "config_entropy_gain": _safe_number(config.get("entropy_gain")),
        "config_min_temperature": _safe_number(config.get("min_temperature")),
        "config_max_temperature": _safe_number(config.get("max_temperature")),
        "config_repression_window": _safe_number(config.get("repression_window")),
        "config_repression_strength": _safe_number(
            config.get("repression_strength")
        ),
        "config_last_token_repression": _safe_number(
            config.get("last_token_repression")
        ),
        "config_ngram_size": _safe_number(config.get("ngram_size")),
        "config_ngram_window": _safe_number(config.get("ngram_window")),
        "config_ngram_repression_strength": _safe_number(
            config.get("ngram_repression_strength")
        ),
        "config_ngram_decay": _safe_number(config.get("ngram_decay")),
        "config_mask_non_top_k": (
            bool(config["mask_non_top_k"]) if "mask_non_top_k" in config else None
        ),
        "config_use_native_zspace": (
            bool(config["use_native_zspace"])
            if "use_native_zspace" in config
            else None
        ),
    }


def _baseline_continuation_hash(rows: Sequence[Mapping[str, object]]) -> object:
    for row in rows:
        if row.get("kind") != "baseline":
            continue
        generation = _run_generation(row)
        value = generation.get("generated_continuation_sha256")
        if value:
            return value
    return None


def _baseline_loop_score(rows: Sequence[Mapping[str, object]]) -> int | float | None:
    for row in rows:
        if row.get("kind") != "baseline":
            continue
        repetition = _run_repetition(row)
        return _safe_number(repetition.get("loop_score"))
    return None


def _ranked_control_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    top_n: int,
) -> list[dict[str, object]]:
    def sort_key(row: Mapping[str, object]) -> tuple[float, float, str]:
        loop_score = _safe_number(row.get("loop_score"))
        changes = _safe_number(row.get("control_top_token_changed_count"))
        return (
            math.inf if loop_score is None else float(loop_score),
            0.0 if changes is None else -float(changes),
            str(row.get("name") or ""),
        )

    ranked = sorted(rows, key=sort_key)
    if top_n >= 0:
        ranked = ranked[:top_n]
    return [dict(row, rank=index) for index, row in enumerate(ranked, 1)]


def _recommended_config(row: Mapping[str, object] | None) -> dict[str, object] | None:
    if row is None or row.get("kind") == "baseline":
        return None
    fields = {
        "top_k": row.get("config_top_k"),
        "curvature": row.get("config_curvature"),
        "temperature": row.get("config_temperature"),
        "entropy_target": row.get("config_entropy_target"),
        "entropy_tolerance": row.get("config_entropy_tolerance"),
        "entropy_gain": row.get("config_entropy_gain"),
        "min_temperature": row.get("config_min_temperature"),
        "max_temperature": row.get("config_max_temperature"),
        "repression_window": row.get("config_repression_window"),
        "repression_strength": row.get("config_repression_strength"),
        "last_token_repression": row.get("config_last_token_repression"),
        "ngram_size": row.get("config_ngram_size"),
        "ngram_window": row.get("config_ngram_window"),
        "ngram_repression_strength": row.get(
            "config_ngram_repression_strength"
        ),
        "ngram_decay": row.get("config_ngram_decay"),
        "mask_non_top_k": row.get("config_mask_non_top_k"),
        "use_native_zspace": row.get("config_use_native_zspace"),
    }
    return {key: value for key, value in fields.items() if value is not None}


def _cli_value(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def zspace_generation_control_processor_kwargs(
    config: Mapping[str, object] | None,
) -> dict[str, object]:
    """Return kwargs suitable for ``build_zspace_repression_logits_processor``."""

    if not config:
        return {}
    allowed = {
        "top_k",
        "curvature",
        "temperature",
        "entropy_target",
        "entropy_tolerance",
        "entropy_gain",
        "min_temperature",
        "max_temperature",
        "repression_window",
        "repression_strength",
        "last_token_repression",
        "ngram_size",
        "ngram_window",
        "ngram_repression_strength",
        "ngram_decay",
        "mask_non_top_k",
        "use_native_zspace",
    }
    return {key: value for key, value in config.items() if key in allowed}


def zspace_generation_control_sweep_cli_args(
    config: Mapping[str, object] | None,
) -> list[str]:
    """Return focused sweep CLI args for a recommended generation config."""

    if not config:
        return []
    flag_map = [
        ("top_k", "--zspace-top-k-values"),
        ("curvature", "--zspace-curvature-values"),
        ("temperature", "--zspace-temperature-values"),
        ("entropy_target", "--zspace-entropy-target-values"),
        ("entropy_gain", "--zspace-entropy-gain-values"),
        ("entropy_tolerance", "--zspace-entropy-tolerance"),
        ("min_temperature", "--zspace-min-temperature"),
        ("max_temperature", "--zspace-max-temperature"),
        ("repression_window", "--repression-window-values"),
        ("repression_strength", "--repression-strength-values"),
        ("last_token_repression", "--last-token-repression-values"),
        ("ngram_size", "--ngram-size-values"),
        ("ngram_window", "--ngram-window-values"),
        ("ngram_repression_strength", "--ngram-repression-strength-values"),
        ("ngram_decay", "--ngram-decay-values"),
    ]
    args: list[str] = []
    for key, flag in flag_map:
        if key not in config:
            continue
        args.extend([flag, _cli_value(config[key])])
    if config.get("mask_non_top_k") is False:
        args.append("--keep-non-top-k")
    if config.get("use_native_zspace") is False:
        args.append("--zspace-no-native")
    return args


def zspace_generation_control_bridge_cli_args(
    config: Mapping[str, object] | None,
    *,
    include_enable_flag: bool = True,
) -> list[str]:
    """Return bridge/sweep CLI args for using one recommended config."""

    if not config:
        return []
    flag_map = [
        ("top_k", "--generation-zspace-top-k"),
        ("curvature", "--generation-zspace-curvature"),
        ("temperature", "--generation-zspace-temperature"),
        ("entropy_target", "--generation-zspace-entropy-target"),
        ("entropy_gain", "--generation-zspace-entropy-gain"),
        ("entropy_tolerance", "--generation-zspace-entropy-tolerance"),
        ("min_temperature", "--generation-zspace-min-temperature"),
        ("max_temperature", "--generation-zspace-max-temperature"),
        ("repression_window", "--generation-repression-window"),
        ("repression_strength", "--generation-repression-strength"),
        ("last_token_repression", "--generation-last-token-repression"),
        ("ngram_size", "--generation-ngram-size"),
        ("ngram_window", "--generation-ngram-window"),
        ("ngram_repression_strength", "--generation-ngram-repression-strength"),
        ("ngram_decay", "--generation-ngram-decay"),
    ]
    args: list[str] = ["--generation-zspace-softmax"] if include_enable_flag else []
    for key, flag in flag_map:
        if key not in config:
            continue
        value = config[key]
        if key == "entropy_target" and value is None:
            continue
        args.extend([flag, _cli_value(value)])
    if config.get("mask_non_top_k") is False:
        args.append("--generation-zspace-keep-non-top-k")
    if config.get("use_native_zspace") is False:
        args.append("--generation-zspace-no-native")
    return args


def _recommendation_reason(
    best: Mapping[str, object] | None,
    *,
    baseline_loop_score: int | float | None,
) -> str | None:
    if best is None:
        return None
    loop_score = _safe_number(best.get("loop_score"))
    top_changes = _safe_number(best.get("control_top_token_changed_count"))
    if baseline_loop_score is not None and loop_score is not None:
        if float(loop_score) < float(baseline_loop_score):
            return (
                "lowest_loop_score_with_baseline_reduction"
                if top_changes is not None and float(top_changes) > 0.0
                else "lowest_loop_score"
            )
    return "lowest_loop_score"


def summarize_zspace_generation_control_sweep(
    report_or_path: str | Path | Mapping[str, object],
    *,
    top_n: int = 5,
) -> dict[str, object]:
    """Summarize a Z-Space generation-control sweep artifact."""

    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    report, source_path = _sweep_payload(report_or_path)
    runs_value = report.get("runs")
    runs = (
        [dict(row) for row in runs_value if isinstance(row, Mapping)]
        if isinstance(runs_value, Sequence) and not isinstance(runs_value, (str, bytes))
        else []
    )
    baseline_hash = _baseline_continuation_hash(runs)
    baseline_loop = _baseline_loop_score(runs)
    summaries = [
        summarize_zspace_generation_control_run(
            row,
            baseline_continuation_sha256=baseline_hash,
            baseline_loop_score=baseline_loop,
        )
        for row in runs
    ]
    completed = [row for row in summaries if row.get("status") == "ok"]
    loop_values = [
        float(value)
        for row in completed
        if (value := _safe_number(row.get("loop_score"))) is not None
    ]
    control_changes = [
        float(value)
        for row in completed
        if (value := _safe_number(row.get("control_top_token_changed_count")))
        is not None
    ]
    changed_from_baseline_count = sum(
        1 for row in completed if row.get("changed_from_baseline") is True
    )
    top_runs = _ranked_control_rows(completed, top_n=top_n)
    best = top_runs[0] if top_runs else None
    recommended_config = _recommended_config(best)
    recommended_processor_kwargs = zspace_generation_control_processor_kwargs(
        recommended_config
    )
    recommended_sweep_cli_args = zspace_generation_control_sweep_cli_args(
        recommended_config
    )
    recommended_bridge_cli_args = zspace_generation_control_bridge_cli_args(
        recommended_config
    )
    best_loop_delta = None if best is None else best.get("loop_score_delta_from_baseline")
    best_loop_ratio = None if best is None else best.get("loop_score_reduction_ratio")
    return {
        "row_type": "zspace_generation_control_sweep_summary",
        "sweep_path": source_path,
        "status": report.get("status"),
        "dry_run": bool(report.get("dry_run")),
        "model_name": report.get("model_name"),
        "prompt": report.get("prompt"),
        "run_count": _safe_number(report.get("run_count")) or len(runs),
        "completed_run_count": len(completed),
        "failed_run_count": sum(1 for row in summaries if row.get("status") != "ok"),
        "changed_from_baseline_count": changed_from_baseline_count,
        "baseline_loop_score": baseline_loop,
        "best_loop_score_run": None if best is None else best.get("name"),
        "best_loop_score": None if best is None else best.get("loop_score"),
        "best_loop_score_delta_from_baseline": best_loop_delta,
        "best_loop_score_reduction_ratio": best_loop_ratio,
        "recommended_run": None if best is None else best.get("name"),
        "recommendation_reason": _recommendation_reason(
            best,
            baseline_loop_score=baseline_loop,
        ),
        "recommended_config": recommended_config,
        "recommended_processor_kwargs": recommended_processor_kwargs,
        "recommended_sweep_cli_args": recommended_sweep_cli_args,
        "recommended_bridge_cli_args": recommended_bridge_cli_args,
        "recommended_cli_args": recommended_sweep_cli_args,
        "min_loop_score": min(loop_values) if loop_values else None,
        "max_loop_score": max(loop_values) if loop_values else None,
        "max_top_token_changed_count": (
            max(control_changes) if control_changes else None
        ),
        "top_runs": top_runs,
        "summaries": summaries,
    }


def summarize_zspace_generation_control_sweep_lines(
    report_or_path: str | Path | Mapping[str, object],
    *,
    top_n: int = 3,
) -> list[str]:
    """Render a compact text summary for a generation-control sweep."""

    summary = summarize_zspace_generation_control_sweep(
        report_or_path,
        top_n=top_n,
    )
    lines = [
        (
            "zspace_generation_control_sweep "
            f"status={summary.get('status')} "
            f"runs={summary.get('completed_run_count')}/{summary.get('run_count')} "
            f"changed={summary.get('changed_from_baseline_count')} "
            f"best={summary.get('best_loop_score_run')} "
            f"best_loop={summary.get('best_loop_score')} "
            f"recommend={summary.get('recommended_run')} "
            f"loop_delta={summary.get('best_loop_score_delta_from_baseline')}"
        )
    ]
    for row in summary.get("top_runs", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "zspace_generation_control_top "
            f"rank={row.get('rank')} "
            f"name={row.get('name')} "
            f"loop={row.get('loop_score')} "
            f"changed={row.get('changed_from_baseline')} "
            f"top_changes={row.get('control_top_token_changed_count')} "
            f"backend={row.get('control_backend')} "
            f"rs={row.get('config_repression_strength')} "
            f"ngram={row.get('config_ngram_size')}/"
            f"{row.get('config_ngram_window')}/"
            f"{row.get('config_ngram_repression_strength')} "
            f"ngram_hits={row.get('control_ngram_repressed_token_total')} "
            f"entropy_target={row.get('config_entropy_target')}"
        )
    return lines
