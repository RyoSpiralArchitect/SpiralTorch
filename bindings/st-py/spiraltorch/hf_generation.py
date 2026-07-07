"""Hugging Face generation helpers backed by SpiralTorch Z-Space controls."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "ZSpaceRepressionLogitsProcessor",
    "build_zspace_repression_logits_processor",
    "build_zspace_softmax_logits_processor",
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
            token_counts, last_token = self._recent_token_counts(input_row)
            adjusted_values, repression = self._apply_repression(
                row_values,
                row_indices,
                token_counts,
                last_token,
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
                "mask_non_top_k": self.mask_non_top_k,
                "rows": rows,
            }
        )
        return processed

    def report(self, *, limit: int | None = None) -> dict[str, object]:
        reports = self._reports[-limit:] if limit is not None else list(self._reports)
        rows = [
            row
            for report in reports
            for row in report.get("rows", [])
            if isinstance(row, Mapping)
        ]
        changed = sum(1 for row in rows if row.get("top_token_changed") is True)
        temperatures = [
            float(row["temperature"])
            for row in rows
            if isinstance(row.get("temperature"), (int, float))
        ]
        entropies = [
            float(row["entropy"])
            for row in rows
            if isinstance(row.get("entropy"), (int, float))
        ]
        return {
            "row_type": "zspace_repression_generation_control",
            "status": "ok" if reports else "unused",
            "processor": "ZSpaceRepressionLogitsProcessor",
            "calls": len(reports),
            "rows": rows,
            "top_token_changed_count": changed,
            "temperature_min": min(temperatures) if temperatures else None,
            "temperature_max": max(temperatures) if temperatures else None,
            "entropy_min": min(entropies) if entropies else None,
            "entropy_max": max(entropies) if entropies else None,
            "native_error": self._native_error,
        }

    def reset_report(self) -> None:
        self._reports.clear()

    def _recent_token_counts(self, input_row: Any) -> tuple[dict[int, int], int | None]:
        if self.repression_window <= 0:
            return {}, None
        tokens = _tensor_row_to_ints(input_row)
        recent = tokens[-self.repression_window :]
        counts: dict[int, int] = {}
        for token in recent:
            counts[token] = counts.get(token, 0) + 1
        return counts, (recent[-1] if recent else None)

    def _apply_repression(
        self,
        values: Sequence[float],
        indices: Sequence[int],
        counts: Mapping[int, int],
        last_token: int | None,
    ) -> tuple[list[float], dict[str, object]]:
        if not values:
            return [], {"repressed_token_count": 0, "max_repression": 0.0}
        adjusted: list[float] = []
        penalties: list[float] = []
        for value, token in zip(values, indices):
            count = counts.get(int(token), 0)
            penalty = self.repression_strength * float(count)
            if last_token is not None and int(token) == int(last_token):
                penalty += self.last_token_repression
            adjusted.append(float(value) - penalty)
            penalties.append(penalty)
        return adjusted, {
            "repressed_token_count": sum(1 for penalty in penalties if penalty > 0.0),
            "max_repression": max(penalties) if penalties else 0.0,
        }

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
