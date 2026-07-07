"""Hugging Face generation helpers backed by SpiralTorch Z-Space controls."""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "ZSpaceRepressionLogitsProcessor",
    "build_zspace_repression_logits_processor",
    "build_zspace_softmax_logits_processor",
    "load_zspace_generation_control_sweep",
    "summarize_zspace_generation_control_run",
    "summarize_zspace_generation_control_sweep",
    "summarize_zspace_generation_control_sweep_lines",
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
        "control_native_error": control.get("native_error"),
        "config_top_k": _safe_number(config.get("top_k")),
        "config_curvature": _safe_number(config.get("curvature")),
        "config_temperature": _safe_number(config.get("temperature")),
        "config_entropy_target": _safe_number(config.get("entropy_target")),
        "config_entropy_gain": _safe_number(config.get("entropy_gain")),
        "config_repression_window": _safe_number(config.get("repression_window")),
        "config_repression_strength": _safe_number(
            config.get("repression_strength")
        ),
        "config_last_token_repression": _safe_number(
            config.get("last_token_repression")
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
        "entropy_gain": row.get("config_entropy_gain"),
        "repression_window": row.get("config_repression_window"),
        "repression_strength": row.get("config_repression_strength"),
        "last_token_repression": row.get("config_last_token_repression"),
    }
    return {key: value for key, value in fields.items() if value is not None}


def _cli_value(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _recommended_cli_args(config: Mapping[str, object] | None) -> list[str]:
    if not config:
        return []
    flag_map = [
        ("top_k", "--zspace-top-k-values"),
        ("curvature", "--zspace-curvature-values"),
        ("temperature", "--zspace-temperature-values"),
        ("entropy_target", "--zspace-entropy-target-values"),
        ("entropy_gain", "--zspace-entropy-gain-values"),
        ("repression_window", "--repression-window-values"),
        ("repression_strength", "--repression-strength-values"),
        ("last_token_repression", "--last-token-repression-values"),
    ]
    args: list[str] = []
    for key, flag in flag_map:
        if key not in config:
            continue
        args.extend([flag, _cli_value(config[key])])
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
        "recommended_cli_args": _recommended_cli_args(recommended_config),
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
            f"entropy_target={row.get('config_entropy_target')}"
        )
    return lines
