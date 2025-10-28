"""Inference helpers that reconstruct Z-space metrics from partial observations."""

from __future__ import annotations

import inspect
import math
import sys
from dataclasses import dataclass
from collections.abc import Iterable, Mapping as MappingABC, MutableMapping
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Sequence, Union
from types import MappingProxyType

from ._zspace_aliases import (
    ZSPACE_METRIC_ALIASES,
    PRIMARY_ZSPACE_METRIC_ALIASES,
)

__all__ = [
    "ZMetrics",
    "ZSpaceDecoded",
    "ZSpaceInference",
    "ZSpacePosterior",
    "ZSpacePartialBundle",
    "ZSpaceTelemetryFrame",
    "ZSpaceInferenceRuntime",
    "ZSpaceInferencePipeline",
    "inference_to_mapping",
    "inference_to_zmetrics",
    "prepare_trainer_step_payload",
    "decode_zspace_embedding",
    "infer_from_partial",
    "infer_with_partials",
    "compile_inference",
    "blend_zspace_partials",
    "canvas_partial_from_snapshot",
    "canvas_coherence_partial",
    "elliptic_partial_from_telemetry",
    "infer_canvas_snapshot",
    "infer_canvas_transformer",
    "coherence_partial_from_diagnostics",
    "infer_coherence_diagnostics",
    "infer_coherence_from_sequencer",
    "infer_canvas_with_coherence",
    "weights_partial_from_dlpack",
    "weights_partial_from_compat",
    "infer_weights_from_dlpack",
    "infer_weights_from_compat",
]


@dataclass(slots=True)
class ZMetrics:
    """Typed metrics container fed into :class:`ZSpaceTrainer`."""

    speed: float
    memory: float
    stability: float
    gradient: Sequence[float] | None = None
    drs: float = 0.0
    telemetry: Mapping[str, float] | None = None


_PRIMARY_ALIAS_GROUPS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        canonical: tuple(
            alias for alias, target in PRIMARY_ZSPACE_METRIC_ALIASES.items() if target == canonical
        )
        for canonical in {"speed", "memory", "stability", "drs", "gradient"}
    }
)


_METRIC_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "speed": "speed",
        "velocity": "speed",
        "mem": "memory",
        "memory": "memory",
        "stab": "stability",
        "stability": "stability",
        "frac": "frac",
        "frac_reg": "frac",
        "fractality": "frac",
        "drs": "drs",
        "drift": "drs",
        "gradient": "gradient",
        "canvas_energy": "canvas_energy",
        "canvas_mean": "canvas_mean",
        "canvas_peak": "canvas_peak",
        "canvas_balance": "canvas_balance",
        "canvas_l1": "canvas_l1",
        "canvas_l2": "canvas_l2",
        "canvas_linf": "canvas_linf",
        "canvas_pixels": "canvas_pixels",
        "canvas_patch_energy": "canvas_patch_energy",
        "canvas_patch_mean": "canvas_patch_mean",
        "canvas_patch_peak": "canvas_patch_peak",
        "canvas_patch_pixels": "canvas_patch_pixels",
        "canvas_patch_balance": "canvas_patch_balance",
        "hypergrad_norm": "hypergrad_norm",
        "hypergrad_balance": "hypergrad_balance",
        "hypergrad_mean": "hypergrad_mean",
        "hypergrad_l1": "hypergrad_l1",
        "hypergrad_l2": "hypergrad_l2",
        "hypergrad_linf": "hypergrad_linf",
        "realgrad_norm": "realgrad_norm",
        "realgrad_balance": "realgrad_balance",
        "realgrad_mean": "realgrad_mean",
        "realgrad_l1": "realgrad_l1",
        "realgrad_l2": "realgrad_l2",
        "realgrad_linf": "realgrad_linf",
        "coherence_mean": "coherence_mean",
        "coherence_entropy": "coherence_entropy",
        "coherence_energy_ratio": "coherence_energy_ratio",
        "coherence_z_bias": "coherence_z_bias",
        "coherence_fractional_order": "coherence_fractional_order",
        "coherence_channels": "coherence_channels",
        "coherence_preserved": "coherence_preserved",
        "coherence_discarded": "coherence_discarded",
        "coherence_dominant": "coherence_dominant",
        "coherence_peak": "coherence_peak",
        "coherence_weight_entropy": "coherence_weight_entropy",
        "coherence_response_peak": "coherence_response_peak",
        "coherence_response_mean": "coherence_response_mean",
        "coherence_strength": "coherence_strength",
        "coherence_prosody": "coherence_prosody",
        "coherence_articulation": "coherence_articulation",
        "import_l1": "import_l1",
        "import_l2": "import_l2",
        "import_linf": "import_linf",
        "import_mean": "import_mean",
        "import_variance": "import_variance",
        "import_energy": "import_energy",
        "import_count": "import_count",
        "import_amplitude": "import_amplitude",
        "import_balance": "import_balance",
        "import_focus": "import_focus",
        "elliptic_curvature": "elliptic_curvature",
        "curvature_radius": "elliptic_curvature",
        "elliptic_curvature_radius": "elliptic_curvature",
        "elliptic_geodesic": "elliptic_geodesic",
        "geodesic_radius": "elliptic_geodesic",
        "elliptic_normalized": "elliptic_normalized",
        "normalized_radius": "elliptic_normalized",
        "elliptic_alignment": "elliptic_alignment",
        "spin_alignment": "elliptic_alignment",
        "elliptic_bias": "elliptic_bias",
        "normal_bias": "elliptic_bias",
        "elliptic_sheet_position": "elliptic_sheet_position",
        "sheet_position": "elliptic_sheet_position",
        "elliptic_sheet_index": "elliptic_sheet_index",
        "sheet_index": "elliptic_sheet_index",
        "elliptic_sheet_count": "elliptic_sheet_count",
        "sheet_count": "elliptic_sheet_count",
        "elliptic_sector": "elliptic_sector",
        "topological_sector": "elliptic_sector",
        "elliptic_homology": "elliptic_homology",
        "homology_index": "elliptic_homology",
        "elliptic_resonance": "elliptic_resonance",
        "resonance_heat": "elliptic_resonance",
        "elliptic_noise": "elliptic_noise",
        "noise_density": "elliptic_noise",
    }
)
_METRIC_ALIASES: Mapping[str, str] = ZSPACE_METRIC_ALIASES


def _normalise_metric_name(name: str) -> str:
    alias = _METRIC_ALIASES.get(name.lower())
    return alias if alias is not None else name.lower()


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_gradient(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, ZMetrics):
        return _coerce_gradient(value.gradient)
    if isinstance(value, MappingABC):
        value = value.values()
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        return None
    try:
        gradient = [float(entry) for entry in value]  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return gradient


def _flatten_telemetry_payload(payload: Any) -> dict[str, float]:
    if payload is None:
        return {}
    if isinstance(payload, ZSpaceTelemetryFrame):
        return dict(payload.payload)
    if isinstance(payload, MappingABC):
        return _flatten_telemetry(payload)
    if hasattr(payload, "payload"):
        inner = getattr(payload, "payload")
        if isinstance(inner, MappingABC):
            return dict(inner)
    if hasattr(payload, "as_dict") and callable(payload.as_dict):
        try:
            mapping = payload.as_dict()
        except Exception:  # pragma: no cover - defensive fallback
            return {}
        inner = mapping.get("payload") if isinstance(mapping, MappingABC) else None
        if isinstance(inner, MappingABC):
            return _flatten_telemetry(inner)
    return {}


def _normalise_metrics_mapping(
    source: Mapping[str, Any] | None,
) -> tuple[dict[str, float], list[float] | None]:
    if source is None:
        return {}, None
    metrics: dict[str, float] = {}
    gradient: list[float] | None = None
    for key, raw_value in source.items():
        canonical = _normalise_metric_name(str(key))
        if canonical == "gradient":
            candidate = _coerce_gradient(raw_value)
            if candidate is not None:
                gradient = candidate
            continue
        converted = _coerce_float(raw_value)
        if converted is not None:
            metrics[canonical] = converted
    return metrics, gradient


def _collect_inference_payload(
    inference: Any,
    *,
    prefer_applied: bool,
) -> tuple[dict[str, float], list[float] | None, dict[str, float] | None]:
    if isinstance(inference, ZMetrics):
        metrics = {
            "speed": float(inference.speed),
            "memory": float(inference.memory),
            "stability": float(inference.stability),
            "drs": float(inference.drs),
        }
        gradient = _coerce_gradient(inference.gradient)
        telemetry = dict(inference.telemetry) if inference.telemetry else None
        return metrics, gradient, telemetry

    if isinstance(inference, MappingABC):
        metrics, gradient = _normalise_metrics_mapping(inference)
        if gradient is not None:
            metrics.pop("gradient", None)
        return metrics, gradient, None

    base_metrics, gradient = _normalise_metrics_mapping(getattr(inference, "metrics", None))
    telemetry = None

    if prefer_applied:
        applied_metrics, applied_gradient = _normalise_metrics_mapping(
            getattr(inference, "applied", None)
        )
        if applied_metrics:
            base_metrics.update(applied_metrics)
        if applied_gradient is not None:
            gradient = applied_gradient

    attr_gradient = _coerce_gradient(getattr(inference, "gradient", None))
    if attr_gradient is not None:
        gradient = attr_gradient

    telemetry = _flatten_telemetry_payload(getattr(inference, "telemetry", None)) or None

    if gradient is not None and "gradient" in base_metrics:
        base_metrics.pop("gradient", None)

    return base_metrics, gradient, telemetry


def _resolve_primary(metrics: Mapping[str, float], canonical: str) -> float:
    value = metrics.get(canonical)
    if value is not None:
        return float(value)
    for alias in _PRIMARY_ALIAS_GROUPS.get(canonical, ()):  # type: ignore[arg-type]
        alias_value = metrics.get(_normalise_metric_name(alias))
        if alias_value is not None:
            return float(alias_value)
    return 0.0


def inference_to_mapping(
    inference: Any,
    *,
    prefer_applied: bool = True,
    canonical: bool = True,
    include_gradient: bool = True,
) -> dict[str, Any]:
    """Convert inference-like payloads into canonical metric dictionaries."""

    metrics, gradient, _ = _collect_inference_payload(
        inference, prefer_applied=prefer_applied
    )

    if canonical:
        resolved: dict[str, Any] = {
            _normalise_metric_name(key): float(value) for key, value in metrics.items()
        }
    else:
        resolved = dict(metrics)

    if include_gradient and gradient is not None:
        resolved["gradient"] = list(gradient)

    return resolved


def inference_to_zmetrics(
    inference: Any,
    *,
    prefer_applied: bool = True,
    include_telemetry: bool = False,
) -> ZMetrics:
    """Convert inference-like payloads into :class:`ZMetrics`."""

    metrics, gradient, telemetry = _collect_inference_payload(
        inference, prefer_applied=prefer_applied
    )

    telemetry_payload = telemetry if include_telemetry else None

    gradient_seq = gradient if gradient else None

    return ZMetrics(
        speed=_resolve_primary(metrics, "speed"),
        memory=_resolve_primary(metrics, "memory"),
        stability=_resolve_primary(metrics, "stability"),
        gradient=gradient_seq,
        drs=_resolve_primary(metrics, "drs"),
        telemetry=telemetry_payload,
    )


if TYPE_CHECKING:
    _PayloadMode = Union[
        None,
        str,
        Callable[["ZSpaceInference"], Any],
    ]
else:
    _PayloadMode = Union[
        None,
        str,
        Callable[[Any], Any],
    ]


def _annotation_mode(annotation: Any) -> str | None:
    if annotation is inspect._empty:
        return None
    if isinstance(annotation, str):
        lower = annotation.lower()
        if "zmetrics" in lower:
            return "zmetrics"
        if "mapping" in lower or "dict" in lower:
            return "mapping"
        if "inference" in lower:
            return "inference"
        return None

    origin = get_origin(annotation)
    if origin is not None:
        if origin in {dict, Dict, MutableMapping, MappingABC, Mapping}:
            return "mapping"
    if annotation in {Mapping, MutableMapping, dict}:
        return "mapping"
    if getattr(annotation, "__name__", "") == "ZMetrics":
        return "zmetrics"
    if getattr(annotation, "__qualname__", "") == "ZSpaceInference":
        return "inference"
    return None


def _detect_payload_mode(step: Callable[..., Any]) -> str | None:
    try:
        signature = inspect.signature(step)
    except (TypeError, ValueError):
        return None

    parameters = list(signature.parameters.values())
    if not parameters:
        return None

    parameter = parameters[0]
    annotation = parameter.annotation
    mode = _annotation_mode(annotation)
    if mode is not None:
        return mode

    annotations = getattr(step, "__annotations__", {})
    if annotations:
        alt = annotations.get(parameter.name)
        if alt is not None:
            mode = _annotation_mode(alt)
            if mode is not None:
                return mode
    return None


def prepare_trainer_step_payload(
    trainer: Any,
    inference: "ZSpaceInference",
    *,
    payload: _PayloadMode = None,
    prefer_applied: bool = True,
    canonical_mapping: bool = True,
) -> Any:
    """Prepare the payload passed to a trainer's ``step`` method."""

    step = getattr(trainer, "step", None)
    if not callable(step):
        raise TypeError("trainer must provide a callable 'step' method")

    if callable(payload):
        return payload(inference)

    mode: str | None
    if payload is None:
        mode = None
    else:
        mode = str(payload).lower()

    if mode in {None, "inference"}:
        chosen = "inference"
    elif mode in {"zmetrics", "metrics"}:
        chosen = "zmetrics"
    elif mode in {"mapping", "dict"}:
        chosen = "mapping"
    elif mode == "auto":
        detected = None
        for hint in (
            getattr(trainer, "__zspace_step_mode__", None),
            getattr(trainer, "zspace_step_mode", None),
        ):
            if hint is not None:
                detected = str(hint).lower()
                break
        if detected is None:
            detected = _detect_payload_mode(step)
        chosen = detected or "inference"
    else:
        raise ValueError(
            "payload must be one of None, 'inference', 'zmetrics', 'mapping', 'auto', or a callable"
        )

    if chosen == "zmetrics":
        return inference_to_zmetrics(inference, prefer_applied=prefer_applied, include_telemetry=True)
    if chosen == "mapping":
        return inference_to_mapping(
            inference,
            prefer_applied=prefer_applied,
            canonical=canonical_mapping,
            include_gradient=True,
        )
    return inference


def _softplus(value: float) -> float:
    if value > 20.0:
        return value
    if value < -20.0:
        return math.exp(value)
    return math.log1p(math.exp(value))


def _ensure_vector(z_state: Sequence[float]) -> list[float]:
    vector = [float(v) for v in z_state]
    if not vector:
        raise ValueError("z_state must contain at least one value")
    return vector


def _rfft(values: Sequence[float]) -> list[complex]:
    n = len(values)
    if n == 0:
        return []
    freq: list[complex] = []
    for k in range(n // 2 + 1):
        real = 0.0
        imag = 0.0
        for t, val in enumerate(values):
            angle = -2.0 * math.pi * k * t / max(1, n)
            real += val * math.cos(angle)
            imag += val * math.sin(angle)
        freq.append(complex(real, imag))
    return freq


def _fractional_energy(values: Sequence[float], alpha: float) -> float:
    spectrum = _rfft(values)
    n = len(spectrum)
    if n <= 1:
        return 0.0
    acc = 0.0
    for idx, coeff in enumerate(spectrum):
        omega = idx / max(1, n - 1)
        weight = omega ** (2.0 * alpha)
        acc += weight * abs(coeff) ** 2
    return acc / n


def _normalise_gradient(values: Sequence[float], length: int) -> list[float]:
    grad = [float(v) for v in values]
    if len(grad) < length:
        grad.extend(0.0 for _ in range(length - len(grad)))
    elif len(grad) > length:
        grad = grad[:length]
    scale = max(1.0, max(abs(v) for v in grad) if grad else 1.0)
    return [math.tanh(v / scale) for v in grad]


def _flatten_telemetry(payload: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in payload.items():
        label = f"{prefix}{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_telemetry(value, prefix=f"{label}."))
            continue
        try:
            flattened[label] = float(value)
        except (TypeError, ValueError):
            continue
    return flattened


def _normalise_telemetry_payload(
    payload: Mapping[str, Any] | "ZSpaceTelemetryFrame" | None,
) -> dict[str, float]:
    if payload is None:
        return {}
    if isinstance(payload, ZSpaceTelemetryFrame):
        return dict(payload.payload)
    if isinstance(payload, Mapping):
        return _flatten_telemetry(payload)
    raise TypeError("telemetry payloads must be provided as mappings")


def _merge_telemetry_payloads(
    *payloads: Mapping[str, Any] | "ZSpaceTelemetryFrame" | None,
) -> dict[str, float]:
    merged: dict[str, float] = {}
    for payload in payloads:
        if payload is None:
            continue
        mapping = _normalise_telemetry_payload(payload)
        if mapping:
            merged.update(mapping)
    return merged


_ELLIPTIC_SAMPLE_KEYS = {
    "curvature_radius",
    "geodesic_radius",
    "normalized_radius",
    "spin_alignment",
    "sheet_index",
    "sheet_position",
    "normal_bias",
    "rotor_transport",
}

_ELLIPTIC_ATTRIBUTE_NAMES = (
    "curvature_radius",
    "geodesic_radius",
    "normalized_radius",
    "spin_alignment",
    "sheet_index",
    "sheet_position",
    "normal_bias",
    "sheet_count",
    "topological_sector",
    "homology_index",
    "resonance_heat",
    "noise_density",
    "rotor_transport",
    "flow_vector",
    "lie_log",
)

_ELLIPTIC_METRIC_SOURCES = {
    "elliptic_curvature": "curvature_radius",
    "elliptic_geodesic": "geodesic_radius",
    "elliptic_normalized": "normalized_radius",
    "elliptic_alignment": "spin_alignment",
    "elliptic_bias": "normal_bias",
    "elliptic_sheet_position": "sheet_position",
    "elliptic_sheet_index": "sheet_index",
    "elliptic_sheet_count": "sheet_count",
    "elliptic_sector": "topological_sector",
    "elliptic_homology": "homology_index",
    "elliptic_resonance": "resonance_heat",
    "elliptic_noise": "noise_density",
}

_ELLIPTIC_VECTOR_CANDIDATES = (
    "rotor_transport",
    "flow_vector",
    "lie_log",
)


def _iter_elliptic_samples(candidate: Any) -> Iterable[Any]:
    stack = [candidate]
    while stack:
        current = stack.pop()
        if current is None:
            continue
        if isinstance(current, (str, bytes, bytearray)):
            continue
        if isinstance(current, Mapping):
            if any(key in current for key in _ELLIPTIC_SAMPLE_KEYS):
                yield current
            else:
                stack.extend(current.values())
            continue
        if hasattr(current, "curvature_radius") or hasattr(current, "as_dict"):
            yield current
            continue
        if isinstance(current, Iterable):
            stack.extend(current)


def _elliptic_payload_mapping(sample: Any) -> dict[str, Any]:
    if isinstance(sample, Mapping):
        return dict(sample)
    as_dict = getattr(sample, "as_dict", None)
    if callable(as_dict):
        try:
            payload = as_dict()
            if isinstance(payload, Mapping):
                return dict(payload)
        except Exception:
            pass
    payload: dict[str, Any] = {}
    for name in _ELLIPTIC_ATTRIBUTE_NAMES:
        if hasattr(sample, name):
            try:
                value = getattr(sample, name)
            except Exception:
                continue
            payload[name] = value
    return payload


def _coerce_float_list(candidate: Any) -> list[float]:
    if candidate is None:
        return []
    if hasattr(candidate, "tolist"):
        try:
            return _coerce_float_list(candidate.tolist())
        except Exception:
            pass
    if hasattr(candidate, "numpy"):
        try:
            return _coerce_float_list(candidate.numpy())
        except Exception:
            pass
    if isinstance(candidate, (bytes, bytearray, str)):
        return []
    if isinstance(candidate, Mapping):
        iterable = candidate.values()
    else:
        try:
            iterable = iter(candidate)
        except TypeError:
            return []
    result: list[float] = []
    for value in iterable:
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            return []
    return result


def _collect_bundle_telemetry(
    partials: Sequence[Mapping[str, Any] | ZSpacePartialBundle | None]
) -> dict[str, float]:
    payloads: list[Mapping[str, Any]] = []
    for partial in partials:
        if isinstance(partial, ZSpacePartialBundle):
            payload = partial.telemetry_payload()
            if payload:
                payloads.append(dict(payload))
    if not payloads:
        return {}
    return _merge_telemetry_payloads(*payloads)


def _flatten_values(candidate: Any) -> list[float]:
    if candidate is None:
        return []
    if hasattr(candidate, "tolist"):
        try:
            return _flatten_values(candidate.tolist())
        except Exception:
            pass
    if hasattr(candidate, "numpy"):
        try:
            return _flatten_values(candidate.numpy())
        except Exception:
            pass
    if isinstance(candidate, (bytes, bytearray, str)):
        return []
    if isinstance(candidate, Mapping):
        flattened: list[float] = []
        for value in candidate.values():
            flattened.extend(_flatten_values(value))
        return flattened
    if isinstance(candidate, Iterable):
        flattened: list[float] = []
        for value in candidate:
            flattened.extend(_flatten_values(value))
        return flattened
    try:
        return [float(candidate)]
    except (TypeError, ValueError):
        return []


def _vector_stats(values: Sequence[float]) -> dict[str, float]:
    data = [float(v) for v in values if not math.isnan(float(v))]
    if not data:
        return {
            "l1": 0.0,
            "l2": 0.0,
            "linf": 0.0,
            "mean": 0.0,
            "variance": 0.0,
            "energy": 0.0,
            "count": 0.0,
            "amplitude": 0.0,
            "positive": 0.0,
            "negative": 0.0,
            "balance": 0.0,
            "focus": 0.0,
        }
    n = len(data)
    l1 = sum(abs(value) for value in data)
    energy = sum(value * value for value in data)
    l2 = math.sqrt(energy)
    linf = max(abs(value) for value in data)
    mean = sum(data) / n
    variance = sum((value - mean) ** 2 for value in data) / n
    amplitude = max(data) - min(data)
    positive = sum(value for value in data if value > 0.0)
    negative = -sum(value for value in data if value < 0.0)
    balance = (positive - negative) / (positive + negative + 1e-9)
    focus = math.tanh(balance * 1.5)
    return {
        "l1": l1,
        "l2": l2,
        "linf": linf,
        "mean": mean,
        "variance": variance,
        "energy": energy,
        "count": float(n),
        "amplitude": amplitude,
        "positive": positive,
        "negative": negative,
        "balance": balance,
        "focus": focus,
    }


def _materialise_imported_weights(candidate: Any) -> list[float]:
    values = _flatten_values(candidate)
    if values:
        return values
    dlpack_capsule = None
    if hasattr(candidate, "__dlpack__"):
        try:
            dlpack_capsule = candidate.__dlpack__()
        except Exception:
            dlpack_capsule = None
    elif hasattr(candidate, "to_dlpack"):
        try:
            dlpack_capsule = candidate.to_dlpack()
        except Exception:
            dlpack_capsule = None
    if dlpack_capsule is not None:
        tensor = None
        module = sys.modules.get("spiraltorch")
        from_dlpack = getattr(module, "from_dlpack", None) if module else None
        if callable(from_dlpack):
            try:
                tensor = from_dlpack(dlpack_capsule)
            except Exception:
                tensor = None
        if tensor is None:
            try:
                import torch.utils.dlpack as torch_dlpack  # type: ignore

                tensor = torch_dlpack.from_dlpack(dlpack_capsule)
            except Exception:
                tensor = None
        if tensor is not None:
            values = _flatten_values(tensor)
            if values:
                return values
        fallback = _flatten_values(dlpack_capsule)
        if fallback:
            return fallback
    compat_module = sys.modules.get("spiraltorch.compat")
    if compat_module is not None:
        adaptors: list[Any] = []
        tensor_from = getattr(compat_module, "tensor_from", None)
        if callable(tensor_from):
            adaptors.append(tensor_from)
        for name in ("torch", "tensorflow", "jax", "numpy"):
            adapter = getattr(compat_module, name, None)
            for attr in ("to_tensor", "to_spiraltorch", "as_tensor", "tensor_from"):
                fn = getattr(adapter, attr, None)
                if callable(fn):
                    adaptors.append(fn)
        for adaptor in adaptors:
            try:
                tensor = adaptor(candidate)
            except Exception:
                continue
            values = _flatten_values(tensor)
            if values:
                return values
    return []


@dataclass(frozen=True)
class ZSpaceTelemetryFrame:
    """Structured PSI telemetry summary available during inference."""

    payload: Mapping[str, float]
    mean: float
    variance: float
    amplitude: float
    energy: float
    balance: float
    focus: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "payload": dict(self.payload),
            "mean": self.mean,
            "variance": self.variance,
            "amplitude": self.amplitude,
            "energy": self.energy,
            "balance": self.balance,
            "focus": self.focus,
        }


def _summarise_telemetry(
    telemetry: Mapping[str, Any] | ZSpaceTelemetryFrame | None,
) -> ZSpaceTelemetryFrame | None:
    if telemetry is None:
        return None
    if isinstance(telemetry, ZSpaceTelemetryFrame):
        return telemetry
    if not isinstance(telemetry, Mapping):
        raise TypeError("telemetry payloads must be provided as mappings")
    flattened = _flatten_telemetry(telemetry)
    if not flattened:
        return ZSpaceTelemetryFrame(
            MappingProxyType({}), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
    values = list(flattened.values())
    stats = _vector_stats(values)
    return ZSpaceTelemetryFrame(
        MappingProxyType(dict(flattened)),
        mean=stats["mean"],
        variance=stats["variance"],
        amplitude=stats["amplitude"],
        energy=stats["energy"],
        balance=stats["balance"],
        focus=stats["focus"],
    )


def _weights_partial_from_values(
    values: Sequence[float],
    *,
    bundle_weight: float,
    origin: str,
    weight_gain: float,
    stability_gain: float,
    focus_gain: float,
    telemetry_prefix: str,
    extra_telemetry: Mapping[str, Any] | None = None,
) -> ZSpacePartialBundle:
    stats = _vector_stats(values)
    count = max(1.0, stats["count"] or 1.0)
    amplitude = max(1e-9, stats["amplitude"] + stats["energy"] / count)
    weight_gain = max(0.0, float(weight_gain))
    stability_gain = max(0.0, float(stability_gain))
    focus_gain = max(0.0, float(focus_gain))
    memory = math.tanh(weight_gain * (stats["l2"] / count))
    speed = math.tanh(stats["mean"] + focus_gain * stats["balance"])
    stability = math.tanh(stability_gain * (1.0 - stats["variance"] / amplitude))
    frac = math.tanh(stats["linf"] / (stats["l2"] + 1e-9))
    drs = math.tanh(stats["balance"])
    partial: dict[str, float] = {
        "speed": speed,
        "memory": memory,
        "stability": stability,
        "frac": frac,
        "drs": drs,
        "import_l1": stats["l1"],
        "import_l2": stats["l2"],
        "import_linf": stats["linf"],
        "import_mean": stats["mean"],
        "import_variance": stats["variance"],
        "import_energy": stats["energy"],
        "import_count": stats["count"],
        "import_amplitude": stats["amplitude"],
        "import_balance": stats["balance"],
        "import_focus": stats["focus"],
    }
    prefix = telemetry_prefix or "psi"
    telemetry_map: dict[str, float] = {
        f"{prefix}.mean": stats["mean"],
        f"{prefix}.variance": stats["variance"],
        f"{prefix}.energy": stats["energy"],
        f"{prefix}.amplitude": stats["amplitude"],
        f"{prefix}.balance": stats["balance"],
        f"{prefix}.focus": stats["focus"],
        f"{prefix}.count": stats["count"],
    }
    if extra_telemetry:
        telemetry_map.update(_flatten_telemetry(extra_telemetry))
    return ZSpacePartialBundle(
        partial,
        weight=max(0.0, float(bundle_weight)),
        origin=origin,
        telemetry=telemetry_map,
    )


@dataclass(frozen=True)
class ZSpacePartialBundle:
    """Container describing a partial observation and its relative weight."""

    metrics: Mapping[str, Any]
    weight: float = 1.0
    origin: str | None = None
    telemetry: Mapping[str, Any] | None = None

    def resolved(self) -> dict[str, Any]:
        """Return the canonicalised metric mapping."""

        return _canonicalise_inputs(self.metrics)

    def telemetry_payload(self) -> Mapping[str, Any] | None:
        """Return a copy of any telemetry payload attached to the bundle."""

        if self.telemetry is None:
            return None
        if not isinstance(self.telemetry, Mapping):
            raise TypeError("telemetry payloads must be mappings")
        return MappingProxyType(dict(self.telemetry))


def _canonicalise_inputs(partial: Mapping[str, Any] | None) -> dict[str, Any]:
    if partial is None:
        return {}
    if not isinstance(partial, Mapping):
        raise TypeError("partial observations must be provided as a mapping")
    resolved: dict[str, Any] = {}
    for key, value in partial.items():
        canonical = _METRIC_ALIASES.get(key.lower())
        if canonical is None:
            raise KeyError(f"unknown metric '{key}'")
        resolved[canonical] = value
    return resolved


def _ensure_iterable(values: Any) -> list[float]:
    if isinstance(values, Mapping):
        values = values.values()
    if not isinstance(values, Iterable):
        return []
    result: list[float] = []
    for value in values:
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            continue
    return result


def _resolve_partial(
    partial: Mapping[str, Any] | ZSpacePartialBundle | None,
    *,
    fallback_weight: float = 1.0,
) -> tuple[dict[str, Any], float] | None:
    if partial is None:
        return None
    weight = fallback_weight
    if isinstance(partial, ZSpacePartialBundle):
        weight = float(partial.weight)
        mapping = partial.resolved()
    else:
        mapping = _canonicalise_inputs(partial)
    if weight <= 0.0:
        return None
    return mapping, weight


def blend_zspace_partials(
    partials: Sequence[Mapping[str, Any] | ZSpacePartialBundle | None],
    *,
    weights: Sequence[float] | None = None,
    strategy: str = "mean",
) -> dict[str, Any]:
    """Fuse several partial observations into a single mapping.

    Parameters
    ----------
    partials:
        Sequence of mappings or :class:`ZSpacePartialBundle` instances. ``None``
        entries are ignored.
    weights:
        Optional per-partial weighting that overrides the bundle's intrinsic
        weight. Negative or zero weights suppress that partial.
    strategy:
        Reduction strategy used when multiple partials define the same metric.
        Supported values are ``"mean"`` (default), ``"last"``, ``"max"`` and
        ``"min"``.
    """

    if not isinstance(partials, Sequence):
        raise TypeError("partials must be provided as a sequence")

    def _reduce(values: list[tuple[float, float]]) -> float:
        if not values:
            return 0.0
        if strategy == "last":
            return values[-1][0]
        if strategy == "max":
            return max(value for value, _ in values)
        if strategy == "min":
            return min(value for value, _ in values)
        # default: weighted mean
        total_weight = sum(weight for _, weight in values)
        if total_weight <= 0.0:
            return values[-1][0]
        return sum(value * weight for value, weight in values) / total_weight

    aggregated: dict[str, list[tuple[float, float]]] = {}
    gradients: list[tuple[list[float], float]] = []
    default_weight = 1.0
    for index, partial in enumerate(partials):
        weight_override = None
        if weights is not None:
            try:
                weight_override = float(weights[index])
            except (IndexError, TypeError, ValueError):
                weight_override = None
        resolved = _resolve_partial(
            partial, fallback_weight=weight_override if weight_override is not None else default_weight
        )
        if resolved is None:
            continue
        mapping, weight = resolved
        gradient = mapping.pop("gradient", None)
        for key, value in mapping.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            aggregated.setdefault(key, []).append((numeric, weight))
        if gradient is not None:
            gradients.append((_ensure_iterable(gradient), weight))

    merged = {key: _reduce(values) for key, values in aggregated.items()}

    if gradients:
        length = max((len(values) for values, _ in gradients), default=0)
        if length:
            total_weight = 0.0
            accumulator = [0.0] * length
            for values, weight in gradients:
                padded = list(values) + [0.0] * (length - len(values))
                for idx in range(length):
                    accumulator[idx] += padded[idx] * weight
                total_weight += weight
            if total_weight > 0.0:
                merged["gradient"] = [value / total_weight for value in accumulator]
            else:
                merged["gradient"] = gradients[-1][0]
    return merged


def _aggregate_values(values: Sequence[float], mode: str) -> float:
    if not values:
        return 0.0
    if mode == "max":
        return max(values)
    if mode == "min":
        return min(values)
    if mode == "last":
        return values[-1]
    if mode == "median":
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])
    if mode == "sum":
        return sum(values)
    return sum(values) / len(values)


def elliptic_partial_from_telemetry(
    telemetry: Any,
    *,
    bundle_weight: float = 1.0,
    origin: str | None = "elliptic",
    telemetry_prefix: str = "elliptic",
    aggregate: str = "mean",
    gradient_source: str = "rotor_transport",
    extra_telemetry: Mapping[str, Any] | None = None,
) -> ZSpacePartialBundle:
    """Convert elliptic telemetry samples into a :class:`ZSpacePartialBundle`."""

    samples = [
        mapping
        for sample in _iter_elliptic_samples(telemetry)
        for mapping in [_elliptic_payload_mapping(sample)]
        if mapping
    ]
    if not samples:
        raise ValueError("elliptic telemetry payload is empty")

    mode = aggregate.lower()
    if mode not in {"mean", "max", "min", "last", "median", "sum"}:
        raise ValueError(f"unsupported aggregate mode '{aggregate}' for elliptic telemetry")

    metric_values: dict[str, list[float]] = {key: [] for key in _ELLIPTIC_METRIC_SOURCES}
    gradient_vectors: list[list[float]] = []

    for payload in samples:
        for canonical, source in _ELLIPTIC_METRIC_SOURCES.items():
            value = payload.get(source)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            metric_values[canonical].append(numeric)

        vector_candidate: Any | None = payload.get(gradient_source) if gradient_source else None
        if vector_candidate is None:
            for candidate in _ELLIPTIC_VECTOR_CANDIDATES:
                vector_candidate = payload.get(candidate)
                if vector_candidate is not None:
                    break
        gradient = _coerce_float_list(vector_candidate)
        if gradient:
            gradient_vectors.append(gradient)

    partial: dict[str, Any] = {}
    summary_values: list[float] = []
    for key, values in metric_values.items():
        if not values:
            continue
        summary_values.extend(values)
        partial[key] = _aggregate_values(values, mode)

    if not partial:
        raise ValueError("no elliptic metrics could be derived from telemetry")

    if gradient_vectors:
        length = max(len(vector) for vector in gradient_vectors)
        if length > 0:
            accumulator = [0.0] * length
            for vector in gradient_vectors:
                for idx in range(length):
                    value = vector[idx] if idx < len(vector) else 0.0
                    accumulator[idx] += value
            count = len(gradient_vectors)
            partial["gradient"] = [accumulator[idx] / count for idx in range(length)]

    telemetry_sources: list[Mapping[str, Any]] = list(samples)
    if extra_telemetry is not None:
        telemetry_sources.append(extra_telemetry)

    telemetry_map: dict[str, float] = {}
    if telemetry_sources:
        merged = _merge_telemetry_payloads(*telemetry_sources)
        if telemetry_prefix:
            telemetry_map = {f"{telemetry_prefix}.{key}": value for key, value in merged.items()}
        else:
            telemetry_map = dict(merged)

    if summary_values:
        stats = _vector_stats(summary_values)
        prefix = telemetry_prefix or "elliptic"
        telemetry_map.setdefault(f"{prefix}.mean", stats["mean"])
        telemetry_map.setdefault(f"{prefix}.variance", stats["variance"])
        telemetry_map.setdefault(f"{prefix}.energy", stats["energy"])
        telemetry_map.setdefault(f"{prefix}.amplitude", stats["amplitude"])
        telemetry_map.setdefault(f"{prefix}.balance", stats["balance"])
        telemetry_map.setdefault(f"{prefix}.focus", stats["focus"])
        telemetry_map.setdefault(f"{prefix}.count", stats["count"])

    weight = max(0.0, float(bundle_weight))
    return ZSpacePartialBundle(
        partial,
        weight=weight,
        origin=origin,
        telemetry=telemetry_map or None,
    )


def _barycentric_from_metrics(metrics: Mapping[str, float]) -> tuple[float, float, float]:
    speed = float(metrics.get("speed", 0.0))
    memory = float(metrics.get("memory", 0.0))
    stability = float(metrics.get("stability", 0.0))
    weights = [_softplus(speed), _softplus(memory), _softplus(stability)]
    total = sum(weights)
    if total <= 0.0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return tuple(weight / total for weight in weights)


def _compute_gradient(values: Sequence[float]) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    grad: list[float] = []
    for idx in range(n):
        left = values[idx] - values[idx - 1] if idx > 0 else values[idx]
        right = values[idx + 1] - values[idx] if idx + 1 < n else -values[idx]
        grad.append(0.5 * (left + right))
    return grad


def _decode_metrics(z_state: Sequence[float], alpha: float) -> tuple[dict[str, float], list[float], tuple[float, float, float], float, float]:
    vector = _ensure_vector(z_state)
    n = len(vector)
    diffs = [vector[i + 1] - vector[i] for i in range(n - 1)]
    curvature = [vector[i + 1] - 2.0 * vector[i] + vector[i - 1] for i in range(1, n - 1)]
    mean_velocity = sum(abs(v) for v in diffs) / max(1, len(diffs))
    curvature_energy = sum(abs(v) for v in curvature) / max(1, len(curvature))
    l2 = math.sqrt(sum(value * value for value in vector))
    centre = sum(vector) / n
    frac_energy = _fractional_energy(vector, alpha)
    total_energy = sum(value * value for value in vector)
    gradient = _normalise_gradient(_compute_gradient(vector), n)
    speed = math.tanh(mean_velocity + 0.25 * l2 / max(1, n))
    memory = math.tanh(centre + 0.25 * total_energy / max(1, n))
    smoothness = 1.0 / (1.0 + curvature_energy)
    drift = 1.0 / (1.0 + mean_velocity)
    stability = math.tanh((smoothness + drift) * 1.5 - 1.0)
    spectrum = _rfft(vector)
    if len(spectrum) > 1:
        half = max(1, len(spectrum) // 2)
        high = sum(abs(coeff) ** 2 for coeff in spectrum[half:])
        low = sum(abs(coeff) ** 2 for coeff in spectrum[:half])
        drs = math.tanh((high - low) / (high + low + 1e-9))
    else:
        drs = 0.0
    frac = math.tanh(frac_energy / (total_energy + 1e-9))
    metrics = {
        "speed": speed,
        "memory": memory,
        "stability": stability,
        "frac": frac,
        "drs": drs,
    }
    barycentric = _barycentric_from_metrics(metrics)
    return metrics, gradient, barycentric, total_energy, frac_energy


@dataclass(frozen=True)
class ZSpaceDecoded:
    """Full set of metrics reconstructed from a latent Z vector."""

    z_state: tuple[float, ...]
    metrics: Mapping[str, float]
    gradient: tuple[float, ...]
    barycentric: tuple[float, float, float]
    energy: float
    frac_energy: float

    def as_dict(self) -> dict[str, Any]:
        data = {
            "z_state": list(self.z_state),
            "metrics": dict(self.metrics),
            "gradient": list(self.gradient),
            "barycentric": self.barycentric,
            "energy": self.energy,
            "frac_energy": self.frac_energy,
        }
        return data


@dataclass(frozen=True)
class ZSpaceInference:
    """Inference result after fusing partial observations with the decoded state."""

    metrics: Mapping[str, float]
    gradient: tuple[float, ...]
    barycentric: tuple[float, float, float]
    residual: float
    confidence: float
    prior: ZSpaceDecoded
    applied: Mapping[str, Any]
    telemetry: ZSpaceTelemetryFrame | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "gradient": list(self.gradient),
            "barycentric": self.barycentric,
            "residual": self.residual,
            "confidence": self.confidence,
            "applied": dict(self.applied),
            "prior": self.prior.as_dict(),
            "telemetry": None if self.telemetry is None else self.telemetry.as_dict(),
        }


class ZSpacePosterior:
    """Posterior over Z-space metrics conditioned on a latent state."""

    def __init__(self, z_state: Sequence[float], *, alpha: float = 0.35) -> None:
        self._z_state = tuple(_ensure_vector(z_state))
        self._alpha = max(1e-6, float(alpha))
        self._decoded: ZSpaceDecoded | None = None

    @property
    def z_state(self) -> list[float]:
        return list(self._z_state)

    @property
    def alpha(self) -> float:
        return self._alpha

    def decode(self) -> ZSpaceDecoded:
        if self._decoded is None:
            metrics, gradient, barycentric, energy, frac_energy = _decode_metrics(
                self._z_state, self._alpha
            )
            self._decoded = ZSpaceDecoded(
                z_state=self._z_state,
                metrics=MappingProxyType(dict(metrics)),
                gradient=tuple(gradient),
                barycentric=barycentric,
                energy=energy,
                frac_energy=frac_energy,
            )
        return self._decoded

    def project(
        self,
        partial: Mapping[str, Any] | None,
        *,
        smoothing: float = 0.35,
        telemetry: Mapping[str, Any] | ZSpaceTelemetryFrame | None = None,
    ) -> ZSpaceInference:
        decoded = self.decode()
        metrics = dict(decoded.metrics)
        gradient = list(decoded.gradient)
        applied: Dict[str, Any] = {}
        updates = _canonicalise_inputs(partial)
        if "gradient" in updates:
            gradient = _normalise_gradient(updates["gradient"], len(self._z_state))
            applied["gradient"] = list(gradient)
        for key, value in updates.items():
            if key == "gradient":
                continue
            metrics[key] = float(value)
            applied[key] = metrics[key]
        override_bary = _barycentric_from_metrics(metrics)
        base_bary = decoded.barycentric
        blend = max(0.0, min(1.0, float(smoothing)))
        barycentric = tuple(
            blend * base + (1.0 - blend) * override
            for base, override in zip(base_bary, override_bary)
        )
        norm = sum(barycentric)
        if norm <= 0.0:
            barycentric = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        else:
            barycentric = tuple(value / norm for value in barycentric)
        diff = 0.0
        for name, base_value in decoded.metrics.items():
            diff += (metrics[name] - base_value) ** 2
        residual = math.sqrt(diff / len(decoded.metrics)) if decoded.metrics else 0.0
        confidence = math.exp(-residual)
        telemetry_frame = _summarise_telemetry(telemetry)
        if telemetry_frame is not None:
            variance_damp = max(1.0, 1.0 + telemetry_frame.variance)
            residual = residual / variance_damp
            focus_gain = max(0.25, 1.0 + 0.25 * telemetry_frame.focus)
            energy_gain = min(1.5, 1.0 + 0.05 * telemetry_frame.energy)
            confidence = max(0.0, min(1.0, confidence * focus_gain * energy_gain))
        else:
            telemetry_frame = None
        return ZSpaceInference(
            metrics=MappingProxyType(dict(metrics)),
            gradient=tuple(gradient),
            barycentric=barycentric,
            residual=residual,
            confidence=confidence,
            prior=decoded,
            applied=MappingProxyType(dict(applied)),
            telemetry=telemetry_frame,
        )


class ZSpaceInferenceRuntime:
    """Stateful helper that incrementally fuses observations into a latent posterior."""

    def __init__(
        self,
        z_state: Sequence[float],
        *,
        alpha: float = 0.35,
        smoothing: float = 0.35,
        accumulate: bool = True,
        telemetry: Mapping[str, Any] | None = None,
    ) -> None:
        self._posterior = ZSpacePosterior(z_state, alpha=alpha)
        self._smoothing = float(smoothing)
        self._accumulate = bool(accumulate)
        self._cached: dict[str, Any] = {}
        self._telemetry: dict[str, float] = _merge_telemetry_payloads(telemetry)

    @property
    def posterior(self) -> ZSpacePosterior:
        """Return the underlying posterior instance."""

        return self._posterior

    @property
    def smoothing(self) -> float:
        """Smoothing factor used when mixing barycentric coordinates."""

        return self._smoothing

    @property
    def accumulate(self) -> bool:
        """Whether successive updates reuse previously supplied observations."""

        return self._accumulate

    @property
    def telemetry(self) -> Mapping[str, float]:
        """Return the currently cached telemetry payload."""

        return MappingProxyType(dict(self._telemetry))

    @property
    def cached_observations(self) -> Mapping[str, Any]:
        """Return the currently cached observation map."""

        return MappingProxyType(dict(self._cached))

    def clear(self) -> None:
        """Forget any cached observations."""

        self._cached.clear()

    def set_telemetry(
        self, telemetry: Mapping[str, Any] | ZSpaceTelemetryFrame | None
    ) -> None:
        """Replace the cached telemetry payload used during inference."""

        self._telemetry = _merge_telemetry_payloads(telemetry)

    def _merge(self, partial: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if partial is None:
            if not self._cached:
                return None
            return self._cached
        updates = _canonicalise_inputs(partial)
        if not self._accumulate:
            self._cached = {}
        if "gradient" in updates:
            gradient = updates.pop("gradient")
            if gradient is not None:
                self._cached["gradient"] = gradient
            else:
                self._cached.pop("gradient", None)
        for key, value in updates.items():
            self._cached[key] = value
        return self._cached

    def update(
        self,
        partial: Mapping[str, Any] | None = None,
        *,
        telemetry: Mapping[str, Any] | ZSpaceTelemetryFrame | None = None,
    ) -> ZSpaceInference:
        """Fuse *partial* with any cached observations and produce an inference."""

        if telemetry is not None:
            self._telemetry = _merge_telemetry_payloads(self._telemetry, telemetry)
        merged = self._merge(partial)
        payload = self._telemetry if self._telemetry else None
        return self._posterior.project(
            merged, smoothing=self._smoothing, telemetry=payload
        )

    def infer(
        self,
        partial: Mapping[str, Any] | None = None,
        *,
        telemetry: Mapping[str, Any] | ZSpaceTelemetryFrame | None = None,
    ) -> ZSpaceInference:
        """Alias for :meth:`update` to mirror the functional helpers."""

        return self.update(partial, telemetry=telemetry)


class ZSpaceInferencePipeline:
    """Composable pipeline that blends heterogeneous partials before inference."""

    def __init__(
        self,
        z_state: Sequence[float],
        *,
        alpha: float = 0.35,
        smoothing: float = 0.35,
        strategy: str = "mean",
        telemetry: Mapping[str, Any] | None = None,
    ) -> None:
        self._runtime = ZSpaceInferenceRuntime(
            z_state,
            alpha=alpha,
            smoothing=smoothing,
            accumulate=False,
            telemetry=telemetry,
        )
        self._strategy = strategy
        self._partials: list[ZSpacePartialBundle] = []

    @property
    def strategy(self) -> str:
        """Return the blending strategy used for partial fusion."""

        return self._strategy

    @property
    def posterior(self) -> ZSpacePosterior:
        """Expose the underlying :class:`ZSpacePosterior`."""

        return self._runtime.posterior

    @property
    def smoothing(self) -> float:
        """Smoothing factor applied during barycentric blending."""

        return self._runtime.smoothing

    def add_partial(
        self,
        partial: Mapping[str, Any] | ZSpacePartialBundle,
        *,
        weight: float | None = None,
        origin: str | None = None,
        telemetry: Mapping[str, Any] | None = None,
    ) -> ZSpacePartialBundle:
        """Register a new partial observation to be included in the next inference."""

        if isinstance(partial, ZSpacePartialBundle):
            bundle = partial
        else:
            bundle = ZSpacePartialBundle(
                partial,
                weight=1.0 if weight is None else weight,
                origin=origin,
                telemetry=telemetry,
            )
        self._partials.append(bundle)
        return bundle

    def add_elliptic_telemetry(
        self,
        telemetry: Any,
        *,
        bundle_weight: float = 1.0,
        origin: str | None = "elliptic",
        telemetry_prefix: str = "elliptic",
        aggregate: str = "mean",
        gradient_source: str = "rotor_transport",
        extra_telemetry: Mapping[str, Any] | None = None,
    ) -> ZSpacePartialBundle:
        """Register elliptic telemetry samples as a partial observation."""

        bundle = elliptic_partial_from_telemetry(
            telemetry,
            bundle_weight=bundle_weight,
            origin=origin,
            telemetry_prefix=telemetry_prefix,
            aggregate=aggregate,
            gradient_source=gradient_source,
            extra_telemetry=extra_telemetry,
        )
        return self.add_partial(bundle)

    def add_elliptic_autograd(
        self,
        warp: Any,
        orientation: Any,
        *,
        bundle_weight: float = 1.0,
        origin: str | None = "elliptic",
        telemetry_prefix: str = "elliptic",
        aggregate: str = "mean",
        gradient_source: str = "rotor_transport",
        extra_telemetry: Mapping[str, Any] | None = None,
        return_features: bool = False,
    ) -> ZSpacePartialBundle | tuple[Any, ZSpacePartialBundle]:
        """Run the elliptic warp and queue its bundle for inference."""

        from .elliptic import elliptic_warp_partial

        result = elliptic_warp_partial(
            warp,
            orientation,
            bundle_weight=bundle_weight,
            origin=origin,
            telemetry_prefix=telemetry_prefix,
            aggregate=aggregate,
            gradient_source=gradient_source,
            extra_telemetry=extra_telemetry,
            return_features=return_features,
        )
        if return_features:
            features, bundle = result
            self.add_partial(bundle)
            return features, bundle
        return self.add_partial(result)

    def add_canvas_snapshot(self, snapshot: Any, **kwargs: Any) -> ZSpacePartialBundle:
        """Derive and register metrics from a Canvas snapshot."""

        partial = canvas_partial_from_snapshot(snapshot, **kwargs)
        return self.add_partial(partial, origin="canvas")

    def add_coherence_diagnostics(
        self, diagnostics: Any, **kwargs: Any
    ) -> ZSpacePartialBundle:
        """Derive and register metrics from coherence diagnostics."""

        partial = coherence_partial_from_diagnostics(diagnostics, **kwargs)
        return self.add_partial(partial, origin="coherence")

    def add_dlpack_weights(self, weights: Any, **kwargs: Any) -> ZSpacePartialBundle:
        """Register DLPack-imported weights as a partial observation."""

        bundle = weights_partial_from_dlpack(weights, **kwargs)
        return self.add_partial(bundle)

    def add_compat_weights(
        self, weights: Any, *, adapter: str | None = None, **kwargs: Any
    ) -> ZSpacePartialBundle:
        """Register compat-imported weights as a partial observation."""

        bundle = weights_partial_from_compat(weights, adapter=adapter, **kwargs)
        return self.add_partial(bundle)

    def clear(self) -> None:
        """Discard any buffered partial observations."""

        self._partials.clear()

    def set_telemetry(
        self, telemetry: Mapping[str, Any] | ZSpaceTelemetryFrame | None
    ) -> None:
        """Forward telemetry to the underlying runtime."""

        self._runtime.set_telemetry(telemetry)

    def infer(
        self,
        *,
        strategy: str | None = None,
        weights: Sequence[float] | None = None,
        clear: bool = True,
        telemetry: Mapping[str, Any] | ZSpaceTelemetryFrame | None = None,
    ) -> ZSpaceInference:
        """Blend registered partials and compute the Z-space inference."""

        chosen_strategy = strategy or self._strategy
        blended = blend_zspace_partials(
            self._partials, strategy=chosen_strategy, weights=weights
        )
        bundle_telemetry = _collect_bundle_telemetry(self._partials)
        merged_telemetry = _merge_telemetry_payloads(bundle_telemetry, telemetry)
        inference = self._runtime.update(
            blended,
            telemetry=merged_telemetry if merged_telemetry else None,
        )
        if clear:
            self.clear()
        return inference

    def infer_and_step(
        self,
        trainer: Any,
        *,
        strategy: str | None = None,
        weights: Sequence[float] | None = None,
        clear: bool = True,
        telemetry: Mapping[str, Any] | ZSpaceTelemetryFrame | None = None,
        prefer_applied: bool = True,
        adapter: Callable[[ZSpaceInference], Any] | None = None,
    ) -> tuple[ZSpaceInference, float]:
        """Run :meth:`infer` and immediately feed the result into a trainer.

        Args:
            trainer: Object exposing a ``step`` method.
            strategy: Optional blend strategy override.
            weights: Optional blend weights.
            clear: Whether to clear buffered partials after inference.
            telemetry: Additional telemetry forwarded to the runtime.
            prefer_applied: Forward preference for applied overrides when the
                trainer supports the ``prefer_applied`` keyword (e.g. the
                Python :class:`~spiraltorch.ZSpaceTrainer`).
            adapter: Optional callable that receives the inference result and
                returns the payload passed into ``trainer.step``.
        """

        step = getattr(trainer, "step", None)
        if not callable(step):
            raise TypeError("trainer must provide a callable 'step' method")

        inference = self.infer(
            strategy=strategy,
            weights=weights,
            clear=clear,
            telemetry=telemetry,
        )
        payload = adapter(inference) if adapter is not None else inference

        call_kwargs: Dict[str, Any] = {}
        if adapter is None:
            try:
                signature = inspect.signature(step)
            except (TypeError, ValueError):
                signature = None
            if signature is not None and "prefer_applied" in signature.parameters:
                call_kwargs["prefer_applied"] = prefer_applied

        try:
            loss = step(payload, **call_kwargs)
        except TypeError as exc:
            if call_kwargs and "unexpected keyword" in str(exc).lower():
                loss = step(payload)
            else:
                raise exc
        return inference, float(loss)


def decode_zspace_embedding(z_state: Sequence[float], *, alpha: float = 0.35) -> ZSpaceDecoded:
    """Decode latent coordinates into a structured metric bundle."""

    return ZSpacePosterior(z_state, alpha=alpha).decode()


def infer_from_partial(
    z_state: Sequence[float],
    partial: Mapping[str, Any] | None,
    *,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    telemetry: Mapping[str, Any] | ZSpaceTelemetryFrame | None = None,
) -> ZSpaceInference:
    """Fuse partial metric observations with a latent state to complete Z-space inference."""

    posterior = ZSpacePosterior(z_state, alpha=alpha)
    return posterior.project(partial, smoothing=smoothing, telemetry=telemetry)


def infer_with_partials(
    z_state: Sequence[float],
    *partials: Mapping[str, Any] | ZSpacePartialBundle | None,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    strategy: str = "mean",
    weights: Sequence[float] | None = None,
    telemetry: Mapping[str, Any] | ZSpaceTelemetryFrame | None = None,
) -> ZSpaceInference:
    """Infer Z-space metrics from multiple partial observations."""

    blended = blend_zspace_partials(partials, weights=weights, strategy=strategy)
    bundle_telemetry = _collect_bundle_telemetry(partials)
    merged_telemetry = _merge_telemetry_payloads(bundle_telemetry, telemetry)
    return infer_from_partial(
        z_state,
        blended,
        alpha=alpha,
        smoothing=smoothing,
        telemetry=merged_telemetry if merged_telemetry else None,
    )


def weights_partial_from_dlpack(
    weights: Any,
    *,
    bundle_weight: float = 1.0,
    label: str | None = None,
    weight_gain: float = 1.25,
    stability_gain: float = 1.5,
    focus_gain: float = 1.0,
    telemetry_prefix: str = "psi",
    telemetry: Mapping[str, Any] | None = None,
) -> ZSpacePartialBundle:
    """Derive a partial bundle from weights imported via DLPack-compatible objects."""

    values = _materialise_imported_weights(weights)
    origin = label or "dlpack"
    return _weights_partial_from_values(
        values,
        bundle_weight=bundle_weight,
        origin=origin,
        weight_gain=weight_gain,
        stability_gain=stability_gain,
        focus_gain=focus_gain,
        telemetry_prefix=telemetry_prefix,
        extra_telemetry=telemetry,
    )


def weights_partial_from_compat(
    weights: Any,
    *,
    adapter: str | None = None,
    bundle_weight: float = 1.0,
    label: str | None = None,
    weight_gain: float = 1.25,
    stability_gain: float = 1.5,
    focus_gain: float = 1.0,
    telemetry_prefix: str = "psi",
    telemetry: Mapping[str, Any] | None = None,
) -> ZSpacePartialBundle:
    """Derive a partial bundle from compat-imported weights."""

    values = _materialise_imported_weights(weights)
    origin = label or (f"compat:{adapter}" if adapter else "compat")
    prefix = telemetry_prefix
    if adapter:
        prefix = f"{telemetry_prefix}.{adapter}" if telemetry_prefix else adapter
    return _weights_partial_from_values(
        values,
        bundle_weight=bundle_weight,
        origin=origin,
        weight_gain=weight_gain,
        stability_gain=stability_gain,
        focus_gain=focus_gain,
        telemetry_prefix=prefix,
        extra_telemetry=telemetry,
    )


def infer_weights_from_dlpack(
    z_state: Sequence[float],
    weights: Any,
    *,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    weight_gain: float = 1.25,
    stability_gain: float = 1.5,
    focus_gain: float = 1.0,
    telemetry_prefix: str = "psi",
    telemetry: Mapping[str, Any] | ZSpaceTelemetryFrame | None = None,
    label: str | None = None,
    bundle_weight: float = 1.0,
) -> ZSpaceInference:
    """Run inference directly from DLPack-imported weights."""

    extra = None
    if isinstance(telemetry, ZSpaceTelemetryFrame):
        extra = telemetry.payload
    elif isinstance(telemetry, Mapping):
        extra = telemetry
    bundle = weights_partial_from_dlpack(
        weights,
        bundle_weight=bundle_weight,
        label=label,
        weight_gain=weight_gain,
        stability_gain=stability_gain,
        focus_gain=focus_gain,
        telemetry_prefix=telemetry_prefix,
        telemetry=extra,
    )
    payload = bundle.telemetry_payload()
    merged = _merge_telemetry_payloads(payload, telemetry)
    return infer_from_partial(
        z_state,
        bundle.resolved(),
        alpha=alpha,
        smoothing=smoothing,
        telemetry=merged if merged else None,
    )


def infer_weights_from_compat(
    z_state: Sequence[float],
    weights: Any,
    *,
    adapter: str | None = None,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    weight_gain: float = 1.25,
    stability_gain: float = 1.5,
    focus_gain: float = 1.0,
    telemetry_prefix: str = "psi",
    telemetry: Mapping[str, Any] | ZSpaceTelemetryFrame | None = None,
    label: str | None = None,
    bundle_weight: float = 1.0,
) -> ZSpaceInference:
    """Run inference from weights sourced via the compat bridges."""

    extra = None
    if isinstance(telemetry, ZSpaceTelemetryFrame):
        extra = telemetry.payload
    elif isinstance(telemetry, Mapping):
        extra = telemetry
    bundle = weights_partial_from_compat(
        weights,
        adapter=adapter,
        bundle_weight=bundle_weight,
        label=label,
        weight_gain=weight_gain,
        stability_gain=stability_gain,
        focus_gain=focus_gain,
        telemetry_prefix=telemetry_prefix,
        telemetry=extra,
    )
    payload = bundle.telemetry_payload()
    merged = _merge_telemetry_payloads(payload, telemetry)
    return infer_from_partial(
        z_state,
        bundle.resolved(),
        alpha=alpha,
        smoothing=smoothing,
        telemetry=merged if merged else None,
    )


def compile_inference(
    fn=None,
    *,
    alpha: float = 0.35,
    smoothing: float = 0.35,
):
    """Wrap a callable so it automatically feeds its output into Z-space inference.

    The returned callable expects a latent ``z_state`` as its first argument and
    delegates any additional positional and keyword arguments to *fn*.  The
    original callable must return either ``None`` (indicating no new
    observations) or a mapping of partial observations compatible with
    :func:`infer_from_partial`.

    The helper can be used directly::

        def collect_metrics(data):
            return {"speed": data["speed"]}

        infer_speed = compile_inference(collect_metrics)
        result = infer_speed(z_state, sample)

    or as a decorator::

        @compile_inference(alpha=0.5)
        def analyze(sample):
            return {"memory": sample.mean()}

    """

    if fn is None:
        return lambda actual: compile_inference(
            actual, alpha=alpha, smoothing=smoothing
        )

    if not callable(fn):
        raise TypeError("compile_inference expects a callable or to be used as a decorator")

    def _compiled(
        z_state: Sequence[float],
        *args,
        telemetry: Mapping[str, Any] | ZSpaceTelemetryFrame | None = None,
        **kwargs,
    ) -> ZSpaceInference:
        partial = fn(*args, **kwargs)
        if partial is not None and not isinstance(partial, Mapping):
            raise TypeError("compiled inference callable must return a mapping or None")
        return infer_from_partial(
            z_state,
            partial,
            alpha=alpha,
            smoothing=smoothing,
            telemetry=telemetry,
        )

    _compiled.__name__ = getattr(fn, "__name__", "compiled_inference")
    _compiled.__doc__ = fn.__doc__
    return _compiled


def _maybe_call(value: Any) -> Any:
    if callable(value):
        try:
            return value()
        except TypeError:
            return value
    return value


def _matrix_stats(matrix: Any) -> dict[str, float]:
    matrix = _maybe_call(matrix)
    if matrix is None or not isinstance(matrix, Iterable):
        return {"l1": 0.0, "l2": 0.0, "linf": 0.0, "mean": 0.0, "count": 0.0}
    flat: list[float] = []
    for row in matrix:
        row = _maybe_call(row)
        if row is None or not isinstance(row, Iterable):
            continue
        for value in row:
            try:
                flat.append(float(value))
            except (TypeError, ValueError):
                continue
    if not flat:
        return {"l1": 0.0, "l2": 0.0, "linf": 0.0, "mean": 0.0, "count": 0.0}
    l1 = sum(abs(value) for value in flat)
    l2 = math.sqrt(sum(value * value for value in flat))
    linf = max(abs(value) for value in flat)
    mean = sum(flat) / len(flat)
    return {"l1": l1, "l2": l2, "linf": linf, "mean": mean, "count": float(len(flat))}


def _merge_summary(stats: dict[str, float], summary: Mapping[str, Any] | None) -> dict[str, float]:
    if not isinstance(summary, Mapping):
        return stats
    merged = dict(stats)
    for key, value in summary.items():
        try:
            merged[key] = float(value)
        except (TypeError, ValueError):
            continue
    return merged


def _canvas_snapshot_stats(snapshot: Any) -> dict[str, dict[str, float]]:
    canvas = _maybe_call(getattr(snapshot, "canvas", None))
    hypergrad = _maybe_call(getattr(snapshot, "hypergrad", None))
    realgrad = _maybe_call(getattr(snapshot, "realgrad", None))
    summary = _maybe_call(getattr(snapshot, "summary", None))
    patch = _maybe_call(getattr(snapshot, "patch", None))
    canvas_stats = _matrix_stats(canvas)
    hyper_stats = _matrix_stats(hypergrad)
    real_stats = _matrix_stats(realgrad)
    if isinstance(summary, Mapping):
        hyper_stats = _merge_summary(hyper_stats, summary.get("hypergrad"))
        real_stats = _merge_summary(real_stats, summary.get("realgrad"))
    patch_stats = _matrix_stats(patch) if patch is not None else None
    stats: dict[str, dict[str, float]] = {
        "canvas": canvas_stats,
        "hypergrad": hyper_stats,
        "realgrad": real_stats,
    }
    if patch_stats is not None:
        stats["patch"] = patch_stats
    return stats


def canvas_partial_from_snapshot(
    snapshot: Any,
    *,
    hyper_gain: float = 2.5,
    memory_gain: float = 2.0,
    stability_gain: float = 2.5,
    patch_gain: float = 1.5,
) -> dict[str, float]:
    """Derive Z-space friendly metrics from a Canvas snapshot."""

    stats = _canvas_snapshot_stats(snapshot)
    canvas = stats.get("canvas", {})
    hyper = stats.get("hypergrad", {})
    real = stats.get("realgrad", {})
    patch = stats.get("patch")

    canvas_norm = float(canvas.get("l2", 0.0))
    hyper_norm = float(hyper.get("l2", 0.0))
    real_norm = float(real.get("l2", 0.0))
    patch_norm = float(patch.get("l2", 0.0)) if patch else 0.0
    total = canvas_norm + hyper_norm + real_norm + 1e-9
    canvas_ratio = canvas_norm / total
    hyper_ratio = hyper_norm / total
    real_ratio = real_norm / total
    patch_ratio = patch_norm / (patch_norm + canvas_norm + 1e-9)

    hyper_gain = max(0.0, float(hyper_gain))
    memory_gain = max(0.0, float(memory_gain))
    stability_gain = max(0.0, float(stability_gain))
    patch_gain = max(0.0, float(patch_gain))

    speed = math.tanh(hyper_gain * hyper_ratio + 0.5 * float(hyper.get("mean", 0.0)))
    memory = math.tanh(memory_gain * canvas_ratio + float(canvas.get("mean", 0.0)))
    stability = math.tanh(
        stability_gain * (1.0 - abs(hyper_ratio - real_ratio)) - 0.5 * stability_gain
    )
    frac_source = float(patch.get("linf", canvas.get("linf", 0.0))) if patch else float(
        canvas.get("linf", 0.0)
    )
    frac = math.tanh(patch_gain * frac_source)
    drs = math.tanh((hyper_ratio - real_ratio) * 2.5)

    partial: dict[str, float] = {
        "speed": speed,
        "memory": memory,
        "stability": stability,
        "frac": frac,
        "drs": drs,
        "canvas_energy": canvas_norm,
        "canvas_mean": float(canvas.get("mean", 0.0)),
        "canvas_peak": float(canvas.get("linf", 0.0)),
        "canvas_l1": float(canvas.get("l1", 0.0)),
        "canvas_l2": canvas_norm,
        "canvas_linf": float(canvas.get("linf", 0.0)),
        "canvas_balance": canvas_ratio,
        "canvas_pixels": float(canvas.get("count", 0.0)),
        "hypergrad_norm": hyper_norm,
        "hypergrad_mean": float(hyper.get("mean", 0.0)),
        "hypergrad_l1": float(hyper.get("l1", 0.0)),
        "hypergrad_l2": hyper_norm,
        "hypergrad_linf": float(hyper.get("linf", 0.0)),
        "hypergrad_balance": hyper_ratio,
        "realgrad_norm": real_norm,
        "realgrad_mean": float(real.get("mean", 0.0)),
        "realgrad_l1": float(real.get("l1", 0.0)),
        "realgrad_l2": real_norm,
        "realgrad_linf": float(real.get("linf", 0.0)),
        "realgrad_balance": real_ratio,
    }
    if patch is not None:
        partial.update(
            {
                "canvas_patch_energy": patch_norm,
                "canvas_patch_mean": float(patch.get("mean", 0.0)),
                "canvas_patch_peak": float(patch.get("linf", 0.0)),
                "canvas_patch_balance": patch_ratio,
                "canvas_patch_pixels": float(patch.get("count", 0.0)),
            }
        )
    return partial


def canvas_coherence_partial(
    snapshot: Any,
    diagnostics: Any,
    *,
    coherence: Any = None,
    contour: Any = None,
    strategy: str = "mean",
    weights: Sequence[float] | None = None,
    canvas_kwargs: Mapping[str, Any] | None = None,
    coherence_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Blend Canvas and coherence-derived partials into a single mapping."""

    canvas_kwargs = dict(canvas_kwargs or {})
    coherence_kwargs = dict(coherence_kwargs or {})
    if coherence is not None:
        coherence_kwargs.setdefault("coherence", coherence)
    if contour is not None:
        coherence_kwargs.setdefault("contour", contour)
    canvas_partial = canvas_partial_from_snapshot(snapshot, **canvas_kwargs)
    coherence_partial = coherence_partial_from_diagnostics(
        diagnostics, **coherence_kwargs
    )
    bundles = [
        ZSpacePartialBundle(canvas_partial, origin="canvas"),
        ZSpacePartialBundle(coherence_partial, origin="coherence"),
    ]
    return blend_zspace_partials(bundles, strategy=strategy, weights=weights)


def infer_canvas_snapshot(
    z_state: Sequence[float],
    snapshot: Any,
    *,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    hyper_gain: float = 2.5,
    memory_gain: float = 2.0,
    stability_gain: float = 2.5,
    patch_gain: float = 1.5,
) -> ZSpaceInference:
    """Project a Canvas snapshot into Z-space inference."""

    partial = canvas_partial_from_snapshot(
        snapshot,
        hyper_gain=hyper_gain,
        memory_gain=memory_gain,
        stability_gain=stability_gain,
        patch_gain=patch_gain,
    )
    return infer_from_partial(z_state, partial, alpha=alpha, smoothing=smoothing)


def infer_canvas_transformer(
    z_state: Sequence[float],
    canvas: Any,
    *,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    hyper_gain: float = 2.5,
    memory_gain: float = 2.0,
    stability_gain: float = 2.5,
    patch_gain: float = 1.5,
) -> ZSpaceInference:
    """Capture a CanvasTransformer snapshot and feed it into inference."""

    snapshot = _maybe_call(getattr(canvas, "snapshot", None))
    if snapshot is None:
        raise AttributeError("canvas object must expose a snapshot() method or property")
    return infer_canvas_snapshot(
        z_state,
        snapshot,
        alpha=alpha,
        smoothing=smoothing,
        hyper_gain=hyper_gain,
        memory_gain=memory_gain,
        stability_gain=stability_gain,
        patch_gain=patch_gain,
    )


def _sequence_floats(values: Any) -> list[float]:
    values = _maybe_call(values)
    if values is None:
        return []
    if isinstance(values, Mapping):
        values = values.values()
    if not isinstance(values, Iterable):
        return []
    out: list[float] = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def coherence_partial_from_diagnostics(
    diagnostics: Any,
    *,
    coherence: Any = None,
    contour: Any = None,
    speed_gain: float = 1.0,
    stability_gain: float = 1.0,
    frac_gain: float = 1.0,
    drs_gain: float = 1.0,
) -> dict[str, float]:
    """Convert coherence diagnostics into Z-space partial observations."""

    speed_gain = max(0.0, float(speed_gain))
    stability_gain = max(0.0, float(stability_gain))
    frac_gain = max(0.0, float(frac_gain))
    drs_gain = max(0.0, float(drs_gain))

    mean_coherence = float(_maybe_call(getattr(diagnostics, "mean_coherence", 0.0)) or 0.0)
    entropy = float(_maybe_call(getattr(diagnostics, "coherence_entropy", 0.0)) or 0.0)
    energy_ratio = float(_maybe_call(getattr(diagnostics, "energy_ratio", 0.0)) or 0.0)
    z_bias = float(_maybe_call(getattr(diagnostics, "z_bias", 0.0)) or 0.0)
    fractional_raw = _maybe_call(getattr(diagnostics, "fractional_order", 0.0))
    fractional_order = float(fractional_raw) if fractional_raw is not None else 0.0
    weights = _sequence_floats(getattr(diagnostics, "normalized_weights", []))
    preserved_raw = _maybe_call(getattr(diagnostics, "preserved_channels", None))
    preserved = float(preserved_raw) if preserved_raw is not None else float(len(weights))
    discarded_raw = _maybe_call(getattr(diagnostics, "discarded_channels", None))
    discarded = float(discarded_raw) if discarded_raw is not None else 0.0
    dominant = _maybe_call(getattr(diagnostics, "dominant_channel", None))

    response = _sequence_floats(coherence)

    partial: dict[str, float] = {
        "speed": math.tanh(speed_gain * mean_coherence),
        "memory": math.tanh(z_bias),
        "stability": math.tanh(stability_gain * (1.0 - entropy)),
        "frac": math.tanh(frac_gain * fractional_order),
        "drs": math.tanh(drs_gain * (energy_ratio - 0.5)),
        "coherence_mean": mean_coherence,
        "coherence_entropy": entropy,
        "coherence_energy_ratio": energy_ratio,
        "coherence_z_bias": z_bias,
        "coherence_fractional_order": fractional_order,
        "coherence_channels": float(len(weights)),
        "coherence_preserved": preserved,
        "coherence_discarded": discarded,
    }
    if dominant is not None:
        try:
            partial["coherence_dominant"] = float(dominant)
        except (TypeError, ValueError):
            partial["coherence_dominant"] = -1.0

    if weights:
        partial["coherence_peak"] = max(weights)
        weight_entropy = -sum(
            weight * math.log(max(weight, 1e-9)) for weight in weights if weight > 0.0
        )
        partial["coherence_weight_entropy"] = weight_entropy
    else:
        partial["coherence_peak"] = 0.0
        partial["coherence_weight_entropy"] = 0.0

    if response:
        partial["coherence_response_peak"] = max(response)
        partial["coherence_response_mean"] = sum(response) / len(response)
    else:
        partial["coherence_response_peak"] = 0.0
        partial["coherence_response_mean"] = 0.0

    if contour is not None:
        for key, attr in (
            ("coherence_strength", "coherence_strength"),
            ("coherence_prosody", "prosody_index"),
            ("coherence_articulation", "articulation_bias"),
        ):
            value = _maybe_call(getattr(contour, attr, None))
            if value is None:
                continue
            try:
                partial[key] = float(value)
            except (TypeError, ValueError):
                partial[key] = 0.0

    return partial


def infer_coherence_diagnostics(
    z_state: Sequence[float],
    diagnostics: Any,
    *,
    coherence: Any = None,
    contour: Any = None,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    speed_gain: float = 1.0,
    stability_gain: float = 1.0,
    frac_gain: float = 1.0,
    drs_gain: float = 1.0,
) -> ZSpaceInference:
    """Fuse coherence diagnostics with a latent state."""

    partial = coherence_partial_from_diagnostics(
        diagnostics,
        coherence=coherence,
        contour=contour,
        speed_gain=speed_gain,
        stability_gain=stability_gain,
        frac_gain=frac_gain,
        drs_gain=drs_gain,
    )
    return infer_from_partial(z_state, partial, alpha=alpha, smoothing=smoothing)


def infer_coherence_from_sequencer(
    z_state: Sequence[float],
    sequencer: Any,
    tensor: Any,
    *,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    method: str = "forward_with_diagnostics",
    include_contour: bool = False,
    return_outputs: bool = False,
    speed_gain: float = 1.0,
    stability_gain: float = 1.0,
    frac_gain: float = 1.0,
    drs_gain: float = 1.0,
):
    """Run a sequencer forward pass and project its diagnostics into Z-space."""

    forward = getattr(sequencer, method, None)
    if forward is None:
        raise AttributeError(f"sequencer has no method '{method}'")
    outputs = forward(tensor)
    if not isinstance(outputs, tuple) or len(outputs) < 3:
        raise ValueError(
            "sequencer forward method must return (tensor, coherence, diagnostics)"
        )
    _, coherence, diagnostics = outputs[:3]
    contour = None
    if include_contour:
        contour_getter = getattr(sequencer, "emit_linguistic_contour", None)
        if callable(contour_getter):
            contour = contour_getter(tensor)
    inference = infer_coherence_diagnostics(
        z_state,
        diagnostics,
        coherence=coherence,
        contour=contour,
        alpha=alpha,
        smoothing=smoothing,
        speed_gain=speed_gain,
        stability_gain=stability_gain,
        frac_gain=frac_gain,
        drs_gain=drs_gain,
    )
    if return_outputs:
        return inference, outputs
    return inference


def infer_canvas_with_coherence(
    z_state: Sequence[float],
    snapshot: Any,
    diagnostics: Any,
    *,
    coherence: Any = None,
    contour: Any = None,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    strategy: str = "mean",
    weights: Sequence[float] | None = None,
    canvas_kwargs: Mapping[str, Any] | None = None,
    coherence_kwargs: Mapping[str, Any] | None = None,
) -> ZSpaceInference:
    """Fuse Canvas and coherence diagnostics before projecting into Z-space."""

    partial = canvas_coherence_partial(
        snapshot,
        diagnostics,
        coherence=coherence,
        contour=contour,
        strategy=strategy,
        weights=weights,
        canvas_kwargs=canvas_kwargs,
        coherence_kwargs=coherence_kwargs,
    )
    return infer_from_partial(z_state, partial, alpha=alpha, smoothing=smoothing)


