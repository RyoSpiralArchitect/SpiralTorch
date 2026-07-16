"""Inference helpers that reconstruct Z-space metrics from partial observations."""

from __future__ import annotations

import inspect
import math
import operator
import sys
from dataclasses import dataclass, replace
from collections.abc import (
    Iterable,
    Mapping as MappingABC,
    MutableMapping,
    Sequence as SequenceABC,
)
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Sequence, Union, get_origin
from types import MappingProxyType

from ._zspace_aliases import (
    ZSPACE_METRIC_ALIASES,
    PRIMARY_ZSPACE_METRIC_ALIASES,
)

__all__ = [
    "ZMetrics",
    "ZSpaceControlGradient",
    "ZSpaceDecoded",
    "ZSpaceInference",
    "ZSpacePosterior",
    "ZSpacePartialBundle",
    "ZSpaceTelemetryFrame",
    "ZSpaceInferenceRuntime",
    "ZSpaceInferencePipeline",
    "ZSPACE_CANONICAL_METRIC_GRADIENT_BASIS",
    "ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS",
    "inference_to_mapping",
    "inference_to_zmetrics",
    "prepare_trainer_step_payload",
    "topos_control_signal",
    "topos_optimizer_snapshot",
    "topos_training_hints",
    "topos_training_plan",
    "topos_inference_hints",
    "topos_inference_plan",
    "topos_runtime_profile",
    "topos_runtime_route",
    "topos_control_partial",
    "zspace_posterior_decode",
    "zspace_posterior_project",
    "zspace_coherence_project",
    "decode_zspace_embedding",
    "infer_from_partial",
    "infer_with_partials",
    "compile_inference",
    "blend_zspace_partials",
    "zspace_partial_fusion",
    "zspace_metric_gradient_projection",
    "zspace_telemetry_fusion",
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
    gradient_basis: str | None = None


_PRIMARY_ALIAS_GROUPS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        canonical: tuple(
            alias
            for alias, target in PRIMARY_ZSPACE_METRIC_ALIASES.items()
            if target == canonical
        )
        for canonical in {"speed", "memory", "stability", "drs", "gradient"}
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


def _unit_interval(value: Any, *, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    if not math.isfinite(numeric):
        numeric = default
    return max(0.0, min(1.0, numeric))


_TOPOS_CONTROL_SIGNAL_CONTRACT_VERSION = "spiraltorch.topos_control_signal.v2"
_TOPOS_CONTROL_SIGNAL_SEMANTIC_OWNER = "st-tensor::pure::topos"
_TOPOS_OPTIMIZER_SNAPSHOT_KIND = "spiraltorch.topos_optimizer_snapshot"
_TOPOS_OPTIMIZER_SNAPSHOT_CONTRACT_VERSION = "spiraltorch.topos_optimizer_snapshot.v3"
_ZSPACE_FUSION_SEMANTIC_OWNER = "st-core::telemetry::zspace_fusion"
_ZSPACE_TELEMETRY_FUSION_KIND = "spiraltorch.zspace_telemetry_fusion"
_ZSPACE_TELEMETRY_FUSION_CONTRACT_VERSION = (
    "spiraltorch.zspace_telemetry_fusion.v1"
)
_ZSPACE_PARTIAL_FUSION_KIND = "spiraltorch.zspace_partial_fusion"
_ZSPACE_PARTIAL_FUSION_CONTRACT_VERSION = "spiraltorch.zspace_partial_fusion.v3"
_ZSPACE_METRIC_GRADIENT_PROJECTION_KIND = (
    "spiraltorch.zspace_metric_gradient_projection"
)
_ZSPACE_METRIC_GRADIENT_PROJECTION_CONTRACT_VERSION = (
    "spiraltorch.zspace_metric_gradient_projection.v1"
)
ZSPACE_CANONICAL_METRIC_GRADIENT_BASIS = (
    "spiraltorch.zspace.canonical_metric_cycle.v1"
)
ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS = (
    "spiraltorch.zspace.latent.central_difference.zero_boundary.v1"
)
_ZSPACE_POSTERIOR_CONTRACT_VERSION = "spiraltorch.zspace_posterior.v2"
_ZSPACE_POSTERIOR_DECODE_KIND = "spiraltorch.zspace_posterior_decode"
_ZSPACE_POSTERIOR_PROJECTION_KIND = "spiraltorch.zspace_posterior_projection"
_ZSPACE_POSTERIOR_SEMANTIC_OWNER = "st-core::inference::zspace_posterior"
_ZSPACE_COHERENCE_PROJECTION_CONTRACT_VERSION = (
    "spiraltorch.zspace_coherence_projection.v1"
)
_ZSPACE_COHERENCE_PROJECTION_KIND = "spiraltorch.zspace_coherence_projection"
_ZSPACE_COHERENCE_PROJECTION_SEMANTIC_OWNER = (
    "st-core::inference::zspace_coherence"
)
_TOPOS_ZSPACE_PROJECTION_CONTRACT_VERSION = (
    "spiraltorch.topos_zspace_projection.v2"
)
_TOPOS_ZSPACE_PROJECTION_KIND = "spiraltorch.topos_zspace_projection"
_TOPOS_ZSPACE_PROJECTION_GRADIENT_BASIS = (
    "spiraltorch.topos.control_signal.axes.v1"
)
_TOPOS_ZSPACE_PROJECTION_GRADIENT_CHANNELS = (
    "openness",
    "guard_strength",
    "stability_hint",
    "exploration_hint",
    "depth_pressure",
    "volume_pressure",
)
_TOPOS_CONTROL_SIGNAL_INPUT_KEYS = frozenset(
    {
        "curvature",
        "tolerance",
        "saturation",
        "porosity",
        "max_depth",
        "max_volume",
        "observed_depth",
        "visited_volume",
    }
)
_TOPOS_CONTROL_SIGNAL_HINT_KEYS = frozenset(
    {
        "training_hints",
        "inference_hints",
    }
)


def _looks_like_topos_runtime_profile(payload: Mapping[str, Any]) -> bool:
    profile_keys = {
        "training_gain",
        "inference_gain",
        "control_energy",
        "closure_risk",
        "exploration_budget",
        "training_rate_scale",
        "training_gradient_bias_scale",
        "inference_temperature",
        "inference_top_p",
        "inference_context_weight",
        "learning_inference_balance",
    }
    signal_keys = _TOPOS_CONTROL_SIGNAL_INPUT_KEYS | {
        "max_depth",
        "max_volume",
        "closure_pressure",
        "training_hints",
        "inference_hints",
        "runtime_profile",
    }
    return bool(profile_keys.intersection(payload)) and not bool(
        signal_keys.intersection(payload)
    )


def _native_topos_runtime_route_from_profile(
    profile: Mapping[str, Any],
) -> dict[str, Any] | None:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    route_from_profile = getattr(native, "_topos_runtime_route_from_profile", None)
    if not callable(route_from_profile):
        return None

    route = route_from_profile(dict(profile))
    if not isinstance(route, MappingABC):
        raise RuntimeError(
            "native Topos runtime router returned a non-mapping contract payload"
        )
    return dict(route)


def _topos_runtime_route_from_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    native = _native_topos_runtime_route_from_profile(profile)
    if native is not None:
        return native
    raise RuntimeError(
        "Topos runtime routing requires the compiled Rust semantic core; "
        "rebuild or reinstall SpiralTorch with _topos_runtime_route_from_profile"
    )


def _native_topos_control_bundle_from_observation(
    payload: Mapping[str, Any],
    *,
    options: Mapping[str, Any] | None = None,
    training_hints: Mapping[str, Any] | None = None,
    inference_hints: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    derive = getattr(native, "_topos_control_bundle_from_observation", None)
    if not callable(derive):
        return None

    bundle = derive(
        dict(payload),
        None if options is None else dict(options),
        None if training_hints is None else dict(training_hints),
        None if inference_hints is None else dict(inference_hints),
    )
    if not isinstance(bundle, MappingABC):
        raise RuntimeError(
            "native Topos control core returned a non-mapping contract payload"
        )
    return dict(bundle)


def _topos_control_bundle_from_observation(
    payload: Mapping[str, Any],
    *,
    options: Mapping[str, Any] | None = None,
    training_hints: Mapping[str, Any] | None = None,
    inference_hints: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bundle = _native_topos_control_bundle_from_observation(
        payload,
        options=options,
        training_hints=training_hints,
        inference_hints=inference_hints,
    )
    if bundle is not None:
        return bundle
    raise RuntimeError(
        "Topos control derivation requires the compiled Rust semantic core; "
        "rebuild or reinstall SpiralTorch with "
        "_topos_control_bundle_from_observation"
    )


def _native_topos_optimizer_snapshot_from_observation(
    payload: Mapping[str, Any],
    *,
    sequence: int,
    hyper_learning_rate: float,
    real_learning_rate: float,
    options: Mapping[str, Any] | None = None,
    training_hints: Mapping[str, Any] | None = None,
    inference_hints: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    derive = getattr(native, "_topos_optimizer_snapshot_from_observation", None)
    if not callable(derive):
        return None

    snapshot = derive(
        dict(payload),
        int(sequence),
        float(hyper_learning_rate),
        float(real_learning_rate),
        None if options is None else dict(options),
        None if training_hints is None else dict(training_hints),
        None if inference_hints is None else dict(inference_hints),
    )
    if not isinstance(snapshot, MappingABC):
        raise RuntimeError(
            "native Topos optimizer core returned a non-mapping contract payload"
        )
    return dict(snapshot)


def _is_native_topos_optimizer_snapshot(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("kind") == _TOPOS_OPTIMIZER_SNAPSHOT_KIND
        and payload.get("contract_version")
        == _TOPOS_OPTIMIZER_SNAPSHOT_CONTRACT_VERSION
        and payload.get("semantic_owner") == _TOPOS_CONTROL_SIGNAL_SEMANTIC_OWNER
        and payload.get("semantic_backend") == "rust"
    )


def _native_topos_zspace_projection_from_observation(
    payload: Mapping[str, Any],
    *,
    gradient_dim: int,
) -> dict[str, Any] | None:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    project = getattr(native, "_topos_zspace_projection_from_observation", None)
    if not callable(project):
        return None

    projection = project(dict(payload), gradient_dim)
    if not isinstance(projection, MappingABC):
        raise RuntimeError(
            "native Topos Z-space core returned a non-mapping contract payload"
        )
    return dict(projection)


def _topos_zspace_projection_from_observation(
    payload: Mapping[str, Any],
    *,
    gradient_dim: int,
) -> dict[str, Any]:
    projection = _native_topos_zspace_projection_from_observation(
        payload,
        gradient_dim=gradient_dim,
    )
    if projection is None:
        raise RuntimeError(
            "Topos Z-space projection requires the compiled Rust semantic core; "
            "rebuild or reinstall SpiralTorch with "
            "_topos_zspace_projection_from_observation"
        )
    if (
        projection.get("kind") != _TOPOS_ZSPACE_PROJECTION_KIND
        or projection.get("contract_version")
        != _TOPOS_ZSPACE_PROJECTION_CONTRACT_VERSION
        or projection.get("semantic_owner")
        != _TOPOS_CONTROL_SIGNAL_SEMANTIC_OWNER
        or projection.get("semantic_backend") != "rust"
    ):
        raise RuntimeError("native Topos Z-space core returned an untrusted contract")
    gradient_channels = projection.get("gradient_channels")
    gradient = projection.get("gradient")
    gradient_formula = projection.get("gradient_formula")
    if (
        projection.get("gradient_basis")
        != _TOPOS_ZSPACE_PROJECTION_GRADIENT_BASIS
        or not isinstance(gradient_channels, SequenceABC)
        or isinstance(gradient_channels, (str, bytes, bytearray))
        or tuple(gradient_channels) != _TOPOS_ZSPACE_PROJECTION_GRADIENT_CHANNELS
        or not isinstance(gradient_formula, str)
        or not gradient_formula
    ):
        raise RuntimeError("native Topos Z-space core returned an unknown gradient basis")
    if (
        not isinstance(gradient, SequenceABC)
        or isinstance(gradient, (str, bytes, bytearray))
        or projection.get("gradient_dim") != len(gradient)
        or projection.get("base_gradient_dim") != len(gradient_channels)
    ):
        raise RuntimeError("native Topos Z-space core returned a malformed gradient")
    return projection


def _is_native_topos_control_signal(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("contract_version") == _TOPOS_CONTROL_SIGNAL_CONTRACT_VERSION
        and payload.get("semantic_owner") == _TOPOS_CONTROL_SIGNAL_SEMANTIC_OWNER
        and payload.get("semantic_backend") == "rust"
    )


def _topos_mapping_section(
    payload: Mapping[str, Any],
    key: str,
) -> dict[str, Any] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, MappingABC):
        raise TypeError("Topos {} must be a mapping".format(key))
    return dict(value)


def _topos_control_request_parts(
    topos: Any | None,
    *,
    curvature: float,
    tolerance: float,
    saturation: float,
    max_depth: int,
    max_volume: int,
    porosity: float | None,
    observed_depth: int,
    visited_volume: int,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    defaults: dict[str, Any] = {
        "curvature": curvature,
        "tolerance": tolerance,
        "saturation": saturation,
        "porosity": 0.2 if porosity is None else porosity,
        "max_depth": max_depth,
        "max_volume": max_volume,
        "observed_depth": observed_depth,
        "visited_volume": visited_volume,
    }

    def _from_mapping(
        source: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
        source_payload = dict(source)
        if any(not isinstance(key, str) for key in source_payload):
            raise TypeError("Topos control signal keys must be strings")
        training_hints = _topos_mapping_section(source_payload, "training_hints")
        inference_hints = _topos_mapping_section(source_payload, "inference_hints")
        if not _is_native_topos_control_signal(source_payload):
            allowed = _TOPOS_CONTROL_SIGNAL_INPUT_KEYS | _TOPOS_CONTROL_SIGNAL_HINT_KEYS
            unsupported = sorted(set(source_payload) - allowed)
            if unsupported:
                raise ValueError(
                    "Topos derived fields are Rust-owned; unsupported overrides: {}".format(
                        ", ".join(unsupported)
                    )
                )
        raw = {
            key: source_payload.get(key, default) for key, default in defaults.items()
        }
        return raw, training_hints, inference_hints

    if isinstance(topos, MappingABC):
        return _from_mapping(topos)

    if topos is None:
        return defaults, None, None

    guard = topos
    if porosity is not None:
        with_porosity = getattr(guard, "with_porosity", None)
        if callable(with_porosity):
            guard = with_porosity(float(porosity))

    control_signal = getattr(guard, "control_signal", None)
    if callable(control_signal):
        try:
            source = control_signal(int(observed_depth), int(visited_volume))
        except TypeError:
            source = control_signal()
        if isinstance(source, MappingABC):
            return _from_mapping(source)

    def _call(name: str, default: Any) -> Any:
        attr = getattr(guard, name, None)
        return attr() if callable(attr) else default

    raw = dict(defaults)
    for key in (
        "curvature",
        "tolerance",
        "saturation",
        "max_depth",
        "max_volume",
    ):
        raw[key] = _call(key, raw[key])
    if porosity is None:
        raw["porosity"] = _call("porosity", raw["porosity"])
    return raw, None, None


def _topos_control_bundle(
    topos: Any | None = None,
    *,
    plan_options: Mapping[str, Any] | None = None,
    curvature: float = -1.0,
    tolerance: float = 1e-3,
    saturation: float = 1.0,
    max_depth: int = 64,
    max_volume: int = 512,
    porosity: float | None = None,
    observed_depth: int = 0,
    visited_volume: int = 0,
) -> dict[str, Any]:
    payload, training_hints, inference_hints = _topos_control_request_parts(
        topos,
        curvature=curvature,
        tolerance=tolerance,
        saturation=saturation,
        max_depth=max_depth,
        max_volume=max_volume,
        porosity=porosity,
        observed_depth=observed_depth,
        visited_volume=visited_volume,
    )
    return _topos_control_bundle_from_observation(
        payload,
        options=plan_options,
        training_hints=training_hints,
        inference_hints=inference_hints,
    )


def _topos_bundle_section(bundle: Mapping[str, Any], key: str) -> dict[str, Any]:
    section = bundle.get(key)
    if not isinstance(section, MappingABC):
        raise RuntimeError("Rust Topos bundle is missing {}".format(key))
    return dict(section)


def topos_control_signal(
    topos: Any | None = None,
    *,
    training_gain: float = 1.0,
    curvature: float = -1.0,
    tolerance: float = 1e-3,
    saturation: float = 1.0,
    max_depth: int = 64,
    max_volume: int = 512,
    porosity: float | None = None,
    observed_depth: int = 0,
    visited_volume: int = 0,
) -> dict[str, Any]:
    """Return the canonical Rust-owned Topos control bundle."""

    return _topos_control_bundle(
        topos,
        plan_options={"training_gain": training_gain},
        curvature=curvature,
        tolerance=tolerance,
        saturation=saturation,
        max_depth=max_depth,
        max_volume=max_volume,
        porosity=porosity,
        observed_depth=observed_depth,
        visited_volume=visited_volume,
    )


def topos_optimizer_snapshot(
    topos: Any | None = None,
    *,
    sequence: int = 0,
    hyper_learning_rate: float,
    real_learning_rate: float,
    gain: float = 1.0,
    training_hints: Mapping[str, Any] | None = None,
    inference_hints: Mapping[str, Any] | None = None,
    curvature: float = -1.0,
    tolerance: float = 1e-3,
    saturation: float = 1.0,
    max_depth: int = 64,
    max_volume: int = 512,
    porosity: float | None = None,
    observed_depth: int = 0,
    visited_volume: int = 0,
) -> dict[str, Any]:
    """Bind one Rust-owned Topos control bundle to a prescribed rate application."""

    payload, embedded_training_hints, embedded_inference_hints = (
        _topos_control_request_parts(
            topos,
            curvature=curvature,
            tolerance=tolerance,
            saturation=saturation,
            max_depth=max_depth,
            max_volume=max_volume,
            porosity=porosity,
            observed_depth=observed_depth,
            visited_volume=visited_volume,
        )
    )
    if training_hints is None:
        used_training_hints = embedded_training_hints
    elif isinstance(training_hints, MappingABC):
        used_training_hints = dict(training_hints)
    else:
        raise TypeError("Topos training_hints must be a mapping")
    if inference_hints is None:
        used_inference_hints = embedded_inference_hints
    elif isinstance(inference_hints, MappingABC):
        used_inference_hints = dict(inference_hints)
    else:
        raise TypeError("Topos inference_hints must be a mapping")

    snapshot = _native_topos_optimizer_snapshot_from_observation(
        payload,
        sequence=sequence,
        hyper_learning_rate=hyper_learning_rate,
        real_learning_rate=real_learning_rate,
        options={"training_gain": gain},
        training_hints=used_training_hints,
        inference_hints=used_inference_hints,
    )
    if snapshot is None:
        raise RuntimeError(
            "Topos optimizer snapshots require the compiled Rust semantic core; "
            "rebuild or reinstall SpiralTorch with "
            "_topos_optimizer_snapshot_from_observation"
        )
    if not _is_native_topos_optimizer_snapshot(snapshot):
        raise RuntimeError("native Topos optimizer core returned an untrusted contract")
    return snapshot


def topos_training_hints(
    topos: Any | None = None,
    **signal_options: Any,
) -> dict[str, Any]:
    """Return Rust-owned optimizer hints for an open-topos signal."""

    bundle = _topos_control_bundle(topos, **signal_options)
    return _topos_bundle_section(bundle, "training_hints")


def topos_training_plan(
    topos: Any | None = None,
    *,
    gain: float = 1.0,
    **signal_options: Any,
) -> dict[str, Any]:
    """Return Rust-owned gain-applied optimizer controls."""

    bundle = _topos_control_bundle(
        topos,
        plan_options={"training_gain": gain},
        **signal_options,
    )
    return _topos_bundle_section(bundle, "training_plan")


def topos_inference_hints(
    topos: Any | None = None,
    **signal_options: Any,
) -> dict[str, Any]:
    """Return Rust-owned hosted-inference hints."""

    bundle = _topos_control_bundle(topos, **signal_options)
    return _topos_bundle_section(bundle, "inference_hints")


def _topos_plan_options(
    *,
    training_gain: float = 1.0,
    inference_gain: float = 1.0,
    base_temperature: float = 1.0,
    base_top_p: float = 1.0,
    min_temperature: float = 0.0,
    max_temperature: float = 2.0,
    min_top_p: float = 0.05,
    max_top_p: float = 1.0,
    base_frequency_penalty: float = 0.0,
    base_presence_penalty: float = 0.0,
) -> dict[str, float]:
    return {
        "training_gain": training_gain,
        "inference_gain": inference_gain,
        "base_temperature": base_temperature,
        "base_top_p": base_top_p,
        "min_temperature": min_temperature,
        "max_temperature": max_temperature,
        "min_top_p": min_top_p,
        "max_top_p": max_top_p,
        "base_frequency_penalty": base_frequency_penalty,
        "base_presence_penalty": base_presence_penalty,
    }


def topos_inference_plan(
    topos: Any | None = None,
    *,
    gain: float = 1.0,
    base_temperature: float = 1.0,
    base_top_p: float = 1.0,
    min_temperature: float = 0.0,
    max_temperature: float = 2.0,
    min_top_p: float = 0.05,
    max_top_p: float = 1.0,
    base_frequency_penalty: float = 0.0,
    base_presence_penalty: float = 0.0,
    **signal_options: Any,
) -> dict[str, Any]:
    """Return Rust-owned concrete hosted-inference controls."""

    bundle = _topos_control_bundle(
        topos,
        plan_options=_topos_plan_options(
            inference_gain=gain,
            base_temperature=base_temperature,
            base_top_p=base_top_p,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            min_top_p=min_top_p,
            max_top_p=max_top_p,
            base_frequency_penalty=base_frequency_penalty,
            base_presence_penalty=base_presence_penalty,
        ),
        **signal_options,
    )
    return _topos_bundle_section(bundle, "inference_plan")


def topos_runtime_profile(
    topos: Any | None = None,
    *,
    training_gain: float = 1.0,
    inference_gain: float = 1.0,
    base_temperature: float = 1.0,
    base_top_p: float = 1.0,
    min_temperature: float = 0.0,
    max_temperature: float = 2.0,
    min_top_p: float = 0.05,
    max_top_p: float = 1.0,
    base_frequency_penalty: float = 0.0,
    base_presence_penalty: float = 0.0,
    **signal_options: Any,
) -> dict[str, Any]:
    """Return one Rust-owned learning/inference profile."""

    bundle = _topos_control_bundle(
        topos,
        plan_options=_topos_plan_options(
            training_gain=training_gain,
            inference_gain=inference_gain,
            base_temperature=base_temperature,
            base_top_p=base_top_p,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            min_top_p=min_top_p,
            max_top_p=max_top_p,
            base_frequency_penalty=base_frequency_penalty,
            base_presence_penalty=base_presence_penalty,
        ),
        **signal_options,
    )
    return _topos_bundle_section(bundle, "runtime_profile")


def topos_runtime_route(
    topos: Any | None = None,
    *,
    runtime_profile: Mapping[str, Any] | None = None,
    training_gain: float = 1.0,
    inference_gain: float = 1.0,
    base_temperature: float = 1.0,
    base_top_p: float = 1.0,
    min_temperature: float = 0.0,
    max_temperature: float = 2.0,
    min_top_p: float = 0.05,
    max_top_p: float = 1.0,
    base_frequency_penalty: float = 0.0,
    base_presence_penalty: float = 0.0,
    **signal_options: Any,
) -> dict[str, Any]:
    """Name the safest route using the shared Rust semantic contract."""

    if runtime_profile is not None:
        profile = dict(runtime_profile)
        profile.setdefault("training_gain", training_gain)
        profile.setdefault("inference_gain", inference_gain)
        return _topos_runtime_route_from_profile(profile)
    if isinstance(topos, MappingABC) and _looks_like_topos_runtime_profile(topos):
        profile = dict(topos)
        profile.setdefault("training_gain", training_gain)
        profile.setdefault("inference_gain", inference_gain)
        return _topos_runtime_route_from_profile(profile)

    bundle = _topos_control_bundle(
        topos,
        plan_options=_topos_plan_options(
            training_gain=training_gain,
            inference_gain=inference_gain,
            base_temperature=base_temperature,
            base_top_p=base_top_p,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            min_top_p=min_top_p,
            max_top_p=max_top_p,
            base_frequency_penalty=base_frequency_penalty,
            base_presence_penalty=base_presence_penalty,
        ),
        **signal_options,
    )
    return _topos_bundle_section(bundle, "runtime_route")


def topos_zspace_projection(
    topos: Any | None = None,
    *,
    gradient_dim: int = 6,
    curvature: float = -1.0,
    tolerance: float = 1e-3,
    saturation: float = 1.0,
    max_depth: int = 64,
    max_volume: int = 512,
    porosity: float | None = None,
    observed_depth: int = 0,
    visited_volume: int = 0,
) -> dict[str, Any]:
    """Project an open-topos signal through the canonical Rust Z-space contract."""

    payload, _, _ = _topos_control_request_parts(
        topos,
        curvature=curvature,
        tolerance=tolerance,
        saturation=saturation,
        max_depth=max_depth,
        max_volume=max_volume,
        porosity=porosity,
        observed_depth=observed_depth,
        visited_volume=visited_volume,
    )
    return _topos_zspace_projection_from_observation(
        payload,
        gradient_dim=gradient_dim,
    )


def topos_control_partial(
    topos: Any | None = None,
    *,
    bundle_weight: float = 1.0,
    origin: str | None = "topos:control",
    telemetry_prefix: str = "topos",
    gradient_dim: int = 6,
    **signal_options: Any,
) -> "ZSpacePartialBundle":
    """Convert an open-topos pressure signal into a Z-space inference partial."""

    signal = topos_control_signal(topos, **signal_options)
    projection = topos_zspace_projection(signal, gradient_dim=gradient_dim)
    metrics = {
        key: projection[key]
        for key in ("speed", "memory", "stability", "drs", "frac", "gradient")
    }
    telemetry_signal = dict(signal)
    telemetry_signal.pop("training_plan", None)
    telemetry_signal.pop("inference_plan", None)
    telemetry_signal.pop("runtime_profile", None)
    runtime_route = telemetry_signal.get("runtime_route")
    if isinstance(runtime_route, MappingABC):
        route_payload = dict(runtime_route)
        route_payload.pop("runtime_profile", None)
        telemetry_signal["runtime_route"] = route_payload
    telemetry_signal["zspace_projection"] = projection
    telemetry = _flatten_telemetry(telemetry_signal, prefix=f"{telemetry_prefix}.")
    return ZSpacePartialBundle(
        metrics,
        weight=max(0.0, float(bundle_weight)),
        origin=origin,
        telemetry=telemetry,
        gradient_basis=projection["gradient_basis"],
    )


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

    base_metrics, gradient = _normalise_metrics_mapping(
        getattr(inference, "metrics", None)
    )
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

    telemetry = (
        _flatten_telemetry_payload(getattr(inference, "telemetry", None)) or None
    )

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
        gradient_basis = (
            inference.get("gradient_basis")
            if isinstance(inference, MappingABC)
            else getattr(inference, "gradient_basis", None)
        )
        if gradient_basis is not None and not isinstance(gradient_basis, str):
            raise TypeError("gradient_basis must be a string")
        if gradient_basis is not None:
            resolved["gradient_basis"] = gradient_basis

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

    if isinstance(inference, MappingABC):
        gradient_basis = inference.get("gradient_basis")
    else:
        gradient_basis = getattr(inference, "gradient_basis", None)
    if gradient_basis is not None and not isinstance(gradient_basis, str):
        raise TypeError("gradient_basis must be a string")

    return ZMetrics(
        speed=_resolve_primary(metrics, "speed"),
        memory=_resolve_primary(metrics, "memory"),
        stability=_resolve_primary(metrics, "stability"),
        gradient=gradient_seq,
        gradient_basis=gradient_basis,
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
        return inference_to_zmetrics(
            inference, prefer_applied=prefer_applied, include_telemetry=True
        )
    if chosen == "mapping":
        return inference_to_mapping(
            inference,
            prefer_applied=prefer_applied,
            canonical=canonical_mapping,
            include_gradient=True,
        )
    return inference


def _flatten_telemetry(
    payload: Mapping[str, Any], prefix: str = ""
) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        raise TypeError("telemetry payloads must be provided as mappings")
    wrapped: Mapping[str, Any] = payload
    label = prefix.rstrip(".")
    if label:
        wrapped = {label: dict(payload)}
    return dict(zspace_telemetry_fusion(wrapped)["payload"])


def _native_zspace_fusion(function_name: str, payload: Any) -> dict[str, Any]:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    fuse = getattr(native, function_name, None)
    if not callable(fuse):
        raise RuntimeError(
            "Z-space fusion requires the compiled Rust semantic core; "
            f"rebuild or reinstall SpiralTorch with {function_name}"
        )
    contract = fuse(payload)
    if not isinstance(contract, MappingABC):
        raise RuntimeError("native Z-space fusion returned a non-mapping payload")
    return dict(contract)


def _native_zspace_posterior(
    function_name: str,
    request: Mapping[str, Any],
    *,
    expected_kind: str,
) -> dict[str, Any]:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    operation = getattr(native, function_name, None)
    if not callable(operation):
        raise RuntimeError(
            "Z-space posterior inference requires the compiled Rust semantic core; "
            f"rebuild or reinstall SpiralTorch with {function_name}"
        )
    contract = operation(dict(request))
    if not isinstance(contract, MappingABC):
        raise RuntimeError("native Z-space posterior returned a non-mapping payload")
    contract = dict(contract)
    _validate_zspace_posterior_contract(contract, expected_kind=expected_kind)
    return contract


def _native_zspace_coherence_projection(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    package = sys.modules.get(__package__ or "spiraltorch")
    native = getattr(package, "_rs", None)
    project = getattr(native, "_zspace_coherence_project", None)
    if not callable(project):
        raise RuntimeError(
            "Z-space coherence projection requires the compiled Rust semantic core; "
            "rebuild or reinstall SpiralTorch with _zspace_coherence_project"
        )
    contract = project(dict(request))
    if not isinstance(contract, MappingABC):
        raise RuntimeError(
            "native Z-space coherence projection returned a non-mapping payload"
        )
    contract = dict(contract)
    if (
        contract.get("kind") != _ZSPACE_COHERENCE_PROJECTION_KIND
        or contract.get("contract_version")
        != _ZSPACE_COHERENCE_PROJECTION_CONTRACT_VERSION
        or contract.get("semantic_owner")
        != _ZSPACE_COHERENCE_PROJECTION_SEMANTIC_OWNER
        or contract.get("semantic_backend") != "rust"
    ):
        raise RuntimeError(
            "native Z-space coherence projection returned an untrusted contract"
        )
    partial = contract.get("partial")
    if not isinstance(partial, MappingABC):
        raise RuntimeError(
            "native Z-space coherence projection returned malformed partial metrics"
        )
    return contract


def _validate_zspace_posterior_contract(
    contract: Mapping[str, Any], *, expected_kind: str
) -> None:
    if (
        contract.get("kind") != expected_kind
        or contract.get("contract_version") != _ZSPACE_POSTERIOR_CONTRACT_VERSION
        or contract.get("semantic_owner") != _ZSPACE_POSTERIOR_SEMANTIC_OWNER
        or contract.get("semantic_backend") != "rust"
    ):
        raise RuntimeError("native Z-space posterior returned an untrusted contract")


def _validate_zspace_fusion_contract(
    contract: Mapping[str, Any], *, kind: str, contract_version: str
) -> None:
    if (
        contract.get("kind") != kind
        or contract.get("contract_version") != contract_version
        or contract.get("semantic_owner") != _ZSPACE_FUSION_SEMANTIC_OWNER
        or contract.get("semantic_backend") != "rust"
    ):
        raise RuntimeError("native Z-space fusion returned an untrusted contract")


def _telemetry_inputs(
    payloads: Sequence[Mapping[str, Any] | "ZSpaceTelemetryFrame" | None],
) -> list[dict[str, Any]]:
    normalised: list[dict[str, Any]] = []
    for payload in payloads:
        if payload is None:
            continue
        if isinstance(payload, ZSpaceTelemetryFrame):
            normalised.append(dict(payload.payload))
        elif isinstance(payload, Mapping):
            normalised.append(dict(payload))
        else:
            raise TypeError("telemetry payloads must be provided as mappings")
    return normalised


def zspace_telemetry_fusion(
    *payloads: Mapping[str, Any]
    | "ZSpaceTelemetryFrame"
    | Sequence[Mapping[str, Any] | "ZSpaceTelemetryFrame" | None]
    | None,
) -> dict[str, Any]:
    """Fuse telemetry through the canonical Rust contract."""

    inputs: Sequence[Mapping[str, Any] | ZSpaceTelemetryFrame | None] = payloads
    if (
        len(payloads) == 1
        and isinstance(payloads[0], SequenceABC)
        and not isinstance(payloads[0], (str, bytes, bytearray))
    ):
        inputs = payloads[0]
    contract = _native_zspace_fusion(
        "_zspace_telemetry_fusion", _telemetry_inputs(inputs)
    )
    _validate_zspace_fusion_contract(
        contract,
        kind=_ZSPACE_TELEMETRY_FUSION_KIND,
        contract_version=_ZSPACE_TELEMETRY_FUSION_CONTRACT_VERSION,
    )
    return contract


def _normalise_telemetry_payload(
    payload: Mapping[str, Any] | "ZSpaceTelemetryFrame" | None,
) -> dict[str, float]:
    if payload is None:
        return {}
    if isinstance(payload, ZSpaceTelemetryFrame):
        return dict(payload.payload)
    if isinstance(payload, Mapping):
        return dict(zspace_telemetry_fusion(payload)["payload"])
    raise TypeError("telemetry payloads must be provided as mappings")


def _merge_telemetry_payloads(
    *payloads: Mapping[str, Any] | "ZSpaceTelemetryFrame" | None,
) -> dict[str, float]:
    return dict(zspace_telemetry_fusion(*payloads)["payload"])


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
_ELLIPTIC_GRADIENT_BASES = MappingProxyType(
    {
        "rotor_transport": "spiraltorch.elliptic.rotor_transport.v1",
        "flow_vector": "spiraltorch.elliptic.flow_vector.v1",
        "lie_log": "spiraltorch.elliptic.lie_log.v1",
    }
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


def _telemetry_frame_from_fusion_contract(
    contract: Mapping[str, Any],
) -> ZSpaceTelemetryFrame:
    _validate_zspace_fusion_contract(
        contract,
        kind=_ZSPACE_TELEMETRY_FUSION_KIND,
        contract_version=_ZSPACE_TELEMETRY_FUSION_CONTRACT_VERSION,
    )
    flattened = contract.get("payload")
    stats = contract.get("summary")
    if not isinstance(flattened, MappingABC):
        raise RuntimeError("native Z-space telemetry payload is malformed")
    if not isinstance(stats, MappingABC):
        raise RuntimeError("native Z-space telemetry summary is malformed")
    return ZSpaceTelemetryFrame(
        MappingProxyType({str(key): float(value) for key, value in flattened.items()}),
        mean=float(stats["mean"]),
        variance=float(stats["variance"]),
        amplitude=float(stats["amplitude"]),
        energy=float(stats["energy"]),
        balance=float(stats["balance"]),
        focus=float(stats["focus"]),
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
    gradient_basis: str | None = None

    def resolved(self) -> dict[str, Any]:
        """Return the canonicalised metric mapping."""

        return _canonicalise_inputs(
            self.metrics,
            gradient_basis=self.gradient_basis,
        )

    def telemetry_payload(self) -> Mapping[str, Any] | None:
        """Return a copy of any telemetry payload attached to the bundle."""

        if self.telemetry is None:
            return None
        if not isinstance(self.telemetry, Mapping):
            raise TypeError("telemetry payloads must be mappings")
        return MappingProxyType(dict(self.telemetry))


def _canonicalise_inputs(
    partial: Mapping[str, Any] | None,
    *,
    gradient_basis: str | None = None,
) -> dict[str, Any]:
    if partial is None:
        return {}
    if not isinstance(partial, Mapping):
        raise TypeError("partial observations must be provided as a mapping")
    metrics, resolved_basis = _split_partial_gradient_basis(partial, gradient_basis)
    source: Mapping[str, Any] | ZSpacePartialBundle = metrics
    if resolved_basis is not None:
        source = ZSpacePartialBundle(metrics, gradient_basis=resolved_basis)
    return _metrics_from_fusion_contract(zspace_partial_fusion([source]))


def _split_partial_gradient_basis(
    partial: Mapping[str, Any],
    explicit_basis: str | None,
) -> tuple[dict[str, Any], str | None]:
    if explicit_basis is not None and not isinstance(explicit_basis, str):
        raise TypeError("gradient_basis must be a string")
    embedded_basis = partial.get("gradient_basis")
    if embedded_basis is not None and not isinstance(embedded_basis, str):
        raise TypeError("gradient_basis must be a string")
    if (
        explicit_basis is not None
        and embedded_basis is not None
        and explicit_basis != embedded_basis
    ):
        raise ValueError("embedded and explicit gradient_basis values disagree")
    resolved_basis = explicit_basis if explicit_basis is not None else embedded_basis
    metrics = {key: value for key, value in partial.items() if key != "gradient_basis"}
    return metrics, resolved_basis


def _metric_input(value: Any) -> float | list[float]:
    if isinstance(value, MappingABC):
        value = value.values()
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        return float(value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Iterable):
        return [float(entry) for entry in value]
    return float(value)


def _partial_fusion_input(
    partial: Mapping[str, Any] | ZSpacePartialBundle | None,
) -> dict[str, Any] | None:
    if partial is None:
        return None
    if isinstance(partial, ZSpacePartialBundle):
        metrics, gradient_basis = _split_partial_gradient_basis(
            partial.metrics, partial.gradient_basis
        )
        weight = float(partial.weight)
        origin = partial.origin
        telemetry = partial.telemetry_payload()
    elif isinstance(partial, Mapping):
        metrics, gradient_basis = _split_partial_gradient_basis(partial, None)
        weight = 1.0
        origin = None
        telemetry = None
    else:
        raise TypeError(
            "partial observations must be mappings or ZSpacePartialBundle instances"
        )
    encoded = {
        "metrics": {str(key): _metric_input(value) for key, value in metrics.items()},
        "weight": weight,
        "origin": origin,
        "gradient_basis": gradient_basis,
        "telemetry": None if telemetry is None else dict(telemetry),
    }
    return encoded


def _partial_telemetry_inputs(
    telemetry: Mapping[str, Any]
    | ZSpaceTelemetryFrame
    | Sequence[Mapping[str, Any] | ZSpaceTelemetryFrame | None]
    | None,
) -> list[dict[str, Any]]:
    if telemetry is None:
        return []
    if isinstance(telemetry, (MappingABC, ZSpaceTelemetryFrame)):
        return _telemetry_inputs((telemetry,))
    if isinstance(telemetry, (str, bytes, bytearray, memoryview)):
        raise TypeError("telemetry payloads must be provided as mappings")
    return _telemetry_inputs(telemetry)


def _contract_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{field} must be an integer") from exc


def zspace_metric_gradient_projection(
    metrics: Mapping[str, Any],
    *,
    gradient_dim: int,
) -> dict[str, Any]:
    """Project canonical metrics through the versioned Rust gradient basis."""

    if not isinstance(metrics, Mapping):
        raise TypeError("metric-gradient projection metrics must be a mapping")
    encoded_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str):
            raise TypeError("metric-gradient projection metric names must be strings")
        if isinstance(value, (bool, str, bytes, bytearray, memoryview)):
            raise TypeError("metric-gradient projection metric values must be numeric")
        encoded_metrics[key] = float(value)
    dimension = _contract_integer(
        gradient_dim,
        field="metric-gradient projection dimension",
    )
    request = {
        "metrics": encoded_metrics,
        "dimension": dimension,
    }
    contract = _native_zspace_fusion(
        "_zspace_metric_gradient_projection",
        request,
    )
    _validate_zspace_fusion_contract(
        contract,
        kind=_ZSPACE_METRIC_GRADIENT_PROJECTION_KIND,
        contract_version=_ZSPACE_METRIC_GRADIENT_PROJECTION_CONTRACT_VERSION,
    )
    gradient = contract.get("gradient")
    if not isinstance(gradient, SequenceABC) or isinstance(
        gradient, (str, bytes, bytearray, memoryview)
    ):
        raise RuntimeError("native metric-gradient projection returned malformed gradient")
    if (
        contract.get("dimension") != request["dimension"]
        or len(gradient) != request["dimension"]
    ):
        raise RuntimeError(
            "native metric-gradient projection returned an unexpected dimension"
        )
    if contract.get("basis") != ZSPACE_CANONICAL_METRIC_GRADIENT_BASIS:
        raise RuntimeError("native metric-gradient projection returned an unknown basis")
    return contract


def zspace_partial_fusion(
    partials: Sequence[Mapping[str, Any] | ZSpacePartialBundle | None],
    *,
    weights: Sequence[float] | None = None,
    strategy: str = "mean",
    gradient_alignment: str = "strict",
    metric_gradient_dimension: int | None = None,
    telemetry: Mapping[str, Any]
    | ZSpaceTelemetryFrame
    | Sequence[Mapping[str, Any] | ZSpaceTelemetryFrame | None]
    | None = None,
) -> dict[str, Any]:
    """Fuse partial metrics and telemetry through the canonical Rust contract."""

    if not isinstance(partials, Sequence) or isinstance(
        partials, (str, bytes, bytearray, memoryview)
    ):
        raise TypeError("partials must be provided as a sequence")
    request: dict[str, Any] = {
        "partials": [_partial_fusion_input(partial) for partial in partials],
        "strategy": str(strategy),
        "gradient_alignment": str(gradient_alignment),
        "telemetry": _partial_telemetry_inputs(telemetry),
    }
    if weights is not None:
        request["weights"] = [float(weight) for weight in weights]
    if metric_gradient_dimension is not None:
        request["metric_gradient_dimension"] = _contract_integer(
            metric_gradient_dimension,
            field="metric_gradient_dimension",
        )
    contract = _native_zspace_fusion("_zspace_partial_fusion", request)
    _validate_zspace_fusion_contract(
        contract,
        kind=_ZSPACE_PARTIAL_FUSION_KIND,
        contract_version=_ZSPACE_PARTIAL_FUSION_CONTRACT_VERSION,
    )
    telemetry_contract = contract.get("telemetry")
    if not isinstance(telemetry_contract, MappingABC):
        raise RuntimeError("native Z-space partial fusion omitted telemetry contract")
    _validate_zspace_fusion_contract(
        telemetry_contract,
        kind=_ZSPACE_TELEMETRY_FUSION_KIND,
        contract_version=_ZSPACE_TELEMETRY_FUSION_CONTRACT_VERSION,
    )
    return contract


def _metrics_from_fusion_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    metrics = contract.get("metrics")
    if not isinstance(metrics, MappingABC):
        raise RuntimeError("native Z-space partial fusion returned malformed metrics")
    fused: dict[str, Any] = dict(metrics)
    gradient = contract.get("gradient")
    if gradient is not None:
        if not isinstance(gradient, Sequence) or isinstance(
            gradient, (str, bytes, bytearray, memoryview)
        ):
            raise RuntimeError("native Z-space partial fusion returned malformed gradient")
        fused["gradient"] = [float(value) for value in gradient]
    return fused


def blend_zspace_partials(
    partials: Sequence[Mapping[str, Any] | ZSpacePartialBundle | None],
    *,
    weights: Sequence[float] | None = None,
    strategy: str = "mean",
    gradient_alignment: str = "strict",
    metric_gradient_dimension: int | None = None,
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
        Supported values are ``"mean"`` (default), ``"last"``, ``"max"``,
        ``"min"``, ``"median"`` and ``"sum"``.
    gradient_alignment:
        ``"strict"`` rejects ragged active gradients. Use ``"pad_zero"`` only
        when explicitly preserving the legacy zero-padding behavior.
    metric_gradient_dimension:
        When set, Rust replaces positional input gradients with one canonical
        projection of the fused named base metrics at this dimension.
    """

    return _metrics_from_fusion_contract(
        zspace_partial_fusion(
            partials,
            weights=weights,
            strategy=strategy,
            gradient_alignment=gradient_alignment,
            metric_gradient_dimension=metric_gradient_dimension,
        )
    )


def elliptic_partial_from_telemetry(
    telemetry: Any,
    *,
    bundle_weight: float = 1.0,
    origin: str | None = "elliptic",
    telemetry_prefix: str = "elliptic",
    aggregate: str = "mean",
    gradient_alignment: str = "strict",
    gradient_source: str = "rotor_transport",
    gradient_basis: str | None = None,
    extra_telemetry: Mapping[str, Any] | None = None,
) -> ZSpacePartialBundle:
    """Convert elliptic telemetry through the Rust-owned partial fusion contract.

    ``aggregate`` applies to both scalar metrics and gradient coordinates.
    Ragged gradients fail by default; ``gradient_alignment="pad_zero"`` opts
    into the audited legacy compatibility path.
    """

    samples = [
        mapping
        for sample in _iter_elliptic_samples(telemetry)
        for mapping in [_elliptic_payload_mapping(sample)]
        if mapping
    ]
    if not samples:
        raise ValueError("elliptic telemetry payload is empty")

    mode = aggregate.lower()

    fusion_inputs: list[Mapping[str, Any] | ZSpacePartialBundle] = []
    summary_values: list[float] = []
    has_scalar_metrics = False

    for payload in samples:
        sample_partial: dict[str, Any] = {}
        for canonical, source in _ELLIPTIC_METRIC_SOURCES.items():
            value = payload.get(source)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            sample_partial[canonical] = numeric
            summary_values.append(numeric)
            has_scalar_metrics = True

        vector_source: str | None = None
        vector_candidate: Any | None = None
        if gradient_source:
            vector_candidate = payload.get(gradient_source)
            if vector_candidate is not None:
                vector_source = gradient_source
        if vector_candidate is None:
            for candidate in _ELLIPTIC_VECTOR_CANDIDATES:
                vector_candidate = payload.get(candidate)
                if vector_candidate is not None:
                    vector_source = candidate
                    break
        gradient = _coerce_float_list(vector_candidate)
        if gradient:
            sample_partial["gradient"] = gradient
        if sample_partial:
            sample_basis = None
            if gradient:
                sample_basis = (
                    gradient_basis
                    if gradient_basis is not None
                    else _ELLIPTIC_GRADIENT_BASES.get(vector_source or "")
                )
                if sample_basis is None:
                    raise ValueError(
                        "gradient_basis is required for custom elliptic gradient sources"
                    )
            fusion_inputs.append(
                ZSpacePartialBundle(sample_partial, gradient_basis=sample_basis)
            )

    if not has_scalar_metrics:
        raise ValueError("no elliptic metrics could be derived from telemetry")
    fusion = zspace_partial_fusion(
        fusion_inputs,
        strategy=mode,
        gradient_alignment=gradient_alignment,
    )
    partial = _metrics_from_fusion_contract(fusion)

    telemetry_sources: list[Mapping[str, Any]] = list(samples)
    if extra_telemetry is not None:
        telemetry_sources.append(extra_telemetry)

    telemetry_map: dict[str, float] = {}
    if telemetry_sources:
        merged = _merge_telemetry_payloads(*telemetry_sources)
        if telemetry_prefix:
            telemetry_map = {
                f"{telemetry_prefix}.{key}": value for key, value in merged.items()
            }
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

    return ZSpacePartialBundle(
        partial,
        weight=float(bundle_weight),
        origin=origin,
        telemetry=telemetry_map or None,
        gradient_basis=fusion.get("gradient_basis"),
    )


@dataclass(frozen=True)
class ZSpaceControlGradient:
    """Basis-tagged external control kept separate from latent posterior gradients."""

    source: str
    basis: str
    values: tuple[float, ...]
    dimension: int
    l2: float
    linf: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "basis": self.basis,
            "values": list(self.values),
            "dimension": self.dimension,
            "l2": self.l2,
            "linf": self.linf,
        }


@dataclass(frozen=True)
class ZSpaceDecoded:
    """Full set of metrics reconstructed from a latent Z vector."""

    z_state: tuple[float, ...]
    metrics: Mapping[str, float]
    gradient: tuple[float, ...]
    gradient_basis: str
    barycentric: tuple[float, float, float]
    energy: float
    spectral_energy: float
    parseval_relative_error: float
    frac_energy: float
    fractional_energy_ratio: float
    spectral_centroid: float
    spectral_bins: int

    def as_dict(self) -> dict[str, Any]:
        data = {
            "z_state": list(self.z_state),
            "metrics": dict(self.metrics),
            "gradient": list(self.gradient),
            "gradient_basis": self.gradient_basis,
            "barycentric": self.barycentric,
            "energy": self.energy,
            "spectral_energy": self.spectral_energy,
            "parseval_relative_error": self.parseval_relative_error,
            "frac_energy": self.frac_energy,
            "fractional_energy_ratio": self.fractional_energy_ratio,
            "spectral_centroid": self.spectral_centroid,
            "spectral_bins": self.spectral_bins,
        }
        return data


def _posterior_sequence(
    payload: Mapping[str, Any], field: str, *, length: int | None = None
) -> tuple[float, ...]:
    raw = payload.get(field)
    if not isinstance(raw, SequenceABC) or isinstance(raw, (str, bytes, bytearray)):
        raise RuntimeError(f"native Z-space posterior field '{field}' is malformed")
    values = tuple(float(value) for value in raw)
    if length is not None and len(values) != length:
        raise RuntimeError(f"native Z-space posterior field '{field}' has invalid length")
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"native Z-space posterior field '{field}' is not finite")
    return values


def _posterior_scalar(payload: Mapping[str, Any], field: str) -> float:
    try:
        value = float(payload[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"native Z-space posterior field '{field}' is malformed"
        ) from exc
    if not math.isfinite(value):
        raise RuntimeError(f"native Z-space posterior field '{field}' is not finite")
    return value


def _posterior_nonnegative_integer(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"native Z-space posterior field '{field}' is malformed")
    return value


def _posterior_gradient_basis(payload: Mapping[str, Any]) -> str:
    basis = payload.get("gradient_basis")
    if basis != ZSPACE_POSTERIOR_LATENT_GRADIENT_BASIS:
        raise RuntimeError("native Z-space posterior returned an unknown latent gradient basis")
    return basis


def _control_gradient_from_posterior_contract(
    contract: Mapping[str, Any],
) -> ZSpaceControlGradient | None:
    payload = contract.get("control_gradient")
    if payload is None:
        return None
    if not isinstance(payload, MappingABC):
        raise RuntimeError("native Z-space posterior control gradient is malformed")
    values = _posterior_sequence(payload, "values")
    source = payload.get("source")
    basis = payload.get("basis")
    dimension = payload.get("dimension")
    if source != "partial" or not isinstance(basis, str) or not basis:
        raise RuntimeError("native Z-space posterior control gradient is untrusted")
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise RuntimeError("native Z-space posterior control gradient dimension is malformed")
    if dimension != len(values):
        raise RuntimeError("native Z-space posterior control gradient dimension does not match")
    return ZSpaceControlGradient(
        source=source,
        basis=basis,
        values=values,
        dimension=dimension,
        l2=_posterior_scalar(payload, "l2"),
        linf=_posterior_scalar(payload, "linf"),
    )


def _decoded_from_posterior_contract(
    contract: Mapping[str, Any],
) -> ZSpaceDecoded:
    _validate_zspace_posterior_contract(
        contract, expected_kind=_ZSPACE_POSTERIOR_DECODE_KIND
    )
    metrics = contract.get("metrics")
    if not isinstance(metrics, MappingABC):
        raise RuntimeError("native Z-space posterior metrics are malformed")
    z_state = _posterior_sequence(contract, "z_state")
    gradient = _posterior_sequence(contract, "gradient", length=len(z_state))
    gradient_basis = _posterior_gradient_basis(contract)
    barycentric = _posterior_sequence(contract, "barycentric", length=3)
    spectral_bins = _posterior_nonnegative_integer(contract, "spectral_bins")
    if spectral_bins != len(z_state) // 2 + 1:
        raise RuntimeError("native Z-space posterior spectral bin count is malformed")
    return ZSpaceDecoded(
        z_state=z_state,
        metrics=MappingProxyType(
            {str(key): float(value) for key, value in metrics.items()}
        ),
        gradient=gradient,
        gradient_basis=gradient_basis,
        barycentric=(barycentric[0], barycentric[1], barycentric[2]),
        energy=_posterior_scalar(contract, "energy"),
        spectral_energy=_posterior_scalar(contract, "spectral_energy"),
        parseval_relative_error=_posterior_scalar(contract, "parseval_relative_error"),
        frac_energy=_posterior_scalar(contract, "frac_energy"),
        fractional_energy_ratio=_posterior_scalar(
            contract, "fractional_energy_ratio"
        ),
        spectral_centroid=_posterior_scalar(contract, "spectral_centroid"),
        spectral_bins=spectral_bins,
    )


def zspace_posterior_decode(
    z_state: Sequence[float], *, alpha: float = 0.35
) -> dict[str, Any]:
    """Decode a latent state through the canonical Rust posterior contract."""

    request = {
        "z_state": [float(value) for value in z_state],
        "alpha": float(alpha),
    }
    return _native_zspace_posterior(
        "_zspace_posterior_decode",
        request,
        expected_kind=_ZSPACE_POSTERIOR_DECODE_KIND,
    )


def zspace_posterior_project(
    z_state: Sequence[float],
    partial: Mapping[str, Any] | None = None,
    *,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    gradient_basis: str | None = None,
    telemetry: Mapping[str, Any] | ZSpaceTelemetryFrame | None = None,
) -> dict[str, Any]:
    """Project partial observations through the canonical Rust posterior contract."""

    if partial is None:
        partial_input: dict[str, Any] = {}
    elif isinstance(partial, Mapping):
        partial, gradient_basis = _split_partial_gradient_basis(
            partial, gradient_basis
        )
        partial_input = {}
        for name, value in partial.items():
            if not isinstance(name, str):
                raise TypeError("partial observation names must be strings")
            partial_input[name] = _metric_input(value)
    else:
        raise TypeError("partial observations must be provided as a mapping")
    request = {
        "z_state": [float(value) for value in z_state],
        "alpha": float(alpha),
        "partial": partial_input,
        "gradient_basis": gradient_basis,
        "smoothing": float(smoothing),
        "telemetry": _telemetry_inputs([telemetry]),
    }
    return _native_zspace_posterior(
        "_zspace_posterior_project",
        request,
        expected_kind=_ZSPACE_POSTERIOR_PROJECTION_KIND,
    )


@dataclass(frozen=True)
class ZSpaceInference:
    """Inference result after fusing partial observations with the decoded state."""

    metrics: Mapping[str, float]
    gradient: Sequence[float]
    gradient_basis: str
    control_gradient: ZSpaceControlGradient | None
    barycentric: tuple[float, float, float]
    residual: float
    residual_metric_count: int
    confidence: float
    telemetry_reliability: float
    prior: ZSpaceDecoded
    applied: Mapping[str, Any]
    telemetry: ZSpaceTelemetryFrame | None = None
    fusion: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "gradient": list(self.gradient),
            "gradient_basis": self.gradient_basis,
            "control_gradient": (
                None if self.control_gradient is None else self.control_gradient.as_dict()
            ),
            "barycentric": self.barycentric,
            "residual": self.residual,
            "residual_metric_count": self.residual_metric_count,
            "confidence": self.confidence,
            "telemetry_reliability": self.telemetry_reliability,
            "applied": dict(self.applied),
            "prior": self.prior.as_dict(),
            "telemetry": None if self.telemetry is None else self.telemetry.as_dict(),
            "fusion": None if self.fusion is None else dict(self.fusion),
        }


class ZSpacePosterior:
    """Posterior over Z-space metrics conditioned on a latent state."""

    def __init__(self, z_state: Sequence[float], *, alpha: float = 0.35) -> None:
        contract = zspace_posterior_decode(z_state, alpha=alpha)
        self._decoded = _decoded_from_posterior_contract(contract)
        self._z_state = self._decoded.z_state
        self._alpha = float(contract["alpha"])

    @property
    def z_state(self) -> list[float]:
        return list(self._z_state)

    @property
    def alpha(self) -> float:
        return self._alpha

    def decode(self) -> ZSpaceDecoded:
        return self._decoded

    def project(
        self,
        partial: Mapping[str, Any] | None,
        *,
        smoothing: float = 0.35,
        gradient_basis: str | None = None,
        telemetry: Mapping[str, Any] | ZSpaceTelemetryFrame | None = None,
    ) -> ZSpaceInference:
        contract = zspace_posterior_project(
            self._z_state,
            partial,
            alpha=self._alpha,
            smoothing=smoothing,
            gradient_basis=gradient_basis,
            telemetry=telemetry,
        )
        metrics = contract.get("metrics")
        applied = contract.get("applied")
        prior = contract.get("prior")
        if not isinstance(metrics, MappingABC) or not isinstance(applied, MappingABC):
            raise RuntimeError("native Z-space posterior projection is malformed")
        if not isinstance(prior, MappingABC):
            raise RuntimeError("native Z-space posterior prior is malformed")
        gradient = _posterior_sequence(
            contract, "gradient", length=len(self._z_state)
        )
        gradient_basis = _posterior_gradient_basis(contract)
        control_gradient = _control_gradient_from_posterior_contract(contract)
        barycentric = _posterior_sequence(contract, "barycentric", length=3)
        telemetry_contract = contract.get("telemetry")
        telemetry_frame = None
        if telemetry_contract is not None:
            if not isinstance(telemetry_contract, MappingABC):
                raise RuntimeError("native Z-space posterior telemetry is malformed")
            telemetry_frame = _telemetry_frame_from_fusion_contract(telemetry_contract)
        return ZSpaceInference(
            metrics=MappingProxyType(
                {str(key): float(value) for key, value in metrics.items()}
            ),
            gradient=list(gradient),
            gradient_basis=gradient_basis,
            control_gradient=control_gradient,
            barycentric=(barycentric[0], barycentric[1], barycentric[2]),
            residual=_posterior_scalar(contract, "residual"),
            residual_metric_count=_posterior_nonnegative_integer(
                contract, "residual_metric_count"
            ),
            confidence=_posterior_scalar(contract, "confidence"),
            telemetry_reliability=_posterior_scalar(
                contract, "telemetry_reliability"
            ),
            prior=_decoded_from_posterior_contract(prior),
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
        self._gradient_basis: str | None = None
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

    @property
    def gradient_basis(self) -> str | None:
        """Return the basis attached to the cached external control gradient."""

        return self._gradient_basis

    def clear(self) -> None:
        """Forget any cached observations."""

        self._cached.clear()
        self._gradient_basis = None

    def set_telemetry(
        self, telemetry: Mapping[str, Any] | ZSpaceTelemetryFrame | None
    ) -> None:
        """Replace the cached telemetry payload used during inference."""

        self._telemetry = _merge_telemetry_payloads(telemetry)

    def _merge(
        self,
        partial: Mapping[str, Any] | None,
        *,
        gradient_basis: str | None,
    ) -> Mapping[str, Any] | None:
        if partial is None:
            if not self._cached:
                return None
            return self._cached
        partial, gradient_basis = _split_partial_gradient_basis(
            partial, gradient_basis
        )
        updates = _canonicalise_inputs(partial, gradient_basis=gradient_basis)
        if not self._accumulate:
            self._cached = {}
            self._gradient_basis = None
        if "gradient" in updates:
            gradient = updates.pop("gradient")
            if gradient is not None:
                self._cached["gradient"] = gradient
                self._gradient_basis = gradient_basis
            else:
                self._cached.pop("gradient", None)
                self._gradient_basis = None
        for key, value in updates.items():
            self._cached[key] = value
        return self._cached

    def update(
        self,
        partial: Mapping[str, Any] | None = None,
        *,
        gradient_basis: str | None = None,
        telemetry: Mapping[str, Any] | ZSpaceTelemetryFrame | None = None,
    ) -> ZSpaceInference:
        """Fuse *partial* with any cached observations and produce an inference."""

        if telemetry is not None:
            self._telemetry = _merge_telemetry_payloads(self._telemetry, telemetry)
        merged = self._merge(partial, gradient_basis=gradient_basis)
        payload = self._telemetry if self._telemetry else None
        return self._posterior.project(
            merged,
            smoothing=self._smoothing,
            gradient_basis=self._gradient_basis,
            telemetry=payload,
        )

    def infer(
        self,
        partial: Mapping[str, Any] | None = None,
        *,
        gradient_basis: str | None = None,
        telemetry: Mapping[str, Any] | ZSpaceTelemetryFrame | None = None,
    ) -> ZSpaceInference:
        """Alias for :meth:`update` to mirror the functional helpers."""

        return self.update(
            partial,
            gradient_basis=gradient_basis,
            telemetry=telemetry,
        )


class ZSpaceInferencePipeline:
    """Composable pipeline that blends heterogeneous partials before inference."""

    def __init__(
        self,
        z_state: Sequence[float],
        *,
        alpha: float = 0.35,
        smoothing: float = 0.35,
        strategy: str = "mean",
        gradient_alignment: str = "strict",
        metric_gradient_dimension: int | None = None,
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
        self._gradient_alignment = gradient_alignment
        self._metric_gradient_dimension = metric_gradient_dimension
        self._partials: list[ZSpacePartialBundle] = []

    @property
    def strategy(self) -> str:
        """Return the blending strategy used for partial fusion."""

        return self._strategy

    @property
    def gradient_alignment(self) -> str:
        """Return the Rust fusion policy used for active gradient dimensions."""

        return self._gradient_alignment

    @property
    def metric_gradient_dimension(self) -> int | None:
        """Return the Rust-owned post-fusion metric projection width, if enabled."""

        return self._metric_gradient_dimension

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
        gradient_basis: str | None = None,
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
                gradient_basis=gradient_basis,
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
        gradient_alignment: str | None = None,
        gradient_source: str = "rotor_transport",
        gradient_basis: str | None = None,
        extra_telemetry: Mapping[str, Any] | None = None,
    ) -> ZSpacePartialBundle:
        """Register elliptic telemetry samples as a partial observation."""

        bundle = elliptic_partial_from_telemetry(
            telemetry,
            bundle_weight=bundle_weight,
            origin=origin,
            telemetry_prefix=telemetry_prefix,
            aggregate=aggregate,
            gradient_alignment=(
                self._gradient_alignment
                if gradient_alignment is None
                else gradient_alignment
            ),
            gradient_source=gradient_source,
            gradient_basis=gradient_basis,
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
        gradient_alignment: str | None = None,
        gradient_source: str = "rotor_transport",
        gradient_basis: str | None = None,
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
            gradient_alignment=(
                self._gradient_alignment
                if gradient_alignment is None
                else gradient_alignment
            ),
            gradient_source=gradient_source,
            gradient_basis=gradient_basis,
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

    def add_topos_control(
        self,
        topos: Any | None = None,
        *,
        bundle_weight: float = 1.0,
        origin: str | None = "topos:control",
        telemetry_prefix: str = "topos",
        gradient_dim: int = 6,
        **signal_options: Any,
    ) -> ZSpacePartialBundle:
        """Register an open-topos pressure signal as queued inference context."""

        bundle = topos_control_partial(
            topos,
            bundle_weight=bundle_weight,
            origin=origin,
            telemetry_prefix=telemetry_prefix,
            gradient_dim=gradient_dim,
            **signal_options,
        )
        return self.add_partial(bundle)

    def add_geometry_probes(
        self,
        probes: Any,
        *,
        max_probes: int | None = None,
        bundle_weight: float = 1.0,
        telemetry_prefix: str = "geometry",
        gradient_dim: int = 8,
        include_consensus: bool = False,
        consensus_only: bool = False,
        consensus_weight: float | None = None,
        consensus_strategy: str = "mean",
        consensus_origin: str = "geometry:consensus",
        return_metadata: bool = False,
    ) -> list[ZSpacePartialBundle] | tuple[list[ZSpacePartialBundle], dict[str, Any]]:
        """Register WASM geometry probes as queued partial observations."""

        from .geometry_context import build_geometry_probe_context

        partials, metadata = build_geometry_probe_context(
            probes,
            max_probes=max_probes,
            bundle_weight=bundle_weight,
            telemetry_prefix=telemetry_prefix,
            gradient_dim=gradient_dim,
            include_consensus=include_consensus,
            consensus_only=consensus_only,
            consensus_weight=consensus_weight,
            consensus_strategy=consensus_strategy,
            consensus_origin=consensus_origin,
        )
        for bundle in partials:
            self.add_partial(bundle)
        if return_metadata:
            return partials, metadata
        return partials

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
        gradient_alignment: str | None = None,
        weights: Sequence[float] | None = None,
        clear: bool = True,
        telemetry: Mapping[str, Any] | ZSpaceTelemetryFrame | None = None,
    ) -> ZSpaceInference:
        """Blend registered partials and compute the Z-space inference."""

        chosen_strategy = self._strategy if strategy is None else strategy
        chosen_gradient_alignment = (
            self._gradient_alignment
            if gradient_alignment is None
            else gradient_alignment
        )
        fusion = zspace_partial_fusion(
            self._partials,
            strategy=chosen_strategy,
            gradient_alignment=chosen_gradient_alignment,
            metric_gradient_dimension=self._metric_gradient_dimension,
            weights=weights,
            telemetry=telemetry,
        )
        blended = _metrics_from_fusion_contract(fusion)
        merged_telemetry = dict(fusion["telemetry"]["payload"])
        inference = replace(
            self._runtime.update(
                blended,
                gradient_basis=fusion.get("gradient_basis"),
                telemetry=merged_telemetry if merged_telemetry else None,
            ),
            fusion=MappingProxyType(dict(fusion)),
        )
        if clear:
            self.clear()
        return inference

    def infer_and_step(
        self,
        trainer: Any,
        *,
        strategy: str | None = None,
        gradient_alignment: str | None = None,
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
            gradient_alignment: Optional Rust gradient-alignment policy override.
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
            gradient_alignment=gradient_alignment,
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


def decode_zspace_embedding(
    z_state: Sequence[float], *, alpha: float = 0.35
) -> ZSpaceDecoded:
    """Decode latent coordinates into a structured metric bundle."""

    return ZSpacePosterior(z_state, alpha=alpha).decode()


def infer_from_partial(
    z_state: Sequence[float],
    partial: Mapping[str, Any] | None,
    *,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    gradient_basis: str | None = None,
    telemetry: Mapping[str, Any] | ZSpaceTelemetryFrame | None = None,
) -> ZSpaceInference:
    """Fuse partial metric observations with a latent state to complete Z-space inference."""

    posterior = ZSpacePosterior(z_state, alpha=alpha)
    return posterior.project(
        partial,
        smoothing=smoothing,
        gradient_basis=gradient_basis,
        telemetry=telemetry,
    )


def infer_with_partials(
    z_state: Sequence[float],
    *partials: Mapping[str, Any] | ZSpacePartialBundle | None,
    alpha: float = 0.35,
    smoothing: float = 0.35,
    strategy: str = "mean",
    gradient_alignment: str = "strict",
    metric_gradient_dimension: int | None = None,
    weights: Sequence[float] | None = None,
    telemetry: Mapping[str, Any] | ZSpaceTelemetryFrame | None = None,
) -> ZSpaceInference:
    """Infer Z-space metrics from multiple partial observations."""

    fusion = zspace_partial_fusion(
        partials,
        weights=weights,
        strategy=strategy,
        gradient_alignment=gradient_alignment,
        metric_gradient_dimension=metric_gradient_dimension,
        telemetry=telemetry,
    )
    blended = _metrics_from_fusion_contract(fusion)
    merged_telemetry = dict(fusion["telemetry"]["payload"])
    return replace(
        infer_from_partial(
            z_state,
            blended,
            alpha=alpha,
            smoothing=smoothing,
            gradient_basis=fusion.get("gradient_basis"),
            telemetry=merged_telemetry if merged_telemetry else None,
        ),
        fusion=MappingProxyType(dict(fusion)),
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
    gradient_basis: str | None = None,
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
            actual,
            alpha=alpha,
            smoothing=smoothing,
            gradient_basis=gradient_basis,
        )

    if not callable(fn):
        raise TypeError(
            "compile_inference expects a callable or to be used as a decorator"
        )

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
            gradient_basis=gradient_basis,
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


def _merge_summary(
    stats: dict[str, float], summary: Mapping[str, Any] | None
) -> dict[str, float]:
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
    frac_source = (
        float(patch.get("linf", canvas.get("linf", 0.0)))
        if patch
        else float(canvas.get("linf", 0.0))
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
    gradient_alignment: str = "strict",
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
    return blend_zspace_partials(
        bundles,
        strategy=strategy,
        gradient_alignment=gradient_alignment,
        weights=weights,
    )


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
        raise AttributeError(
            "canvas object must expose a snapshot() method or property"
        )
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


_COHERENCE_MISSING = object()


def _coherence_source_value(source: Any, *names: str) -> Any:
    if isinstance(source, MappingABC):
        for name in names:
            if name in source:
                return _maybe_call(source[name])
    else:
        for name in names:
            value = getattr(source, name, _COHERENCE_MISSING)
            if value is not _COHERENCE_MISSING:
                return _maybe_call(value)
    return _COHERENCE_MISSING


def _coherence_sequence(values: Any, *, field: str) -> list[float]:
    values = _maybe_call(values)
    if values is None or values is _COHERENCE_MISSING:
        return []
    if isinstance(values, MappingABC):
        values = values.values()
    if isinstance(values, (str, bytes, bytearray, memoryview)) or not isinstance(
        values, Iterable
    ):
        raise TypeError(f"coherence field '{field}' must be a sequence")
    return [float(value) for value in values]


def _coherence_count(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"coherence field '{field}' must be a non-negative integer")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise ValueError(f"coherence field '{field}' must be a non-negative integer")
    return int(numeric)


def zspace_coherence_project(
    diagnostics: Any,
    *,
    coherence: Any = None,
    contour: Any = None,
    speed_gain: float = 1.0,
    stability_gain: float = 1.0,
    frac_gain: float = 1.0,
    drs_gain: float = 1.0,
    background_energy_ratio_max: float = 1.0e-5,
    cascade_energy_ratio_min: float = 0.7,
) -> dict[str, Any]:
    """Project coherence diagnostics through the canonical Rust contract."""

    diagnostics_input: dict[str, Any] = {}
    for canonical, aliases in (
        ("mean_coherence", ("mean_coherence",)),
        ("coherence_entropy", ("coherence_entropy", "entropy")),
        ("energy_ratio", ("energy_ratio",)),
        ("z_bias", ("z_bias",)),
        ("fractional_order", ("fractional_order",)),
    ):
        value = _coherence_source_value(diagnostics, *aliases)
        if value is _COHERENCE_MISSING or value is None:
            raise ValueError(
                f"coherence diagnostics are missing required field '{canonical}'"
            )
        diagnostics_input[canonical] = float(value)

    weights = _coherence_source_value(diagnostics, "normalized_weights")
    diagnostics_input["normalized_weights"] = _coherence_sequence(
        weights, field="normalized_weights"
    )
    for field in ("preserved_channels", "discarded_channels", "dominant_channel"):
        value = _coherence_source_value(diagnostics, field)
        if value is not _COHERENCE_MISSING and value is not None:
            diagnostics_input[field] = _coherence_count(value, field=field)

    response_source = coherence
    if response_source is None:
        response_source = _coherence_source_value(diagnostics, "coherence")

    contour_input: dict[str, float] | None = None
    if contour is not None:
        contour_input = {}
        for canonical, source_name in (
            ("coherence_strength", "coherence_strength"),
            ("prosody_index", "prosody_index"),
            ("articulation_bias", "articulation_bias"),
        ):
            value = _coherence_source_value(contour, source_name)
            if value is _COHERENCE_MISSING or value is None:
                raise ValueError(
                    f"linguistic contour is missing required field '{source_name}'"
                )
            contour_input[canonical] = float(value)
        timbre = _coherence_source_value(contour, "timbre_spread")
        if timbre is not _COHERENCE_MISSING and timbre is not None:
            contour_input["timbre_spread"] = float(timbre)

    request = {
        "diagnostics": diagnostics_input,
        "coherence": _coherence_sequence(response_source, field="coherence"),
        "contour": contour_input,
        "config": {
            "speed_gain": float(speed_gain),
            "stability_gain": float(stability_gain),
            "frac_gain": float(frac_gain),
            "drs_gain": float(drs_gain),
        },
        "classification_policy": {
            "background_energy_ratio_max": float(background_energy_ratio_max),
            "cascade_energy_ratio_min": float(cascade_energy_ratio_min),
        },
    }
    return _native_zspace_coherence_projection(request)


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
    """Return partial metrics from the canonical Rust coherence contract."""

    contract = zspace_coherence_project(
        diagnostics,
        coherence=coherence,
        contour=contour,
        speed_gain=speed_gain,
        stability_gain=stability_gain,
        frac_gain=frac_gain,
        drs_gain=drs_gain,
    )
    return {str(key): float(value) for key, value in contract["partial"].items()}


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
    gradient_alignment: str = "strict",
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
        gradient_alignment=gradient_alignment,
        weights=weights,
        canvas_kwargs=canvas_kwargs,
        coherence_kwargs=coherence_kwargs,
    )
    return infer_from_partial(z_state, partial, alpha=alpha, smoothing=smoothing)
