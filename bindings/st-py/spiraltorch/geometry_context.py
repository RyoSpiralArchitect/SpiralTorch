from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .fractal_field_probe import fractal_field_probe_to_zspace_partial
from .log_z_series_probe import log_z_series_probe_to_zspace_partial
from .scale_stack_probe import scale_stack_probe_to_zspace_partial
from .zspace_inference import ZSpacePartialBundle, blend_zspace_partials

__all__ = [
    "build_geometry_probe_context",
    "build_geometry_probe_context_artifact",
    "geometry_probe_consensus_partial",
    "geometry_probe_summary",
    "geometry_probe_to_zspace_partial",
    "geometry_probes_to_zspace_partials",
    "load_geometry_probe_context_artifact",
    "write_geometry_probe_context_artifact",
]

_GEOMETRY_CONTEXT_SCHEMA = "spiraltorch.geometry_probe_context.v1"
_CONSENSUS_STRATEGIES = {"mean": 1.0, "last": 2.0, "max": 3.0, "min": 4.0}

_ROUTES = {
    "spiraltorch.wasm_scale_stack_probe": (
        "scale_stack",
        "st-frac::scale_stack",
        scale_stack_probe_to_zspace_partial,
    ),
    "spiraltorch.wasm_fractal_field_probe": (
        "fractal_field",
        "st-frac::fractal_field",
        fractal_field_probe_to_zspace_partial,
    ),
    "spiraltorch.wasm_log_z_series_probe": (
        "log_z_series",
        "st-frac::cosmology",
        log_z_series_probe_to_zspace_partial,
    ),
}


def _probe_kind(probe: Mapping[str, Any]) -> str:
    return str(probe.get("kind", ""))


def _route_for_probe(probe: Mapping[str, Any]):
    kind = _probe_kind(probe)
    route = _ROUTES.get(kind)
    if route is None:
        known = ", ".join(sorted(_ROUTES))
        raise ValueError(f"unsupported geometry probe kind '{kind}', expected one of: {known}")
    return route


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric == numeric and abs(numeric) != float("inf") else None


def _count_from_mapping(payload: Any, key: str) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    value = _finite_float(payload.get(key))
    return None if value is None else max(0, int(value))


def _iter_probe_items(probes: Any) -> list[tuple[str | None, Mapping[str, Any]]]:
    if probes is None:
        return []
    if isinstance(probes, Mapping):
        if "kind" in probes or "source_crate" in probes:
            return [(None, probes)]
        return [
            (str(label), probe)
            for label, probe in probes.items()
            if isinstance(probe, Mapping)
        ]
    if isinstance(probes, (str, bytes, bytearray, os.PathLike)):
        raise TypeError("geometry probes must be mappings or sequences of mappings")
    try:
        iterator = iter(probes)
    except TypeError as exc:
        raise TypeError("geometry probes must be mappings or sequences of mappings") from exc
    items: list[tuple[str | None, Mapping[str, Any]]] = []
    for probe in iterator:
        if probe is None:
            continue
        if not isinstance(probe, Mapping):
            raise TypeError("geometry probe sequences must contain mapping entries")
        items.append((None, probe))
    return items


def _label_for(index: int, label: str | None, family: str) -> str:
    return label if label else f"{family}-{index}"


def _telemetry_prefix(base: str, family: str, index: int) -> str:
    base = base or "geometry"
    return f"{base}.{family}.{index}"


def _normalise_consensus_strategy(strategy: str) -> str:
    value = str(strategy or "mean").strip().lower()
    if value not in _CONSENSUS_STRATEGIES:
        known = ", ".join(sorted(_CONSENSUS_STRATEGIES))
        raise ValueError(
            f"unsupported geometry consensus strategy '{strategy}', expected one of: {known}"
        )
    return value


def _partial_payload(partial: ZSpacePartialBundle) -> dict[str, Any]:
    telemetry = partial.telemetry_payload()
    return {
        "metrics": partial.resolved(),
        "weight": partial.weight,
        "origin": partial.origin,
        "telemetry": None if telemetry is None else dict(telemetry),
    }


def _partial_from_payload(payload: Any) -> ZSpacePartialBundle:
    if not isinstance(payload, Mapping):
        raise ValueError("geometry context partial rows must be JSON objects")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("geometry context partial rows must contain metrics")
    telemetry = payload.get("telemetry")
    if telemetry is not None and not isinstance(telemetry, Mapping):
        raise ValueError("geometry context partial telemetry must be an object")
    return ZSpacePartialBundle(
        dict(metrics),
        weight=float(payload.get("weight", 1.0)),
        origin=None if payload.get("origin") is None else str(payload.get("origin")),
        telemetry=None if telemetry is None else dict(telemetry),
    )


def geometry_probe_summary(
    probe: Mapping[str, Any],
    *,
    label: str | None = None,
) -> dict[str, Any]:
    """Return a compact metadata row for a supported WASM geometry probe."""

    if not isinstance(probe, Mapping):
        raise TypeError("geometry probes must be mappings")
    family, expected_source, _converter = _route_for_probe(probe)
    source = str(probe.get("source_crate", ""))
    sample_count = _count_from_mapping(probe, "sample_count")
    projection = probe.get("projection")
    summary: dict[str, Any] = {
        "label": label,
        "kind": _probe_kind(probe),
        "family": family,
        "source_crate": source,
        "source_expected": source == expected_source,
        "mode": probe.get("mode"),
        "sample_count": sample_count,
    }
    if family == "scale_stack":
        summary["threshold"] = _finite_float(probe.get("threshold"))
        summary["interface_density"] = _finite_float(probe.get("interface_density"))
        summary["coherence_break_count"] = _count_from_mapping(
            {"value": len(probe.get("coherence_profile", ()) or ())},
            "value",
        )
    elif family == "fractal_field":
        summary["energy"] = _finite_float(probe.get("energy"))
        summary["total_variation"] = _finite_float(probe.get("total_variation"))
        summary["coherence_score"] = _finite_float(probe.get("coherence_score"))
    elif family == "log_z_series":
        summary["z_count"] = _count_from_mapping(probe, "z_count")
        if isinstance(projection, Mapping):
            summary["projection_energy"] = _finite_float(projection.get("energy"))
            summary["projection_stability"] = _finite_float(
                projection.get("stability_score")
            )
    return summary


def geometry_probe_to_zspace_partial(
    probe: Mapping[str, Any],
    *,
    bundle_weight: float = 1.0,
    origin: str | None = None,
    telemetry_prefix: str = "geometry",
    gradient_dim: int = 8,
) -> ZSpacePartialBundle:
    """Route one supported WASM geometry probe into a Z-space partial bundle."""

    if not isinstance(probe, Mapping):
        raise TypeError("geometry probes must be mappings")
    family, _source, converter = _route_for_probe(probe)
    return converter(
        probe,
        bundle_weight=bundle_weight,
        origin=origin,
        telemetry_prefix=telemetry_prefix or family,
        gradient_dim=gradient_dim,
    )


def geometry_probes_to_zspace_partials(
    probes: Any,
    *,
    bundle_weight: float = 1.0,
    telemetry_prefix: str = "geometry",
    gradient_dim: int = 8,
) -> list[ZSpacePartialBundle]:
    """Convert one or more geometry probes into Z-space partial bundles."""

    partials: list[ZSpacePartialBundle] = []
    for index, (label, probe) in enumerate(_iter_probe_items(probes), start=1):
        family, _source, _converter = _route_for_probe(probe)
        origin = f"geometry:{_label_for(index, label, family)}"
        partials.append(
            geometry_probe_to_zspace_partial(
                probe,
                bundle_weight=bundle_weight,
                origin=origin,
                telemetry_prefix=_telemetry_prefix(telemetry_prefix, family, index),
                gradient_dim=gradient_dim,
            )
        )
    return partials


def _consensus_telemetry(
    metadata: Mapping[str, Any],
    *,
    telemetry_prefix: str,
    strategy: str,
    metric_count: int,
) -> dict[str, float]:
    prefix = telemetry_prefix or "geometry"
    consensus_prefix = f"{prefix}.consensus"
    families = metadata.get("families")
    family_counts = dict(families) if isinstance(families, Mapping) else {}
    telemetry: dict[str, float] = {
        f"{consensus_prefix}.probe_count": float(metadata.get("probe_count", 0)),
        f"{consensus_prefix}.candidate_count": float(
            metadata.get("candidate_count", 0)
        ),
        f"{consensus_prefix}.family_count": float(len(family_counts)),
        f"{consensus_prefix}.metric_count": float(metric_count),
        f"{consensus_prefix}.gradient_dim": float(metadata.get("gradient_dim", 0)),
        f"{consensus_prefix}.strategy_code": _CONSENSUS_STRATEGIES[strategy],
    }
    for family, count in sorted(family_counts.items()):
        telemetry[f"{consensus_prefix}.family_{family}_count"] = float(count)
    return telemetry


def _consensus_partial_from_context(
    partials: Sequence[ZSpacePartialBundle],
    metadata: Mapping[str, Any],
    *,
    bundle_weight: float,
    telemetry_prefix: str,
    strategy: str,
    origin: str,
) -> ZSpacePartialBundle:
    strategy = _normalise_consensus_strategy(strategy)
    metrics = blend_zspace_partials(partials, strategy=strategy)
    telemetry = _consensus_telemetry(
        metadata,
        telemetry_prefix=telemetry_prefix,
        strategy=strategy,
        metric_count=len(metrics),
    )
    return ZSpacePartialBundle(
        metrics,
        weight=max(0.0, float(bundle_weight)),
        origin=origin or "geometry:consensus",
        telemetry=telemetry,
    )


def geometry_probe_consensus_partial(
    probes: Any,
    *,
    max_probes: int | None = None,
    bundle_weight: float = 1.0,
    consensus_weight: float | None = None,
    telemetry_prefix: str = "geometry",
    gradient_dim: int = 8,
    strategy: str = "mean",
    origin: str = "geometry:consensus",
) -> tuple[ZSpacePartialBundle, dict[str, Any]]:
    """Fuse supported WASM geometry probes into one runtime-ready partial."""

    partials, metadata = build_geometry_probe_context(
        probes,
        max_probes=max_probes,
        bundle_weight=bundle_weight,
        telemetry_prefix=telemetry_prefix,
        gradient_dim=gradient_dim,
        include_consensus=False,
    )
    if not partials:
        raise ValueError("at least one geometry probe is required for consensus")
    consensus = _consensus_partial_from_context(
        partials,
        metadata,
        bundle_weight=bundle_weight if consensus_weight is None else consensus_weight,
        telemetry_prefix=telemetry_prefix,
        strategy=strategy,
        origin=origin,
    )
    consensus_metadata = dict(metadata)
    consensus_metadata["consensus"] = {
        "origin": consensus.origin,
        "strategy": _normalise_consensus_strategy(strategy),
        "weight": consensus.weight,
        "metric_count": len(consensus.resolved()),
    }
    return consensus, consensus_metadata


def build_geometry_probe_context(
    probes: Any,
    *,
    max_probes: int | None = None,
    bundle_weight: float = 1.0,
    telemetry_prefix: str = "geometry",
    gradient_dim: int = 8,
    include_consensus: bool = False,
    consensus_weight: float | None = None,
    consensus_strategy: str = "mean",
    consensus_origin: str = "geometry:consensus",
) -> tuple[list[ZSpacePartialBundle], dict[str, Any]]:
    """Build Z-space context partials plus metadata from WASM geometry probes."""

    candidate_items = _iter_probe_items(probes)
    selected_items = (
        candidate_items
        if max_probes is None
        else candidate_items[: max(0, int(max_probes))]
    )
    partials: list[ZSpacePartialBundle] = []
    summaries: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}

    for index, (label, probe) in enumerate(selected_items, start=1):
        summary = geometry_probe_summary(probe, label=label)
        family = str(summary["family"])
        source = str(summary.get("source_crate") or "unknown")
        kind = str(summary.get("kind") or "unknown")
        origin = f"geometry:{_label_for(index, label, family)}"
        partial = geometry_probe_to_zspace_partial(
            probe,
            bundle_weight=bundle_weight,
            origin=origin,
            telemetry_prefix=_telemetry_prefix(telemetry_prefix, family, index),
            gradient_dim=gradient_dim,
        )
        partials.append(partial)
        summaries.append(summary)
        family_counts[family] = family_counts.get(family, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    metadata: dict[str, Any] = {
        "candidate_count": len(candidate_items),
        "probe_count": len(selected_items),
        "gradient_dim": int(gradient_dim),
        "bundle_weight": float(bundle_weight),
        "telemetry_prefix": telemetry_prefix,
        "selection": {"max_probes": max_probes},
        "families": family_counts,
        "sources": source_counts,
        "kinds": kind_counts,
        "probes": summaries,
        "context_origins": [partial.origin for partial in partials],
    }
    if include_consensus and partials:
        consensus = _consensus_partial_from_context(
            partials,
            metadata,
            bundle_weight=bundle_weight if consensus_weight is None else consensus_weight,
            telemetry_prefix=telemetry_prefix,
            strategy=consensus_strategy,
            origin=consensus_origin,
        )
        partials.append(consensus)
        metadata["consensus"] = {
            "origin": consensus.origin,
            "strategy": _normalise_consensus_strategy(consensus_strategy),
            "weight": consensus.weight,
            "metric_count": len(consensus.resolved()),
        }
        metadata["context_origins"] = [partial.origin for partial in partials]
    elif include_consensus:
        metadata["consensus"] = None
    return partials, metadata


def build_geometry_probe_context_artifact(
    probes: Any,
    *,
    max_probes: int | None = None,
    bundle_weight: float = 1.0,
    telemetry_prefix: str = "geometry",
    gradient_dim: int = 8,
    include_consensus: bool = False,
    consensus_weight: float | None = None,
    consensus_strategy: str = "mean",
    consensus_origin: str = "geometry:consensus",
) -> dict[str, Any]:
    """Return a portable JSON-ready artifact for geometry-probe context partials."""

    partials, metadata = build_geometry_probe_context(
        probes,
        max_probes=max_probes,
        bundle_weight=bundle_weight,
        telemetry_prefix=telemetry_prefix,
        gradient_dim=gradient_dim,
        include_consensus=include_consensus,
        consensus_weight=consensus_weight,
        consensus_strategy=consensus_strategy,
        consensus_origin=consensus_origin,
    )
    return {
        "schema": _GEOMETRY_CONTEXT_SCHEMA,
        "kind": "spiraltorch.geometry_probe_context",
        "metadata": metadata,
        "context_partials": [_partial_payload(partial) for partial in partials],
    }


def write_geometry_probe_context_artifact(
    path: str | os.PathLike[str],
    probes: Any,
    *,
    max_probes: int | None = None,
    bundle_weight: float = 1.0,
    telemetry_prefix: str = "geometry",
    gradient_dim: int = 8,
    include_consensus: bool = False,
    consensus_weight: float | None = None,
    consensus_strategy: str = "mean",
    consensus_origin: str = "geometry:consensus",
) -> str:
    """Write a geometry-probe context handoff artifact and return its path."""

    artifact = build_geometry_probe_context_artifact(
        probes,
        max_probes=max_probes,
        bundle_weight=bundle_weight,
        telemetry_prefix=telemetry_prefix,
        gradient_dim=gradient_dim,
        include_consensus=include_consensus,
        consensus_weight=consensus_weight,
        consensus_strategy=consensus_strategy,
        consensus_origin=consensus_origin,
    )
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(out_path)


def load_geometry_probe_context_artifact(
    path: str | os.PathLike[str],
) -> tuple[list[ZSpacePartialBundle], dict[str, Any]]:
    """Load a geometry-probe context artifact into partials plus metadata."""

    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("geometry context artifact must be a JSON object")
    partial_rows = payload.get("context_partials")
    if not isinstance(partial_rows, Sequence) or isinstance(
        partial_rows, (str, bytes, bytearray)
    ):
        raise ValueError("geometry context artifact must contain context_partials")
    metadata = payload.get("metadata")
    if metadata is None:
        metadata_payload: dict[str, Any] = {}
    elif isinstance(metadata, Mapping):
        metadata_payload = dict(metadata)
    else:
        raise ValueError("geometry context artifact metadata must be an object")
    metadata_payload.setdefault("artifact_path", str(artifact_path))
    metadata_payload.setdefault("artifact_schema", payload.get("schema"))
    return [_partial_from_payload(row) for row in partial_rows], metadata_payload
