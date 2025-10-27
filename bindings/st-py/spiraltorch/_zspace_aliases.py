"""Shared alias tables for Z-space metrics.

These mappings are consumed by both the public Python API shim and the
inference helpers so that the accepted aliases stay in lockstep. Historically
each module maintained its own copy and they inevitably drifted as new metrics
were added in one place but not the other. Centralising the definitions keeps
the mapping coherent and makes it easier to extend Z-space diagnostics without
chasing down duplicate tables.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

__all__ = [
    "ZSPACE_METRIC_ALIASES",
    "PRIMARY_ZSPACE_METRIC_ALIASES",
]


_ALIAS_ENTRIES = {
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
    "grad": "gradient",
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


PRIMARY_ZSPACE_METRICS = frozenset({"speed", "memory", "stability", "drs", "gradient"})


ZSPACE_METRIC_ALIASES: Mapping[str, str] = MappingProxyType(dict(_ALIAS_ENTRIES))


PRIMARY_ZSPACE_METRIC_ALIASES: Mapping[str, str] = MappingProxyType(
    {alias: target for alias, target in _ALIAS_ENTRIES.items() if target in PRIMARY_ZSPACE_METRICS}
)

