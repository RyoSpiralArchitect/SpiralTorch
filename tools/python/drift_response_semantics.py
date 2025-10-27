# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
# Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

"""Compatibility shim linking the Drift-Response Semantics helpers to DRL.

The original DRS prototype shipped a lighter analysis pass; the updated
``drift_response_linguistics`` module has since grown richer diagnostics
including curvature-aware signatures and directional probes.  Rather than keep
two implementations drifting apart, this module now **routes every public call**
to the linguistics engine while preserving the historical naming.

Importers that depended on ``drift_response_semantics`` continue to work, but
they automatically gain access to the extended metrics (directional signatures,
tipping radii, etc.) without code changes.  The adapter keeps the original
function signatures and re-exports the shared data classes so downstream code
can gradually adopt the unified vocabulary.
"""

from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping

from drift_response_linguistics import (
    DEFAULT_THRESHOLDS,
    DirectionQuery,
    DirectionalAxis,
    DirectionalSignature,
    DRLMetrics,
    FrameSignature,
    FrameState,
    FrameThreshold,
    WordState,
    aggregate_penalty as _aggregate_penalty,
    analyse_word as _analyse_word,
    existence_load as _existence_load,
    frame_hazard as _frame_hazard,
    frame_summary as _frame_summary,
    safe_radius as _safe_radius,
    trainer_penalty as _trainer_penalty,
)

# Historical alias kept for callers that imported ``DRSMetrics`` directly.
DRSMetrics = DRLMetrics


def existence_load(word: WordState) -> float:
    """Return the existential load using the shared DRL logic."""

    return _existence_load(word)


def frame_hazard(word: WordState, frame_name: str, frame: FrameState) -> float:
    """Compatibility wrapper that delegates to the DRL hazard computation."""

    return _frame_hazard(word, frame_name, frame)


def safe_radius(
    word: WordState,
    thresholds: Mapping[str, FrameThreshold],
) -> MutableMapping[str, float]:
    """Compute safe radii per frame via the unified DRL implementation."""

    return _safe_radius(word, thresholds)


def analyse_word(
    word: WordState,
    thresholds: Mapping[str, FrameThreshold],
    *,
    hazard_cut: float | None = None,
    min_radius: float = 0.2,
    direction_queries: Mapping[str, Iterable[DirectionQuery]] | None = None,
) -> DRSMetrics:
    """Run the full DRL analysis but keep the legacy return type alias."""

    return _analyse_word(
        word,
        thresholds,
        hazard_cut=hazard_cut,
        min_radius=min_radius,
        direction_queries=direction_queries,
    )


def trainer_penalty(metrics: DRSMetrics, *, min_radius: float = 0.2) -> float:
    """Delegate to the DRL penalty aggregator so tuning stays consistent."""

    return _trainer_penalty(metrics, min_radius=min_radius)


def aggregate_penalty(
    metrics: Iterable[DRSMetrics],
    *,
    min_radius: float = 0.2,
) -> float:
    """Aggregate penalties using the shared DRL helper."""

    return _aggregate_penalty(metrics, min_radius=min_radius)


def frame_summary(metrics: DRSMetrics) -> MutableMapping[str, float]:
    """Expose the frame summary produced by the linguistics helper."""

    return _frame_summary(metrics)


__all__ = [
    "DRSMetrics",
    "DEFAULT_THRESHOLDS",
    "DirectionalAxis",
    "DirectionalSignature",
    "DirectionQuery",
    "FrameSignature",
    "FrameState",
    "FrameThreshold",
    "WordState",
    "aggregate_penalty",
    "analyse_word",
    "existence_load",
    "frame_hazard",
    "frame_summary",
    "safe_radius",
    "trainer_penalty",
]

