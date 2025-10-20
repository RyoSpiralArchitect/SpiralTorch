# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
# Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

"""Drift-Response Linguistics (DRL) helpers for Z-space language training.

The module implements the quantities described in the Drift-Response
Linguistics (DRL) note.  It lets callers estimate a word's existential load,
safe radius, and hazard signature, then converts the summary into a scalar
penalty suitable for feeding into the lightweight :class:`ZSpaceTrainer`.

Example
-------
>>> from drift_response_linguistics import (
...     DEFAULT_THRESHOLDS,
...     FrameState,
...     WordState,
...     analyse_word,
...     trainer_penalty,
... )
>>> word = WordState(
...     name="AI",
...     definition_entropy=0.72,
...     timing_signal=1.4,
...     frames={
...         "Normative": FrameState(
...             phi=0.65,
...             c=0.9,
...             S=0.8,
...             a_den=-0.05,
...             a_con=0.2,
...             b_den=0.4,
...             b_con=0.8,
...             kappa=0.35,
...         )
...     },
... )
>>> metrics = analyse_word(word, DEFAULT_THRESHOLDS)
>>> penalty = trainer_penalty(metrics)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Dict, Iterable, Mapping, MutableMapping


@dataclass(frozen=True)
class FrameThreshold:
    """Policy thresholds for a frame.

    Parameters
    ----------
    tau:
        Minimum acceptable comprehension rate (``1 - tau`` is the maximum
        tolerated drop).
    rho:
        Maximum tolerated loss for the frame.
    hazard:
        Hazard cutoff used when counting the number of risky frames
        (``CHI`` in the note).
    """

    tau: float
    rho: float
    hazard: float = 1.0


@dataclass
class FrameState:
    """Observed or estimated state for a word-frame pair."""

    phi: float
    c: float
    S: float
    a_den: float
    a_con: float
    b_den: float
    b_con: float
    kappa: float
    timing_scale: float = 1.0
    curvature_a_den: float = 0.0
    curvature_a_con: float = 0.0
    curvature_b_den: float = 0.0
    curvature_b_con: float = 0.0
    kappa_slope: float = 0.0

    def mix_a(self) -> float:
        return (1.0 - self.phi) * self.a_den + self.phi * self.a_con

    def mix_b(self) -> float:
        return (1.0 - self.phi) * self.b_den + self.phi * self.b_con

    def mix_curvature_a(self) -> float:
        return (1.0 - self.phi) * self.curvature_a_den + self.phi * self.curvature_a_con

    def mix_curvature_b(self) -> float:
        return (1.0 - self.phi) * self.curvature_b_den + self.phi * self.curvature_b_con


@dataclass
class WordState:
    """Container for per-word DRL measurements."""

    name: str
    definition_entropy: float
    frames: Dict[str, FrameState]
    timing_signal: float = 0.0
    base_lambda: float = 1.0
    beta: float = 1.0


@dataclass(frozen=True)
class FrameSignature:
    """Local linear and quadratic response statistics for a frame."""

    value_slope: float
    risk_slope: float
    net_slope: float
    value_curvature: float
    risk_curvature: float
    net_curvature: float
    hazard_multiplier: float
    safe_radius: float | None
    kappa_slope: float


@dataclass
class DRLMetrics:
    """Summary of DRL statistics for a word."""

    word: WordState
    existence_load: float
    frame_hazards: Dict[str, float]
    safe_radii: Dict[str, float]
    chi: int
    strict_mode: bool
    frame_signatures: Dict[str, FrameSignature]


# Backwards compatibility for earlier drafts that used the "semantics" label.
DRSMetrics = DRLMetrics


DEFAULT_THRESHOLDS: Dict[str, FrameThreshold] = {
    "Physical": FrameThreshold(tau=0.05, rho=0.1, hazard=0.8),
    "Normative": FrameThreshold(tau=0.1, rho=0.08, hazard=0.9),
    "Social": FrameThreshold(tau=0.15, rho=0.06, hazard=0.7),
    "Protocol": FrameThreshold(tau=0.02, rho=0.05, hazard=0.95),
    "MetaLanguage": FrameThreshold(tau=0.08, rho=0.07, hazard=0.75),
    "Mythic": FrameThreshold(tau=0.2, rho=0.05, hazard=0.6),
}
"""Reasonable defaults that prioritise high-safety frames."""


def _hazard_multiplier(word: WordState, frame: FrameState) -> float:
    timing = max(0.0, word.timing_signal * frame.timing_scale)
    if timing == 0.0 or word.definition_entropy == 0.0 or frame.phi == 0.0:
        return 1.0
    exponent = word.beta * word.definition_entropy * frame.phi * timing
    # Clamp to avoid floating overflows when timing spikes.
    exponent = max(-30.0, min(30.0, exponent))
    return math.exp(exponent)


def frame_hazard(word: WordState, frame_name: str, frame: FrameState) -> float:
    a_mix = frame.mix_a()
    b_mix = frame.mix_b()
    base = max(0.0, -(a_mix - word.base_lambda * b_mix * frame.S))
    if base == 0.0:
        return 0.0
    multiplier = _hazard_multiplier(word, frame)
    return frame.c * multiplier * base


def existence_load(word: WordState) -> float:
    total = 0.0
    for frame in word.frames.values():
        a_mix = frame.mix_a()
        b_mix = frame.mix_b()
        base = max(0.0, -(a_mix - word.base_lambda * b_mix * frame.S))
        if base == 0.0:
            continue
        amplifier = 1.0 + word.beta * word.definition_entropy * frame.phi
        total += frame.c * base * amplifier
    return total


def safe_radius(
    word: WordState,
    thresholds: Mapping[str, FrameThreshold],
) -> Dict[str, float]:
    radii: Dict[str, float] = {}
    for name, frame in word.frames.items():
        threshold = thresholds.get(name)
        if threshold is None:
            continue
        kappa = max(frame.kappa, 1e-6)
        comprehension_limit = (1.0 - threshold.tau) / kappa
        b_mix = frame.mix_b()
        risk_denom = max(b_mix * frame.S, 1e-9)
        risk_limit = threshold.rho / risk_denom
        radii[name] = min(comprehension_limit, risk_limit)
    return radii


def analyse_word(
    word: WordState,
    thresholds: Mapping[str, FrameThreshold],
    *,
    hazard_cut: float | None = None,
    min_radius: float = 0.2,
) -> DRLMetrics:
    frame_hazards: Dict[str, float] = {}
    signatures: Dict[str, FrameSignature] = {}
    for name, frame in word.frames.items():
        hazard = frame_hazard(word, name, frame)
        frame_hazards[name] = hazard
        value_slope = frame.mix_a()
        risk_slope = word.base_lambda * frame.mix_b() * frame.S
        net_slope = value_slope - risk_slope
        value_curvature = frame.mix_curvature_a()
        risk_curvature = word.base_lambda * frame.mix_curvature_b() * frame.S
        net_curvature = value_curvature - risk_curvature
        signatures[name] = FrameSignature(
            value_slope=value_slope,
            risk_slope=risk_slope,
            net_slope=net_slope,
            value_curvature=value_curvature,
            risk_curvature=risk_curvature,
            net_curvature=net_curvature,
            hazard_multiplier=_hazard_multiplier(word, frame),
            safe_radius=None,
            kappa_slope=frame.kappa_slope,
        )

    radii = safe_radius(word, thresholds)
    for name, radius in radii.items():
        if name in signatures:
            signatures[name] = replace(signatures[name], safe_radius=radius)

    hazard_counts = 0
    for name, hz in frame_hazards.items():
        threshold = thresholds.get(name)
        if threshold is None:
            continue
        cut = hazard_cut if hazard_cut is not None else threshold.hazard
        if hz >= cut:
            hazard_counts += 1

    min_radius_observed = min(radii.values()) if radii else float("inf")
    existence = existence_load(word)
    strict = (
        hazard_counts >= 4
        or min_radius_observed <= min_radius
        or existence >= 1.0
    )

    return DRLMetrics(
        word=word,
        existence_load=existence,
        frame_hazards=frame_hazards,
        safe_radii=radii,
        chi=hazard_counts,
        strict_mode=strict,
        frame_signatures=signatures,
    )


def trainer_penalty(metrics: DRLMetrics, *, min_radius: float = 0.2) -> float:
    penalty = metrics.existence_load
    if metrics.safe_radii:
        min_radius_observed = min(metrics.safe_radii.values())
        if min_radius_observed < min_radius:
            penalty += (min_radius - min_radius_observed) / max(min_radius, 1e-6)
    penalty += float(metrics.chi)
    if metrics.strict_mode:
        penalty *= 1.25
    return penalty


def aggregate_penalty(
    metrics: Iterable[DRLMetrics],
    *,
    min_radius: float = 0.2,
) -> float:
    total = 0.0
    for item in metrics:
        total += trainer_penalty(item, min_radius=min_radius)
    return total


def frame_summary(metrics: DRLMetrics) -> MutableMapping[str, float]:
    summary: MutableMapping[str, float] = {}
    for name, hazard in metrics.frame_hazards.items():
        radius = metrics.safe_radii.get(name)
        summary[name] = hazard if radius is None else hazard / max(radius, 1e-6)
    return summary


__all__ = [
    "DRLMetrics",
    "DRSMetrics",
    "DEFAULT_THRESHOLDS",
    "FrameState",
    "FrameSignature",
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
