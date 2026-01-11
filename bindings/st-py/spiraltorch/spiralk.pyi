from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from spiraltorch import RankPlan
from spiraltorch import (
    SpiralKAiRewriteConfig,
    SpiralKAiRewritePrompt,
    SpiralKContext,
    SpiralKHeuristicHint,
    SpiralKWilsonMetrics,
)


class SpiralKFftPlan:
    def __init__(self, radix: int, tile_cols: int, segments: int, subgroup: bool) -> None: ...

    @staticmethod
    def from_rank_plan(plan: RankPlan) -> "SpiralKFftPlan": ...

    def radix(self) -> int: ...

    def tile_cols(self) -> int: ...

    def segments(self) -> int: ...

    def subgroup(self) -> bool: ...

    def workgroup_size(self) -> int: ...

    def emit_wgsl(self) -> str: ...

    def emit_spiralk_hint(self) -> str: ...


FftPlan = SpiralKFftPlan


class MaxwellSpiralKHint:
    @property
    def channel(self) -> str: ...

    @property
    def blocks(self) -> int: ...

    @property
    def z_score(self) -> float: ...

    @property
    def z_bias(self) -> float: ...

    @property
    def weight(self) -> float: ...

    def script_line(self) -> str: ...


MaxwellHint = MaxwellSpiralKHint


class MaxwellSpiralKBridge:
    def __init__(
        self,
        base_program: Optional[str] = None,
        min_weight: float = 0.55,
        max_weight: float = 0.95,
    ) -> None: ...

    def set_base_program(self, program: Optional[str]) -> None: ...

    def set_weight_bounds(self, min_weight: float, max_weight: float) -> None: ...

    def push_pulse(
        self,
        channel: str,
        blocks: int,
        mean: float,
        standard_error: float,
        z_score: float,
        band_energy: Tuple[float, float, float],
        z_bias: float,
    ) -> MaxwellSpiralKHint: ...

    def hints(self) -> List[MaxwellSpiralKHint]: ...

    def is_empty(self) -> bool: ...

    def script(self) -> Optional[str]: ...


MaxwellBridge = MaxwellSpiralKBridge


def required_blocks(
    target_z: float,
    sigma: float,
    kappa: float,
    lambda_: float,
) -> Optional[float]: ...


def wilson_lower_bound(wins: int, trials: int, z: float) -> float: ...


def should_rewrite(
    metrics: SpiralKWilsonMetrics,
    min_gain: float = ...,
    min_confidence: float = ...,
) -> bool: ...


def synthesize_program(
    base_src: str,
    hints: Sequence[SpiralKHeuristicHint],
) -> str: ...


def rewrite_with_wilson(
    base_src: str,
    ctx: SpiralKContext,
    metrics: SpiralKWilsonMetrics,
    hints: Sequence[SpiralKHeuristicHint],
    min_gain: float = ...,
    min_confidence: float = ...,
) -> tuple[Dict[str, object], str]: ...


def rewrite_with_ai(
    base_src: str,
    ctx: SpiralKContext,
    config: SpiralKAiRewriteConfig,
    prompt: SpiralKAiRewritePrompt,
    generator: Callable[
        [SpiralKAiRewriteConfig, SpiralKAiRewritePrompt],
        Sequence[SpiralKHeuristicHint],
    ]
    | None = ...,
) -> tuple[Dict[str, object], str, Sequence[SpiralKHeuristicHint]]: ...


class MaxwellFingerprint:
    def __init__(
        self,
        gamma: float,
        modulation_depth: float,
        tissue_response: float,
        shielding_db: float,
        transmit_gain: float,
        polarization_angle: float,
        distance_m: float,
    ) -> None: ...

    @property
    def gamma(self) -> float: ...

    @property
    def modulation_depth(self) -> float: ...

    @property
    def tissue_response(self) -> float: ...

    @property
    def shielding_db(self) -> float: ...

    @property
    def transmit_gain(self) -> float: ...

    @property
    def polarization_angle(self) -> float: ...

    @property
    def distance_m(self) -> float: ...

    def shielding_factor(self) -> float: ...

    def polarization_alignment(self) -> float: ...

    def lambda_(self) -> float: ...

    def expected_block_mean(self, kappa: float) -> float: ...


class MeaningGate:
    def __init__(self, physical_gain: float, semantic_gain: float) -> None: ...

    @property
    def physical_gain(self) -> float: ...

    @physical_gain.setter
    def physical_gain(self, value: float) -> None: ...

    @property
    def semantic_gain(self) -> float: ...

    @semantic_gain.setter
    def semantic_gain(self, value: float) -> None: ...

    def envelope(self, rho: float) -> float: ...


class SequentialZ:
    def __init__(self) -> None: ...

    def push(self, sample: float) -> Optional[float]: ...

    def extend(self, samples: Iterable[float]) -> Optional[float]: ...

    def len(self) -> int: ...

    def is_empty(self) -> bool: ...

    def mean(self) -> float: ...

    def variance(self) -> Optional[float]: ...

    def standard_error(self) -> Optional[float]: ...

    def z_stat(self) -> Optional[float]: ...

    def reset(self) -> None: ...


class MaxwellPulse:
    def __init__(
        self,
        blocks: int,
        mean: float,
        standard_error: float,
        z_score: float,
        band_energy: Tuple[float, float, float],
        z_bias: float,
    ) -> None: ...

    @property
    def blocks(self) -> int: ...

    @property
    def mean(self) -> float: ...

    @property
    def standard_error(self) -> float: ...

    @property
    def z_score(self) -> float: ...

    @property
    def band_energy(self) -> Tuple[float, float, float]: ...

    @property
    def z_bias(self) -> float: ...

    def magnitude(self) -> float: ...


class MaxwellProjector:
    def __init__(
        self,
        rank: int,
        weight: float,
        bias_gain: float = ...,
        min_blocks: int = ...,
        min_z: float = ...,
    ) -> None: ...

    def project(self, tracker: SequentialZ) -> Optional[MaxwellPulse]: ...

    def last_pulse(self) -> Optional[MaxwellPulse]: ...

    def bias_gain(self) -> float: ...

    def set_bias_gain(self, bias_gain: float) -> None: ...

    def min_blocks(self) -> int: ...

    def set_min_blocks(self, min_blocks: int) -> None: ...

    def min_z(self) -> float: ...

    def set_min_z(self, min_z: float) -> None: ...

    def rank(self) -> int: ...

    def weight(self) -> float: ...


__all__ = [
    "FftPlan",
    "MaxwellBridge",
    "MaxwellHint",
    "MaxwellFingerprint",
    "MeaningGate",
    "SequentialZ",
    "MaxwellPulse",
    "MaxwellProjector",
    "required_blocks",
]
