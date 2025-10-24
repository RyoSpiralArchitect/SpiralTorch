# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
# Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

"""High-level helpers that expose the pure bridge functionality to Python callers.

The command line interface under :mod:`tools.python.pure_bridge_cli` exercises the
full bridge feature set, but scripting against it previously required either
shelling out or re-implementing the CLI wiring.  This module provides a Pythonic
surface over the same capabilities so automation, notebooks, and tests can drive
hypergradient computations or encoder invocations directly.
"""
from __future__ import annotations

import os
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from .pure_bridge import LibraryLoadError, OpenCartesianTopos, PurePythonBridge
from .pure_bridge_io import (
    FloatPair,
    load_pairs_from_path,
    load_pairs_from_sources,
    load_pairs_from_text,
    load_texts_from_path,
    load_texts_from_sources,
    load_texts_from_text,
    load_weights_from_path,
    load_weights_from_text,
    reshape,
    select_entries,
    summarize,
)


@dataclass(slots=True)
class SelectionOptions:
    """Options that control dataset sampling."""

    shuffle: bool = False
    limit: int | None = None
    offset: int = 0
    every: int | None = None

    def applies(self) -> bool:
        return self.shuffle or self.limit is not None or self.offset or self.every is not None


Pathish = os.PathLike[str] | str


@dataclass(slots=True)
class PairSources:
    """Collection of sources that provide prediction/target pairs."""

    direct: Sequence[FloatPair] = ()
    files: Sequence[Pathish] = ()
    directories: Sequence[Pathish] = ()
    patterns: Sequence[str] | None = None
    recursive: bool = False
    texts: Sequence[str] = ()
    selection: SelectionOptions = field(default_factory=SelectionOptions)


@dataclass(slots=True)
class TextSources:
    """Collection of sources that provide text samples."""

    direct: Sequence[str] = ()
    files: Sequence[Pathish] = ()
    directories: Sequence[Pathish] = ()
    patterns: Sequence[str] | None = None
    recursive: bool = False
    texts: Sequence[str] = ()
    selection: SelectionOptions = field(default_factory=SelectionOptions)


@dataclass(slots=True)
class WeightSource:
    """Defines how initial weights should be populated."""

    values: Sequence[float] | None = None
    file: Pathish | None = None
    text: str | None = None


@dataclass(slots=True)
class ToposConfig:
    """Configuration values for :class:`OpenCartesianTopos`."""

    curvature: float
    tolerance: float
    saturation: float
    max_depth: int
    max_volume: int

    def instantiate(self, bridge: PurePythonBridge) -> OpenCartesianTopos:
        return bridge.topos(
            self.curvature,
            self.tolerance,
            self.saturation,
            self.max_depth,
            self.max_volume,
        )


@dataclass(slots=True)
class EncoderConfig:
    """Parameters forwarded to :class:`~pure_bridge.LanguageWaveEncoder`."""

    curvature: float | None = None
    temperature: float = 0.5


@dataclass(slots=True)
class HypergradRequest:
    """Full request payload for :func:`run_hypergrad`."""

    rows: int
    cols: int
    curvature: float = -1.0
    learning_rate: float = 0.05
    pairs: PairSources = field(default_factory=PairSources)
    text: TextSources = field(default_factory=TextSources)
    weights: WeightSource = field(default_factory=WeightSource)
    seed: int | None = None
    summarize: bool = False
    matrix: bool = False
    emit_weights: bool = False
    topos: ToposConfig | None = None
    encoder: EncoderConfig | None = None


@dataclass(slots=True)
class HypergradResult:
    """Result object returned by :func:`run_hypergrad`."""

    gradient: List[float]
    matrix: List[List[float]] | None = None
    summary: dict[str, object] | None = None
    weights: List[float] | None = None

    def as_dict(self) -> dict[str, object]:
        """Return the result as a JSON serialisable dictionary."""

        payload: dict[str, object] = {"gradient": self.gradient}
        if self.matrix is not None:
            payload["matrix"] = self.matrix
        if self.summary is not None:
            payload["summary"] = self.summary
        if self.weights is not None:
            payload["weights"] = self.weights
        return payload


def encode_text(
    text: str,
    *,
    curvature: float = -1.0,
    temperature: float = 0.5,
    library: Optional[os.PathLike[str]] = None,
    bridge: PurePythonBridge | None = None,
) -> List[float]:
    """Encode ``text`` into z-space coordinates via the pure bridge."""

    if bridge is not None and library is not None:
        raise ValueError("Provide either 'bridge' or 'library', not both")

    with _bridge_context(bridge, library) as active:
        with active.encoder(curvature, temperature) as encoder:
            return encoder.encode_z_space(text)


def run_hypergrad(
    request: HypergradRequest,
    *,
    library: Optional[os.PathLike[str]] = None,
    bridge: PurePythonBridge | None = None,
    text_seed_offset: int = 1,
) -> HypergradResult:
    """Execute a hypergradient accumulation using the provided ``request``."""

    if bridge is not None and library is not None:
        raise ValueError("Provide either 'bridge' or 'library', not both")

    pairs = _gather_pairs(request.pairs)
    if request.pairs.selection.applies():
        pairs = select_entries(
            pairs,
            shuffle=request.pairs.selection.shuffle,
            limit=request.pairs.selection.limit,
            seed=request.seed,
            offset=request.pairs.selection.offset,
            every=request.pairs.selection.every,
        )

    text_samples = _gather_text(request.text)
    if request.text.selection.applies():
        text_seed = None if request.seed is None else request.seed + text_seed_offset
        text_samples = select_entries(
            text_samples,
            shuffle=request.text.selection.shuffle,
            limit=request.text.selection.limit,
            seed=text_seed,
            offset=request.text.selection.offset,
            every=request.text.selection.every,
        )

    weights = _resolve_weights(request.weights)

    applied_weights: List[float] | None = None
    gradient: List[float] = []

    with _bridge_context(bridge, library) as active_bridge:
        with _maybe_topos(active_bridge, request.topos) as topos:
            with active_bridge.hypergrad(
                request.curvature,
                request.learning_rate,
                request.rows,
                request.cols,
                topos,
            ) as hypergrad:
                for prediction, target in pairs:
                    hypergrad.accumulate_pair(prediction, target)

                if text_samples:
                    encoder_cfg = request.encoder or EncoderConfig()
                    encoder_curvature = (
                        encoder_cfg.curvature
                        if encoder_cfg.curvature is not None
                        else request.curvature
                    )
                    with active_bridge.encoder(
                        encoder_curvature, encoder_cfg.temperature
                    ) as encoder:
                        for sample in text_samples:
                            hypergrad.absorb_text(encoder, sample)

                if weights is not None:
                    applied_weights = hypergrad.apply(weights)

                gradient = hypergrad.gradient()

    result = HypergradResult(gradient=gradient)

    if request.matrix:
        result.matrix = reshape(gradient, request.rows, request.cols)
    if request.summarize:
        result.summary = summarize(gradient)
    if request.emit_weights:
        result.weights = applied_weights

    return result


def _gather_pairs(config: PairSources) -> List[FloatPair]:
    collected: List[FloatPair] = []

    for prediction, target in config.direct:
        collected.append(
            (
                [float(value) for value in prediction],
                [float(value) for value in target],
            )
        )

    for text in config.texts:
        collected.extend(load_pairs_from_text(text))
    for path in config.files:
        collected.extend(load_pairs_from_path(Path(path)))
    if config.directories:
        patterns = config.patterns or ["*.json", "*.txt"]
        collected.extend(
            load_pairs_from_sources(
                [Path(directory) for directory in config.directories],
                patterns=patterns,
                recursive=config.recursive,
            )
        )

    return collected


def _gather_text(config: TextSources) -> List[str]:
    samples: List[str] = list(config.direct)

    for text in config.texts:
        samples.extend(load_texts_from_text(text))
    for path in config.files:
        samples.extend(load_texts_from_path(Path(path)))
    if config.directories:
        patterns = config.patterns or ["*.txt", "*.json"]
        samples.extend(
            load_texts_from_sources(
                [Path(directory) for directory in config.directories],
                patterns=patterns,
                recursive=config.recursive,
            )
        )

    return samples


def _resolve_weights(config: WeightSource) -> List[float] | None:
    options = [config.values is not None, config.file is not None, config.text is not None]
    if sum(1 for option in options if option) > 1:
        raise ValueError("Provide at most one weight source")

    if config.values is not None:
        return [float(value) for value in config.values]
    if config.file is not None:
        return load_weights_from_path(Path(config.file))
    if config.text is not None:
        return load_weights_from_text(config.text)
    return None


class _ToposContext:
    def __init__(self, bridge: PurePythonBridge, config: ToposConfig | None) -> None:
        self._bridge = bridge
        self._config = config
        self._resource: OpenCartesianTopos | None = None

    def __enter__(self) -> OpenCartesianTopos | None:
        if self._config is None:
            return None
        self._resource = self._config.instantiate(self._bridge)
        return self._resource

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._resource is not None:
            self._resource.close()
            self._resource = None


def _maybe_topos(bridge: PurePythonBridge, config: ToposConfig | None) -> _ToposContext:
    return _ToposContext(bridge, config)


class _BridgeContext:
    def __init__(
        self,
        bridge: PurePythonBridge | None,
        library: Optional[os.PathLike[str]],
    ) -> None:
        self._external = bridge
        self._library = library
        self._owned: PurePythonBridge | None = None

    def __enter__(self) -> PurePythonBridge:
        if self._external is not None:
            return self._external

        owned = PurePythonBridge(self._library)
        owned.__enter__()
        self._owned = owned
        return owned

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._owned is not None:
            self._owned.__exit__(exc_type, exc, tb)
            self._owned = None


def _bridge_context(
    bridge: PurePythonBridge | None, library: Optional[os.PathLike[str]]
) -> _BridgeContext:
    return _BridgeContext(bridge, library)


class PureBridgeSession:
    """Context manager that keeps a bridge loaded for multiple operations."""

    def __init__(self, library: Optional[os.PathLike[str]] = None) -> None:
        self._library = library
        self._stack: ExitStack | None = None
        self._bridge: PurePythonBridge | None = None

    def __enter__(self) -> "PureBridgeSession":
        stack = ExitStack()
        self._stack = stack
        self._bridge = stack.enter_context(PurePythonBridge(self._library))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None
        self._bridge = None

    @property
    def bridge(self) -> PurePythonBridge:
        if self._bridge is None:
            raise RuntimeError("Session has not been entered")
        return self._bridge

    def encode_text(
        self,
        text: str,
        *,
        curvature: float = -1.0,
        temperature: float = 0.5,
    ) -> List[float]:
        return encode_text(
            text,
            curvature=curvature,
            temperature=temperature,
            bridge=self.bridge,
        )

    def run_hypergrad(
        self,
        request: HypergradRequest,
        *,
        text_seed_offset: int = 1,
    ) -> HypergradResult:
        return run_hypergrad(
            request,
            bridge=self.bridge,
            text_seed_offset=text_seed_offset,
        )

    def open_topos(self, config: ToposConfig) -> OpenCartesianTopos:
        if self._stack is None:
            raise RuntimeError("Session has not been entered")
        topos = config.instantiate(self.bridge)
        return self._stack.enter_context(topos)


__all__ = [
    "EncoderConfig",
    "HypergradRequest",
    "HypergradResult",
    "PureBridgeSession",
    "PairSources",
    "SelectionOptions",
    "TextSources",
    "ToposConfig",
    "WeightSource",
    "encode_text",
    "run_hypergrad",
    "LibraryLoadError",
]
