# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
# Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

"""Utility helpers shared across the pure bridge tooling suite."""

from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import Iterable, Iterator, List, Sequence, Tuple, TypeVar, Union

FloatPair = Tuple[List[float], List[float]]
T = TypeVar("T")


def parse_float_sequence(raw: Union[str, Sequence[float], Iterable[float]]) -> List[float]:
    """Parse a float sequence from JSON, comma separated text, or iterable values."""

    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            items = [segment.strip() for segment in text.split(",") if segment.strip()]
        else:
            if not isinstance(data, (list, tuple)):
                raise ValueError("Expected an array of numbers")
            items = data
    else:
        items = raw

    floats: List[float] = []
    for item in items:
        floats.append(float(item))
    return floats


def load_pairs_from_path(path: Path) -> List[FloatPair]:
    """Load prediction/target pairs from a JSON or pipe-separated text file."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Failed to read pairs file '{path}': {exc}") from exc
    return load_pairs_from_text(text)


def load_pairs_from_sources(
    sources: Sequence[Path],
    *,
    patterns: Sequence[str] | None = None,
    recursive: bool = False,
) -> List[FloatPair]:
    """Load prediction/target pairs from the provided files or directories."""

    files = _collect_data_files(sources, patterns=patterns, recursive=recursive)
    pairs: List[FloatPair] = []
    for path in files:
        pairs.extend(load_pairs_from_path(path))
    return pairs


def load_pairs_from_text(text: str) -> List[FloatPair]:
    """Load prediction/target pairs from JSON or line separated text."""

    stripped_lines = [line.strip() for line in text.splitlines() if line.strip()]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [_parse_pair_line(line) for line in stripped_lines]

    if not isinstance(data, list):
        raise ValueError("Pairs input must be a JSON array")

    parsed: List[FloatPair] = []
    for idx, item in enumerate(data):
        try:
            parsed.append(_pair_from_obj(item))
        except ValueError as exc:
            raise ValueError(f"Entry {idx}: {exc}") from exc
    return parsed


def load_weights_from_path(path: Path) -> List[float]:
    """Load a float sequence from disk."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Failed to read weights file '{path}': {exc}") from exc
    return parse_float_sequence(text)


def load_weights_from_text(text: str) -> List[float]:
    """Load a float sequence from stdin text."""

    return parse_float_sequence(text)


def load_texts_from_path(path: Path) -> List[str]:
    """Load newline or JSON encoded text samples from disk."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Failed to read text file '{path}': {exc}") from exc
    return load_texts_from_text(text)


def load_texts_from_sources(
    sources: Sequence[Path],
    *,
    patterns: Sequence[str] | None = None,
    recursive: bool = False,
) -> List[str]:
    """Load newline or JSON encoded samples from the provided files or directories."""

    files = _collect_data_files(sources, patterns=patterns, recursive=recursive)
    samples: List[str] = []
    for path in files:
        samples.extend(load_texts_from_path(path))
    return samples


def load_texts_from_text(text: str) -> List[str]:
    """Load text samples from stdin or CLI input."""

    stripped_lines = [line.strip() for line in text.splitlines() if line.strip()]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return stripped_lines

    if not isinstance(data, list):
        raise ValueError("Text input must be a JSON array")

    samples: List[str] = []
    for idx, item in enumerate(data):
        if not isinstance(item, str):
            raise ValueError(f"Entry {idx}: expected a string")
        samples.append(item)
    return samples


def summarize(values: Sequence[float]) -> dict:
    """Return basic statistics over a sequence of floats."""

    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "l1": 0.0,
            "l2": 0.0,
        }

    count = len(values)
    total = sum(values)
    l1 = sum(abs(value) for value in values)
    l2 = (sum(value * value for value in values)) ** 0.5
    return {
        "count": count,
        "min": min(values),
        "max": max(values),
        "mean": total / float(count),
        "l1": l1,
        "l2": l2,
    }


def reshape(values: Sequence[float], rows: int, cols: int) -> List[List[float]]:
    """Reshape a flat list of values into a matrix with ``rows`` x ``cols`` entries."""

    expected = rows * cols
    if len(values) != expected:
        raise ValueError(
            f"Cannot reshape gradient of length {len(values)} into matrix with {rows}x{cols} entries"
        )

    matrix: List[List[float]] = []
    for r in range(rows):
        offset = r * cols
        matrix.append([float(values[offset + c]) for c in range(cols)])
    return matrix


def select_entries(
    values: Sequence[T],
    *,
    limit: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
) -> List[T]:
    """Return a possibly shuffled and truncated list of ``values``."""

    items = list(values)
    if shuffle:
        Random(seed).shuffle(items)
    if limit is not None:
        if limit < 0:
            raise ValueError("Limit must be a non-negative integer")
        items = items[:limit]
    return items


def _collect_data_files(
    sources: Sequence[Path],
    *,
    patterns: Sequence[str] | None,
    recursive: bool,
) -> List[Path]:
    """Return a stable list of files collected from ``sources``.

    Each entry in ``sources`` may refer to a file or directory.  Directories are
    scanned using the provided glob ``patterns``.  Duplicate files are skipped
    while preserving the order they are first encountered.
    """

    expanded: List[Path] = []
    for source in sources:
        expanded.append(source.expanduser().resolve())

    patterns = [pattern for pattern in (patterns or ["*"]) if pattern]
    seen: set[Path] = set()
    collected: List[Path] = []

    for source in expanded:
        if source.is_file():
            if source not in seen:
                seen.add(source)
                collected.append(source)
            continue
        if source.is_dir():
            iterator: Iterator[Path]
            for pattern in patterns:
                iterator = source.rglob(pattern) if recursive else source.glob(pattern)
                for candidate in iterator:
                    candidate = candidate.resolve()
                    if candidate.is_file() and candidate not in seen:
                        seen.add(candidate)
                        collected.append(candidate)
            continue
        raise ValueError(f"Data source '{source}' does not exist or is not accessible")

    if not collected:
        roots = ", ".join(str(path) for path in expanded) or "<none>"
        pattern_text = ", ".join(patterns)
        raise ValueError(
            "No files matched the provided data sources."
            f" Sources: {roots}. Patterns: {pattern_text}."
        )

    return collected


def _parse_pair_line(value: str) -> FloatPair:
    try:
        pred_raw, tgt_raw = value.split("|", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid pair '{value}': {exc}") from exc
    return parse_float_sequence(pred_raw), parse_float_sequence(tgt_raw)


def _pair_from_obj(obj: object) -> FloatPair:
    if isinstance(obj, dict):
        if "prediction" in obj and "target" in obj:
            pred = parse_float_sequence(obj["prediction"])
            tgt = parse_float_sequence(obj["target"])
        elif "pred" in obj and "tgt" in obj:
            pred = parse_float_sequence(obj["pred"])
            tgt = parse_float_sequence(obj["tgt"])
        else:
            raise ValueError("missing prediction/target keys")
        return pred, tgt

    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        return parse_float_sequence(obj[0]), parse_float_sequence(obj[1])

    raise ValueError("must be [prediction, target] or expose prediction/target keys")


__all__ = [
    "FloatPair",
    "load_pairs_from_sources",
    "load_pairs_from_path",
    "load_pairs_from_text",
    "load_weights_from_path",
    "load_weights_from_text",
    "load_texts_from_sources",
    "load_texts_from_path",
    "load_texts_from_text",
    "parse_float_sequence",
    "reshape",
    "select_entries",
    "summarize",
]

