# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
# Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

"""Command line interface for the pure SpiralTorch bridge primitives."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable as IterableABC, Mapping, Sequence as SequenceABC
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from pure_bridge import (  # type: ignore[no-redef]
        LibraryLoadError,
        OpenCartesianTopos,
        PurePythonBridge,
        resolve_library_path,
    )
else:
    from .pure_bridge import (
        LibraryLoadError,
        OpenCartesianTopos,
        PurePythonBridge,
        resolve_library_path,
    )

FloatPair = Tuple[List[float], List[float]]


def _float_sequence_from_obj(obj: Any) -> List[float]:
    if isinstance(obj, str):
        return _parse_float_sequence(obj)
    if isinstance(obj, Mapping):
        raise ValueError("Mapping inputs must expose 'prediction'/'target' keys, not nested maps")
    if isinstance(obj, SequenceABC):
        return [float(item) for item in obj]
    if isinstance(obj, IterableABC):
        return [float(item) for item in obj]
    raise ValueError("Expected a list of numbers or a comma/JSON encoded string")


def _parse_float_sequence(raw: str) -> List[float]:
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
    floats: List[float] = []
    for item in items:
        floats.append(float(item))
    return floats


def _float_sequence_argument(value: str) -> List[float]:
    try:
        return _parse_float_sequence(value)
    except ValueError as exc:  # pragma: no cover - argparse maps to CLI error
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _pair_argument(value: str) -> FloatPair:
    try:
        prediction_raw, target_raw = value.split("|", 1)
    except ValueError as exc:  # pragma: no cover - CLI validation
        raise argparse.ArgumentTypeError(f"Invalid pair '{value}': {exc}") from exc
    try:
        return _parse_float_sequence(prediction_raw), _parse_float_sequence(target_raw)
    except ValueError as exc:  # pragma: no cover - CLI validation
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _load_pairs_from_file(path: Path) -> List[FloatPair]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise argparse.ArgumentTypeError(f"Failed to read pairs file '{path}': {exc}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return [_pair_argument(line) for line in lines]

    if not isinstance(data, list):
        raise argparse.ArgumentTypeError("Pairs file must contain a JSON array")

    parsed: List[FloatPair] = []
    for idx, item in enumerate(data):
        if isinstance(item, dict):
            if "prediction" in item and "target" in item:
                pred = _float_sequence_from_obj(item["prediction"])
                tgt = _float_sequence_from_obj(item["target"])
            elif "pred" in item and "tgt" in item:
                pred = _float_sequence_from_obj(item["pred"])
                tgt = _float_sequence_from_obj(item["tgt"])
            else:
                raise argparse.ArgumentTypeError(
                    f"Entry {idx} missing 'prediction'/'target' keys"
                )
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            pred = _float_sequence_from_obj(item[0])
            tgt = _float_sequence_from_obj(item[1])
        else:
            raise argparse.ArgumentTypeError(
                f"Entry {idx} must be [prediction, target] or have prediction/target keys"
            )
        parsed.append((pred, tgt))
    return parsed


def _load_pairs_from_text(text: str) -> List[FloatPair]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [_pair_argument(line) for line in lines]

    if not isinstance(data, list):
        raise argparse.ArgumentTypeError("Pairs input must be a JSON array when read from stdin")

    parsed: List[FloatPair] = []
    for idx, item in enumerate(data):
        if isinstance(item, dict):
            if "prediction" in item and "target" in item:
                pred = _float_sequence_from_obj(item["prediction"])
                tgt = _float_sequence_from_obj(item["target"])
            elif "pred" in item and "tgt" in item:
                pred = _float_sequence_from_obj(item["pred"])
                tgt = _float_sequence_from_obj(item["tgt"])
            else:
                raise argparse.ArgumentTypeError(
                    f"Entry {idx} missing 'prediction'/'target' keys"
                )
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            pred = _float_sequence_from_obj(item[0])
            tgt = _float_sequence_from_obj(item[1])
        else:
            raise argparse.ArgumentTypeError(
                f"Entry {idx} must contain prediction and target sequences"
            )
        parsed.append((pred, tgt))
    return parsed


def _load_weights_from_file(path: Path) -> List[float]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise argparse.ArgumentTypeError(f"Failed to read weights file '{path}': {exc}") from exc
    return _float_sequence_argument(text)


def _load_weights_from_text(text: str) -> List[float]:
    return _float_sequence_argument(text)


def _summarize(values: Sequence[float]) -> Dict[str, Union[float, int, None]]:
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
    l2 = math.sqrt(sum(value * value for value in values))
    return {
        "count": count,
        "min": min(values),
        "max": max(values),
        "mean": total / float(count),
        "l1": l1,
        "l2": l2,
    }


def _emit_json(data: Any, indent: Optional[int], output: Optional[Path]) -> None:
    text = json.dumps(data, indent=indent)
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interact with the pure SpiralTorch bridge library",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=None,
        help="Path to the compiled SpiralTorch cdylib.",
    )
    parser.add_argument(
        "--show-library",
        action="store_true",
        help="Print the resolved cdylib path and exit.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="Indentation to use for JSON output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file to write JSON output to.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    encode = subparsers.add_parser(
        "encode",
        help="Encode UTF-8 text into z-space coordinates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    encode.add_argument("text", help="Text to encode")
    encode.add_argument("--curvature", type=float, default=-1.0)
    encode.add_argument("--temperature", type=float, default=0.5)

    hypergrad = subparsers.add_parser(
        "hypergrad",
        help="Accumulate prediction/target pairs and inspect the resulting gradient",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    hypergrad.add_argument("--curvature", type=float, default=-1.0)
    hypergrad.add_argument("--learning-rate", type=float, default=0.05)
    hypergrad.add_argument("--rows", type=int, required=True)
    hypergrad.add_argument("--cols", type=int, required=True)
    hypergrad.add_argument(
        "--pairs",
        action="append",
        type=_pair_argument,
        default=[],
        metavar="PRED|TARGET",
        help="Prediction and target sequences separated by '|' (comma or JSON encoded).",
    )
    hypergrad.add_argument(
        "--pairs-file",
        type=Path,
        help="Load prediction/target pairs from a file (JSON array or 'pred|target' per line).",
    )
    hypergrad.add_argument(
        "--pairs-stdin",
        action="store_true",
        help="Read prediction/target pairs from stdin (JSON array or 'pred|target' per line).",
    )
    hypergrad.add_argument(
        "--weights",
        type=_float_sequence_argument,
        default=None,
        help="Initial weights sequence to apply before sampling the gradient.",
    )
    hypergrad.add_argument(
        "--weights-file",
        type=Path,
        help="Load initial weights from a file (comma or JSON encoded sequence).",
    )
    hypergrad.add_argument(
        "--weights-stdin",
        action="store_true",
        help="Read initial weights from stdin (comma or JSON encoded sequence).",
    )
    hypergrad.add_argument(
        "--summarize",
        action="store_true",
        help="Include summary statistics alongside the gradient output.",
    )
    topos_group = hypergrad.add_argument_group(
        "topos",
        "Optional OpenCartesianTopos parameters to couple the hypergrad against",
    )
    topos_group.add_argument("--topos-curvature", type=float)
    topos_group.add_argument("--topos-tolerance", type=float)
    topos_group.add_argument("--topos-saturation", type=float)
    topos_group.add_argument("--topos-max-depth", type=int)
    topos_group.add_argument("--topos-max-volume", type=int)

    return parser


def _validate_topos_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Optional[Tuple[float, float, float, int, int]]:
    values = (
        args.topos_curvature,
        args.topos_tolerance,
        args.topos_saturation,
        args.topos_max_depth,
        args.topos_max_volume,
    )
    provided = [value is not None for value in values]
    if any(provided) and not all(provided):
        parser.error("All --topos-* options must be provided together")
    if all(provided):
        curvature, tolerance, saturation, max_depth, max_volume = values  # type: ignore[assignment]
        assert curvature is not None and tolerance is not None and saturation is not None
        assert max_depth is not None and max_volume is not None
        return (
            float(curvature),
            float(tolerance),
            float(saturation),
            int(max_depth),
            int(max_volume),
        )
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.show_library:
        try:
            path = resolve_library_path(args.library)
        except LibraryLoadError as exc:
            parser.error(str(exc))
        print(path)
        return 0

    pairs: List[FloatPair] = []
    topos_values = None
    stdin_cache: Optional[str] = None
    if args.command == "hypergrad":
        pairs = list(args.pairs)
        if args.pairs_file is not None:
            pairs.extend(_load_pairs_from_file(args.pairs_file))
        if args.pairs_stdin:
            if args.weights_stdin:
                parser.error("Cannot combine --pairs-stdin with --weights-stdin")
            stdin_cache = sys.stdin.read()
            pairs.extend(_load_pairs_from_text(stdin_cache))
        topos_values = _validate_topos_arguments(args, parser)

    with ExitStack() as stack:
        try:
            bridge = PurePythonBridge(args.library)
        except LibraryLoadError as exc:
            parser.error(str(exc))
        stack.enter_context(bridge)

        if args.command == "encode":
            with bridge.encoder(args.curvature, args.temperature) as encoder:
                encoded = encoder.encode_z_space(args.text)
            _emit_json(encoded, args.indent, args.output)
            return 0

        if args.command == "hypergrad":
            topos_ctx: Optional[OpenCartesianTopos] = None
            if topos_values is not None:
                topos_ctx = stack.enter_context(
                    bridge.topos(*topos_values)
                )
            weights = args.weights
            if args.weights_file is not None:
                if weights is not None:
                    parser.error("Provide only one of --weights or --weights-file")
                weights = _load_weights_from_file(args.weights_file)
            if args.weights_stdin:
                if weights is not None:
                    parser.error("Provide only one weights source")
                if stdin_cache is None:
                    stdin_cache = sys.stdin.read()
                weights = _load_weights_from_text(stdin_cache)
            with bridge.hypergrad(
                args.curvature,
                args.learning_rate,
                args.rows,
                args.cols,
                topos_ctx,
            ) as hypergrad:
                for prediction, target in pairs:
                    hypergrad.accumulate_pair(prediction, target)
                if weights is not None:
                    hypergrad.apply(weights)
                gradient = hypergrad.gradient()
            if args.summarize:
                payload = {
                    "gradient": gradient,
                    "summary": _summarize(gradient),
                }
            else:
                payload = gradient
            _emit_json(payload, args.indent, args.output)
            return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
