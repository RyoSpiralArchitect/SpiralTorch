# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
# Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

"""Command line interface for the pure SpiralTorch bridge primitives."""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from pure_bridge import (  # type: ignore[no-redef]
        LibraryLoadError,
        OpenCartesianTopos,
        PurePythonBridge,
        resolve_library_path,
    )
    from pure_bridge_io import (  # type: ignore[no-redef]
        FloatPair,
        load_pairs_from_path,
        load_pairs_from_sources,
        load_pairs_from_text,
        load_weights_from_path,
        load_weights_from_text,
        load_texts_from_path,
        load_texts_from_sources,
        load_texts_from_text,
        parse_float_sequence,
        reshape,
        select_entries,
        summarize,
    )
else:
    from .pure_bridge import (
        LibraryLoadError,
        OpenCartesianTopos,
        PurePythonBridge,
        resolve_library_path,
    )
    from .pure_bridge_io import (
        FloatPair,
        load_pairs_from_path,
        load_pairs_from_sources,
        load_pairs_from_text,
        load_weights_from_path,
        load_weights_from_text,
        load_texts_from_path,
        load_texts_from_sources,
        load_texts_from_text,
        parse_float_sequence,
        reshape,
        select_entries,
        summarize,
    )


def _float_sequence_argument(value: str) -> List[float]:
    try:
        return parse_float_sequence(value)
    except ValueError as exc:  # pragma: no cover - argparse maps to CLI error
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _pair_argument(value: str) -> FloatPair:
    try:
        prediction_raw, target_raw = value.split("|", 1)
    except ValueError as exc:  # pragma: no cover - CLI validation
        raise argparse.ArgumentTypeError(f"Invalid pair '{value}': {exc}") from exc
    try:
        return parse_float_sequence(prediction_raw), parse_float_sequence(target_raw)
    except ValueError as exc:  # pragma: no cover - CLI validation
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _load_pairs_from_file(path: Path) -> List[FloatPair]:
    try:
        return load_pairs_from_path(path)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _load_pairs_from_text(text: str) -> List[FloatPair]:
    try:
        return load_pairs_from_text(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _load_weights_from_file(path: Path) -> List[float]:
    try:
        return load_weights_from_path(path)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _load_weights_from_text(text: str) -> List[float]:
    try:
        return load_weights_from_text(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _load_texts_from_file(path: Path) -> List[str]:
    try:
        return load_texts_from_path(path)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _load_texts_from_text(text: str) -> List[str]:
    try:
        return load_texts_from_text(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


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
        action="append",
        default=[],
        help="Load prediction/target pairs from file(s) (JSON array or 'pred|target' per line).",
    )
    hypergrad.add_argument(
        "--pairs-dir",
        type=Path,
        action="append",
        default=[],
        help="Load pairs from every file inside a directory (supports glob patterns).",
    )
    hypergrad.add_argument(
        "--pairs-pattern",
        action="append",
        metavar="GLOB",
        help="Glob pattern(s) to match when scanning --pairs-dir directories (default: *.json, *.txt).",
    )
    hypergrad.add_argument(
        "--pairs-recursive",
        action="store_true",
        help="Recurse into subdirectories when scanning --pairs-dir directories.",
    )
    hypergrad.add_argument(
        "--pairs-shuffle",
        action="store_true",
        help="Shuffle loaded pairs before accumulation (requires deterministic seed for reproducibility).",
    )
    hypergrad.add_argument(
        "--pairs-limit",
        type=int,
        default=None,
        help="Limit the number of pairs processed after optional shuffling.",
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
    hypergrad.add_argument(
        "--matrix",
        action="store_true",
        help="Reshape the gradient using the provided rows/cols before emission.",
    )
    hypergrad.add_argument(
        "--emit-weights",
        action="store_true",
        help="Include the final weight vector in the JSON payload when applying weights.",
    )
    hypergrad.add_argument(
        "--text",
        action="append",
        default=[],
        metavar="SAMPLE",
        help="Text sample to absorb via the encoder before sampling the gradient.",
    )
    hypergrad.add_argument(
        "--text-file",
        type=Path,
        action="append",
        default=[],
        help="Load text samples from file(s) (JSON array or one entry per line).",
    )
    hypergrad.add_argument(
        "--text-dir",
        type=Path,
        action="append",
        default=[],
        help="Load text samples from every file inside a directory (supports glob patterns).",
    )
    hypergrad.add_argument(
        "--text-pattern",
        action="append",
        metavar="GLOB",
        help="Glob pattern(s) when scanning --text-dir directories (default: *.txt, *.json).",
    )
    hypergrad.add_argument(
        "--text-recursive",
        action="store_true",
        help="Recurse into subdirectories when scanning --text-dir directories.",
    )
    hypergrad.add_argument(
        "--text-shuffle",
        action="store_true",
        help="Shuffle loaded text samples before absorption (requires deterministic seed for reproducibility).",
    )
    hypergrad.add_argument(
        "--text-limit",
        type=int,
        default=None,
        help="Limit the number of text samples absorbed after optional shuffling.",
    )
    hypergrad.add_argument(
        "--text-stdin",
        action="store_true",
        help="Read text samples from stdin (JSON array or one entry per line).",
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

    encoder_group = hypergrad.add_argument_group(
        "encoder",
        "LanguageWaveEncoder parameters for --text inputs",
    )
    encoder_group.add_argument(
        "--encoder-curvature",
        type=float,
        default=None,
        help="Curvature to use when constructing the encoder (defaults to hypergrad curvature).",
    )
    encoder_group.add_argument(
        "--encoder-temperature",
        type=float,
        default=0.5,
        help="Temperature to use when constructing the encoder (default: 0.5).",
    )

    hypergrad.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to use when shuffling pairs or text samples for deterministic ordering.",
    )

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
    text_samples: List[str] = []
    topos_values = None
    stdin_cache: Optional[str] = None
    if args.command == "hypergrad":
        pairs = list(args.pairs)
        for path in args.pairs_file:
            pairs.extend(_load_pairs_from_file(path))
        if args.pairs_dir:
            pair_patterns = args.pairs_pattern or ["*.json", "*.txt"]
            try:
                pairs.extend(
                    load_pairs_from_sources(
                        args.pairs_dir,
                        patterns=pair_patterns,
                        recursive=args.pairs_recursive,
                    )
                )
            except ValueError as exc:
                parser.error(str(exc))
        text_samples = list(args.text)
        for path in args.text_file:
            text_samples.extend(_load_texts_from_file(path))
        if args.text_dir:
            text_patterns = args.text_pattern or ["*.txt", "*.json"]
            try:
                text_samples.extend(
                    load_texts_from_sources(
                        args.text_dir,
                        patterns=text_patterns,
                        recursive=args.text_recursive,
                    )
                )
            except ValueError as exc:
                parser.error(str(exc))
        stdin_flags = [args.pairs_stdin, args.weights_stdin, args.text_stdin]
        if sum(1 for flag in stdin_flags if flag) > 1:
            parser.error(
                "At most one of --pairs-stdin, --weights-stdin, or --text-stdin may read from stdin"
            )
        if args.pairs_stdin:
            stdin_cache = sys.stdin.read()
            pairs.extend(_load_pairs_from_text(stdin_cache))
        if args.text_stdin:
            if stdin_cache is None:
                stdin_cache = sys.stdin.read()
            text_samples.extend(_load_texts_from_text(stdin_cache))
        if args.pairs_shuffle or args.pairs_limit is not None:
            try:
                pairs = select_entries(
                    pairs,
                    shuffle=args.pairs_shuffle,
                    limit=args.pairs_limit,
                    seed=args.seed,
                )
            except ValueError as exc:
                parser.error(str(exc))
        if args.text_shuffle or args.text_limit is not None:
            text_seed = None if args.seed is None else args.seed + 1
            try:
                text_samples = select_entries(
                    text_samples,
                    shuffle=args.text_shuffle,
                    limit=args.text_limit,
                    seed=text_seed,
                )
            except ValueError as exc:
                parser.error(str(exc))
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
            applied_weights: Optional[List[float]] = None
            with bridge.hypergrad(
                args.curvature,
                args.learning_rate,
                args.rows,
                args.cols,
                topos_ctx,
            ) as hypergrad:
                for prediction, target in pairs:
                    hypergrad.accumulate_pair(prediction, target)
                if text_samples:
                    encoder_curvature = (
                        args.encoder_curvature
                        if args.encoder_curvature is not None
                        else args.curvature
                    )
                    encoder_temperature = args.encoder_temperature
                    with bridge.encoder(encoder_curvature, encoder_temperature) as encoder:
                        for sample in text_samples:
                            hypergrad.absorb_text(encoder, sample)
                if weights is not None:
                    applied_weights = hypergrad.apply(weights)
                gradient = hypergrad.gradient()
            if args.summarize or args.matrix or args.emit_weights:
                payload: Dict[str, object] = {"gradient": gradient}
                if args.matrix:
                    try:
                        payload["matrix"] = reshape(gradient, args.rows, args.cols)
                    except ValueError as exc:
                        parser.error(str(exc))
                if args.summarize:
                    payload["summary"] = summarize(gradient)
                if args.emit_weights and applied_weights is not None:
                    payload["weights"] = applied_weights
                elif args.emit_weights:
                    payload["weights"] = None
                _emit_json(payload, args.indent, args.output)
            else:
                _emit_json(gradient, args.indent, args.output)
            return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
