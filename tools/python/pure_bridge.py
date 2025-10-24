# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
# Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
# Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

"""Minimal ctypes helpers for driving the pure SpiralTorch stack from Python.

The previous revision of this module bundled a quick CLI entry point alongside
the bridge primitives. To keep the bridge importable without argparse baggage
we now focus the module purely on resource wrappers while exposing helpers that
other tools (notably :mod:`tools.python.pure_bridge_cli`) can reuse.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from ctypes.util import find_library
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


class LibraryLoadError(RuntimeError):
    """Raised when the SpiralTorch cdylib cannot be located."""


def _default_library_name() -> str:
    if sys.platform.startswith("linux"):
        return "libst_tensor.so"
    if sys.platform == "darwin":
        return "libst_tensor.dylib"
    if sys.platform in ("win32", "cygwin"):
        return "st_tensor.dll"
    raise RuntimeError(f"Unsupported platform: {sys.platform}")


def _candidate_directories() -> Iterable[Path]:
    """Yield directories that may contain the compiled pure bridge library."""

    root = Path(__file__).resolve().parents[2]
    target = root / "target"

    # Common cargo build output locations.
    yield target / "release"
    yield target / "debug"
    yield target / "release" / "deps"
    yield target / "debug" / "deps"

    # Maturin/maturin develop builds when working on the Python bindings.
    python_bindings = root / "bindings" / "python"
    yield python_bindings / "target" / "release"
    yield python_bindings / "target" / "debug"

    # `just build-python` targets emit wheels into maturin/targets.
    maturin_dir = root / "maturin"
    for profile in ("release", "debug"):
        yield maturin_dir / "target" / profile


def resolve_library_path(explicit: Optional[os.PathLike[str]] = None) -> Path:
    """Resolve the path to the pure bridge shared library.

    The resolution order favours caller intent and then falls back to common
    build directories.  Users may also provide ``SPIRALTORCH_PURE_LIB`` to
    override the discovery logic when embedding the module in standalone tools.
    """

    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser().resolve())

    env_override = os.environ.get("SPIRALTORCH_PURE_LIB")
    if env_override:
        candidates.append(Path(env_override).expanduser().resolve())

    name = _default_library_name()

    for directory in _candidate_directories():
        candidates.append((directory / name).resolve())

    located = find_library("st_tensor")
    if located:
        candidates.append(Path(located).resolve())

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n".join(str(path) for path in candidates)
    raise LibraryLoadError(
        "Unable to locate the SpiralTorch pure bridge cdylib.\n"
        "Searched:\n"
        f"{searched}\n"
        "Build the library via 'cargo build -p st-tensor --release' or set\n"
        "SPIRALTORCH_PURE_LIB to the compiled artifact."
    )


class _Lib:
    def __init__(self, path: Optional[os.PathLike[str]] = None) -> None:
        resolved = resolve_library_path(path)
        self._cdll = ctypes.CDLL(str(resolved))
        self._configure()

    def _configure(self) -> None:
        lib = self._cdll
        c_float = ctypes.c_float
        c_size = ctypes.c_size_t
        c_char_p = ctypes.c_char_p
        c_void_p = ctypes.c_void_p
        c_int = ctypes.c_int

        lib.st_pure_last_error.restype = c_char_p
        lib.st_pure_clear_last_error.argtypes = []
        lib.st_pure_clear_last_error.restype = None

        lib.st_pure_topos_new.argtypes = [c_float, c_float, c_float, c_size, c_size]
        lib.st_pure_topos_new.restype = c_void_p
        lib.st_pure_topos_free.argtypes = [c_void_p]
        lib.st_pure_topos_free.restype = None

        lib.st_pure_hypergrad_new.argtypes = [c_float, c_float, c_size, c_size]
        lib.st_pure_hypergrad_new.restype = c_void_p
        lib.st_pure_hypergrad_with_topos.argtypes = [c_float, c_float, c_size, c_size, c_void_p]
        lib.st_pure_hypergrad_with_topos.restype = c_void_p
        lib.st_pure_hypergrad_free.argtypes = [c_void_p]
        lib.st_pure_hypergrad_free.restype = None
        lib.st_pure_hypergrad_shape.argtypes = [c_void_p, ctypes.POINTER(c_size), ctypes.POINTER(c_size)]
        lib.st_pure_hypergrad_shape.restype = c_int
        lib.st_pure_hypergrad_accumulate_pair.argtypes = [
            c_void_p,
            ctypes.POINTER(c_float),
            c_size,
            ctypes.POINTER(c_float),
            c_size,
        ]
        lib.st_pure_hypergrad_accumulate_pair.restype = c_int
        lib.st_pure_hypergrad_apply.argtypes = [c_void_p, ctypes.POINTER(c_float), c_size]
        lib.st_pure_hypergrad_apply.restype = c_int
        lib.st_pure_hypergrad_gradient.argtypes = [c_void_p, ctypes.POINTER(c_float), c_size]
        lib.st_pure_hypergrad_gradient.restype = c_size
        lib.st_pure_hypergrad_absorb_text.argtypes = [c_void_p, c_void_p, c_char_p]
        lib.st_pure_hypergrad_absorb_text.restype = c_int

        lib.st_pure_encoder_new.argtypes = [c_float, c_float]
        lib.st_pure_encoder_new.restype = c_void_p
        lib.st_pure_encoder_free.argtypes = [c_void_p]
        lib.st_pure_encoder_free.restype = None
        lib.st_pure_encoder_encode_z_space.argtypes = [
            c_void_p,
            c_char_p,
            ctypes.POINTER(c_float),
            c_size,
        ]
        lib.st_pure_encoder_encode_z_space.restype = c_size

        self.lib = lib

    def last_error(self) -> Optional[str]:
        raw = self.lib.st_pure_last_error()
        if raw:
            return raw.decode("utf-8")
        return None

    def clear(self) -> None:
        self.lib.st_pure_clear_last_error()

    def check(self, status: int) -> None:
        if status != 0:
            message = self.last_error() or "unknown error"
            raise RuntimeError(message)


class OpenCartesianTopos:
    def __init__(
        self,
        bridge: _Lib,
        curvature: float,
        tolerance: float,
        saturation: float,
        max_depth: int,
        max_volume: int,
    ) -> None:
        self._bridge = bridge
        ptr = bridge.lib.st_pure_topos_new(
            curvature, tolerance, saturation, max_depth, max_volume
        )
        if not ptr:
            raise RuntimeError(bridge.last_error() or "failed to create OpenCartesianTopos")
        self._ptr = ptr

    @property
    def ptr(self) -> ctypes.c_void_p:
        return ctypes.c_void_p(self._ptr)

    def __enter__(self) -> "OpenCartesianTopos":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._ptr:
            self._bridge.lib.st_pure_topos_free(self._ptr)
            self._ptr = None

    def __del__(self) -> None:  # pragma: no cover - destructor safety
        try:
            self.close()
        except Exception:
            pass


class LanguageWaveEncoder:
    def __init__(self, bridge: _Lib, curvature: float, temperature: float) -> None:
        self._bridge = bridge
        ptr = bridge.lib.st_pure_encoder_new(curvature, temperature)
        if not ptr:
            raise RuntimeError(bridge.last_error() or "failed to create LanguageWaveEncoder")
        self._ptr = ptr

    @property
    def ptr(self) -> ctypes.c_void_p:
        return ctypes.c_void_p(self._ptr)

    def encode_z_space(self, text: str) -> List[float]:
        message = text.encode("utf-8")
        required = self._bridge.lib.st_pure_encoder_encode_z_space(self._ptr, message, None, 0)
        if required == 0:
            self._bridge.check(-1)
        buffer = (ctypes.c_float * required)()
        copied = self._bridge.lib.st_pure_encoder_encode_z_space(
            self._ptr, message, buffer, required
        )
        if copied != required:
            self._bridge.check(-1)
        return list(buffer)

    def __enter__(self) -> "LanguageWaveEncoder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._ptr:
            self._bridge.lib.st_pure_encoder_free(self._ptr)
            self._ptr = None

    def __del__(self) -> None:  # pragma: no cover - destructor safety
        try:
            self.close()
        except Exception:
            pass


class AmegaHypergrad:
    def __init__(
        self,
        bridge: _Lib,
        curvature: float,
        learning_rate: float,
        rows: int,
        cols: int,
        topos: Optional[OpenCartesianTopos] = None,
    ) -> None:
        self._bridge = bridge
        if topos is None:
            ptr = bridge.lib.st_pure_hypergrad_new(curvature, learning_rate, rows, cols)
        else:
            ptr = bridge.lib.st_pure_hypergrad_with_topos(
                curvature, learning_rate, rows, cols, topos._ptr
            )
        if not ptr:
            raise RuntimeError(bridge.last_error() or "failed to create AmegaHypergrad")
        self._ptr = ptr

    @property
    def ptr(self) -> ctypes.c_void_p:
        return ctypes.c_void_p(self._ptr)

    def shape(self) -> Tuple[int, int]:
        rows = ctypes.c_size_t()
        cols = ctypes.c_size_t()
        status = self._bridge.lib.st_pure_hypergrad_shape(self._ptr, ctypes.byref(rows), ctypes.byref(cols))
        self._bridge.check(status)
        return int(rows.value), int(cols.value)

    def accumulate_pair(self, prediction: Sequence[float], target: Sequence[float]) -> None:
        pred_buf = _as_float_buffer(prediction)
        tgt_buf = _as_float_buffer(target)
        status = self._bridge.lib.st_pure_hypergrad_accumulate_pair(
            self._ptr,
            pred_buf,
            len(prediction),
            tgt_buf,
            len(target),
        )
        self._bridge.check(status)

    def absorb_text(self, encoder: LanguageWaveEncoder, text: str) -> None:
        message = text.encode("utf-8")
        status = self._bridge.lib.st_pure_hypergrad_absorb_text(self._ptr, encoder._ptr, message)
        self._bridge.check(status)

    def apply(self, weights: Sequence[float]) -> List[float]:
        buffer = _as_float_buffer(weights)
        status = self._bridge.lib.st_pure_hypergrad_apply(self._ptr, buffer, len(weights))
        self._bridge.check(status)
        return list(buffer)

    def gradient(self) -> List[float]:
        required = self._bridge.lib.st_pure_hypergrad_gradient(self._ptr, None, 0)
        if required == 0:
            self._bridge.check(-1)
        buffer = (ctypes.c_float * required)()
        copied = self._bridge.lib.st_pure_hypergrad_gradient(self._ptr, buffer, required)
        if copied != required:
            self._bridge.check(-1)
        return list(buffer)

    def __enter__(self) -> "AmegaHypergrad":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._ptr:
            self._bridge.lib.st_pure_hypergrad_free(self._ptr)
            self._ptr = None

    def __del__(self) -> None:  # pragma: no cover - destructor safety
        try:
            self.close()
        except Exception:
            pass


def _as_float_buffer(values: Sequence[float]) -> ctypes.Array[ctypes.c_float]:
    if isinstance(values, ctypes.Array):
        return values  # type: ignore[return-value]
    buf = (ctypes.c_float * len(values))()
    for idx, value in enumerate(values):
        buf[idx] = float(value)
    return buf


class PurePythonBridge:
    """High-level convenience wrapper.

    Example usage::

        bridge = PurePythonBridge()
        encoder = bridge.encoder(curvature=-1.0, temperature=0.5)
        hypergrad = bridge.hypergrad(curvature=-1.0, learning_rate=0.05, rows=1, cols=8)
        hypergrad.absorb_text(encoder, "hello spiral torch")
        weights = hypergrad.apply([0.0] * 8)
    """

    def __init__(self, library_path: Optional[os.PathLike[str]] = None) -> None:
        self._lib = _Lib(library_path)

    def __enter__(self) -> "PurePythonBridge":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Release any outstanding resources."""
        self._lib.clear()

    def encoder(self, curvature: float, temperature: float) -> LanguageWaveEncoder:
        return LanguageWaveEncoder(self._lib, curvature, temperature)

    def hypergrad(
        self,
        curvature: float,
        learning_rate: float,
        rows: int,
        cols: int,
        topos: Optional[OpenCartesianTopos] = None,
    ) -> AmegaHypergrad:
        return AmegaHypergrad(self._lib, curvature, learning_rate, rows, cols, topos)

    def topos(
        self,
        curvature: float,
        tolerance: float,
        saturation: float,
        max_depth: int,
        max_volume: int,
    ) -> OpenCartesianTopos:
        return OpenCartesianTopos(self._lib, curvature, tolerance, saturation, max_depth, max_volume)

    def last_error(self) -> Optional[str]:
        return self._lib.last_error()

    def clear_error(self) -> None:
        self._lib.clear()


def _parse_float_sequence(raw: str) -> List[float]:
    """Parse a string containing a sequence of floats.

    The parser first attempts to treat the input as JSON. If parsing fails it
    falls back to comma-separated values so the CLI remains ergonomic.
    """

    text = raw.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        items: Iterable[str] = text.split(",")
    else:
        if isinstance(data, (list, tuple)):
            items = data  # type: ignore[assignment]
        else:
            raise ValueError("Expected a list of numbers")

    floats: List[float] = []
    for item in items:
        floats.append(float(item))
    return floats


def _parse_weight_source(value: Optional[str], value_file: Optional[Path]) -> Optional[List[float]]:
    if value is None and value_file is None:
        return None

    if value is not None and value_file is not None:
        raise ValueError("Specify either --weights or --weights-file, not both.")

    if value_file is not None:
        try:
            text = value_file.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - filesystem failure
            raise SystemExit(f"Failed to read weights file '{value_file}': {exc}") from exc
        value = text.strip()

    if value is None:
        return None

    return _parse_float_sequence(value)


def _parse_pairs_sources(values: Iterable[str], files: Iterable[Path]) -> List[Tuple[List[float], List[float]]]:
    pairs: List[Tuple[List[float], List[float]]] = []
    for entry in values:
        prediction, target = _split_pair(entry)
        pairs.append((prediction, target))

    for file in files:
        try:
            lines = file.read_text(encoding="utf-8").splitlines()
        except OSError as exc:  # pragma: no cover - filesystem failure
            raise SystemExit(f"Failed to read pairs file '{file}': {exc}") from exc
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            prediction_raw, target_raw = _ensure_pair_format(stripped)
            prediction = _parse_float_sequence(prediction_raw)
            target = _parse_float_sequence(target_raw)
            pairs.append((prediction, target))

    return pairs


def _split_pair(entry: str) -> Tuple[List[float], List[float]]:
    prediction_raw, target_raw = _ensure_pair_format(entry)
    prediction = _parse_float_sequence(prediction_raw)
    target = _parse_float_sequence(target_raw)
    return prediction, target


def _ensure_pair_format(entry: str) -> Tuple[str, str]:
    parts = entry.split("|", 1)
    if len(parts) != 2:
        raise SystemExit(f"Invalid pair '{entry}': expected 'prediction|target'")
    return parts[0], parts[1]


def _iter_text_inputs(texts: Iterable[str], files: Iterable[Path]) -> Iterator[str]:
    for text in texts:
        stripped = text.strip()
        if stripped:
            yield stripped
    for file in files:
        try:
            content = file.read_text(encoding="utf-8").strip()
        except OSError as exc:  # pragma: no cover - filesystem failure
            raise SystemExit(f"Failed to read text file '{file}': {exc}") from exc
        if content:
            yield content


def _emit(data: object, pretty: bool) -> None:
    if pretty:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(data, ensure_ascii=False))


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Interact with the pure SpiralTorch bridge")
    parser.add_argument(
        "--library",
        type=Path,
        default=None,
        help="Path to the compiled SpiralTorch cdylib (defaults to release build).",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    encode = subparsers.add_parser("encode", help="Encode text into z-space coordinates")
    encode.add_argument("text", help="UTF-8 text to encode")
    encode.add_argument("--curvature", type=float, default=-1.0)
    encode.add_argument("--temperature", type=float, default=0.5)

    gradient = subparsers.add_parser(
        "hypergrad", help="Accumulate pairs and inspect gradients for a hypergrad"
    )
    gradient.add_argument("--curvature", type=float, default=-1.0)
    gradient.add_argument("--learning-rate", type=float, default=0.05)
    gradient.add_argument("--rows", type=int, required=True)
    gradient.add_argument("--cols", type=int, required=True)
    gradient.add_argument(
        "--pairs",
        action="append",
        default=[],
        metavar="PRED|TARGET",
        help=(
            "Prediction and target sequences separated by '|' (each list can be JSON or comma separated)."
        ),
    )
    gradient.add_argument(
        "--pairs-file",
        action="append",
        type=Path,
        default=[],
        metavar="FILE",
        help="File containing prediction|target pairs (one per line, supports comments).",
    )
    gradient.add_argument(
        "--weights",
        default=None,
        help="Optional initial weights sequence to apply before reading the gradient.",
    )
    gradient.add_argument(
        "--weights-file",
        type=Path,
        default=None,
        help="Load initial weights from a text file (JSON or comma separated).",
    )
    gradient.add_argument(
        "--topos",
        action="store_true",
        help="Construct a reusable OpenCartesianTopos for the hypergrad context.",
    )
    gradient.add_argument("--topos-curvature", type=float, default=-1.0)
    gradient.add_argument("--topos-tolerance", type=float, default=0.05)
    gradient.add_argument("--topos-saturation", type=float, default=1.0)
    gradient.add_argument("--topos-max-depth", type=int, default=8)
    gradient.add_argument("--topos-max-volume", type=int, default=4096)
    gradient.add_argument(
        "--text",
        action="append",
        default=[],
        metavar="TEXT",
        help="Direct text inputs to absorb via the encoder before gradient inspection.",
    )
    gradient.add_argument(
        "--text-file",
        action="append",
        type=Path,
        default=[],
        metavar="FILE",
        help="File containing UTF-8 text to absorb (can be provided multiple times).",
    )
    gradient.add_argument("--encoder-curvature", type=float, default=-1.0)
    gradient.add_argument("--encoder-temperature", type=float, default=0.5)

    info = subparsers.add_parser("info", help="Inspect the resolved library path and last error")
    info.add_argument(
        "--clear",
        action="store_true",
        help="Clear the last error after printing the diagnostics.",
    )

    args = parser.parse_args()

    with PurePythonBridge(args.library) as bridge:
        if args.command == "encode":
            with bridge.encoder(args.curvature, args.temperature) as encoder:
                encoded = encoder.encode_z_space(args.text)
            _emit(encoded, args.pretty)
            return

        if args.command == "hypergrad":
            pairs = _parse_pairs_sources(args.pairs, args.pairs_file)
            try:
                weights = _parse_weight_source(args.weights, args.weights_file)
            except ValueError as exc:  # pragma: no cover - CLI validation
                raise SystemExit(str(exc)) from exc

            topos_context = nullcontext(None)
            if args.topos:
                topos_context = bridge.topos(
                    args.topos_curvature,
                    args.topos_tolerance,
                    args.topos_saturation,
                    args.topos_max_depth,
                    args.topos_max_volume,
                )

            with topos_context as topos:
                with bridge.hypergrad(
                    args.curvature, args.learning_rate, args.rows, args.cols, topos
                ) as hypergrad:
                    for prediction, target in pairs:
                        hypergrad.accumulate_pair(prediction, target)

                    if args.text or args.text_file:
                        with bridge.encoder(
                            args.encoder_curvature, args.encoder_temperature
                        ) as encoder:
                            for text_input in _iter_text_inputs(args.text, args.text_file):
                                hypergrad.absorb_text(encoder, text_input)

                    if weights is not None:
                        hypergrad.apply(weights)

                    gradient_values = hypergrad.gradient()

            _emit(gradient_values, args.pretty)
            return

        if args.command == "info":
            diagnostics = {
                "library": str(bridge.library_path),
                "last_error": bridge.last_error(),
            }
            if args.clear:
                bridge.clear_error()
                diagnostics["cleared"] = True
            _emit(diagnostics, args.pretty)
            return

    raise SystemExit(f"Unknown command: {args.command}")


__all__ = [
    "AmegaHypergrad",
    "LanguageWaveEncoder",
    "OpenCartesianTopos",
    "PurePythonBridge",
    "LibraryLoadError",
    "resolve_library_path",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    _cli()
