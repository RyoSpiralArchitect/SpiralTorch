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

import ctypes
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


__all__ = [
    "AmegaHypergrad",
    "LanguageWaveEncoder",
    "OpenCartesianTopos",
    "PurePythonBridge",
    "LibraryLoadError",
    "resolve_library_path",
]
