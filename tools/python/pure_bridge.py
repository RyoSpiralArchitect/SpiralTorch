"""Minimal ctypes helpers for driving the pure SpiralTorch stack from Python.

This module purposefully avoids NumPy and PyTorch so pure CPython environments can
pair with the zero-dependency Rust pipeline. Only the standard library is used
and buffers are exchanged as Python lists.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


class _Lib:
    def __init__(self, path: Optional[os.PathLike[str]] = None) -> None:
        if path is None:
            path = self._default_path()
        self._cdll = ctypes.CDLL(str(path))
        self._configure()

    @staticmethod
    def _default_path() -> Path:
        root = Path(__file__).resolve().parents[2]
        target = root / "target" / "release"
        if sys.platform.startswith("linux"):
            name = "libst_tensor.so"
        elif sys.platform == "darwin":
            name = "libst_tensor.dylib"
        elif sys.platform in ("win32", "cygwin"):
            name = "st_tensor.dll"
        else:
            raise RuntimeError(f"Unsupported platform: {sys.platform}")
        candidate = target / name
        if not candidate.exists():
            raise FileNotFoundError(
                f"Could not locate {candidate}. Build the cdylib with 'cargo build --release -p st-tensor'."
            )
        return candidate

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
]
