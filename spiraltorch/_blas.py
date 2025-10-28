"""Minimal BLAS wrapper used by the SpiralTorch stub runtime."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from array import array
from threading import Lock
from typing import Iterable

__all__ = ["blas_available", "dgemm"]

_CBLAS_ROW_MAJOR = 101
_CBLAS_NO_TRANS = 111

_LIB: ctypes.CDLL | None = None
_DGEMM: ctypes._CFuncPtr | None = None  # type: ignore[attr-defined]
_ERROR: BaseException | None = None
_LOCK = Lock()


def _candidate_paths() -> Iterable[str]:
    hint = os.environ.get("SPIRALTORCH_BLAS_LIB", "").strip()
    if hint:
        for entry in hint.split(os.pathsep):
            entry = entry.strip()
            if entry:
                yield entry
    for name in (
        "openblas",
        "blas",
        "cblas",
        "mkl_rt",
        "Accelerate",
        "vecLib",
    ):
        resolved = ctypes.util.find_library(name)
        if resolved:
            yield resolved
        yield name


def _load_library() -> tuple[ctypes.CDLL | None, BaseException | None]:
    for candidate in _candidate_paths():
        try:
            return ctypes.CDLL(candidate), None
        except OSError as exc:
            last_error: BaseException = exc
    else:
        return None, last_error if "last_error" in locals() else None


def _initialise() -> None:
    global _LIB, _DGEMM, _ERROR
    if _LIB is not None or _ERROR is not None:
        return
    lib, err = _load_library()
    if lib is None:
        _ERROR = err or RuntimeError("no usable BLAS library located")
        return
    try:
        func = lib.cblas_dgemm  # type: ignore[attr-defined]
    except AttributeError as exc:
        _ERROR = exc
        return

    func.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    func.restype = None

    _LIB = lib
    _DGEMM = func


_initialise()


def blas_available() -> bool:
    """Return ``True`` when a BLAS backend has been successfully initialised."""

    return _DGEMM is not None


def _as_double_buffer(buffer: array) -> ctypes.Array[ctypes.c_double]:
    if not isinstance(buffer, array) or buffer.typecode != "d":
        raise TypeError("BLAS buffers must be array('d') instances")
    return (ctypes.c_double * len(buffer)).from_buffer(buffer)


def dgemm(
    m: int,
    n: int,
    k: int,
    a: array,
    b: array,
    c: array,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> None:
    """Compute ``C = alpha * A @ B + beta * C`` using row-major ``array('d')`` buffers."""

    if not blas_available():
        raise RuntimeError("BLAS backend is unavailable")

    if m < 0 or n < 0 or k < 0:
        raise ValueError("matrix dimensions must be non-negative")

    expected_a = m * k
    expected_b = k * n
    expected_c = m * n
    if len(a) != expected_a:
        raise ValueError(f"matrix A has {len(a)} elements, expected {expected_a}")
    if len(b) != expected_b:
        raise ValueError(f"matrix B has {len(b)} elements, expected {expected_b}")
    if len(c) != expected_c:
        raise ValueError(f"matrix C has {len(c)} elements, expected {expected_c}")

    if expected_c == 0:
        return

    a_ptr = _as_double_buffer(a)
    b_ptr = _as_double_buffer(b)
    c_ptr = _as_double_buffer(c)

    with _LOCK:
        _DGEMM(  # type: ignore[misc]
            _CBLAS_ROW_MAJOR,
            _CBLAS_NO_TRANS,
            _CBLAS_NO_TRANS,
            int(m),
            int(n),
            int(k),
            float(alpha),
            a_ptr,
            int(k if k else 1),
            b_ptr,
            int(n if n else 1),
            float(beta),
            c_ptr,
            int(n if n else 1),
        )
