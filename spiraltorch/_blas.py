"""Minimal-but-extensible BLAS wrapper for the SpiralTorch stub runtime.

The helpers in this module intentionally avoid pulling in heavy Python
dependencies while still providing a reasonably ergonomic interface for the
stub tensor implementation.  The design goals are:

* Prefer a BLAS backend when it is present but degrade gracefully when it is
  missing.
* Make it possible to override or extend the search path from user code.
* Keep the call surface close to what the Rust bindings expose so that
  exercising the stub feels representative of the compiled module.

The initial iteration of this module focussed solely on discovering
``cblas_dgemm`` and providing a tiny façade over it.  The current version keeps
that simplicity while offering extra introspection and configurability hooks so
that future improvements can layer additional routines on top without having to
reimplement the loader plumbing each time.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from array import array
from threading import Lock
from typing import Iterable

__all__ = ["blas_available", "blas_info", "dgemm", "set_blas_libraries"]

_CBLAS_ROW_MAJOR = 101
_CBLAS_NO_TRANS = 111
_CBLAS_TRANS = 112

_LIB: ctypes.CDLL | None = None
_DGEMM: ctypes._CFuncPtr | None = None  # type: ignore[attr-defined]
_ERROR: BaseException | None = None
_LOCK = Lock()
_USER_CANDIDATES: tuple[str, ...] = ()


def _candidate_paths() -> Iterable[str]:
    hint = os.environ.get("SPIRALTORCH_BLAS_LIB", "").strip()
    if hint:
        for entry in hint.split(os.pathsep):
            entry = entry.strip()
            if entry:
                yield entry
    yield from _USER_CANDIDATES
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


def blas_info() -> dict[str, object]:
    """Return diagnostic information about the BLAS backend state."""

    library = None
    if _LIB is not None:
        library = getattr(_LIB, "_name", None) or getattr(_LIB, "_Handle", None)
    error = _ERROR
    return {
        "available": _DGEMM is not None,
        "library": library,
        "error": None if error is None else repr(error),
        "user_candidates": list(_USER_CANDIDATES),
    }


def set_blas_libraries(*candidates: str) -> None:
    """Override the probe order for BLAS discovery and reinitialise the handle."""

    normalized: list[str] = []
    for candidate in candidates:
        value = str(candidate).strip()
        if value:
            normalized.append(value)

    global _USER_CANDIDATES
    with _LOCK:
        _USER_CANDIDATES = tuple(normalized)
        global _LIB, _DGEMM, _ERROR
        _LIB = None
        _DGEMM = None
        _ERROR = None
        _initialise()


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
    transpose_a: bool = False,
    transpose_b: bool = False,
    lda: int | None = None,
    ldb: int | None = None,
    ldc: int | None = None,
) -> None:
    """Compute ``C = alpha * A @ B + beta * C`` using row-major ``array('d')`` buffers."""

    if not blas_available():
        raise RuntimeError("BLAS backend is unavailable")

    if m < 0 or n < 0 or k < 0:
        raise ValueError("matrix dimensions must be non-negative")

    a_rows, a_cols = (k, m) if transpose_a else (m, k)
    b_rows, b_cols = (n, k) if transpose_b else (k, n)
    expected_a = a_rows * a_cols
    expected_b = b_rows * b_cols
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

    def _resolve_leading_dim(value: int | None, fallback: int, label: str) -> int:
        fallback = int(fallback)
        if value is None:
            return fallback if fallback > 0 else 1
        resolved = int(value)
        if resolved <= 0:
            raise ValueError(f"{label} leading dimension must be a positive integer")
        return resolved

    lda_val = _resolve_leading_dim(
        lda,
        a_cols if not transpose_a else a_rows,
        "matrix A",
    )
    ldb_val = _resolve_leading_dim(
        ldb,
        b_cols if not transpose_b else b_rows,
        "matrix B",
    )
    ldc_val = _resolve_leading_dim(ldc, n, "matrix C")

    trans_a_flag = _CBLAS_TRANS if transpose_a else _CBLAS_NO_TRANS
    trans_b_flag = _CBLAS_TRANS if transpose_b else _CBLAS_NO_TRANS

    with _LOCK:
        _DGEMM(  # type: ignore[misc]
            _CBLAS_ROW_MAJOR,
            trans_a_flag,
            trans_b_flag,
            int(m),
            int(n),
            int(k),
            float(alpha),
            a_ptr,
            int(lda_val if lda_val else 1),
            b_ptr,
            int(ldb_val if ldb_val else 1),
            float(beta),
            c_ptr,
            int(ldc_val if ldc_val else 1),
        )
