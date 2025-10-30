"""Minimal BLAS wrapper used by the SpiralTorch stub runtime."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import warnings
from array import array
from contextlib import contextmanager
from threading import Lock
from typing import Iterable

__all__ = [
    "blas_available",
    "configure_threads",
    "current_thread_count",
    "thread_controls_available",
    "temporary_thread_count",
    "blas_vendor",
    "dgemm",
]

_CBLAS_ROW_MAJOR = 101
_CBLAS_NO_TRANS = 111

_LIB: ctypes.CDLL | None = None
_DGEMM: ctypes._CFuncPtr | None = None  # type: ignore[attr-defined]
_THREAD_SETTER: ctypes._CFuncPtr | None = None  # type: ignore[attr-defined]
_THREAD_GETTER: ctypes._CFuncPtr | None = None  # type: ignore[attr-defined]
_ERROR: BaseException | None = None
_LOCK = Lock()
_VENDOR: str | None = None
_THREAD_LAST_SET: int | None = None


def _candidate_paths() -> Iterable[str]:
    hint = os.environ.get("SPIRALTORCH_BLAS_LIB", "").strip()
    if hint:
        for entry in hint.split(os.pathsep):
            entry = entry.strip()
            if entry:
                yield entry
    for name in (
        "spiraltorch_blas",  # allow custom builds to be hinted first
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


def _locate_thread_controls(lib: ctypes.CDLL) -> None:
    """Attempt to locate thread control helpers exposed by the BLAS library."""

    global _THREAD_SETTER, _THREAD_GETTER

    setter_candidates: tuple[tuple[str, type[ctypes._SimpleCData]], ...] = (
        ("openblas_set_num_threads64", ctypes.c_longlong),
        ("openblas_set_num_threads64_v2", ctypes.c_longlong),
        ("openblas_set_num_threads", ctypes.c_int),
        ("openblas_set_num_threads_v2", ctypes.c_int),
        ("cblas_set_num_threads", ctypes.c_int),
        ("MKL_Set_Num_Threads", ctypes.c_int),
        ("MKL_Set_Num_Threads_Loc", ctypes.c_int),
        ("MKL_Set_Num_Threads_Local", ctypes.c_int),
        ("bli_thread_set_num_threads", ctypes.c_int),
        ("flexiblas_set_num_threads", ctypes.c_int),
        ("goto_set_num_threads", ctypes.c_int),
        ("veclib_set_num_threads", ctypes.c_int),
        ("omp_set_num_threads", ctypes.c_int),
    )

    getter_candidates: tuple[tuple[str, type[ctypes._SimpleCData]], ...] = (
        ("openblas_get_num_threads64", ctypes.c_longlong),
        ("openblas_get_num_threads", ctypes.c_int),
        ("cblas_get_num_threads", ctypes.c_int),
        ("MKL_Get_Max_Threads", ctypes.c_int),
        ("bli_thread_get_num_threads", ctypes.c_int),
        ("flexiblas_get_num_threads", ctypes.c_int),
        ("goto_get_num_threads", ctypes.c_int),
        ("veclib_get_num_threads", ctypes.c_int),
        ("omp_get_max_threads", ctypes.c_int),
    )

    for name, argtype in setter_candidates:
        try:
            func = getattr(lib, name)
        except AttributeError:
            continue
        try:
            func.argtypes = [argtype]
            func.restype = None
        except TypeError:
            continue
        _THREAD_SETTER = func
        break

    for name, restype in getter_candidates:
        try:
            func = getattr(lib, name)
        except AttributeError:
            continue
        try:
            func.argtypes = []
            func.restype = restype
        except TypeError:
            continue
        _THREAD_GETTER = func
        break


def _decode_bytes(value: bytes | None) -> str | None:
    if not value:
        return None
    try:
        return value.decode("utf-8").strip()
    except UnicodeDecodeError:
        return value.decode("latin1", "ignore").strip()


def _identify_vendor(lib: ctypes.CDLL) -> None:
    global _VENDOR

    # OpenBLAS exposes detailed build configuration strings.
    try:
        get_config = getattr(lib, "openblas_get_config")
        get_config.argtypes = []
        get_config.restype = ctypes.c_char_p
        config = _decode_bytes(get_config())
        if config:
            corename: str | None = None
            try:
                get_core = getattr(lib, "openblas_get_corename")
                get_core.argtypes = []
                get_core.restype = ctypes.c_char_p
                corename = _decode_bytes(get_core())
            except AttributeError:
                corename = None
            if corename:
                _VENDOR = f"OpenBLAS ({corename}; {config})"
            else:
                _VENDOR = f"OpenBLAS ({config})"
            return
    except AttributeError:
        pass

    # BLIS publishes its version through the info helper.
    try:
        get_blis_version = getattr(lib, "bli_info_get_version_str")
        get_blis_version.argtypes = []
        get_blis_version.restype = ctypes.c_char_p
        version = _decode_bytes(get_blis_version())
        if version:
            _VENDOR = f"BLIS ({version})"
            return
    except AttributeError:
        pass

    # FlexiBLAS provides indirection over multiple backends and exposes helper APIs.
    try:
        get_backend = getattr(lib, "flexiblas_get_current_backend")
        get_backend.argtypes = []
        get_backend.restype = ctypes.c_char_p
        backend = _decode_bytes(get_backend())

        get_version = getattr(lib, "flexiblas_get_version")
        get_version.argtypes = []
        get_version.restype = ctypes.c_char_p
        version = _decode_bytes(get_version())

        details = backend or "unknown backend"
        if version:
            details = f"{details}; {version}"
        _VENDOR = f"FlexiBLAS ({details})"
        return
    except AttributeError:
        pass

    # Intel MKL provides a descriptive version string via MKL_Get_Version_String.
    try:
        get_mkl_version = getattr(lib, "MKL_Get_Version_String")
        get_mkl_version.argtypes = [ctypes.c_char_p, ctypes.c_int]
        get_mkl_version.restype = None
        buffer = ctypes.create_string_buffer(512)
        get_mkl_version(buffer, ctypes.sizeof(buffer))
        version = _decode_bytes(buffer.value)
        if version:
            _VENDOR = f"Intel MKL ({version})"
            return
    except AttributeError:
        pass
    except OSError:
        # Some MKL builds lazily resolve symbols; ignore failures and fall through.
        pass

    # Apple's Accelerate / vecLib stack does not expose query helpers, but
    # the shared library name is often descriptive enough.
    libname = getattr(lib, "_name", None)
    if isinstance(libname, str):
        if "Accelerate" in libname:
            _VENDOR = "Apple Accelerate"
            return
        if "vecLib" in libname:
            _VENDOR = "Apple vecLib"
            return
        if "spiraltorch" in libname:
            _VENDOR = "SpiralTorch custom BLAS"
            return

    # Fall back to a generic identifier if nothing more specific is available.
    if not _VENDOR:
        _VENDOR = "generic BLAS"


def _configure_threads_from_env() -> None:
    value = os.environ.get("SPIRALTORCH_BLAS_THREADS")
    if not value:
        return
    try:
        requested = int(value)
    except ValueError:
        warnings.warn(
            "Ignoring invalid SPIRALTORCH_BLAS_THREADS value; expected an integer",
            RuntimeWarning,
        )
        return

    try:
        configure_threads(requested)
    except RuntimeError as exc:
        warnings.warn(
            f"Unable to configure BLAS thread count: {exc}",
            RuntimeWarning,
        )
    except ValueError as exc:
        warnings.warn(str(exc), RuntimeWarning)


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

    _locate_thread_controls(lib)
    _identify_vendor(lib)

    _LIB = lib
    _DGEMM = func

    _configure_threads_from_env()


_initialise()


def blas_available() -> bool:
    """Return ``True`` when a BLAS backend has been successfully initialised."""

    return _DGEMM is not None


def configure_threads(threads: int) -> None:
    """Configure the number of threads used by the underlying BLAS library."""

    global _THREAD_LAST_SET

    if not blas_available():
        raise RuntimeError("BLAS backend is unavailable")

    if threads <= 0:
        raise ValueError("thread count must be a positive integer")

    if _THREAD_SETTER is None:
        raise RuntimeError("loaded BLAS library does not expose thread control APIs")

    _THREAD_SETTER(int(threads))  # type: ignore[misc]
    _THREAD_LAST_SET = int(threads)


def current_thread_count() -> int | None:
    """Return the current BLAS thread count, or ``None`` if unavailable."""

    global _THREAD_LAST_SET

    if not blas_available():
        raise RuntimeError("BLAS backend is unavailable")

    if _THREAD_GETTER is None:
        return _THREAD_LAST_SET

    current = int(_THREAD_GETTER())
    _THREAD_LAST_SET = current
    return current


def thread_controls_available() -> bool:
    """Return ``True`` when the loaded BLAS exposes thread control helpers."""

    if not blas_available():
        raise RuntimeError("BLAS backend is unavailable")

    return _THREAD_SETTER is not None


@contextmanager
def temporary_thread_count(threads: int):
    """Temporarily override the BLAS thread count within the managed context."""

    target = int(threads)

    if target <= 0:
        raise ValueError("thread count must be a positive integer")

    if not thread_controls_available():
        raise RuntimeError("loaded BLAS library does not expose thread control APIs")

    previous: int | None
    try:
        previous = current_thread_count()
    except RuntimeError:
        previous = None

    if previous == target:
        yield
        return

    configure_threads(target)
    try:
        yield
    finally:
        if previous is not None and previous > 0:
            try:
                configure_threads(previous)
            except Exception as exc:  # pragma: no cover - defensive fallback
                warnings.warn(
                    f"Failed to restore previous BLAS thread count: {exc}",
                    RuntimeWarning,
                )


def blas_vendor() -> str:
    """Return a descriptive string for the detected BLAS implementation."""

    if not blas_available():
        raise RuntimeError("BLAS backend is unavailable")

    return _VENDOR or "generic BLAS"


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
