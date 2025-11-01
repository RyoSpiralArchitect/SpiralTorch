"""Minimal BLAS wrapper used by the SpiralTorch stub runtime.

The helpers in this module have gradually expanded beyond merely loading a
``dgemm`` symbol.  The thread-tuning primitives we expose are now aware of the
wider environment so callers can coordinate BLAS parallelism with the rest of
the system.  Beyond the CPU the routines now integrate with the WGPU autotuning
artefacts emitted by the Rust runtime, allowing SpiralTorch to reserve host
threads for GPU command submission so compute on both sides can progress
without contending for the same CPU cores.  The system-facing helpers also
interrogate Linux control groups so container limits and CPU reservations are
captured before tuning decisions are made.  The expanded system snapshot now
records CPU topology from ``/sys/devices/system/cpu`` so physical cores can be
weighed against logical CPUs when scheduling work alongside GPU pipelines while
Linux Pressure Stall Information (PSI) metrics feed into the heuristics so BLAS
threads back off when the host is already saturated.  WGSL kernel metadata
persisted in the autotune store is summarised so GPU dispatch characteristics
can feed back into the CPU reservation heuristic.  The CPU accounting helpers
also honour kernel-level isolation controls such as ``isolcpus`` and
``nohz_full`` so cores reserved for real-time workloads remain untouched by BLAS
threads.

The scope has now broadened further.  A compute-only WGSL pipeline allows the
module to dispatch GEMM workloads directly to WGPU-capable GPUs so SpiralTorch
can bypass CPU BLAS implementations entirely when the caller opts into GPU
execution.  The dispatcher keeps track of device and pipeline state so repeated
invocations amortise setup overheads and the interface gracefully degrades when
WGPU is unavailable.  Thread recommendations are no longer purely heuristic:
observed execution times are persisted and folded back into future tuning
decisions via a lightweight bandit learner that explores thread counts while
guarding against pathological regressions.  Finally a WebAssembly-oriented
fallback exposes BLAS-style helpers that continue to work when the runtime is
executed inside a browser through Pyodide or similar hosts, using typed-array
buffers to deliver a pure Web environment implementation that remains API
compatible with the CPU BLAS entry points.
"""

from __future__ import annotations

import atexit
import asyncio
import ctypes
import ctypes.util
import importlib
import json
import math
import os
import random
import struct
import sys
import time
import warnings
from array import array
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Iterable

__all__ = [
    "blas_available",
    "configure_threads",
    "current_thread_count",
    "thread_controls_available",
    "temporary_thread_count",
    "system_cpu_count",
    "system_cpu_topology",
    "system_cpu_isolations",
    "recommended_thread_count",
    "synchronise_thread_hints",
    "auto_tune_threads",
    "blas_vendor",
    "thread_learning_snapshot",
    "wgpu_adapter_info",
    "wgpu_kernel_profiles",
    "gpu_host_thread_reservation",
    "gpu_blas_available",
    "gpu_dgemm",
    "wasm_blas_available",
    "wasm_dgemm",
    "available_dgemm_backends",
    "select_dgemm_backend",
    "auto_dgemm",
    "process_cpu_budget",
    "cgroup_cpu_quota",
    "system_cpu_pressure",
    "dgemm",
]

try:  # pragma: no cover - optional dependency detection
    _WGPU_MODULE = importlib.import_module("wgpu")
    try:  # initialise the default backend eagerly when possible
        importlib.import_module("wgpu.backends.rs")
    except Exception:
        pass
    try:
        from wgpu.utils import get_default_device as _wgpu_get_default_device
    except Exception:  # pragma: no cover - utils may not be available
        _wgpu_get_default_device = None  # type: ignore[assignment]
    _wgpu_read_buffer = None
except Exception:  # pragma: no cover - wgpu entirely unavailable
    _WGPU_MODULE = None
    _wgpu_get_default_device = None  # type: ignore[assignment]
    _wgpu_read_buffer = None  # type: ignore[assignment]

try:  # pragma: no cover - only on Pyodide / WASM environments
    import js  # type: ignore[attr-defined]
    from pyodide.ffi import create_proxy, to_js  # type: ignore
except Exception:  # pragma: no cover - desktop CPython
    js = None  # type: ignore[assignment]
    create_proxy = None  # type: ignore[assignment]
    to_js = None  # type: ignore[assignment]

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
_THREAD_HINT_SOURCES: tuple[str, ...] = (
    "SPIRALTORCH_BLAS_THREADS",
    "SPIRALTORCH_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_GPU_HOST_HINT_VAR = "SPIRALTORCH_WGPU_HOST_THREADS"

_WGPU_ADAPTER_INFO: dict[str, object] | None = None
_WGPU_KERNEL_PROFILES: dict[str, object] | None = None
_GPU_HOST_RESERVATION: int | None = None
_GPU_HOST_RESERVATION_BUDGET: int | None = None
_WGPU_DEVICE: object | None = None
_WGPU_QUEUE: object | None = None
_WGPU_PIPELINES: dict[int, object] = {}
_WGPU_BIND_LAYOUT: object | None = None
_WGPU_PARAMS_SIZE = struct.calcsize("<8I4f")

_CPU_BUDGET_CACHE: dict[str, object] | None = None
_CPU_TOPOLOGY_CACHE: tuple[int, tuple[int, ...] | None, dict[str, object]] | None = None
_CPU_ISOLATION_CACHE: tuple[int, dict[str, object] | None] | None = None
_CPU_PRESSURE_CACHE: tuple[int, float, dict[str, object] | None] | None = None

_CPU_PRESSURE_CACHE_TTL = 2.0

_CGROUP_SELF_CACHE: dict[str, str] | None = None
_CGROUP_QUOTA_CACHE: tuple[int, float | None] | None = None
_CGROUP_CPUSET_CACHE: tuple[int, int | None] | None = None

_LEARNING_CACHE: dict[str, object] | None = None
_LEARNING_CACHE_MTIME: float = 0.0
_LEARNING_CACHE_DIRTY = False
_LEARNING_LAST_FLUSH = 0.0
_LEARNING_LOCK = Lock()
_LEARNING_MAX_SAMPLES = 64
_LEARNING_MIN_OBSERVATIONS = 3
try:
    _LEARNING_EPSILON = float(
        os.environ.get("SPIRALTORCH_BLAS_LEARNING_EPSILON", "0.15") or 0.15
    )
except ValueError:
    _LEARNING_EPSILON = 0.15

_WASM_STATE: dict[str, object] | None = None

_INTEGRATED_GPU_VENDORS = {
    0x8086,  # Intel
    0x106B,  # Apple
    0x13B5,  # ARM
    0x5143,  # Qualcomm
    0x1AF4,  # virtio
    0x1010,  # Imagination Technologies
}
_SOFTWARE_GPU_VENDORS = {
    0x1414,  # Microsoft (WARP)
}

_SOFTWARE_GPU_KEYWORDS = {
    "swiftshader",
    "llvmpipe",
    "software",
    "warp",
}

_WGPU_PROFILE_STALE_SECONDS = 90 * 24 * 60 * 60
_WGPU_HEAVY_WORKGROUP_THRESHOLD = 256
_WGPU_HEAVY_VOLUME_THRESHOLD = 2_000_000


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


def _read_int_env(variable: str) -> int | None:
    value = os.environ.get(variable)
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        if variable.startswith("SPIRALTORCH"):
            warnings.warn(
                f"Ignoring invalid {variable} value; expected an integer",
                RuntimeWarning,
            )
        return None


def _normalise_wgpu_token(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.replace("_", " ").replace("-", " ")
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    result = cleaned.strip()
    return result or None


def _classify_wgpu_adapter(
    vendor: int | None, *, backend: str | None, name: str | None
) -> str:
    lowered_name = (name or "").lower()
    if vendor is not None:
        if vendor in _SOFTWARE_GPU_VENDORS:
            return "software"
        if vendor in _INTEGRATED_GPU_VENDORS:
            return "integrated"
    if any(keyword in lowered_name for keyword in _SOFTWARE_GPU_KEYWORDS):
        return "software"
    backend_lower = (backend or "").lower()
    if vendor is None:
        if backend_lower in {"gl", "gles"}:
            return "integrated"
        if lowered_name:
            return "unknown"
    if backend_lower in {"metal", "vulkan"} and vendor in _INTEGRATED_GPU_VENDORS:
        return "integrated"
    if vendor is None:
        return "unknown"
    return "discrete"


def _parse_wgpu_autotune_key(key: str) -> dict[str, object] | None:
    if not key.startswith("wgpu."):
        return None
    parts = key.split("|")
    if len(parts) < 4:
        return None
    name_token = parts[1] if len(parts) > 1 else None
    vendor_token = parts[2] if len(parts) > 2 else None
    device_token = parts[3] if len(parts) > 3 else None
    backend = parts[4] if len(parts) > 4 else None
    driver = parts[5] if len(parts) > 5 else None
    driver_info = parts[6] if len(parts) > 6 else None

    try:
        vendor = int(vendor_token, 16) if vendor_token else None
    except ValueError:
        vendor = None
    try:
        device = int(device_token, 16) if device_token else None
    except ValueError:
        device = None

    name = _normalise_wgpu_token(name_token)
    classification = _classify_wgpu_adapter(vendor, backend=backend, name=name)

    return {
        "name": name,
        "vendor_id": vendor,
        "vendor_hex": vendor_token,
        "device_id": device,
        "device_hex": device_token,
        "backend": backend,
        "driver": _normalise_wgpu_token(driver),
        "driver_info": _normalise_wgpu_token(driver_info),
        "classification": classification,
        "key": key,
    }


def _autotune_store_candidates() -> Iterable[str]:
    seen: set[str] = set()
    env_path = os.environ.get("SPIRALTORCH_AUTOTUNE_STORE")
    if env_path:
        env_path = env_path.strip()
        if env_path:
            seen.add(env_path)
            yield env_path
    home = os.environ.get("HOME")
    if home:
        default_path = os.path.join(home, ".spiraltorch", "kernels.json")
        if default_path not in seen:
            yield default_path


def _load_wgpu_adapter_from_store() -> dict[str, object] | None:
    for candidate in _autotune_store_candidates():
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        items = list(data.items())
        for key, _ in reversed(items):
            info = _parse_wgpu_autotune_key(str(key))
            if info:
                info["source"] = candidate
                return info
    return None


def _detect_wgpu_adapter() -> dict[str, object] | None:
    global _WGPU_ADAPTER_INFO
    if _WGPU_ADAPTER_INFO is not None:
        return _WGPU_ADAPTER_INFO

    env_hint = os.environ.get("SPIRALTORCH_WGPU_ADAPTER_INFO")
    if env_hint:
        parsed: dict[str, object] | None
        try:
            maybe = json.loads(env_hint)
            parsed = maybe if isinstance(maybe, dict) else None
        except json.JSONDecodeError:
            parsed = None
        if parsed is None:
            parsed = _parse_wgpu_autotune_key(env_hint)
        if parsed is not None:
            _WGPU_ADAPTER_INFO = parsed
            return _WGPU_ADAPTER_INFO

    _WGPU_ADAPTER_INFO = _load_wgpu_adapter_from_store()
    return _WGPU_ADAPTER_INFO


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        rounded = int(value)
        return rounded if rounded >= 0 and abs(value - rounded) < 1e-6 else None
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            parsed = int(candidate, 0)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def _coerce_non_negative_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        candidate = float(value)
        if not math.isfinite(candidate) or candidate < 0:
            return None
        return candidate
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            parsed = float(candidate)
        except ValueError:
            return None
        if not math.isfinite(parsed) or parsed < 0:
            return None
        return parsed
    return None


def _estimate_context_volume(context: object) -> int | None:
    if not isinstance(context, dict):
        return None
    dims: list[int] = []
    for key, raw in context.items():
        if not isinstance(key, str):
            continue
        lowered = key.lower()
        if any(token in lowered for token in ("rows", "cols", "inner", "m", "n", "k", "width", "height", "depth", "size", "len", "length", "volume")):
            value = _coerce_positive_int(raw)
            if value:
                dims.append(value)
    dims = [value for value in dims if value]
    if not dims:
        return None
    dims.sort(reverse=True)
    volume = 1
    for value in dims[:3]:
        volume *= value
    return volume


def _estimate_workgroup_size(params: object) -> int | None:
    if not isinstance(params, dict):
        return None
    direct = _coerce_positive_int(params.get("workgroup_size"))
    if direct:
        return direct
    tile_m = _coerce_positive_int(params.get("tile_m"))
    tile_n = _coerce_positive_int(params.get("tile_n"))
    tile_k = _coerce_positive_int(params.get("tile_k"))
    if tile_m and tile_n:
        if tile_k and tile_k > 1:
            return tile_m * tile_n * tile_k
        return tile_m * tile_n
    axes: list[int] = []
    for key in ("workgroup_x", "workgroup_y", "workgroup_z", "local_size_x", "local_size_y", "local_size_z", "wg_rows", "wg_cols"):
        value = _coerce_positive_int(params.get(key))
        if value:
            axes.append(value)
    if axes:
        size = 1
        for axis in axes:
            size *= axis
        return size
    for key, raw in params.items():
        if not isinstance(key, str):
            continue
        lowered = key.lower()
        if lowered.endswith("_workgroup") or lowered.endswith("_workgroup_size"):
            value = _coerce_positive_int(raw)
            if value:
                return value
    return None


def _summarise_wgpu_buckets(store: object) -> dict[str, object] | None:
    if not isinstance(store, dict):
        return None
    kernels: dict[str, dict[str, object]] = {}
    latest: int | None = None
    total_entries = 0
    for raw_key, bucket in store.items():
        key = str(raw_key)
        if not key.startswith("wgpu."):
            continue
        info = _parse_wgpu_autotune_key(key)
        kernel_prefix = key.split("|", 1)[0]
        parts = kernel_prefix.split(".")
        kernel_name = parts[1] if len(parts) > 1 else kernel_prefix
        entries: list[dict[str, object]] = []
        if isinstance(bucket, dict):
            maybe_entries = bucket.get("entries")
            if isinstance(maybe_entries, list):
                entries = [entry for entry in maybe_entries if isinstance(entry, dict)]
            elif {"params", "score", "updated_unix"}.issubset(bucket.keys()):
                entries = [bucket]
        elif isinstance(bucket, list):
            entries = [entry for entry in bucket if isinstance(entry, dict)]
        if not entries:
            continue
        summary = kernels.setdefault(
            kernel_name,
            {
                "entries": 0,
                "max_workgroup": 0,
                "max_volume": 0,
                "recent_update": None,
                "prefix": kernel_prefix,
            },
        )
        if info:
            summary.setdefault("adapter", info.get("name"))
            summary.setdefault("backend", info.get("backend"))
            summary.setdefault("classification", info.get("classification"))
        for entry in entries:
            total_entries += 1
            summary["entries"] = int(summary.get("entries", 0)) + 1
            updated = _coerce_positive_int(entry.get("updated_unix"))
            if updated is not None:
                summary["recent_update"] = max(summary.get("recent_update") or 0, updated)
                if latest is None or updated > latest:
                    latest = updated
            context = entry.get("context")
            volume = _estimate_context_volume(context)
            if volume is not None:
                summary["max_volume"] = max(int(summary.get("max_volume", 0)), volume)
            params = entry.get("params")
            workgroup = _estimate_workgroup_size(params)
            if workgroup is not None:
                summary["max_workgroup"] = max(int(summary.get("max_workgroup", 0)), workgroup)
        summary["entries"] = int(summary.get("entries", 0))
        summary["max_workgroup"] = int(summary.get("max_workgroup", 0))
        summary["max_volume"] = int(summary.get("max_volume", 0))
        if summary.get("recent_update") is None:
            summary.pop("recent_update", None)
    if not kernels:
        return None
    profile: dict[str, object] = {
        "kernels": kernels,
        "entries": total_entries,
    }
    if latest is not None:
        profile["most_recent"] = latest
    return profile


def _collect_wgpu_kernel_profiles() -> dict[str, object] | None:
    for candidate in _autotune_store_candidates():
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            continue
        profile = _summarise_wgpu_buckets(data)
        if profile:
            profile["source"] = candidate
            return profile
    return None


def wgpu_kernel_profiles(*, refresh: bool = False) -> dict[str, object] | None:
    """Return aggregated metadata about autotuned WGPU/WGSL kernels."""

    global _WGPU_KERNEL_PROFILES
    if _WGPU_KERNEL_PROFILES is not None and not refresh:
        return dict(_WGPU_KERNEL_PROFILES)

    profile = _collect_wgpu_kernel_profiles()
    _WGPU_KERNEL_PROFILES = dict(profile) if profile is not None else None
    return dict(profile) if profile is not None else None


def _wgpu_kernel_pressure_details() -> tuple[bool, int, int]:
    profile = wgpu_kernel_profiles()
    if not profile:
        return False, 0, 0
    kernels = profile.get("kernels")
    if not isinstance(kernels, dict):
        return False, 0, 0
    max_workgroup = 0
    max_volume = 0
    for stats in kernels.values():
        if not isinstance(stats, dict):
            continue
        workgroup = _coerce_positive_int(stats.get("max_workgroup"))
        if workgroup is not None:
            max_workgroup = max(max_workgroup, workgroup)
        volume = _coerce_positive_int(stats.get("max_volume"))
        if volume is not None:
            max_volume = max(max_volume, volume)
    stale = False
    most_recent = profile.get("most_recent")
    if isinstance(most_recent, (int, float)):
        try:
            stale = (time.time() - float(most_recent)) > _WGPU_PROFILE_STALE_SECONDS
        except OverflowError:
            stale = False
    heavy = False
    if not stale:
        if max_workgroup >= _WGPU_HEAVY_WORKGROUP_THRESHOLD:
            heavy = True
        elif max_volume >= _WGPU_HEAVY_VOLUME_THRESHOLD:
            heavy = True
    return heavy, max_workgroup, max_volume


def _apply_budget_cap(reservation: int, budget: int | None) -> int:
    reservation = max(0, reservation)
    if budget is None:
        return reservation
    cap = max(0, budget - 1)
    if cap == 0:
        return 0
    return min(reservation, cap)


def _compute_gpu_host_reservation(cpu_budget: int | None) -> int | None:
    explicit = _read_int_env(_GPU_HOST_HINT_VAR)
    if explicit is not None:
        return max(0, explicit)

    info = _detect_wgpu_adapter()
    if not info:
        return None

    heavy_gpu, max_workgroup, max_volume = _wgpu_kernel_pressure_details()
    classification = str(info.get("classification", "unknown"))
    if classification == "software":
        return None
    if classification == "integrated":
        if cpu_budget is None or cpu_budget <= 2:
            reservation = 1
        else:
            reservation = min(2, max(1, cpu_budget // 4))
        if heavy_gpu:
            target = max(reservation + 1, 2)
            if max_volume >= (_WGPU_HEAVY_VOLUME_THRESHOLD * 2):
                target = max(target, reservation + 2)
            reservation = _apply_budget_cap(target, cpu_budget)
            if reservation < 1:
                reservation = 1
        return reservation
    if classification == "discrete":
        if cpu_budget is not None and cpu_budget >= 8:
            reservation = 2
        else:
            reservation = 1
        if heavy_gpu:
            target = reservation + 1
            if max_workgroup >= _WGPU_HEAVY_WORKGROUP_THRESHOLD * 2:
                target = max(target, 3)
            reservation = _apply_budget_cap(target, cpu_budget)
            if reservation < 1 and (cpu_budget is None or cpu_budget > 1):
                reservation = 1
        return reservation
    if classification == "unknown":
        if cpu_budget is None or cpu_budget <= 1:
            return None
        reservation = 1
        if heavy_gpu:
            target = reservation + 1
            reservation = _apply_budget_cap(target, cpu_budget)
            if reservation < 1:
                reservation = 1
        return reservation
    return None


def _propagate_gpu_host_hint(reservation: int | None) -> None:
    if reservation is None:
        os.environ.pop(_GPU_HOST_HINT_VAR, None)
        return
    os.environ[_GPU_HOST_HINT_VAR] = str(int(reservation))


def gpu_host_thread_reservation(
    cpu_budget: int | None = None, *, refresh: bool = False
) -> int | None:
    """Return the number of CPU threads reserved for GPU host work.

    Parameters
    ----------
    cpu_budget:
        Optional upper bound on the CPU threads that should remain available to
        BLAS workloads.  When provided the reservation heuristic scales its
        result relative to this limit instead of the raw system CPU count.
    refresh:
        Force a recomputation of the cached reservation value even when the
        inputs have not changed.
    """

    global _GPU_HOST_RESERVATION, _GPU_HOST_RESERVATION_BUDGET

    if cpu_budget is not None:
        try:
            budget = max(0, int(cpu_budget))
        except (TypeError, ValueError):
            budget = None
    else:
        budget = system_cpu_count()

    if refresh or _GPU_HOST_RESERVATION is None or _GPU_HOST_RESERVATION_BUDGET != budget:
        _GPU_HOST_RESERVATION = _compute_gpu_host_reservation(budget)
        _GPU_HOST_RESERVATION_BUDGET = budget

    return _GPU_HOST_RESERVATION


def _thread_learning_state_path() -> Path:
    base = os.environ.get("SPIRALTORCH_STATE_DIR") or os.environ.get(
        "SPIRALTORCH_CACHE"
    )
    if base:
        return Path(base).expanduser() / "blas_thread_learning.json"

    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        root = Path(cache_home).expanduser() / "spiraltorch"
    else:
        try:
            home = Path.home()
        except Exception:  # pragma: no cover - Pyodide / sandboxed envs
            home = Path(os.getcwd())
        root = home / ".cache" / "spiraltorch"
    return root / "blas_thread_learning.json"


def _thread_learning_flush(force: bool = False) -> None:
    global _LEARNING_CACHE_DIRTY, _LEARNING_LAST_FLUSH

    if not _LEARNING_CACHE_DIRTY or _LEARNING_CACHE is None:
        return

    now = time.time()
    if not force and (now - _LEARNING_LAST_FLUSH) < 1.5:
        return

    path = _thread_learning_state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    tmp_path = path.with_suffix(".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(_LEARNING_CACHE, handle, sort_keys=True)
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        return

    _LEARNING_CACHE_DIRTY = False
    _LEARNING_LAST_FLUSH = now


atexit.register(_thread_learning_flush, True)


def _load_learning_cache() -> dict[str, object]:
    global _LEARNING_CACHE, _LEARNING_CACHE_MTIME

    if _LEARNING_CACHE is not None:
        return _LEARNING_CACHE

    path = _thread_learning_state_path()
    cache: dict[str, object]
    if path.exists():
        try:
            payload = path.read_text(encoding="utf-8")
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                raise TypeError("learning cache must be a JSON object")
            cache = parsed
            cache.setdefault("buckets", {})
            try:
                _LEARNING_CACHE_MTIME = path.stat().st_mtime
            except OSError:
                _LEARNING_CACHE_MTIME = 0.0
        except Exception:
            cache = {"buckets": {}}
            _LEARNING_CACHE_MTIME = 0.0
    else:
        cache = {"buckets": {}}
        _LEARNING_CACHE_MTIME = 0.0

    _LEARNING_CACHE = cache
    return cache


def _thread_problem_bucket(problem_size: tuple[int, int, int]) -> str:
    dims = [max(1, int(d)) for d in problem_size]
    ordering = sorted(dims)
    logs = [str(int(math.log2(v))) if v > 1 else "0" for v in ordering]
    baseline = max(1, ordering[0])
    ratios = [str(int(round(v / baseline))) for v in dims]
    return "|".join(logs + ["x".join(ratios)])


def _choose_learned_threads(
    problem_size: tuple[int, int, int],
    limit: int | None,
) -> int | None:
    with _LEARNING_LOCK:
        cache = _load_learning_cache()
        buckets = cache.get("buckets")
        if not isinstance(buckets, dict):
            return None

        bucket_key = _thread_problem_bucket(problem_size)
        entry = buckets.get(bucket_key)
        if not isinstance(entry, dict) or not entry:
            return None

        observed: list[tuple[float, int, int]] = []
        under_observed: list[int] = []
        for key, stats in entry.items():
            try:
                threads = int(key)
            except Exception:
                continue
            if threads <= 0:
                continue
            if limit is not None and threads > limit:
                continue
            if not isinstance(stats, dict):
                continue
            samples = int(stats.get("samples", 0))
            mean = float(stats.get("mean", 0.0))
            variance = float(stats.get("m2", 0.0))
            if samples < _LEARNING_MIN_OBSERVATIONS:
                under_observed.append(threads)
                continue
            deviation = 0.0
            if samples > 1 and variance > 0.0:
                deviation = math.sqrt(max(0.0, variance / (samples - 1)))
            score = mean + (deviation / max(1, samples))
            observed.append((score, threads, samples))

        if observed:
            observed.sort()
            _, best_threads, _ = observed[0]
            if _LEARNING_EPSILON > 0.0 and len(observed) > 1:
                if random.random() < _LEARNING_EPSILON:
                    _, alt_threads, _ = random.choice(observed[1:])
                    return max(1, alt_threads)
            if _LEARNING_EPSILON > 0.0 and under_observed:
                if random.random() < (_LEARNING_EPSILON * 0.5):
                    return max(1, random.choice(under_observed))
            return max(1, best_threads)

        if under_observed:
            return max(1, min(under_observed))

        return None


def _record_thread_outcome(
    problem_size: tuple[int, int, int],
    threads: int | None,
    duration: float,
) -> None:
    if threads is None or threads <= 0:
        return
    if duration <= 0.0 or any(dim <= 0 for dim in problem_size):
        return

    bucket_key = _thread_problem_bucket(problem_size)
    now = time.time()

    with _LEARNING_LOCK:
        cache = _load_learning_cache()
        buckets = cache.setdefault("buckets", {})
        if not isinstance(buckets, dict):
            buckets = {}
            cache["buckets"] = buckets
        entry = buckets.setdefault(bucket_key, {})
        if not isinstance(entry, dict):
            entry = {}
            buckets[bucket_key] = entry

        stats = entry.get(str(int(threads)))
        if not isinstance(stats, dict):
            stats = {"samples": 0, "mean": 0.0, "m2": 0.0}

        samples = int(stats.get("samples", 0))
        mean = float(stats.get("mean", 0.0))
        m2 = float(stats.get("m2", 0.0))
        if samples >= _LEARNING_MAX_SAMPLES:
            samples = max(1, _LEARNING_MAX_SAMPLES - 1)
            mean *= samples / max(1, samples + 1)
            m2 *= samples / max(1, samples + 1)

        samples += 1
        delta = duration - mean
        mean += delta / samples
        m2 += delta * (duration - mean)

        stats.update(
            {
                "samples": samples,
                "mean": mean,
                "m2": m2,
                "last": now,
            }
        )
        entry[str(int(threads))] = stats

        global _LEARNING_CACHE_DIRTY
        _LEARNING_CACHE_DIRTY = True
        _thread_learning_flush()


def thread_learning_snapshot() -> dict[str, object]:
    """Return a deep copy of the adaptive thread tuning dataset."""

    with _LEARNING_LOCK:
        cache = _load_learning_cache()
        return json.loads(json.dumps(cache))


def _sync_await(value: object) -> object:
    if not asyncio.iscoroutine(value):
        return value
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(value)  # type: ignore[arg-type]
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def _ensure_wgpu_device(
    device: object | None = None, queue: object | None = None
) -> tuple[object, object]:
    if _WGPU_MODULE is None:
        raise RuntimeError("wgpu is not available")

    if device is not None:
        resolved_queue = queue or getattr(device, "queue", None)
        if resolved_queue is None:
            raise RuntimeError("provided WGPU device lacks an associated queue")
        return device, resolved_queue

    if queue is not None:
        resolved_device = getattr(queue, "device", None)
        if resolved_device is None:
            raise RuntimeError("provided WGPU queue does not expose its device")
        return resolved_device, queue

    global _WGPU_DEVICE, _WGPU_QUEUE
    if _WGPU_DEVICE is not None and _WGPU_QUEUE is not None:
        return _WGPU_DEVICE, _WGPU_QUEUE

    if _wgpu_get_default_device is not None:
        try:
            dev = _wgpu_get_default_device()
        except Exception as exc:  # pragma: no cover - adapter failures
            raise RuntimeError(f"failed to acquire default WGPU device: {exc}") from exc
    else:
        adapter = _WGPU_MODULE.request_adapter(power_preference="high-performance")
        adapter = _sync_await(adapter)
        if adapter is None:
            raise RuntimeError("unable to acquire a WGPU adapter")
        dev = adapter.request_device()
        dev = _sync_await(dev)

    queue_obj = getattr(dev, "queue", None)
    if queue_obj is None:
        raise RuntimeError("WGPU device does not expose a queue")

    _WGPU_DEVICE, _WGPU_QUEUE = dev, queue_obj
    return dev, queue_obj


_GPU_DGEMM_SHADER_TEMPLATE = """
struct Matrix {
    data: array<f32>;
};

struct Params {
    dims: vec4<u32>;
    strides: vec4<u32>;
    scalars: vec4<f32>;
};

@group(0) @binding(0) var<storage, read> matrix_a: Matrix;
@group(0) @binding(1) var<storage, read> matrix_b: Matrix;
@group(0) @binding(2) var<storage, read_write> matrix_c: Matrix;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size({wg_size}, {wg_size}, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let m = params.dims.x;
    let n = params.dims.y;
    let k = params.dims.z;
    let flags = params.dims.w;
    let row = global_id.y;
    let col = global_id.x;
    if (row >= m || col >= n) {
        return;
    }

    let lda = params.strides.x;
    let ldb = params.strides.y;
    let ldc = params.strides.z;
    let alpha = params.scalars.x;
    let beta = params.scalars.y;
    let trans_a = (flags & 1u) != 0u;
    let trans_b = (flags & 2u) != 0u;

    var kk: u32 = 0u;
    var acc: f32 = 0.0;
    loop {
        if kk >= k {
            break;
        }
        let a_row = if trans_a { kk } else { row };
        let a_col = if trans_a { row } else { kk };
        let b_row = if trans_b { col } else { kk };
        let b_col = if trans_b { kk } else { col };
        let a_index = a_row * lda + a_col;
        let b_index = b_row * ldb + b_col;
        acc = acc + matrix_a.data[a_index] * matrix_b.data[b_index];
        kk = kk + 1u;
    }

    let c_index = row * ldc + col;
    let scaled = acc * alpha;
    if beta != 0.0 {
        matrix_c.data[c_index] = scaled + matrix_c.data[c_index] * beta;
    } else {
        matrix_c.data[c_index] = scaled;
    }
}
"""


def _wgpu_coerce_float32_buffer(
    data: Iterable[float] | array, expected: int, label: str
) -> array:
    try:
        buf = array("f", data)
    except TypeError:
        buf = array("f", list(data))
    if len(buf) != expected:
        raise ValueError(f"matrix {label} has {len(buf)} elements, expected {expected}")
    return buf


def _ensure_gpu_pipeline(device: object, workgroup_size: int) -> tuple[object, object]:
    global _WGPU_BIND_LAYOUT
    workgroup_size = max(1, min(32, int(workgroup_size)))
    pipeline = _WGPU_PIPELINES.get(workgroup_size)
    if pipeline is not None and _WGPU_BIND_LAYOUT is not None:
        return pipeline, _WGPU_BIND_LAYOUT

    shader_source = _GPU_DGEMM_SHADER_TEMPLATE.format(wg_size=workgroup_size)
    module = device.create_shader_module(code=shader_source)

    if _WGPU_BIND_LAYOUT is None:
        _WGPU_BIND_LAYOUT = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": _WGPU_MODULE.ShaderStage.COMPUTE,
                    "buffer": {
                        "type": _WGPU_MODULE.BufferBindingType.read_only_storage,
                        "has_dynamic_offset": False,
                        "min_binding_size": 0,
                    },
                },
                {
                    "binding": 1,
                    "visibility": _WGPU_MODULE.ShaderStage.COMPUTE,
                    "buffer": {
                        "type": _WGPU_MODULE.BufferBindingType.read_only_storage,
                        "has_dynamic_offset": False,
                        "min_binding_size": 0,
                    },
                },
                {
                    "binding": 2,
                    "visibility": _WGPU_MODULE.ShaderStage.COMPUTE,
                    "buffer": {
                        "type": _WGPU_MODULE.BufferBindingType.storage,
                        "has_dynamic_offset": False,
                        "min_binding_size": 0,
                    },
                },
                {
                    "binding": 3,
                    "visibility": _WGPU_MODULE.ShaderStage.COMPUTE,
                    "buffer": {
                        "type": _WGPU_MODULE.BufferBindingType.uniform,
                        "has_dynamic_offset": False,
                        "min_binding_size": _WGPU_PARAMS_SIZE,
                    },
                },
            ]
        )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[_WGPU_BIND_LAYOUT]
    )
    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": module, "entry_point": "main"},
    )
    _WGPU_PIPELINES[workgroup_size] = pipeline
    return pipeline, _WGPU_BIND_LAYOUT


def _create_buffer_with_data(
    device: object,
    queue: object,
    data: bytes,
    *,
    usage: int,
    label: str | None = None,
) -> object:
    if hasattr(device, "create_buffer_with_data"):
        return device.create_buffer_with_data(data=data, usage=usage, label=label)
    buffer = device.create_buffer(size=len(data), usage=usage, label=label)
    if data:
        queue.write_buffer(buffer, 0, data)
    return buffer


def _readback_gpu_buffer(device: object, queue: object, buffer: object, size: int) -> bytes:
    read_usage = _WGPU_MODULE.BufferUsage.COPY_DST | _WGPU_MODULE.BufferUsage.MAP_READ
    readback = device.create_buffer(size=size, usage=read_usage, label="st.gpu_dgemm.readback")
    encoder = device.create_command_encoder(label="st.gpu_dgemm.readback.encoder")
    encoder.copy_buffer_to_buffer(buffer, 0, readback, 0, size)
    command_buffer = encoder.finish()
    queue.submit([command_buffer])
    try:
        device.poll(True)
    except Exception:
        pass
    map_future = readback.map_async(_WGPU_MODULE.MapMode.READ, 0, size)
    _sync_await(map_future)
    data = bytes(readback.get_mapped_range(0, size))
    readback.unmap()
    return data


def gpu_blas_available() -> bool:
    """Return ``True`` when a WGPU device suitable for GEMM dispatch is available."""

    if _WGPU_MODULE is None:
        return False
    try:
        _ensure_wgpu_device()
    except Exception:
        return False
    return True


def gpu_dgemm(
    m: int,
    n: int,
    k: int,
    a: Iterable[float] | array,
    b: Iterable[float] | array,
    c: array | None = None,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    transpose_a: bool = False,
    transpose_b: bool = False,
    workgroup_size: int = 16,
    device: object | None = None,
    queue: object | None = None,
) -> array:
    """Dispatch a GEMM workload to a WGPU compute device using WGSL kernels."""

    if _WGPU_MODULE is None:
        raise RuntimeError("wgpu is not available")
    if m < 0 or n < 0 or k < 0:
        raise ValueError("matrix dimensions must be non-negative")

    expected_a = m * k
    expected_b = k * n
    expected_c = m * n

    a_buf = _wgpu_coerce_float32_buffer(a, expected_a, "A")
    b_buf = _wgpu_coerce_float32_buffer(b, expected_b, "B")

    if expected_c == 0:
        return array("f")

    if c is None:
        base_c = array("f", [0.0] * expected_c)
        target = None
    else:
        if not isinstance(c, array):
            raise TypeError("matrix C must be an array('d') or array('f')")
        if len(c) != expected_c:
            raise ValueError(f"matrix C has {len(c)} elements, expected {expected_c}")
        if c.typecode not in {"f", "d"}:
            raise TypeError("matrix C must use 'f' or 'd' typecode")
        base_c = array("f", c)
        target = c

    device, queue = _ensure_wgpu_device(device, queue)
    pipeline, bind_layout = _ensure_gpu_pipeline(device, workgroup_size)

    lda = k if not transpose_a else m
    ldb = n if not transpose_b else k
    ldc = n
    flags = (1 if transpose_a else 0) | (2 if transpose_b else 0)

    params = struct.pack(
        "<8I4f",
        int(m),
        int(n),
        int(k),
        int(flags),
        int(lda if lda else 1),
        int(ldb if ldb else 1),
        int(ldc if ldc else 1),
        0,
        float(alpha),
        float(beta),
        0.0,
        0.0,
    )

    usage_readonly = (
        _WGPU_MODULE.BufferUsage.STORAGE
        | _WGPU_MODULE.BufferUsage.COPY_DST
    )
    usage_storage = (
        _WGPU_MODULE.BufferUsage.STORAGE
        | _WGPU_MODULE.BufferUsage.COPY_DST
        | _WGPU_MODULE.BufferUsage.COPY_SRC
    )

    buf_a = _create_buffer_with_data(
        device,
        queue,
        a_buf.tobytes(),
        usage=usage_readonly,
        label="st.gpu_dgemm.A",
    )
    buf_b = _create_buffer_with_data(
        device,
        queue,
        b_buf.tobytes(),
        usage=usage_readonly,
        label="st.gpu_dgemm.B",
    )
    buf_c = _create_buffer_with_data(
        device,
        queue,
        base_c.tobytes(),
        usage=usage_storage,
        label="st.gpu_dgemm.C",
    )
    buf_params = _create_buffer_with_data(
        device,
        queue,
        params,
        usage=_WGPU_MODULE.BufferUsage.UNIFORM | _WGPU_MODULE.BufferUsage.COPY_DST,
        label="st.gpu_dgemm.params",
    )

    bind_group = device.create_bind_group(
        layout=bind_layout,
        entries=[
            {
                "binding": 0,
                "resource": {"buffer": buf_a, "offset": 0, "size": len(a_buf) * 4},
            },
            {
                "binding": 1,
                "resource": {"buffer": buf_b, "offset": 0, "size": len(b_buf) * 4},
            },
            {
                "binding": 2,
                "resource": {"buffer": buf_c, "offset": 0, "size": len(base_c) * 4},
            },
            {
                "binding": 3,
                "resource": {"buffer": buf_params, "offset": 0, "size": _WGPU_PARAMS_SIZE},
            },
        ],
        label="st.gpu_dgemm.bind_group",
    )

    encoder = device.create_command_encoder(label="st.gpu_dgemm.encoder")
    pass_encoder = encoder.begin_compute_pass(label="st.gpu_dgemm.pass")
    pass_encoder.set_pipeline(pipeline)
    pass_encoder.set_bind_group(0, bind_group, [])
    groups_x = (n + workgroup_size - 1) // workgroup_size
    groups_y = (m + workgroup_size - 1) // workgroup_size
    pass_encoder.dispatch_workgroups(groups_x, groups_y, 1)
    pass_encoder.end()

    command_buffer = encoder.finish()
    queue.submit([command_buffer])
    try:
        device.poll(True)
    except Exception:
        pass

    result_bytes = _readback_gpu_buffer(device, queue, buf_c, expected_c * 4)
    result = array("f")
    result.frombytes(result_bytes)

    if target is not None:
        if target.typecode == "d":
            target[:] = array("d", result)
            return target
        target[:] = result
        return target
    return result


def wasm_blas_available() -> bool:
    """Return ``True`` when the pure WebAssembly BLAS fallback can be used."""

    hint = os.environ.get("SPIRALTORCH_WASM_BLAS", "").strip().lower()
    if hint and hint not in {"0", "false", "no", "off"}:
        return True
    return js is not None


def _ensure_wasm_kernel() -> None:
    global _WASM_STATE
    if _WASM_STATE is None:
        _WASM_STATE = {}
    if _WASM_STATE.get("kernel_ready"):
        return
    if js is None or to_js is None:
        _WASM_STATE["kernel_ready"] = False
        return
    try:
        kernel = js.Function(
            "m",
            "n",
            "k",
            "alpha",
            "beta",
            "a",
            "b",
            "c",
            """
            const out = new Float64Array(m * n);
            const left = new Float64Array(a);
            const right = new Float64Array(b);
            const baseC = c ? new Float64Array(c) : null;
            for (let row = 0; row < m; row++) {
                for (let col = 0; col < n; col++) {
                    let acc = 0.0;
                    const rowOffset = row * k;
                    for (let kk = 0; kk < k; kk++) {
                        acc += left[rowOffset + kk] * right[kk * n + col];
                    }
                    const idx = row * n + col;
                    out[idx] = alpha * acc + (baseC ? beta * baseC[idx] : 0.0);
                }
            }
            return out;
            """,
        )
        _WASM_STATE["kernel"] = kernel
        _WASM_STATE["kernel_ready"] = True
    except Exception:
        _WASM_STATE["kernel_ready"] = False


def wasm_dgemm(
    m: int,
    n: int,
    k: int,
    a: Iterable[float] | array,
    b: Iterable[float] | array,
    c: array | None = None,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> array:
    """Compute GEMM using a WebAssembly-friendly fallback implementation."""

    if m < 0 or n < 0 or k < 0:
        raise ValueError("matrix dimensions must be non-negative")

    expected_a = m * k
    expected_b = k * n
    expected_c = m * n

    lhs = array("d", a)
    rhs = array("d", b)

    if len(lhs) != expected_a:
        raise ValueError(f"matrix A has {len(lhs)} elements, expected {expected_a}")
    if len(rhs) != expected_b:
        raise ValueError(f"matrix B has {len(rhs)} elements, expected {expected_b}")

    if expected_c == 0:
        return array("d")

    if c is not None:
        if not isinstance(c, array):
            raise TypeError("matrix C must be an array('d') or array('f')")
        if len(c) != expected_c:
            raise ValueError(f"matrix C has {len(c)} elements, expected {expected_c}")
        base_c = array("d", c)
    else:
        base_c = array("d", [0.0] * expected_c)

    use_js = False
    result_py: array | None = None
    if wasm_blas_available() and js is not None and to_js is not None:
        global _WASM_STATE
        if _WASM_STATE is None:
            _WASM_STATE = {}
        _ensure_wasm_kernel()
        kernel = _WASM_STATE.get("kernel") if _WASM_STATE else None
        if kernel is not None and _WASM_STATE.get("kernel_ready"):
            try:
                lhs_js = to_js(lhs, dtype="float64")
                rhs_js = to_js(rhs, dtype="float64")
                base_js = to_js(base_c, dtype="float64") if beta != 0.0 else None
                result_js = kernel(m, n, k, float(alpha), float(beta), lhs_js, rhs_js, base_js)
                result_py = array("d", result_js.to_py())
                use_js = True
            except Exception:
                result_py = None
                use_js = False
        else:
            result_py = None
    else:
        result_py = None

    if not use_js:
        result_py = array("d", [0.0] * expected_c)
        for row in range(m):
            for col in range(n):
                acc = 0.0
                base = row * k
                for kk in range(k):
                    acc += lhs[base + kk] * rhs[kk * n + col]
                idx = row * n + col
                value = alpha * acc
                if beta != 0.0:
                    value += beta * base_c[idx]
                result_py[idx] = value

    if result_py is None:
        raise RuntimeError("wasm_dgemm failed to compute a result")

    if c is None:
        return result_py

    if c.typecode == "f":
        c[:] = array("f", result_py)
    elif c.typecode == "d":
        c[:] = result_py
    else:
        raise TypeError("matrix C must use 'f' or 'd' typecode")
    return c


def _coerce_double_array(
    data: Iterable[float] | array, expected: int, label: str
) -> array:
    try:
        buf = array("d", data)
    except TypeError:
        buf = array("d", list(data))
    if len(buf) != expected:
        raise ValueError(f"matrix {label} has {len(buf)} elements, expected {expected}")
    return buf


def available_dgemm_backends() -> list[str]:
    """Return an ordered list of the GEMM backends that are ready to use."""

    backends: list[str] = []
    try:
        if blas_available():
            backends.append("cpu")
    except Exception:
        pass

    try:
        if gpu_blas_available():
            backends.append("gpu")
    except Exception:
        pass

    try:
        if wasm_blas_available():
            backends.append("wasm")
    except Exception:
        pass

    return backends


def _backend_preference_order(preference: str) -> tuple[str, ...]:
    base = ("cpu", "gpu", "wasm")
    if preference == "gpu":
        return ("gpu", "cpu", "wasm")
    if preference == "wasm":
        return ("wasm", "cpu", "gpu")
    if preference == "cpu":
        return base
    if preference == "none":
        return tuple()
    return base


def _backend_is_available(name: str) -> bool:
    if name == "cpu":
        return blas_available()
    if name == "gpu":
        return gpu_blas_available()
    if name == "wasm":
        return wasm_blas_available()
    return False


def select_dgemm_backend(preference: str | None = None) -> str:
    """Choose a GEMM backend honouring configuration hints and availability."""

    env_preference = os.environ.get("SPIRALTORCH_DGEMM_BACKEND", "").strip().lower()
    hint = (preference or "").strip().lower() or env_preference or "auto"

    if hint in {"auto", ""}:
        order = ("cpu", "gpu", "wasm")
    else:
        order = _backend_preference_order(hint)
        if not order:
            raise RuntimeError("no GEMM backend preference available")

    fallback_warning = False
    for backend in order:
        try:
            if _backend_is_available(backend):
                if fallback_warning:
                    warnings.warn(
                        "Falling back to GEMM backend '%s'" % backend,
                        RuntimeWarning,
                    )
                return backend
        except Exception:
            fallback_warning = True
            continue

        if hint not in {"auto", ""} and backend == order[0]:
            fallback_warning = True

    raise RuntimeError("no GEMM backend is currently available")


def _ensure_target_buffer(
    c: array | None, expected: int
) -> tuple[array, array | None]:
    if c is None:
        return array("d", [0.0] * expected), None
    if not isinstance(c, array) or c.typecode != "d":
        raise TypeError("matrix C must be an array('d') buffer")
    if len(c) != expected:
        raise ValueError(f"matrix C has {len(c)} elements, expected {expected}")
    return c, c


def auto_dgemm(
    m: int,
    n: int,
    k: int,
    a: Iterable[float] | array,
    b: Iterable[float] | array,
    c: array | None = None,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    backend: str | None = None,
) -> array:
    """Compute GEMM using the best available backend with a uniform interface."""

    if m < 0 or n < 0 or k < 0:
        raise ValueError("matrix dimensions must be non-negative")

    expected_a = m * k
    expected_b = k * n
    expected_c = m * n

    lhs = _coerce_double_array(a, expected_a, "A")
    rhs = _coerce_double_array(b, expected_b, "B")

    if expected_c == 0:
        return array("d")

    target, original = _ensure_target_buffer(c, expected_c)

    backend_name = select_dgemm_backend(backend)
    if backend_name == "cpu":
        dgemm(m, n, k, lhs, rhs, target, alpha=alpha, beta=beta)
        return target

    if backend_name == "gpu":
        gpu_result = gpu_dgemm(
            m,
            n,
            k,
            lhs,
            rhs,
            target if original is None else array("f", target),
            alpha=alpha,
            beta=beta,
        )
        if original is None:
            if gpu_result.typecode == "d":
                return gpu_result
            return array("d", gpu_result)
        if gpu_result.typecode == "f":
            original[:] = array("d", gpu_result)
            return original
        return gpu_result

    if backend_name == "wasm":
        wasm_result = wasm_dgemm(
            m,
            n,
            k,
            lhs,
            rhs,
            target if original is not None else None,
            alpha=alpha,
            beta=beta,
        )
        if original is None:
            return wasm_result
        original[:] = wasm_result
        return original

    raise RuntimeError(f"unsupported GEMM backend '{backend_name}'")


def wgpu_adapter_info() -> dict[str, object] | None:
    """Return metadata describing the detected WGPU adapter, if any."""

    info = _detect_wgpu_adapter()
    if info is None:
        return None
    return dict(info)


def _read_text_file(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except (FileNotFoundError, OSError):
        return None


def _cpu_topology_from_sysfs(affinity_filter: set[int] | None) -> dict[str, object] | None:
    base = "/sys/devices/system/cpu"
    try:
        entries = os.listdir(base)
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
        return None

    logical_ids: list[int] = []
    cores: dict[tuple[int, int], int] = {}
    packages: set[int] = set()

    for name in entries:
        if not name.startswith("cpu"):
            continue
        suffix = name[3:]
        if not suffix.isdigit():
            continue
        cpu_id = int(suffix)
        if affinity_filter is not None and cpu_id not in affinity_filter:
            continue

        topology_dir = os.path.join(base, name, "topology")
        core_id_text = _read_text_file(os.path.join(topology_dir, "core_id"))
        package_id_text = _read_text_file(os.path.join(topology_dir, "physical_package_id"))

        try:
            core_id = int(core_id_text) if core_id_text is not None else None
        except ValueError:
            core_id = None
        try:
            package_id = int(package_id_text) if package_id_text is not None else None
        except ValueError:
            package_id = None

        key: tuple[int, int] | None = None
        if core_id is not None and package_id is not None:
            key = (package_id, core_id)
        elif core_id is not None:
            key = (-1, core_id)
        elif package_id is not None:
            key = (package_id, -1)

        logical_ids.append(cpu_id)
        if package_id is not None:
            packages.add(package_id)
        if key is not None:
            cores[key] = cores.get(key, 0) + 1

    if not logical_ids:
        return None

    physical_cores = len(cores) if cores else None
    packages_count = len(packages) if packages else None
    threads_per_core = max(cores.values()) if cores else None

    return {
        "logical_ids": logical_ids,
        "physical_cores": physical_cores,
        "packages": packages_count,
        "threads_per_core": threads_per_core,
    }


def _self_cgroup_paths(*, refresh: bool = False) -> dict[str, str]:
    global _CGROUP_SELF_CACHE
    if _CGROUP_SELF_CACHE is not None and not refresh:
        return dict(_CGROUP_SELF_CACHE)

    mapping: dict[str, str] = {}
    try:
        with open("/proc/self/cgroup", "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split(":", 2)
                if len(parts) != 3:
                    continue
                _, controllers, path = parts
                mapping[controllers] = path or "/"
    except (FileNotFoundError, OSError):
        mapping = {}

    _CGROUP_SELF_CACHE = dict(mapping)
    return mapping


def _resolve_cgroup_path(base: str, relative: str) -> str:
    rel = relative.lstrip("/")
    if not rel:
        return base
    return os.path.join(base, rel)


def _parse_cpu_quota(quota: str, period: str | None) -> float | None:
    try:
        quota_value = int(quota)
    except (TypeError, ValueError):
        return None
    if quota_value <= 0:
        return None

    if period is None:
        period_value = 100000
    else:
        try:
            period_value = int(period)
        except (TypeError, ValueError):
            return None
        if period_value <= 0:
            return None

    return quota_value / period_value


def _cgroup_v2_cpu_quota(paths: dict[str, str]) -> float | None:
    relative = paths.get("")
    if relative is None:
        return None

    base = _resolve_cgroup_path("/sys/fs/cgroup", relative)
    data = _read_text_file(os.path.join(base, "cpu.max"))
    if not data:
        return None

    parts = data.split()
    if not parts or parts[0] == "max":
        return None

    period = parts[1] if len(parts) > 1 else None
    return _parse_cpu_quota(parts[0], period)


_CGROUP_V1_CPU_BASES: tuple[str, ...] = (
    "/sys/fs/cgroup/cpu",
    "/sys/fs/cgroup/cpuacct",
    "/sys/fs/cgroup/cpu,cpuacct",
    "/sys/fs/cgroup/cpuacct,cpu",
)


def _cgroup_v1_cpu_quota(paths: dict[str, str]) -> float | None:
    candidates = []
    for controllers, relative in paths.items():
        if not controllers:
            continue
        controller_set = {token.strip() for token in controllers.split(",") if token.strip()}
        if "cpu" not in controller_set:
            continue
        for base in _CGROUP_V1_CPU_BASES:
            full_base = _resolve_cgroup_path(base, relative)
            if os.path.exists(full_base):
                candidates.append(full_base)
                break
        else:
            for base in _CGROUP_V1_CPU_BASES:
                if os.path.exists(base):
                    candidates.append(base)
                    break

    for candidate in candidates:
        quota_path = os.path.join(candidate, "cpu.cfs_quota_us")
        period_path = os.path.join(candidate, "cpu.cfs_period_us")
        quota_raw = _read_text_file(quota_path)
        period_raw = _read_text_file(period_path)
        if quota_raw is None or period_raw is None:
            continue
        quota_raw = quota_raw.strip()
        if quota_raw == "-1":
            continue
        quota = _parse_cpu_quota(quota_raw, period_raw.strip())
        if quota is not None:
            return quota
    return None


def _cgroup_cpu_quota_raw(*, refresh: bool = False) -> float | None:
    global _CGROUP_QUOTA_CACHE
    if not refresh and _CGROUP_QUOTA_CACHE is not None:
        cached_pid, cached = _CGROUP_QUOTA_CACHE
        if cached_pid == os.getpid():
            return cached

    paths = _self_cgroup_paths(refresh=refresh)
    quota = _cgroup_v2_cpu_quota(paths)
    if quota is None:
        quota = _cgroup_v1_cpu_quota(paths)

    _CGROUP_QUOTA_CACHE = (os.getpid(), quota)  # PID ensures cache scoped to process
    return quota


def _expand_cpu_list(spec: str) -> tuple[int, ...] | None:
    include: set[int] = set()
    exclude: set[int] = set()

    tokens = spec.replace("\n", ",").replace("\t", ",").split(",")
    for raw_segment in tokens:
        for token in raw_segment.split():
            item = token.strip()
            if not item:
                continue
            if ":" in item:
                item = item.rsplit(":", 1)[1].strip()
                if not item:
                    continue
            negate = item.startswith("^")
            if negate:
                item = item[1:].strip()
                if not item:
                    continue
            stride = 1
            if "/" in item:
                item, stride_str = item.split("/", 1)
                try:
                    stride = int(stride_str, 10)
                except ValueError:
                    return None
                if stride <= 0:
                    return None
            if "-" in item:
                start_str, end_str = item.split("-", 1)
            else:
                start_str = end_str = item
            try:
                start = int(start_str, 10)
                end = int(end_str, 10)
            except ValueError:
                return None
            if end < start:
                return None
            values = range(start, end + 1, stride)
            if negate:
                exclude.update(values)
            else:
                include.update(values)

    if not include:
        if exclude:
            include = set()
        else:
            return None

    include.difference_update(exclude)
    return tuple(sorted(include))


def _parse_cpuset_list(spec: str) -> int | None:
    expanded = _expand_cpu_list(spec)
    if expanded is None:
        return None
    return len(expanded)


def _read_cpu_list_file(path: str) -> tuple[int, ...] | None:
    raw = _read_text_file(path)
    if raw is None:
        return None
    expanded = _expand_cpu_list(raw)
    if expanded is None:
        return None
    return expanded


_CGROUP_V1_CPUSET_BASES: tuple[str, ...] = (
    "/sys/fs/cgroup/cpuset",
    "/sys/fs/cgroup/cpuset,cpu",
)


def _cgroup_cpuset_count(paths: dict[str, str], *, refresh: bool = False) -> int | None:
    global _CGROUP_CPUSET_CACHE
    if not refresh and _CGROUP_CPUSET_CACHE is not None:
        cached_pid, cached = _CGROUP_CPUSET_CACHE
        if cached_pid == os.getpid():
            return cached

    # cgroup v2 exposes cpuset files alongside cpu.max
    relative_v2 = paths.get("")
    cpuset_paths: list[str] = []
    if relative_v2 is not None:
        base = _resolve_cgroup_path("/sys/fs/cgroup", relative_v2)
        for name in ("cpuset.cpus.effective", "cpuset.cpus"):
            candidate = os.path.join(base, name)
            if os.path.exists(candidate):
                cpuset_paths.append(candidate)

    # cgroup v1 cpuset controller
    for controllers, relative in paths.items():
        controller_set = {token.strip() for token in controllers.split(",") if token.strip()}
        if "cpuset" not in controller_set:
            continue
        for base in _CGROUP_V1_CPUSET_BASES:
            full_base = _resolve_cgroup_path(base, relative)
            if os.path.exists(full_base):
                for name in ("cpuset.cpus.effective", "cpuset.cpus"):
                    candidate = os.path.join(full_base, name)
                    if os.path.exists(candidate):
                        cpuset_paths.append(candidate)
                break

    for path in cpuset_paths:
        data = _read_text_file(path)
        if not data:
            continue
        count = _parse_cpuset_list(data)
        if count is not None:
            _CGROUP_CPUSET_CACHE = (os.getpid(), count)
            return count

    _CGROUP_CPUSET_CACHE = (os.getpid(), None)
    return None


_CPU_ISOLATION_PATHS: tuple[tuple[str, str], ...] = (
    ("isolated", "/sys/devices/system/cpu/isolated"),
    ("nohz_full", "/sys/devices/system/cpu/nohz_full"),
    ("rcu_nocbs", "/sys/devices/system/cpu/rcu_nocbs"),
    ("managed_irq", "/sys/devices/system/cpu/managed_irq"),
)

_CPU_ISOLATION_BASIS: tuple[str, ...] = (
    "/sys/devices/system/cpu/online",
    "/sys/devices/system/cpu/possible",
)


def _cpu_isolation_snapshot() -> dict[str, object] | None:
    snapshot: dict[str, object] = {}
    reserved_sets: list[set[int]] = []

    for key, path in _CPU_ISOLATION_PATHS:
        cpus = _read_cpu_list_file(path)
        if cpus is None:
            continue
        snapshot[key] = cpus
        if cpus:
            reserved_sets.append(set(cpus))

    basis: tuple[int, ...] | None = None
    for candidate in _CPU_ISOLATION_BASIS:
        basis = _read_cpu_list_file(candidate)
        if basis is not None:
            snapshot["available"] = basis
            break

    if reserved_sets:
        reserved_union: set[int] = set()
        for group in reserved_sets:
            reserved_union.update(group)
        snapshot["reserved"] = tuple(sorted(reserved_union))
        snapshot["reserved_count"] = len(reserved_union)
    elif snapshot:
        snapshot["reserved_count"] = 0

    if basis is not None:
        available_set = set(basis)
        for group in reserved_sets:
            available_set.difference_update(group)
        effective = tuple(sorted(available_set))
        snapshot["effective"] = effective
        snapshot["effective_count"] = len(effective)
    elif snapshot:
        snapshot["effective_count"] = None

    if not snapshot:
        return None

    return snapshot


def system_cpu_isolations(*, refresh: bool = False) -> dict[str, object] | None:
    """Return kernel-level CPU isolation metadata when available.

    The returned mapping contains any detected isolation sets such as
    ``"isolated"``, ``"nohz_full"``, ``"rcu_nocbs"`` and ``"managed_irq"``
    alongside the CPUs considered generally available (``"available"``), the
    union of reserved CPUs (``"reserved"``/``"reserved_count"``) and the number
    of CPUs left after respecting those reservations
    (``"effective"``/``"effective_count"``).  ``None`` is returned when the host
    does not expose any isolation metadata.
    """

    global _CPU_ISOLATION_CACHE

    if (
        not refresh
        and _CPU_ISOLATION_CACHE is not None
        and _CPU_ISOLATION_CACHE[0] == os.getpid()
    ):
        cached = _CPU_ISOLATION_CACHE[1]
        if cached is None:
            return None
        return {
            key: (tuple(value) if isinstance(value, tuple) else value)
            for key, value in cached.items()
        }

    snapshot = _cpu_isolation_snapshot()
    if snapshot is None:
        _CPU_ISOLATION_CACHE = (os.getpid(), None)
        return None

    structured = {
        key: (tuple(value) if isinstance(value, tuple) else value)
        for key, value in snapshot.items()
    }
    _CPU_ISOLATION_CACHE = (os.getpid(), structured)
    return {key: value for key, value in structured.items()}


def cgroup_cpu_quota(*, refresh: bool = False) -> float | None:
    """Return the CPU quota imposed by control groups, if any.

    The returned value represents the number of CPU cores available to the
    process according to cgroup ``cpu.max`` or ``cpu.cfs_*`` limits.  A ``None``
    result indicates that no quota could be detected or the quota is unlimited.
    """

    return _cgroup_cpu_quota_raw(refresh=refresh)


def _parse_pressure_line(line: str) -> tuple[str, dict[str, float | int]] | None:
    tokens = line.strip().split()
    if not tokens:
        return None

    kind = tokens[0].strip()
    if not kind:
        return None

    metrics: dict[str, float | int] = {}
    for token in tokens[1:]:
        if "=" not in token:
            continue
        key, raw = token.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if not key or not raw:
            continue
        if key == "total":
            value = _coerce_positive_int(raw)
            if value is None:
                continue
            metrics[key] = value
            continue
        value_f = _coerce_non_negative_float(raw)
        if value_f is None:
            continue
        metrics[key] = value_f

    if not metrics:
        return None

    return kind, metrics


def _cpu_pressure_snapshot() -> dict[str, dict[str, float | int]] | None:
    try:
        with open("/proc/pressure/cpu", "r", encoding="utf-8") as handle:
            snapshot: dict[str, dict[str, float | int]] = {}
            for line in handle:
                parsed = _parse_pressure_line(line)
                if parsed is None:
                    continue
                kind, metrics = parsed
                if kind not in {"some", "full"}:
                    continue
                snapshot[kind] = dict(metrics)
    except (FileNotFoundError, OSError):
        return None

    return snapshot or None


def system_cpu_pressure(*, refresh: bool = False) -> dict[str, dict[str, float | int]] | None:
    """Return Linux Pressure Stall Information (PSI) metrics for the host.

    The returned mapping exposes the ``"some"`` and ``"full"`` CPU pressure
    buckets when available, including their ``avg10``/``avg60``/``avg300``
    windows as floating-point percentages alongside the accumulated ``total``
    stall time in microseconds.  ``None`` is returned when PSI support is not
    available on the running kernel.
    """

    global _CPU_PRESSURE_CACHE

    now = time.time()
    if (
        not refresh
        and _CPU_PRESSURE_CACHE is not None
        and _CPU_PRESSURE_CACHE[0] == os.getpid()
        and (now - _CPU_PRESSURE_CACHE[1]) <= _CPU_PRESSURE_CACHE_TTL
    ):
        cached = _CPU_PRESSURE_CACHE[2]
        if cached is None:
            return None
        return {bucket: dict(metrics) for bucket, metrics in cached.items()}

    snapshot = _cpu_pressure_snapshot()
    if snapshot is None:
        _CPU_PRESSURE_CACHE = (os.getpid(), now, None)
        return None

    structured = {bucket: dict(metrics) for bucket, metrics in snapshot.items()}
    _CPU_PRESSURE_CACHE = (os.getpid(), now, structured)
    return {bucket: dict(metrics) for bucket, metrics in structured.items()}


def _system_affinity_size() -> int | None:
    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        return None
    if not affinity:
        return None
    return len(affinity)


def process_cpu_budget(*, refresh: bool = False) -> dict[str, object]:
    """Return a snapshot of CPU availability signals for the current process.

    The returned dictionary contains the raw system CPU count (``"system"``),
    the size of the POSIX affinity mask (``"affinity"``), any detected cgroup
    quota expressed both as a float (``"cgroup_quota"``) and clamped integer
    limit (``"cgroup_quota_limit"``), the number of CPUs exposed through cgroup
    cpuset files (``"cgroup_cpuset"``) and the final minimum of the available
    signals (``"effective"``).  When kernel isolation controls such as
    ``isolcpus`` or ``nohz_full`` are detected their reserved CPUs are reported
    via ``"isolated_cpus"`` while ``"isolated_effective"`` reflects the
    remaining CPUs after honouring those reservations.
    """

    global _CPU_BUDGET_CACHE
    if _CPU_BUDGET_CACHE is not None and not refresh:
        return dict(_CPU_BUDGET_CACHE)

    affinity = _system_affinity_size()
    system_total = os.cpu_count()
    system_total = system_total if isinstance(system_total, int) and system_total > 0 else None

    paths = _self_cgroup_paths(refresh=refresh)
    quota = _cgroup_cpu_quota_raw(refresh=refresh)
    quota_limit: int | None
    if quota is None:
        quota_limit = None
    else:
        quota_limit = max(1, int(math.floor(quota))) if quota >= 1 else 1

    cpuset = _cgroup_cpuset_count(paths, refresh=refresh)

    candidates = [value for value in (affinity, quota_limit, cpuset, system_total) if value]
    effective: int | None = None
    if candidates:
        effective = max(1, min(candidates))

    isolation = system_cpu_isolations(refresh=refresh)
    isolated_effective: int | None = None
    isolated_reserved: int | None = None
    if isolation is not None:
        reserved = isolation.get("reserved_count")
        if isinstance(reserved, int):
            isolated_reserved = max(0, reserved)
        isolation_effective = isolation.get("effective_count")
        if isinstance(isolation_effective, int):
            isolated_effective = max(0, isolation_effective)
            adjusted = max(1, isolated_effective) if isolated_effective > 0 else 1
            if effective is None:
                effective = adjusted
            else:
                effective = max(1, min(effective, adjusted))

    budget = {
        "system": system_total,
        "affinity": affinity,
        "cgroup_quota": quota,
        "cgroup_quota_limit": quota_limit,
        "cgroup_cpuset": cpuset,
        "effective": effective,
        "isolated_cpus": isolated_reserved,
        "isolated_effective": isolated_effective if isolated_effective is not None else None,
    }

    _CPU_BUDGET_CACHE = dict(budget)
    return budget


def system_cpu_count() -> int | None:
    """Return the best-effort number of CPUs available to the process."""

    budget = process_cpu_budget()
    effective = budget.get("effective")
    return int(effective) if isinstance(effective, int) else None


def system_cpu_topology(*, refresh: bool = False) -> dict[str, object]:
    """Return detected CPU topology details scoped to the current process.

    The returned dictionary exposes the logical CPU count seen by the system
    (``"logical_cpus"``), the number of logical CPUs currently available via the
    affinity mask (``"available_logical_cpus"``) and the effective limit that
    considers affinity, cpusets and cgroup quotas (``"effective_logical_cpus"``).
    When the host exposes topology metadata through ``/sys/devices/system/cpu``
    additional keys describe the number of physical cores (``"physical_cores"``),
    CPU packages (``"packages"``) and threads per core
    (``"threads_per_core"``).  A boolean ``"smt_active"`` key indicates whether
    symmetric multithreading appears to be enabled for the accessible CPUs.  If
    kernel isolation controls reserve CPUs the derived counts are also surfaced
    via ``"isolated_effective_cpus"`` and ``"isolated_cpu_count"``.
    """

    global _CPU_TOPOLOGY_CACHE

    try:
        affinity_members = set(os.sched_getaffinity(0))  # type: ignore[arg-type]
    except (AttributeError, OSError):
        affinity_members = None

    signature: tuple[int, ...] | None
    if affinity_members is None:
        signature = None
    else:
        signature = tuple(sorted(affinity_members))

    if (
        not refresh
        and _CPU_TOPOLOGY_CACHE is not None
        and _CPU_TOPOLOGY_CACHE[0] == os.getpid()
        and _CPU_TOPOLOGY_CACHE[1] == signature
    ):
        return dict(_CPU_TOPOLOGY_CACHE[2])

    budget = process_cpu_budget(refresh=refresh)
    logical_total = budget.get("system")
    affinity_total = budget.get("affinity")
    effective = budget.get("effective")
    isolated_effective = budget.get("isolated_effective")
    isolated_reserved = budget.get("isolated_cpus")

    topology = _cpu_topology_from_sysfs(affinity_members)
    physical_cores = topology.get("physical_cores") if topology else None
    packages = topology.get("packages") if topology else None
    threads_per_core = topology.get("threads_per_core") if topology else None

    result = {
        "logical_cpus": logical_total if isinstance(logical_total, int) else None,
        "available_logical_cpus": affinity_total if isinstance(affinity_total, int) else None,
        "effective_logical_cpus": effective if isinstance(effective, int) else None,
        "physical_cores": physical_cores if isinstance(physical_cores, int) else None,
        "packages": packages if isinstance(packages, int) else None,
        "threads_per_core": threads_per_core if isinstance(threads_per_core, int) else None,
        "smt_active": bool(threads_per_core and isinstance(threads_per_core, int) and threads_per_core > 1),
    }

    if isinstance(isolated_effective, int):
        result["isolated_effective_cpus"] = max(0, isolated_effective)
    if isinstance(isolated_reserved, int):
        result["isolated_cpu_count"] = max(0, isolated_reserved)

    _CPU_TOPOLOGY_CACHE = (os.getpid(), signature, dict(result))
    return result


def recommended_thread_count(
    problem_size: tuple[int, int, int] | None = None,
    *,
    clamp_to_env: bool = True,
    consider_gpu: bool = True,
) -> int | None:
    """Return a recommended thread count based on the system and workload.

    Parameters
    ----------
    problem_size:
        Optional ``(m, n, k)`` triple describing the GEMM dimensions.  When
        provided we balance the recommendation against the matrix sizes so small
        problems do not consume the entire machine.
    clamp_to_env:
        When ``True`` (the default) the result honours stricter environment
        hints such as ``OMP_NUM_THREADS`` if they are already configured.
    consider_gpu:
        When ``True`` (the default) host thread reservations derived from the
        active WGPU adapter are respected so CPU BLAS workloads leave breathing
        room for GPU command submission.  The heuristic also inspects Linux PSI
        CPU pressure metrics to back off when the host is already congested.
    """

    system_threads = system_cpu_count()

    env_limit: int | None = None
    if clamp_to_env:
        hints = [_read_int_env(var) for var in _THREAD_HINT_SOURCES]
        hints = [hint for hint in hints if hint is not None and hint > 0]
        if hints:
            env_limit = min(hints)

    if system_threads is None and env_limit is None:
        return None

    available = system_threads if system_threads is not None else env_limit
    if env_limit is not None:
        available = min(available, env_limit) if available is not None else env_limit

    if available is None:
        return None

    if consider_gpu:
        reservation = gpu_host_thread_reservation(available, refresh=True)
        if reservation is not None and available > 0:
            deductible = available - 1 if available > 1 else 0
            if deductible > 0:
                available = max(1, available - min(reservation, deductible))

        topology = system_cpu_topology()
        physical = topology.get("physical_cores") if topology else None
        threads_per_core = topology.get("threads_per_core") if topology else None
        if (
            isinstance(physical, int)
            and physical > 0
            and isinstance(threads_per_core, int)
            and threads_per_core > 1
            and reservation is not None
            and reservation > 0
        ):
            available = max(1, min(available, physical))

    pressure = system_cpu_pressure()
    if pressure and available and available > 1:
        peak_avg = 0.0
        sustained_avg = 0.0
        for bucket in ("some", "full"):
            metrics = pressure.get(bucket)
            if not isinstance(metrics, dict):
                continue
            peak = _coerce_non_negative_float(metrics.get("avg10"))
            if peak is not None:
                peak_avg = max(peak_avg, peak)
            sustained = _coerce_non_negative_float(metrics.get("avg60"))
            if sustained is not None:
                sustained_avg = max(sustained_avg, sustained)

        if peak_avg >= 10.0:
            if peak_avg >= 50.0:
                reduction_ratio = 0.5
            elif peak_avg >= 20.0:
                reduction_ratio = 0.3
            else:
                reduction_ratio = 0.15
            if sustained_avg >= 20.0:
                reduction_ratio = min(0.6, reduction_ratio + 0.1)
            scaled = int(max(1, math.floor(available * (1 - reduction_ratio))))
            if scaled < available:
                available = scaled

    available = max(1, int(available))

    if problem_size:
        learned = _choose_learned_threads(
            problem_size, available if clamp_to_env else None
        )
        if learned is not None:
            return max(1, min(available, learned))

    if not problem_size:
        return available

    m, n, k = (max(0, int(d)) for d in problem_size)
    if m == 0 or n == 0 or k == 0:
        return 1

    # The heuristic balances against the number of independent tiles implied by
    # the operands: we avoid spawning more threads than there are rows/columns to
    # process while still scaling for tall or wide matrices.  The square root of
    # the total work keeps cubic workloads from immediately consuming the entire
    # CPU budget for tiny matrices.
    work_estimate = max(1, m * n * k)
    tile_capacity = max(m, n, k)
    dynamic_budget = max(1, int(math.sqrt(work_estimate)) // max(1, min(m, n, k)))
    proposed = max(1, min(tile_capacity, available, dynamic_budget or 1))
    return max(1, min(available, proposed))


def _determine_thread_hint() -> tuple[int | None, str | None]:
    for variable in _THREAD_HINT_SOURCES:
        hint = _read_int_env(variable)
        if hint and hint > 0:
            return hint, variable

    policy = os.environ.get("SPIRALTORCH_BLAS_AUTOTUNE", "").strip().lower()
    if policy in {"1", "true", "yes", "auto"}:
        recommended = recommended_thread_count()
        if recommended:
            return recommended, "system_cpu"
    return None, None


def _propagate_thread_hint(threads: int) -> None:
    for variable in _THREAD_HINT_SOURCES[2:]:
        os.environ[variable] = str(threads)


def synchronise_thread_hints(*, propagate: bool = False) -> int | None:
    """Apply the strongest thread hint and optionally mirror it to the env.

    The function inspects common environment variables (``OMP_NUM_THREADS`` and
    friends) to determine how many threads the process *should* run with.  When
    a usable hint exists the BLAS backend is configured accordingly and the last
    set thread count is returned.  When ``propagate`` is ``True`` the resolved
    value is written back to the ecosystem variables so future subprocesses and
    OpenMP runtimes see a consistent value.
    """

    hint, source = _determine_thread_hint()
    if not hint:
        return None

    try:
        configure_threads(hint)
    except (RuntimeError, ValueError) as exc:
        warnings.warn(
            f"Unable to honour thread hint from {source or 'environment'}: {exc}",
            RuntimeWarning,
        )
        return None

    if propagate:
        _propagate_thread_hint(hint)
        _propagate_gpu_host_hint(
            gpu_host_thread_reservation(hint, refresh=True)
        )
    return hint


def auto_tune_threads(
    problem_size: tuple[int, int, int] | None = None,
    *,
    clamp_to_env: bool = True,
    propagate: bool = False,
) -> int | None:
    """Automatically configure BLAS threads according to the workload.

    The routine combines :func:`recommended_thread_count` with
    :func:`configure_threads` and mirrors the result to environment hints when
    requested.  It returns the configured thread count or ``None`` when BLAS is
    unavailable or the heuristics could not determine an appropriate value.
    """

    try:
        recommendation = recommended_thread_count(problem_size, clamp_to_env=clamp_to_env)
    except Exception:
        recommendation = None

    if not recommendation:
        return synchronise_thread_hints(propagate=propagate)

    try:
        configure_threads(recommendation)
    except (RuntimeError, ValueError) as exc:
        warnings.warn(
            f"Unable to apply recommended BLAS thread count {recommendation}: {exc}",
            RuntimeWarning,
        )
        return None

    if propagate:
        _propagate_thread_hint(recommendation)
        _propagate_gpu_host_hint(
            gpu_host_thread_reservation(recommendation, refresh=True)
        )
    return recommendation


def _configure_threads_from_env() -> None:
    if synchronise_thread_hints() is not None:
        return

    policy = os.environ.get("SPIRALTORCH_BLAS_AUTOTUNE", "").strip().lower()
    if policy in {"1", "true", "yes", "auto"}:
        auto_tune_threads()


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

    start = time.perf_counter()
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

    duration = time.perf_counter() - start
    try:
        threads = current_thread_count()
    except Exception:
        threads = _THREAD_LAST_SET
    _record_thread_outcome((m, n, k), threads, duration)
