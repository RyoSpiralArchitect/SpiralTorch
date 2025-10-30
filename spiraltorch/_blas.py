"""Minimal BLAS wrapper used by the SpiralTorch stub runtime.

The helpers in this module have gradually expanded beyond merely loading a
``dgemm`` symbol.  The thread-tuning primitives we expose are now aware of the
wider environment so callers can coordinate BLAS parallelism with the rest of
the system.  Beyond the CPU the routines now integrate with the WGPU autotuning
artefacts emitted by the Rust runtime, allowing SpiralTorch to reserve host
threads for GPU command submission so compute on both sides can progress
without contending for the same CPU cores.  The system-facing helpers also
interrogate Linux control groups so container limits and CPU reservations are
captured before tuning decisions are made.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import math
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
    "system_cpu_count",
    "recommended_thread_count",
    "synchronise_thread_hints",
    "auto_tune_threads",
    "blas_vendor",
    "wgpu_adapter_info",
    "gpu_host_thread_reservation",
    "process_cpu_budget",
    "cgroup_cpu_quota",
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
_GPU_HOST_RESERVATION: int | None = None
_GPU_HOST_RESERVATION_BUDGET: int | None = None

_CPU_BUDGET_CACHE: dict[str, object] | None = None

_CGROUP_SELF_CACHE: dict[str, str] | None = None
_CGROUP_QUOTA_CACHE: tuple[int, float | None] | None = None
_CGROUP_CPUSET_CACHE: tuple[int, int | None] | None = None

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


def _compute_gpu_host_reservation(cpu_budget: int | None) -> int | None:
    explicit = _read_int_env(_GPU_HOST_HINT_VAR)
    if explicit is not None:
        return max(0, explicit)

    info = _detect_wgpu_adapter()
    if not info:
        return None

    classification = str(info.get("classification", "unknown"))
    if classification == "software":
        return None
    if classification == "integrated":
        if cpu_budget is None or cpu_budget <= 2:
            return 1
        return min(2, max(1, cpu_budget // 4))
    if classification == "discrete":
        if cpu_budget is not None and cpu_budget >= 8:
            return 2
        return 1
    if classification == "unknown":
        if cpu_budget is None or cpu_budget <= 1:
            return None
        return 1
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


def _parse_cpuset_list(spec: str) -> int | None:
    total = 0
    for segment in spec.split(","):
        segment = segment.strip()
        if not segment:
            continue
        if "-" in segment:
            start_str, end_str = segment.split("-", 1)
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                return None
            if end < start:
                return None
            total += end - start + 1
        else:
            try:
                int(segment)
            except ValueError:
                return None
            total += 1
    if total <= 0:
        return None
    return total


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


def cgroup_cpu_quota(*, refresh: bool = False) -> float | None:
    """Return the CPU quota imposed by control groups, if any.

    The returned value represents the number of CPU cores available to the
    process according to cgroup ``cpu.max`` or ``cpu.cfs_*`` limits.  A ``None``
    result indicates that no quota could be detected or the quota is unlimited.
    """

    return _cgroup_cpu_quota_raw(refresh=refresh)


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
    signals (``"effective"``).
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

    budget = {
        "system": system_total,
        "affinity": affinity,
        "cgroup_quota": quota,
        "cgroup_quota_limit": quota_limit,
        "cgroup_cpuset": cpuset,
        "effective": effective,
    }

    _CPU_BUDGET_CACHE = dict(budget)
    return budget


def system_cpu_count() -> int | None:
    """Return the best-effort number of CPUs available to the process."""

    budget = process_cpu_budget()
    effective = budget.get("effective")
    return int(effective) if isinstance(effective, int) else None


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
        room for GPU command submission.
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

    available = max(1, int(available))

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
