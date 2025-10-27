from __future__ import annotations

import cmath as _cmath
import math as _math
import threading as _threading
import types as _types
import sys
import importlib.abc as _importlib_abc
import importlib.util as _importlib_util
import weakref as _weakref
from collections import deque as _deque
from collections.abc import Iterable as _IterableABC, Sequence as _SequenceABC
from dataclasses import dataclass as _dataclass
from importlib import import_module
from typing import (
    Any as _Any,
    Callable as _Callable,
    Dict as _Dict,
    Iterable as _Iterable,
    List as _List,
    Mapping as _Mapping,
    MutableSequence as _MutableSequence,
    Optional as _Optional,
    Sequence as _Sequence,
    Tuple as _Tuple,
)
from importlib.metadata import version as _pkg_version, PackageNotFoundError

from ._meta import (
    BUILD_FINGERPRINT,
    BUILD_ID,
    BUILD_MANIFEST,
    BUILD_MANIFEST_JSON,
)
from ._zspace_aliases import (
    PRIMARY_ZSPACE_METRIC_ALIASES,
    ZSPACE_METRIC_ALIASES,
)

_rs: _types.ModuleType | None = None

_PREDECLARED_SUBMODULES: list[tuple[str, str]] = [
    ("nn", "SpiralTorch neural network primitives"),
    ("frac", "Fractal & fractional tools"),
    ("dataset", "Datasets & loaders"),
    ("linalg", "Linear algebra utilities"),
    ("planner", "Planning & device heuristics"),
    ("spiralk", "SpiralK DSL & hint bridges"),
    ("spiral_rl", "Reinforcement learning components"),
    ("rec", "Reconstruction / signal processing"),
    ("telemetry", "Telemetry / dashboards / metrics"),
    ("ecosystem", "Integrations & ecosystem glue"),
    ("selfsup", "Self-supervised objectives"),
    ("export", "Model export & compression"),
    ("compat", "Interoperability bridges"),
    ("hpo", "Hyper-parameter optimization tools"),
    ("inference", "Safety inference runtime & auditing"),
    ("zspace", "Z-space training helpers"),
    ("vision", "SpiralTorchVision orchestration"),
    ("canvas", "Canvas transformer utilities"),
]

_RENAMED_EXPORTS: dict[str, str] = {
    "DqnAgent": "stAgent",
}

_CANONICAL_METRIC_NAMES: dict[str, str] = {
    str(alias).lower(): str(canonical).lower()
    for alias, canonical in ZSPACE_METRIC_ALIASES.items()
}
for _canonical in set(_CANONICAL_METRIC_NAMES.values()):
    _CANONICAL_METRIC_NAMES.setdefault(_canonical.lower(), _canonical.lower())

_RELEVANT_ZSPACE_METRICS = {"speed", "memory", "stability", "drs", "gradient"}


def _safe_getattr(obj: _Any, name: str, default: _Any = None) -> _Any:
    if obj is None or not name:
        return default
    try:
        return getattr(obj, name)
    except AttributeError:
        return default


def _resolve_rs_attr(candidate: str) -> _Any | None:
    if not candidate:
        return None
    target: _Any = _rs
    for part in candidate.split("."):
        target = _safe_getattr(target, part, None)
        if target is None:
            return None
    return target


_parent_module = sys.modules[__name__]
for _name, _doc in _PREDECLARED_SUBMODULES:
    _fq = f"{__name__}.{_name}"
    _module = sys.modules.get(_fq)
    if _module is None:
        _module = _types.ModuleType(_fq, _doc)
        sys.modules[_fq] = _module
    elif _doc and not getattr(_module, "__doc__", None):
        _module.__doc__ = _doc
    setattr(_parent_module, _name, _module)
    globals()[_name] = _module

if "spiral_rl" not in sys.modules:
    _shim = _types.ModuleType("spiral_rl")
    # 参照される両方の候補名を用意しておく（実体は後で本物に差し替え）
    _shim.DqnAgent = type("DqnAgent", (), {})  # placeholder
    _shim.PyDqnAgent = type("PyDqnAgent", (), {})  # placeholder
    _shim.__spiraltorch_placeholder__ = True
    sys.modules["spiral_rl"] = _shim

try:
    _rs = import_module("spiraltorch.spiraltorch")
except ModuleNotFoundError as exc:
    if exc.name not in {"spiraltorch.spiraltorch", "spiraltorch"}:
        raise
    try:
        _rs = import_module("spiraltorch.spiraltorch_native")
    except ModuleNotFoundError as _native_exc:
        try:
            _rs = import_module("spiraltorch_native")
        except ModuleNotFoundError as _final_exc:
            if _native_exc.name not in {
                "spiraltorch.spiraltorch_native",
                "spiraltorch_native",
            }:
                raise
            if _final_exc.name not in {"spiraltorch.spiraltorch_native", "spiraltorch_native"}:
                raise
            _rs = None

# --- begin: promote real rl submodule & alias DqnAgent->stAgent ---
try:
    _spiral_rl = globals().get("spiral_rl")
    if isinstance(_spiral_rl, _types.ModuleType):
        sys.modules["spiral_rl"] = _spiral_rl
        if hasattr(_spiral_rl, "stAgent") and not hasattr(_spiral_rl, "DqnAgent"):
            setattr(_spiral_rl, "DqnAgent", getattr(_spiral_rl, "stAgent"))
except Exception:
    pass


def _is_valid_rl_module(module: _types.ModuleType | None) -> bool:
    return bool(
        isinstance(module, _types.ModuleType)
        and not getattr(module, "__spiraltorch_placeholder__", False)
        and hasattr(module, "stAgent")
    )


def _spiraltorch_rl_module(*, load: bool) -> _types.ModuleType | None:
    parent = sys.modules.get(__name__)
    module = _safe_getattr(parent, "rl")
    if _is_valid_rl_module(module):
        return module
    for cached in ("spiraltorch.rl", "spiraltorch.spiral_rl"):
        candidate = sys.modules.get(cached)
        if _is_valid_rl_module(candidate):
            return candidate
    if load:
        for candidate in ("spiraltorch.rl", "spiraltorch.spiral_rl"):
            try:
                resolved = import_module(candidate)
            except ModuleNotFoundError:
                continue
            if _is_valid_rl_module(resolved):
                return resolved
    return None


class _SpiralTorchRLLazyLoader(_importlib_abc.Loader):
    def create_module(self, spec):  # type: ignore[override]
        module = _spiraltorch_rl_module(load=True)
        if module is None:
            raise ModuleNotFoundError("spiraltorch.rl module is unavailable")
        sys.modules.setdefault(spec.name, module)
        return module

    def exec_module(self, module):  # type: ignore[override]
        sys.modules.setdefault("rl", module)


class _SpiralTorchRLAliasFinder(_importlib_abc.MetaPathFinder):
    def __init__(self) -> None:
        self._loader = _SpiralTorchRLLazyLoader()

    def find_spec(self, fullname, path, target=None):  # type: ignore[override]
        if fullname != "rl" or fullname in sys.modules:
            return None
        if _spiraltorch_rl_module(load=False) is None:
            return None
        return _importlib_util.spec_from_loader(fullname, self._loader, origin="spiraltorch")


_RL_ALIAS_FINDER: _SpiralTorchRLAliasFinder | None = None


def _ensure_rl_lazy_alias() -> None:
    global _RL_ALIAS_FINDER
    if "rl" in sys.modules:
        return
    if _spiraltorch_rl_module(load=False) is None:
        return
    if _RL_ALIAS_FINDER is None:
        _RL_ALIAS_FINDER = _SpiralTorchRLAliasFinder()
    for finder in sys.meta_path:
        if finder is _RL_ALIAS_FINDER:
            break
    else:
        sys.meta_path.insert(0, _RL_ALIAS_FINDER)


_ensure_rl_lazy_alias()
try:
    __version__ = _pkg_version("spiraltorch")
except PackageNotFoundError:
    __version__ = "0.0.0+local"


def print_build_id(*, verbose: bool = False) -> None:
    """Display the build identifier embedded in the wheel."""

    if verbose:
        print(f"[SpiralTorch] Build manifest: {BUILD_MANIFEST_JSON}")
    else:
        print(f"[SpiralTorch] Build ID: {BUILD_ID} ({BUILD_FINGERPRINT})")


def build_manifest() -> dict[str, _Any]:
    """Return a copy of the structured build metadata."""

    return dict(BUILD_MANIFEST)

from .zspace_inference import (
    ZMetrics,
    ZSpaceDecoded,
    ZSpaceInference,
    ZSpacePosterior,
    ZSpacePartialBundle,
    ZSpaceTelemetryFrame,
    ZSpaceInferencePipeline,
    inference_to_mapping,
    inference_to_zmetrics,
    prepare_trainer_step_payload,
    canvas_partial_from_snapshot,
    canvas_coherence_partial,
    elliptic_partial_from_telemetry,
    coherence_partial_from_diagnostics,
    decode_zspace_embedding,
    blend_zspace_partials,
    infer_canvas_snapshot,
    infer_canvas_transformer,
    infer_coherence_diagnostics,
    infer_coherence_from_sequencer,
    infer_canvas_with_coherence,
    infer_with_partials,
    infer_from_partial,
    weights_partial_from_dlpack,
    weights_partial_from_compat,
    infer_weights_from_dlpack,
    infer_weights_from_compat,
)

from .elliptic import (
    EllipticWarpFunction,
    elliptic_warp_autograd,
    elliptic_warp_features,
    elliptic_warp_partial,
)

# 追加API（Rust側でエクスポート済みのやつだけ拾う）
_EXTRAS = [
    "golden_ratio","golden_angle","set_global_seed",
    "capture","share","compat",
    "fibonacci_pacing","pack_nacci_chunks",
    "pack_tribonacci_chunks","pack_tetranacci_chunks",
    "generate_plan_batch_ex","plan","plan_topk",
    "describe_device","hip_probe","z_space_barycenter",
    "hypergrad","hypergrad_topos","encode_zspace","z_metrics",
]
for _n in _EXTRAS:
    _value = _safe_getattr(_rs, _n, None)
    if _value is not None:
        globals()[_n] = _value

_COMPAT_ALIAS = {
    "Tensor":   ("Tensor", "PyTensor"),
    "Device":   ("Device", "PyDevice"),
    "Dataset":  ("Dataset", "PyDataset"),
    "Plan":     ("Plan", "PyPlan"),
}
for _pub, _cands in _COMPAT_ALIAS.items():
    for _c in _cands:
        _value = _safe_getattr(_rs, _c, None)
        if _value is not None:
            globals()[_pub] = _value
            if _pub == "Tensor":
                globals()["PyTensor"] = _value
            break

_TENSOR_BASE = globals().get("Tensor")

if _TENSOR_BASE is not None:
    globals()["TensorBase"] = _TENSOR_BASE

    _TENSOR_NO_DATA = object()

    def _tensor_is_sequence(obj: _Any) -> bool:
        return isinstance(obj, _SequenceABC) and not isinstance(
            obj, (str, bytes, bytearray, memoryview)
        )


    def _tensor_is_iterable(obj: _Any) -> bool:
        return isinstance(obj, _IterableABC) and not isinstance(
            obj, (str, bytes, bytearray, memoryview)
        )


    def _tensor_coerce_index(value: _Any, label: str) -> int:
        try:
            index = int(value)
        except Exception as exc:  # noqa: BLE001 - surface Pythonic error message
            raise TypeError(f"Tensor {label} must be an integer, got {value!r}") from exc
        if index < 0:
            raise ValueError(f"Tensor {label} must be non-negative, got {index}")
        return index


    def _tensor_coerce_shape(value: _Any, label: str) -> tuple[int, int]:
        if not _tensor_is_sequence(value):
            raise TypeError(f"Tensor {label} must be a sequence of two integers")
        dims = list(value)
        if len(dims) != 2:
            raise ValueError(
                f"Tensor {label} must contain exactly two dimensions, got {len(dims)}"
            )
        rows = _tensor_coerce_index(dims[0], f"{label}[0]")
        cols = _tensor_coerce_index(dims[1], f"{label}[1]")
        return rows, cols


    def _tensor_maybe_shape(value: _Any) -> tuple[int, int] | None:
        if not _tensor_is_sequence(value):
            return None
        dims = list(value)
        if len(dims) != 2:
            return None
        try:
            return _tensor_coerce_shape(dims, "shape")
        except (TypeError, ValueError):
            return None


    def _tensor_normalize_row(row: _Any, *, allow_empty: bool) -> list[float]:
        if isinstance(row, _TENSOR_BASE):
            row = row.tolist()
        elif hasattr(row, "tolist") and not _tensor_is_sequence(row):
            row = row.tolist()
        if _tensor_is_sequence(row):
            seq = list(row)
        elif _tensor_is_iterable(row):
            seq = list(row)
        else:
            raise TypeError("Tensor rows must be sequences of numbers")
        if not allow_empty and not seq:
            raise ValueError("Tensor rows must not be empty")
        return [float(value) for value in seq]


    def _tensor_flatten_data(data: _Any) -> tuple[int, int, list[float]]:
        if isinstance(data, _TENSOR_BASE):
            rows, cols = (int(dim) for dim in data.shape())
            nested = data.tolist()
            flat: list[float] = [float(value) for row in nested for value in row]
            return rows, cols, flat

        if hasattr(data, "tolist") and not _tensor_is_sequence(data):
            return _tensor_flatten_data(data.tolist())

        if _tensor_is_sequence(data):
            items = list(data)
        elif _tensor_is_iterable(data):
            items = list(data)
        else:
            raise TypeError(
                "Tensor data must be an iterable of floats or nested iterables"
            )

        if not items:
            # Allow callers to explicitly describe zero-length tensors.
            return 0, 0, []

        head = items[0]
        if isinstance(head, _TENSOR_BASE):
            head = head.tolist()
        elif hasattr(head, "tolist") and not _tensor_is_sequence(head):
            head = head.tolist()

        if _tensor_is_sequence(head) or _tensor_is_iterable(head):
            rows = len(items)
            cols: int | None = None
            flat: list[float] = []
            for row in items:
                # Allow empty rows; downstream shape checks ensure that only
                # tensors with zero columns make it through.
                normalized = _tensor_normalize_row(row, allow_empty=True)
                if cols is None:
                    cols = len(normalized)
                elif len(normalized) != cols:
                    raise ValueError("Tensor rows must all share the same length")
                flat.extend(normalized)
            return rows, (0 if cols is None else cols), flat

        flat = [float(value) for value in items]
        return 1, len(flat), flat


    def _normalize_tensor_ctor_args(
        *args: _Any, **kwargs: _Any
    ) -> tuple[int, int, list[float] | object]:
        data_value = kwargs.pop("data", _TENSOR_NO_DATA)
        shape_value = kwargs.pop("shape", None)
        rows_value = kwargs.pop("rows", None)
        cols_value = kwargs.pop("cols", None)

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Tensor() got unexpected keyword arguments: {unexpected}")

        if data_value is None:
            data_value = _TENSOR_NO_DATA

        rows: int | None = None
        cols: int | None = None

        if shape_value is not None:
            rows, cols = _tensor_coerce_shape(shape_value, "shape")

        if rows_value is not None:
            rows = _tensor_coerce_index(rows_value, "rows")
        if cols_value is not None:
            cols = _tensor_coerce_index(cols_value, "cols")

        positional = list(args)
        if len(positional) == 1:
            candidate = positional[0]
            maybe_shape = None if rows is not None or cols is not None else _tensor_maybe_shape(candidate)
            if maybe_shape is not None:
                rows, cols = maybe_shape
            else:
                if data_value is not _TENSOR_NO_DATA:
                    raise TypeError("Tensor() got multiple values for data")
                data_value = _TENSOR_NO_DATA if candidate is None else candidate
        elif len(positional) == 2:
            first, second = positional
            maybe_shape = None if rows is not None or cols is not None else _tensor_maybe_shape(first)
            if maybe_shape is not None:
                rows, cols = maybe_shape
                if data_value is not _TENSOR_NO_DATA:
                    raise TypeError("Tensor() got multiple values for data")
                data_value = _TENSOR_NO_DATA if second is None else second
            else:
                inferred_rows = _tensor_coerce_index(first, "rows")
                inferred_cols = _tensor_coerce_index(second, "cols")
                if rows is not None and rows != inferred_rows:
                    raise ValueError(
                        f"Tensor rows argument conflicts with shape: {rows} != {inferred_rows}"
                    )
                if cols is not None and cols != inferred_cols:
                    raise ValueError(
                        f"Tensor cols argument conflicts with shape: {cols} != {inferred_cols}"
                    )
                rows = inferred_rows
                cols = inferred_cols
        elif len(positional) == 3:
            first, second, third = positional
            inferred_rows = _tensor_coerce_index(first, "rows")
            inferred_cols = _tensor_coerce_index(second, "cols")
            if rows is not None and rows != inferred_rows:
                raise ValueError(
                    f"Tensor rows argument conflicts with shape: {rows} != {inferred_rows}"
                )
            if cols is not None and cols != inferred_cols:
                raise ValueError(
                    f"Tensor cols argument conflicts with shape: {cols} != {inferred_cols}"
                )
            rows = inferred_rows
            cols = inferred_cols
            if data_value is not _TENSOR_NO_DATA:
                raise TypeError("Tensor() got multiple values for data")
            data_value = _TENSOR_NO_DATA if third is None else third
        elif len(positional) > 3:
            raise TypeError(
                "Tensor() takes at most 3 positional arguments"
                f" but {len(positional)} were given"
            )

        if data_value is _TENSOR_NO_DATA:
            if rows is None or cols is None:
                raise TypeError("Tensor() requires a shape when data is omitted")
            return rows, cols, _TENSOR_NO_DATA

        inferred_rows, inferred_cols, flat = _tensor_flatten_data(data_value)
        total = len(flat)

        def _infer_missing_dimension(
            total_elems: int, known: int, *, known_label: str
        ) -> int:
            """Derive the complementary dimension from a known axis length."""

            if known == 0:
                if total_elems != 0:
                    raise ValueError(
                        f"Tensor data of length {total_elems} cannot fill ({known}) {known_label}"
                    )
                return 0
            if total_elems % known != 0:
                raise ValueError(
                    f"Tensor data of length {total_elems} cannot fill ({known}) {known_label}"
                )
            return total_elems // known

        if rows is None and cols is None:
            rows, cols = inferred_rows, inferred_cols
        elif rows is None:
            if cols is None:
                raise TypeError("Tensor() could not determine rows from provided inputs")
            rows = _infer_missing_dimension(total, cols, known_label="columns")
        elif cols is None:
            cols = _infer_missing_dimension(total, rows, known_label="rows")
        else:
            if rows * cols != total:
                raise ValueError(
                    f"Tensor data of length {total} cannot be reshaped to ({rows}, {cols})"
                )

        if rows is None or cols is None:
            raise TypeError("Tensor() could not determine both rows and cols from the provided data")

        if (rows == 0 or cols == 0) and total != 0:
            raise ValueError(
                f"Tensor shape ({rows}, {cols}) is incompatible with {total} data elements"
            )

        return rows, cols, flat


    class TensorMeta(type(_TENSOR_BASE)):
        def __instancecheck__(cls, instance: _Any) -> bool:  # noqa: D401 - delegated check
            return isinstance(instance, _TENSOR_BASE)

        def __subclasscheck__(cls, subclass: _Any) -> bool:  # noqa: D401 - delegated check
            try:
                return issubclass(subclass, _TENSOR_BASE)
            except TypeError:
                return False


    class Tensor(_TENSOR_BASE, metaclass=TensorMeta):
        """Flexible front-end wrapper around the native SpiralTorch tensor."""

        __doc__ = getattr(_TENSOR_BASE, "__doc__", None)

        def __new__(cls, *args: _Any, **kwargs: _Any):
            rows, cols, payload = _normalize_tensor_ctor_args(*args, **kwargs)
            if payload is _TENSOR_NO_DATA:
                return super().__new__(cls, rows, cols)
            return super().__new__(cls, rows, cols, payload)


    Tensor.__module__ = __name__
    globals()["Tensor"] = Tensor
    _TensorFastType = Tensor

else:
    _TensorFastType = globals().get("Tensor")


if isinstance(_TensorFastType, type):
    _native_tensor_matmul = getattr(_TensorFastType, "matmul", None)
    _tensor_matmul_simd_prepacked = getattr(
        _TensorFastType, "matmul_simd_prepacked", None
    )
    _tensor_storage_token = getattr(_TensorFastType, "storage_token", None)
    _cpu_simd_prepack_rhs = globals().get("cpu_simd_prepack_rhs")

    if (
        callable(_native_tensor_matmul)
        and callable(_tensor_matmul_simd_prepacked)
        and callable(_tensor_storage_token)
        and callable(_cpu_simd_prepack_rhs)
    ):

        @_dataclass
        class _SimdPackEntry:
            pack: _Any
            shape: _Tuple[int, int]
            token: int


        class _SimdPackCache:
            __slots__ = ("_packs", "_lock")

            def __init__(self) -> None:
                self._packs: "_weakref.WeakKeyDictionary[_Any, _SimdPackEntry]" = (
                    _weakref.WeakKeyDictionary()
                )
                self._lock = _threading.RLock()

            def fetch(self, rhs: _Any) -> _SimdPackEntry | None:
                try:
                    shape = rhs.shape()
                    token = int(_tensor_storage_token(rhs))
                except Exception:
                    return None

                with self._lock:
                    cached = self._packs.get(rhs)
                    if (
                        cached is not None
                        and cached.shape == shape
                        and cached.token == token
                    ):
                        return cached

                try:
                    pack = _cpu_simd_prepack_rhs(rhs)
                except Exception:
                    return None

                entry = _SimdPackEntry(pack=pack, shape=shape, token=token)
                with self._lock:
                    existing = self._packs.get(rhs)
                    if (
                        existing is not None
                        and existing.shape == shape
                        and existing.token == token
                    ):
                        return existing
                    self._packs[rhs] = entry
                return entry

            def invalidate(self, rhs: _Any) -> None:
                with self._lock:
                    self._packs.pop(rhs, None)

            def clear(self) -> None:
                with self._lock:
                    self._packs.clear()


        _SIMD_PACK_CACHE = _SimdPackCache()
        _SIMD_FAST_LABELS = {"python-simd", "cpu-simd-prepack", "cpu-simd", "simd"}

        def _tensor_matmul_fastpath(
            self: _Any,
            other: _Any,
            *,
            backend: str | None = None,
            out: _Any = None,
        ) -> _Any:
            label = backend or "auto"
            fallback_backend = backend

            if label in {"python-simd", "cpu-simd-prepack"}:
                fallback_backend = "cpu-simd"

            if label in _SIMD_FAST_LABELS:
                entry = _SIMD_PACK_CACHE.fetch(other)
                if entry is not None:
                    try:
                        if out is None:
                            return _tensor_matmul_simd_prepacked(self, entry.pack)
                        return _tensor_matmul_simd_prepacked(self, entry.pack, out=out)
                    except Exception:
                        _SIMD_PACK_CACHE.invalidate(other)

            return _native_tensor_matmul(self, other, backend=fallback_backend, out=out)


        _TensorFastType.matmul = _tensor_matmul_fastpath  # type: ignore[assignment]

        def clear_simd_prepack_cache() -> None:
            """Evict cached SIMD RHS packs used by the Python fast-path."""

            _SIMD_PACK_CACHE.clear()


        globals()["clear_simd_prepack_cache"] = clear_simd_prepack_cache
        _EXTRAS.append("clear_simd_prepack_cache")

def _extract_shape_like(candidate: _Any, label: str) -> tuple[int, int] | None:
    if isinstance(candidate, globals().get("Tensor", ())):
        rows, cols = candidate.shape()
        return int(rows), int(cols)
    if hasattr(candidate, "shape"):
        shape_attr = candidate.shape
        if callable(shape_attr):
            dims = shape_attr()
        else:
            dims = shape_attr
        if _tensor_is_sequence(dims):
            try:
                return _tensor_coerce_shape(dims, label)
            except (TypeError, ValueError):
                return None
    maybe_shape = _tensor_maybe_shape(candidate)
    if maybe_shape is not None:
        return maybe_shape
    return None


def _normalize_hypergrad_shape(
    *args: _Any,
    shape: _Any | None = None,
    rows: _Any | None = None,
    cols: _Any | None = None,
) -> tuple[int, int]:
    resolved_rows = _tensor_coerce_index(rows, "rows") if rows is not None else None
    resolved_cols = _tensor_coerce_index(cols, "cols") if cols is not None else None

    inferred_shape = None
    if shape is not None:
        inferred_shape = _extract_shape_like(shape, "shape")
        if inferred_shape is None:
            inferred_shape = _tensor_coerce_shape(shape, "shape")

    positional = list(args)
    if len(positional) == 1:
        candidate = positional[0]
        maybe_shape = _extract_shape_like(candidate, "shape")
        if maybe_shape is not None:
            inferred_shape = maybe_shape if inferred_shape is None else inferred_shape
            if inferred_shape != maybe_shape:
                raise ValueError(
                    f"hypergrad shape {maybe_shape} conflicts with declared shape {inferred_shape}"
                )
        else:
            value = _tensor_coerce_index(candidate, "rows")
            if resolved_rows is not None and resolved_rows != value:
                raise ValueError(
                    f"hypergrad rows {value} conflicts with declared rows {resolved_rows}"
                )
            resolved_rows = value
    elif len(positional) == 2:
        if inferred_shape is not None:
            raise TypeError("hypergrad() received multiple shape specifications")
        first, second = positional
        inferred_rows = _tensor_coerce_index(first, "rows")
        inferred_cols = _tensor_coerce_index(second, "cols")
        if resolved_rows is not None and resolved_rows != inferred_rows:
            raise ValueError(
                f"hypergrad rows {inferred_rows} conflicts with declared rows {resolved_rows}"
            )
        if resolved_cols is not None and resolved_cols != inferred_cols:
            raise ValueError(
                f"hypergrad cols {inferred_cols} conflicts with declared cols {resolved_cols}"
            )
        resolved_rows = inferred_rows
        resolved_cols = inferred_cols
    elif len(positional) > 2:
        raise TypeError(
            "hypergrad() takes at most 2 positional arguments"
            f" but {len(positional)} were given"
        )

    if inferred_shape is not None:
        rows_candidate, cols_candidate = inferred_shape
        if resolved_rows is not None and resolved_rows != rows_candidate:
            raise ValueError(
                f"hypergrad rows {rows_candidate} conflicts with declared rows {resolved_rows}"
            )
        if resolved_cols is not None and resolved_cols != cols_candidate:
            raise ValueError(
                f"hypergrad cols {cols_candidate} conflicts with declared cols {resolved_cols}"
            )
        resolved_rows, resolved_cols = rows_candidate, cols_candidate

    if resolved_rows is None or resolved_cols is None:
        raise TypeError("hypergrad() requires a shape or explicit rows/cols")

    return resolved_rows, resolved_cols


def _require_rs_class(name: str) -> _Any:
    existing = globals().get(name)
    if existing is not None:
        return existing
    resolved = _resolve_rs_attr(name)
    if resolved is None:
        raise AttributeError(f"SpiralTorch native attribute '{name}' is unavailable")
    globals()[name] = resolved
    return resolved


def _coerce_topos(topos: _Any | None) -> _Any | None:
    if topos is None:
        return None
    topos_cls = _require_rs_class("OpenCartesianTopos")
    if isinstance(topos, topos_cls):
        return topos
    if isinstance(topos, _Mapping):
        curvature = float(topos.get("curvature", -1.0))
        tolerance = float(topos.get("tolerance", 1e-3))
        saturation = float(topos.get("saturation", 1.0))
        depth = int(topos.get("max_depth", topos.get("depth", 64)))
        volume = int(topos.get("max_volume", topos.get("volume", 512)))
        return topos_cls(curvature, tolerance, saturation, depth, volume)
    if _tensor_is_sequence(topos):
        items = list(topos)
        if len(items) != 5:
            raise ValueError(
                "topos sequences must provide (curvature, tolerance, saturation, depth, volume)"
            )
        curvature, tolerance, saturation, depth, volume = items
        return topos_cls(
            float(curvature),
            float(tolerance),
            float(saturation),
            int(depth),
            int(volume),
        )
    raise TypeError(
        "topos must be an OpenCartesianTopos, mapping, or sequence"
    )


def hypergrad(
    *shape_args: _Any,
    curvature: float = -1.0,
    learning_rate: float = 0.05,
    shape: _Any | None = None,
    rows: _Any | None = None,
    cols: _Any | None = None,
    topos: _Any | None = None,
) -> _Any:
    rows_value, cols_value = _normalize_hypergrad_shape(
        *shape_args, shape=shape, rows=rows, cols=cols
    )
    tape_cls = _require_rs_class("Hypergrad")
    guard = _coerce_topos(topos)
    if guard is not None:
        return tape_cls(curvature, learning_rate, rows_value, cols_value, guard)
    return tape_cls(curvature, learning_rate, rows_value, cols_value)


def hypergrad_topos(
    *,
    curvature: float = -1.0,
    tolerance: float = 1e-3,
    saturation: float = 1.0,
    max_depth: int = 64,
    max_volume: int = 512,
) -> _Any:
    topos_cls = _require_rs_class("OpenCartesianTopos")
    return topos_cls(curvature, tolerance, saturation, int(max_depth), int(max_volume))


def encode_zspace(
    text: str,
    *,
    curvature: float = -1.0,
    temperature: float = 0.5,
    encoder: _Any | None = None,
) -> _Any:
    encoder_cls = _require_rs_class("LanguageWaveEncoder")
    tensor_cls = _require_rs_class("Tensor")
    created = False
    if encoder is None:
        encoder = encoder_cls(curvature, temperature)
        created = True
    elif not isinstance(encoder, encoder_cls):
        raise TypeError("encoder must be a LanguageWaveEncoder instance")
    try:
        tensor = encoder.encode_z_space(text)
    finally:
        if created:
            try:
                encoder.close()
            except AttributeError:
                pass
    if not isinstance(tensor, tensor_cls):
        tensor = tensor_cls(tensor)
    return tensor


_Z_METRIC_ALIAS = PRIMARY_ZSPACE_METRIC_ALIASES


_Z_PARTIAL_ALIAS = ZSPACE_METRIC_ALIASES


def _coerce_gradient_values(value: _Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, ZMetrics):
        value = value.gradient
    if value is None:
        return None
    if isinstance(value, _Mapping):
        value = value.values()
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("gradient metrics must be provided as an iterable of floats")
    try:
        return [float(entry) for entry in value]
    except TypeError as exc:  # noqa: BLE001 - surface a user-friendly error message
        raise TypeError("gradient metrics must be provided as an iterable of floats") from exc


def _metrics_to_mapping(metrics: ZMetrics) -> dict[str, _Any]:
    payload: dict[str, _Any] = {
        "speed": float(metrics.speed),
        "memory": float(metrics.memory),
        "stability": float(metrics.stability),
        "drs": float(metrics.drs),
    }
    gradient = _coerce_gradient_values(metrics.gradient)
    if gradient is not None:
        payload["gradient"] = gradient
    return payload


def _canonicalise_partial_mapping(payload: _Mapping[str, _Any] | None) -> dict[str, _Any]:
    if payload is None:
        return {}
    resolved: dict[str, _Any] = {}
    for key, value in payload.items():
        alias = _Z_PARTIAL_ALIAS.get(key.lower())
        if alias is None:
            raise KeyError(f"unknown Z-space metric '{key}'")
        if alias == "gradient":
            gradient = _coerce_gradient_values(value)
            if gradient is not None:
                resolved[alias] = gradient
            continue
        try:
            resolved[alias] = float(value)
        except (TypeError, ValueError) as exc:  # noqa: BLE001 - user feedback
            raise TypeError(
                f"Z-space metric '{key}' must be a real number, got {value!r}"
            ) from exc
    return resolved


def _flatten_telemetry(payload: _Mapping[str, _Any], *, prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in payload.items():
        label = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, _Mapping):
            flattened.update(_flatten_telemetry(value, prefix=label))
            continue
        try:
            flattened[label] = float(value)
        except (TypeError, ValueError):
            continue
    return flattened


def _normalise_telemetry_arg(payload: _Any) -> dict[str, float]:
    if payload is None:
        return {}
    if isinstance(payload, ZSpacePartialBundle):
        mapping = payload.telemetry_payload()
        return dict(mapping or {})
    if isinstance(payload, ZSpaceTelemetryFrame):
        return dict(payload.payload)
    if isinstance(payload, _Mapping):
        return _flatten_telemetry(payload)
    raise TypeError("telemetry payloads must be mappings or telemetry frames")


def z_metrics(
    *,
    speed: float | None = None,
    memory: float | None = None,
    stability: float | None = None,
    drs: float | None = None,
    gradient: _Any | None = None,
    **aliases: _Any,
) -> ZMetrics:
    values: dict[str, _Any] = {
        "speed": speed,
        "memory": memory,
        "stability": stability,
        "drs": drs,
        "gradient": gradient,
    }
    for key, value in aliases.items():
        alias = _Z_METRIC_ALIAS.get(key.lower())
        if alias is None:
            raise KeyError(f"unknown Z-space metric alias '{key}'")
        values[alias] = value

    grad_value = values.get("gradient")
    grad_list = None
    if grad_value is not None:
        if isinstance(grad_value, ZMetrics):
            base_grad = grad_value.gradient
            if base_grad is not None:
                grad_list = [float(v) for v in base_grad]
        else:
            grad_list = [float(v) for v in grad_value]

    return ZMetrics(
        speed=float(values.get("speed", 0.0) or 0.0),
        memory=float(values.get("memory", 0.0) or 0.0),
        stability=float(values.get("stability", 0.0) or 0.0),
        gradient=grad_list,
        drs=float(values.get("drs", 0.0) or 0.0),
    )


class _HypergradPartial:
    """Callable proxy that bakes a shape into :func:`hypergrad`."""

    __slots__ = ("_args", "_kwargs")

    def __init__(self, args: _Tuple[_Any, ...], kwargs: _Dict[str, _Any]) -> None:
        self._args = args
        self._kwargs = kwargs

    def __call__(self, *args: _Any, **kwargs: _Any) -> _Any:
        if args:
            raise TypeError(
                "hypergrad notation binds the shape already; pass configuration as keyword arguments"
            )
        merged: _Dict[str, _Any] = dict(self._kwargs)
        for key in ("shape", "rows", "cols"):
            if key in merged and key in kwargs:
                raise TypeError(
                    f"hypergrad() shape component '{key}' was provided by the notation and cannot be overridden"
                )
        merged.update(kwargs)
        return hypergrad(*self._args, **merged)

    def with_topos(self, *, topos: _Any | None = None, **kwargs: _Any) -> _Any:
        """Return a tape while constructing (or reusing) a guard inline."""

        if topos is not None:
            if kwargs:
                raise TypeError("with_topos() cannot mix 'topos=' with additional guard kwargs")
            return self(topos=topos)
        if not kwargs:
            raise TypeError("with_topos() requires either 'topos=' or guard keyword arguments")
        return self(topos=hypergrad_topos(**kwargs))


class _HypergradNotation:
    """Lightweight DSL that shortens hypergrad tape construction."""

    __slots__ = ()

    def __call__(self, *shape_args: _Any, **kwargs: _Any) -> _Any:
        return hypergrad(*shape_args, **kwargs)

    def __getitem__(self, selector: _Any) -> _HypergradPartial:
        if isinstance(selector, slice):
            if selector.step is not None:
                raise TypeError("hypergrad slice notation does not support step")
            base_kwargs: _Dict[str, _Any] = {}
            if selector.start is not None:
                base_kwargs["rows"] = _tensor_coerce_index(selector.start, "rows")
            if selector.stop is not None:
                base_kwargs["cols"] = _tensor_coerce_index(selector.stop, "cols")
            if not base_kwargs:
                raise TypeError("hypergrad[:] requires at least rows or cols")
            return _HypergradPartial((), base_kwargs)
        if isinstance(selector, tuple):
            if not selector:
                raise TypeError("hypergrad[] requires a shape or tensor")
            if len(selector) == 1:
                return _HypergradPartial((selector[0],), {})
            if len(selector) == 2:
                return _HypergradPartial((), {"shape": tuple(selector)})
            raise TypeError("hypergrad[...] accepts at most two entries")
        return _HypergradPartial((selector,), {})

    def topos(self, **kwargs: _Any) -> _Any:
        return hypergrad_topos(**kwargs)

    guard = topos

class _ZSpaceNotation:
    """Syntactic sugar for encoding text and metrics into Z-space."""

    __slots__ = ()

    def __call__(self, text: str, **kwargs: _Any) -> _Any:
        return encode_zspace(text, **kwargs)

    def _module(self) -> _types.ModuleType:
        return _ensure_submodule("zspace")

    def __getitem__(self, selector: _Any) -> _Any:
        if isinstance(selector, str):
            return encode_zspace(selector)
        if isinstance(selector, tuple):
            if not selector:
                raise TypeError("z[] requires a text payload")
            text = selector[0]
            if not isinstance(text, str):
                raise TypeError("z[...] expects the first element to be text")
            options: _Dict[str, _Any] = {}
            for extra in selector[1:]:
                if isinstance(extra, _Mapping):
                    options.update(extra)
                elif isinstance(extra, (int, float)):
                    if "temperature" in options:
                        raise TypeError("temperature supplied multiple times in z[...] notation")
                    options["temperature"] = float(extra)
                elif isinstance(extra, tuple) and len(extra) == 2 and isinstance(extra[0], str):
                    key, value = extra
                    if key in options:
                        raise TypeError(f"argument '{key}' supplied multiple times in z[...] notation")
                    options[key] = value
                else:
                    raise TypeError("unsupported z[...] argument; use mappings, scalars, or (key, value) pairs")
            return encode_zspace(text, **options)
        raise TypeError("z[...] expects text or (text, …) tuples")

    def metrics(self, **kwargs: _Any) -> ZMetrics:
        return z_metrics(**kwargs)

    def describe(self, *, latest: bool = True, feedback: bool = False) -> _Any:
        module = self._module()
        describe = _safe_getattr(module, "describe") or _safe_getattr(module, "describe_zspace")
        if describe is None:
            raise RuntimeError("z.describe() is unavailable in this build")
        return describe(latest=latest, feedback=feedback)

    def feedback(self) -> _Any:
        module = self._module()
        feedback = _safe_getattr(module, "feedback") or _safe_getattr(module, "softlogic_feedback")
        if feedback is None:
            raise RuntimeError("z.feedback() is unavailable in this build")
        return feedback()

    def snapshot(self) -> _Any:
        module = self._module()
        snapshot = _safe_getattr(module, "snapshot") or _safe_getattr(module, "zspace_snapshot")
        if snapshot is None:
            raise RuntimeError("z.snapshot() is unavailable in this build")
        return snapshot()

    def signal(self) -> _Any:
        module = self._module()
        signal = _safe_getattr(module, "softlogic_signal")
        if signal is None:
            raise RuntimeError("z.signal() is unavailable in this build")
        return signal()

    def partial(
        self,
        *args: _Any,
        weight: float | None = None,
        origin: str | None = None,
        telemetry: _Any | None = None,
        **metrics: _Any,
    ) -> ZSpacePartialBundle:
        base_metrics: dict[str, _Any] = {}
        base_weight: float = 1.0
        base_origin: str | None = None
        telemetry_payload: dict[str, float] | None = None

        if len(args) > 1:
            raise TypeError("z.partial() accepts at most one positional argument")

        if args:
            source = args[0]
            if isinstance(source, ZSpacePartialBundle):
                base_metrics = source.resolved()
                base_weight = float(source.weight)
                base_origin = source.origin
                telemetry_payload = dict(source.telemetry_payload() or {}) or None
            elif isinstance(source, ZMetrics):
                base_metrics = _metrics_to_mapping(source)
            elif isinstance(source, _Mapping):
                base_metrics = _canonicalise_partial_mapping(source)
            elif source is None:
                base_metrics = {}
            else:
                raise TypeError(
                    "z.partial() positional argument must be a mapping, ZMetrics, or ZSpacePartialBundle"
                )

        extra_metrics = _canonicalise_partial_mapping(metrics) if metrics else {}
        merged_metrics = dict(base_metrics)
        if extra_metrics:
            merged_metrics.update(extra_metrics)

        if "gradient" in merged_metrics and merged_metrics["gradient"] is None:
            merged_metrics.pop("gradient")

        final_weight = float(weight) if weight is not None else base_weight
        final_origin = origin if origin is not None else base_origin

        if telemetry is not None:
            telemetry_payload = dict(telemetry_payload or {})
            telemetry_payload.update(_normalise_telemetry_arg(telemetry))
        elif telemetry_payload is not None:
            telemetry_payload = dict(telemetry_payload)

        return ZSpacePartialBundle(
            merged_metrics,
            weight=final_weight,
            origin=final_origin,
            telemetry=telemetry_payload,
        )

    def bundle(
        self,
        *partials: _Any,
        strategy: str = "mean",
        weights: _Sequence[float] | None = None,
    ) -> dict[str, _Any]:
        if len(partials) == 1 and isinstance(partials[0], _SequenceABC):
            sequence = list(partials[0])
        else:
            sequence = list(partials)

        normalised: list[ZSpacePartialBundle | dict[str, _Any] | None] = []
        for partial in sequence:
            if partial is None:
                normalised.append(None)
                continue
            if isinstance(partial, ZSpacePartialBundle):
                normalised.append(partial)
                continue
            if isinstance(partial, ZMetrics):
                normalised.append(ZSpacePartialBundle(_metrics_to_mapping(partial)))
                continue
            if isinstance(partial, _Mapping):
                normalised.append(_canonicalise_partial_mapping(partial))
                continue
            raise TypeError(
                "z.bundle() expects partial bundles, mappings, or ZMetrics entries"
            )

        return blend_zspace_partials(normalised, strategy=strategy, weights=weights)

    blend = bundle


hg = _HypergradNotation()
z = _ZSpaceNotation()

_FORWARDING_HINTS: dict[str, dict[str, tuple[str, ...]]] = {
    "nn": {
        "NonLiner": ("NonLiner",),
        "Dropout": ("Dropout",),
        "Dataset": ("_NnDataset",),
        "DataLoader": ("_NnDataLoader",),
        "DataLoaderIter": ("_NnDataLoaderIter",),
        "ZConv": ("PyZConv",),
        "ZConv6DA": ("PyZConv6DA",),
        "ZPooling": ("PyZPooling",),
        "from_samples": ("nn_from_samples", "dataset_from_samples"),
        "CurvatureScheduler": ("CurvatureScheduler",),
        "CurvatureDecision": ("CurvatureDecision",),
        "softlogic_signal": ("softlogic_signal",),
    },
    "compat": {
        "capture": ("capture",),
        "share": ("share",),
    },
    "planner": {
        "RankPlan": ("PyRankPlan",),
        "plan": (),
        "plan_topk": (),
        "describe_device": (),
        "hip_probe": (),
        "generate_plan_batch_ex": (),
    },
    "spiralk": {
        "FftPlan": (),
        "MaxwellBridge": (),
        "MaxwellHint": (),
        "MaxwellFingerprint": (),
        "MeaningGate": (),
        "SequentialZ": (),
        "MaxwellPulse": (),
        "MaxwellProjector": (),
        "required_blocks": (),
    },
    "compat.torch": {
        "to_torch": ("compat_to_torch", "to_torch"),
        "from_torch": ("compat_from_torch", "from_torch"),
    },
    "compat.jax": {
        "to_jax": ("compat_to_jax", "to_jax"),
        "from_jax": ("compat_from_jax", "from_jax"),
    },
    "compat.tensorflow": {
        "to_tensorflow": ("compat_to_tensorflow", "to_tensorflow"),
        "from_tensorflow": ("compat_from_tensorflow", "from_tensorflow"),
    },
    "spiral_rl": {
        "stAgent": ("PyDqnAgent", "DqnAgent", "StAgent"),
        "PpoAgent": ("PyPpoAgent",),
        "SacAgent": ("PySacAgent",),
    },
    "rec": {
        "QueryPlan": ("PyQueryPlan",),
        "RecEpochReport": ("PyRecEpochReport",),
        "Recommender": ("PyRecommender",),
    },
    "telemetry": {
        "DashboardMetric": ("PyDashboardMetric",),
        "DashboardEvent": ("PyDashboardEvent",),
        "DashboardFrame": ("PyDashboardFrame",),
        "DashboardRing": ("PyDashboardRing",),
        "DashboardRingIter": ("PyDashboardRingIter",),
        "current": ("current",),
        "SoftlogicZFeedback": ("SoftlogicZFeedback",),
        "ZSpaceRegionDescriptor": ("ZSpaceRegionDescriptor",),
    },
    "zspace": {
        "ZSpaceSpinBand": ("ZSpaceSpinBand",),
        "ZSpaceRadiusBand": ("ZSpaceRadiusBand",),
        "ZSpaceRegionKey": ("ZSpaceRegionKey",),
        "ZSpaceRegionDescriptor": ("ZSpaceRegionDescriptor",),
        "SoftlogicEllipticSample": ("SoftlogicEllipticSample",),
        "SoftlogicZFeedback": ("SoftlogicZFeedback",),
        "snapshot": ("snapshot", "zspace_snapshot"),
        "feedback": ("feedback", "softlogic_feedback"),
        "describe": ("describe", "describe_zspace"),
        "softlogic_signal": ("softlogic_signal",),
    },
    "export": {
        "QatObserver": ("PyQatObserver",),
        "QuantizationReport": ("PyQuantizationReport",),
        "StructuredPruningReport": ("PyStructuredPruningReport",),
        "CompressionReport": ("PyCompressionReport",),
        "structured_prune": (),
        "compress_weights": (),
    },
    "hpo": {
        "SearchLoop": ("PySearchLoop",),
    },
    "selfsup": {
        "info_nce": ("selfsup.info_nce",),
        "masked_mse": ("selfsup.masked_mse",),
    },
}


@_dataclass(frozen=True)
class Axis:
    """Named axis descriptor used by :class:`LabeledTensor`."""

    name: str
    size: int | None = None

    def __post_init__(self) -> None:  # pragma: no cover - sanity guard
        label = str(self.name).strip()
        if not label:
            raise ValueError("axis name must be a non-empty string")
        object.__setattr__(self, "name", label)
        if self.size is not None:
            value = int(self.size)
            if value <= 0:
                raise ValueError("axis size must be positive")
            object.__setattr__(self, "size", value)

    def with_size(self, size: int) -> "Axis":
        """Return a copy with the provided concrete size."""

        if size <= 0:
            raise ValueError("size must be positive")
        return Axis(self.name, size)

    def __str__(self) -> str:  # pragma: no cover - representation helper
        suffix = self.size if self.size is not None else "?"
        return f"{self.name}[{suffix}]"


def _ensure_tensor_type() -> type:
    tensor_type = globals().get("Tensor")
    if tensor_type is None:
        raise RuntimeError("Tensor export is unavailable in this build")
    return tensor_type


def _prepare_rows(data: _Any) -> _List[_List[float]]:
    if hasattr(data, "tolist"):
        data = data.tolist()
    if not isinstance(data, _Sequence):
        raise TypeError("tensor data must be a sequence")
    if not data:
        raise ValueError("tensor data must contain at least one row")
    rows: _List[_List[float]] = []
    if isinstance(data[0], _Sequence):  # type: ignore[index]
        width: int | None = None
        for row in data:  # type: ignore[assignment]
            if hasattr(row, "tolist"):
                row = row.tolist()
            if not isinstance(row, _Sequence):
                raise TypeError("tensor rows must be sequences of numbers")
            values = [float(value) for value in row]
            if not values:
                raise ValueError("tensor rows cannot be empty")
            if width is None:
                width = len(values)
            elif len(values) != width:
                raise ValueError("all rows must share the same length")
            rows.append(values)
    else:
        values = [float(value) for value in data]  # type: ignore[assignment]
        if not values:
            raise ValueError("tensor data must contain at least one element")
        rows.append(values)
    return rows


def _tensor_from_data(data: _Any):
    tensor_type = _ensure_tensor_type()
    if isinstance(data, tensor_type):
        return data
    rows = _prepare_rows(data)
    height = len(rows)
    width = len(rows[0])
    flat: _List[float] = [value for row in rows for value in row]
    return tensor_type(height, width, flat)


def _coerce_axis(axis: "Axis | str") -> Axis:
    if isinstance(axis, Axis):
        return axis
    if isinstance(axis, str):
        return Axis(axis)
    raise TypeError("axes must be Axis instances or strings")


def _resolve_axis_size(axis: Axis, size: int) -> Axis:
    if size <= 0:
        raise ValueError("tensor dimensions must be positive")
    if axis.size is None:
        return axis.with_size(size)
    if axis.size != size:
        raise ValueError(
            f"axis '{axis.name}' expects size {axis.size}, received {size}"
        )
    return axis


def _normalise_axes(axes: _Sequence["Axis | str"]) -> tuple[Axis, Axis]:
    seq = list(axes)
    if len(seq) != 2:
        raise ValueError("exactly two axes are required for a 2D tensor")
    first = _coerce_axis(seq[0])
    second = _coerce_axis(seq[1])
    return (first, second)


class LabeledTensor:
    """Tensor wrapper that carries human-readable axis annotations."""

    def __init__(self, data: _Any, axes: _Sequence["Axis | str"]) -> None:
        base = _tensor_from_data(data)
        resolved = _normalise_axes(axes)
        self._tensor = base
        self._axes = (
            _resolve_axis_size(resolved[0], base.rows),
            _resolve_axis_size(resolved[1], base.cols),
        )

    @property
    def tensor(self):
        return self._tensor

    @property
    def axes(self) -> tuple[Axis, Axis]:
        return self._axes

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def rows(self) -> int:
        return self._tensor.rows

    @property
    def cols(self) -> int:
        return self._tensor.cols

    def to_tensor(self):
        return self._tensor

    def tolist(self) -> _List[_List[float]]:
        return self._tensor.tolist()

    def rename(self, axes: _Sequence["Axis | str"]) -> "LabeledTensor":
        return LabeledTensor(self._tensor, axes)

    def with_axes(self, axes: _Sequence["Axis | str"]) -> "LabeledTensor":
        return self.rename(axes)

    def transpose(self) -> "LabeledTensor":
        return LabeledTensor(self._tensor.transpose(), (self._axes[1], self._axes[0]))

    def row_softmax(self, *, backend: str | None = None) -> "LabeledTensor":
        return LabeledTensor(
            self._tensor.row_softmax(backend=backend),
            self._axes,
        )

    def __matmul__(self, other: "LabeledTensor") -> "LabeledTensor":
        if not isinstance(other, LabeledTensor):
            return NotImplemented
        left_axis = self._axes[1]
        right_axis = other._axes[0]
        if left_axis.name != right_axis.name:
            raise ValueError(
                f"axis mismatch: cannot contract '{left_axis.name}' with '{right_axis.name}'"
            )
        if (
            left_axis.size is not None
            and right_axis.size is not None
            and left_axis.size != right_axis.size
        ):
            raise ValueError(
                f"axis '{left_axis.name}' expects size {left_axis.size}, received {right_axis.size}"
            )
        return LabeledTensor(
            self._tensor.matmul(other._tensor),
            (self._axes[0], other._axes[1]),
        )

    def describe(self) -> dict[str, _Any]:
        return {
            "axes": [axis.name for axis in self._axes],
            "axis_sizes": [axis.size for axis in self._axes],
            "shape": self.shape,
        }

    def axis_names(self) -> tuple[str, str]:
        return (self._axes[0].name, self._axes[1].name)

    def __iter__(self):  # pragma: no cover - simple iterator proxy
        return iter(self.tolist())

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        axis_repr = ", ".join(str(axis) for axis in self._axes)
        return f"LabeledTensor(shape={self.shape}, axes=({axis_repr}))"

def tensor(
    data: _Any,
    *,
    axes: _Optional[_Sequence["Axis | str"]] = None,
):
    """Construct a :class:`Tensor` (optionally annotated with axes)."""

    base = _tensor_from_data(data)
    if axes is None:
        return base
    return LabeledTensor(base, axes)


def label_tensor(tensor_obj: _Any, axes: _Sequence["Axis | str"]) -> LabeledTensor:
    """Attach axis annotations to an existing tensor-like object."""

    return LabeledTensor(tensor_obj, axes)


def _coerce_gradient_sequence(payload: _Any) -> list[float] | None:
    if payload is None or isinstance(payload, (str, bytes, bytearray, memoryview)):
        return None
    if isinstance(payload, _Mapping):
        return None
    if isinstance(payload, _IterableABC):
        try:
            return [float(value) for value in payload]
        except (TypeError, ValueError):
            return None
    return None


def _extract_zmetrics_components(
    *mappings: _Mapping[str, _Any]
) -> tuple[dict[str, float], list[float] | None]:
    scalars: dict[str, float] = {}
    gradient: list[float] | None = None
    for mapping in mappings:
        for key, value in mapping.items():
            canonical = _CANONICAL_METRIC_NAMES.get(str(key).lower())
            if canonical not in _RELEVANT_ZSPACE_METRICS:
                continue
            if canonical == "gradient":
                seq = _coerce_gradient_sequence(value)
                if seq is not None:
                    gradient = seq
                continue
            try:
                scalars[canonical] = float(value)
            except (TypeError, ValueError):
                continue
    return scalars, gradient


@_dataclass
class ZMetrics:
    """Typed metrics container fed into :class:`ZSpaceTrainer`."""

    speed: float
    memory: float
    stability: float
    gradient: _Optional[_Sequence[float]] = None
    drs: float = 0.0

    @classmethod
    def from_mapping(
        cls,
        mapping: _Mapping[str, _Any],
        *,
        gradient: _Optional[_Sequence[float]] = None,
    ) -> "ZMetrics":
        scalars, gradient_override = _extract_zmetrics_components(mapping)
        grad_source = gradient_override
        if grad_source is None and gradient is not None:
            coerced = _coerce_gradient_sequence(gradient)
            if coerced is not None:
                grad_source = coerced
        gradient_payload = None
        if grad_source:
            gradient_payload = tuple(float(value) for value in grad_source)
        return cls(
            speed=float(scalars.get("speed", 0.0)),
            memory=float(scalars.get("memory", 0.0)),
            stability=float(scalars.get("stability", 0.0)),
            gradient=gradient_payload,
            drs=float(scalars.get("drs", 0.0)),
        )

    @classmethod
    def from_payload(
        cls,
        payload: "ZSpaceInference | ZMetrics | _Mapping[str, _Any]",
        *,
        prefer_applied: bool = True,
    ) -> "ZMetrics":
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, ZSpaceInference):
            return inference_to_zmetrics(payload, prefer_applied=prefer_applied)
        if isinstance(payload, _Mapping):
            return cls.from_mapping(payload)
        raise TypeError("payload must be mapping, ZMetrics or ZSpaceInference")


def inference_to_zmetrics(
    inference: "ZSpaceInference", *, prefer_applied: bool = True
) -> ZMetrics:
    """Convert a :class:`ZSpaceInference` result into :class:`ZMetrics`.

    Args:
        inference: Inference result produced by :class:`ZSpaceInferencePipeline`
            or :func:`infer_with_partials`.
        prefer_applied: When ``True`` (default), prefer values from
            :attr:`ZSpaceInference.applied` when a metric was explicitly
            rewritten by a partial observation. Falling back to
            :attr:`ZSpaceInference.metrics` preserves the decoded baseline.

    Returns:
        A :class:`ZMetrics` instance populated with canonical speed, memory,
        stability, DRS and gradient signals extracted from the inference.
    """

    if not isinstance(inference, ZSpaceInference):
        raise TypeError("inference must be a ZSpaceInference instance")

    metrics_mapping = dict(inference.metrics)
    applied_mapping = dict(inference.applied) if inference.applied else {}

    scalars, gradient_override = _extract_zmetrics_components(
        metrics_mapping, applied_mapping if prefer_applied else {}
    )

    gradient_seq = list(float(entry) for entry in inference.gradient)
    if not gradient_seq:
        gradient_seq = gradient_override or []
    elif gradient_override is not None:
        gradient_seq = gradient_override

    gradient_payload: _Optional[_Sequence[float]]
    gradient_payload = tuple(gradient_seq) if gradient_seq else None

    return ZMetrics(
        speed=float(scalars.get("speed", 0.0)),
        memory=float(scalars.get("memory", 0.0)),
        stability=float(scalars.get("stability", 0.0)),
        gradient=gradient_payload,
        drs=float(scalars.get("drs", 0.0)),
    )


def ensure_zmetrics(
    payload: "ZSpaceInference | ZMetrics | _Mapping[str, _Any]",
    *,
    prefer_applied: bool = True,
) -> ZMetrics:
    """Normalize any supported payload into a :class:`ZMetrics` instance."""

    return ZMetrics.from_payload(payload, prefer_applied=prefer_applied)


def _clone_volume(volume: _Sequence[_Sequence[_Sequence[float]]]) -> _List[_List[_List[float]]]:
    return [[list(row) for row in slice_] for slice_ in volume]


def _coerce_slice(
    data: _Sequence[_Sequence[float]] | _Any,
    height: _Optional[int] = None,
    width: _Optional[int] = None,
) -> _List[_List[float]]:
    if hasattr(data, "tolist"):
        data = data.tolist()
    if not isinstance(data, _Sequence):
        raise TypeError("slice must be a sequence of rows")
    rows_seq = list(data)
    rows: _List[_List[float]] = []
    if height is None:
        height = len(rows_seq)
    if len(rows_seq) != height:
        raise ValueError(f"expected {height} rows, received {len(rows_seq)}")
    for row in rows_seq:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, _Sequence):
            raise TypeError("slice rows must be sequences")
        values = [float(v) for v in row]
        if width is None:
            width = len(values)
        if len(values) != width:
            raise ValueError(f"expected row width {width}, received {len(values)}")
        rows.append(values)
    return rows


def _coerce_volume(
    volume: _Sequence[_Sequence[_Sequence[float]]],
    depth: int,
    height: int,
    width: int,
) -> _List[_List[_List[float]]]:
    if hasattr(volume, "tolist"):
        volume = volume.tolist()  # type: ignore[assignment]
    if len(volume) != depth:
        raise ValueError(f"expected {depth} slices, received {len(volume)}")
    slices: _List[_List[_List[float]]] = []
    for slice_data in volume:
        slices.append(_coerce_slice(slice_data, height, width))
    return slices


def _spectral_window(name: str | None, depth: int) -> _List[float]:
    if depth <= 0:
        return []
    if name is None:
        return [1.0] * depth
    key = name.lower()
    if key == "hann":
        return [0.5 - 0.5 * _math.cos(2.0 * _math.pi * n / max(1, depth - 1)) for n in range(depth)]
    if key == "hamming":
        return [0.54 - 0.46 * _math.cos(2.0 * _math.pi * n / max(1, depth - 1)) for n in range(depth)]
    if key == "blackman":
        return [
            0.42
            - 0.5 * _math.cos(2.0 * _math.pi * n / max(1, depth - 1))
            + 0.08 * _math.cos(4.0 * _math.pi * n / max(1, depth - 1))
            for n in range(depth)
        ]
    if key == "gaussian":
        centre = 0.5 * (depth - 1)
        sigma = max(depth * 0.17, 1.0)
        return [
            _math.exp(-0.5 * ((n - centre) / sigma) ** 2)
            for n in range(depth)
        ]
    raise ValueError(f"unknown spectral window '{name}'")


def _blend_volumes(
    current: _Sequence[_Sequence[_Sequence[float]]],
    update: _Sequence[_Sequence[_Sequence[float]]],
    alpha: float,
) -> _List[_List[_List[float]]]:
    blended: _List[_List[_List[float]]] = []
    for cur_slice, upd_slice in zip(current, update):
        upd_rows = _coerce_slice(upd_slice)
        width = len(upd_rows[0]) if upd_rows else None
        cur_rows = _coerce_slice(cur_slice, len(upd_rows), width)
        rows: _List[_List[float]] = []
        for cur_row, upd_row in zip(cur_rows, upd_rows):
            rows.append([
                (1.0 - alpha) * cur_val + alpha * upd_val
                for cur_val, upd_val in zip(cur_row, upd_row)
            ])
        blended.append(rows)
    return blended


class TemporalResonanceBuffer:
    """Maintains an exponential moving average over recent Z-space volumes."""

    def __init__(self, capacity: int = 4, alpha: float = 0.2) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._history: _deque[_List[_List[_List[float]]]] = _deque(maxlen=capacity)
        self._alpha = max(1e-6, min(1.0, float(alpha)))
        self._ema: _Optional[_List[_List[_List[float]]]] = None

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def capacity(self) -> int:
        maxlen = self._history.maxlen
        return maxlen if maxlen is not None else len(self._history)

    def update(self, volume: _Sequence[_Sequence[_Sequence[float]]]) -> _List[_List[_List[float]]]:
        snapshot = _clone_volume(volume)
        self._history.append(snapshot)
        if self._ema is None:
            self._ema = snapshot
        else:
            self._ema = _blend_volumes(self._ema, snapshot, self._alpha)
        return _clone_volume(self._ema)

    def state(self) -> _Optional[_List[_List[_List[float]]]]:
        if self._ema is not None:
            return _clone_volume(self._ema)
        if self._history:
            return _clone_volume(self._history[-1])
        return None

    def history(self) -> _List[_List[_List[_List[float]]]]:
        return [_clone_volume(volume) for volume in self._history]

    def state_dict(self) -> _Dict[str, _Any]:
        return {
            "capacity": self.capacity,
            "alpha": self._alpha,
            "history": self.history(),
            "ema": _clone_volume(self._ema) if self._ema is not None else None,
        }

    def load_state_dict(self, state: _Mapping[str, _Any]) -> None:
        if not isinstance(state, _Mapping):
            raise TypeError("state must be a mapping")
        capacity = int(state.get("capacity", self.capacity) or self.capacity)
        if capacity <= 0:
            raise ValueError("state capacity must be positive")
        self._alpha = max(1e-6, min(1.0, float(state.get("alpha", self._alpha))))
        self._history = _deque(maxlen=capacity)
        history = state.get("history", [])
        if history:
            if not isinstance(history, _Sequence):
                raise TypeError("history must be a sequence of volumes")
            for volume in history:
                self._history.append(_clone_volume(volume))
        ema = state.get("ema")
        self._ema = _clone_volume(ema) if ema is not None else None


@_dataclass
class SliceProfile:
    mean: float
    std: float
    energy: float


class SpiralTorchVision:
    """Minimal Python orchestrator for SpiralTorchVision pipelines."""

    def __init__(
        self,
        depth: int,
        height: int,
        width: int,
        *,
        alpha: float = 0.2,
        window: str | None = "hann",
        temporal: int = 4,
    ) -> None:
        if depth <= 0 or height <= 0 or width <= 0:
            raise ValueError("depth, height, and width must be positive")
        self.depth = depth
        self.height = height
        self.width = width
        self._alpha = max(1e-6, min(1.0, float(alpha)))
        self._window_name = window
        self._window = _spectral_window(window, depth)
        self._buffer_capacity = max(1, int(temporal))
        self._volume: _List[_List[_List[float]]] = [
            [[0.0 for _ in range(width)] for _ in range(height)]
            for _ in range(depth)
        ]
        self._buffer = TemporalResonanceBuffer(capacity=self._buffer_capacity, alpha=self._alpha)

    @property
    def volume(self) -> _List[_List[_List[float]]]:
        return _clone_volume(self._volume)

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def temporal_capacity(self) -> int:
        return self._buffer_capacity

    @property
    def temporal_state(self) -> _Optional[_List[_List[_List[float]]]]:
        return self._buffer.state()

    @property
    def window(self) -> _List[float]:
        return list(self._window)

    def reset(self) -> None:
        for slice_ in self._volume:
            for row in slice_:
                for idx in range(len(row)):
                    row[idx] = 0.0
        self._buffer = TemporalResonanceBuffer(capacity=self._buffer_capacity, alpha=self._alpha)

    def update_window(self, window: str | _Sequence[float] | None) -> None:
        if window is None or isinstance(window, str):
            self._window_name = window
            self._window = _spectral_window(window, self.depth)
            if not self._window:
                self._window = [1.0] * self.depth
            return
        values = [float(v) for v in window]
        if values and len(values) != self.depth:
            raise ValueError(
                f"expected window with {self.depth} coefficients, received {len(values)}"
            )
        if not values:
            values = [1.0] * self.depth
        self._window_name = None
        self._window = values

    def accumulate(self, volume: _Sequence[_Sequence[_Sequence[float]]], weight: float = 1.0) -> None:
        if hasattr(volume, "tolist"):
            volume = volume.tolist()
        if len(volume) != self.depth:
            raise ValueError(f"expected {self.depth} slices, received {len(volume)}")
        w = max(0.0, float(weight))
        alpha = self._alpha * (w if w else 1.0)
        for idx, slice_data in enumerate(volume):
            rows = _coerce_slice(slice_data, self.height, self.width)
            for r_idx, row in enumerate(rows):
                target_row = self._volume[idx][r_idx]
                for c_idx, value in enumerate(row):
                    target_row[c_idx] = (1.0 - alpha) * target_row[c_idx] + alpha * value
        self._buffer.update(self._volume)

    def accumulate_slices(self, slices: _Sequence[_Sequence[_Sequence[float]]]) -> None:
        self.accumulate(slices)

    def accumulate_sequence(
        self,
        frames: _Iterable[_Sequence[_Sequence[_Sequence[float]]]],
        weights: _Optional[_Sequence[float]] = None,
    ) -> None:
        if weights is None:
            for frame in frames:
                self.accumulate(frame)
            return
        for frame, weight in zip(frames, weights):
            self.accumulate(frame, weight)

    def project(self, *, normalise: bool = True) -> _List[_List[float]]:
        window = self._window or [1.0] * self.depth
        if not window:
            window = [1.0] * self.depth
        total: _List[_List[float]] = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        weight_sum = 0.0
        for coeff, slice_ in zip(window, self._volume):
            if coeff == 0.0:
                continue
            weight_sum += coeff
            for r_idx, row in enumerate(slice_):
                target = total[r_idx]
                for c_idx, value in enumerate(row):
                    target[c_idx] += coeff * value
        if normalise and weight_sum:
            inv = 1.0 / weight_sum
            for row in total:
                for idx in range(len(row)):
                    row[idx] *= inv
        return total

    def volume_energy(self) -> float:
        acc = 0.0
        for slice_ in self._volume:
            for row in slice_:
                for value in row:
                    acc += value * value
        return acc

    def slice_profile(self) -> _List[SliceProfile]:
        profiles: _List[SliceProfile] = []
        for slice_ in self._volume:
            flat = [value for row in slice_ for value in row]
            if not flat:
                profiles.append(SliceProfile(0.0, 0.0, 0.0))
                continue
            mean = sum(flat) / len(flat)
            var = sum((value - mean) ** 2 for value in flat) / len(flat)
            energy = sum(value * value for value in flat) / len(flat)
            profiles.append(SliceProfile(mean, _math.sqrt(var), energy))
        return profiles

    def snapshot(self) -> _Dict[str, _Any]:
        return {
            "volume": self.volume,
            "profiles": self.slice_profile(),
            "energy": self.volume_energy(),
            "temporal": self._buffer.state(),
        }

    def state_dict(self) -> _Dict[str, _Any]:
        return {
            "depth": self.depth,
            "height": self.height,
            "width": self.width,
            "alpha": self._alpha,
            "window": list(self._window),
            "window_name": self._window_name,
            "buffer": self._buffer.state_dict(),
            "volume": self.volume,
        }

    def load_state_dict(self, state: _Mapping[str, _Any], *, strict: bool = True) -> None:
        if not isinstance(state, _Mapping):
            raise TypeError("state must be a mapping")
        depth = int(state.get("depth", self.depth))
        height = int(state.get("height", self.height))
        width = int(state.get("width", self.width))
        if strict and (depth != self.depth or height != self.height or width != self.width):
            raise ValueError("state dimensions do not match the vision volume")
        alpha = float(state.get("alpha", self._alpha))
        self.depth = depth
        self.height = height
        self.width = width
        self._alpha = max(1e-6, min(1.0, alpha))
        buffer_state = state.get("buffer")
        capacity = self._buffer_capacity
        if isinstance(buffer_state, _Mapping):
            capacity = int(buffer_state.get("capacity", capacity) or capacity)
        if capacity <= 0:
            capacity = self._buffer_capacity
        if capacity != self._buffer_capacity:
            self._buffer_capacity = capacity
            self._buffer = TemporalResonanceBuffer(capacity=capacity, alpha=self._alpha)
        if isinstance(buffer_state, _Mapping):
            self._buffer.load_state_dict(buffer_state)
        else:
            self._buffer._alpha = self._alpha  # keep alpha in sync when no buffer state is supplied
        window_name = state.get("window_name")
        window_values = state.get("window")
        if window_name is not None:
            self.update_window(window_name)
        elif isinstance(window_values, _Sequence):
            self.update_window(list(window_values))
        volume_data = state.get("volume")
        if volume_data is not None:
            coerced = _coerce_volume(volume_data, self.depth, self.height, self.width)
            self._volume = coerced
        temporal_state = state.get("temporal")
        if temporal_state is not None:
            current = self._buffer.state_dict()
            current["ema"] = temporal_state
            self._buffer.load_state_dict(current)


class ZSpaceTrainer:
    """Lightweight Adam optimiser operating on a Z vector."""

    def __init__(
        self,
        z_dim: int = 4,
        *,
        alpha: float = 0.35,
        lam_speed: float = 0.5,
        lam_mem: float = 0.3,
        lam_stab: float = 0.2,
        lam_frac: float = 0.1,
        lam_drs: float = 0.0,
        lr: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        if z_dim <= 0:
            raise ValueError("z_dim must be positive")
        self._z: _List[float] = [0.0] * z_dim
        self._alpha = max(1e-6, float(alpha))
        self._lam = (float(lam_speed), float(lam_mem), float(lam_stab), float(lam_frac), float(lam_drs))
        self._lr = float(lr)
        self._beta1 = float(beta1)
        self._beta2 = float(beta2)
        self._eps = float(eps)
        self._m: _List[float] = [0.0] * z_dim
        self._v: _List[float] = [0.0] * z_dim
        self._t = 0

    @property
    def state(self) -> _List[float]:
        return list(self._z)

    def reset(self) -> None:
        for arr in (self._z, self._m, self._v):
            for idx in range(len(arr)):
                arr[idx] = 0.0
        self._t = 0

    def state_dict(self) -> _Dict[str, _Any]:
        return {
            "z": list(self._z),
            "moment": list(self._m),
            "velocity": list(self._v),
            "step": self._t,
            "hyperparams": {
                "alpha": self._alpha,
                "lambda": self._lam,
                "lr": self._lr,
                "beta1": self._beta1,
                "beta2": self._beta2,
                "eps": self._eps,
            },
        }

    def load_state_dict(self, state: _Mapping[str, _Any], *, strict: bool = True) -> None:
        if not isinstance(state, _Mapping):
            raise TypeError("state must be a mapping")
        z = state.get("z")
        moment = state.get("moment")
        velocity = state.get("velocity")
        if z is None or moment is None or velocity is None:
            if strict:
                missing = [
                    key
                    for key, value in (("z", z), ("moment", moment), ("velocity", velocity))
                    if value is None
                ]
                raise KeyError(f"missing keys in state: {missing}")
            z = z or self._z
            moment = moment or self._m
            velocity = velocity or self._v
        self._assign_vector(self._z, z, strict)
        self._assign_vector(self._m, moment, strict)
        self._assign_vector(self._v, velocity, strict)
        self._t = int(state.get("step", self._t))
        hyper = state.get("hyperparams")
        if isinstance(hyper, _Mapping):
            alpha = float(hyper.get("alpha", self._alpha))
            self._alpha = max(1e-6, alpha)
            lam = hyper.get("lambda")
            if isinstance(lam, _Sequence) and len(lam) == 5:
                self._lam = tuple(float(value) for value in lam)
            self._lr = float(hyper.get("lr", self._lr))
            self._beta1 = float(hyper.get("beta1", self._beta1))
            self._beta2 = float(hyper.get("beta2", self._beta2))
            self._eps = float(hyper.get("eps", self._eps))

    def _assign_vector(self, target: _MutableSequence[float], values: _Any, strict: bool) -> None:
        data = [float(v) for v in values]
        if len(data) != len(target):
            if strict:
                raise ValueError(
                    f"expected vector of length {len(target)}, received {len(data)}"
                )
            if len(data) < len(target):
                data.extend(0.0 for _ in range(len(target) - len(data)))
            else:
                data = data[: len(target)]
        for idx, value in enumerate(data):
            target[idx] = value

    def step_batch(
        self,
        metrics: _Iterable[_Mapping[str, float] | ZMetrics],
    ) -> _List[float]:
        losses: _List[float] = []
        for sample in metrics:
            losses.append(self.step(sample))
        return losses

    def _rfft(self, values: _Sequence[float]) -> _List[complex]:
        n = len(values)
        if n == 0:
            return []
        freq: _List[complex] = []
        for k in range(n // 2 + 1):
            total = 0.0j
            for t, val in enumerate(values):
                angle = -2.0 * _math.pi * k * t / max(1, n)
                total += complex(val, 0.0) * _cmath.exp(1j * angle)
            freq.append(total)
        return freq

    def _frac_reg(self, values: _Sequence[float]) -> float:
        spectrum = self._rfft(values)
        n = len(spectrum)
        if n <= 1:
            return 0.0
        acc = 0.0
        for idx, coeff in enumerate(spectrum):
            omega = idx / max(1, n - 1)
            weight = omega ** (2.0 * self._alpha)
            acc += weight * abs(coeff) ** 2
        return acc / n

    def _frac_grad(self) -> _List[float]:
        grad: _List[float] = []
        base = self._frac_reg(self._z)
        step = 1e-4
        for i in range(len(self._z)):
            original = self._z[i]
            self._z[i] = original + step
            plus = self._frac_reg(self._z)
            self._z[i] = original - step
            minus = self._frac_reg(self._z)
            self._z[i] = original
            grad.append((plus - minus) / (2.0 * step))
        scale = max(1.0, max(abs(g) for g in grad) if grad else 1.0)
        return [g / scale for g in grad]

    def _normalise(self, value: float) -> float:
        return _math.tanh(value)

    def _normalise_gradient(self, grad: _Sequence[float] | None) -> _List[float]:
        if not grad:
            return [0.0] * len(self._z)
        grad_list = list(grad)
        if len(grad_list) == len(self._z):
            return [self._normalise(g) for g in grad_list]
        out: _List[float] = []
        for idx in range(len(self._z)):
            out.append(self._normalise(grad_list[idx % len(grad_list)]))
        return out

    def _adam_update(self, grad: _Sequence[float]) -> None:
        self._t += 1
        for i, g in enumerate(grad):
            self._m[i] = self._beta1 * self._m[i] + (1.0 - self._beta1) * g
            self._v[i] = self._beta2 * self._v[i] + (1.0 - self._beta2) * (g * g)
            m_hat = self._m[i] / (1.0 - self._beta1 ** self._t)
            v_hat = self._v[i] / (1.0 - self._beta2 ** self._t)
            self._z[i] -= self._lr * m_hat / (_math.sqrt(v_hat) + self._eps)

    def step(
        self,
        metrics: _Mapping[str, float] | ZMetrics | "ZSpaceInference",
        *,
        prefer_applied: bool = True,
    ) -> float:
        normalized = ZMetrics.from_payload(
            metrics, prefer_applied=prefer_applied
        )

        speed = float(normalized.speed)
        memory = float(normalized.memory)
        stability = float(normalized.stability)
        gradient = normalized.gradient
        drs_signal = float(normalized.drs)
        lam_speed, lam_mem, lam_stab, lam_frac, lam_drs = self._lam
        penalty = (
            lam_speed * self._normalise(speed)
            + lam_mem * self._normalise(memory)
            + lam_stab * self._normalise(stability)
        )
        if lam_drs:
            penalty += lam_drs * self._normalise(drs_signal)
        frac_reg = self._frac_reg(self._z)
        loss = penalty + lam_frac * frac_reg
        grad_metric = self._normalise_gradient(gradient)
        frac_grad = self._frac_grad()
        total_grad = [grad_metric[idx] + lam_frac * frac_grad[idx] for idx in range(len(self._z))]
        self._adam_update(total_grad)
        return loss


def step_many(
    trainer: ZSpaceTrainer,
    samples: _Iterable[_Mapping[str, float] | ZMetrics | ZSpaceInference],
) -> _List[float]:
    for metrics in samples:
        trainer.step(metrics)
    return trainer.state


def stream_zspace_training(
    trainer: ZSpaceTrainer,
    samples: _Iterable[_Mapping[str, float] | ZMetrics | ZSpaceInference],
    *,
    on_step: _Optional[_Callable[[int, _List[float], float], None]] = None,
) -> _List[float]:
    losses: _List[float] = []
    for index, metrics in enumerate(samples):
        loss = trainer.step(metrics)
        losses.append(loss)
        if on_step is not None:
            on_step(index, trainer.state, loss)
    return losses


def _matrix_summary(matrix: _Sequence[_Sequence[float]]) -> _Dict[str, float]:
    flat = [float(value) for row in matrix for value in row]
    if not flat:
        return {"l1": 0.0, "l2": 0.0, "linf": 0.0, "mean": 0.0}
    l1 = sum(abs(value) for value in flat)
    l2 = _math.sqrt(sum(value * value for value in flat))
    linf = max(abs(value) for value in flat)
    mean = sum(flat) / len(flat)
    return {"l1": l1, "l2": l2, "linf": linf, "mean": mean}


def _coerce_matrix(matrix: _Any, height: int, width: int) -> _List[_List[float]]:
    rows = _coerce_slice(matrix, height, width)
    return rows


class _ForwardingModule(_types.ModuleType):
    """Module stub that forwards attribute lookups to the Rust backend."""

    def __init__(self, name: str, doc: str, key: str) -> None:
        super().__init__(name, doc)
        self.__dict__["_forward_key"] = key

    @property
    def _forward_key(self) -> str:
        return self.__dict__["_forward_key"]

    def __getattr__(self, attr: str) -> _Any:
        if attr.startswith("_"):
            raise AttributeError(f"module '{self.__name__}' has no attribute '{attr}'")

        # Prefer already-exposed globals so top-level mirrors stay consistent.
        if attr in globals():
            value = globals()[attr]
            setattr(self, attr, value)
            _register_module_export(self, attr)
            return value

        hints = _FORWARDING_HINTS.get(self._forward_key, {})
        candidates: list[str] = []
        aliases = hints.get(attr)
        if aliases:
            candidates.extend(aliases)

        namespace_parts = self._forward_key.split(".")
        suffix = namespace_parts[-1]
        flat_suffix = "_".join(namespace_parts)
        candidates.extend(
            [
                attr,
                f"{suffix}_{attr}",
                f"{suffix}_{attr.lower()}",
                f"{flat_suffix}_{attr}",
                f"{flat_suffix}_{attr.lower()}",
            ]
        )

        for candidate in dict.fromkeys(candidates):
            value = _resolve_rs_attr(candidate)
            if value is not None:
                setattr(self, attr, value)
                _register_module_export(self, attr)
                return value

        raise AttributeError(f"module '{self.__name__}' has no attribute '{attr}'")

    def __dir__(self) -> list[str]:
        exported = set(getattr(self, "__all__", ()))
        exported.update(super().__dir__())
        hints = _FORWARDING_HINTS.get(self._forward_key, {})
        exported.update(hints.keys())
        suffix = self._forward_key.split(".")[-1] + "_"
        flat_suffix = "_".join(self._forward_key.split(".")) + "_"
        if _rs is not None:
            for name in dir(_rs):
                trimmed = None
                if name.startswith(suffix):
                    trimmed = name[len(suffix):]
                elif name.startswith(flat_suffix):
                    trimmed = name[len(flat_suffix):]
                if not trimmed:
                    continue
                trimmed = _RENAMED_EXPORTS.get(trimmed, trimmed)
                exported.add(trimmed)
        return sorted(exported)


def _register_module_export(module: _types.ModuleType, name: str) -> None:
    exported = set(getattr(module, "__all__", ()))
    exported.add(name)
    module.__all__ = sorted(exported)


def _ensure_submodule(name: str, doc: str = "") -> _types.ModuleType:
    """Return or create a synthetic child module without touching the native core."""

    parts = name.split(".")
    fq = __name__
    parent: _types.ModuleType = sys.modules[__name__]
    for idx, part in enumerate(parts):
        fq = f"{fq}.{part}"
        module = sys.modules.get(fq)
        final = idx == len(parts) - 1
        doc_for_part = doc if final else ""
        if module is None:
            key = ".".join(parts[: idx + 1])
            module = _ForwardingModule(fq, doc_for_part, key)
            sys.modules[fq] = module
        elif doc_for_part and not getattr(module, "__doc__", None):
            module.__doc__ = doc_for_part

        setattr(parent, part, module)
        if idx == 0:
            globals()[part] = module
        parent = module
    return parent


def _expose_from_rs(name: str, *aliases: str) -> None:
    if name in globals():
        return
    for candidate in (name, *aliases):
        value = _resolve_rs_attr(candidate)
        if value is not None:
            globals()[name] = value
            return


def _mirror_into_module(
    name: str,
    members: _Iterable[str] | _Mapping[str, _Iterable[str]],
    *,
    reexport: bool = True,
) -> _types.ModuleType:
    module = _ensure_submodule(name)
    exported: set[str] = set(getattr(module, "__all__", ()))
    items: _Iterable[tuple[str, _Iterable[str]]] \
        = members.items() if isinstance(members, _Mapping) else ((m, ()) for m in members)
    for member, aliases in items:
        value = None
        if reexport:
            _expose_from_rs(member, *aliases)
            value = globals().get(member)
        else:
            if member in globals():
                value = globals()[member]
            if value is None:
                for candidate in (member, *aliases):
                    value = _safe_getattr(_rs, candidate, None)
                    if value is not None:
                        break
        if value is None:
            continue
        if reexport:
            globals()[member] = value
        setattr(module, member, value)
        exported.add(member)
    if exported:
        module.__all__ = sorted(exported)
    return module


for _name, _doc in _PREDECLARED_SUBMODULES:
    _module = _ensure_submodule(_name, _doc)
    if not isinstance(_module, _ForwardingModule):
        _fq = f"{__name__}.{_name}"
        _forward = _ForwardingModule(_fq, getattr(_module, "__doc__", _doc), _name)
        for _key, _value in vars(_module).items():
            if _key in {"__dict__", "__weakref__"}:
                continue
            if _key == "__name__":
                continue
            setattr(_forward, _key, _value)
        sys.modules[_fq] = _forward
        setattr(sys.modules[__name__], _name, _forward)
        globals()[_name] = _forward


_compat_children = {
    "torch": "PyTorch interoperability helpers",
    "jax": "JAX interoperability helpers",
    "tensorflow": "TensorFlow interoperability helpers",
}
for _child, _doc in _compat_children.items():
    _ensure_submodule(f"compat.{_child}", _doc)
_compat_module = globals().get("compat")
if isinstance(_compat_module, _types.ModuleType):
    _compat_exports = set(getattr(_compat_module, "__all__", ()))
    _compat_exports.update(_compat_children.keys())
    _compat_module.__all__ = sorted(_compat_exports)


_mirror_into_module(
    "inference",
    [
        "SafetyViolation","SafetyVerdict","AuditEvent","AuditLog",
        "InferenceResult","InferenceRuntime",
    ],
)


_mirror_into_module(
    "hpo",
    {
        "SearchLoop": ("PySearchLoop",),
    },
)


_mirror_into_module(
    "export",
    {
        "QatObserver": ("PyQatObserver",),
        "QuantizationReport": ("PyQuantizationReport",),
        "StructuredPruningReport": ("PyStructuredPruningReport",),
        "CompressionReport": ("PyCompressionReport",),
        "structured_prune": (),
        "compress_weights": (),
    },
)


_mirror_into_module(
    "nn",
    {
        "Dataset": ("_NnDataset",),
        "DataLoader": ("_NnDataLoader",),
        "DataLoaderIter": ("_NnDataLoaderIter",),
        "ZConv": ("PyZConv",),
        "ZPooling": ("PyZPooling",),
        "from_samples": ("nn_from_samples", "dataset_from_samples"),
    },
)
_mirror_into_module(
    "frac",
    {
        "gl_coeffs_adaptive": ("frac.gl_coeffs_adaptive",),
        "fracdiff_gl_1d": ("frac.fracdiff_gl_1d",),
    },
)
_mirror_into_module(
    "spiral_rl",
    {
        "stAgent": ("PyDqnAgent", "DqnAgent", "StAgent"),
        "PpoAgent": ("PyPpoAgent",),
        "SacAgent": ("PySacAgent",),
    },
)
_mirror_into_module(
    "rec",
    {
        "QueryPlan": ("PyQueryPlan",),
        "RecEpochReport": ("PyRecEpochReport",),
        "Recommender": ("PyRecommender",),
    },
)
_mirror_into_module(
    "telemetry",
    {
        "DashboardMetric": ("PyDashboardMetric",),
        "DashboardEvent": ("PyDashboardEvent",),
        "DashboardFrame": ("PyDashboardFrame",),
        "DashboardRing": ("PyDashboardRing",),
    },
)


_mirror_into_module(
    "compat",
    [
        "capture",
        "share",
    ],
    reexport=False,
)
_mirror_into_module(
    "compat.torch",
    {
        "to_torch": ("compat_to_torch", "to_torch"),
        "from_torch": ("compat_from_torch", "from_torch"),
    },
    reexport=False,
)
_mirror_into_module(
    "compat.jax",
    {
        "to_jax": ("compat_to_jax", "to_jax"),
        "from_jax": ("compat_from_jax", "from_jax"),
    },
    reexport=False,
)
_mirror_into_module(
    "compat.tensorflow",
    {
        "to_tensorflow": ("compat_to_tensorflow", "to_tensorflow"),
        "from_tensorflow": ("compat_from_tensorflow", "from_tensorflow"),
    },
    reexport=False,
)


_mirror_into_module(
    "zspace",
    [
        "ZMetrics",
        "ZSpaceTrainer",
        "z_metrics",
        "step_many",
        "stream_zspace_training",
        "inference_to_zmetrics",
        "ensure_zmetrics",
    ],
    reexport=False,
)
_mirror_into_module(
    "vision",
    [
        "SpiralTorchVision",
        "TemporalResonanceBuffer",
        "SliceProfile",
    ],
    reexport=False,
)
_mirror_into_module(
    "canvas",
    [
        "CanvasTransformer",
        "CanvasSnapshot",
        "apply_vision_update",
    ],
    reexport=False,
)


_mirror_into_module(
    "selfsup",
    {
        "info_nce": ("selfsup.info_nce",),
        "masked_mse": ("selfsup.masked_mse",),
    },
)


_mirror_into_module(
    "spiralk",
    {
        "SpiralKFftPlan": (),
        "MaxwellSpiralKBridge": (),
        "MaxwellSpiralKHint": (),
        "SpiralKContext": (),
        "SpiralKWilsonMetrics": (),
        "SpiralKHeuristicHint": (),
        "wilson_lower_bound": (),
        "should_rewrite": (),
        "synthesize_program": (),
        "rewrite_with_wilson": (),
    },
    reexport=False,
)


_mirror_into_module(
    "planner",
    {
        "RankPlan": (),
        "plan": (),
        "plan_topk": (),
        "describe_device": (),
        "hip_probe": (),
        "generate_plan_batch_ex": (),
    },
    reexport=False,
)


_mirror_into_module(
    "spiralk",
    {
        "FftPlan": (),
        "MaxwellBridge": (),
        "MaxwellHint": (),
        "required_blocks": (),
    },
    reexport=False,
)


class SpiralSession:
    """Lightweight execution context for quick experimentation."""

    backend: str
    seed: int | None
    device: str

    def __init__(self, backend: str = "auto", seed: int | None = None) -> None:
        self.backend = backend
        self.seed = seed
        if backend == "hip":
            init_backend("hip")
            self.device = "hip"
        elif backend == "wgpu":
            self.device = "wgpu"
        else:
            self.device = "cpu"

    def plan_topk(self, rows: int, cols: int, k: int):
        return plan_topk(rows, cols, k, backend=self.backend)

    def close(self) -> None:
        """Release any session-scoped resources (currently a no-op)."""


_EXTRAS.append("SpiralSession")
_EXTRAS.extend(
    [
        "ZSpaceDecoded",
        "ZSpaceInference",
        "ZSpacePosterior",
        "ZSpaceInferenceRuntime",
        "decode_zspace_embedding",
        "infer_from_partial",
        "compile_inference",
        "inference_to_zmetrics",
        "ensure_zmetrics",
    ]
)


for _key, _hint in _FORWARDING_HINTS.items():
    _module = _ensure_submodule(_key)
    if not _hint:
        continue
    _exports = set(getattr(_module, "__all__", ()))
    _exports.update(_hint.keys())
    _module.__all__ = sorted(_exports)


_CORE_EXPORTS = [
    "Tensor","ComplexTensor","OpenCartesianTopos","LanguageWaveEncoder",
    "GradientSummary","Hypergrad","TensorBiome",
    "LinearModel",
    "BarycenterIntermediate","ZSpaceBarycenter",
    "QueryPlan","RecEpochReport","Recommender",
    "stAgent","PpoAgent","SacAgent",
    "DashboardMetric","DashboardEvent","DashboardFrame","DashboardRing",
    "AuditEvent","AuditLog","InferenceResult","InferenceRuntime",
    "SafetyVerdict","SafetyViolation",
    "SearchLoop",
    "QatObserver","QuantizationReport","StructuredPruningReport","CompressionReport",
    "structured_prune","compress_weights",
    "ModuleTrainer","NonLiner","ZConv","ZPooling","ZSpaceTrainer","ZSpaceCoherenceSequencer","PreDiscardTelemetry","PreDiscardPolicy",
    "CoherenceObservation","CoherenceSignature","CoherenceChannelReport","CoherenceDiagnostics","is_swap_invariant",
    "TemporalResonanceBuffer","SpiralTorchVision",
    "CanvasTransformer","CanvasSnapshot","apply_vision_update",
    "ZMetrics","SliceProfile","step_many","stream_zspace_training",
    "info_nce","masked_mse","mean_squared_error",
    "init_backend",
]
for _name in _CORE_EXPORTS:
    _expose_from_rs(_name)


def __getattr__(name: str) -> _Any:
    """Defer missing attributes to the Rust extension module.

    This keeps the Python façade lightweight while still exposing the rich
    surface area implemented in Rust.
    """

    if name.startswith("_"):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    redirect = _RENAMED_EXPORTS.get(name)
    if redirect is not None:
        _expose_from_rs(redirect)
        if redirect in globals():
            return globals()[redirect]
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    _expose_from_rs(name)
    if name in globals():
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    _public = set(__all__)
    if _rs is not None:
        for _name in dir(_rs):
            if _name.startswith("_"):
                continue
            _public.add(_RENAMED_EXPORTS.get(_name, _name))
    return sorted(_public)


_EXPORTED = {
    *_EXTRAS,
    *_CORE_EXPORTS,
    *[n for n in _COMPAT_ALIAS if n in globals()],
    "nn","frac","dataset","linalg","spiral_rl","rec","telemetry","ecosystem",
    "selfsup","export","compat","hpo","inference","zspace","vision","canvas",
    "planner","spiralk",
    "hg","z",
    "__version__",
}
_EXPORTED.update(
    n
    for n in _safe_getattr(_rs, "__all__", ())
    if isinstance(n, str) and not n.startswith("_")
)
__all__ = sorted(_EXPORTED)
