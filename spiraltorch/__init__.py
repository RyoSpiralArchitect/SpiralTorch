"""Development shim for the SpiralTorch Python bindings.

This module lets ``import spiraltorch`` succeed directly from a source
checkout without first installing the wheel.  It delegates to the real
package that lives under ``bindings/st-py`` and improves the error message
when the compiled extension has not been built yet.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Sequence
import importlib
import importlib.machinery
import importlib.util
import math
import random
import pathlib
import sys
import types
import warnings
from typing import Any, NoReturn


_TENSOR_NO_DATA = object()


# Hardmax/spiral consensus constants mirrored from the native implementation.
_GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0
_GOLDEN_RATIO_CONJUGATE = 1.0 / _GOLDEN_RATIO
_GOLDEN_RATIO_BIAS = 1.0 - _GOLDEN_RATIO_CONJUGATE
_SPIRAL_PROJECTOR_RANK = 24
_SPIRAL_PROJECTOR_WEIGHT = 0.75
_SPIRAL_PROJECTOR_RAMANUJAN_ITERS = 6
_SPIRAL_LEECH_PACKING_DENSITY = 0.001_929_574_309_403_922_5


def _ramanujan_pi(iterations: int) -> float:
    """Return the Ramanujan π approximation used by the stub consensus."""

    iterations = max(1, int(iterations))
    prefactor = (2.0 * math.sqrt(2.0)) / 9801.0
    base = 396.0**4
    factor = 1.0
    series_sum = 0.0
    approximation = math.pi

    for k in range(iterations):
        kf = float(k)
        series_sum += factor * (1103.0 + 26390.0 * kf)
        approximation = 1.0 / (prefactor * series_sum)
        if k + 1 == iterations:
            break
        next_k = float(k + 1)
        numerator = (
            (4.0 * next_k - 3.0)
            * (4.0 * next_k - 2.0)
            * (4.0 * next_k - 1.0)
            * (4.0 * next_k)
        )
        denominator = (next_k**4) * base
        factor *= numerator / denominator

    return approximation


_SPIRAL_RAMANUJAN_PI = _ramanujan_pi(_SPIRAL_PROJECTOR_RAMANUJAN_ITERS)
_SPIRAL_RAMANUJAN_RATIO = (
    math.pi / _SPIRAL_RAMANUJAN_PI if _SPIRAL_RAMANUJAN_PI > 1e-12 else 1.0
)
_SPIRAL_RAMANUJAN_DELTA = abs(_SPIRAL_RAMANUJAN_PI - math.pi)
_SPIRAL_LEECH_SCALE = (
    _SPIRAL_PROJECTOR_WEIGHT
    * _SPIRAL_LEECH_PACKING_DENSITY
    * math.sqrt(float(_SPIRAL_PROJECTOR_RANK))
    * _SPIRAL_RAMANUJAN_RATIO
)


def _spiral_softmax_hardmax_consensus_python(
    softmax: Sequence[float],
    hardmax: Sequence[float],
    rows: int,
    cols: int,
) -> tuple[list[float], dict[str, float]]:
    """Blend softmax probabilities with hardmax masks using spiral consensus."""

    expected = rows * cols
    metrics: dict[str, float] = {
        "phi": _GOLDEN_RATIO,
        "phi_conjugate": _GOLDEN_RATIO_CONJUGATE,
        "phi_bias": _GOLDEN_RATIO_BIAS,
        "ramanujan_ratio": _SPIRAL_RAMANUJAN_RATIO,
        "ramanujan_delta": _SPIRAL_RAMANUJAN_DELTA,
        "average_enrichment": 0.0,
        "mean_entropy": 0.0,
        "mean_hardmass": 0.0,
        "spiral_coherence": 0.0,
    }

    if expected == 0 or len(softmax) != expected or len(hardmax) != expected:
        return [0.0] * expected, metrics

    fused = [0.0] * expected
    total_entropy = 0.0
    total_hardmass = 0.0
    total_enrichment = 0.0
    total_coherence = 0.0

    for row in range(rows):
        offset = row * cols
        row_soft = softmax[offset : offset + cols]
        row_hard = hardmax[offset : offset + cols]

        entropy = 0.0
        hardmass = 0.0

        for prob, mask in zip(row_soft, row_hard):
            p = float(prob)
            if p > 0.0:
                entropy -= p * math.log(p)
            hardmass += float(mask) if mask > 0.0 else 0.0

        geodesic = entropy * _SPIRAL_RAMANUJAN_RATIO + hardmass * _GOLDEN_RATIO
        enrichment = _SPIRAL_LEECH_SCALE * geodesic if geodesic > 1e-12 else 0.0
        scale = 1.0 + enrichment

        total_entropy += entropy
        total_hardmass += hardmass
        total_enrichment += enrichment

        entropy_norm = (entropy / (entropy + 1.0)) if entropy > 0.0 else 0.0
        entropy_norm = max(0.0, min(1.0, entropy_norm))
        hardmass_norm = hardmass / cols if cols else 0.0
        hardmass_norm = max(0.0, min(1.0, hardmass_norm))
        enrichment_norm = enrichment / (1.0 + abs(enrichment)) if enrichment != 0.0 else 0.0
        enrichment_norm = max(0.0, min(1.0, enrichment_norm))
        total_coherence += (entropy_norm + hardmass_norm + enrichment_norm) / 3.0

        for index, (prob, mask) in enumerate(zip(row_soft, row_hard)):
            fused_value = (
                _GOLDEN_RATIO_CONJUGATE * float(prob)
                + _GOLDEN_RATIO_BIAS * float(mask)
            )
            fused[offset + index] = scale * fused_value

    if rows > 0:
        inv_rows = 1.0 / rows
        metrics.update(
            {
                "average_enrichment": total_enrichment * inv_rows,
                "mean_entropy": total_entropy * inv_rows,
                "mean_hardmass": total_hardmass * inv_rows,
                "spiral_coherence": total_coherence * inv_rows,
            }
        )

    return fused, metrics


def _tensor_is_sequence(obj: object) -> bool:
    return isinstance(obj, Sequence) and not isinstance(
        obj, (str, bytes, bytearray, memoryview)
    )


def _tensor_is_iterable(obj: object) -> bool:
    return isinstance(obj, Iterable) and not isinstance(
        obj, (str, bytes, bytearray, memoryview)
    )


def _tensor_coerce_index(value: object, label: str) -> int:
    try:
        index = int(value)
    except Exception as exc:  # noqa: BLE001 - provide friendly error context
        raise TypeError(f"Tensor {label} must be an integer, got {value!r}") from exc
    if index < 0:
        raise ValueError(f"Tensor {label} must be non-negative, got {index}")
    return index


def _tensor_coerce_shape(value: object, label: str) -> tuple[int, int]:
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


def _tensor_maybe_shape(value: object) -> tuple[int, int] | None:
    if not _tensor_is_sequence(value):
        return None
    dims = list(value)
    if len(dims) != 2:
        return None
    try:
        return _tensor_coerce_shape(dims, "shape")
    except (TypeError, ValueError):
        return None


def _tensor_normalize_row(row: object, *, allow_empty: bool) -> list[float]:
    tensor_type = globals().get("Tensor")
    if tensor_type is not None and isinstance(row, tensor_type):
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


def _tensor_flatten_data(data: object) -> tuple[int, int, list[float]]:
    tensor_type = globals().get("Tensor")
    if tensor_type is not None and isinstance(data, tensor_type):
        rows, cols = (int(dim) for dim in data.shape())
        nested = data.tolist()
        flat = [float(value) for row in nested for value in row]
        return rows, cols, flat

    if hasattr(data, "tolist") and not _tensor_is_sequence(data):
        return _tensor_flatten_data(data.tolist())

    if _tensor_is_sequence(data):
        items = list(data)
    elif _tensor_is_iterable(data):
        items = list(data)
    else:
        raise TypeError("Tensor data must be an iterable of floats or nested iterables")

    if not items:
        return 0, 0, []

    head = items[0]
    tensor_type = globals().get("Tensor")
    if tensor_type is not None and isinstance(head, tensor_type):
        head = head.tolist()
    elif hasattr(head, "tolist") and not _tensor_is_sequence(head):
        head = head.tolist()

    if _tensor_is_sequence(head) or _tensor_is_iterable(head):
        rows = len(items)
        cols: int | None = None
        flat: list[float] = []
        for row in items:
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
    *args,
    **kwargs,
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
            "Tensor() takes at most 3 positional arguments "
            f"but {len(positional)} were given"
        )

    if data_value is _TENSOR_NO_DATA:
        if rows is None or cols is None:
            raise TypeError("Tensor() requires a shape when data is omitted")
        return rows, cols, _TENSOR_NO_DATA

    inferred_rows, inferred_cols, flat = _tensor_flatten_data(data_value)
    total = len(flat)

    def _infer_missing_dimension(total_elems: int, known: int, *, known_label: str) -> int:
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


_TENSOR_NO_DATA = object()


def _tensor_is_sequence(obj: object) -> bool:
    return isinstance(obj, Sequence) and not isinstance(
        obj, (str, bytes, bytearray, memoryview)
    )


def _tensor_is_iterable(obj: object) -> bool:
    return isinstance(obj, Iterable) and not isinstance(
        obj, (str, bytes, bytearray, memoryview)
    )


def _tensor_coerce_index(value: object, label: str) -> int:
    try:
        index = int(value)
    except Exception as exc:  # noqa: BLE001 - provide friendly error context
        raise TypeError(f"Tensor {label} must be an integer, got {value!r}") from exc
    if index < 0:
        raise ValueError(f"Tensor {label} must be non-negative, got {index}")
    return index


def _tensor_coerce_shape(value: object, label: str) -> tuple[int, int]:
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


def _tensor_maybe_shape(value: object) -> tuple[int, int] | None:
    if not _tensor_is_sequence(value):
        return None
    dims = list(value)
    if len(dims) != 2:
        return None
    try:
        return _tensor_coerce_shape(dims, "shape")
    except (TypeError, ValueError):
        return None


def _tensor_normalize_row(row: object, *, allow_empty: bool) -> list[float]:
    tensor_type = globals().get("Tensor")
    if tensor_type is not None and isinstance(row, tensor_type):
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


def _tensor_flatten_data(data: object) -> tuple[int, int, list[float]]:
    tensor_type = globals().get("Tensor")
    if tensor_type is not None and isinstance(data, tensor_type):
        rows, cols = (int(dim) for dim in data.shape())
        nested = data.tolist()
        flat = [float(value) for row in nested for value in row]
        return rows, cols, flat

    if hasattr(data, "tolist") and not _tensor_is_sequence(data):
        return _tensor_flatten_data(data.tolist())

    if _tensor_is_sequence(data):
        items = list(data)
    elif _tensor_is_iterable(data):
        items = list(data)
    else:
        raise TypeError("Tensor data must be an iterable of floats or nested iterables")

    if not items:
        return 0, 0, []

    head = items[0]
    tensor_type = globals().get("Tensor")
    if tensor_type is not None and isinstance(head, tensor_type):
        head = head.tolist()
    elif hasattr(head, "tolist") and not _tensor_is_sequence(head):
        head = head.tolist()

    if _tensor_is_sequence(head) or _tensor_is_iterable(head):
        rows = len(items)
        cols: int | None = None
        flat: list[float] = []
        for row in items:
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
    *args,
    **kwargs,
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
            "Tensor() takes at most 3 positional arguments "
            f"but {len(positional)} were given"
        )

    if data_value is _TENSOR_NO_DATA:
        if rows is None or cols is None:
            raise TypeError("Tensor() requires a shape when data is omitted")
        return rows, cols, _TENSOR_NO_DATA

    inferred_rows, inferred_cols, flat = _tensor_flatten_data(data_value)
    total = len(flat)

    def _infer_missing_dimension(total_elems: int, known: int, *, known_label: str) -> int:
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


def _load_native_package() -> None:
    package_root = pathlib.Path(__file__).resolve().parent
    crate_root = package_root.parent / "bindings" / "st-py"
    impl_init = crate_root / "spiraltorch" / "__init__.py"

    if not impl_init.exists():
        raise ModuleNotFoundError(
            "spiraltorch",
            name="spiraltorch",
            path=str(impl_init),
        )

    # Ensure Python can locate the compiled extension modules that maturin builds
    # inside ``bindings/st-py`` (e.g. ``spiraltorch/spiraltorch*.so``).
    crate_path = str(crate_root)
    if crate_path not in sys.path:
        sys.path.insert(0, crate_path)

    loader = importlib.machinery.SourceFileLoader(__name__, str(impl_init))
    spec = importlib.util.spec_from_loader(__name__, loader, origin=str(impl_init))
    module = sys.modules[__name__]
    module.__file__ = str(impl_init)
    module.__package__ = __name__
    module.__path__ = [str(impl_init.parent)]
    module.__spec__ = spec

    try:
        loader.exec_module(module)
    except ModuleNotFoundError as exc:
        missing = {"spiraltorch.spiraltorch", "spiraltorch.spiraltorch_native", "spiraltorch_native"}
        if exc.name in missing:
            _install_stub_bindings(module, exc)
            return
        raise
    except Exception as exc:  # pragma: no cover - defensive fallback
        warnings.warn(
            "Failed to load the native SpiralTorch bindings; falling back to the Python stub.",
            RuntimeWarning,
            stacklevel=2,
        )
        placeholder = ModuleNotFoundError("spiraltorch", name="spiraltorch")
        _install_stub_bindings(module, placeholder)
        module.__dict__["__native_import_error__"] = exc


def _install_stub_bindings(module, error: ModuleNotFoundError) -> None:
    warnings.warn(
        "Using SpiralTorch Python stub because the native extension is missing. "
        "Run `maturin develop -m bindings/st-py/Cargo.toml` for the optimized bindings.",
        RuntimeWarning,
        stacklevel=2,
    )

    PY_STUB_MATMUL_INNER_TILE = 64
    PY_STUB_MATMUL_COL_TILE = 64
    PY_STUB_FMA = getattr(math, "fma", None)

    _TENSOR_NO_DATA = object()

    def _tensor_is_sequence(obj: Any) -> bool:
        return isinstance(obj, Sequence) and not isinstance(
            obj, (str, bytes, bytearray, memoryview)
        )

    def _tensor_is_iterable(obj: Any) -> bool:
        return isinstance(obj, Iterable) and not isinstance(
            obj, (str, bytes, bytearray, memoryview)
        )

    def _tensor_coerce_index(value: Any, label: str) -> int:
        try:
            index = int(value)
        except Exception as exc:  # pragma: no cover - align with native errors
            raise TypeError(f"Tensor {label} must be an integer, got {value!r}") from exc
        if index < 0:
            raise ValueError(f"Tensor {label} must be non-negative, got {index}")
        return index

    def _tensor_coerce_shape(value: Any, label: str) -> tuple[int, int]:
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

    def _tensor_maybe_shape(value: Any) -> tuple[int, int] | None:
        if not _tensor_is_sequence(value):
            return None
        dims = list(value)
        if len(dims) != 2:
            return None
        try:
            return _tensor_coerce_shape(dims, "shape")
        except (TypeError, ValueError):
            return None

    def _tensor_normalize_row(row: Any, *, allow_empty: bool) -> list[float]:
        tensor_type = module.__dict__.get("Tensor")
        if tensor_type is not None and isinstance(row, tensor_type):
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

    def _tensor_flatten_data(data: Any) -> tuple[int, int, list[float]]:
        tensor_type = module.__dict__.get("Tensor")
        if tensor_type is not None and isinstance(data, tensor_type):
            rows, cols = (int(dim) for dim in data.shape())
            nested = data.tolist()
            flat = [float(value) for row in nested for value in row]
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
            return 0, 0, []

        head = items[0]
        tensor_type = module.__dict__.get("Tensor")
        if tensor_type is not None and isinstance(head, tensor_type):
            head = head.tolist()
        elif hasattr(head, "tolist") and not _tensor_is_sequence(head):
            head = head.tolist()

        if _tensor_is_sequence(head) or _tensor_is_iterable(head):
            rows = len(items)
            cols: int | None = None
            flat: list[float] = []
            for row in items:
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
        *args: Any, **kwargs: Any
    ) -> tuple[int, int, list[float] | object]:
        data_value = kwargs.pop("data", _TENSOR_NO_DATA)
        shape_value = kwargs.pop("shape", None)
        rows_value = kwargs.pop("rows", None)
        cols_value = kwargs.pop("cols", None)

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(
                f"Tensor() got unexpected keyword arguments: {unexpected}"
            )

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
            maybe_shape = (
                None if rows is not None or cols is not None else _tensor_maybe_shape(candidate)
            )
            if maybe_shape is not None:
                rows, cols = maybe_shape
            else:
                if data_value is not _TENSOR_NO_DATA:
                    raise TypeError("Tensor() got multiple values for data")
                data_value = _TENSOR_NO_DATA if candidate is None else candidate
        elif len(positional) == 2:
            first, second = positional
            maybe_shape = (
                None if rows is not None or cols is not None else _tensor_maybe_shape(first)
            )
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
                        "Tensor rows argument conflicts with shape: "
                        f"{rows} != {inferred_rows}"
                    )
                if cols is not None and cols != inferred_cols:
                    raise ValueError(
                        "Tensor cols argument conflicts with shape: "
                        f"{cols} != {inferred_cols}"
                    )
                rows = inferred_rows
                cols = inferred_cols
        elif len(positional) == 3:
            first, second, third = positional
            inferred_rows = _tensor_coerce_index(first, "rows")
            inferred_cols = _tensor_coerce_index(second, "cols")
            if rows is not None and rows != inferred_rows:
                raise ValueError(
                    "Tensor rows argument conflicts with shape: "
                    f"{rows} != {inferred_rows}"
                )
            if cols is not None and cols != inferred_cols:
                raise ValueError(
                    "Tensor cols argument conflicts with shape: "
                    f"{cols} != {inferred_cols}"
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

        def _infer_missing_dimension(total_elems: int, known: int, *, known_label: str) -> int:
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
            raise TypeError(
                "Tensor() could not determine both rows and cols from the provided data"
            )

        if (rows == 0 or cols == 0) and total != 0:
            raise ValueError(
                f"Tensor shape ({rows}, {cols}) is incompatible with {total} data elements"
            )

        return rows, cols, flat

    try:  # Prefer a NumPy-backed shim when available for better performance.
        import numpy as _np  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        _np = None  # type: ignore

    try:  # pragma: no cover - optional dependency
        from . import _blas as _blas_impl
    except Exception:  # pragma: no cover - best-effort optional import
        _blas_impl = None  # type: ignore

    NUMPY_AVAILABLE = _np is not None
    BLAS_AVAILABLE = bool(_blas_impl and _blas_impl.blas_available())

    def _resolve_stub_backend(
        requested, *, op: str, allow_numpy: bool = True
    ) -> str:
        """Normalize backend hints for the Python stub implementation."""

        if requested is None:
            return "numpy" if allow_numpy and NUMPY_AVAILABLE else "python"

        normalized = str(requested).lower()
        if normalized == "auto":
            return "numpy" if allow_numpy and NUMPY_AVAILABLE else "python"
        if normalized in {"python", "cpu"}:
            return "python"
        if normalized == "numpy":
            if not allow_numpy:
                raise ValueError(f"{op} does not support the NumPy backend")
            if not NUMPY_AVAILABLE:
                raise RuntimeError(
                    f"NumPy backend requested for {op} but NumPy is not installed"
                )
            return "numpy"
        if normalized in {"gpu", "wgpu"}:
            raise RuntimeError(
                f"{op} backend '{requested}' is unavailable in the stub bindings"
            )
        raise ValueError(f"Unsupported backend '{requested}' for {op}")

    _TENSOR_NO_DATA = object()

    def _tensor_is_sequence(obj) -> bool:
        return isinstance(obj, Sequence) and not isinstance(
            obj, (str, bytes, bytearray, memoryview)
        )

    def _tensor_is_iterable(obj) -> bool:
        return isinstance(obj, Iterable) and not isinstance(
            obj, (str, bytes, bytearray, memoryview)
        )

    def _tensor_coerce_index(value, label: str) -> int:
        try:
            index = int(value)
        except Exception as exc:  # noqa: BLE001 - mirror native error surface
            raise TypeError(f"Tensor {label} must be an integer, got {value!r}") from exc
        if index < 0:
            raise ValueError(f"Tensor {label} must be non-negative, got {index}")
        return index

    def _tensor_coerce_shape(value, label: str) -> tuple[int, int]:
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

    def _tensor_maybe_shape(value) -> tuple[int, int] | None:
        if not _tensor_is_sequence(value):
            return None
        dims = list(value)
        if len(dims) != 2:
            return None
        try:
            return _tensor_coerce_shape(dims, "shape")
        except (TypeError, ValueError):
            return None

    def _tensor_normalize_row(row, *, allow_empty: bool) -> list[float]:
        if isinstance(row, Tensor):
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

    def _tensor_flatten_data(data):
        if isinstance(data, Tensor):
            rows, cols = (int(dim) for dim in data.shape())
            nested = data.tolist()
            flat = [float(value) for row in nested for value in row]
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
            return 0, 0, []

        head = items[0]
        if isinstance(head, Tensor):
            head = head.tolist()
        elif hasattr(head, "tolist") and not _tensor_is_sequence(head):
            head = head.tolist()

        if _tensor_is_sequence(head) or _tensor_is_iterable(head):
            rows = len(items)
            cols: int | None = None
            flat: list[float] = []
            for row in items:
                normalized = _tensor_normalize_row(row, allow_empty=True)
                if cols is None:
                    cols = len(normalized)
                elif len(normalized) != cols:
                    raise ValueError("Tensor rows must all share the same length")
                flat.extend(normalized)
            return rows, 0 if cols is None else cols, flat

        flat = [float(value) for value in items]
        return 1, len(flat), flat

    def _normalize_tensor_ctor_args(*args, **kwargs):
        data_value = kwargs.pop("data", _TENSOR_NO_DATA)
        shape_value = kwargs.pop("shape", None)
        rows_value = kwargs.pop("rows", None)
        cols_value = kwargs.pop("cols", None)

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(
                f"Tensor() got unexpected keyword arguments: {unexpected}"
            )

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
            maybe_shape = (
                None if rows is not None or cols is not None else _tensor_maybe_shape(candidate)
            )
            if maybe_shape is not None:
                rows, cols = maybe_shape
                if data_value is not _TENSOR_NO_DATA:
                    raise TypeError("Tensor() got multiple values for data")
                data_value = _TENSOR_NO_DATA
            else:
                if data_value is not _TENSOR_NO_DATA:
                    raise TypeError("Tensor() got multiple values for data")
                data_value = _TENSOR_NO_DATA if candidate is None else candidate
        elif len(positional) == 2:
            first, second = positional
            maybe_shape = (
                None if rows is not None or cols is not None else _tensor_maybe_shape(first)
            )
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

        def _infer_missing_dimension(total_elems: int, known: int, *, known_label: str) -> int:
            """Derive the complementary dimension from a known axis length."""

            if known == 0:
                if total_elems != 0:
                    raise ValueError(
                        f"Tensor data of length {total_elems} cannot fill ({known}) {known_label}"
                    )
                return 0
            if total_elems % known != 0:
                raise ValueError(
                    f"Tensor data of length {total_elems} is incompatible with "
                    f"{known_label}={known}"
                )
            return total_elems // known

        if rows is None and cols is None:
            rows = inferred_rows
            cols = inferred_cols
        elif rows is None:
            rows = _infer_missing_dimension(total, cols, known_label="cols")
        elif cols is None:
            cols = _infer_missing_dimension(total, rows, known_label="rows")

        if rows * cols != total:
            raise ValueError(
                f"Tensor data of length {total} cannot fill a {rows}x{cols} tensor"
            )

        return rows, cols, flat

    class _ShapeView(tuple):
        def __new__(cls, tensor: "Tensor", getter):
            rows, cols = getter(tensor)
            obj = super().__new__(cls, (rows, cols))
            obj._tensor = tensor
            obj._getter = getter
            return obj

        def __call__(self) -> tuple[int, int]:
            return self._getter(self._tensor)


    class _ShapeDescriptor:
        __slots__ = ("_func", "__doc__")

        def __init__(self, func):
            self._func = func
            self.__doc__ = getattr(func, "__doc__", None)

        def __get__(self, instance, owner):
            if instance is None:
                return self._func
            return _ShapeView(instance, self._func)


    _UNSET = object()

    class Tensor:
        """Featureful stand-in for the Rust ``Tensor`` exposed by the stub bindings."""

        __slots__ = ("_rows", "_cols", "_data", "_backend")

        #: str: Message guiding users to enable DLPack interoperability.
        DLPACK_UNAVAILABLE_MESSAGE = (
            "DLPack interoperability requires NumPy support in the stub Tensor backend."
        )
        @staticmethod
        def _resolve_backend(label: str | None, *, allow_gpu: bool = False) -> str:
            """Normalize backend labels to the internal python/numpy choices."""

            if label is None:
                return "numpy" if NUMPY_AVAILABLE else "python"

            normalized = str(label).lower()
            if normalized == "auto":
                return "numpy" if NUMPY_AVAILABLE else "python"
            if normalized in {"numpy"}:
                if not NUMPY_AVAILABLE:
                    raise RuntimeError("NumPy backend requested but NumPy is not installed")
                return "numpy"
            if normalized in {"python", "cpu"}:
                return "python"
            if allow_gpu and normalized in {"wgpu", "gpu"}:
                return "numpy" if NUMPY_AVAILABLE else "python"
            raise ValueError(
                "backend must be one of 'auto', 'numpy', 'python', 'cpu', or None"
            )

        def __init__(
            self,
            rows=_UNSET,
            cols=_UNSET,
            data=_UNSET,
            *args,
            backend: str | None = None,
            **kwargs,
        ):
            backend_hint = backend
            if "backend" in kwargs:
                raise TypeError("Tensor() got multiple values for keyword argument 'backend'")
            positional_head: list[Any] = []
            explicit_pairs = (("rows", rows), ("cols", cols), ("data", data))
            for name, value in explicit_pairs:
                if value is _UNSET:
                    break
                positional_head.append(value)

            ctor_kwargs = dict(kwargs)
            for name, value in explicit_pairs[len(positional_head) :]:
                if value is not _UNSET:
                    ctor_kwargs[name] = value

            ctor_args = (*positional_head, *args)

            rows, cols, payload = _normalize_tensor_ctor_args(*ctor_args, **ctor_kwargs)
            if backend_hint is not None and backend_hint not in {"numpy", "python", "blas"}:
                raise ValueError("backend must be 'numpy', 'python', 'blas', or None")
            if backend_hint == "numpy" and not NUMPY_AVAILABLE:
                raise RuntimeError("NumPy backend requested but NumPy is not installed")
            if backend_hint == "blas" and not BLAS_AVAILABLE:
                raise RuntimeError("BLAS backend requested but no BLAS library was detected")
            rows = int(rows)
            cols = int(cols)

            if payload is _TENSOR_NO_DATA:
                total = rows * cols
                canonical = array("d") if total == 0 else array("d", [0.0]) * total
            elif isinstance(payload, array) and payload.typecode == "d":
                canonical = array("d", payload)
            else:
                canonical = array("d", (float(x) for x in payload))
            if rows * cols != len(canonical):
                raise ValueError("data length does not match matrix dimensions")

            self._rows = rows
            self._cols = cols

            preferred_backend = backend_hint or (
                "blas" if BLAS_AVAILABLE else ("numpy" if NUMPY_AVAILABLE else "python")
            )
            if preferred_backend == "numpy":
                arr = (
                    _np.frombuffer(canonical, dtype=_np.float64, count=rows * cols)
                    .reshape(rows, cols)
                    .copy()
                )
                self._data = arr
                self._backend = "numpy"
            else:
                self._data = canonical
                self._backend = "blas" if preferred_backend == "blas" else "python"

        # noqa: D401 - mirror real signature from the native extension
        def matmul(self, other: "Tensor", *, backend: str | None = None):
            if not isinstance(other, Tensor):
                raise TypeError("matmul expects another Tensor instance")
            if self._cols != other._rows:
                raise ValueError("inner dimensions do not match for matmul")

            if backend is not None and backend not in {"numpy", "python", "blas"}:
                raise ValueError("backend must be 'numpy', 'python', 'blas', or None")
            target_backend = backend or (
                "blas" if BLAS_AVAILABLE else ("numpy" if NUMPY_AVAILABLE else "python")
            )

            if target_backend == "numpy":
                if not NUMPY_AVAILABLE:
                    raise RuntimeError("NumPy backend requested but NumPy is not installed")
                return self._matmul_numpy(other)
            if target_backend == "blas":
                if not BLAS_AVAILABLE:
                    raise RuntimeError("BLAS backend requested but no BLAS library was detected")
                return self._matmul_blas(other)
            return self._matmul_python(other)

        def _matmul_blas(self, other: "Tensor") -> "Tensor":
            rows, cols, inner = self._rows, other._cols, self._cols
            if rows == 0 or cols == 0:
                return Tensor._from_python_array(rows, cols, array("d"), backend="blas")
            if inner == 0:
                return Tensor._from_python_array(
                    rows, cols, array("d", [0.0]) * (rows * cols), backend="blas"
                )

            out = array("d", [0.0]) * (rows * cols)
            left = self._row_major_python()
            right = other._row_major_python()
            _blas_impl.dgemm(rows, cols, inner, left, right, out)  # type: ignore[union-attr]
            return Tensor._from_python_array(rows, cols, out, backend="blas")
        def row_softmax(self, *, backend: str | None = None) -> "Tensor":
            """Compute the row-wise softmax of the tensor."""

            target_backend = self._resolve_backend(backend)
            cls = type(self)

            if target_backend == "numpy":
                matrix = self._to_numpy(copy=False)
                if matrix.size == 0 or matrix.shape[1] == 0:
                    return cls._from_numpy_array(matrix.copy())

                rows, cols = matrix.shape
                soft = _np.zeros((rows, cols), dtype=_np.float64)

                for r in range(rows):
                    row = matrix[r]
                    finite_mask = ~_np.isnan(row)
                    if not finite_mask.any():
                        continue

                    finite_indices = _np.flatnonzero(finite_mask)
                    finite_values = row[finite_mask]
                    max_value = float(finite_values.max())
                    max_is_inf = math.isinf(max_value)

                    if max_is_inf:
                        shifted = _np.where(finite_values == max_value, 1.0, 0.0)
                    else:
                        shifted = _np.exp(finite_values - max_value)

                    denom = float(shifted.sum())
                    if math.isfinite(denom) and denom > 0.0:
                        soft_values = shifted / denom
                    else:
                        soft_values = _np.zeros_like(shifted)

                    soft[r, finite_indices] = soft_values

                return cls._from_numpy_array(soft)

            rows, cols = self._rows, self._cols
            total = rows * cols
            buffer = array("d", [0.0]) * total if total else array("d")
            if cols == 0 or total == 0:
                return cls._from_python_array(rows, cols, buffer)

            flat = self._row_major_python()
            for r in range(rows):
                offset = r * cols
                row_slice = [float(flat[offset + c]) for c in range(cols)]
                finite_indices: list[int] = []
                max_value = -math.inf
                for idx, value in enumerate(row_slice):
                    if math.isnan(value):
                        continue
                    finite_indices.append(idx)
                    if value > max_value:
                        max_value = value

                if not finite_indices:
                    continue

                accum = 0.0
                exps = [0.0] * cols
                max_is_inf = math.isinf(max_value)
                for idx in finite_indices:
                    value = row_slice[idx]
                    shifted = 1.0 if max_is_inf and value == max_value else math.exp(value - max_value)
                    exps[idx] = shifted
                    accum += shifted

                inv = 1.0 / accum if accum > 0.0 and math.isfinite(accum) else 0.0
                for idx in finite_indices:
                    buffer[offset + idx] = exps[idx] * inv if inv > 0.0 else 0.0

            return cls._from_python_array(rows, cols, buffer)

        def _compute_row_softmax_hardmax_python(self) -> tuple[array, array]:
            rows, cols = self._rows, self._cols
            total = rows * cols
            soft_buffer = array("d", [0.0]) * total if total else array("d")
            hard_buffer = array("d", [0.0]) * total if total else array("d")
            if cols == 0 or total == 0:
                return soft_buffer, hard_buffer

            flat = self._row_major_python()
            for r in range(rows):
                offset = r * cols
                row_slice = [float(flat[offset + c]) for c in range(cols)]
                finite_indices: list[int] = []
                max_value = -math.inf
                argmax_index = -1
                for idx, value in enumerate(row_slice):
                    if math.isnan(value):
                        continue
                    finite_indices.append(idx)
                    if value > max_value or argmax_index == -1:
                        max_value = value
                        argmax_index = idx

                if argmax_index == -1:
                    continue

                accum = 0.0
                exps = [0.0] * cols
                max_is_inf = math.isinf(max_value)
                for idx in finite_indices:
                    value = row_slice[idx]
                    shifted = 1.0 if max_is_inf and value == max_value else math.exp(value - max_value)
                    exps[idx] = shifted
                    accum += shifted

                inv = 1.0 / accum if accum > 0.0 and math.isfinite(accum) else 0.0
                for idx in finite_indices:
                    soft_buffer[offset + idx] = exps[idx] * inv if inv > 0.0 else 0.0

                if 0 <= argmax_index < cols:
                    hard_buffer[offset + argmax_index] = 1.0
            return soft_buffer, hard_buffer

        def _compute_row_hardmax_python(self) -> array:
            rows, cols = self._rows, self._cols
            total = rows * cols
            buffer = array("d", [0.0]) * total if total else array("d")
            if cols == 0 or total == 0:
                return buffer

            flat = self._row_major_python()
            for r in range(rows):
                offset = r * cols
                row_slice = [float(flat[offset + c]) for c in range(cols)]
                max_value = -math.inf
                argmax_index = -1
                for idx, value in enumerate(row_slice):
                    if math.isnan(value):
                        continue
                    if value > max_value or argmax_index == -1:
                        max_value = value
                        argmax_index = idx
                if 0 <= argmax_index < cols:
                    buffer[offset + argmax_index] = 1.0
            return buffer

        def row_softmax_hardmax(
            self, *, backend: str | None = None
        ) -> tuple["Tensor", "Tensor"]:
            """Return row-wise softmax probabilities paired with the hardmax mask."""

            target_backend = self._resolve_backend(backend)
            cls = type(self)

            if target_backend == "numpy":
                if not NUMPY_AVAILABLE:
                    raise RuntimeError("NumPy backend requested but NumPy is not installed")
                matrix = self._to_numpy(copy=False)
                if matrix.size == 0 or matrix.shape[1] == 0:
                    zeros = _np.zeros_like(matrix)
                    return cls._from_numpy_array(zeros.copy()), cls._from_numpy_array(zeros)

                rows, cols = matrix.shape
                soft = _np.zeros((rows, cols), dtype=_np.float64)
                hard = _np.zeros((rows, cols), dtype=_np.float64)

                for r in range(rows):
                    row = matrix[r]
                    finite_mask = ~_np.isnan(row)
                    if not finite_mask.any():
                        continue

                    finite_indices = _np.flatnonzero(finite_mask)
                    finite_values = row[finite_mask]
                    max_idx_local = int(finite_values.argmax())
                    max_value = float(finite_values[max_idx_local])
                    max_is_inf = math.isinf(max_value)

                    if max_is_inf:
                        shifted = _np.where(finite_values == max_value, 1.0, 0.0)
                    else:
                        shifted = _np.exp(finite_values - max_value)

                    denom = float(shifted.sum())
                    if math.isfinite(denom) and denom > 0.0:
                        soft_values = shifted / denom
                    else:
                        soft_values = _np.zeros_like(shifted)

                    soft[r, finite_indices] = soft_values

                    argmax_col = int(finite_indices[max_idx_local])
                    hard[r, argmax_col] = 1.0

                return cls._from_numpy_array(soft), cls._from_numpy_array(hard)

            rows, cols = self._rows, self._cols
            soft_buffer, hard_buffer = self._compute_row_softmax_hardmax_python()
            return (
                cls._from_python_array(rows, cols, soft_buffer),
                cls._from_python_array(rows, cols, hard_buffer),
            )

        def row_hardmax(self, *, backend: str | None = None) -> "Tensor":
            """Return the row-wise hardmax mask."""

            target_backend = self._resolve_backend(backend)
            cls = type(self)

            if target_backend == "numpy":
                if not NUMPY_AVAILABLE:
                    raise RuntimeError("NumPy backend requested but NumPy is not installed")
                matrix = self._to_numpy(copy=False)
                if matrix.size == 0 or matrix.shape[1] == 0:
                    return cls._from_numpy_array(matrix.copy())

                rows, cols = matrix.shape
                mask = _np.zeros((rows, cols), dtype=_np.float64)

                for r in range(rows):
                    row = matrix[r]
                    finite_mask = ~_np.isnan(row)
                    if not finite_mask.any():
                        continue
                    finite_values = row[finite_mask]
                    argmax_local = int(finite_values.argmax())
                    argmax_col = int(_np.flatnonzero(finite_mask)[argmax_local])
                    mask[r, argmax_col] = 1.0

                return cls._from_numpy_array(mask)

            rows, cols = self._rows, self._cols
            buffer = self._compute_row_hardmax_python()
            return cls._from_python_array(rows, cols, buffer)

        def row_softmax_hardmax_spiral(
            self, *, backend: str | None = None
        ) -> tuple["Tensor", "Tensor", "Tensor", dict[str, float]]:
            """Return softmax, hardmax, and spiral consensus tensors with metrics."""

            soft, hard = self.row_softmax_hardmax(backend=backend)
            rows, cols = self._rows, self._cols

            if NUMPY_AVAILABLE and getattr(soft, "_backend", None) == "numpy":
                soft_matrix = soft._to_numpy(copy=False)
                hard_matrix = hard._to_numpy(copy=False)
                fused, metrics = _spiral_softmax_hardmax_consensus_python(
                    soft_matrix.reshape(-1).tolist(),
                    hard_matrix.reshape(-1).tolist(),
                    rows,
                    cols,
                )
                spiral_matrix = _np.asarray(fused, dtype=_np.float64).reshape(rows, cols)
                spiral = type(self)._from_numpy_array(spiral_matrix)
            else:
                soft_flat = [float(value) for value in soft._row_major_python()]
                hard_flat = [float(value) for value in hard._row_major_python()]
                fused, metrics = _spiral_softmax_hardmax_consensus_python(
                    soft_flat,
                    hard_flat,
                    rows,
                    cols,
                )
                spiral = type(self)._from_python_array(rows, cols, array("d", fused))

            return soft, hard, spiral, metrics

        def scaled_dot_attention(
            self,
            keys: "Tensor",
            values: "Tensor",
            *,
            contexts: int,
            sequence: int,
            scale: float,
            z_bias: "Tensor" | None = None,
            attn_bias: "Tensor" | None = None,
            backend: str | None = None,
        ) -> "Tensor":
            """Compute fused scaled dot-product attention with optional biases."""

            if not isinstance(keys, Tensor) or not isinstance(values, Tensor):
                raise TypeError("scaled_dot_attention expects Tensor inputs for keys and values")
            if z_bias is not None and not isinstance(z_bias, Tensor):
                raise TypeError("z_bias must be a Tensor when provided")
            if attn_bias is not None and not isinstance(attn_bias, Tensor):
                raise TypeError("attn_bias must be a Tensor when provided")

            contexts = int(contexts)
            sequence = int(sequence)
            if contexts <= 0 or sequence <= 0:
                raise ValueError("contexts and sequence must be positive integers")
            head_dim = self._cols
            if head_dim <= 0:
                raise ValueError("scaled_dot_attention requires tensors with at least one column")

            expected_rows = contexts * sequence
            if (
                self._rows != expected_rows
                or keys._rows != expected_rows
                or values._rows != expected_rows
            ):
                raise ValueError("query, key, and value tensors must have contexts*sequence rows")
            if keys._cols != head_dim or values._cols != head_dim:
                raise ValueError("query, key, and value tensors must share the same column count")

            if z_bias is not None and z_bias.shape() != (contexts, sequence):
                raise ValueError("z_bias must have shape (contexts, sequence)")
            if attn_bias is not None and attn_bias.shape() != (expected_rows, sequence):
                raise ValueError("attn_bias must have shape (contexts*sequence, sequence)")

            target_backend = self._resolve_backend(backend, allow_gpu=True)
            cls = type(self)
            scale_value = float(scale)

            if target_backend == "numpy":
                queries = self._to_numpy(copy=False).reshape(contexts, sequence, head_dim)
                keys_arr = keys._to_numpy(copy=False).reshape(contexts, sequence, head_dim)
                values_arr = values._to_numpy(copy=False).reshape(contexts, sequence, head_dim)
                logits = _np.matmul(queries, keys_arr.transpose(0, 2, 1)) * scale_value
                if z_bias is not None:
                    zb = z_bias._to_numpy(copy=False).reshape(contexts, 1, sequence)
                    logits = logits + zb
                if attn_bias is not None:
                    ab = attn_bias._to_numpy(copy=False).reshape(contexts, sequence, sequence)
                    logits = logits + ab
                max_logits = logits.max(axis=2, keepdims=True)
                shifted = logits - max_logits
                weights = _np.exp(shifted)
                sums = weights.sum(axis=2, keepdims=True)
                weights = _np.divide(weights, sums, out=_np.zeros_like(weights), where=sums != 0.0)
                output = weights @ values_arr
                return cls._from_numpy_array(output.reshape(expected_rows, head_dim))

            buffer = self._scaled_dot_attention_python(
                keys,
                values,
                contexts=contexts,
                sequence=sequence,
                scale=scale_value,
                z_bias=z_bias,
                attn_bias=attn_bias,
            )
            return cls._from_python_array(expected_rows, head_dim, buffer)

        def _scaled_dot_attention_python(
            self,
            keys: "Tensor",
            values: "Tensor",
            *,
            contexts: int,
            sequence: int,
            scale: float,
            z_bias: "Tensor" | None,
            attn_bias: "Tensor" | None,
        ) -> array:
            rows = contexts * sequence
            head_dim = self._cols
            total = rows * head_dim
            out = array("d", [0.0]) * total if total else array("d")

            if total == 0:
                return out

            queries = self._row_major_python()
            keys_flat = keys._row_major_python()
            values_flat = values._row_major_python()
            z_bias_flat = None if z_bias is None else z_bias._row_major_python()
            attn_bias_flat = None if attn_bias is None else attn_bias._row_major_python()

            for context in range(contexts):
                context_offset = context * sequence
                for query_idx in range(sequence):
                    query_row = context_offset + query_idx
                    query_offset = query_row * head_dim
                    running_max = -1.0e30
                    running_sum = 0.0
                    accum = [0.0] * head_dim
                    for key_idx in range(sequence):
                        key_row = context_offset + key_idx
                        key_offset = key_row * head_dim
                        dot = 0.0
                        for dim in range(head_dim):
                            dot += queries[query_offset + dim] * keys_flat[key_offset + dim]

                        logit = dot * scale
                        if z_bias_flat is not None:
                            logit += z_bias_flat[context_offset + key_idx]
                        if attn_bias_flat is not None:
                            logit += attn_bias_flat[query_row * sequence + key_idx]

                        new_max = running_max if running_max > logit else logit
                        scaled_sum = (
                            running_sum * math.exp(running_max - new_max)
                            if running_sum > 0.0
                            else 0.0
                        )
                        exp_curr = math.exp(logit - new_max)
                        denom = scaled_sum + exp_curr
                        alpha = scaled_sum / denom if denom > 0.0 else 0.0
                        weight = exp_curr / denom if denom > 0.0 else 0.0
                        running_max = new_max
                        running_sum = denom

                        for dim in range(head_dim):
                            accum[dim] = accum[dim] * alpha + weight * values_flat[key_offset + dim]

                    for dim in range(head_dim):
                        out[query_offset + dim] = accum[dim]

            return out

        def _matmul_python(self, other: "Tensor") -> "Tensor":
            rows, cols, inner = self._rows, other._cols, self._cols
            if rows == 0 or cols == 0:
                return Tensor._from_python_array(rows, cols, array("d"))
            if inner == 0:
                return Tensor._from_python_array(rows, cols, array("d", [0.0]) * (rows * cols))

            out = array("d", [0.0]) * (rows * cols)
            left = self._row_major_python()
            right = other._row_major_python()
            col_tile = PY_STUB_MATMUL_COL_TILE
            if col_tile > cols:
                col_tile = cols
            inner_tile = PY_STUB_MATMUL_INNER_TILE
            if inner_tile > inner:
                inner_tile = inner
            fma = PY_STUB_FMA

            for i in range(rows):
                lhs_row_base = i * inner
                out_row_base = i * cols
                for col_start in range(0, cols, col_tile):
                    col_end = min(col_start + col_tile, cols)
                    block_width = col_end - col_start
                    for k_start in range(0, inner, inner_tile):
                        k_end = min(k_start + inner_tile, inner)
                        for k in range(k_start, k_end):
                            scale = left[lhs_row_base + k]
                            if scale == 0.0:
                                continue
                            rhs_base = k * cols + col_start
                            out_base = out_row_base + col_start
                            full = block_width - (block_width % 4)
                            offset = 0
                            while offset < full:
                                rhs_index = rhs_base + offset
                                out_index = out_base + offset
                                if fma is not None:
                                    out[out_index] = fma(scale, right[rhs_index], out[out_index])
                                    out[out_index + 1] = fma(scale, right[rhs_index + 1], out[out_index + 1])
                                    out[out_index + 2] = fma(scale, right[rhs_index + 2], out[out_index + 2])
                                    out[out_index + 3] = fma(scale, right[rhs_index + 3], out[out_index + 3])
                                else:
                                    out[out_index] += scale * right[rhs_index]
                                    out[out_index + 1] += scale * right[rhs_index + 1]
                                    out[out_index + 2] += scale * right[rhs_index + 2]
                                    out[out_index + 3] += scale * right[rhs_index + 3]
                                offset += 4
                            for tail in range(full, block_width):
                                idx = out_base + tail
                                rhs_idx = rhs_base + tail
                                if fma is not None:
                                    out[idx] = fma(scale, right[rhs_idx], out[idx])
                                else:
                                    out[idx] += scale * right[rhs_idx]

            return Tensor._from_python_array(rows, cols, out)

        def _matmul_numpy(self, other: "Tensor") -> "Tensor":
            result = self._to_numpy(copy=False) @ other._to_numpy(copy=False)
            return Tensor._from_numpy_array(result)

        def add(self, other: "Tensor") -> "Tensor":
            """Element-wise addition."""

            if not isinstance(other, Tensor):
                raise TypeError("add expects another Tensor instance")
            if self.shape() != other.shape():
                raise ValueError("tensor shapes must match for addition")

            cls = type(self)
            if NUMPY_AVAILABLE and (
                self._backend == "numpy" or other._backend == "numpy"
            ):
                result = self._to_numpy(copy=False) + other._to_numpy(copy=False)
                return cls._from_numpy_array(result)

            data = array(
                "d",
                (
                    a + b
                    for a, b in zip(self._row_major_python(), other._row_major_python())
                ),
            )
            return cls._from_python_array(self._rows, self._cols, data)

        def sub(self, other: "Tensor") -> "Tensor":
            """Element-wise subtraction."""

            if not isinstance(other, Tensor):
                raise TypeError("sub expects another Tensor instance")
            if self.shape() != other.shape():
                raise ValueError("tensor shapes must match for subtraction")

            cls = type(self)
            if NUMPY_AVAILABLE and (
                self._backend == "numpy" or other._backend == "numpy"
            ):
                result = self._to_numpy(copy=False) - other._to_numpy(copy=False)
                return cls._from_numpy_array(result)

            data = array(
                "d",
                (
                    a - b
                    for a, b in zip(self._row_major_python(), other._row_major_python())
                ),
            )
            return cls._from_python_array(self._rows, self._cols, data)

        def scale(self, value: float) -> "Tensor":
            """Return a new tensor with every element scaled by ``value``."""

            scalar = float(value)
            cls = type(self)
            if NUMPY_AVAILABLE and self._backend == "numpy":
                scaled = self._to_numpy(copy=False) * scalar
                return cls._from_numpy_array(scaled)

            data = array("d", (elem * scalar for elem in self._row_major_python()))
            return cls._from_python_array(self._rows, self._cols, data)

        def hadamard(self, other: "Tensor") -> "Tensor":
            if not isinstance(other, Tensor):
                raise TypeError("hadamard expects another Tensor instance")
            if self.shape != other.shape:
                raise ValueError("tensor shapes must match for hadamard product")
            if NUMPY_AVAILABLE:
                product = self._to_numpy(copy=False) * other._to_numpy(copy=False)
                return Tensor._from_numpy_array(product)
            data = array(
                "d",
                (
                    a * b
                    for a, b in zip(self._row_major_python(), other._row_major_python())
                ),
            )
            backend = (
                "blas"
                if BLAS_AVAILABLE
                and (self._backend == "blas" or other._backend == "blas")
                else "python"
            )
            return Tensor._from_python_array(
                self._rows, self._cols, data, backend=backend
            )

        def row_softmax(self, *, backend: str | None = None) -> "Tensor":
            rows, cols = self._rows, self._cols
            target_backend = _resolve_stub_backend(backend, op="row_softmax")

            if target_backend == "numpy":
                matrix = self._to_numpy(copy=False)
                if cols == 0:
                    result = _np.empty((rows, 0), dtype=_np.float64)
                else:
                    shifted = matrix - matrix.max(axis=1, keepdims=True)
                    exp = _np.exp(shifted)
                    sums = exp.sum(axis=1, keepdims=True)
                    result = _np.divide(
                        exp,
                        sums,
                        out=_np.zeros_like(exp),
                        where=sums > 0.0,
                    )
                return Tensor._from_numpy_array(result)

            total = rows * cols
            buffer = array("d") if total == 0 else array("d", [0.0]) * total
            data = self._row_major_python()
            for r in range(rows):
                offset = r * cols
                if cols == 0:
                    continue
                row_slice = data[offset : offset + cols]
                max_value = max(row_slice) if row_slice else 0.0
                running_sum = 0.0
                for c in range(cols):
                    value = math.exp(row_slice[c] - max_value)
                    buffer[offset + c] = value
                    running_sum += value
                scale = (1.0 / running_sum) if running_sum > 0.0 else 0.0
                for c in range(cols):
                    buffer[offset + c] *= scale
            return Tensor._from_python_array(rows, cols, buffer)

        def scaled_dot_attention(
            self,
            keys: "Tensor",
            values: "Tensor",
            *,
            contexts: int,
            sequence: int,
            scale: float,
            z_bias: "Tensor" | None = None,
            attn_bias: "Tensor" | None = None,
            backend: str | None = None,
        ) -> "Tensor":
            if not isinstance(keys, Tensor) or not isinstance(values, Tensor):
                raise TypeError("scaled_dot_attention expects Tensor inputs")

            contexts = int(contexts)
            sequence = int(sequence)
            if contexts <= 0 or sequence <= 0:
                raise ValueError("contexts and sequence must be positive integers")

            head_dim = self._cols
            expected_rows = contexts * sequence
            if self._rows != expected_rows:
                raise ValueError(
                    "query tensor has shape {} but expected ({} * {}, {})".format(
                        self.shape(), contexts, sequence, head_dim
                    )
                )
            if keys._rows != expected_rows or keys._cols != head_dim:
                raise ValueError("keys tensor must match query shape")
            if values._rows != expected_rows or values._cols != head_dim:
                raise ValueError("values tensor must match query shape")

            z_bias_buffer = None
            if z_bias is not None:
                if not isinstance(z_bias, Tensor):
                    raise TypeError("z_bias must be a Tensor or None")
                if z_bias.shape() != (contexts, sequence):
                    raise ValueError("z_bias must have shape (contexts, sequence)")
                z_bias_buffer = z_bias._row_major_python()

            attn_bias_buffer = None
            if attn_bias is not None:
                if not isinstance(attn_bias, Tensor):
                    raise TypeError("attn_bias must be a Tensor or None")
                if attn_bias.shape() != (expected_rows, sequence):
                    raise ValueError(
                        "attn_bias must have shape (contexts * sequence, sequence)"
                    )
                attn_bias_buffer = attn_bias._row_major_python()

            target_backend = _resolve_stub_backend(
                backend, op="scaled_dot_attention"
            )

            queries = self._row_major_python()
            keys_buffer = keys._row_major_python()
            values_buffer = values._row_major_python()
            total = expected_rows * head_dim
            buffer = array("d") if total == 0 else array("d", [0.0]) * total
            accum = [0.0] * head_dim
            scale_value = float(scale)

            for context in range(contexts):
                context_offset = context * sequence
                for query_idx in range(sequence):
                    query_row = context_offset + query_idx
                    query_offset = query_row * head_dim
                    if head_dim:
                        accum[:] = [0.0] * head_dim
                    logits: list[float] = []
                    for key_idx in range(sequence):
                        key_row = context_offset + key_idx
                        key_offset = key_row * head_dim
                        dot = 0.0
                        for dim in range(head_dim):
                            dot += (
                                queries[query_offset + dim]
                                * keys_buffer[key_offset + dim]
                            )
                        logit = dot * scale_value
                        if z_bias_buffer is not None:
                            logit += z_bias_buffer[context_offset + key_idx]
                        if attn_bias_buffer is not None:
                            logit += attn_bias_buffer[query_row * sequence + key_idx]
                        logits.append(logit)

                    if logits:
                        max_logit = max(logits)
                        exp_values = [math.exp(value - max_logit) for value in logits]
                        denom = sum(exp_values)
                        if denom > 0.0:
                            weights = [value / denom for value in exp_values]
                        else:
                            weights = [0.0] * len(exp_values)
                    else:
                        weights = []

                    for key_idx, weight in enumerate(weights):
                        if weight == 0.0:
                            continue
                        key_row = context_offset + key_idx
                        key_offset = key_row * head_dim
                        for dim in range(head_dim):
                            accum[dim] += weight * values_buffer[key_offset + dim]

                    out_offset = query_offset
                    for dim in range(head_dim):
                        buffer[out_offset + dim] = accum[dim]

            if target_backend == "numpy":
                matrix = _np.asarray(buffer, dtype=_np.float64).reshape(
                    expected_rows, head_dim
                )
                return Tensor._from_numpy_array(matrix)
            return Tensor._from_python_array(expected_rows, head_dim, buffer)

        def add(self, other: "Tensor") -> "Tensor":
            if not isinstance(other, Tensor):
                raise TypeError("add expects another Tensor instance")
            if self.shape != other.shape:
                raise ValueError("tensor shapes must match for add")
            if NUMPY_AVAILABLE and self._backend == "numpy" and other._backend == "numpy":
                result = self._data + other._data
                return Tensor._from_numpy_array(result)
            left = self._row_major_python()
            right = other._row_major_python()
            data = array("d", (a + b for a, b in zip(left, right)))
            return Tensor._from_python_array(self._rows, self._cols, data)

        def sub(self, other: "Tensor") -> "Tensor":
            if not isinstance(other, Tensor):
                raise TypeError("sub expects another Tensor instance")
            if self.shape != other.shape:
                raise ValueError("tensor shapes must match for sub")
            if NUMPY_AVAILABLE and self._backend == "numpy" and other._backend == "numpy":
                result = self._data - other._data
                return Tensor._from_numpy_array(result)
            left = self._row_major_python()
            right = other._row_major_python()
            data = array("d", (a - b for a, b in zip(left, right)))
            return Tensor._from_python_array(self._rows, self._cols, data)

        def scale(self, value: float) -> "Tensor":
            factor = float(value)
            if NUMPY_AVAILABLE and self._backend == "numpy":
                result = self._data * factor
                return Tensor._from_numpy_array(result)
            data = self._row_major_python()
            buffer = array("d", (factor * elem for elem in data))
            return Tensor._from_python_array(self._rows, self._cols, buffer)

        def add_scaled_(self, other: "Tensor", scale: float) -> None:
            if not isinstance(other, Tensor):
                raise TypeError("add_scaled_ expects another Tensor instance")
            if self.shape != other.shape:
                raise ValueError("tensor shapes must match for add_scaled_")
            factor = float(scale)
            if NUMPY_AVAILABLE and self._backend == "numpy":
                other_matrix = other._to_numpy(copy=False)
                self._data += other_matrix * factor
                return
            dest = self._data
            other_data = other._row_major_python()
            for index, value in enumerate(other_data):
                dest[index] += factor * value

        def add_row_inplace(self, bias: Sequence[float]) -> None:
            bias_values = [float(value) for value in bias]
            if len(bias_values) != self._cols:
                raise ValueError("bias length must match tensor columns")
            if self._rows == 0 or self._cols == 0:
                return
            if NUMPY_AVAILABLE and self._backend == "numpy":
                self._data += _np.asarray(bias_values, dtype=_np.float64)
                return
            cols = self._cols
            dest = self._data
            for r in range(self._rows):
                base = r * cols
                for c, value in enumerate(bias_values):
                    dest[base + c] += value

        def squared_l2_norm(self) -> float:
            if NUMPY_AVAILABLE and self._backend == "numpy":
                return float(_np.square(self._data).sum())
            total = 0.0
            for value in self._row_major_python():
                total += value * value
            return total

        def project_to_poincare(self, curvature: float) -> "Tensor":
            curvature = float(curvature)
            if curvature >= 0.0:
                raise ValueError("curvature must be negative for Poincaré projection")
            rows, cols = self._rows, self._cols
            data = self._row_major_python()
            total = rows * cols
            buffer = array("d") if total == 0 else array("d", [0.0]) * total
            scale = math.sqrt(-curvature)
            for r in range(rows):
                offset = r * cols
                row_slice = data[offset : offset + cols]
                norm_sq = sum(value * value for value in row_slice)
                norm = math.sqrt(norm_sq)
                if norm > 0.0:
                    clip = math.tanh(norm / scale)
                    factor = clip / norm
                    for c in range(cols):
                        buffer[offset + c] = row_slice[c] * factor
                else:
                    for c in range(cols):
                        buffer[offset + c] = row_slice[c]
            if self._backend == "numpy" and NUMPY_AVAILABLE:
                matrix = _np.asarray(buffer, dtype=_np.float64).reshape(rows, cols)
                return Tensor._from_numpy_array(matrix)
            return Tensor._from_python_array(rows, cols, buffer)

        def hyperbolic_distance(
            self, other: "Tensor", curvature: float
        ) -> float:
            if not isinstance(other, Tensor):
                raise TypeError("hyperbolic_distance expects another Tensor instance")
            if self.shape != other.shape:
                raise ValueError("tensor shapes must match for hyperbolic distance")
            curvature = float(curvature)
            if curvature >= 0.0:
                raise ValueError("curvature must be negative for hyperbolic distance")
            scale = math.sqrt(-curvature)
            sum_norm = 0.0
            sum_inner = 0.0
            for a, b in zip(self._row_major_python(), other._row_major_python()):
                pa = a / scale
                pb = b / scale
                diff = pa - pb
                sum_norm += diff * diff
                sum_inner += (1.0 - pa * pa) * (1.0 - pb * pb)
            denom = math.sqrt(max(sum_inner, 1e-6))
            return float(2.0 * math.acosh(1.0 + (sum_norm / denom)))

        @staticmethod
        def from_dlpack(_: object) -> "Tensor":
            raise RuntimeError(
                "DLPack interchange is unavailable in the SpiralTorch stub bindings."
            )

        def to_dlpack(self) -> object:
            raise RuntimeError(
                "DLPack interchange is unavailable in the SpiralTorch stub bindings."
            )

        def __dlpack__(self, *, stream=None):  # pragma: no cover - interoperability hook
            raise RuntimeError(
                "DLPack interchange is unavailable in the SpiralTorch stub bindings."
            )

        def __dlpack_device__(self):  # pragma: no cover - interoperability hook
            raise RuntimeError(
                "DLPack interchange is unavailable in the SpiralTorch stub bindings."
            )

        def numpy(self, *, copy: bool = True):
            if not NUMPY_AVAILABLE:
                raise RuntimeError("NumPy is not available in the stub bindings")
            return self._to_numpy(copy=copy)

        def _to_numpy(self, *, copy: bool) -> "_np.ndarray":
            if not NUMPY_AVAILABLE:
                raise RuntimeError("NumPy is not available in the stub bindings")
            if self._backend == "numpy":
                return self._data.copy() if copy else self._data
            array_view = _np.frombuffer(self._data, dtype=_np.float64, count=self._rows * self._cols)
            matrix = array_view.reshape(self._rows, self._cols)
            return matrix.copy() if copy else matrix

        @classmethod
        def _raise_dlpack_unavailable(cls) -> NoReturn:
            """DLPack interoperability requires NumPy support in the stub Tensor backend."""

            raise RuntimeError(cls.DLPACK_UNAVAILABLE_MESSAGE)

        @classmethod
        def from_dlpack(cls, capsule: Any) -> "Tensor":
            """Create a ``Tensor`` from a DLPack capsule.

            Raises:
                RuntimeError: DLPack interoperability requires NumPy support in the stub
                    Tensor backend.
            """

            if not NUMPY_AVAILABLE or _np is None or not hasattr(_np, "from_dlpack"):
                cls._raise_dlpack_unavailable()
            matrix = _np.from_dlpack(capsule)
            matrix = _np.asarray(matrix, dtype=_np.float64)
            if matrix.ndim != 2:
                raise ValueError("Tensor expects a 2D array")
            return cls._from_numpy_array(matrix)

        def to_dlpack(self) -> Any:
            """Export the tensor data as a DLPack capsule.

            Raises:
                RuntimeError: DLPack interoperability requires NumPy support in the stub
                    Tensor backend.
            """

            if not NUMPY_AVAILABLE or _np is None:
                self._raise_dlpack_unavailable()
            array = self._to_numpy(copy=False)
            dlpack = getattr(array, "__dlpack__", None)
            if dlpack is None:
                self._raise_dlpack_unavailable()
            return dlpack()

        def __dlpack__(self, stream: Any | None = None) -> Any:
            if not NUMPY_AVAILABLE or _np is None:
                self._raise_dlpack_unavailable()
            array = self._to_numpy(copy=False)
            dlpack = getattr(array, "__dlpack__", None)
            if dlpack is None:
                self._raise_dlpack_unavailable()
            if stream is None:
                return dlpack()
            return dlpack(stream=stream)

        def __dlpack_device__(self) -> tuple[int, int]:
            if not NUMPY_AVAILABLE or _np is None:
                self._raise_dlpack_unavailable()
            array = self._to_numpy(copy=False)
            dlpack_device = getattr(array, "__dlpack_device__", None)
            if dlpack_device is None:
                self._raise_dlpack_unavailable()
            return dlpack_device()

        def _row_major_python(self):
            """Return the tensor data as a row-major ``array('d')`` buffer.

            The returned buffer always contains ``self._rows * self._cols`` floating
            point values.
            """
            if self._backend == "python":
                return self._data
            return array("d", self._data.reshape(-1))

        @classmethod
        def _from_numpy_array(cls, array: "_np.ndarray") -> "Tensor":
            if not NUMPY_AVAILABLE:
                raise RuntimeError("NumPy is not available in the stub bindings")
            instance = cls.__new__(cls)
            matrix = _np.asarray(array, dtype=_np.float64)
            if matrix.ndim != 2:
                raise ValueError("Tensor expects a 2D array")
            instance._rows = int(matrix.shape[0])
            instance._cols = int(matrix.shape[1])
            instance._data = matrix.copy()
            instance._backend = "numpy"
            return instance

        @classmethod
        def _from_python_array(
            cls, rows: int, cols: int, buffer: array, *, backend: str = "python"
        ) -> "Tensor":
            if buffer.typecode != "d":
                raise TypeError("python backend tensors must use array('d') storage")
            if len(buffer) != rows * cols:
                raise ValueError("buffer does not match requested tensor shape")
            if backend not in {"python", "blas"}:
                raise ValueError("python array tensors must have 'python' or 'blas' backend")
            instance = cls.__new__(cls)
            instance._rows = int(rows)
            instance._cols = int(cols)
            instance._data = buffer
            instance._backend = backend
            return instance

        @_ShapeDescriptor
        def shape(self) -> tuple[int, int]:
            return (self._rows, self._cols)

        @property
        def rows(self) -> int:
            return int(self._rows)

        @property
        def cols(self) -> int:
            return int(self._cols)

        @property
        def backend(self) -> str:
            return self._backend

        def reshape(self, rows: int, cols: int) -> "Tensor":
            rows = int(rows)
            cols = int(cols)
            if rows < 0 or cols < 0:
                raise ValueError("tensor dimensions must be non-negative")
            total = rows * cols
            if total != self._rows * self._cols:
                raise ValueError(
                    "Tensor data of length {} cannot be reshaped to ({}, {})".format(
                        self._rows * self._cols, rows, cols
                    )
                )
            cls = type(self)
            if self._backend == "numpy":
                matrix = self._to_numpy(copy=False).reshape(rows, cols).copy()
                return cls._from_numpy_array(matrix)
            flat = self._row_major_python()
            buffer = array("d", flat)
            backend = "blas" if self._backend == "blas" else "python"
            return cls._from_python_array(rows, cols, buffer, backend=backend)

        def transpose(self) -> "Tensor":
            rows, cols = self._rows, self._cols
            cls = type(self)
            if self._backend == "numpy":
                matrix = self._to_numpy(copy=False).transpose().copy()
                return cls._from_numpy_array(matrix)
            flat = self._row_major_python()
            total = rows * cols
            transposed = array("d", [0.0]) * total if total else array("d")
            for r in range(rows):
                row_offset = r * cols
                for c in range(cols):
                    transposed[c * rows + r] = flat[row_offset + c]
            backend = "blas" if self._backend == "blas" else "python"
            return cls._from_python_array(cols, rows, transposed, backend=backend)

        def sum_axis0(self) -> list[float]:
            cols = self._cols
            if cols == 0:
                return []
            if self._backend == "numpy":
                summed = self._to_numpy(copy=False).sum(axis=0)
                return [float(value) for value in summed.tolist()]
            totals = [0.0] * cols
            flat = self._row_major_python()
            for r in range(self._rows):
                base = r * cols
                for c in range(cols):
                    totals[c] += flat[base + c]
            return totals

        def sum_axis1(self) -> list[float]:
            rows = self._rows
            if rows == 0:
                return []
            if self._backend == "numpy":
                summed = self._to_numpy(copy=False).sum(axis=1)
                return [float(value) for value in summed.tolist()]
            cols = self._cols
            totals = [0.0] * rows
            flat = self._row_major_python()
            for r in range(rows):
                base = r * cols
                row_total = 0.0
                for c in range(cols):
                    row_total += flat[base + c]
                totals[r] = row_total
            return totals

        def squared_l2_norm(self) -> float:
            """Return the squared L2 norm of the tensor."""

            if NUMPY_AVAILABLE and self._backend == "numpy":
                matrix = self._to_numpy(copy=False)
                return float(_np.sum(matrix * matrix))

            total = 0.0
            for value in self._row_major_python():
                total += value * value
            return float(total)

        def project_to_poincare(self, curvature: float) -> "Tensor":
            """Project each row onto the Poincaré ball for the given curvature."""

            if curvature >= 0.0:
                raise ValueError("curvature must be negative for hyperbolic projection")
            scale = math.sqrt(-float(curvature))
            cls = type(self)

            if NUMPY_AVAILABLE and self._backend == "numpy":
                matrix = self._to_numpy(copy=False)
                if matrix.size == 0:
                    return cls._from_numpy_array(matrix.copy())
                norms = _np.linalg.norm(matrix, axis=1, keepdims=True)
                clipped = _np.tanh(
                    _np.divide(norms, scale, out=_np.zeros_like(norms), where=norms != 0.0)
                )
                factors = _np.divide(
                    clipped,
                    norms,
                    out=_np.ones_like(norms),
                    where=norms != 0.0,
                )
                projected = matrix * factors
                return cls._from_numpy_array(projected)

            rows, cols = self._rows, self._cols
            total = rows * cols
            data = self._row_major_python()
            buffer = array("d", [0.0]) * total if total else array("d")
            if cols == 0 or total == 0:
                return cls._from_python_array(rows, cols, buffer)
            for r in range(rows):
                offset = r * cols
                segment = [float(data[offset + c]) for c in range(cols)]
                norm = math.sqrt(sum(value * value for value in segment))
                if norm > 0.0:
                    clip = math.tanh(norm / scale)
                    factor = clip / norm
                    for c in range(cols):
                        buffer[offset + c] = segment[c] * factor
                else:
                    for c in range(cols):
                        buffer[offset + c] = segment[c]
            return cls._from_python_array(rows, cols, buffer)

        def hyperbolic_distance(self, other: "Tensor", curvature: float) -> float:
            """Estimate the hyperbolic distance between two projected points."""

            if not isinstance(other, Tensor):
                raise TypeError("hyperbolic_distance expects another Tensor instance")
            if self.shape() != other.shape():
                raise ValueError("tensor shapes must match for hyperbolic distance")
            if curvature >= 0.0:
                raise ValueError("curvature must be negative for hyperbolic distance")

            scale = math.sqrt(-float(curvature))

            if NUMPY_AVAILABLE and (
                self._backend == "numpy" or other._backend == "numpy"
            ):
                a = self._to_numpy(copy=False) / scale
                b = other._to_numpy(copy=False) / scale
                diff = a - b
                sum_norm = float(_np.sum(diff * diff))
                sum_inner = float(_np.sum((1.0 - a * a) * (1.0 - b * b)))
                denom = math.sqrt(max(sum_inner, 1e-6))
                return float(2.0 * math.acosh(1.0 + sum_norm / denom))

            sum_norm = 0.0
            sum_inner = 0.0
            for lhs, rhs in zip(self._row_major_python(), other._row_major_python()):
                pa = lhs / scale
                pb = rhs / scale
                diff = pa - pb
                sum_norm += diff * diff
                sum_inner += (1.0 - pa * pa) * (1.0 - pb * pb)
            denom = math.sqrt(max(sum_inner, 1e-6))
            return float(2.0 * math.acosh(1.0 + sum_norm / denom))

        def tolist(self):
            rows, cols = self._rows, self._cols

            if rows == 0:
                return []
            if cols == 0:
                return [[] for _ in range(rows)]

            if self._backend == "numpy":
                matrix = self._to_numpy(copy=False).reshape(rows, cols)
                return [[float(value) for value in row] for row in matrix]

            flat = self._row_major_python()
            return [
                [float(flat[row_offset + c]) for c in range(cols)]
                for row_offset in range(0, rows * cols, cols)
            ]

        @staticmethod
        def zeros(rows: int, cols: int) -> "Tensor":
            rows = _tensor_coerce_index(rows, "rows")
            cols = _tensor_coerce_index(cols, "cols")
            total = rows * cols
            if NUMPY_AVAILABLE:
                matrix = _np.zeros((rows, cols), dtype=_np.float64)
                return Tensor._from_numpy_array(matrix)
            buffer = array("d", [0.0]) * total if total else array("d")
            backend = "blas" if BLAS_AVAILABLE else "python"
            return Tensor._from_python_array(rows, cols, buffer, backend=backend)

        @staticmethod
        def randn(
            rows: int,
            cols: int,
            mean: float = 0.0,
            std: float = 1.0,
            seed: int | None = None,
        ) -> "Tensor":
            rows = _tensor_coerce_index(rows, "rows")
            cols = _tensor_coerce_index(cols, "cols")
            total = rows * cols
            if NUMPY_AVAILABLE:
                rng = _np.random.default_rng(seed)
                matrix = rng.normal(loc=mean, scale=std, size=(rows, cols)).astype(
                    _np.float64
                )
                return Tensor._from_numpy_array(matrix)
            rng = random.Random(seed)
            values = [rng.gauss(mean, std) for _ in range(total)]
            buffer = array("d", values)
            backend = "blas" if BLAS_AVAILABLE else "python"
            return Tensor._from_python_array(rows, cols, buffer, backend=backend)

        @staticmethod
        def rand(
            rows: int,
            cols: int,
            min: float = 0.0,
            max: float = 1.0,
            seed: int | None = None,
        ) -> "Tensor":
            rows = _tensor_coerce_index(rows, "rows")
            cols = _tensor_coerce_index(cols, "cols")
            if max < min:
                raise ValueError("max must be greater than or equal to min")
            total = rows * cols
            if NUMPY_AVAILABLE:
                rng = _np.random.default_rng(seed)
                matrix = rng.uniform(low=min, high=max, size=(rows, cols)).astype(
                    _np.float64
                )
                return Tensor._from_numpy_array(matrix)
            rng = random.Random(seed)
            values = [rng.uniform(min, max) for _ in range(total)]
            buffer = array("d", values)
            backend = "blas" if BLAS_AVAILABLE else "python"
            return Tensor._from_python_array(rows, cols, buffer, backend=backend)

        @staticmethod
        def cat_rows(tensors: Sequence["Tensor"]) -> "Tensor":
            tensors = list(tensors)
            if not tensors:
                raise ValueError("cat_rows requires at least one tensor")
            if not all(isinstance(tensor, Tensor) for tensor in tensors):
                raise TypeError("cat_rows expects a sequence of Tensor instances")
            cols = tensors[0]._cols
            for tensor in tensors:
                if tensor._cols != cols:
                    raise ValueError("all tensors must have the same number of columns")
            total_rows = sum(tensor._rows for tensor in tensors)
            use_numpy = NUMPY_AVAILABLE and any(
                tensor._backend == "numpy" for tensor in tensors
            )
            if use_numpy:
                matrices = [tensor._to_numpy(copy=False) for tensor in tensors]
                concatenated = _np.concatenate(matrices, axis=0)
                cls = type(tensors[0])
                return cls._from_numpy_array(concatenated)
            cls = type(tensors[0])
            data = array("d")
            for tensor in tensors:
                data.extend(tensor._row_major_python())
            return cls._from_python_array(total_rows, cols, data)
            backend = "blas" if BLAS_AVAILABLE else "python"
            return Tensor._from_python_array(total_rows, cols, data, backend=backend)

        def __matmul__(self, other) -> "Tensor":  # pragma: no cover - convenience wrapper
            if isinstance(other, Tensor):
                return self.matmul(other)
            return NotImplemented

        def __rmatmul__(self, other) -> "Tensor":  # pragma: no cover - convenience wrapper
            if isinstance(other, Tensor):
                return other.matmul(self)
            return NotImplemented

        def __mul__(self, other) -> "Tensor":  # pragma: no cover - align with Hadamard
            if isinstance(other, Tensor):
                return self.hadamard(other)
            return NotImplemented

        def __array__(self):  # pragma: no cover - interoperability hook
            return self._to_numpy(copy=True)

        def __repr__(self) -> str:  # pragma: no cover - debugging helper
            return f"Tensor(shape={self.shape()}, backend='{self._backend}')"

    Tensor.__module__ = module.__name__

    def available_stub_backends() -> tuple[str, ...]:
        options: list[str] = []
        if BLAS_AVAILABLE:
            options.append("blas")
        if NUMPY_AVAILABLE:
            options.append("numpy")
        options.append("python")
        return tuple(options)

    module.Tensor = Tensor
    module.available_stub_backends = available_stub_backends
    module.default_stub_backend = (
        "blas"
        if BLAS_AVAILABLE
        else ("numpy" if NUMPY_AVAILABLE else "python")
    )

    class Axis:
        """Named axis descriptor used by :class:`LabeledTensor` in the stub runtime."""

        __slots__ = ("name", "size")

        def __init__(self, name: Any, size: int | None = None) -> None:
            label = str(name).strip()
            if not label:
                raise ValueError("axis name must be a non-empty string")
            self.name = label
            if size is None:
                self.size = None
            else:
                value = int(size)
                if value <= 0:
                    raise ValueError("axis size must be positive")
                self.size = value

        def with_size(self, size: int) -> "Axis":
            value = int(size)
            if value <= 0:
                raise ValueError("size must be positive")
            return Axis(self.name, value)

        def __repr__(self) -> str:  # pragma: no cover - debugging helper
            suffix = self.size if self.size is not None else "?"
            return f"Axis(name={self.name!r}, size={suffix})"

    def _ensure_tensor_type() -> type[Tensor]:
        tensor_type = module.__dict__.get("Tensor")
        if tensor_type is None:
            raise RuntimeError("Tensor export is unavailable in this build")
        return tensor_type

    def _prepare_rows(data: Any) -> list[list[float]]:
        if hasattr(data, "tolist") and not _tensor_is_sequence(data):
            data = data.tolist()
        if isinstance(data, _ensure_tensor_type()):
            rows = data.tolist()
            return [[float(value) for value in row] for row in rows]
        if not _tensor_is_sequence(data):
            raise TypeError("tensor data must be a sequence")
        if not data:
            raise ValueError("tensor data must contain at least one row")
        rows: list[list[float]] = []
        head = data[0]
        if hasattr(head, "tolist") and not _tensor_is_sequence(head):
            head = head.tolist()
        if _tensor_is_sequence(head):
            width: int | None = None
            for row in data:  # type: ignore[assignment]
                if hasattr(row, "tolist") and not _tensor_is_sequence(row):
                    row = row.tolist()
                if not _tensor_is_sequence(row):
                    raise TypeError("tensor rows must be sequences of numbers")
                values = [float(value) for value in row]
                if not values:
                    raise ValueError("tensor rows cannot be empty")
                if width is None:
                    width = len(values)
                elif len(values) != width:
                    raise ValueError("all rows must share the same length")
                rows.append(values)
            return rows
        values = [float(value) for value in data]
        if not values:
            raise ValueError("tensor data must contain at least one element")
        return [values]

    def _tensor_from_data(data: Any):
        tensor_type = _ensure_tensor_type()
        if isinstance(data, tensor_type):
            return data
        rows = _prepare_rows(data)
        height = len(rows)
        width = len(rows[0])
        flat: list[float] = [value for row in rows for value in row]
        return tensor_type(height, width, flat)

    def _coerce_axis(axis: Axis | str) -> Axis:
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

    def _normalise_axes(axes: Sequence[Axis | str]) -> tuple[Axis, Axis]:
        seq = list(axes)
        if len(seq) != 2:
            raise ValueError("exactly two axes are required for a 2D tensor")
        first = _coerce_axis(seq[0])
        second = _coerce_axis(seq[1])
        return (first, second)

    class LabeledTensor:
        """Tensor wrapper that carries human-readable axis annotations."""

        def __init__(self, data: Any, axes: Sequence[Axis | str]) -> None:
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

        def tolist(self) -> list[list[float]]:
            return self._tensor.tolist()

        def rename(self, axes: Sequence[Axis | str]) -> "LabeledTensor":
            return LabeledTensor(self._tensor, axes)

        def with_axes(self, axes: Sequence[Axis | str]) -> "LabeledTensor":
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

        def describe(self) -> dict[str, Any]:
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
        data: Any,
        *,
        axes: Sequence[Axis | str] | None = None,
    ):
        base = _tensor_from_data(data)
        if axes is None:
            return base
        return LabeledTensor(base, axes)

    def label_tensor(tensor_obj: Any, axes: Sequence[Axis | str]) -> LabeledTensor:
        return LabeledTensor(tensor_obj, axes)

    _SCALESTACK_STUB_MESSAGE = (
        "ScaleStack is unavailable in the stub bindings. Build the native extension "
        "via `maturin develop -m bindings/st-py/Cargo.toml`."
    )

    class ScaleStack:
        """Placeholder implementation for the native :class:`ScaleStack`.

        Build the native extension via ``maturin develop -m bindings/st-py/Cargo.toml``
        to access the full feature set.
        """

        __slots__ = ("_mode", "_threshold", "_meta")

        def __init__(self, *, mode: str, threshold: float, meta: dict[str, Any] | None = None) -> None:
            self._mode = str(mode)
            self._threshold = float(threshold)
            self._meta = dict(meta or {})

        @property
        def threshold(self) -> float:
            return self._threshold

        @property
        def mode(self) -> str:
            return self._mode

        @property
        def meta(self) -> dict[str, Any]:
            """Return metadata captured when constructing the stub stack."""

            return dict(self._meta)

        def _raise_unavailable(self, method: str) -> None:
            raise RuntimeError(
                f"ScaleStack.{method} is unavailable in the stub bindings. "
                "Build the native extension via `maturin develop -m bindings/st-py/Cargo.toml`."
            )

        def samples(self) -> list[tuple[float, float]]:
            self._raise_unavailable("samples")

        def persistence(self) -> list[tuple[float, float, float]]:
            self._raise_unavailable("persistence")

        def interface_density(self) -> float | None:
            self._raise_unavailable("interface_density")

        def moment(self, order: int) -> float:
            self._raise_unavailable("moment")

        def boundary_dimension(self, ambient_dim: float, window: int) -> float | None:
            self._raise_unavailable("boundary_dimension")

        def coherence_break_scale(self, level: float) -> float | None:
            self._raise_unavailable("coherence_break_scale")

        def coherence_profile(self, levels: Sequence[float]) -> list[float | None]:
            self._raise_unavailable("coherence_profile")

        def __repr__(self) -> str:  # pragma: no cover - debugging helper
            return f"ScaleStack(mode={self._mode!r}, threshold={self._threshold!r}, stub=True)"

    def _coerce_float_sequence(values: Any, label: str) -> list[float]:
        if not _tensor_is_iterable(values):
            raise TypeError(f"{label} must be an iterable of numbers")
        sequence = [float(value) for value in values]
        if not sequence:
            raise ValueError(f"{label} must not be empty")
        return sequence

    def scalar_scale_stack(
        field: Sequence[float],
        shape: Sequence[int],
        scales: Sequence[float],
        threshold: float,
    ) -> ScaleStack:
        rows, cols = _tensor_coerce_shape(shape, "shape")
        field_values = _coerce_float_sequence(field, "field")
        if len(field_values) != rows * cols:
            raise ValueError(
                "field length does not match the provided shape in the stub bindings"
            )
        scale_values = _coerce_float_sequence(scales, "scales")
        return ScaleStack(
            mode="scalar",
            threshold=float(threshold),
            meta={
                "shape": (rows, cols),
                "field": tuple(field_values),
                "scales": tuple(scale_values),
            },
        )

    def semantic_scale_stack(
        embeddings: Sequence[Sequence[float]],
        scales: Sequence[float],
        threshold: float,
        metric: str = "cosine",
    ) -> ScaleStack:
        if not _tensor_is_sequence(embeddings):
            raise TypeError("embeddings must be a sequence of sequences")
        prepared = _prepare_rows(embeddings)
        scale_values = _coerce_float_sequence(scales, "scales")
        return ScaleStack(
            mode="semantic",
            threshold=float(threshold),
            meta={
                "shape": (len(prepared), len(prepared[0])),
                "scales": tuple(scale_values),
                "metric": str(metric),
            },
        )

    module.Axis = Axis
    module.LabeledTensor = LabeledTensor
    module.tensor = tensor
    module.label_tensor = label_tensor
    module.ScaleStack = ScaleStack
    module.scalar_scale_stack = scalar_scale_stack
    module.semantic_scale_stack = semantic_scale_stack

    all_exports = module.__dict__.setdefault("__all__", [])
    for symbol in {
        "Tensor",
        "available_stub_backends",
        "default_stub_backend",
        "Axis",
        "LabeledTensor",
        "tensor",
        "label_tensor",
        "ScaleStack",
        "scalar_scale_stack",
        "semantic_scale_stack",
    }:
        if symbol not in all_exports:
            all_exports.append(symbol)
    module.__dict__.setdefault("__version__", "0.0.0+stub")

    def _stub_runtime_error(feature: str) -> RuntimeError:
        return RuntimeError(
            "The SpiralTorch stub bindings cannot provide "
            f"{feature!r}; build the native extension ("
            "`maturin develop -m bindings/st-py/Cargo.toml`) for full functionality."
        )

    def _register_stub_module(name: str, *, doc: str | None = None) -> types.ModuleType:
        qualname = f"{module.__name__}.{name}"
        stub_module = types.ModuleType(qualname, doc)

        def _stub_getattr(attribute: str, *, _qualname: str = qualname):
            raise _stub_runtime_error(f"{_qualname}.{attribute}")

        stub_module.__getattr__ = _stub_getattr  # type: ignore[attr-defined]
        sys.modules[qualname] = stub_module
        setattr(module, name, stub_module)
        if name not in all_exports:
            all_exports.append(name)
        return stub_module

    _PLACEHOLDER_MODULES = {
        "dataset": "Datasets & loaders are only available once the SpiralTorch native extension is built.",
        "linalg": "Linear algebra helpers require the SpiralTorch native extension.",
        "rec": "Signal reconstruction tools require the SpiralTorch native extension.",
        "telemetry": "Telemetry integrations require the SpiralTorch native extension.",
        "ecosystem": "Ecosystem integrations require the SpiralTorch native extension.",
    }

    for _name, _doc in _PLACEHOLDER_MODULES.items():
        _register_stub_module(_name, doc=_doc)

    def _install_spiral_rl_stub() -> types.ModuleType:
        stub = _register_stub_module(
            "spiral_rl",
            doc=(
                "Stub reinforcement learning harness. Build the native SpiralTorch extension "
                "to access training agents."
            ),
        )
        sys.modules["spiral_rl"] = stub
        module_name = f"{module.__name__}.spiral_rl"

        class _StubAgent:
            """Stub placeholder for SpiralTorch reinforcement learning agents."""

            __slots__ = ()

            def __init__(self, *args, **kwargs):
                raise _stub_runtime_error(f"{module_name}.stAgent")

            def _fail(self) -> None:
                raise _stub_runtime_error(f"{module_name}.stAgent")

            def select_action(self, *args, **kwargs):
                self._fail()

            def select_actions(self, *args, **kwargs):
                self._fail()

            def update(self, *args, **kwargs):
                self._fail()

            def update_batch(self, *args, **kwargs):
                self._fail()

            @property
            def epsilon(self):
                self._fail()

            def set_epsilon(self, *args, **kwargs):
                self._fail()

            def set_exploration(self, *args, **kwargs):
                self._fail()

            def state_dict(self):
                self._fail()

            def load_state_dict(self, *args, **kwargs):
                self._fail()

        _StubAgent.__name__ = "stAgent"
        _StubAgent.__module__ = "spiral_rl"

        class _PpoAgent(_StubAgent):
            __name__ = "PpoAgent"

            def __init__(self, *args, **kwargs):
                raise _stub_runtime_error(f"{module_name}.PpoAgent")

        _PpoAgent.__module__ = "spiral_rl"

        class _SacAgent(_StubAgent):
            __name__ = "SacAgent"

            def __init__(self, *args, **kwargs):
                raise _stub_runtime_error(f"{module_name}.SacAgent")

        _SacAgent.__module__ = "spiral_rl"

        stub.stAgent = _StubAgent
        stub.DqnAgent = _StubAgent
        stub.PyDqnAgent = _StubAgent
        stub.PpoAgent = _PpoAgent
        stub.SacAgent = _SacAgent
        stub.__all__ = ["stAgent", "DqnAgent", "PyDqnAgent", "PpoAgent", "SacAgent"]

        def _spiral_rl_getattr(attribute: str, *, _module_name: str = module_name):
            raise _stub_runtime_error(f"{_module_name}.{attribute}")

        stub.__getattr__ = _spiral_rl_getattr  # type: ignore[attr-defined]
        return stub

    _install_spiral_rl_stub()

    def _bridge_pure_python_namespace() -> None:
        base_dir = pathlib.Path(__file__).resolve().parents[1] / "bindings" / "python" / "spiral"
        init_py = base_dir / "__init__.py"
        if not init_py.exists():
            return

        if importlib.util.find_spec("numpy") is None:
            return

        bridge_name = "_spiraltorch_stub_py_bridge"
        spec = importlib.util.spec_from_file_location(
            bridge_name,
            init_py,
            submodule_search_locations=[str(base_dir)],
        )
        if spec is None or spec.loader is None:
            return

        bridge_module = importlib.util.module_from_spec(spec)
        bridge_module.__package__ = bridge_name
        bridge_module.__path__ = [str(base_dir)]
        sys.modules[bridge_name] = bridge_module
        try:
            spec.loader.exec_module(bridge_module)
        except ModuleNotFoundError:
            sys.modules.pop(bridge_name, None)
            return

        def _register(name: str, value: object) -> None:
            if name.startswith("_"):
                return
            module.__dict__[name] = value
            exports = module.__dict__.setdefault("__all__", [])
            if name not in exports:
                exports.append(name)

        def _merge_public_members(source: types.ModuleType) -> None:
            exports: Iterable[str] | None = getattr(source, "__all__", None)
            if not exports:
                exports = (name for name in dir(source) if not name.startswith("_"))
            for name in exports:
                value = getattr(source, name, None)
                if value is not None:
                    _register(name, value)

        def _rebind_module(submodule: types.ModuleType, *, relative_name: str) -> None:
            target_name = f"{module.__name__}.{relative_name}"
            submodule.__name__ = target_name
            if "." in relative_name:
                submodule.__package__ = f"{module.__name__}.{relative_name.rsplit('.', 1)[0]}"
            else:
                submodule.__package__ = module.__name__
            sys.modules[target_name] = submodule
            head, _, tail = relative_name.partition(".")
            if not tail:
                setattr(module, head, submodule)
                _register(head, submodule)

        _merge_public_members(bridge_module)

        prefix = f"{bridge_name}."
        for name, value in list(sys.modules.items()):
            if name == bridge_name or not name.startswith(prefix):
                continue
            if not isinstance(value, types.ModuleType):
                continue
            relative_name = name[len(prefix) :]
            _rebind_module(value, relative_name=relative_name)

        for submodule in ("data", "hypergrad", "inference"):
            target = f"{bridge_name}.{submodule}"
            loaded: types.ModuleType | None
            if target in sys.modules:
                candidate = sys.modules[target]
                loaded = candidate if isinstance(candidate, types.ModuleType) else None
            else:
                spec = importlib.util.find_spec(target)
                if spec is None:
                    continue
                candidate = importlib.import_module(target)
                loaded = candidate if isinstance(candidate, types.ModuleType) else None
            if loaded is None:
                continue
            _rebind_module(loaded, relative_name=submodule)

    _bridge_pure_python_namespace()

    def _missing_attr(name: str):
        raise AttributeError(
            f"spiraltorch.{name} is unavailable in the stub bindings. Build the native extension "
            "for full functionality."
        )

    def __getattr__(name: str):  # pragma: no cover - dynamic attribute guard
        if name in {"Tensor", "available_stub_backends", "default_stub_backend", "__all__", "__version__"}:
            return module.__dict__[name]
        raise _missing_attr(name)

    module.__getattr__ = __getattr__

    STUB_TENSOR_TYPES = (Tensor,)


_load_native_package()
del _load_native_package
