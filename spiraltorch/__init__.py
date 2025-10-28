"""Development shim for the SpiralTorch Python bindings.

This module lets ``import spiraltorch`` succeed directly from a source
checkout without first installing the wheel.  It delegates to the real
package that lives under ``bindings/st-py`` and improves the error message
when the compiled extension has not been built yet.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Sequence
import importlib.machinery
import importlib.util
import math
import random
import pathlib
import sys
import types
import warnings
from typing import Any


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

        def _row_major_python(self):
            """Return the matrix data flattened row-major into an ``array('d')`` buffer."""
            if self._backend in {"python", "blas"}:
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

        def tolist(self):
            rows, cols = self._rows, self._cols

            if rows == 0:
                return []
            if cols == 0:
                return [[] for _ in range(rows)]

            if self._backend == "numpy":
                matrix = self._to_numpy(copy=False)
                return [
                    [float(matrix[r, c]) for c in range(cols)]
                    for r in range(rows)
                ]

            flat = self._row_major_python()
            return [
                [float(flat[row_offset + c]) for c in range(cols)]
                for row_offset in range(0, rows * cols, cols)
            ]

        @staticmethod
        def zeros(rows: int, cols: int) -> "Tensor":
            rows = int(rows)
            cols = int(cols)
            if rows < 0 or cols < 0:
                raise ValueError("tensor dimensions must be non-negative")
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
            rows = int(rows)
            cols = int(cols)
            if rows < 0 or cols < 0:
                raise ValueError("tensor dimensions must be non-negative")
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
            rows = int(rows)
            cols = int(cols)
            if rows < 0 or cols < 0:
                raise ValueError("tensor dimensions must be non-negative")
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
                return Tensor._from_numpy_array(concatenated)
            data = array("d")
            for tensor in tensors:
                data.extend(tensor._row_major_python())
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
