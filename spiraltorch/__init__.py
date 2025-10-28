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

    NUMPY_AVAILABLE = _np is not None

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
            if backend_hint is not None and backend_hint not in {"numpy", "python"}:
                raise ValueError("backend must be 'numpy', 'python', or None")
            if backend_hint == "numpy" and not NUMPY_AVAILABLE:
                raise RuntimeError("NumPy backend requested but NumPy is not installed")
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

            preferred_backend = backend_hint or ("numpy" if NUMPY_AVAILABLE else "python")
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
                self._backend = "python"

        # noqa: D401 - mirror real signature from the native extension
        def matmul(self, other: "Tensor", *, backend: str | None = None):
            if not isinstance(other, Tensor):
                raise TypeError("matmul expects another Tensor instance")
            if self._cols != other._rows:
                raise ValueError("inner dimensions do not match for matmul")

            if backend is not None and backend not in {"numpy", "python"}:
                raise ValueError("backend must be 'numpy', 'python', or None")
            target_backend = backend or ("numpy" if NUMPY_AVAILABLE else "python")

            if target_backend == "numpy":
                if not NUMPY_AVAILABLE:
                    raise RuntimeError("NumPy backend requested but NumPy is not installed")
                return self._matmul_numpy(other)
            return self._matmul_python(other)

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
            return Tensor._from_python_array(self._rows, self._cols, data)

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

        def _row_major_python(self):
            """Return the tensor data as a 1D row-major ``array('d')`` buffer."""
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
        def _from_python_array(cls, rows: int, cols: int, buffer: array) -> "Tensor":
            if buffer.typecode != "d":
                raise TypeError("python backend tensors must use array('d') storage")
            if len(buffer) != rows * cols:
                raise ValueError("buffer does not match requested tensor shape")
            instance = cls.__new__(cls)
            instance._rows = int(rows)
            instance._cols = int(cols)
            instance._data = buffer
            instance._backend = "python"
            return instance

        @_ShapeDescriptor
        def shape(self) -> tuple[int, int]:
            return (self._rows, self._cols)

        @property
        def rows(self) -> int:
            return self._rows

        @property
        def cols(self) -> int:
            return self._cols

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
            if self._backend == "numpy":
                matrix = self._to_numpy(copy=False).reshape(rows, cols).copy()
                return Tensor._from_numpy_array(matrix)
            flat = self._row_major_python()
            return Tensor._from_python_array(rows, cols, array("d", flat))

        def transpose(self) -> "Tensor":
            rows, cols = self._rows, self._cols
            if self._backend == "numpy":
                matrix = self._to_numpy(copy=False).transpose().copy()
                return Tensor._from_numpy_array(matrix)
            flat = self._row_major_python()
            total = rows * cols
            transposed = array("d", [0.0]) * total if total else array("d")
            for r in range(rows):
                row_offset = r * cols
                for c in range(cols):
                    transposed[c * rows + r] = flat[row_offset + c]
            return Tensor._from_python_array(cols, rows, transposed)

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

            if self._backend == "python":
                flat = self._data
                return [
                    [float(flat[r * cols + c]) for c in range(cols)]
                    for r in range(rows)
                ]

            matrix = self._to_numpy(copy=False)
            return [
                [float(matrix[r, c]) for c in range(cols)]
                for r in range(rows)
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
            return Tensor._from_python_array(rows, cols, buffer)

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
            return Tensor._from_python_array(rows, cols, buffer)

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
            return Tensor._from_python_array(rows, cols, buffer)

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
            return Tensor._from_python_array(total_rows, cols, data)

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
        return ("numpy", "python") if NUMPY_AVAILABLE else ("python",)

    module.Tensor = Tensor
    module.available_stub_backends = available_stub_backends
    module.default_stub_backend = "numpy" if NUMPY_AVAILABLE else "python"
    all_exports = module.__dict__.setdefault("__all__", [])
    for symbol in {"Tensor", "available_stub_backends", "default_stub_backend"}:
        if symbol not in all_exports:
            all_exports.append(symbol)
    module.__dict__.setdefault("__version__", "0.0.0+stub")

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
