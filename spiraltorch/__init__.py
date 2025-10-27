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
import pathlib
import sys
import warnings


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

    try:  # Prefer a NumPy-backed shim when available for better performance.
        import numpy as _np  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        _np = None  # type: ignore

    NUMPY_AVAILABLE = _np is not None

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
        except Exception as exc:  # noqa: BLE001 - surface Pythonic error message
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

    def _tensor_flatten_data(data: object) -> tuple[int, int, list[float]]:
        if isinstance(data, Tensor):
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
            raise ValueError("Tensor data cannot be empty")

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
                normalized = _tensor_normalize_row(row, allow_empty=rows == 0)
                if cols is None:
                    cols = len(normalized)
                    if cols == 0 and rows != 0:
                        raise ValueError("Tensor rows must not be empty")
                elif len(normalized) != cols:
                    raise ValueError("Tensor rows must all share the same length")
                flat.extend(normalized)
            return rows, (0 if cols is None else cols), flat

        flat = [float(value) for value in items]
        return 1, len(flat), flat

    def _normalize_tensor_ctor_args(
        *args: object, **kwargs: object
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

        if rows is None and cols is None:
            rows, cols = inferred_rows, inferred_cols
        elif rows is None:
            if cols is None:
                raise TypeError("Tensor() could not determine rows from provided inputs")
            if cols == 0:
                if total != 0:
                    raise ValueError(
                        f"Tensor data of length {total} cannot fill ({cols}) columns"
                    )
                rows = 0
            else:
                if total % cols != 0:
                    raise ValueError(
                        f"Tensor data of length {total} cannot fill ({cols}) columns"
                    )
                rows = total // cols
        elif cols is None:
            if rows == 0:
                if total != 0:
                    raise ValueError(
                        f"Tensor data of length {total} cannot fill ({rows}) rows"
                    )
                cols = 0
            else:
                if total % rows != 0:
                    raise ValueError(
                        f"Tensor data of length {total} cannot fill ({rows}) rows"
                    )
                cols = total // rows
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

    class _ShapeAccessor(tuple):
        __slots__ = ()

        def __call__(self) -> tuple[int, int]:
            return tuple(self)


    class _ShapeProperty(property):
        def __get__(self, instance, owner=None):  # type: ignore[override]
            if instance is None:
                return self.fget
            values = self.fget(instance)  # type: ignore[misc]
            if not isinstance(values, tuple):
                values = tuple(values)
            return _ShapeAccessor(values)


    class Tensor:
        """Featureful stand-in for the Rust ``Tensor`` exposed by the stub bindings."""

        __slots__ = ("_rows", "_cols", "_data", "_backend")

        def __init__(self, *args, backend: str | None = None, **kwargs):
            backend_hint = backend
            rows, cols, payload = _normalize_tensor_ctor_args(*args, **kwargs)
            if backend_hint is not None and backend_hint not in {"numpy", "python"}:
                raise ValueError("backend must be 'numpy', 'python', or None")
            if backend_hint == "numpy" and not NUMPY_AVAILABLE:
                raise RuntimeError("NumPy backend requested but NumPy is not installed")

            rows = int(rows)
            cols = int(cols)
            if rows < 0 or cols < 0:
                raise ValueError("tensor dimensions must be non-negative")

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

        @_ShapeProperty
        def shape(self):
            return (self._rows, self._cols)

        @property
        def backend(self) -> str:
            return self._backend

        def tolist(self):
            rows, cols = self.shape()
            if self._backend == "python":
                flat = list(self._data)
            else:
                flat = self._data.reshape(-1).tolist()
            if rows == 0 or cols == 0:
                return [[] for _ in range(rows)]
            return [flat[i * cols : (i + 1) * cols] for i in range(rows)]

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


_load_native_package()
del _load_native_package
