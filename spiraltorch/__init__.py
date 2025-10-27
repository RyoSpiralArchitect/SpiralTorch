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


    class Tensor:
        """Featureful stand-in for the Rust ``Tensor`` exposed by the stub bindings."""

        __slots__ = ("_rows", "_cols", "_data", "_backend")

        def __init__(
            self,
            rows: int | tuple[int, int],
            cols: int | None = None,
            data=None,
            *,
            backend: str | None = None,
        ):
            backend_hint = backend
            rows, cols, payload = _normalize_tensor_ctor_args(*args, **kwargs)
            if backend_hint is not None and backend_hint not in {"numpy", "python"}:
                raise ValueError("backend must be 'numpy', 'python', or None")
            if backend_hint == "numpy" and not NUMPY_AVAILABLE:
                raise RuntimeError("NumPy backend requested but NumPy is not installed")

            if isinstance(rows, tuple) and cols is None and data is None:
                if len(rows) != 2:
                    raise ValueError("shape tuple must describe a 2D tensor")
                inferred_rows, inferred_cols = rows
                rows = int(inferred_rows)
                cols = int(inferred_cols)
                data = [0.0] * (rows * cols)

            if cols is None or data is None:
                raise TypeError(
                    "Tensor constructor expects rows, cols, and data or a shape tuple"
                )

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

        def tolist(self):
            rows, cols = self._rows, self._cols
            if self._backend == "python":
                flat = list(self._data)
                matrix = (flat[r * cols : (r + 1) * cols] for r in range(rows))
                return [row[:] for row in matrix]
            return self._data.reshape(rows, cols).tolist()

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


_load_native_package()
del _load_native_package
