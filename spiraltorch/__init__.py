"""Development shim for the SpiralTorch Python bindings.

This module lets ``import spiraltorch`` succeed directly from a source
checkout without first installing the wheel.  It delegates to the real
package that lives under ``bindings/st-py`` and improves the error message
when the compiled extension has not been built yet.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
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


def _install_stub_bindings(module, error: ModuleNotFoundError) -> None:
    warnings.warn(
        "Using SpiralTorch Python stub because the native extension is missing. "
        "Run `maturin develop -m bindings/st-py/Cargo.toml` for the optimized bindings.",
        RuntimeWarning,
        stacklevel=2,
    )

    try:  # Prefer a NumPy-backed shim when available for better performance.
        import numpy as _np  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        _np = None  # type: ignore

    NUMPY_AVAILABLE = _np is not None

    class Tensor:
        """Featureful stand-in for the Rust ``Tensor`` exposed by the stub bindings."""

        __slots__ = ("_rows", "_cols", "_data", "_backend")

        def __init__(self, rows: int, cols: int, data, *, backend: str | None = None):
            backend_hint = backend
            if backend_hint is not None and backend_hint not in {"numpy", "python"}:
                raise ValueError("backend must be 'numpy', 'python', or None")
            if backend_hint == "numpy" and not NUMPY_AVAILABLE:
                raise RuntimeError("NumPy backend requested but NumPy is not installed")

            rows = int(rows)
            cols = int(cols)
            if rows < 0 or cols < 0:
                raise ValueError("tensor dimensions must be non-negative")

            # ``data`` is expected to be a flat iterable of length rows*cols.  Convert once
            # so we can verify the length while remaining agnostic to the original container.
            flat_data = list(data)
            if rows * cols != len(flat_data):
                raise ValueError("data length does not match matrix dimensions")

            self._rows = rows
            self._cols = cols

            preferred_backend = backend_hint or ("numpy" if NUMPY_AVAILABLE else "python")
            if preferred_backend == "numpy":
                arr = _np.asarray(flat_data, dtype=_np.float64).reshape(rows, cols)
                self._data = arr
                self._backend = "numpy"
            else:
                self._data = [float(x) for x in flat_data]
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
            out = [0.0] * (rows * cols)
            left = self._row_major_python()
            right = other._row_major_python()
            for i in range(rows):
                for k in range(inner):
                    aik = left[i * inner + k]
                    if aik == 0.0:
                        continue
                    row_offset = k * cols
                    out_offset = i * cols
                    for j in range(cols):
                        out[out_offset + j] += aik * right[row_offset + j]
            return Tensor(rows, cols, out, backend="python")

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
            out = [a * b for a, b in zip(self._row_major_python(), other._row_major_python())]
            return Tensor(self._rows, self._cols, out, backend="python")

        def numpy(self, *, copy: bool = True):
            if not NUMPY_AVAILABLE:
                raise RuntimeError("NumPy is not available in the stub bindings")
            return self._to_numpy(copy=copy)

        def _to_numpy(self, *, copy: bool) -> "_np.ndarray":
            if not NUMPY_AVAILABLE:
                raise RuntimeError("NumPy is not available in the stub bindings")
            if self._backend == "numpy":
                return self._data.copy() if copy else self._data
            return _np.asarray(self._data, dtype=_np.float64).reshape(self._rows, self._cols)

        def _row_major_python(self):
            if self._backend == "python":
                return self._data
            # ``tolist`` returns a new Python list which is ideal for nested loops.
            return self._data.reshape(-1).tolist()

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

        @property
        def shape(self):
            return (self._rows, self._cols)

        @property
        def backend(self) -> str:
            return self._backend

        def tolist(self):
            if self._backend == "python":
                return list(self._data)
            return self._data.reshape(-1).tolist()

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
            return f"Tensor(shape={self.shape}, backend='{self._backend}')"

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
