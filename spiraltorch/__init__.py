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

    class Tensor:
        """Minimal pure Python stand-in for the Rust Tensor."""

        __slots__ = ("_rows", "_cols", "_data")

        def __init__(self, rows: int, cols: int, data):
            if rows * cols != len(data):
                raise ValueError("data length does not match matrix dimensions")
            self._rows = int(rows)
            self._cols = int(cols)
            self._data = [float(x) for x in data]

        def matmul(self, other: "Tensor", *, backend: str | None = None):  # noqa: D401 - mirror real signature
            if self._cols != other._rows:
                raise ValueError("inner dimensions do not match for matmul")
            rows, cols, inner = self._rows, other._cols, self._cols
            out = [0.0] * (rows * cols)
            for i in range(rows):
                for k in range(inner):
                    aik = self._data[i * inner + k]
                    if aik == 0.0:
                        continue
                    row_offset = k * other._cols
                    out_offset = i * cols
                    for j in range(cols):
                        out[out_offset + j] += aik * other._data[row_offset + j]
            return Tensor(rows, cols, out)

        def hadamard(self, other: "Tensor"):
            if self.shape != other.shape:
                raise ValueError("tensor shapes must match for hadamard product")
            out = [a * b for a, b in zip(self._data, other._data)]
            return Tensor(self._rows, self._cols, out)

        @property
        def shape(self):
            return (self._rows, self._cols)

        def tolist(self):
            return list(self._data)

        def __repr__(self) -> str:  # pragma: no cover - debugging helper
            return f"Tensor(shape={self.shape})"

    Tensor.__module__ = module.__name__

    module.Tensor = Tensor
    all_exports = module.__dict__.setdefault("__all__", [])
    if "Tensor" not in all_exports:
        all_exports.append("Tensor")
    module.__dict__.setdefault("__version__", "0.0.0+stub")

    def _missing_attr(name: str):
        raise AttributeError(
            f"spiraltorch.{name} is unavailable in the stub bindings. Build the native extension "
            "for full functionality."
        )

    def __getattr__(name: str):  # pragma: no cover - dynamic attribute guard
        if name in {"Tensor", "__all__", "__version__"}:
            return module.__dict__[name]
        raise _missing_attr(name)

    module.__getattr__ = __getattr__


_load_native_package()
del _load_native_package
