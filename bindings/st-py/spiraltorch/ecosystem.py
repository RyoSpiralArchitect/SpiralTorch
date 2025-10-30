"""Convenience wrappers around SpiralTorch's ecosystem bridges."""

from __future__ import annotations

from typing import Any

from . import Tensor, compat

__all__ = [
    "tensor_to_torch",
    "torch_to_tensor",
    "tensor_to_jax",
    "jax_to_tensor",
    "tensor_to_cupy",
    "cupy_to_tensor",
    "tensor_to_tensorflow",
    "tensorflow_to_tensor",
]

_NATIVE_EXTENSION_HINT = (
    "Build the SpiralTorch native extension (e.g. `maturin develop -m "
    "bindings/st-py/Cargo.toml`) to enable spiraltorch.compat helpers."
)


def _compat_namespace(name: str) -> Any:
    """Return a compat child module or raise a descriptive error."""

    try:
        module = getattr(compat, name)
    except AttributeError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(
            f"spiraltorch.compat.{name} is unavailable. {_NATIVE_EXTENSION_HINT}"
        ) from exc
    return module


def _compat_call(namespace: str, attr: str, *args: Any, **kwargs: Any) -> Any:
    module = _compat_namespace(namespace)
    try:
        func = getattr(module, attr)
    except AttributeError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(
            f"spiraltorch.compat.{namespace}.{attr} is unavailable. {_NATIVE_EXTENSION_HINT}"
        ) from exc
    return func(*args, **kwargs)


def tensor_to_torch(
    tensor: Tensor,
    *,
    dtype: Any | None = None,
    device: Any | None = None,
    requires_grad: bool | None = None,
    copy: bool | None = None,
    memory_format: Any | None = None,
) -> Any:
    """Share a :class:`~spiraltorch.Tensor` with PyTorch."""

    return _compat_call(
        "torch",
        "to_torch",
        tensor,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        copy=copy,
        memory_format=memory_format,
    )


def torch_to_tensor(
    tensor: Any,
    *,
    dtype: Any | None = None,
    device: Any | None = None,
    ensure_cpu: bool | None = None,
    copy: bool | None = None,
    require_contiguous: bool | None = None,
) -> Tensor:
    """Convert a ``torch.Tensor`` into a SpiralTorch tensor."""

    return _compat_call(
        "torch",
        "from_torch",
        tensor,
        dtype=dtype,
        device=device,
        ensure_cpu=ensure_cpu,
        copy=copy,
        require_contiguous=require_contiguous,
    )


def tensor_to_jax(tensor: Tensor) -> Any:
    """Share a :class:`~spiraltorch.Tensor` with JAX."""

    return _compat_call("jax", "to_jax", tensor)


def jax_to_tensor(array: Any) -> Tensor:
    """Convert a ``jax.Array`` (or compatible object) into a SpiralTorch tensor."""

    return _compat_call("jax", "from_jax", array)


def tensor_to_tensorflow(tensor: Tensor) -> Any:
    """Share a :class:`~spiraltorch.Tensor` with TensorFlow."""

    return _compat_call("tensorflow", "to_tensorflow", tensor)


def tensorflow_to_tensor(value: Any) -> Tensor:
    """Convert a ``tf.Tensor`` (or compatible object) into a SpiralTorch tensor."""

    return _compat_call("tensorflow", "from_tensorflow", value)


def _require_module(name: str) -> Any:
    try:
        return __import__(name)
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(
            f"Optional dependency '{name}' is required for this interoperability helper."
        ) from exc


def tensor_to_cupy(tensor: Tensor) -> Any:
    """Share a :class:`~spiraltorch.Tensor` with CuPy via DLPack."""

    cupy = _require_module("cupy")
    exporter = getattr(cupy, "from_dlpack", None)
    if exporter is None:  # pragma: no cover - defensive guard
        raise RuntimeError("cupy.from_dlpack is unavailable")
    capsule = tensor.to_dlpack()
    return exporter(capsule)


def cupy_to_tensor(array: Any) -> Tensor:
    """Convert a ``cupy.ndarray`` (or compatible object) into a SpiralTorch tensor."""

    cupy = _require_module("cupy")
    if hasattr(array, "toDlpack"):
        capsule = array.toDlpack()
    elif hasattr(array, "to_dlpack"):
        capsule = array.to_dlpack()
    elif hasattr(cupy, "toDlpack"):
        capsule = cupy.toDlpack(array)
    elif hasattr(cupy, "to_dlpack"):
        capsule = cupy.to_dlpack(array)
    elif hasattr(array, "__dlpack__"):
        capsule = array.__dlpack__()
    else:  # pragma: no cover - defensive guard
        raise TypeError("Object does not expose a DLPack-compatible exporter")
    return Tensor.from_dlpack(capsule)
