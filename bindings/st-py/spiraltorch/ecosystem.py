"""Convenience wrappers around SpiralTorch's ecosystem bridges."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Iterable

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


def _supports_stream_parameter(func: Callable[..., Any]) -> bool | None:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return None

    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.KEYWORD_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            if parameter.name == "stream":
                return True
        elif parameter.kind is parameter.VAR_KEYWORD:
            return True
    return False


def _call_with_optional_stream(
    func: Callable[..., Any],
    positional: Iterable[Any],
    *,
    stream: Any | None,
) -> Any:
    positional = tuple(positional)
    if stream is None:
        return func(*positional)

    supports_stream = _supports_stream_parameter(func)
    if supports_stream:
        return func(*positional, stream=stream)
    if supports_stream is False:
        return func(*positional)

    try:
        return func(*positional, stream=stream)
    except TypeError as exc:
        if "stream" in str(exc):
            return func(*positional)
        raise


def _dlpack_from_array(array: Any, *, stream: Any | None) -> Any:
    if hasattr(array, "__dlpack__"):
        method = getattr(array, "__dlpack__")
        return _call_with_optional_stream(method, (), stream=stream)
    if hasattr(array, "toDlpack"):
        method = getattr(array, "toDlpack")
        return _call_with_optional_stream(method, (), stream=stream)
    if hasattr(array, "to_dlpack"):
        method = getattr(array, "to_dlpack")
        return _call_with_optional_stream(method, (), stream=stream)
    cupy = _require_module("cupy")
    if hasattr(cupy, "toDlpack"):
        function = getattr(cupy, "toDlpack")
        return _call_with_optional_stream(function, (array,), stream=stream)
    if hasattr(cupy, "to_dlpack"):
        function = getattr(cupy, "to_dlpack")
        return _call_with_optional_stream(function, (array,), stream=stream)
    raise TypeError("Object does not expose a DLPack-compatible exporter")


def tensor_to_cupy(tensor: Tensor, *, stream: Any | None = None) -> Any:
    """Share a :class:`~spiraltorch.Tensor` with CuPy via DLPack."""

    cupy = _require_module("cupy")
    exporter = getattr(cupy, "from_dlpack", None)
    if exporter is None:  # pragma: no cover - defensive guard
        raise RuntimeError("cupy.from_dlpack is unavailable")
    capsule = tensor.to_dlpack()
    return _call_with_optional_stream(exporter, (capsule,), stream=stream)


def cupy_to_tensor(array: Any, *, stream: Any | None = None) -> Tensor:
    """Convert a ``cupy.ndarray`` (or compatible object) into a SpiralTorch tensor."""

    capsule = _dlpack_from_array(array, stream=stream)
    return Tensor.from_dlpack(capsule)
