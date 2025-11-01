"""Convenience wrappers around SpiralTorch's ecosystem bridges."""

from __future__ import annotations

import inspect
import numbers
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


def _coerce_stream_pointer(value: Any | None) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)

    attr_value = getattr(value, "value", None)
    if isinstance(attr_value, numbers.Integral):
        return int(attr_value)

    converter = getattr(value, "__int__", None)
    if converter is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _resolve_cupy_stream(stream: Any | None, *, cupy: Any) -> Any:
    if stream is None:
        return None

    cuda = getattr(cupy, "cuda", None)

    if hasattr(stream, "ptr"):
        return stream

    if isinstance(stream, str):
        keyword = stream.lower()
        if keyword in {"current", "auto"}:
            if cuda is None or not hasattr(cuda, "get_current_stream"):
                raise RuntimeError(
                    "CuPy does not expose cuda.get_current_stream; cannot resolve 'current' stream alias."
                )
            return cuda.get_current_stream()
        if keyword in {"null", "default"}:
            stream_class = getattr(cuda, "Stream", None) if cuda is not None else None
            null_stream = getattr(stream_class, "null", None)
            if null_stream is None:
                raise RuntimeError(
                    "CuPy does not expose cuda.Stream.null; cannot resolve 'null' stream alias."
                )
            return null_stream
        raise ValueError(f"Unknown CuPy stream alias: {stream!r}")

    pointer = _coerce_stream_pointer(stream)
    if pointer is not None:
        external_stream = getattr(cuda, "ExternalStream", None) if cuda is not None else None
        if external_stream is None:
            raise RuntimeError(
                "CuPy does not expose cuda.ExternalStream; cannot wrap raw stream pointer."
            )
        return external_stream(pointer)

    return stream


def _dlpack_from_array(array: Any, *, stream: Any | None, cupy_module: Any | None = None) -> Any:
    if hasattr(array, "__dlpack__"):
        method = getattr(array, "__dlpack__")
        return _call_with_optional_stream(method, (), stream=stream)
    if hasattr(array, "toDlpack"):
        method = getattr(array, "toDlpack")
        return _call_with_optional_stream(method, (), stream=stream)
    if hasattr(array, "to_dlpack"):
        method = getattr(array, "to_dlpack")
        return _call_with_optional_stream(method, (), stream=stream)
    cupy = cupy_module or _require_module("cupy")
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
    stream = _resolve_cupy_stream(stream, cupy=cupy)
    exporter = getattr(cupy, "from_dlpack", None)
    if exporter is None:  # pragma: no cover - defensive guard
        raise RuntimeError("cupy.from_dlpack is unavailable")
    capsule = tensor.to_dlpack()
    return _call_with_optional_stream(exporter, (capsule,), stream=stream)


def cupy_to_tensor(array: Any, *, stream: Any | None = None) -> Tensor:
    """Convert a ``cupy.ndarray`` (or compatible object) into a SpiralTorch tensor."""

    cupy = _require_module("cupy")
    stream = _resolve_cupy_stream(stream, cupy=cupy)
    capsule = _dlpack_from_array(array, stream=stream, cupy_module=cupy)
    return Tensor.from_dlpack(capsule)
