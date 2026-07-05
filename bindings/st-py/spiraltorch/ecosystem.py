"""Convenience wrappers around SpiralTorch's ecosystem bridges."""

from __future__ import annotations

import importlib
import inspect
import numbers
from collections.abc import Mapping
from typing import Any, Callable, Iterable

from . import Tensor

__all__ = [
    "bound_external_state_tensors",
    "checkpoint_from_external_state",
    "external_tensor_last_token",
    "external_tensor_metadata",
    "external_tensor_shape",
    "external_tensor_to_list",
    "slice_external_tensor",
    "tensor_from_external",
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
        compat = importlib.import_module(f"{__package__}.compat")
        module = getattr(compat, name)
    except (AttributeError, ModuleNotFoundError) as exc:  # pragma: no cover - exercised via tests
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


def _looks_like_spiraltorch_tensor(value: Any) -> bool:
    return (
        hasattr(value, "rows")
        and hasattr(value, "cols")
        and callable(getattr(value, "data", None))
    )


def _materialize_external_tensor(value: Any) -> Any:
    tensor = value
    for method_name in ["detach", "cpu", "float"]:
        method = getattr(tensor, method_name, None)
        if callable(method):
            tensor = method()
    return tensor


def _shape_from_sequence(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} has no shape metadata and is not a list/tuple")
    if not value:
        raise ValueError(f"{name} is empty")
    if isinstance(value[0], (list, tuple)):
        child_shape = _shape_from_sequence(value[0], name)
        for row in value:
            if not isinstance(row, (list, tuple)):
                raise ValueError(f"{name} must be a rectangular sequence")
            if _shape_from_sequence(row, name) != child_shape:
                raise ValueError(f"{name} must be a rectangular sequence")
        return (len(value), *child_shape)
    return (len(value),)


def _external_shape_from_materialized(value: Any, name: str) -> tuple[int, ...]:
    if _looks_like_spiraltorch_tensor(value):
        return (int(value.rows), int(value.cols))
    shape = getattr(value, "shape", None)
    if shape is None:
        size = getattr(value, "size", None)
        shape = size() if callable(size) else None
    if shape is None:
        return _shape_from_sequence(value, name)
    return tuple(int(dim) for dim in shape)


def _external_device_kind(device: Any | None) -> str | None:
    if device is None:
        return None
    label = str(device).strip().lower()
    if not label:
        return None
    return label.split(":", 1)[0]


def _external_tensor_backend(value: Any, device: Any | None) -> str:
    if _looks_like_spiraltorch_tensor(value):
        return "spiraltorch"
    if isinstance(value, (list, tuple)):
        return "python_sequence"

    module = type(value).__module__.split(".", 1)[0]
    if module in {"torch", "numpy", "jax", "tensorflow", "cupy", "mlx"}:
        return module
    if callable(getattr(value, "detach", None)) and _external_device_kind(device) in {
        "cpu",
        "cuda",
        "mps",
    }:
        return "torch"
    if module and module != "builtins":
        return module
    return type(value).__name__


def _external_bool_metadata(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return None


def _external_shape_label(shape: tuple[int, ...] | None) -> str | None:
    if shape is None:
        return None
    return "x".join(str(dim) for dim in shape)


def external_tensor_shape(value: Any, *, name: str = "tensor") -> tuple[int, ...]:
    """Return shape metadata for a torch/numpy/list-like external tensor."""

    materialized = _materialize_external_tensor(value)
    return _external_shape_from_materialized(materialized, name)


def external_tensor_metadata(value: Any, *, name: str = "tensor") -> dict[str, Any]:
    """Return non-materializing metadata for a torch/numpy/list-like tensor.

    Unlike conversion helpers, this intentionally avoids ``cpu()``/``float()`` so
    traces can preserve the runtime/device where an external tensor originated.
    """

    shape = None
    shape_error = None
    try:
        shape = _external_shape_from_materialized(value, name)
    except (TypeError, ValueError) as exc:
        shape_error = f"{exc.__class__.__name__}: {exc}"

    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    requires_grad = getattr(value, "requires_grad", None)
    module = type(value).__module__
    return {
        "name": name,
        "type": type(value).__name__,
        "module": module,
        "backend": _external_tensor_backend(value, device),
        "shape": shape,
        "shape_label": _external_shape_label(shape),
        "shape_rank": None if shape is None else len(shape),
        "shape_error": shape_error,
        "dtype": None if dtype is None else str(dtype),
        "device": None if device is None else str(device),
        "device_kind": _external_device_kind(device),
        "requires_grad": _external_bool_metadata(requires_grad),
    }


def _flatten_sequence(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        return [item for row in value for item in row]
    return list(value)


def _flatten_nested_sequence(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_nested_sequence(item))
        return flattened
    return [value]


def _flat_external_data(value: Any, name: str) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return _flatten_sequence(value)
    reshaped = None
    reshape = getattr(value, "reshape", None)
    if callable(reshape):
        try:
            reshaped = reshape(-1)
        except TypeError:
            reshaped = None
    if reshaped is None:
        flatten = getattr(value, "flatten", None)
        if callable(flatten):
            reshaped = flatten()
    source = reshaped if reshaped is not None else value
    tolist = getattr(source, "tolist", None)
    if callable(tolist):
        data = tolist()
    else:
        data = source
    if isinstance(data, (list, tuple)):
        return _flatten_sequence(data)
    raise TypeError(f"{name} data could not be flattened into a Python sequence")


def _slice_sequence_external_tensor(
    value: Any,
    name: str,
    shape: tuple[int, ...],
    rows: int,
    cols: int,
) -> list[Any]:
    if len(shape) == 1:
        limit = min(shape[0], cols)
        return list(value[:limit])
    row_limit = min(shape[0], rows)
    col_limit = min(shape[1], cols)
    sliced = []
    for row in value[:row_limit]:
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"{name} must be a rectangular 2D sequence")
        sliced.append(list(row[:col_limit]))
    return sliced


def _slice_spiraltorch_tensor(
    value: Any,
    shape: tuple[int, ...],
    rows: int,
    cols: int,
) -> Tensor:
    if len(shape) == 1:
        limit = min(shape[0], cols)
        return Tensor(1, limit, value.data()[:limit])
    row_limit = min(shape[0], rows)
    col_limit = min(shape[1], cols)
    data = []
    source = value.data()
    for row in range(row_limit):
        offset = row * value.cols
        data.extend(source[offset : offset + col_limit])
    return Tensor(row_limit, col_limit, data)


def slice_external_tensor(
    value: Any,
    *,
    rows: int,
    cols: int,
    name: str = "tensor",
) -> Any:
    """Return a bounded slice/copy of a torch/numpy/list-like tensor.

    This keeps checkpoint and Transformers preflight code from materializing a
    full model-sized tensor when only the overlapping SpiralTorch block matters.
    """

    if rows <= 0 or cols <= 0:
        raise ValueError(f"{name} bounds must be positive, got {(rows, cols)!r}")
    materialized = _materialize_external_tensor(value)
    shape = _external_shape_from_materialized(materialized, name)
    if len(shape) not in {1, 2}:
        raise ValueError(f"{name} must be 1D or 2D to slice, got shape={shape}")
    if isinstance(materialized, (list, tuple)):
        return _slice_sequence_external_tensor(materialized, name, shape, rows, cols)
    if _looks_like_spiraltorch_tensor(materialized):
        return _slice_spiraltorch_tensor(materialized, shape, rows, cols)

    if len(shape) == 1:
        limit = min(shape[0], cols)
        try:
            return materialized[:limit]
        except TypeError:
            pass
    else:
        row_limit = min(shape[0], rows)
        col_limit = min(shape[1], cols)
        try:
            return materialized[:row_limit, :col_limit]
        except TypeError:
            pass

    tolist = getattr(materialized, "tolist", None)
    if callable(tolist):
        return _slice_sequence_external_tensor(tolist(), name, shape, rows, cols)
    raise TypeError(f"{name} could not be sliced before tensor materialization")


def bound_external_state_tensors(
    state: Mapping[str, Any],
    tensor_bounds: Mapping[str, tuple[int, int]] | None,
) -> dict[str, Any]:
    """Slice selected external checkpoint tensors before conversion."""

    if not tensor_bounds:
        return dict(state)
    bounded = {}
    for name, value in state.items():
        bound = tensor_bounds.get(name)
        if bound is None:
            bounded[name] = value
            continue
        rows, cols = bound
        bounded[name] = slice_external_tensor(value, rows=rows, cols=cols, name=name)
    return bounded


def _external_numeric_value(value: Any, name: str, index: int) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} contains boolean checkpoint value at index {index}")
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} contains non-numeric checkpoint value at index {index}")
    return float(value)


def external_tensor_to_list(value: Any, *, name: str = "tensor") -> list[float]:
    """Materialize and flatten a torch/numpy/list-like tensor into floats."""

    materialized = _materialize_external_tensor(value)
    tolist = getattr(materialized, "tolist", None)
    if callable(tolist):
        materialized = tolist()
    values = _flatten_nested_sequence(materialized)
    return [
        _external_numeric_value(item, name, index)
        for index, item in enumerate(values)
    ]


def external_tensor_last_token(value: Any, *, name: str = "tensor") -> Any:
    """Select the final token row from a 2D/3D tensor-like value.

    Transformers logits commonly arrive as ``(batch, sequence, vocab)`` and
    hidden states as ``(batch, sequence, hidden)``. This helper extracts the
    last sequence vector while keeping unknown-shape values unchanged.
    """

    materialized = _materialize_external_tensor(value)
    try:
        shape = _external_shape_from_materialized(materialized, name)
    except (TypeError, ValueError):
        return materialized

    if len(shape) == 3:
        try:
            return materialized[0, shape[1] - 1, :]
        except (TypeError, IndexError):
            return materialized[0][shape[1] - 1]
    if len(shape) == 2:
        try:
            return materialized[shape[0] - 1, :]
        except (TypeError, IndexError):
            return materialized[shape[0] - 1]
    return materialized


def tensor_from_external(value: Any, *, name: str = "tensor") -> Tensor:
    """Convert a torch/numpy/list-like 1D or 2D tensor into ``Tensor``.

    2D arrays become ``(rows, cols)`` tensors. 1D arrays are imported as a
    single row so common bias vectors and last-token hidden vectors are easy to
    route into Z-Space probes.
    """

    if _looks_like_spiraltorch_tensor(value):
        return value
    materialized = _materialize_external_tensor(value)
    shape = _external_shape_from_materialized(materialized, name)
    if len(shape) == 1:
        rows, cols = 1, shape[0]
    elif len(shape) == 2:
        rows, cols = shape
    else:
        raise ValueError(
            f"{name} must be 1D or 2D for SpiralTorch conversion, got shape={shape}"
        )
    data = [
        _external_numeric_value(item, name, index)
        for index, item in enumerate(_flat_external_data(materialized, name))
    ]
    expected = rows * cols
    if len(data) != expected:
        raise ValueError(f"{name} shape={shape} expects {expected} values, got {len(data)}")
    return Tensor(rows, cols, data)


def checkpoint_from_external_state(
    state: Mapping[str, Any],
    *,
    include: Iterable[str] | None = None,
    tensor_bounds: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, Tensor]:
    """Convert selected external checkpoint values into SpiralTorch tensors."""

    include_names = None if include is None else set(include)
    selected_state = {
        name: value
        for name, value in state.items()
        if include_names is None or name in include_names
    }
    bounded_state = bound_external_state_tensors(selected_state, tensor_bounds)
    checkpoint = {}
    for name, value in bounded_state.items():
        checkpoint[name] = tensor_from_external(value, name=name)
    return checkpoint


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
