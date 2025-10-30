"""Interoperability helpers for popular ML ecosystems.

These helpers rely on DLPack interchange to bridge SpiralTorch tensors with
external frameworks without materialising data copies when possible.
"""

from __future__ import annotations

import importlib
from typing import Any, TYPE_CHECKING

from . import Tensor

if TYPE_CHECKING:  # pragma: no cover - optional dependency hints
    import torch  # noqa: F401
    import jax  # noqa: F401

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


def _require_module(name: str) -> Any:
    """Import *name* or raise a descriptive runtime error."""

    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised indirectly in tests
        raise RuntimeError(
            f"Optional dependency '{name}' is required for this interoperability helper."
        ) from exc


def tensor_to_torch(
    tensor: Tensor,
    *,
    device: Any | None = None,
    dtype: Any | None = None,
    copy: bool | None = None,
) -> Any:
    """Convert a :class:`~spiraltorch.Tensor` into a ``torch.Tensor``.

    Args:
        tensor: The SpiralTorch tensor to convert.
        device: Optional device placement forwarded to ``torch.Tensor.to``.
        dtype: Optional dtype forwarded to ``torch.Tensor.to``.
        copy: If provided, forwarded to ``torch.Tensor.to``'s ``copy`` argument.

    Returns:
        A ``torch.Tensor`` sharing storage with the SpiralTorch tensor whenever
        the backend supports zero-copy DLPack interchange.
    """

    torch = _require_module("torch")
    utils = getattr(torch, "utils", None)
    dlpack_mod = getattr(utils, "dlpack", None) if utils is not None else None
    if dlpack_mod is None or not hasattr(dlpack_mod, "from_dlpack"):
        raise RuntimeError("torch.utils.dlpack.from_dlpack is unavailable")

    capsule = tensor.to_dlpack()
    torch_tensor = dlpack_mod.from_dlpack(capsule)

    kwargs: dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None:
        kwargs["dtype"] = dtype
    if copy is not None:
        kwargs["copy"] = copy
    if kwargs:
        torch_tensor = torch_tensor.to(**kwargs)
    return torch_tensor


def torch_to_tensor(tensor: Any, *, clone: bool = False) -> Tensor:
    """Convert a ``torch.Tensor`` into a :class:`~spiraltorch.Tensor`.

    Args:
        tensor: The source ``torch.Tensor`` instance.
        clone: Whether to clone the PyTorch tensor before exporting to DLPack.

    Returns:
        A SpiralTorch :class:`~spiraltorch.Tensor` containing the same values.
    """

    torch = _require_module("torch")
    if not isinstance(tensor, getattr(torch, "Tensor", ())):
        raise TypeError("torch_to_tensor expects a torch.Tensor instance")

    utils = getattr(torch, "utils", None)
    dlpack_mod = getattr(utils, "dlpack", None) if utils is not None else None
    if dlpack_mod is None:
        raise RuntimeError("torch.utils.dlpack is unavailable")

    candidate = tensor.clone() if clone else tensor

    if hasattr(dlpack_mod, "to_dlpack"):
        capsule = dlpack_mod.to_dlpack(candidate)
    elif hasattr(candidate, "to_dlpack"):
        capsule = candidate.to_dlpack()
    elif hasattr(candidate, "__dlpack__"):
        capsule = candidate.__dlpack__()
    else:  # pragma: no cover - defensive guard
        raise RuntimeError("Torch tensor does not expose a DLPack exporter")

    return Tensor.from_dlpack(capsule)


def tensor_to_jax(tensor: Tensor, *, device: Any | None = None) -> Any:
    """Convert a :class:`~spiraltorch.Tensor` into a ``jax.Array``."""

    jax_dlpack = _require_module("jax.dlpack")
    capsule = tensor.to_dlpack()
    jax_array = jax_dlpack.from_dlpack(capsule)
    if device is None:
        return jax_array

    jax = _require_module("jax")
    return jax.device_put(jax_array, device)


def jax_to_tensor(array: Any) -> Tensor:
    """Convert a ``jax.Array`` (or compatible object) into a SpiralTorch tensor."""

    jax_dlpack = _require_module("jax.dlpack")
    if hasattr(jax_dlpack, "to_dlpack"):
        capsule = jax_dlpack.to_dlpack(array)
    elif hasattr(array, "__dlpack__"):
        capsule = array.__dlpack__()
    else:  # pragma: no cover - defensive guard
        raise TypeError("Object does not expose a DLPack-compatible exporter")
    return Tensor.from_dlpack(capsule)


def tensor_to_cupy(tensor: Tensor) -> Any:
    """Convert a :class:`~spiraltorch.Tensor` into a ``cupy.ndarray``."""

    cupy = _require_module("cupy")
    if not hasattr(cupy, "from_dlpack"):
        raise RuntimeError("cupy.from_dlpack is unavailable")

    capsule = tensor.to_dlpack()
    return cupy.from_dlpack(capsule)


def cupy_to_tensor(array: Any) -> Tensor:
    """Convert a ``cupy.ndarray`` (or compatible object) into a SpiralTorch tensor."""

    cupy = _require_module("cupy")
    if hasattr(array, "toDlpack"):
        capsule = array.toDlpack()  # legacy CuPy exporter name
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


def tensor_to_tensorflow(tensor: Tensor) -> Any:
    """Convert a :class:`~spiraltorch.Tensor` into a ``tf.Tensor``."""

    tf = _require_module("tensorflow")
    dlpack = getattr(tf.experimental, "dlpack", None)
    if dlpack is None or not hasattr(dlpack, "from_dlpack"):
        raise RuntimeError("tensorflow.experimental.dlpack.from_dlpack is unavailable")

    capsule = tensor.to_dlpack()
    return dlpack.from_dlpack(capsule)


def tensorflow_to_tensor(tensor: Any) -> Tensor:
    """Convert a ``tf.Tensor`` (or compatible object) into a SpiralTorch tensor."""

    tf = _require_module("tensorflow")
    dlpack = getattr(tf.experimental, "dlpack", None)
    if dlpack is None:
        raise RuntimeError("tensorflow.experimental.dlpack is unavailable")

    if hasattr(dlpack, "to_dlpack"):
        capsule = dlpack.to_dlpack(tensor)
    elif hasattr(tensor, "__dlpack__"):
        capsule = tensor.__dlpack__()
    else:  # pragma: no cover - defensive guard
        raise TypeError("Object does not expose a DLPack-compatible exporter")
    return Tensor.from_dlpack(capsule)
