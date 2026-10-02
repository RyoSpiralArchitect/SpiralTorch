"""First-order Torch transport for Rust-owned geometric dynamics.

This initial bridge executes f32 on the CPU, including for GPU inputs. Copies
are explicit overhead, not a resident accelerator implementation. The Rust
operator, not this module, owns the recurrence and its exact finite-unroll VJP.
"""

from __future__ import annotations

import json
import math
from typing import Any

try:
    import torch
except ImportError:  # Optional dependency; importing SpiralTorch remains valid.
    torch = None

__all__ = ["ToposResonatorAdapter", "topos_resonator_autograd"]
_SCHEMA = "spiraltorch.topos_resonator_adapter.v1"


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for geometric autograd adapters")


def _kernel(**options: Any) -> Any:
    from . import ToposResonatorKernel

    return ToposResonatorKernel(**options)


def _strength(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("strength must be finite and in [0, 1]")
    return value


def _input(value: Any) -> None:
    _require_torch()
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        raise TypeError("geometric autograd currently requires float32 tensors")
    if value.ndim < 1 or value.shape[-1] == 0:
        raise ValueError("input needs a nonempty final feature axis")


def _values(tensor: Any) -> list[float]:
    return tensor.detach().to(device="cpu").reshape(-1).tolist()


if torch is not None:

    class _ToposResonatorFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, value: Any, gate: Any, kernel: Any) -> Any:
            features = value.shape[-1]
            rows = value.numel() // features
            output = kernel.forward(_values(value), _values(gate), rows, features)
            ctx.save_for_backward(value, gate)
            ctx.kernel = kernel
            ctx.rows = rows
            ctx.features = features
            return torch.tensor(output, device=value.device, dtype=value.dtype).reshape(
                value.shape
            )

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, Any, None]:
            value, gate = ctx.saved_tensors
            dx, dg = ctx.kernel.backward(
                _values(value),
                _values(gate),
                _values(grad_output),
                ctx.rows,
                ctx.features,
            )
            return (
                torch.tensor(dx, device=value.device, dtype=value.dtype).reshape(
                    value.shape
                ),
                torch.tensor(dg, device=gate.device, dtype=gate.dtype).reshape(
                    gate.shape
                ),
                None,
            )


def topos_resonator_autograd(value: Any, gate: Any, *, kernel: Any = None) -> Any:
    """Apply the Rust resonance with first-order gradients for input and gate.

    The gate may broadcast to the input. PyTorch reduces its returned VJP
    through that broadcast; the geometric derivative itself is computed in Rust.
    Float32 only; higher-order gradients, AMP and compilation are not promised.
    """
    _input(value)
    if (
        not isinstance(gate, torch.Tensor)
        or gate.dtype != value.dtype
        or gate.device != value.device
    ):
        raise TypeError("gate must be a float32 tensor on the input device")
    expanded_gate = gate.expand_as(value)
    if kernel is None:
        kernel = _kernel()
    from . import ToposResonatorKernel

    if not isinstance(kernel, ToposResonatorKernel):
        raise TypeError("kernel must be an immutable Rust ToposResonatorKernel")
    if value.numel() > kernel.max_values:
        raise ValueError("input exceeds the geometric kernel's value budget")
    return _ToposResonatorFunction.apply(value, expanded_gate, kernel)


if torch is None:

    class ToposResonatorAdapter:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_torch()

else:

    class ToposResonatorAdapter(torch.nn.Module):
        """Pointwise causal residual: ``x + strength * resonance(x, gate)``.

        A per-feature trainable gate starts at zero, so initial outputs are the
        original hidden states while gate gradients can already be nonzero.
        Place this explicitly after a tensor-valued block; it does not discover
        or monkey-patch model layers. It never mixes tokens or padding positions.
        ``state_dict`` carries both learned gates and the canonical Rust recipe.
        """

        def __init__(
            self, features: int, *, strength: float = 0.1, **kernel_options: Any
        ) -> None:
            super().__init__()
            if (
                isinstance(features, bool)
                or not isinstance(features, int)
                or features <= 0
            ):
                raise ValueError("features must be a positive integer")
            self.features = features
            self.strength = _strength(strength)
            self._kernel = _kernel(**kernel_options)
            self.gate = torch.nn.Parameter(torch.zeros(features, dtype=torch.float32))

        @property
        def execution_backend(self) -> str:
            return self._kernel.execution_backend

        def forward(self, value: Any) -> Any:
            _input(value)
            if value.shape[-1] != self.features:
                raise ValueError("input feature width differs from the adapter")
            strength = _strength(self.strength)
            if strength == 0.0:
                return value
            return value + strength * topos_resonator_autograd(
                value, self.gate, kernel=self._kernel
            )

        def get_extra_state(self) -> dict[str, Any]:
            return {
                "schema": _SCHEMA,
                "features": self.features,
                "strength": _strength(self.strength),
                "kernel": json.loads(self._kernel.configuration_json()),
            }

        def set_extra_state(self, state: dict[str, Any]) -> None:
            if (
                not isinstance(state, dict)
                or set(state) != {"schema", "features", "strength", "kernel"}
                or state["schema"] != _SCHEMA
                or state["features"] != self.features
            ):
                raise ValueError("incompatible geometric adapter state")
            strength = _strength(state["strength"])
            kernel = _kernel(**state["kernel"])
            self.strength = strength
            self._kernel = kernel
