"""First-order transport for Rust's causal fractional difference, not a new rule.

CPU f32 host copies are explicit. Sequence adapters require complete unpadded
prefixes, no KV cache, and no packed-document boundaries within one sequence.
"""

from __future__ import annotations

import json
import math
from typing import Any

from .geometry_autograd import _input, _require_torch, _strength, _values, torch

__all__ = ["fractional_gl_autograd", "fractional_gl_history_autograd",
           "FractionalMemoryAdapter", "FractionalHistoryAdapter"]


def _kernel(**options: Any) -> Any:
    from . import FractionalGlKernel

    return FractionalGlKernel(**options)


if torch is not None:
    class _FractionalGlFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, kernel: Any, value: Any, alpha: Any, axis: int,
                    history_only: bool) -> Any:
            operation = kernel.forward_history if history_only else kernel.forward
            ctx.snapshot = operation(_values(value), list(value.shape), axis, alpha.detach().item())
            ctx.save_for_backward(value, alpha)
            ctx.save_for_forward(value, alpha)
            return torch.tensor(ctx.snapshot.output, dtype=value.dtype, device=value.device).reshape_as(value)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx: Any, upstream: Any) -> tuple[Any, ...]:
            value, alpha = ctx.saved_tensors
            need_input, need_alpha = ctx.needs_input_grad[1:3]
            dx = da = None
            if need_input and need_alpha:
                dx, da = ctx.snapshot.vjp(_values(upstream))
            elif need_input:
                dx = ctx.snapshot.vjp_input(_values(upstream))
            elif need_alpha:
                da = ctx.snapshot.vjp_alpha(_values(upstream))
            return (
                None,
                None if dx is None else torch.tensor(dx, dtype=value.dtype, device=value.device).reshape_as(value),
                None if da is None else torch.tensor(da, dtype=alpha.dtype, device=alpha.device),
                None,
                None,
            )

        @staticmethod
        def jvp(ctx: Any, _kernel: Any, dx: Any, da: Any, _axis: Any,
                _history_only: Any) -> Any:
            value, _ = ctx.saved_tensors
            dx = torch.zeros_like(value) if dx is None else dx
            result = ctx.snapshot.jvp(_values(dx), 0.0 if da is None else da.detach().item())
            return torch.tensor(result, dtype=value.dtype, device=value.device).reshape_as(value)


def fractional_gl_autograd(value: Any, alpha: Any, *, axis: int, kernel: Any = None) -> Any:
    """Causal zero-padded GL on an ND axis, with true input and scalar-alpha VJPs.

    First-order reverse/forward AD only; no higher-order, vmap or torch.func.
    Alpha must be finite and positive. The Rust kernel owns h^-alpha scaling.
    """
    return _fractional_apply(value, alpha, axis, kernel, False)


def fractional_gl_history_autograd(value: Any, alpha: Any, *, axis: int,
                                  kernel: Any = None) -> Any:
    """Strictly past GL taps, with Rust-owned input/order VJPs and joint JVP.

    The zero-lag tap and its alpha derivative are removed before convolution,
    not by subtracting two rounded full outputs. Same limits as full GL AD.
    """
    return _fractional_apply(value, alpha, axis, kernel, True)


def _fractional_apply(value: Any, alpha: Any, axis: int, kernel: Any,
                      history_only: bool) -> Any:
    _input(value)
    if not isinstance(alpha, torch.Tensor) or alpha.ndim != 0 or alpha.dtype != torch.float32:
        raise TypeError("alpha must be a scalar float32 tensor")
    if isinstance(axis, bool) or not isinstance(axis, int) or not 0 <= axis < value.ndim:
        raise ValueError("axis must be a valid nonnegative integer")
    from . import FractionalGlKernel

    kernel = _kernel() if kernel is None else kernel
    if not isinstance(kernel, FractionalGlKernel):
        raise TypeError("kernel must be an immutable Rust FractionalGlKernel")
    return _FractionalGlFunction.apply(kernel, value, alpha, axis, history_only)


if torch is None:
    class FractionalMemoryAdapter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()

    class FractionalHistoryAdapter(FractionalMemoryAdapter):
        pass
else:
    class FractionalMemoryAdapter(torch.nn.Module):
        """Identity-initialized [batch,time,feature] residual with learnable order.

        ``x + strength*tanh(gate)*D_alpha(x)``, alpha=exp(log_alpha).
        Ordinary Torch gates/reparameterization surround the Rust GL map. No
        hidden cache or trainer hook is installed; use complete unpadded prefixes
        with the base model's use_cache=False. State carries the exact GL recipe.
        """

        _state_schema = "spiraltorch.fractional_memory_adapter.v1"

        def __init__(self, features: int, *, initial_alpha: float = 0.5,
                     strength: float = 0.1, **kernel_options: Any) -> None:
            super().__init__()
            if type(features) is not int or features <= 0:
                raise ValueError("features must be a positive integer")
            if not math.isfinite(initial_alpha) or initial_alpha <= 0:
                raise ValueError("initial_alpha must be finite and positive")
            self.features = features
            self.strength = _strength(strength)
            self._kernel = _kernel(**kernel_options)
            self.gate = torch.nn.Parameter(torch.zeros(features, dtype=torch.float32))
            self.log_alpha = torch.nn.Parameter(torch.tensor(math.log(initial_alpha), dtype=torch.float32))

        @property
        def execution_backend(self) -> str:
            return self._kernel.execution_backend

        def _checked_input(self, value: Any) -> float:
            _input(value)
            if value.ndim != 3 or value.shape[-1] != self.features:
                raise ValueError("fractional memory needs [batch, time, features]")
            if self.gate.dtype != value.dtype or self.log_alpha.dtype != value.dtype:
                raise TypeError("fractional memory parameters must remain float32")
            if value.device != self.gate.device or value.device != self.log_alpha.device:
                raise ValueError("input and adapter parameters must share a device")
            return _strength(self.strength)

        def forward(self, value: Any) -> Any:
            strength = self._checked_input(value)
            if strength == 0.0:
                return value
            difference = fractional_gl_autograd(value, self.log_alpha.exp(), axis=1, kernel=self._kernel)
            return value + strength * self.gate.tanh() * difference

        def get_extra_state(self) -> dict[str, Any]:
            return {"schema": self._state_schema,
                    "features": self.features, "strength": _strength(self.strength),
                    "kernel": json.loads(self._kernel.configuration_json())}

        def set_extra_state(self, state: dict[str, Any]) -> None:
            if (not isinstance(state, dict)
                    or set(state) != {"schema", "features", "strength", "kernel"}
                    or state["schema"] != self._state_schema
                    or state["features"] != self.features):
                raise ValueError("incompatible fractional memory adapter state")
            strength = _strength(state["strength"])
            kernel = _kernel(**state["kernel"])
            self.strength, self._kernel = strength, kernel

    class FractionalHistoryAdapter(FractionalMemoryAdapter):
        """Independent local and strictly-past feature gates, 2*F+1 parameters.

        ``gate`` scales Rust GL history only; ``local_gate`` scales the current
        feature. Both start at zero. Disabling history recovers the ordinary
        pointwise gate without tying its gain to the learned fractional order.
        Full unpadded prefixes, float32 and use_cache=False are still required.
        """

        _state_schema = "spiraltorch.fractional_history_adapter.v1"

        def __init__(self, features: int, *, initial_alpha: float = 0.5,
                     strength: float = 0.1, **kernel_options: Any) -> None:
            super().__init__(features, initial_alpha=initial_alpha, strength=strength,
                             **kernel_options)
            self.local_gate = torch.nn.Parameter(torch.zeros_like(self.gate))

        def forward(self, value: Any) -> Any:
            strength = self._checked_input(value)
            if self.local_gate.dtype != value.dtype:
                raise TypeError("fractional history parameters must remain float32")
            if self.local_gate.device != value.device:
                raise ValueError("input and adapter parameters must share a device")
            if strength == 0.0:
                return value
            history = fractional_gl_history_autograd(
                value, self.log_alpha.exp(), axis=1, kernel=self._kernel)
            local = value + strength * self.local_gate.tanh() * value
            return local + strength * self.gate.tanh() * history
