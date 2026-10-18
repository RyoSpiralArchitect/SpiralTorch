"""First-order transport for the Rust-owned ordinary channel-affine control."""
from __future__ import annotations

from typing import Any

from ._torch_transport import _buffer_transport_available, _buffer_values, _transport_output, torch
from .geometry_autograd import _input, _require_torch, _strength, _values

__all__ = ["ChannelAffineAdapter", "channel_affine_autograd"]
_SCHEMA = "spiraltorch.channel_affine_adapter.v1"


if torch is not None:
    class _AffineFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, gate, bias, kernel):
            ctx.buffers = _buffer_transport_available()
            transport = _buffer_values if ctx.buffers else _values
            operation = kernel.forward_buffer if ctx.buffers else kernel.forward
            ctx.snapshot = operation(transport(value), transport(gate), transport(bias),
                                     value.numel() // value.shape[-1], value.shape[-1])
            ctx.save_for_backward(value, gate, bias)
            output = ctx.snapshot.output_buffer() if ctx.buffers else ctx.snapshot.output
            return _transport_output(output, value, ctx.buffers)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, upstream):
            parameters = ctx.saved_tensors
            operation = ctx.snapshot.vjp_buffer if ctx.buffers else ctx.snapshot.vjp
            direction = _buffer_values(upstream) if ctx.buffers else _values(upstream)
            gradients = operation(direction)
            return tuple(_transport_output(g, p, ctx.buffers) for g, p in zip(gradients, parameters)) + (None,)


def channel_affine_autograd(value: Any, gate: Any, bias: Any, *, kernel: Any = None) -> Any:
    """Rust x * gate + bias on [...,features], with sum-reduced parameter VJPs.

    No saturation, geometry, token mixing, higher-order AD or implicit optimizer.
    Uses the same CPU tensor primitives as WaveGate's affine/reduction path.
    GPU tensors explicitly copy through CPU; this is not a resident GPU control.
    """
    _input(value)
    for name, parameter in (("gate", gate), ("bias", bias)):
        if (not isinstance(parameter, torch.Tensor) or parameter.dtype != value.dtype
                or parameter.device != value.device):
            raise TypeError(f"{name} must be float32 on the input device")
        if parameter.shape != (value.shape[-1],):
            raise ValueError(f"{name} must be a shared feature vector")
    from . import ChannelAffineKernel
    kernel = ChannelAffineKernel() if kernel is None else kernel
    if not isinstance(kernel, ChannelAffineKernel):
        raise TypeError("kernel must be an immutable Rust ChannelAffineKernel")
    if max(value.numel(), value.shape[-1]) > kernel.max_values:
        raise ValueError("input exceeds the affine kernel value budget")
    return _AffineFunction.apply(value, gate, bias, kernel)


if torch is None:
    class ChannelAffineAdapter:
        def __init__(self, *_args, **_kwargs):
            _require_torch()
else:
    class ChannelAffineAdapter(torch.nn.Module):
        """Identity-start x + strength * (x * gate + bias), with 2F parameters.

        Ordinary, non-geometric control. State and derivatives remain Rust-owned
        through the same portable artifact and HF Trainer path as geometry.
        """
        def __init__(self, features: int, *, strength: float = .1, max_values: int = 1_048_576):
            super().__init__()
            if type(features) is not int or features <= 0:
                raise ValueError("features must be a positive integer")
            from . import ChannelAffineKernel
            self._kernel = ChannelAffineKernel(max_values=max_values)
            if features > self._kernel.max_values:
                raise ValueError("features exceed the affine kernel value budget")
            self.features = features
            self.strength = _strength(strength)
            self.gate = torch.nn.Parameter(torch.zeros(features, dtype=torch.float32))
            self.bias = torch.nn.Parameter(torch.zeros(features, dtype=torch.float32))

        @property
        def execution_backend(self) -> str:
            return self._kernel.execution_backend

        def forward(self, value):
            _input(value)
            if value.shape[-1] != self.features:
                raise ValueError("input feature width differs from the adapter")
            strength = _strength(self.strength)
            if strength == 0.:
                return value
            result = value + strength * channel_affine_autograd(value, self.gate, self.bias, kernel=self._kernel)
            if not bool(torch.isfinite(result).all()):
                raise ValueError("channel-affine residual must remain finite in float32")
            return result

        def get_extra_state(self):
            return {"schema": _SCHEMA, "features": self.features,
                    "strength": _strength(self.strength), "max_values": self._kernel.max_values}

        def set_extra_state(self, state):
            if (not isinstance(state, dict) or set(state) != {"schema", "features", "strength", "max_values"}
                    or state["schema"] != _SCHEMA or type(state["features"]) is not int
                    or state["features"] != self.features):
                raise ValueError("incompatible channel-affine adapter state")
            from . import ChannelAffineKernel
            strength = _strength(state["strength"])
            kernel = ChannelAffineKernel(max_values=state["max_values"])
            if self.features > kernel.max_values:
                raise ValueError("features exceed the affine kernel value budget")
            self.strength, self._kernel = strength, kernel
