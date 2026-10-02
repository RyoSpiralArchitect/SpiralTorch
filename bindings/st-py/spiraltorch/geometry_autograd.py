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

__all__ = [
    "ToposResonatorAdapter",
    "topos_resonator_autograd",
    "EllipticResidualAdapter",
    "WaveGateAdapter",
    "wave_gate_autograd",
]
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

    class _WaveGateFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: Any, value: Any, gate: Any, bias: Any, kernel: Any, conditioning: Any
        ) -> Any:
            features = value.shape[-1]
            ctx.snapshot = kernel.forward(
                _values(value),
                _values(gate),
                _values(bias),
                value.numel() // features,
                features,
            )
            # Keep Torch's version checks, but the mathematical pullback reads
            # only the owned Rust snapshot, never current adapter parameters.
            ctx.save_for_backward(value, gate, bias)
            if conditioning is not None:
                conditioning.update(json.loads(ctx.snapshot.conditioning_json()))
            return torch.tensor(
                ctx.snapshot.output, device=value.device, dtype=value.dtype
            ).reshape(value.shape)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, Any, Any, None, None]:
            value, gate, bias = ctx.saved_tensors
            gradients = ctx.snapshot.vjp(_values(grad_output))
            return tuple(
                torch.tensor(
                    gradient, device=parameter.device, dtype=parameter.dtype
                ).reshape(parameter.shape)
                for gradient, parameter in zip(gradients, (value, gate, bias))
            ) + (None, None)

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


def wave_gate_autograd(
    value: Any,
    gate: Any,
    bias: Any,
    *,
    kernel: Any = None,
    return_conditioning: bool = False,
) -> Any:
    """Rust WaveGate with shared feature vectors and sum-reduced parameter VJPs.

    Leading axes are independent rows; the final feature axis is coupled by the
    radial projection. Float32 CPU transport only, with first-order gradients.
    No parameter-gradient averaging or training-policy rewrite is applied.
    When requested, return (output, Rust scalar conditioning) from this same
    forward snapshot. It measures local projection gains, not full loss gradients.
    """
    _input(value)
    for name, parameter in (("gate", gate), ("bias", bias)):
        if (
            not isinstance(parameter, torch.Tensor)
            or parameter.dtype != value.dtype
            or parameter.device != value.device
        ):
            raise TypeError(f"{name} must be a float32 tensor on the input device")
        if parameter.shape != (value.shape[-1],):
            raise ValueError(f"{name} must be a shared vector of feature width")
    from . import WaveGateKernel

    if kernel is None:
        kernel = WaveGateKernel()
    if not isinstance(kernel, WaveGateKernel):
        raise TypeError("kernel must be an immutable Rust WaveGateKernel")
    if max(value.numel(), value.shape[-1]) > kernel.max_values:
        raise ValueError("input exceeds the geometric kernel's value budget")
    conditioning = {} if return_conditioning else None
    output = _WaveGateFunction.apply(value, gate, bias, kernel, conditioning)
    return (output, conditioning) if return_conditioning else output


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

    class EllipticResidualAdapter:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_torch()

    class WaveGateAdapter:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_torch()

else:

    class WaveGateAdapter(torch.nn.Module):
        """Identity-start residual through a Rust WaveGate, with 2F parameters.

        Zero gate/bias keep the original hidden state and allow both to learn
        immediately. No token mixing or implicit model patching is performed.
        The native module's text encoder is not part of this adapter.
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
            from . import WaveGateKernel

            self.features = features
            self.strength = _strength(strength)
            self._kernel = WaveGateKernel(**kernel_options)
            if features > self._kernel.max_values:
                raise ValueError("features exceed the geometric kernel's value budget")
            self.gate = torch.nn.Parameter(torch.zeros(features, dtype=torch.float32))
            self.bias = torch.nn.Parameter(torch.zeros(features, dtype=torch.float32))

        @property
        def execution_backend(self) -> str:
            return self._kernel.execution_backend

        def forward(self, value: Any) -> Any:
            return self._forward(value, return_conditioning=False)

        def forward_with_conditioning(self, value: Any) -> Any:
            """Return (residual output, local Rust conditioning); bypass has None."""
            return self._forward(value, return_conditioning=True)

        def _forward(self, value: Any, *, return_conditioning: bool) -> Any:
            _input(value)
            if value.shape[-1] != self.features:
                raise ValueError("input feature width differs from the adapter")
            strength = _strength(self.strength)
            if strength == 0.0:
                return (value, None) if return_conditioning else value
            result = wave_gate_autograd(
                value,
                self.gate,
                self.bias,
                kernel=self._kernel,
                return_conditioning=return_conditioning,
            )
            if return_conditioning:
                output, conditioning = result
                return value + strength * output, conditioning
            return value + strength * result

        def get_extra_state(self) -> dict[str, Any]:
            return {
                "schema": "spiraltorch.wave_gate_adapter.v1",
                "features": self.features,
                "strength": _strength(self.strength),
                "kernel": json.loads(self._kernel.configuration_json()),
            }

        def set_extra_state(self, state: dict[str, Any]) -> None:
            if (
                not isinstance(state, dict)
                or set(state) != {"schema", "features", "strength", "kernel"}
                or state["schema"] != "spiraltorch.wave_gate_adapter.v1"
                or state["features"] != self.features
            ):
                raise ValueError("incompatible WaveGate adapter state")
            from . import WaveGateKernel

            strength = _strength(state["strength"])
            kernel = WaveGateKernel(**state["kernel"])
            if self.features > kernel.max_values:
                raise ValueError("features exceed the geometric kernel's value budget")
            self.strength, self._kernel = strength, kernel

    class EllipticResidualAdapter(torch.nn.Module):
        """Identity-initialized residual through the Rust elliptic feature map.

        A trainable F->2 projection supplies the local chart (1, u, v), which
        avoids poles and the azimuth seam for finite coordinates. A zero-start
        9->F readout restores hidden width. This is a pointwise hemisphere chart,
        not a global spherical atlas or a resident GPU implementation.
        """

        def __init__(
            self,
            features: int,
            *,
            strength: float = 0.1,
            curvature_radius: float = 1.0,
            sheet_count: int = 2,
            spin_harmonics: int = 1,
        ) -> None:
            super().__init__()
            if (
                isinstance(features, bool)
                or not isinstance(features, int)
                or features <= 0
            ):
                raise ValueError("features must be a positive integer")
            from . import EllipticWarp

            self.features = features
            self.strength = _strength(strength)
            self._warp = EllipticWarp(curvature_radius, sheet_count, spin_harmonics)
            self.orientation = torch.nn.Linear(features, 2, dtype=torch.float32)
            self.readout = torch.nn.Linear(9, features, bias=False, dtype=torch.float32)
            torch.nn.init.zeros_(self.readout.weight)

        @property
        def execution_backend(self) -> str:
            return "rust_f32_cpu"

        def forward(self, value: Any) -> Any:
            _input(value)
            if value.shape[-1] != self.features:
                raise ValueError("input feature width differs from the adapter")
            strength = _strength(self.strength)
            if strength == 0.0:
                return value
            from .elliptic import elliptic_warp_autograd

            coordinates = self.orientation(value)
            orientation = torch.cat(
                (torch.ones_like(coordinates[..., :1]), coordinates), dim=-1
            )
            features = elliptic_warp_autograd(self._warp, orientation)
            return value + strength * self.readout(features)

        def get_extra_state(self) -> dict[str, Any]:
            return {
                "schema": "spiraltorch.elliptic_residual_adapter.v1",
                "features": self.features,
                "strength": _strength(self.strength),
                "warp": {
                    "curvature_radius": self._warp.curvature_radius,
                    "sheet_count": self._warp.sheet_count,
                    "spin_harmonics": self._warp.spin_harmonics,
                },
            }

        def set_extra_state(self, state: dict[str, Any]) -> None:
            if (
                not isinstance(state, dict)
                or set(state) != {"schema", "features", "strength", "warp"}
                or state["schema"] != "spiraltorch.elliptic_residual_adapter.v1"
                or state["features"] != self.features
            ):
                raise ValueError("incompatible elliptic adapter state")
            from . import EllipticWarp

            strength = _strength(state["strength"])
            warp = EllipticWarp(**state["warp"])
            self.strength, self._warp = strength, warp

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
