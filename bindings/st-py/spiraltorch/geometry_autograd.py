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
    "EllipticCausalResidualAdapter",
    "elliptic_causal_autograd",
    "EllipticGatedCausalResidualAdapter",
    "elliptic_gated_causal_autograd",
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

    class _EllipticGatedCausalFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: Any, warp: Any, orientation: Any, raw_mix: Any, max_pairs: int
        ) -> Any:
            batch, sequence, _ = orientation.shape
            ctx.snapshot = warp.map_gated_causal_batch(
                _values(orientation),
                batch_size=batch,
                sequence_length=sequence,
                raw_mix=raw_mix.detach().item(),
                max_pairs=max_pairs,
            )
            ctx.save_for_backward(orientation, raw_mix)
            return torch.tensor(
                ctx.snapshot.features, device=orientation.device, dtype=orientation.dtype
            ).reshape(batch, sequence, 9)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx: Any, upstream: Any) -> tuple[Any, ...]:
            orientation, raw_mix = ctx.saved_tensors
            dx, dg = ctx.snapshot.vjp(_values(upstream))
            return (
                None,
                torch.tensor(
                    dx, device=orientation.device, dtype=orientation.dtype
                ).reshape_as(orientation),
                torch.tensor(dg, device=raw_mix.device, dtype=raw_mix.dtype),
                None,
            )

    class _EllipticCausalFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, warp: Any, orientation: Any, max_pairs: int) -> Any:
            batch, sequence, _ = orientation.shape
            ctx.snapshot = warp.map_causal_batch(
                _values(orientation),
                batch_size=batch,
                sequence_length=sequence,
                max_pairs=max_pairs,
            )
            ctx.save_for_backward(orientation)
            return torch.tensor(
                ctx.snapshot.features,
                device=orientation.device,
                dtype=orientation.dtype,
            ).reshape(batch, sequence, 9)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx: Any, upstream: Any) -> tuple[Any, Any, None]:
            (orientation,) = ctx.saved_tensors
            gradient = torch.tensor(
                ctx.snapshot.vjp(_values(upstream)),
                device=orientation.device,
                dtype=orientation.dtype,
            ).reshape_as(orientation)
            return None, gradient, None

    class _WaveGateFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: Any,
            value: Any,
            gate: Any,
            bias: Any,
            log_radius: Any,
            kernel: Any,
            conditioning: Any,
        ) -> Any:
            features = value.shape[-1]
            arguments = (
                _values(value),
                _values(gate),
                _values(bias),
                value.numel() // features,
                features,
            )
            ctx.has_radius = log_radius is not None
            ctx.snapshot = (
                kernel.forward_with_log_radius(*arguments, _values(log_radius)[0])
                if ctx.has_radius
                else kernel.forward(*arguments)
            )
            # Keep Torch's version checks, but the mathematical pullback reads
            # only the owned Rust snapshot, never current adapter parameters.
            ctx.save_for_backward(value, gate, bias, log_radius)
            if conditioning is not None:
                conditioning.update(json.loads(ctx.snapshot.conditioning_json()))
            return torch.tensor(
                ctx.snapshot.output, device=value.device, dtype=value.dtype
            ).reshape(value.shape)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, ...]:
            value, gate, bias, log_radius = ctx.saved_tensors
            gradients = (
                ctx.snapshot.vjp_with_log_radius(_values(grad_output))
                if ctx.has_radius
                else ctx.snapshot.vjp(_values(grad_output))
            )
            parameters = (
                (value, gate, bias, log_radius)
                if ctx.has_radius
                else (value, gate, bias)
            )
            result = tuple(
                torch.tensor(
                    gradient, device=parameter.device, dtype=parameter.dtype
                ).reshape(parameter.shape)
                for gradient, parameter in zip(gradients, parameters)
            )
            return result + ((None, None) if ctx.has_radius else (None, None, None))

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


def elliptic_causal_autograd(
    warp: Any, orientation: Any, *, max_pairs: int = 1_048_576
) -> Any:
    """Rust elliptic Q=K=V attention on unpadded [B,T,3] orientations.

    Full contexts only, structurally causal within each batch item. Scale is
    1/sqrt(9); no dropout, padding mask or KV cache. The complete first-order
    chart/attention VJP lives in Rust. Device tensors use explicit CPU transport.
    """
    _causal_input(warp, orientation, max_pairs)
    return _EllipticCausalFunction.apply(warp, orientation, max_pairs)


def _causal_input(warp: Any, orientation: Any, max_pairs: int) -> None:
    _input(orientation)
    from . import EllipticWarp

    if not isinstance(warp, EllipticWarp):
        raise TypeError("warp must be a spiraltorch.EllipticWarp instance")
    if type(max_pairs) is not int or max_pairs <= 0:
        raise ValueError("max_pairs must be a positive integer")
    if (
        orientation.ndim != 3
        or orientation.shape[-1] != 3
        or min(orientation.shape[:2]) == 0
    ):
        raise ValueError("causal orientations require nonempty [batch, sequence, 3]")
    batch, sequence, _ = orientation.shape
    if batch * sequence > 65_536 or batch * sequence * sequence > max_pairs:
        raise ValueError("causal orientations exceed the row or score-pair budget")


def elliptic_gated_causal_autograd(
    warp: Any, orientation: Any, raw_mix: Any, *, max_pairs: int = 1_048_576
) -> Any:
    """Rust local + tanh(raw_mix) * (causal - local), including both VJPs.

    The signed shared scalar controls an ambient feature correction, not a
    manifold interpolation. Zero exactly preserves pointwise features and their
    orientation VJP, while the gate can learn. Same full-context CPU transport
    contract as elliptic_causal_autograd; the gate gradient is summed, not averaged.
    """
    _causal_input(warp, orientation, max_pairs)
    if (
        not isinstance(raw_mix, torch.Tensor)
        or raw_mix.dtype != orientation.dtype
        or raw_mix.device != orientation.device
    ):
        raise TypeError("raw_mix must be a float32 tensor on the input device")
    if raw_mix.ndim != 0:
        raise ValueError("raw_mix must be a shared scalar")
    return _EllipticGatedCausalFunction.apply(warp, orientation, raw_mix, max_pairs)


def wave_gate_autograd(
    value: Any,
    gate: Any,
    bias: Any,
    *,
    log_radius: Any = None,
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
    if log_radius is not None:
        if (
            not isinstance(log_radius, torch.Tensor)
            or log_radius.dtype != value.dtype
            or log_radius.device != value.device
        ):
            raise TypeError("log_radius must be a float32 tensor on the input device")
        if log_radius.ndim != 0:
            raise ValueError("log_radius must be a shared scalar")
    from . import WaveGateKernel

    if kernel is None:
        kernel = WaveGateKernel()
    if not isinstance(kernel, WaveGateKernel):
        raise TypeError("kernel must be an immutable Rust WaveGateKernel")
    if max(value.numel(), value.shape[-1]) > kernel.max_values:
        raise ValueError("input exceeds the geometric kernel's value budget")
    conditioning = {} if return_conditioning else None
    output = _WaveGateFunction.apply(
        value, gate, bias, log_radius, kernel, conditioning
    )
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

    class EllipticCausalResidualAdapter:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_torch()

    class EllipticGatedCausalResidualAdapter:
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
            self,
            features: int,
            *,
            strength: float = 0.1,
            log_radius: float | None = None,
            learnable_radius: bool = False,
            **kernel_options: Any,
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
            if not isinstance(learnable_radius, bool):
                raise TypeError("learnable_radius must be a bool")
            if log_radius is None and not learnable_radius:
                self.register_parameter("log_radius", None)
            else:
                radius = torch.tensor(
                    0.0 if log_radius is None else float(log_radius),
                    dtype=torch.float32,
                )
                # Rust owns the parameter domain as well as the map and pullback.
                self._kernel.forward_with_log_radius(
                    [],
                    _values(self.gate),
                    _values(self.bias),
                    0,
                    features,
                    radius.item(),
                )
                if learnable_radius:
                    self.log_radius = torch.nn.Parameter(radius)
                else:
                    self.register_buffer("log_radius", radius)

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
                log_radius=self.log_radius,
                kernel=self._kernel,
                return_conditioning=return_conditioning,
            )
            if return_conditioning:
                output, conditioning = result
                return value + strength * output, conditioning
            return value + strength * result

        def get_extra_state(self) -> dict[str, Any]:
            state = {
                "schema": "spiraltorch.wave_gate_adapter.v1",
                "features": self.features,
                "strength": _strength(self.strength),
                "kernel": json.loads(self._kernel.configuration_json()),
            }
            if self.log_radius is not None:
                state["schema"] = "spiraltorch.wave_gate_adapter.v2"
                state["learnable_radius"] = isinstance(
                    self.log_radius, torch.nn.Parameter
                )
            return state

        def set_extra_state(self, state: dict[str, Any]) -> None:
            expected_schema = (
                "spiraltorch.wave_gate_adapter.v1"
                if self.log_radius is None
                else "spiraltorch.wave_gate_adapter.v2"
            )
            expected_keys = {"schema", "features", "strength", "kernel"}
            if self.log_radius is not None:
                expected_keys.add("learnable_radius")
            if (
                not isinstance(state, dict)
                or set(state) != expected_keys
                or state["schema"] != expected_schema
                or state["features"] != self.features
                or (
                    self.log_radius is not None
                    and (
                        not isinstance(state["learnable_radius"], bool)
                        or state["learnable_radius"]
                        != isinstance(self.log_radius, torch.nn.Parameter)
                    )
                )
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
            coordinates = self.orientation(value)
            orientation = torch.cat(
                (torch.ones_like(coordinates[..., :1]), coordinates), dim=-1
            )
            features = self._map_features(orientation)
            return value + strength * self.readout(features)

        def _map_features(self, orientation: Any) -> Any:
            from .elliptic import elliptic_warp_autograd

            return elliptic_warp_autograd(self._warp, orientation)

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

    class EllipticCausalResidualAdapter(EllipticResidualAdapter):
        """Identity-start residual mixing preceding tokens via Rust elliptic features.

        Same 11*F+2 trainable parameters as the pointwise adapter. Requires full,
        unpadded [B,T,F] contexts; a cached single-token call is not equivalent.
        At most 65536 rows and 1048576 potential score pairs per forward.
        """

        def forward(self, value: Any) -> Any:
            _input(value)
            if value.ndim != 3 or min(value.shape[:2]) == 0:
                raise ValueError(
                    "causal adapter requires nonempty [batch, sequence, features]"
                )
            if (
                value.shape[0] * value.shape[1] > 65_536
                or value.shape[0] * value.shape[1] ** 2 > 1_048_576
            ):
                raise ValueError("causal adapter exceeds the row or score-pair budget")
            return super().forward(value)

        def _map_features(self, orientation: Any) -> Any:
            return elliptic_causal_autograd(self._warp, orientation)

        def get_extra_state(self) -> dict[str, Any]:
            return {
                **super().get_extra_state(),
                "schema": "spiraltorch.elliptic_causal_residual_adapter.v1",
            }

        def set_extra_state(self, state: dict[str, Any]) -> None:
            if (
                not isinstance(state, dict)
                or state.get("schema")
                != "spiraltorch.elliptic_causal_residual_adapter.v1"
            ):
                raise ValueError("incompatible causal elliptic adapter state")
            super().set_extra_state(
                {**state, "schema": "spiraltorch.elliptic_residual_adapter.v1"}
            )

    class EllipticGatedCausalResidualAdapter(EllipticCausalResidualAdapter):
        """Pointwise-start features with a learned signed contextual correction.

        11*F+3 parameters; the zero readout starts the entire adapter at identity.
        The new shared gate starts at zero without consuming random numbers.
        Full unpadded contexts only; no cache, padding mask or resident GPU path.
        """

        def __init__(self, features: int, *, raw_mix: float = 0.0, **options: Any) -> None:
            value = torch.tensor(float(raw_mix), dtype=torch.float32)
            if not torch.isfinite(value):
                raise ValueError("raw_mix must be finite and representable as float32")
            super().__init__(features, **options)
            self.raw_mix = torch.nn.Parameter(value)

        def _map_features(self, orientation: Any) -> Any:
            return elliptic_gated_causal_autograd(self._warp, orientation, self.raw_mix)

        def get_extra_state(self) -> dict[str, Any]:
            return {
                **super().get_extra_state(),
                "schema": "spiraltorch.elliptic_gated_causal_residual_adapter.v1",
            }

        def set_extra_state(self, state: dict[str, Any]) -> None:
            if (
                not isinstance(state, dict)
                or state.get("schema")
                != "spiraltorch.elliptic_gated_causal_residual_adapter.v1"
            ):
                raise ValueError("incompatible gated causal elliptic adapter state")
            super().set_extra_state(
                {**state, "schema": "spiraltorch.elliptic_causal_residual_adapter.v1"}
            )

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
