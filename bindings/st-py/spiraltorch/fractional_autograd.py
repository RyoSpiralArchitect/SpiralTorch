"""First-order transport for Rust's causal fractional difference, not a new rule.

CPU f32 host copies are explicit. Sequence adapters require complete unpadded
prefixes, no KV cache, and no packed-document boundaries within one sequence.
Bulk buffers avoid Python scalar boxing when Torch/NumPy interop is available;
the sequence transport remains available without that optional capability.
"""

from __future__ import annotations

import json
import math
import struct
from typing import Any

from .geometry_autograd import _input, _require_torch, _strength, _values, torch
from ._torch_transport import _buffer_transport_available, _buffer_values, _transport_output

__all__ = ["fractional_gl_autograd", "fractional_gl_history_autograd",
           "fractional_gl_history_l2_autograd", "FractionalMemoryAdapter",
           "FractionalHistoryAdapter", "FractionalL2HistoryAdapter",
           "fractional_gl_history_log_gain_autograd", "FractionalGainHistoryAdapter",
           "fractional_gl_angle_autograd", "FractionalAngleGainHistoryAdapter"]


def _kernel(**options: Any) -> Any:
    from . import FractionalGlKernel

    return FractionalGlKernel(**options)


def _capture_fractional(ctx: Any, kernel: Any, value: Any, alpha: Any,
                        axis: int, name: str, *options: Any) -> Any:
    ctx.buffer_transport = _buffer_transport_available()
    operation = getattr(kernel, name + ("_buffer" if ctx.buffer_transport else ""))
    values = _buffer_values(value) if ctx.buffer_transport else _values(value)
    ctx.snapshot = operation(values, list(value.shape), axis, alpha.detach().item(), *options)
    ctx.save_for_backward(value, alpha)
    ctx.save_for_forward(value, alpha)
    result = ctx.snapshot.output_buffer() if ctx.buffer_transport else ctx.snapshot.output
    return _transport_output(result, value, ctx.buffer_transport)


def _history_gain(gain: float) -> float:
    if isinstance(gain, bool) or not isinstance(gain, (int, float)):
        raise TypeError("history gain must be a constant number, not a tensor")
    try:
        gain = struct.unpack("=f", struct.pack("=f", gain))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("history gain must be finite and positive in float32") from error
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("history gain must be finite and positive in float32")
    return gain


def _history_lag_window(window: Any, kernel: Any) -> tuple[int, int] | None:
    if window is None:
        return None
    if not isinstance(window, (tuple, list)) or len(window) != 2 or any(type(v) is not int for v in window):
        raise TypeError("lag_window must contain two integer bounds")
    kernel.validate_history_window(*window)
    return tuple(window)


if torch is not None:
    class _FractionalGlAngleFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, angle: Any) -> Any:
            from . import FractionalGlAngleChart

            ctx.chart = FractionalGlAngleChart(angle.detach().item())
            ctx.save_for_backward(angle)
            ctx.save_for_forward(angle)
            return torch.tensor(ctx.chart.alpha, dtype=angle.dtype, device=angle.device)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx: Any, upstream: Any) -> tuple[Any]:
            angle, = ctx.saved_tensors
            gradient = ctx.chart.vjp(upstream.detach().item())
            return (torch.tensor(gradient, dtype=angle.dtype, device=angle.device),)

        @staticmethod
        def jvp(ctx: Any, tangent: Any) -> Any:
            angle, = ctx.saved_tensors
            value = ctx.chart.jvp(0.0 if tangent is None else tangent.detach().item())
            return torch.tensor(value, dtype=angle.dtype, device=angle.device)


    class _FractionalGlFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, kernel: Any, value: Any, alpha: Any, axis: int,
                    history_only: bool) -> Any:
            name = "forward_history" if history_only else "forward"
            return _capture_fractional(ctx, kernel, value, alpha, axis, name)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx: Any, upstream: Any) -> tuple[Any, ...]:
            value, alpha = ctx.saved_tensors
            need_input, need_alpha = ctx.needs_input_grad[1:3]
            if not (need_input or need_alpha):
                return None, None, None, None, None
            dx = da = None
            suffix = "_buffer" if ctx.buffer_transport else ""
            direction = _buffer_values(upstream) if ctx.buffer_transport else _values(upstream)
            if need_input and need_alpha:
                dx, da = getattr(ctx.snapshot, "vjp" + suffix)(direction)
            elif need_input:
                dx = getattr(ctx.snapshot, "vjp_input" + suffix)(direction)
            elif need_alpha:
                da = getattr(ctx.snapshot, "vjp_alpha" + suffix)(direction)
            return (
                None,
                None if dx is None else _transport_output(dx, value, ctx.buffer_transport),
                None if da is None else torch.tensor(da, dtype=alpha.dtype, device=alpha.device),
                None,
                None,
            )

        @staticmethod
        def jvp(ctx: Any, _kernel: Any, dx: Any, da: Any, _axis: Any,
                _history_only: Any) -> Any:
            value, _ = ctx.saved_tensors
            dx = torch.zeros_like(value) if dx is None else dx
            operation = ctx.snapshot.jvp_buffer if ctx.buffer_transport else ctx.snapshot.jvp
            direction = _buffer_values(dx) if ctx.buffer_transport else _values(dx)
            result = operation(direction, 0.0 if da is None else da.detach().item())
            return _transport_output(result, value, ctx.buffer_transport)


    class _FractionalGlHistoryL2Function(_FractionalGlFunction):
        @staticmethod
        def forward(ctx: Any, kernel: Any, value: Any, alpha: Any, axis: int,
                    gain: float) -> Any:
            return _capture_fractional(ctx, kernel, value, alpha, axis,
                                       "forward_history_l2", gain)


    class _FractionalGlLogGainFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, kernel: Any, value: Any, alpha: Any, log_gain: Any, axis: int,
                    lag_window: tuple[int, int] | None) -> Any:
            ctx.buffer_transport = _buffer_transport_available()
            name = "forward_history_log_gain" + ("_window" if lag_window is not None else "")
            operation = getattr(kernel, name + ("_buffer" if ctx.buffer_transport else ""))
            values = _buffer_values(value) if ctx.buffer_transport else _values(value)
            ctx.snapshot = operation(values, list(value.shape), axis, alpha.detach().item(),
                                     log_gain.detach().item(), *(lag_window or ()))
            ctx.save_for_backward(value, alpha, log_gain)
            ctx.save_for_forward(value, alpha, log_gain)
            output = ctx.snapshot.output_buffer() if ctx.buffer_transport else ctx.snapshot.output
            return _transport_output(output, value, ctx.buffer_transport)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx: Any, upstream: Any) -> tuple[Any, ...]:
            value, alpha, log_gain = ctx.saved_tensors
            need_input, need_alpha, need_gain = ctx.needs_input_grad[1:4]
            direction = _buffer_values(upstream) if ctx.buffer_transport else _values(upstream)
            suffix = "_buffer" if ctx.buffer_transport else ""
            dx = da = dg = None
            if need_input and need_alpha and need_gain:
                dx, da, dg = getattr(ctx.snapshot, "vjp" + suffix)(direction)
            else:
                if need_input:
                    dx = getattr(ctx.snapshot, "vjp_input" + suffix)(direction)
                if need_alpha and need_gain:
                    da, dg = getattr(ctx.snapshot, "vjp_parameters" + suffix)(direction)
                elif need_alpha:
                    da = getattr(ctx.snapshot, "vjp_alpha" + suffix)(direction)
                elif need_gain:
                    dg = getattr(ctx.snapshot, "vjp_log_gain" + suffix)(direction)
            return (None, None if dx is None else _transport_output(dx, value, ctx.buffer_transport),
                    None if da is None else torch.tensor(da, dtype=alpha.dtype, device=alpha.device),
                    None if dg is None else torch.tensor(dg, dtype=log_gain.dtype, device=log_gain.device), None, None)

        @staticmethod
        def jvp(ctx: Any, _kernel: Any, dx: Any, da: Any, dg: Any, _axis: Any, _lag_window: Any) -> Any:
            value, _, _ = ctx.saved_tensors
            dx = torch.zeros_like(value) if dx is None else dx
            direction = _buffer_values(dx) if ctx.buffer_transport else _values(dx)
            operation = ctx.snapshot.jvp_buffer if ctx.buffer_transport else ctx.snapshot.jvp
            result = operation(direction, 0.0 if da is None else da.detach().item(),
                               0.0 if dg is None else dg.detach().item())
            return _transport_output(result, value, ctx.buffer_transport)


def fractional_gl_angle_autograd(angle: Any) -> Any:
    """Rust alpha=1+2*tan(angle), with first-order scalar VJP/JVP.

    Domain: -atan(1/2) < angle < pi/2 after float32 conversion. No clipping,
    wrapping, higher-order AD or torch.func. Compose with the existing GL map;
    this does not replace its convolution or promise cross-chart quality parity.
    """
    _require_torch()
    if not isinstance(angle, torch.Tensor) or angle.ndim != 0 or angle.dtype != torch.float32:
        raise TypeError("angle must be a scalar float32 tensor")
    return _FractionalGlAngleFunction.apply(angle)


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
    kernel = _checked_kernel(value, alpha, axis, kernel)
    return _FractionalGlFunction.apply(kernel, value, alpha, axis, history_only)


def fractional_gl_history_l2_autograd(value: Any, alpha: Any, *, axis: int,
                                     kernel: Any = None, gain: float = 1.0) -> Any:
    """Strictly-past GL with full-kernel coefficient L2 norm fixed to gain.

    Rust differentiates both taps and normalization with respect to alpha.
    Positive sample-spacing scale cancels. Gain is constant, not trainable;
    this fixes filter energy, not the variance of correlated hidden states.
    Same first-order AD and complete-prefix limits as the raw history route.
    """
    kernel = _checked_kernel(value, alpha, axis, kernel)
    return _FractionalGlHistoryL2Function.apply(kernel, value, alpha, axis, _history_gain(gain))


def _checked_kernel(value: Any, alpha: Any, axis: int, kernel: Any) -> Any:
    _input(value)
    if not isinstance(alpha, torch.Tensor) or alpha.ndim != 0 or alpha.dtype != torch.float32:
        raise TypeError("alpha must be a scalar float32 tensor")
    if isinstance(axis, bool) or not isinstance(axis, int) or not 0 <= axis < value.ndim:
        raise ValueError("axis must be a valid nonnegative integer")
    from . import FractionalGlKernel

    kernel = _kernel() if kernel is None else kernel
    if not isinstance(kernel, FractionalGlKernel):
        raise TypeError("kernel must be an immutable Rust FractionalGlKernel")
    return kernel


def fractional_gl_history_log_gain_autograd(value: Any, alpha: Any, log_gain: Any, *,
                                           axis: int, kernel: Any = None,
                                           lag_window: tuple[int, int] | None = None) -> Any:
    """Rust normalized history with independently differentiable log amplitude.

    Rust owns exp(log_gain), normalization and all three first-order derivatives.
    Alpha and log_gain must be scalar float32 tensors. Gain overflow/underflow
    fails closed; no clipping, higher-order AD or persistent prefix cache.
    Optional half-open lag_window masks taps after full-kernel normalization;
    it does not shorten or renormalize the kernel. Rust also masks its VJP/JVP.
    """
    kernel = _checked_kernel(value, alpha, axis, kernel)
    if not isinstance(log_gain, torch.Tensor) or log_gain.ndim != 0 or log_gain.dtype != torch.float32:
        raise TypeError("log_gain must be a scalar float32 tensor")
    window = _history_lag_window(lag_window, kernel)
    return _FractionalGlLogGainFunction.apply(kernel, value, alpha, log_gain, axis, window)


if torch is None:
    class FractionalMemoryAdapter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()

    class FractionalHistoryAdapter(FractionalMemoryAdapter):
        pass

    class FractionalL2HistoryAdapter(FractionalHistoryAdapter):
        pass

    class FractionalGainHistoryAdapter(FractionalHistoryAdapter):
        pass

    class FractionalAngleGainHistoryAdapter(FractionalHistoryAdapter):
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

        def _order_parameter(self) -> Any:
            return self.log_alpha

        def _alpha_tensor(self) -> Any:
            return self.log_alpha.exp()

        def _checked_input(self, value: Any) -> float:
            _input(value)
            if value.ndim != 3 or value.shape[-1] != self.features:
                raise ValueError("fractional memory needs [batch, time, features]")
            order = self._order_parameter()
            if self.gate.dtype != value.dtype or order.dtype != value.dtype:
                raise TypeError("fractional memory parameters must remain float32")
            if value.device != self.gate.device or value.device != order.device:
                raise ValueError("input and adapter parameters must share a device")
            return _strength(self.strength)

        def forward(self, value: Any) -> Any:
            strength = self._checked_input(value)
            if strength == 0.0:
                return value
            difference = fractional_gl_autograd(value, self._alpha_tensor(), axis=1, kernel=self._kernel)
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

        def _history(self, value: Any, alpha: Any) -> Any:
            return fractional_gl_history_autograd(value, alpha, axis=1, kernel=self._kernel)

        def forward(self, value: Any) -> Any:
            strength = self._checked_input(value)
            if self.local_gate.dtype != value.dtype:
                raise TypeError("fractional history parameters must remain float32")
            if self.local_gate.device != value.device:
                raise ValueError("input and adapter parameters must share a device")
            if strength == 0.0:
                return value
            history = self._history(value, self._alpha_tensor())
            local = value + strength * self.local_gate.tanh() * value
            return local + strength * self.gate.tanh() * history

    class FractionalL2HistoryAdapter(FractionalHistoryAdapter):
        """Separate local/history gates with a fixed-energy fractional filter.

        2*F+1 parameters, identity initialization and explicit full-prefix use.
        The constant gain and distinct state schema prevent silent loading as
        raw GL history. Normalization and its differentials live in Rust.
        """

        _state_schema = "spiraltorch.fractional_l2_history_adapter.v1"

        def __init__(self, features: int, *, initial_alpha: float = 0.5,
                     strength: float = 0.1, gain: float = 1.0, **kernel_options: Any) -> None:
            gain = _history_gain(gain)
            super().__init__(features, initial_alpha=initial_alpha, strength=strength,
                             **kernel_options)
            self._history_gain = gain

        @property
        def gain(self) -> float:
            return self._history_gain

        def _history(self, value: Any, alpha: Any) -> Any:
            return fractional_gl_history_l2_autograd(
                value, alpha, axis=1, kernel=self._kernel, gain=self.gain)

        def get_extra_state(self) -> dict[str, Any]:
            return {**super().get_extra_state(), "gain": self.gain}

        def set_extra_state(self, state: dict[str, Any]) -> None:
            if (not isinstance(state, dict)
                    or set(state) != {"schema", "features", "strength", "kernel", "gain"}):
                raise ValueError("incompatible fractional L2 history adapter state")
            gain = _history_gain(state["gain"])
            super().set_extra_state({key: value for key, value in state.items() if key != "gain"})
            self._history_gain = gain


    class FractionalGainHistoryAdapter(FractionalHistoryAdapter):
        """2*F+2 parameters: local/history gates, log order, independent log gain.

        Starts as identity. The positive gain and signed feature gate are
        redundant amplitude controls, not an identifiability guarantee.
        Full unpadded prefixes, float32 and use_cache=False remain required.
        """

        _state_schema = "spiraltorch.fractional_gain_history_adapter.v1"

        def __init__(self, features: int, *, initial_alpha: float = 0.5,
                     initial_gain: float = 1.0, strength: float = 0.1,
                     lag_window: tuple[int, int] | None = None, **kernel_options: Any) -> None:
            initial_gain = _history_gain(initial_gain)
            super().__init__(features, initial_alpha=initial_alpha, strength=strength, **kernel_options)
            self._lag_window = _history_lag_window(lag_window, self._kernel)
            self.log_gain = torch.nn.Parameter(torch.tensor(math.log(initial_gain), dtype=torch.float32))

        @property
        def lag_window(self) -> tuple[int, int] | None:
            return self._lag_window

        def get_extra_state(self) -> dict[str, Any]:
            state = super().get_extra_state()
            if self.lag_window is not None:
                state["lag_window"] = list(self.lag_window)
            return state

        def set_extra_state(self, state: dict[str, Any]) -> None:
            if self.lag_window is None:
                return super().set_extra_state(state)
            if not isinstance(state, dict) or state.get("lag_window") != list(self.lag_window):
                raise ValueError("incompatible fractional history lag_window state")
            # The saved kernel may differ from the constructor's recipe.
            kernel = _kernel(**state["kernel"])
            _history_lag_window(state["lag_window"], kernel)
            super().set_extra_state({key: value for key, value in state.items() if key != "lag_window"})

        @property
        def gain(self) -> float:
            """Current checked Rust f32 amplitude, without a history allocation."""
            if self.log_gain.ndim != 0 or self.log_gain.dtype != torch.float32:
                raise TypeError("fractional history log_gain must remain scalar float32")
            return self._kernel.gain_from_log_gain(float(self.log_gain.detach()))

        def _checked_input(self, value: Any) -> float:
            strength = super()._checked_input(value)
            if self.log_gain.dtype != value.dtype:
                raise TypeError("fractional history log_gain must remain float32")
            if self.log_gain.device != value.device:
                raise ValueError("input and log_gain must share a device")
            return strength

        def _history(self, value: Any, alpha: Any) -> Any:
            return fractional_gl_history_log_gain_autograd(value, alpha, self.log_gain,
                                                          axis=1, kernel=self._kernel, lag_window=self.lag_window)


    class FractionalAngleGainHistoryAdapter(FractionalGainHistoryAdapter):
        """2*F+2 parameters with Rust's bounded angular GL shape chart.

        Same identity feature gates and log-gain as the log-order adapter,
        but a distinct recipe and parameter name prevent silent checkpoint
        substitution. Complete unpadded prefixes, float32 and no KV cache.
        Leaving the chart fails; no parameter projection is performed.
        """

        _state_schema = "spiraltorch.fractional_angle_gain_history_adapter.v1"

        def __init__(self, features: int, *, initial_angle: float = 0.0,
                     initial_gain: float = 1.0, strength: float = 0.1,
                     lag_window: tuple[int, int] | None = None, **kernel_options: Any) -> None:
            from . import FractionalGlAngleChart

            if isinstance(initial_angle, bool) or not isinstance(initial_angle, (int, float)):
                raise TypeError("initial_angle must be a number")
            chart = FractionalGlAngleChart(initial_angle)
            super().__init__(features, initial_alpha=chart.alpha, initial_gain=initial_gain,
                             strength=strength, lag_window=lag_window, **kernel_options)
            gain = self.log_gain
            del self.log_alpha
            del self.log_gain
            self.history_angle = torch.nn.Parameter(torch.tensor(chart.angle, dtype=torch.float32))
            self.log_gain = gain

        def _order_parameter(self) -> Any:
            return self.history_angle

        def _alpha_tensor(self) -> Any:
            return fractional_gl_angle_autograd(self.history_angle)

        @property
        def alpha(self) -> float:
            """Read the native chart, including its domain guard, without AD."""
            from . import FractionalGlAngleChart

            if self.history_angle.ndim != 0 or self.history_angle.dtype != torch.float32:
                raise TypeError("history_angle must remain scalar float32")
            return FractionalGlAngleChart(float(self.history_angle.detach())).alpha
