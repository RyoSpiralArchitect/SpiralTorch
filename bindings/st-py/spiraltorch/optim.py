"""Optimizers built on top of SpiralTorch gradient tapes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

__all__ = ["Amegagrad", "amegagrad"]

_NATIVE_EXTENSION_HINT = (
    "Build the SpiralTorch native extension (e.g. `maturin develop -m "
    "bindings/st-py/Cargo.toml`) to enable spiraltorch.optim."
)


def _require_native(st: Any) -> None:
    missing: list[str] = []
    for name in ("Hypergrad", "Realgrad", "GradientSummary"):
        try:
            getattr(st, name)
        except AttributeError:
            missing.append(name)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "spiraltorch.optim requires the compiled SpiralTorch extension "
            f"(missing: {joined}). {_NATIVE_EXTENSION_HINT}"
        )


def _require_tensor(st: Any, value: Any, *, label: str) -> Any:
    helper = getattr(st, "_session_require_tensor", None)
    if callable(helper):
        return helper(value, label=label)
    tensor_type = getattr(st, "Tensor", None)
    if isinstance(tensor_type, type):
        return tensor_type(value)
    raise TypeError(f"{label} must be a SpiralTorch Tensor. {_NATIVE_EXTENSION_HINT}")


def _set_tape_learning_rate(tape: Any, target: float) -> None:
    current = float(tape.learning_rate())
    if current <= 0.0 or target <= 0.0:
        return
    factor = target / current
    if factor != 1.0:
        tape.scale_learning_rate(float(factor))


def _finite_float(value: Any, *, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _finite_int(value: Any, *, default: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, numeric)


def _flatten_numeric_mapping(
    payload: Mapping[str, Any],
    *,
    prefix: str,
    out: dict[str, float],
) -> None:
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            _flatten_numeric_mapping(value, prefix=path, out=out)
            continue
        if isinstance(value, (str, bytes, bytearray)):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            out[path] = numeric


class Amegagrad:
    """Couple Hypergrad + Realgrad into a single `step()`-driven optimizer."""

    def __init__(
        self,
        *shape_args: Any,
        curvature: float = -1.0,
        hyper_learning_rate: float = 0.05,
        real_learning_rate: float = 0.01,
        shape: Any | None = None,
        rows: Any | None = None,
        cols: Any | None = None,
        topos: Any | None = None,
        gain: float = 1.0,
        topos_control_gain: float | None = None,
        topos_observed_depth: int | None = None,
        topos_visited_volume: int | None = None,
    ) -> None:
        import spiraltorch as st

        _require_native(st)

        self.curvature = float(curvature)
        self.hyper_learning_rate = float(hyper_learning_rate)
        self.real_learning_rate = float(real_learning_rate)
        self.gain = float(gain)

        self.hyper = st.hypergrad(
            *shape_args,
            curvature=self.curvature,
            learning_rate=self.hyper_learning_rate,
            shape=shape,
            rows=rows,
            cols=cols,
            topos=topos,
        )
        self.real = st.realgrad(
            *shape_args,
            learning_rate=self.real_learning_rate,
            shape=shape,
            rows=rows,
            cols=cols,
        )

        if self.hyper.shape() != self.real.shape():
            raise ValueError(
                f"Amegagrad hyper/real shapes differ: hyper={self.hyper.shape()} real={self.real.shape()}"
            )

        self.topos = self._resolve_hyper_topos(fallback=topos)
        rows_value, cols_value = self.shape()
        default_visited_volume = rows_value * cols_value
        self.topos_observed_depth = _finite_int(topos_observed_depth, default=1)
        self.topos_visited_volume = _finite_int(
            topos_visited_volume,
            default=default_visited_volume,
        )
        default_topos_gain = 0.0
        self.topos_control_gain = _finite_float(
            default_topos_gain if topos_control_gain is None else topos_control_gain,
            default=default_topos_gain,
        )
        if self.topos_control_gain < 0.0:
            self.topos_control_gain = 0.0
        self.last_control: Any | None = None
        self.last_topos_signal: dict[str, Any] | None = None
        self.last_topos_hints: dict[str, Any] | None = None
        self.last_topos_profile: dict[str, Any] | None = None
        self.last_topos_effect: dict[str, float] | None = None

    def shape(self) -> tuple[int, int]:
        return self.hyper.shape()

    def _resolve_hyper_topos(self, *, fallback: Any | None = None) -> Any | None:
        topos = getattr(self.hyper, "topos", None)
        if callable(topos):
            try:
                return topos()
            except Exception:
                pass
        return fallback

    def zero_grad(self) -> None:
        self.hyper.reset()
        self.real.reset()

    reset = zero_grad

    def accumulate_wave(self, tensor: Any) -> None:
        import spiraltorch as st

        tensor = _require_tensor(st, tensor, label="tensor")
        self.hyper.accumulate_wave(tensor)
        self.real.accumulate_wave(tensor)

    def accumulate_complex_wave(self, wave: Any) -> None:
        self.hyper.accumulate_complex_wave(wave)
        self.real.accumulate_complex_wave(wave)

    def absorb_text(self, encoder: Any, text: str) -> None:
        import spiraltorch as st

        curvature_fn = getattr(encoder, "curvature", None)
        if callable(curvature_fn):
            encoder_curvature = float(curvature_fn())
            if abs(encoder_curvature - self.curvature) > 1e-6:
                raise ValueError(
                    "encoder curvature must match Amegagrad.curvature "
                    f"(encoder={encoder_curvature}, optimizer={self.curvature})"
                )

        encode = getattr(encoder, "encode_z_space", None)
        if not callable(encode):
            raise TypeError("encoder must provide encode_z_space(text) -> Tensor")

        encoded = encode(str(text))
        tolist = getattr(encoded, "tolist", None)
        if not callable(tolist):
            raise TypeError("encode_z_space(text) must return a Tensor exposing tolist()")

        flat = [float(value) for row in tolist() for value in row]
        rows, cols = self.shape()
        total = rows * cols
        if len(flat) < total:
            flat.extend(0.0 for _ in range(total - len(flat)))
        elif len(flat) > total:
            flat = flat[:total]

        self.accumulate_wave(st.Tensor(rows, cols, flat))

    def accumulate_pair(self, prediction: Any, target: Any) -> None:
        import spiraltorch as st

        prediction = _require_tensor(st, prediction, label="prediction")
        target = _require_tensor(st, target, label="target")
        self.hyper.accumulate_pair(prediction, target)
        self.real.accumulate_pair(prediction, target)

    def desire_control(self, *, gain: float | None = None) -> Any:
        used_gain = self.gain if gain is None else float(gain)
        return self.hyper.desire_control(self.real.summary(), gain=used_gain)

    def topos_control_signal(
        self,
        *,
        observed_depth: int | None = None,
        visited_volume: int | None = None,
        **signal_options: Any,
    ) -> dict[str, Any]:
        """Return and cache the optimizer's open-topos pressure signal."""

        import spiraltorch as st

        guard = self.topos if self.topos is not None else self._resolve_hyper_topos()
        if guard is None:
            raise RuntimeError("Amegagrad has no topos available for control hints")
        used_observed_depth = (
            self.topos_observed_depth if observed_depth is None else int(observed_depth)
        )
        used_visited_volume = (
            self.topos_visited_volume if visited_volume is None else int(visited_volume)
        )
        signal = st.topos_control_signal(
            guard,
            observed_depth=used_observed_depth,
            visited_volume=used_visited_volume,
            **signal_options,
        )
        self.last_topos_signal = dict(signal)
        training_hints = signal.get("training_hints")
        self.last_topos_hints = (
            dict(training_hints) if isinstance(training_hints, Mapping) else None
        )
        return self.last_topos_signal

    def topos_training_hints(
        self,
        *,
        observed_depth: int | None = None,
        visited_volume: int | None = None,
        **signal_options: Any,
    ) -> dict[str, Any]:
        """Return named optimizer hints derived from this optimizer's topos."""

        signal = self.topos_control_signal(
            observed_depth=observed_depth,
            visited_volume=visited_volume,
            **signal_options,
        )
        hints = signal.get("training_hints")
        if not isinstance(hints, Mapping):
            return {}
        return dict(hints)

    def _topos_rate_scale(self, hints: Mapping[str, Any]) -> tuple[float, float, float]:
        learning_rate_scale = _finite_float(
            hints.get("learning_rate_scale"),
            default=1.0,
        )
        clip_scale = _finite_float(hints.get("clip_scale"), default=1.0)
        raw_scale = max(0.01, min(2.0, learning_rate_scale * clip_scale))
        blended_scale = 1.0 + self.topos_control_gain * (raw_scale - 1.0)
        if not math.isfinite(blended_scale) or blended_scale <= 0.0:
            blended_scale = 1.0
        blended_scale = max(0.01, min(2.0, blended_scale))
        return learning_rate_scale, clip_scale, blended_scale

    def _apply_topos_learning_rate_scale(
        self,
        hyper_target: float,
        real_target: float,
        *,
        hints: Mapping[str, Any] | None = None,
        observed_depth: int | None = None,
        visited_volume: int | None = None,
    ) -> tuple[float, float]:
        import spiraltorch as st

        if hints is None:
            signal = self.topos_control_signal(
                observed_depth=observed_depth,
                visited_volume=visited_volume,
            )
            profile_source: Mapping[str, Any] = signal
            hints = signal.get("training_hints")
            if not isinstance(hints, Mapping):
                hints = {}
            plan = st.topos_training_plan(signal, gain=self.topos_control_gain)
        else:
            hints = dict(hints)
            self.last_topos_hints = dict(hints)
            profile_source = self.last_topos_signal or {"training_hints": hints}
            plan = st.topos_training_plan(
                {"training_hints": hints},
                gain=self.topos_control_gain,
            )
        learning_rate_scale = _finite_float(
            plan.get("learning_rate_scale"),
            default=_finite_float(hints.get("learning_rate_scale"), default=1.0),
        )
        clip_scale = _finite_float(
            plan.get("clip_scale"),
            default=_finite_float(hints.get("clip_scale"), default=1.0),
        )
        rate_scale = _finite_float(
            plan.get("rate_scale"),
            default=self._topos_rate_scale(hints)[2],
        )
        hyper_scaled = hyper_target * rate_scale
        real_scaled = real_target * rate_scale
        self.last_topos_effect = {
            "learning_rate_scale": learning_rate_scale,
            "clip_scale": clip_scale,
            "raw_rate_scale": _finite_float(
                plan.get("raw_rate_scale"),
                default=learning_rate_scale * clip_scale,
            ),
            "rate_scale": rate_scale,
            "effective_gradient_bias_scale": _finite_float(
                plan.get("effective_gradient_bias_scale"),
                default=0.0,
            ),
            "effective_momentum_damping": _finite_float(
                plan.get("effective_momentum_damping"),
                default=0.0,
            ),
            "hyper_learning_rate": hyper_scaled,
            "real_learning_rate": real_scaled,
        }
        self.last_topos_profile = st.topos_runtime_profile(
            profile_source,
            training_gain=self.topos_control_gain,
        )
        return hyper_scaled, real_scaled

    def topos_telemetry_payload(
        self,
        signal: Mapping[str, Any] | None = None,
    ) -> dict[str, float]:
        """Flatten cached topos control state into `topos.*` telemetry keys."""

        if signal is None:
            signal = self.last_topos_signal
        if signal is None:
            signal = self.topos_control_signal()
        telemetry: dict[str, float] = {}
        signal_payload = dict(signal)
        signal_payload.pop("runtime_profile", None)
        _flatten_numeric_mapping(signal_payload, prefix="topos", out=telemetry)
        profile = self.last_topos_profile
        if profile is None:
            import spiraltorch as st

            profile = st.topos_runtime_profile(
                signal,
                training_gain=self.topos_control_gain,
            )
        _flatten_numeric_mapping(
            profile,
            prefix="topos.runtime_profile",
            out=telemetry,
        )
        if self.last_topos_effect is not None:
            _flatten_numeric_mapping(
                self.last_topos_effect,
                prefix="topos.optimizer_effect",
                out=telemetry,
            )
        return telemetry

    def topos_diagnostics(self) -> dict[str, Any]:
        """Return the cached topos signal, hints, and optimizer effect."""

        return {
            "signal": dict(self.last_topos_signal) if self.last_topos_signal else None,
            "training_hints": dict(self.last_topos_hints) if self.last_topos_hints else None,
            "runtime_profile": dict(self.last_topos_profile)
            if self.last_topos_profile
            else None,
            "effect": dict(self.last_topos_effect) if self.last_topos_effect else None,
        }

    def tune(
        self,
        control: Any | None = None,
        *,
        gain: float | None = None,
        use_topos: bool | None = None,
        topos_hints: Mapping[str, Any] | None = None,
        observed_depth: int | None = None,
        visited_volume: int | None = None,
    ) -> Any:
        if control is None:
            control = self.desire_control(gain=gain)

        hyper_target = self.hyper_learning_rate * float(control.hyper_rate_scale())
        real_target = self.real_learning_rate * float(control.real_rate_scale())
        self.last_control = control
        should_use_topos = self.topos_control_gain > 0.0 if use_topos is None else bool(use_topos)
        if should_use_topos:
            hyper_target, real_target = self._apply_topos_learning_rate_scale(
                hyper_target,
                real_target,
                hints=topos_hints,
                observed_depth=observed_depth,
                visited_volume=visited_volume,
            )
        else:
            self.last_topos_profile = None
            self.last_topos_effect = None
        _set_tape_learning_rate(self.hyper, hyper_target)
        _set_tape_learning_rate(self.real, real_target)
        return control

    def step(
        self,
        weights: Any,
        *,
        tune: bool = True,
        gain: float | None = None,
        control: Any | None = None,
        use_topos: bool | None = None,
        topos_hints: Mapping[str, Any] | None = None,
        observed_depth: int | None = None,
        visited_volume: int | None = None,
    ) -> Any:
        import spiraltorch as st

        weights = _require_tensor(st, weights, label="weights")
        if tune:
            self.tune(
                control=control,
                gain=gain,
                use_topos=use_topos,
                topos_hints=topos_hints,
                observed_depth=observed_depth,
                visited_volume=visited_volume,
            )
        self.hyper.apply(weights)
        self.real.apply(weights)
        return weights


def amegagrad(
    *shape_args: Any,
    curvature: float = -1.0,
    hyper_learning_rate: float = 0.05,
    real_learning_rate: float = 0.01,
    shape: Any | None = None,
    rows: Any | None = None,
    cols: Any | None = None,
    topos: Any | None = None,
    gain: float = 1.0,
    topos_control_gain: float | None = None,
    topos_observed_depth: int | None = None,
    topos_visited_volume: int | None = None,
) -> Amegagrad:
    return Amegagrad(
        *shape_args,
        curvature=curvature,
        hyper_learning_rate=hyper_learning_rate,
        real_learning_rate=real_learning_rate,
        shape=shape,
        rows=rows,
        cols=cols,
        topos=topos,
        gain=gain,
        topos_control_gain=topos_control_gain,
        topos_observed_depth=topos_observed_depth,
        topos_visited_volume=topos_visited_volume,
    )
