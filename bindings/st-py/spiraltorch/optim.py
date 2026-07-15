"""Optimizers built on top of SpiralTorch gradient tapes."""

from __future__ import annotations

import math
from copy import deepcopy
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


def _configure_amegagrad_optimizer(
    hyper: Any,
    real: Any,
    hyper_learning_rate: float,
    real_learning_rate: float,
    optimizer_state: Mapping[str, Any] | None,
) -> None:
    import spiraltorch as st

    native = getattr(st, "_rs", None)
    configure = getattr(native, "_configure_amegagrad_optimizer", None)
    if not callable(configure):
        raise RuntimeError(
            "Amegagrad.tune requires the Rust atomic optimizer-configuration contract"
        )
    transported_state = dict(optimizer_state) if optimizer_state is not None else None
    configure(
        hyper,
        real,
        float(hyper_learning_rate),
        float(real_learning_rate),
        transported_state,
    )


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


def _required_rust_mapping(
    payload: Mapping[str, Any], field: str, *, label: str
) -> dict[str, Any]:
    value = payload.get(field)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Rust {label} returned invalid '{field}'")
    return dict(value)


def _required_rust_float(
    payload: Mapping[str, Any], field: str, *, label: str
) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Rust {label} returned invalid '{field}'")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RuntimeError(f"Rust {label} returned non-finite '{field}'")
    return numeric


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
        self._last_topos_signal: dict[str, Any] | None = None
        self._last_topos_snapshot: dict[str, Any] | None = None
        self._topos_snapshot_sequence = 0

    @property
    def last_topos_signal(self) -> dict[str, Any] | None:
        return deepcopy(self._last_topos_signal)

    @property
    def last_topos_snapshot(self) -> dict[str, Any] | None:
        return deepcopy(self._last_topos_snapshot)

    def _snapshot_control(self) -> dict[str, Any] | None:
        if self._last_topos_snapshot is None:
            return None
        return deepcopy(
            _required_rust_mapping(
                self._last_topos_snapshot,
                "control",
                label="Topos optimizer snapshot",
            )
        )

    @property
    def last_topos_hints(self) -> dict[str, Any] | None:
        control = self._snapshot_control() or self.last_topos_signal
        if control is None:
            return None
        hints = control.get("training_hints")
        return dict(hints) if isinstance(hints, Mapping) else None

    @property
    def last_topos_profile(self) -> dict[str, Any] | None:
        control = self._snapshot_control() or self.last_topos_signal
        if control is None:
            return None
        profile = control.get("runtime_profile")
        return dict(profile) if isinstance(profile, Mapping) else None

    @property
    def last_topos_effect(self) -> dict[str, Any] | None:
        if self._last_topos_snapshot is None:
            return None
        application = self._last_topos_snapshot.get("optimizer_application")
        return deepcopy(dict(application)) if isinstance(application, Mapping) else None

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
        used_training_gain = float(
            signal_options.pop("training_gain", self.topos_control_gain)
        )
        signal = st.topos_control_signal(
            guard,
            training_gain=used_training_gain,
            observed_depth=used_observed_depth,
            visited_volume=used_visited_volume,
            **signal_options,
        )
        self._last_topos_signal = deepcopy(dict(signal))
        return deepcopy(self._last_topos_signal)

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

    def _prepare_topos_optimizer_snapshot(
        self,
        hyper_target: float,
        real_target: float,
        *,
        hints: Mapping[str, Any] | None = None,
        observed_depth: int | None = None,
        visited_volume: int | None = None,
    ) -> tuple[float, float, dict[str, Any]]:
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
        next_sequence = self._topos_snapshot_sequence + 1
        snapshot = st.topos_optimizer_snapshot(
            guard,
            sequence=next_sequence,
            hyper_learning_rate=hyper_target,
            real_learning_rate=real_target,
            gain=self.topos_control_gain,
            training_hints=hints,
            observed_depth=used_observed_depth,
            visited_volume=used_visited_volume,
        )
        if not isinstance(snapshot, Mapping):
            raise RuntimeError(
                "Rust Topos optimizer snapshot returned a non-mapping payload"
            )
        returned_sequence = snapshot.get("sequence")
        if (
            isinstance(returned_sequence, bool)
            or not isinstance(returned_sequence, int)
            or returned_sequence != next_sequence
        ):
            raise RuntimeError(
                "Rust Topos optimizer snapshot returned an invalid sequence"
            )
        control = _required_rust_mapping(
            snapshot,
            "control",
            label="Topos optimizer snapshot",
        )
        plan = _required_rust_mapping(control, "training_plan", label="Topos control")
        application = _required_rust_mapping(
            snapshot,
            "optimizer_application",
            label="Topos optimizer snapshot",
        )
        planned_rate_scale = _required_rust_float(
            plan,
            "rate_scale",
            label="Topos training plan",
        )
        applied_rate_scale = _required_rust_float(
            application,
            "rate_scale",
            label="Topos optimizer application",
        )
        if planned_rate_scale != applied_rate_scale:
            raise RuntimeError(
                "Rust Topos optimizer snapshot disagrees on planned and applied rate_scale"
            )
        hyper_scaled = _required_rust_float(
            application,
            "hyper_learning_rate",
            label="Topos optimizer application",
        )
        real_scaled = _required_rust_float(
            application,
            "real_learning_rate",
            label="Topos optimizer application",
        )

        return hyper_scaled, real_scaled, dict(snapshot)

    def topos_telemetry_contract(self) -> dict[str, Any]:
        """Fuse cached Topos state through the canonical Rust telemetry contract."""

        import spiraltorch as st

        if self._last_topos_snapshot is None:
            control = self.last_topos_signal or self.topos_control_signal()
            topos_payload = dict(control)
        else:
            control = self._snapshot_control()
            if control is None:  # pragma: no cover - guarded by snapshot validation
                raise RuntimeError("Rust Topos optimizer snapshot omitted its control")
            application = _required_rust_mapping(
                self._last_topos_snapshot,
                "optimizer_application",
                label="Topos optimizer snapshot",
            )
            topos_payload = dict(control)
            # Preserve historical flat telemetry names as projections of one snapshot.
            topos_payload["optimizer_effect"] = application
            topos_payload["optimizer_snapshot"] = {
                "sequence": self._last_topos_snapshot["sequence"]
            }
        contract = st.zspace_telemetry_fusion([{"topos": topos_payload}])
        if not isinstance(contract, Mapping):
            raise RuntimeError(
                "Rust Z-space telemetry fusion returned a non-mapping payload"
            )
        return dict(contract)

    def topos_telemetry_payload(self) -> dict[str, float]:
        """Project the Rust-owned Topos telemetry contract to its flat payload."""

        contract = self.topos_telemetry_contract()
        payload = contract.get("payload")
        if not isinstance(payload, Mapping):
            raise RuntimeError("Rust Z-space telemetry fusion omitted its payload")
        return {str(key): float(value) for key, value in payload.items()}

    def topos_diagnostics(self) -> dict[str, Any]:
        """Return projections of the authoritative Topos optimizer snapshot."""

        snapshot = self.last_topos_snapshot
        signal = self._snapshot_control() or self.last_topos_signal
        training_plan = None
        if signal is not None and isinstance(signal.get("training_plan"), Mapping):
            training_plan = dict(signal["training_plan"])

        return {
            "snapshot": snapshot,
            "signal": dict(signal) if signal else None,
            "training_hints": dict(self.last_topos_hints)
            if self.last_topos_hints
            else None,
            "training_plan": training_plan,
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
        should_use_topos = self.topos_control_gain > 0.0 if use_topos is None else bool(use_topos)
        pending_snapshot: dict[str, Any] | None = None
        if should_use_topos:
            hyper_target, real_target, pending_snapshot = (
                self._prepare_topos_optimizer_snapshot(
                    hyper_target,
                    real_target,
                    hints=topos_hints,
                    observed_depth=observed_depth,
                    visited_volume=visited_volume,
                )
            )
        application = (
            None
            if pending_snapshot is None
            else _required_rust_mapping(
                pending_snapshot,
                "optimizer_application",
                label="Topos optimizer snapshot",
            )
        )
        _configure_amegagrad_optimizer(
            self.hyper,
            self.real,
            hyper_target,
            real_target,
            application,
        )

        self.last_control = control
        if pending_snapshot is None:
            self._last_topos_snapshot = None
        else:
            snapshot_control = _required_rust_mapping(
                pending_snapshot,
                "control",
                label="Topos optimizer snapshot",
            )
            self._topos_snapshot_sequence = int(pending_snapshot["sequence"])
            self._last_topos_snapshot = deepcopy(pending_snapshot)
            self._last_topos_signal = deepcopy(snapshot_control)
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
        native = getattr(st, "_rs", None)
        apply_step = getattr(native, "_apply_amegagrad_step", None)
        if not callable(apply_step):
            raise RuntimeError(
                "Amegagrad.step requires the Rust atomic combined-step contract"
            )
        apply_step(self.hyper, self.real, weights)
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
