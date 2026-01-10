"""Optimizers built on top of SpiralTorch gradient tapes."""

from __future__ import annotations

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

    def shape(self) -> tuple[int, int]:
        return self.hyper.shape()

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

    def tune(self, control: Any | None = None, *, gain: float | None = None) -> Any:
        if control is None:
            control = self.desire_control(gain=gain)

        hyper_target = self.hyper_learning_rate * float(control.hyper_rate_scale())
        real_target = self.real_learning_rate * float(control.real_rate_scale())
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
    ) -> Any:
        import spiraltorch as st

        weights = _require_tensor(st, weights, label="weights")
        if tune:
            self.tune(control=control, gain=gain)
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
    )
