"""Matched learned-amplitude GL history versus an ordinary two-tap filter.

The ordinary Torch control is an independent reference, not a fractional
backend. All GL coefficients, gain conversion and pullbacks remain in Rust.
"""

import math
from pathlib import Path

import torch
import spiraltorch as st
import spiraltorch.fractional_autograd as fractional_bridge

import hf_fractional_lag_study as lag

study = lag.study
ARMS = ["ordinary_gain_short", "history_gain_short", "history_gain_full"]
INITIAL_ALPHA = 2.0
INITIAL_GAIN = math.sqrt(5.0)
INITIAL_ANGLE = 0.4636476090008061  # Frozen recipe, not host-libm atan rounding.


def validate_protocol(config):
    study.require(config.get("schema") == "spiraltorch.fractional_gain_protocol.v1",
                  "gain protocol differs")
    study.require(config.get("arms") == ARMS
                  and config.get("reference_arm") == ARMS[0], "incomplete gain design")
    for key, expected in (("initial_alpha", INITIAL_ALPHA), ("initial_gain", INITIAL_GAIN),
                          ("initial_history_angle", INITIAL_ANGLE)):
        study.require(type(config.get(key)) in (int, float) and config[key] == expected,
                      f"initial gain-study coordinate differs: {key}")
    kernel = config.get("kernel", {})
    study.require(type(config.get("short_kernel_len")) is int
                  and config["short_kernel_len"] == 3, "short history needs two past taps")
    study.require(type(kernel.get("kernel_len")) is int and kernel["kernel_len"] > 3
                  and type(kernel.get("step")) in (int, float) and kernel["step"] == 1.,
                  "gain-study kernel differs")


class OrdinaryGainShort(lag.CausalLagGate):
    """Two past taps exp(log_gain)*[-cos(angle), sin(angle)], ordinary Torch.

    Shape and amplitude are learned with two scalars, in addition to the same
    feature gates as GL. Angle and log-order are different optimizer charts.
    """

    def __init__(self, features, **options):
        super().__init__(features, **options)
        study.require(self.kernel["kernel_len"] == 3, "ordinary control needs two past taps")
        self.history_angle = torch.nn.Parameter(torch.tensor(INITIAL_ANGLE, dtype=torch.float32))
        self.log_gain = torch.nn.Parameter(torch.tensor(math.log(INITIAL_GAIN), dtype=torch.float32))

    @property
    def gain(self):
        study.require(self.log_gain.ndim == 0 and self.log_gain.dtype == torch.float32,
                      "ordinary gain requires scalar float32")
        value = self.log_gain.detach().exp()
        study.require(bool(torch.isfinite(value)) and bool(value > 0),
                      "ordinary gain must be positive finite f32")
        return float(value)

    def history(self, value):
        super().history(value)
        self.gain  # Validate the same f32 chart domain before the reference map.
        angle = self.history_angle.double()
        coefficients = (self.log_gain.exp().double()
                        * torch.stack((-angle.cos(), angle.sin()))).float()
        work = value.double()
        first = torch.cat((torch.zeros_like(work[:, :1]), work[:, :-1]), dim=1)
        second = torch.cat((torch.zeros_like(work[:, :min(2, work.shape[1])]),
                            work[:, :-2]), dim=1)
        output = (coefficients[0].double()*first + coefficients[1].double()*second).float()
        study.require(bool(torch.isfinite(output).all()), "nonfinite ordinary history")
        return output

    def get_extra_state(self):
        return {**super().get_extra_state(), "schema": "spiraltorch.ordinary_gain_short.v1",
                "arm": ARMS[0], "initial_history_angle": INITIAL_ANGLE,
                "initial_gain": INITIAL_GAIN, "accumulation_dtype": "float64"}


class StudyGainHistory(st.FractionalGainHistoryAdapter):
    def __init__(self, features, *, arm, **options):
        super().__init__(features, initial_alpha=INITIAL_ALPHA,
                         initial_gain=INITIAL_GAIN, **options)
        self.arm = arm

    def get_extra_state(self):
        return {**super().get_extra_state(), "study_schema": "spiraltorch.fractional_gain_control.v1",
                "arm": self.arm, "initial_alpha": INITIAL_ALPHA, "initial_gain": INITIAL_GAIN}

    def set_extra_state(self, state):
        study.require(state == self.get_extra_state(), "gain study recipe differs")


def adapter_for(arm, config, seed):
    validate_protocol(config)
    study.require(arm in ARMS, "unrecognized gain study arm")
    kernel = dict(config["kernel"])
    if arm.endswith("_short"):
        kernel["kernel_len"] = config["short_kernel_len"]
    with torch.device("cpu"):
        if arm == ARMS[0]:
            return OrdinaryGainShort(config["features"], strength=config["strength"], **kernel)
        return StudyGainHistory(config["features"], arm=arm,
                                strength=config["strength"], **kernel)


def main():
    study.main(
        arms=ARMS, adapter_factory=adapter_for,
        adapter_sources={"gain_study": Path(__file__), "lag_control": Path(lag.__file__),
                         "fractional_bridge": Path(fractional_bridge.__file__)},
        result_schema="spiraltorch.fractional_gain_study.v1",
    )


if __name__ == "__main__":
    main()
