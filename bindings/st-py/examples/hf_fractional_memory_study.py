"""Offline paired pointwise/fixed-order/learned-order LM study.

No-adapter evaluation is the shared frozen baseline. Fractional mathematics
comes only from the public Rust-backed adapter; the pointwise control is an
ordinary feature gate, with neither token mixing nor a dummy alpha parameter.
"""

from pathlib import Path

import torch
import spiraltorch as st
import spiraltorch.fractional_autograd as fractional_bridge
from spiraltorch.geometry_autograd import _input, _strength

import hf_wave_gate_long_horizon as study

ARMS = ["pointwise", "fractional_fixed", "fractional_learned"]


class PointwiseGate(torch.nn.Module):
    def __init__(self, features, *, strength):
        super().__init__()
        study.require(type(features) is int and features > 0, "invalid features")
        self.features, self.strength = features, _strength(strength)
        self.gate = torch.nn.Parameter(torch.zeros(features, dtype=torch.float32))

    def forward(self, value):
        _input(value)
        study.require(value.ndim == 3 and value.shape[-1] == self.features, "expected BTF input")
        study.require(value.dtype == self.gate.dtype and value.device == self.gate.device,
                      "pointwise input/parameter dtype or device differs")
        return value + self.strength * self.gate.tanh() * value

    def get_extra_state(self):
        return {"schema": "spiraltorch.fractional_pointwise_control.v1",
                "features": self.features, "strength": self.strength}

    def set_extra_state(self, state):
        study.require(state == self.get_extra_state(), "pointwise recipe differs")


class StudyFractional(st.FractionalMemoryAdapter):
    def __init__(self, features, *, learnable_alpha, **options):
        super().__init__(features, **options)
        study.require(type(learnable_alpha) is bool, "invalid alpha mode")
        self.learnable_alpha = learnable_alpha
        self.log_alpha.requires_grad_(learnable_alpha)

    def get_extra_state(self):
        return {**super().get_extra_state(), "study_schema": "spiraltorch.fractional_control.v1",
                "learnable_alpha": self.learnable_alpha}

    def set_extra_state(self, state):
        # In this study the plan fixes the complete recipe and frozen/trainable mode.
        study.require(state == self.get_extra_state(), "fractional study recipe differs")


def adapter_for(arm, config, seed):
    study.require(arm in ARMS, "unrecognized fractional arm")
    study.require(config["kernel"]["step"] == 1.0, "pointwise gain control requires step=1")
    study.require(type(config["kernel"]["kernel_len"]) is int and config["kernel"]["kernel_len"] > 1,
                  "history comparison requires more than one GL coefficient")
    # Zero initialization needs no RNG and must not inherit a caller's meta/GPU device.
    with torch.device("cpu"):
        if arm == "pointwise":
            return PointwiseGate(config["features"], strength=config["strength"])
        return StudyFractional(
            config["features"], strength=config["strength"],
            initial_alpha=config["initial_alpha"],
            learnable_alpha=arm == "fractional_learned", **config["kernel"],
        )


def main():
    study.main(
        arms=ARMS, adapter_factory=adapter_for,
        adapter_sources={"fractional_study": Path(__file__),
                         "fractional_bridge": Path(fractional_bridge.__file__)},
        result_schema="spiraltorch.fractional_memory_study.v1",
    )


if __name__ == "__main__":
    main()
