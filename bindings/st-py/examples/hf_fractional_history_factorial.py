"""Offline history-length x coefficient-energy learning comparison.

All four arms use Rust maps and differentials. This client only chooses the
declared recipe; it does not implement coefficients or their normalization.
"""

import math
from pathlib import Path

import torch
import spiraltorch as st
import spiraltorch.fractional_autograd as fractional_bridge

import hf_wave_gate_long_horizon as study

ARMS = ["history_raw_short", "history_raw_full", "history_l2_short", "history_l2_full"]
INITIAL_ALPHA = 2.0
L2_GAIN = math.sqrt(5.0)


def validate_protocol(config):
    study.require(config.get("schema") == "spiraltorch.fractional_history_factorial_protocol.v1",
                  "history factorial protocol differs")
    study.require(config.get("arms") == ARMS and config.get("reference_arm") == "history_raw_full",
                  "incomplete history factorial design")
    study.require(type(config.get("initial_alpha")) in (int, float)
                  and config["initial_alpha"] == INITIAL_ALPHA, "initial order differs")
    study.require(type(config.get("history_l2_gain")) in (int, float)
                  and config["history_l2_gain"] == L2_GAIN, "initial filter energy differs")
    study.require(type(config.get("short_kernel_len")) is int
                  and config["short_kernel_len"] == 3, "short history needs exactly two past taps")
    kernel = config.get("kernel", {})
    study.require(type(kernel.get("step")) in (int, float) and kernel["step"] == 1.0,
                  "history factorial requires step=1")
    study.require(type(kernel.get("kernel_len")) is int and kernel["kernel_len"] > 3,
                  "full history must extend past the short filter")


class _StudyRecipe:
    def __init__(self, features, *, arm, **options):
        super().__init__(features, initial_alpha=INITIAL_ALPHA, **options)
        self.arm = arm

    def get_extra_state(self):
        return {**super().get_extra_state(),
                "study_schema": "spiraltorch.fractional_history_factorial_control.v1",
                "arm": self.arm, "initial_alpha": INITIAL_ALPHA,
                "learnable_alpha": self.log_alpha.requires_grad}

    def set_extra_state(self, state):
        study.require(state == self.get_extra_state(), "history factorial recipe differs")


class RawHistory(_StudyRecipe, st.FractionalHistoryAdapter):
    pass


class NormalizedHistory(_StudyRecipe, st.FractionalL2HistoryAdapter):
    pass


def adapter_for(arm, config, seed):
    validate_protocol(config)
    study.require(arm in ARMS, "unrecognized history factorial arm")
    kernel = dict(config["kernel"])
    if arm.endswith("_short"):
        kernel["kernel_len"] = config["short_kernel_len"]
    normalized = arm.startswith("history_l2_")
    constructor = NormalizedHistory if normalized else RawHistory
    options = {"gain": config["history_l2_gain"]} if normalized else {}
    with torch.device("cpu"):
        return constructor(config["features"], arm=arm,
                           strength=config["strength"], **kernel, **options)


def main():
    study.main(
        arms=ARMS, adapter_factory=adapter_for,
        adapter_sources={"history_factorial": Path(__file__),
                         "fractional_bridge": Path(fractional_bridge.__file__)},
        result_schema="spiraltorch.fractional_history_factorial_study.v1",
    )


if __name__ == "__main__":
    main()
