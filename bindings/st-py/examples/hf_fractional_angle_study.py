"""Matched angular GL learning against an independent ordinary short filter.

Every arm shares Rust's positive-order chart domain. The ordinary control
still evaluates its filter and gradients in Torch, never via GL coefficients.
"""

import math
from pathlib import Path

import torch
import spiraltorch as st

import hf_fractional_gain_study as gain

study = gain.study
ARMS = ["ordinary_angle_short", "history_angle_short", "history_angle_full"]
DOMAIN_POLICY = "terminal_all_arms_no_projection_no_endpoints"


def validate_protocol(config):
    study.require(config.get("schema") == "spiraltorch.fractional_angle_protocol.v1"
                  and config.get("arms") == ARMS and config.get("reference_arm") == ARMS[0],
                  "incomplete angular design")
    study.require(config.get("angle_domain_policy") == DOMAIN_POLICY,
                  "angular domain policy differs")
    # Reuse the unchanged amplitude, initialization and kernel contracts.
    gain.validate_protocol({**config, "schema": "spiraltorch.fractional_gain_protocol.v1",
                            "arms": gain.ARMS, "reference_arm": gain.ARMS[0]})


class BoundedStudyAngle:
    @property
    def alpha(self):
        coordinate = self.history_angle
        study.require(coordinate.ndim == 0 and coordinate.dtype == torch.float32,
                      "angular study requires scalar float32")
        angle = float(coordinate.detach())
        try:
            return st.FractionalGlAngleChart(angle).alpha
        except ValueError as error:
            raise study.TerminalStudyError("angular chart domain exit", details={
                "condition": "angle_domain_exit", "arm": self.arm,
                "observed_angle": angle if math.isfinite(angle) else repr(angle),
                "policy": DOMAIN_POLICY,
            }) from error


class OrdinaryAngleShort(BoundedStudyAngle, gain.OrdinaryGainShort):
    def __init__(self, features, **options):
        super().__init__(features, **options)
        self.arm = ARMS[0]

    def history(self, value):
        self.alpha  # Domain observation only; the independent map stays in Torch.
        return super().history(value)

    def get_extra_state(self):
        return {**super().get_extra_state(), "schema": "spiraltorch.ordinary_angle_short.v1",
                "arm": self.arm, "angle_domain_policy": DOMAIN_POLICY}


class StudyAngularHistory(BoundedStudyAngle, st.FractionalAngleGainHistoryAdapter):
    def __init__(self, features, *, arm, **options):
        super().__init__(features, initial_angle=gain.INITIAL_ANGLE,
                         initial_gain=gain.INITIAL_GAIN, **options)
        self.arm = arm

    def _alpha_tensor(self):
        self.alpha
        return super()._alpha_tensor()

    def get_extra_state(self):
        return {**super().get_extra_state(), "study_schema": "spiraltorch.fractional_angle_control.v1",
                "arm": self.arm, "angle_domain_policy": DOMAIN_POLICY,
                "initial_history_angle": gain.INITIAL_ANGLE, "initial_gain": gain.INITIAL_GAIN}

    def set_extra_state(self, state):
        study.require(state == self.get_extra_state(), "angular study recipe differs")


def adapter_for(arm, config, seed):
    validate_protocol(config)
    study.require(arm in ARMS, "unrecognized angular study arm")
    kernel = dict(config["kernel"])
    if arm.endswith("_short"):
        kernel["kernel_len"] = config["short_kernel_len"]
    with torch.device("cpu"):
        if arm == ARMS[0]:
            return OrdinaryAngleShort(config["features"], strength=config["strength"], **kernel)
        return StudyAngularHistory(config["features"], arm=arm, strength=config["strength"], **kernel)


def main():
    study.main(
        arms=ARMS, adapter_factory=adapter_for,
        adapter_sources={"angle_study": Path(__file__), "gain_control": Path(gain.__file__),
                         "lag_control": Path(gain.lag.__file__),
                         "fractional_bridge": Path(gain.fractional_bridge.__file__)},
        result_schema="spiraltorch.fractional_angle_study.v1",
    )


if __name__ == "__main__":
    main()
