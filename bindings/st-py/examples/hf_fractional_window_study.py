"""Matched learning with fixed full-K normalization and different lag support."""

from pathlib import Path

import torch

import hf_fractional_angle_study as angle

gain, study = angle.gain, angle.study
ARMS = ["history_window_short", "history_window_full"]
WINDOWS = {ARMS[0]: [1, 3], ARMS[1]: None}
NORMALIZATION = "full_declared_kernel_before_window"
DOMAIN_POLICY = angle.DOMAIN_POLICY


def validate_protocol(config):
    study.require(config.get("schema") == "spiraltorch.fractional_window_protocol.v1"
                  and config.get("arms") == ARMS and config.get("reference_arm") == ARMS[0],
                  "incomplete window design")
    study.require(config.get("normalization_policy") == NORMALIZATION
                  and study.pilot.equal_state(config.get("lag_windows"), WINDOWS)
                  and all(type(value) is int for value in config["lag_windows"][ARMS[0]]),
                  "window support or normalization differs")
    angle.validate_protocol({**config, "schema": "spiraltorch.fractional_angle_protocol.v1",
                             "arms": angle.ARMS, "reference_arm": angle.ARMS[0]})


class StudyWindowHistory(angle.StudyAngularHistory):
    def get_extra_state(self):
        return {**super().get_extra_state(), "study_schema": "spiraltorch.fractional_window_control.v1",
                "normalization_policy": NORMALIZATION}


def adapter_for(arm, config, seed):
    validate_protocol(config)
    study.require(arm in ARMS, "unrecognized window study arm")
    window = config["lag_windows"][arm]
    with torch.device("cpu"):
        return StudyWindowHistory(config["features"], arm=arm, strength=config["strength"],
                                  lag_window=None if window is None else tuple(window), **config["kernel"])


def main():
    study.main(
        arms=ARMS, adapter_factory=adapter_for,
        adapter_sources={"window_study": Path(__file__), "angle_control": Path(angle.__file__),
                         "gain_control": Path(gain.__file__), "lag_control": Path(gain.lag.__file__),
                         "fractional_bridge": Path(gain.fractional_bridge.__file__)},
        result_schema="spiraltorch.fractional_window_study.v1",
    )


if __name__ == "__main__":
    main()
