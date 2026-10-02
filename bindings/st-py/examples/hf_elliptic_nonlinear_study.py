"""Paired directional geometry controls using the existing restartable study loop.

Only the elliptic arm evaluates a geometric map: the public adapter delegates
its features and VJP to Rust. Ordinary controls consume Rust's anchor/Jacobian
but do not approximate the full map or make a matched-compute claim.
"""

from pathlib import Path

import torch
import spiraltorch as st
import spiraltorch.elliptic as elliptic_bridge

import hf_wave_gate_long_horizon as study


ARMS = ["tangent", "tanh_control", "elliptic"]


class ChartControl(st.EllipticResidualAdapter):
    def __init__(self, features, *, control, **options):
        if control not in {"tangent", "tanh_control"}:
            raise ValueError("unrecognized ordinary chart control")
        super().__init__(features, **options)
        self.control = control
        warp = st.EllipticWarp(**super().get_extra_state()["warp"])
        snapshot = warp.map_orientations_batch([1.0, 0.0, 0.0])
        self.register_buffer(
            "anchor", torch.tensor(snapshot.features, dtype=torch.float32)
        )
        rows = [snapshot.vjp([float(i == j) for j in range(9)])[1:] for i in range(9)]
        self.register_buffer("tangent", torch.tensor(rows, dtype=torch.float32))

    def chart_features(self, coordinates):
        displacement = coordinates @ self.tangent.T
        if self.control == "tanh_control":
            displacement = torch.tanh(displacement)
        return self.anchor + displacement

    def forward(self, value):
        if self.strength == 0:
            return value
        return value + self.strength * self.readout(
            self.chart_features(self.orientation(value))
        )

    def get_extra_state(self):
        return {**super().get_extra_state(), "control": self.control}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state.get("control") != self.control:
            raise ValueError("checkpoint control differs")
        super().set_extra_state(
            {key: value for key, value in state.items() if key != "control"}
        )


def adapter_for(arm, config, seed):
    if arm not in ARMS:
        raise ValueError("unrecognized elliptic study arm")
    options = {"strength": config["strength"], **config["warp"]}
    # This CPU study must neither seed nor draw from accelerator generators.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        if arm == "elliptic":
            return st.EllipticResidualAdapter(config["features"], **options)
        return ChartControl(config["features"], control=arm, **options)


def main():
    study.main(
        arms=ARMS,
        adapter_factory=adapter_for,
        adapter_sources={
            "study_adapter": Path(__file__),
            "elliptic_bridge": Path(elliptic_bridge.__file__),
        },
        result_schema="spiraltorch.elliptic_nonlinear_study.v1",
    )


if __name__ == "__main__":
    main()
