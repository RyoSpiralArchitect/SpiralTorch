"""Paired fixed-anchor versus learned-context geometry training.

All four arms have the same parameters, initialization and update schedule.
Rust owns the nonlinear feature maps and VJPs; Torch implements only the ordinary
linearized control. This is not a matched-compute or speed comparison.
"""

from pathlib import Path

import torch
import spiraltorch as st

import hf_elliptic_gated_study as gated


study = gated.study
ARMS = ["anchored_tangent", "anchored_elliptic", "gated_tangent", "gated_elliptic"]


def anchored_mix(features, anchor, raw_mix):
    gate = raw_mix.double().tanh()
    return ((1 - gate) * features.double() + gate * anchor.double()).float()


class AnchoredTangentControl(st.EllipticAnchoredResidualAdapter):
    """Ordinary affine features and a fixed anchor, not the nonlinear map."""

    def __init__(self, features, **options):
        super().__init__(features, **options)
        warp = st.EllipticWarp(**super().get_extra_state()["warp"])
        snapshot = warp.map_orientations_batch([1.0, 0.0, 0.0])
        self.register_buffer("anchor", torch.tensor(snapshot.features, dtype=torch.float32))
        rows = [snapshot.vjp([float(i == j) for j in range(9)])[1:] for i in range(9)]
        self.register_buffer("tangent", torch.tensor(rows, dtype=torch.float32))

    @property
    def execution_backend(self):
        return "torch_cpu_rust_anchor"

    def _map_features(self, orientation):
        features = self.anchor + orientation[..., 1:] @ self.tangent.T
        return anchored_mix(features, self.anchor, self.raw_mix)

    def get_extra_state(self):
        return {
            **super().get_extra_state(),
            "schema": "spiraltorch.elliptic_anchored_tangent_control.v1",
        }

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or state.get("schema") != "spiraltorch.elliptic_anchored_tangent_control.v1"
        ):
            raise ValueError("incompatible anchored tangent control state")
        super().set_extra_state(
            {**state, "schema": "spiraltorch.elliptic_anchored_residual_adapter.v1"}
        )


def adapter_for(arm, config, seed):
    if arm not in ARMS:
        raise ValueError("unrecognized anchored study arm")
    if arm.startswith("gated_"):
        return gated.adapter_for(arm, config, seed)
    options = {"strength": config["strength"], **config["warp"]}
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        kind = AnchoredTangentControl if arm == "anchored_tangent" else st.EllipticAnchoredResidualAdapter
        return kind(config["features"], **options)


def main():
    study.main(
        arms=ARMS,
        adapter_factory=adapter_for,
        adapter_sources={
            "study_adapter": Path(__file__),
            "gated_control": Path(gated.__file__),
            "causal_control": Path(gated.causal.__file__),
            "pointwise_adapter": Path(gated.causal.pointwise.__file__),
            "elliptic_bridge": Path(gated.causal.pointwise.elliptic_bridge.__file__),
        },
        result_schema="spiraltorch.elliptic_anchored_study.v1",
    )


if __name__ == "__main__":
    main()
