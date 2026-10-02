"""Paired local versus learned-context study with an ordinary tangent control.

The production geometry/blend/VJP is Rust-owned. Only the ordinary experimental
control spells out the matched blend in Torch. No speed or matched-compute claim.
"""

from pathlib import Path

import torch
import spiraltorch as st

import hf_elliptic_causal_study as causal


study = causal.study
ARMS = ["tangent", "elliptic", "gated_tangent", "gated_elliptic"]


def gated_mix(features, raw_mix):
    local = features.double()
    context = causal.causal_mix(features).double()
    gate = raw_mix.double().tanh()
    return ((1 - gate) * local + gate * context).float()


class GatedTangentControl(causal.CausalTangentControl):
    """Same shared scalar and causal contract, without the nonlinear geometry."""

    def __init__(self, features, **options):
        super().__init__(features, **options)
        self.raw_mix = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def _map_features(self, orientation):
        features = self.anchor + orientation[..., 1:] @ self.tangent.T
        return gated_mix(features, self.raw_mix)

    def get_extra_state(self):
        return {
            **super().get_extra_state(),
            "schema": "spiraltorch.elliptic_gated_tangent_control.v1",
        }

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or state.get("schema") != "spiraltorch.elliptic_gated_tangent_control.v1"
        ):
            raise ValueError("incompatible gated tangent control state")
        super().set_extra_state(
            {**state, "schema": "spiraltorch.elliptic_causal_tangent_control.v1"}
        )


def adapter_for(arm, config, seed):
    if arm not in ARMS:
        raise ValueError("unrecognized gated study arm")
    if arm in {"tangent", "elliptic"}:
        return causal.pointwise.adapter_for(arm, config, seed)
    options = {"strength": config["strength"], **config["warp"]}
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        kind = GatedTangentControl if arm == "gated_tangent" else st.EllipticGatedCausalResidualAdapter
        return kind(config["features"], **options)


def main():
    study.main(
        arms=ARMS,
        adapter_factory=adapter_for,
        adapter_sources={
            "study_adapter": Path(__file__),
            "causal_control": Path(causal.__file__),
            "pointwise_adapter": Path(causal.pointwise.__file__),
            "elliptic_bridge": Path(causal.pointwise.elliptic_bridge.__file__),
        },
        result_schema="spiraltorch.elliptic_gated_study.v1",
    )


if __name__ == "__main__":
    main()
