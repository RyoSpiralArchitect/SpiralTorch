"""Paired 2x2 study of a geometric map and causal token mixing.

Rust owns the elliptic map and the complete geometric causal VJP. The ordinary
tangent control uses Rust's anchor/Jacobian and explicit Torch attention. This
is a learning comparison, not matched computation or a speed benchmark.
"""

from pathlib import Path

import torch
import spiraltorch as st

import hf_elliptic_nonlinear_study as pointwise


study = pointwise.study
ARMS = ["tangent", "elliptic", "causal_tangent", "causal_elliptic"]


def causal_mix(features):
    """Ordinary tied Q/K/V oracle, with the same scale and structural mask."""
    values = features.double()
    scores = values @ values.transpose(-1, -2) / 3
    mask = torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool).tril()
    return (scores.masked_fill(~mask, -torch.inf).softmax(-1) @ values).float()


class CausalTangentControl(st.EllipticCausalResidualAdapter):
    """Linearized feature map with ordinary causal mixing, not a geometric map."""

    def __init__(self, features, **options):
        super().__init__(features, **options)
        warp = st.EllipticWarp(**super().get_extra_state()["warp"])
        snapshot = warp.map_orientations_batch([1.0, 0.0, 0.0])
        self.register_buffer(
            "anchor", torch.tensor(snapshot.features, dtype=torch.float32)
        )
        rows = [snapshot.vjp([float(i == j) for j in range(9)])[1:] for i in range(9)]
        self.register_buffer("tangent", torch.tensor(rows, dtype=torch.float32))

    @property
    def execution_backend(self):
        return "torch_cpu_rust_anchor"

    def _map_features(self, orientation):
        features = self.anchor + orientation[..., 1:] @ self.tangent.T
        return causal_mix(features)

    def get_extra_state(self):
        return {
            **super().get_extra_state(),
            "schema": "spiraltorch.elliptic_causal_tangent_control.v1",
        }

    def set_extra_state(self, state):
        if (
            not isinstance(state, dict)
            or state.get("schema") != "spiraltorch.elliptic_causal_tangent_control.v1"
        ):
            raise ValueError("incompatible causal tangent control state")
        super().set_extra_state(
            {**state, "schema": "spiraltorch.elliptic_causal_residual_adapter.v1"}
        )


def adapter_for(arm, config, seed):
    if arm not in ARMS:
        raise ValueError("unrecognized causal study arm")
    if arm in {"tangent", "elliptic"}:
        return pointwise.adapter_for(arm, config, seed)
    options = {"strength": config["strength"], **config["warp"]}
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        kind = (
            CausalTangentControl
            if arm == "causal_tangent"
            else st.EllipticCausalResidualAdapter
        )
        return kind(config["features"], **options)


def main():
    study.main(
        arms=ARMS,
        adapter_factory=adapter_for,
        adapter_sources={
            "study_adapter": Path(__file__),
            "pointwise_adapter": Path(pointwise.__file__),
            "elliptic_bridge": Path(pointwise.elliptic_bridge.__file__),
        },
        result_schema="spiraltorch.elliptic_causal_study.v1",
    )


if __name__ == "__main__":
    main()
