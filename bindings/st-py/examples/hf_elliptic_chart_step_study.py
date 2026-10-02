"""Matched Adam proposals with explicit Rust chart-direction preconditioning.

No gradient rewrite: Adam consumes the true loss gradient and keeps its ordinary
moments. Only the orientation projection's proposed displacement is transformed.
The mean two-coordinate metric omits hidden-input covariance and downstream
readout/loss curvature; this is not full natural gradient or a loss Hessian.
"""

import copy
from pathlib import Path

import torch
import spiraltorch as st

import hf_elliptic_anchored_study as anchored

study = anchored.study
ARMS = ["adam_tangent", "adam_elliptic", "chart_tangent", "chart_elliptic"]


class CaptureChart:
    def _map_features(self, orientation):
        if torch.is_grad_enabled():
            self._chart_orientation = orientation.detach().clone()
            self._chart_forward_count = getattr(self, "_chart_forward_count", 0) + 1
        return super()._map_features(orientation)


class ChartTangent(CaptureChart, anchored.AnchoredTangentControl):
    pass


class ChartElliptic(CaptureChart, st.EllipticAnchoredResidualAdapter):
    pass


class ChartAdam(torch.optim.Adam):
    """Experiment-only, single-forward CPU-f32 Adam proposal orchestrator."""

    def __init__(self, adapter, *, enabled, tangent, learning_rate, relative_damping):
        study.require(
            type(enabled) is bool and type(tangent) is bool, "invalid chart flags"
        )
        study.require(1e-6 <= relative_damping <= 1.0, "invalid relative damping")
        self.adapter = adapter
        self.recipe = {
            "schema": "spiraltorch.chart_adam.v1",
            "enabled": enabled,
            "tangent": tangent,
            "relative_damping": relative_damping,
        }
        self.last_step_diagnostics = None
        super().__init__(adapter.parameters(), lr=learning_rate)

    def zero_grad(self, set_to_none=True):
        self.adapter._chart_orientation = None
        self.adapter._chart_forward_count = 0
        return super().zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {**super().state_dict(), "chart_recipe": dict(self.recipe)}

    def load_state_dict(self, state):
        study.require(
            state.get("chart_recipe") == self.recipe, "chart optimizer recipe mismatch"
        )
        super().load_state_dict(
            {key: value for key, value in state.items() if key != "chart_recipe"}
        )
        self.adapter._chart_orientation = None
        self.adapter._chart_forward_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        study.require(closure is None, "chart optimizer does not support closures")
        if not self.recipe["enabled"]:
            result = super().step()
            self.last_step_diagnostics = {"enabled": False}
            return result
        adapter = self.adapter
        orientation = getattr(adapter, "_chart_orientation", None)
        study.require(
            orientation is not None and adapter._chart_forward_count == 1,
            "one fresh forward required; gradient accumulation is unsupported",
        )
        parameters = list(adapter.parameters())
        study.require(
            all(
                p.device.type == "cpu" and p.dtype == torch.float32 for p in parameters
            ),
            "chart study requires CPU float32 parameters",
        )
        study.require(
            all(
                p.grad is not None and bool(torch.isfinite(p.grad).all())
                for p in parameters
            ),
            "finite gradients required",
        )
        points = (
            [1.0, 0.0, 0.0]
            if self.recipe["tangent"]
            else orientation.flatten().tolist()
        )
        snapshot = adapter._warp.map_orientations_batch(points)
        weight, bias = adapter.orientation.weight, adapter.orientation.bias
        before_weight, before_bias = weight.clone(), bias.clone()
        before_parameters = [p.clone() for p in parameters]
        before_state = copy.deepcopy(super().state_dict())
        try:
            result = super().step()
            proposal = torch.cat(
                (weight - before_weight, (bias - before_bias)[:, None]), dim=1
            )
            receipt = snapshot.chart_step(
                proposal.flatten().tolist(), self.recipe["relative_damping"]
            )
            direction = torch.tensor(receipt.values).reshape_as(proposal)
            if receipt.proposal_l2 > 0:
                weight.copy_(before_weight + direction[:, :-1])
                bias.copy_(before_bias + direction[:, -1])
            actual = torch.cat(
                (weight - before_weight, (bias - before_bias)[:, None]), dim=1
            )
            study.require(
                all(bool(torch.isfinite(p).all()) for p in parameters),
                "nonfinite chart update",
            )
            gradient = torch.cat((weight.grad, bias.grad[:, None]), dim=1)
            self.last_step_diagnostics = {
                "enabled": True,
                "metric": receipt.metric,
                "damped_condition": receipt.damped_condition,
                "proposal_l2": receipt.proposal_l2,
                "step_l2": receipt.step_l2,
                "applied_step_l2": float(actual.double().norm()),
                "cosine": receipt.cosine,
                "gradient_dot_proposal": float(
                    (gradient.double() * proposal.double()).sum()
                ),
                "gradient_dot_applied_step": float(
                    (gradient.double() * actual.double()).sum()
                ),
            }
        except BaseException:
            for parameter, previous in zip(parameters, before_parameters):
                parameter.copy_(previous)
            super().load_state_dict(before_state)
            self.last_step_diagnostics = None
            raise
        finally:
            adapter._chart_orientation = None
            adapter._chart_forward_count = 0
        return result


def adapter_for(arm, config, seed):
    study.require(arm in ARMS, "unrecognized chart study arm")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        kind = ChartTangent if arm.endswith("tangent") else ChartElliptic
        return kind(config["features"], strength=config["strength"], **config["warp"])


def optimizer_for(adapter, arm, config):
    study.require(arm in ARMS, "unrecognized chart optimizer arm")
    return ChartAdam(
        adapter,
        enabled=arm.startswith("chart_"),
        tangent=arm.endswith("tangent"),
        learning_rate=config["learning_rate"],
        relative_damping=config["relative_damping"],
    )


def main():
    study.main(
        arms=ARMS,
        adapter_factory=adapter_for,
        optimizer_factory=optimizer_for,
        adapter_sources={
            "chart_study": Path(__file__),
            "anchored_control": Path(anchored.__file__),
            "gated_control": Path(anchored.gated.__file__),
            "causal_control": Path(anchored.gated.causal.__file__),
            "pointwise_control": Path(anchored.gated.causal.pointwise.__file__),
            "elliptic_bridge": Path(
                anchored.gated.causal.pointwise.elliptic_bridge.__file__
            ),
        },
        result_schema="spiraltorch.elliptic_chart_step_study.v1",
    )


if __name__ == "__main__":
    main()
