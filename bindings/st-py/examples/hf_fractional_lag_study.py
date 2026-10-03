"""Offline one-lag parity and learned-order initialization controls.

The ordinary control is a Torch shift. All fractional maps and differentials,
including the order derivative at alpha=1, remain owned by the Rust core.
"""

from pathlib import Path

import torch
import spiraltorch as st
import spiraltorch.fractional_autograd as fractional_bridge
from spiraltorch.geometry_autograd import _input, _strength

import hf_wave_gate_long_horizon as study

ARMS = ["lag1", "history_fixed_one", "history_learned_one", "history_learned_half"]
INITIAL_ORDERS = {"history_fixed_one": 1.0, "history_learned_one": 1.0,
                  "history_learned_half": 0.5}


class CausalLagGate(torch.nn.Module):
    """Independent local/history gains with h[t]=-x[t-1], h[0]=0.

    The sign matches strictly-past GL at alpha=step=1. The shared K-based
    input budget is a comparison constraint, not the cost of this one-tap map.
    """

    def __init__(self, features, *, strength, kernel_len, step=1.0,
                 max_values=1_048_576, max_products=16_777_216):
        super().__init__()
        study.require(type(features) is int and features > 0, "invalid features")
        study.require(type(step) in (int, float) and step == 1.0,
                      "lag comparison requires step=1")
        study.require(all(type(v) is int and v > 0 for v in
                          (kernel_len, max_values, max_products))
                      and 1 < kernel_len <= max_products, "invalid history budget")
        self.features, self.strength = features, _strength(strength)
        self.kernel = {"kernel_len": kernel_len, "step": step,
                       "max_values": max_values, "max_products": max_products}
        self.gate = torch.nn.Parameter(torch.zeros(features, dtype=torch.float32))
        self.local_gate = torch.nn.Parameter(torch.zeros(features, dtype=torch.float32))

    def history(self, value):
        _input(value)
        study.require(value.ndim == 3 and min(value.shape) > 0
                      and value.shape[-1] == self.features, "expected nonempty BTF input")
        study.require(all(p.dtype == value.dtype and p.device == value.device
                          for p in self.parameters()), "lag dtype or device differs")
        study.require(bool(torch.isfinite(value).all())
                      and all(bool(torch.isfinite(p).all()) for p in self.parameters()),
                      "nonfinite lag input or parameters")
        study.require(value.numel() <= self.kernel["max_values"]
                      and value.numel() * self.kernel["kernel_len"] <= self.kernel["max_products"],
                      "history budget exceeded")
        return torch.cat((torch.zeros_like(value[:, :1]), -value[:, :-1]), dim=1)

    def forward(self, value):
        history = self.history(value)
        local = value + self.strength * self.local_gate.tanh() * value
        return local + self.strength * self.gate.tanh() * history

    def get_extra_state(self):
        return {"schema": "spiraltorch.ordinary_causal_lag_control.v1",
                "features": self.features, "strength": self.strength,
                "kernel": dict(self.kernel)}

    def set_extra_state(self, state):
        study.require(state == self.get_extra_state(), "lag study recipe differs")


class StudyHistory(st.FractionalHistoryAdapter):
    def __init__(self, features, *, arm, **options):
        study.require(arm in INITIAL_ORDERS, "invalid history arm")
        super().__init__(features, initial_alpha=INITIAL_ORDERS[arm], **options)
        self.arm = arm
        self.log_alpha.requires_grad_(arm != "history_fixed_one")

    def get_extra_state(self):
        return {**super().get_extra_state(),
                "study_schema": "spiraltorch.fractional_lag_control.v1",
                "arm": self.arm, "initial_alpha": INITIAL_ORDERS[self.arm],
                "learnable_alpha": self.log_alpha.requires_grad}

    def set_extra_state(self, state):
        study.require(state == self.get_extra_state(), "history study recipe differs")


def adapter_for(arm, config, seed):
    study.require(arm in ARMS, "unrecognized lag arm")
    orders = config.get("initial_orders")
    study.require(isinstance(orders, dict) and orders == INITIAL_ORDERS
                  and all(type(v) in (int, float) for v in orders.values()),
                  "lag initial orders differ")
    kernel = config["kernel"]
    study.require(type(kernel["step"]) in (int, float) and kernel["step"] == 1.0,
                  "lag comparison requires step=1")
    study.require(type(kernel["kernel_len"]) is int and kernel["kernel_len"] > 1,
                  "history requires more than one tap")
    with torch.device("cpu"):
        if arm == "lag1":
            return CausalLagGate(config["features"], strength=config["strength"], **kernel)
        return StudyHistory(config["features"], arm=arm,
                            strength=config["strength"], **kernel)


def main():
    study.main(
        arms=ARMS, adapter_factory=adapter_for,
        adapter_sources={"lag_study": Path(__file__),
                         "fractional_bridge": Path(fractional_bridge.__file__)},
        result_schema="spiraltorch.fractional_lag_study.v1",
    )


if __name__ == "__main__":
    main()
