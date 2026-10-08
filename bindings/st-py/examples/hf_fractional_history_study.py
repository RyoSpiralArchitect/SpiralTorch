"""Offline local/ordinary-causal/fractional-history comparison.

Only the ordinary EMA control is implemented with Torch operations here.
Fractional history and its differentials come from SpiralTorch's Rust core.
"""

import math
from pathlib import Path

import torch
import spiraltorch as st
import spiraltorch.fractional_autograd as fractional_bridge
from spiraltorch.geometry_autograd import _input, _strength

import hf_fractional_memory_study as memory
import hf_wave_gate_long_horizon as study

ARMS = ["pointwise", "ema_learned", "history_fixed", "history_learned"]


class CausalEmaGate(torch.nn.Module):
    """Finite, zero-padded, strictly-past EMA plus independent local gain.

    h[t] = (1-d) * sum(d**(k-1) * x[t-k], k=1..K-1), d=sigmoid(logit_decay).
    No prefix renormalization, persistent state or fractional coefficients.
    """

    def __init__(self, features, *, strength, initial_decay, kernel_len,
                 step=1.0, max_values=1_048_576, max_products=16_777_216):
        super().__init__()
        study.require(type(features) is int and features > 0, "invalid features")
        study.require(type(initial_decay) in (int, float) and math.isfinite(initial_decay)
                      and 0 < initial_decay < 1, "invalid initial decay")
        study.require(step == 1.0, "EMA control requires step=1")
        study.require(all(type(v) is int and v > 0 for v in
                          (kernel_len, max_values, max_products))
                      and 1 < kernel_len <= max_products, "invalid history budget")
        self.features, self.strength = features, _strength(strength)
        self.initial_decay = initial_decay
        self.kernel = {"kernel_len": kernel_len, "step": step,
                       "max_values": max_values, "max_products": max_products}
        self.gate = torch.nn.Parameter(torch.zeros(features, dtype=torch.float32))
        self.local_gate = torch.nn.Parameter(torch.zeros(features, dtype=torch.float32))
        self.logit_decay = torch.nn.Parameter(torch.tensor(
            math.log(initial_decay / (1 - initial_decay)), dtype=torch.float32))

    def history(self, value):
        _input(value)
        study.require(value.ndim == 3 and min(value.shape) > 0
                      and value.shape[-1] == self.features, "expected nonempty BTF input")
        study.require(all(p.dtype == value.dtype and p.device == value.device
                          for p in self.parameters()), "EMA dtype or device differs")
        study.require(bool(torch.isfinite(value).all())
                      and all(bool(torch.isfinite(p).all()) for p in self.parameters()),
                      "nonfinite EMA input or parameters")
        k = self.kernel["kernel_len"]
        study.require(value.numel() <= self.kernel["max_values"]
                      and value.numel() * k <= self.kernel["max_products"], "history budget exceeded")
        decay = self.logit_decay.sigmoid()
        study.require(bool((decay > 0) & (decay < 1)), "decay is not representable in (0,1)")
        weights = (1 - decay) * decay ** torch.arange(k - 1, device=value.device)
        # Torch correlation sees [oldest ... previous, zero current tap].
        weights = torch.cat((weights.flip(0), weights.new_zeros(1))).reshape(1, 1, k)
        b, t, f = value.shape
        lanes = value.transpose(1, 2).reshape(b * f, 1, t)
        result = torch.nn.functional.conv1d(torch.nn.functional.pad(lanes, (k - 1, 0)), weights)
        return result.reshape(b, f, t).transpose(1, 2)

    def forward(self, value):
        history = self.history(value)
        local = value + self.strength * self.local_gate.tanh() * value
        return local + self.strength * self.gate.tanh() * history

    def get_extra_state(self):
        return {"schema": "spiraltorch.ordinary_causal_ema_control.v1",
                "features": self.features, "strength": self.strength,
                "initial_decay": self.initial_decay, "kernel": dict(self.kernel)}

    def set_extra_state(self, state):
        study.require(state == self.get_extra_state(), "EMA study recipe differs")


class StudyHistory(st.FractionalHistoryAdapter):
    def __init__(self, features, *, learnable_alpha, **options):
        super().__init__(features, **options)
        study.require(type(learnable_alpha) is bool, "invalid alpha mode")
        self.learnable_alpha = learnable_alpha
        self.log_alpha.requires_grad_(learnable_alpha)

    def get_extra_state(self):
        return {**super().get_extra_state(),
                "study_schema": "spiraltorch.fractional_history_control.v1",
                "learnable_alpha": self.learnable_alpha}

    def set_extra_state(self, state):
        study.require(state == self.get_extra_state(), "history study recipe differs")


def adapter_for(arm, config, seed):
    study.require(arm in ARMS, "unrecognized history arm")
    study.require(config["kernel"]["step"] == 1.0, "history comparison requires step=1")
    study.require(type(config["kernel"]["kernel_len"]) is int
                  and config["kernel"]["kernel_len"] > 1, "history requires more than one tap")
    with torch.device("cpu"):
        if arm == "pointwise":
            return memory.PointwiseGate(config["features"], strength=config["strength"])
        if arm == "ema_learned":
            return CausalEmaGate(config["features"], strength=config["strength"],
                                 initial_decay=config["initial_decay"], **config["kernel"])
        return StudyHistory(config["features"], strength=config["strength"],
                            initial_alpha=config["initial_alpha"],
                            learnable_alpha=arm == "history_learned", **config["kernel"])


def main():
    study.main(
        arms=ARMS, adapter_factory=adapter_for,
        adapter_sources={"history_study": Path(__file__),
                         "memory_study": Path(memory.__file__),
                         "fractional_bridge": Path(fractional_bridge.__file__)},
        result_schema="spiraltorch.fractional_history_study.v1",
    )


if __name__ == "__main__":
    main()
