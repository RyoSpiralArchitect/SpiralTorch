"""Offline two-lag parity and learnable departure from integer order two.

The ordinary two-tap control is an independent Torch reference, not a new
production fractional backend. Rust retains all GL maps and differentials.
"""

from pathlib import Path

import torch
import spiraltorch as st
import spiraltorch.fractional_autograd as fractional_bridge

import hf_fractional_lag_study as one_lag

study = one_lag.study
ARMS = ["lag2", "history_fixed_two", "history_learned_two", "history_learned_one"]
INITIAL_ORDERS = {"history_fixed_two": 2.0, "history_learned_two": 2.0,
                  "history_learned_one": 1.0}


class CausalTwoLagGate(one_lag.CausalLagGate):
    """Independent local/history gains with h[t] = -2*x[t-1] + x[t-2].

    Missing taps are zero, never wrapped. Double accumulation matches the Rust
    map before its f32 output boundary, including cancellation near f32 limits.
    This reference is a quality/parity control, not a throughput competitor.
    """

    def __init__(self, features, **options):
        super().__init__(features, **options)
        study.require(self.kernel["kernel_len"] > 2, "two-lag history needs at least three taps")

    def history(self, value):
        # Reuse the existing fail-closed BTF, finite, device and budget checks.
        super().history(value)
        work = value.to(dtype=torch.float64)
        first = torch.cat((torch.zeros_like(work[:, :1]), work[:, :-1]), dim=1)
        second = torch.cat((torch.zeros_like(work[:, :min(2, work.shape[1])]),
                            work[:, :-2]), dim=1)
        result = (-2 * first + second).to(dtype=value.dtype)
        study.require(bool(torch.isfinite(result).all()), "nonfinite two-lag history")
        return result

    def get_extra_state(self):
        return {**super().get_extra_state(),
                "schema": "spiraltorch.ordinary_two_lag_control.v1",
                "history_coefficients": [-2.0, 1.0], "accumulation_dtype": "float64"}


class StudyHistory(st.FractionalHistoryAdapter):
    def __init__(self, features, *, arm, **options):
        study.require(arm in INITIAL_ORDERS, "invalid two-lag history arm")
        super().__init__(features, initial_alpha=INITIAL_ORDERS[arm], **options)
        self.arm = arm
        self.log_alpha.requires_grad_(arm != "history_fixed_two")

    def get_extra_state(self):
        return {**super().get_extra_state(),
                "study_schema": "spiraltorch.fractional_two_lag_control.v1",
                "arm": self.arm, "initial_alpha": INITIAL_ORDERS[self.arm],
                "learnable_alpha": self.log_alpha.requires_grad}

    def set_extra_state(self, state):
        study.require(state == self.get_extra_state(), "two-lag study recipe differs")


def adapter_for(arm, config, seed):
    study.require(arm in ARMS, "unrecognized two-lag arm")
    study.require(config.get("schema") == "spiraltorch.fractional_two_lag_protocol.v1",
                  "two-lag protocol differs")
    orders = config.get("initial_orders")
    study.require(isinstance(orders, dict) and orders == INITIAL_ORDERS
                  and all(type(v) in (int, float) for v in orders.values()),
                  "two-lag initial orders differ")
    kernel = config["kernel"]
    study.require(type(kernel["step"]) in (int, float) and kernel["step"] == 1.0,
                  "two-lag comparison requires step=1")
    study.require(type(kernel["kernel_len"]) is int and kernel["kernel_len"] > 2,
                  "two-lag history needs at least three taps")
    with torch.device("cpu"):
        if arm == "lag2":
            return CausalTwoLagGate(config["features"], strength=config["strength"], **kernel)
        return StudyHistory(config["features"], arm=arm,
                            strength=config["strength"], **kernel)


def main():
    study.main(
        arms=ARMS, adapter_factory=adapter_for,
        adapter_sources={"two_lag_study": Path(__file__),
                         "one_lag_control": Path(one_lag.__file__),
                         "fractional_bridge": Path(fractional_bridge.__file__)},
        result_schema="spiraltorch.fractional_two_lag_study.v1",
    )


if __name__ == "__main__":
    main()
