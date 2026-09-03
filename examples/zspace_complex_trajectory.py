"""Fit a shared potential through a three-step Rust-owned complex trajectory.

Run against an installed wheel. This is a numerical learning demonstration,
not evidence of language-model quality or physical-system identification.
"""

from __future__ import annotations

import json

import spiraltorch as st


def rollout(potential):
    real, imaginary = [0.8, -0.3, 0.4], [0.2, 0.5, -0.6]
    receipts = []
    for _ in range(3):
        receipt = st.zspace_stochastic_schrodinger_complex_step({
            "forward_request": {
                "input": real, "potential": potential, "rows": 1, "features": 3,
                "standard_normal": [0.0] * 3,
                "config": {"time_step": 0.3, "hopping_rate": 0.6},
            },
            "input_imaginary": imaginary,
        })
        receipts.append(receipt)
        real, imaginary = receipt["step"]["output_real"], receipt["step"]["output_imaginary"]
    return receipts


def fit(iterations=80):
    target = rollout([0.4, -0.5, 0.3])[-1]["step"]
    potential = [0.0] * 3
    losses = []
    for iteration in range(iterations + 1):
        receipts = rollout(potential)
        last = receipts[-1]["step"]
        real_error = [a - b for a, b in zip(last["output_real"], target["output_real"])]
        imaginary_error = [a - b for a, b in zip(last["output_imaginary"], target["output_imaginary"])]
        losses.append(0.5 * sum(e * e for e in real_error + imaginary_error))
        if iteration == iterations:
            break
        cotangent = {"real": real_error, "imaginary": imaginary_error}
        gradient = [0.0] * 3
        for receipt in reversed(receipts):
            backward = st.zspace_stochastic_schrodinger_complex_step({**receipt["request"], "cotangent": cotangent})
            grad = backward["gradient"]
            gradient = [a + b for a, b in zip(gradient, grad["grad_potential"])]
            cotangent = {"real": grad["grad_input_real"], "imaginary": grad["grad_input_imaginary"]}
        potential = [p - 0.3 * g for p, g in zip(potential, gradient)]
    for receipt in receipts:
        st.validate_zspace_stochastic_schrodinger_complex(receipt)
    return {"iterations": iterations, "trajectory_steps": 3, "initial_loss": losses[0], "final_loss": losses[-1], "learned_potential": potential, "final_state": {"real": last["output_real"], "imaginary": last["output_imaginary"]}, "semantic_backend": "rust", "language_quality_claim": False}


if __name__ == "__main__":
    print(json.dumps(fit(), indent=2))
