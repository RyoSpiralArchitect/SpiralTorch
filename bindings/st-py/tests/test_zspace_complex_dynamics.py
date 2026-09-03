from __future__ import annotations

import cmath
import copy
import json
import math
import runpy
from pathlib import Path

import pytest

import spiraltorch as st


def request():
    return {
        "forward_request": {
            "input": [0.2, -0.8, 0.5, 0.3, 0.7, -0.4],
            "potential": [0.3, -0.6, 0.2],
            "standard_normal": [0.3, -0.5, 0.7, -0.2, 0.1, 0.4],
            "rows": 2,
            "features": 3,
            "config": {"time_step": 0.3, "hopping_rate": -0.8, "loss_rate": 0.07, "noise_scale": 0.2},
        },
        "input_imaginary": [0.5, 0.1, -0.2, 0.6, -0.9, 0.2],
    }


def test_complex_forward_matches_independent_split_operator():
    req = request()
    forward = req["forward_request"]
    c = forward["config"]
    state = [complex(r, i) for r, i in zip(forward["input"], req["input_imaginary"])]
    phase = [c["time_step"] * forward["potential"][i % 3] + c["noise_scale"] * math.sqrt(c["time_step"]) * n for i, n in enumerate(forward["standard_normal"])]
    diagonal = [cmath.exp(-0.5j * p) for p in phase]
    left = [d * x for d, x in zip(diagonal, state)]
    middle = left.copy()
    angle = c["time_step"] * c["hopping_rate"]
    for i in [0, 3]:
        middle[i] = math.cos(angle) * left[i] - 1j * math.sin(angle) * left[i + 1]
        middle[i + 1] = math.cos(angle) * left[i + 1] - 1j * math.sin(angle) * left[i]
    expected = [math.exp(-0.5 * c["loss_rate"] * c["time_step"]) * d * x for d, x in zip(diagonal, middle)]
    receipt = st.zspace_stochastic_schrodinger_complex_step(req)
    assert receipt["step"]["output_real"] == pytest.approx([x.real for x in expected], abs=2e-7)
    assert receipt["step"]["output_imaginary"] == pytest.approx([x.imag for x in expected], abs=2e-7)
    assert receipt["step"]["final_norm_squared"] == pytest.approx(sum(abs(x) ** 2 for x in state) * math.exp(-c["loss_rate"] * c["time_step"]), abs=5e-7)
    assert receipt["gradient"] is None
    assert receipt["request"]["cotangent"] is None
    assert st.validate_zspace_stochastic_schrodinger_complex(json.loads(json.dumps(receipt))) == receipt


def trajectory(req):
    first = st.zspace_stochastic_schrodinger_complex_step(req)
    second_request = copy.deepcopy(req)
    second_request["forward_request"]["input"] = first["step"]["output_real"]
    second_request["input_imaginary"] = first["step"]["output_imaginary"]
    second = st.zspace_stochastic_schrodinger_complex_step(second_request)
    return first, second


def test_two_step_adjoint_matches_state_and_shared_potential_differences():
    req = request()
    first, second = trajectory(req)
    cotangent = {"real": [0.6, -0.2, 0.8, -0.3, 0.5, 0.1], "imaginary": [-0.4, 0.7, -0.6, 0.2, 0.8, -0.5]}
    backward_request = copy.deepcopy(second["request"])
    backward_request["cotangent"] = cotangent
    last_grad = st.zspace_stochastic_schrodinger_complex_step(backward_request)["gradient"]
    backward_request = copy.deepcopy(first["request"])
    backward_request["cotangent"] = {"real": last_grad["grad_input_real"], "imaginary": last_grad["grad_input_imaginary"]}
    first_grad = st.zspace_stochastic_schrodinger_complex_step(backward_request)["gradient"]
    expected_potential = [a + b for a, b in zip(first_grad["grad_potential"], last_grad["grad_potential"])]

    def loss(payload):
        output = trajectory(payload)[1]["step"]
        return sum(x * g for x, g in zip(output["output_real"], cotangent["real"])) + sum(x * g for x, g in zip(output["output_imaginary"], cotangent["imaginary"]))

    for key, expected in [("input", first_grad["grad_input_real"]), ("input_imaginary", first_grad["grad_input_imaginary"]), ("potential", expected_potential)]:
        for index, analytic in enumerate(expected):
            plus, minus = copy.deepcopy(req), copy.deepcopy(req)
            a = plus if key == "input_imaginary" else plus["forward_request"]
            b = minus if key == "input_imaginary" else minus["forward_request"]
            a[key][index] += 1e-3
            b[key][index] -= 1e-3
            numerical = (loss(plus) - loss(minus)) / 2e-3
            assert numerical == pytest.approx(analytic, abs=1.2e-4), (key, index)


@pytest.mark.parametrize("field", ["input_imaginary", "gradient", "phase"])
def test_complex_replay_rejects_tampering(field):
    req = request()
    req["cotangent"] = {"real": [1.0] * 6, "imaginary": [0.2] * 6}
    receipt = st.zspace_stochastic_schrodinger_complex_step(req)
    if field == "input_imaginary":
        receipt["request"][field][0] = 9.0
    elif field == "gradient":
        receipt["gradient"]["grad_input_imaginary"][0] = 9.0
    else:
        receipt["step"]["phase"][0] = 9.0
    with pytest.raises(ValueError, match="canonical Rust complex"):
        st.validate_zspace_stochastic_schrodinger_complex(receipt)


def test_complex_ingress_has_no_python_fallback_or_active_hooks(monkeypatch):
    class Active:
        @property
        def __class__(self):
            raise AssertionError("must not execute __class__")

        def __iter__(self):
            raise AssertionError("must not iterate an active container")

    req = request()
    req["input_imaginary"] = Active()
    with pytest.raises((TypeError, ValueError)):
        st.zspace_stochastic_schrodinger_complex_step(req)
    req = request()
    req["cotangent"] = {"real": [0.0] * 6}
    with pytest.raises(ValueError, match="imaginary"):
        st.zspace_stochastic_schrodinger_complex_step(req)
    monkeypatch.setattr(st, "_rs", None)
    with pytest.raises(RuntimeError, match="compiled Rust"):
        st.zspace_stochastic_schrodinger_complex_step(request())


def test_shared_potential_learns_through_three_complex_steps():
    example = Path(__file__).resolve().parents[3] / "examples" / "zspace_complex_trajectory.py"
    report = runpy.run_path(str(example))["fit"]()
    assert report["final_loss"] < report["initial_loss"] * 0.01
    assert report["trajectory_steps"] == 3
    assert report["language_quality_claim"] is False
