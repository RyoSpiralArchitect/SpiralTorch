from __future__ import annotations

import math

import spiraltorch as st


def test_dropout_eval_is_identity() -> None:
    layer = st.nn.Dropout(0.4, seed=7, training=False)
    tensor = st.Tensor(2, 3, [1.0, -2.0, 3.0, -4.0, 5.5, -6.5])

    output = layer(tensor)
    assert output.tolist() == tensor.tolist()

    grad_output = st.Tensor(2, 3, [0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    grad_input = layer.backward(tensor, grad_output)
    assert grad_input.tolist() == grad_output.tolist()

    assert not layer.training
    layer.train()
    assert layer.training
    layer.set_training(False)
    assert not layer.training


def test_dropout_probability_property() -> None:
    layer = st.nn.Dropout(0.25, seed=3)
    assert math.isclose(layer.probability, 0.25, rel_tol=1e-6, abs_tol=1e-6)
