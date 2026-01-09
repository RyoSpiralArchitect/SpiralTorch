import pytest

import spiraltorch as st


def test_realgrad_helper_accepts_tuple_shape() -> None:
    tape = st.realgrad((1, 4), learning_rate=0.02)
    assert tape.shape() == (1, 4)
    assert tape.learning_rate() == 0.02


def test_realgrad_notation_square_brackets() -> None:
    tape = st.rg[2, 3](learning_rate=0.03)
    assert tape.shape() == (2, 3)
    assert tape.learning_rate() == 0.03


def test_realgrad_apply_updates_weights() -> None:
    tape = st.Realgrad(0.1, 1, 3)
    weights = st.Tensor((1, 3), data=[1.0, 2.0, 3.0])
    grad = st.Tensor((1, 3), data=[0.5, -1.0, 2.0])
    tape.accumulate_wave(grad)
    tape.apply(weights)
    assert weights.tolist() == pytest.approx([[0.95, 2.1, 2.8]])
    assert tape.gradient() == pytest.approx([0.0, 0.0, 0.0])

