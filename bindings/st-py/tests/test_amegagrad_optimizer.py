from __future__ import annotations

import pytest

import spiraltorch as st


def _require_native() -> None:
    try:
        st.hypergrad((1, 1))
        st.realgrad((1, 1))
    except Exception as exc:  # pragma: no cover - depends on native build
        pytest.skip(f"native SpiralTorch extension required: {exc}", allow_module_level=True)


def test_amegagrad_available_in_optim_module() -> None:
    _require_native()
    assert hasattr(st, "optim")
    assert hasattr(st.optim, "Amegagrad")
    assert hasattr(st.optim, "amegagrad")


def test_amegagrad_step_updates_weights() -> None:
    _require_native()

    opt = st.optim.Amegagrad((1, 3), curvature=-0.9, hyper_learning_rate=0.03, real_learning_rate=0.02)
    assert opt.shape() == (1, 3)

    weights = st.Tensor((1, 3), data=[0.2, -0.1, 0.05])
    signal = st.Tensor((1, 3), data=[0.4, -0.6, 0.2])
    opt.accumulate_wave(signal)

    hyper_grad = opt.hyper.gradient()
    real_grad = opt.real.gradient()
    assert len(hyper_grad) == 3
    assert len(real_grad) == 3
    assert any(abs(value) > 0.0 for value in hyper_grad)
    assert any(abs(value) > 0.0 for value in real_grad)

    before = weights.tolist()
    updated = opt.step(weights, tune=False)
    after = weights.tolist()
    assert updated is weights
    assert after != before

