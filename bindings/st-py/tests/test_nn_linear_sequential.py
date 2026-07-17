from __future__ import annotations

import importlib
import sys
import types

import pytest


def _ensure_torch_stub() -> None:
    if "torch" in sys.modules:
        return
    torch_stub = types.ModuleType("torch")
    torch_stub.autograd = types.SimpleNamespace(Function=object)
    sys.modules["torch"] = torch_stub


def _load_native() -> types.ModuleType | None:
    _ensure_torch_stub()
    try:
        module = importlib.import_module("spiraltorch")
    except ModuleNotFoundError:
        return None

    for native_name in (
        "spiraltorch.spiraltorch",
        "spiraltorch.spiraltorch_native",
        "spiraltorch_native",
    ):
        try:
            importlib.import_module(native_name)
        except ModuleNotFoundError:
            continue
        return module
    return None


def test_linear_forward_backward_smoke() -> None:
    st = _load_native()
    if st is None:
        pytest.skip("native SpiralTorch extension unavailable")
    assert hasattr(st, "nn")
    assert hasattr(st.nn, "Linear")
    assert hasattr(st.nn, "MeanSquaredError")

    Linear = st.nn.Linear
    MeanSquaredError = st.nn.MeanSquaredError

    layer = Linear("fc", 2, 1)
    loss = MeanSquaredError()

    x = st.Tensor((1, 2), data=[0.5, -0.1])
    target = st.Tensor((1, 1), data=[0.2])

    pred = layer.forward(x)
    loss_value = loss.forward(pred, target)
    grad_pred = loss.backward(pred, target)
    _ = layer.backward(x, grad_pred)
    layer.apply_step(0.05)
    layer.zero_accumulators()

    pred_after = layer.forward(x)
    assert pred.shape() == pred_after.shape()
    assert pred.tolist() != pred_after.tolist()
    assert loss_value.shape() == (1, 1)


def test_lora_linear_exposes_the_rust_adapter_contract() -> None:
    st = _load_native()
    if st is None:
        pytest.skip("native SpiralTorch extension unavailable")
    assert hasattr(st, "nn")
    assert hasattr(st.nn, "LoraLinear")

    with pytest.raises(ValueError, match="lora_alpha"):
        st.nn.LoraLinear(2, 1, 1, alpha=0.0, name="adapter")
    with pytest.raises(ValueError, match="lora_rank"):
        st.nn.LoraLinear(2, 1, 2, alpha=2.0, name="adapter")

    layer = st.nn.LoraLinear(2, 1, 1, alpha=2.0, name="adapter")
    assert layer.rank == 1
    assert layer.alpha == pytest.approx(2.0)
    assert layer.scale == pytest.approx(2.0)
    assert [name for name, _ in layer.state_dict()] == [
        "adapter::lora_a",
        "adapter::lora_b",
    ]
    assert [name for name, _ in layer.base_state_dict()] == [
        "adapter::bias",
        "adapter::weight",
    ]

    x = st.Tensor((1, 2), data=[0.5, -0.1])
    grad_output = st.Tensor((1, 1), data=[0.3])
    before = layer.forward(x)
    _ = layer.backward(x, grad_output)
    layer.apply_step(0.05)
    layer.zero_accumulators()
    after = layer.forward(x)

    assert before.shape() == after.shape()
    assert before.tolist() != after.tolist()


def test_sequential_add_and_train_step() -> None:
    st = _load_native()
    if st is None:
        pytest.skip("native SpiralTorch extension unavailable")
    assert hasattr(st, "nn")
    for name in ("Sequential", "Linear", "Relu", "MeanSquaredError"):
        assert hasattr(st.nn, name)

    model = st.nn.Sequential()
    linear = st.nn.Linear("l1", 2, 4)
    model.add(linear)
    model.add(st.nn.Relu())
    model.add(st.nn.Linear("l2", 4, 1))

    assert model.len() == 3
    assert len(model) == 3
    assert not model.is_empty()

    x = st.Tensor((1, 2), data=[0.5, -0.1])
    target = st.Tensor((1, 1), data=[0.2])
    loss = st.nn.MeanSquaredError()

    pred = model.forward(x)
    grad_pred = loss.backward(pred, target)
    _ = model.backward(x, grad_pred)
    model.apply_step(0.05)
    model.zero_accumulators()

    pred_after = model.forward(x)
    assert pred.tolist() != pred_after.tolist()

    with pytest.raises(ValueError):
        linear.forward(x)


def test_sequential_add_accepts_dropout_and_pooling() -> None:
    st = _load_native()
    if st is None:
        pytest.skip("native SpiralTorch extension unavailable")
    assert hasattr(st, "nn")
    for name in ("Sequential", "Dropout", "Pool2d", "ZPooling"):
        assert hasattr(st.nn, name)

    model = st.nn.Sequential()
    dropout = st.nn.Dropout(0.5, seed=11)
    model.add(dropout)
    pool = st.nn.Pool2d("max", 1, 2, 2, (2, 2))
    model.add(pool)
    zpool = st.nn.ZPooling(1, (2, 2), (2, 2))
    model.add(zpool)

    x = st.Tensor((1, 4), data=[1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        dropout.forward(x)
    with pytest.raises(ValueError):
        pool.forward(x)
    with pytest.raises(ValueError):
        zpool.forward(x)


def test_sequential_add_rejects_unknown_layer() -> None:
    st = _load_native()
    if st is None:
        pytest.skip("native SpiralTorch extension unavailable")
    assert hasattr(st, "nn")
    assert hasattr(st.nn, "Sequential")

    model = st.nn.Sequential()
    with pytest.raises(TypeError):
        model.add(object())
