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
    except Exception:
        return None

    for native_name in (
        "spiraltorch.spiraltorch",
        "spiraltorch.spiraltorch_native",
        "spiraltorch_native",
    ):
        try:
            importlib.import_module(native_name)
        except Exception:
            continue
        return module
    return None


def test_module_trainer_prepare_step_zero_and_realgrad_controls() -> None:
    st = _load_native()
    if st is None:
        pytest.skip("native SpiralTorch extension unavailable")
    assert hasattr(st, "nn")
    assert hasattr(st.nn, "ModuleTrainer")
    assert hasattr(st.nn, "Sequential")
    assert hasattr(st.nn, "Linear")
    assert hasattr(st.nn, "MeanSquaredError")

    trainer = st.nn.ModuleTrainer(
        backend="cpu",
        curvature=-1.0,
        hyper_learning_rate=1e-2,
        fallback_learning_rate=1e-2,
    )
    assert trainer.real_learning_rate is None
    trainer.enable_realgrad(5e-3)
    assert trainer.real_learning_rate == pytest.approx(5e-3)
    trainer.disable_realgrad()
    assert trainer.real_learning_rate is None

    model = st.nn.Sequential()
    model.add(st.nn.Linear("l1", 2, 1))
    loss = st.nn.MeanSquaredError()
    x = st.Tensor.rand(2, 2, seed=21)
    y = st.Tensor.rand(2, 1, seed=22)

    pred_before = model.forward(x)
    trainer.prepare(model)
    grad_pred = loss.backward(pred_before, y)
    _ = model.backward(x, grad_pred)
    trainer.step(model)
    trainer.zero(model)
    pred_after = model.forward(x)

    assert pred_before.shape() == pred_after.shape()
    assert pred_before.tolist() != pred_after.tolist()


def test_module_trainer_curvature_scheduler_metrics_roundtrip() -> None:
    st = _load_native()
    if st is None:
        pytest.skip("native SpiralTorch extension unavailable")
    assert hasattr(st, "nn")
    assert hasattr(st.nn, "CurvatureScheduler")
    assert hasattr(st.nn, "RoundtableConfig")

    trainer = st.nn.ModuleTrainer(
        backend="cpu",
        curvature=-1.0,
        hyper_learning_rate=1e-1,
        fallback_learning_rate=1e-2,
    )
    scheduler = st.nn.CurvatureScheduler(
        initial=-1.0,
        min_curvature=-2.0,
        max_curvature=-0.2,
        target_pressure=0.0,
        step=0.2,
        tolerance=0.0,
        smoothing=1.0,
    )
    trainer.enable_curvature_scheduler(scheduler)

    model = st.nn.Sequential()
    model.add(st.nn.Linear("l1", 2, 1))
    trainer.prepare(model)
    loss = st.nn.MeanSquaredError()
    schedule = trainer.roundtable(
        2,
        1,
        st.nn.RoundtableConfig(top_k=1, mid_k=1, bottom_k=1, here_tolerance=1e-5),
    )
    x1 = st.Tensor.rand(2, 2, seed=31)
    y1 = st.Tensor.rand(2, 1, seed=32)
    x2 = st.Tensor.rand(2, 2, seed=33)
    y2 = st.Tensor.rand(2, 1, seed=34)
    stats = trainer.train_epoch(model, loss, [(x1, y1), (x2, y2)], schedule)
    assert stats.batches == 2

    metrics = trainer.curvature_metrics()
    assert isinstance(metrics, dict)
    assert "raw_pressure" in metrics
    assert "smoothed_pressure" in metrics
    assert "curvature" in metrics
    assert trainer.curvature == pytest.approx(float(metrics["curvature"]))

    trainer.disable_curvature_scheduler()
    assert trainer.curvature_metrics() is None
