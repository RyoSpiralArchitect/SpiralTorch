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


def test_evaluate_at_k_returns_expected_fields() -> None:
    st = _load_native()
    if st is None:
        pytest.skip("native SpiralTorch extension unavailable")

    report = st.evaluate_at_k(recommended=[2, 1, 3], relevant=[2, 3], k=2)
    assert report["hits"] == 1
    assert report["relevant"] == 2
    assert report["precision"] == pytest.approx(0.5)
    assert report["recall"] == pytest.approx(0.5)
    assert report["hit_rate"] == pytest.approx(1.0)
    assert "ndcg" in report
    assert "average_precision" in report

