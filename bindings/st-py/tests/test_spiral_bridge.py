from __future__ import annotations

import importlib
import sys
import types


def _reset_spiral_modules() -> None:
    for key in list(sys.modules):
        if key in {"spiral", "spiraltorch"} or key.startswith("spiral.") or key.startswith("_spiral_py_bridge"):
            sys.modules.pop(key, None)


def test_spiral_bridge_merges_pure_python_helpers(monkeypatch) -> None:
    _reset_spiral_modules()

    inference_stub = types.SimpleNamespace(
        SafetyViolation=object,
        SafetyVerdict=object,
        InferenceResult=object,
        AuditEvent=object,
        AuditLog=types.SimpleNamespace(entries=lambda: []),
        InferenceRuntime=types.SimpleNamespace,
    )

    def _dummy_hypergrad(*args, **kwargs):  # noqa: ANN001, ANN002 - test helper signature mirrors runtime
        return types.SimpleNamespace(
            reset=lambda: None,
            summary=lambda: types.SimpleNamespace(
                l1=lambda: 0.0,
                l2=lambda: 0.0,
                linf=lambda: 0.0,
                mean_abs=lambda: 0.0,
                rms=lambda: 0.0,
                count=lambda: 0,
                sum_squares=lambda: 0.0,
            ),
            shape=lambda: (1, 1),
            curvature=lambda: 0.0,
            learning_rate=lambda: 0.1,
            gradient=lambda: [0.0],
        )

    spiraltorch_stub = types.ModuleType("spiraltorch")
    spiraltorch_stub.inference = inference_stub
    spiraltorch_stub.hypergrad = _dummy_hypergrad
    spiraltorch_stub.export = types.SimpleNamespace(
        PyQatObserver=lambda **_: types.SimpleNamespace(),
        compress_weights=lambda weights, observer, pruning_cfg, latency_hint: (
            list(weights),
            types.SimpleNamespace(as_dict=lambda: {}),
        ),
    )
    monkeypatch.setitem(sys.modules, "spiraltorch", spiraltorch_stub)

    module = importlib.import_module("spiral")

    assert callable(module.format_chat_prompt)
    assert callable(module.hypergrad_session)
    assert hasattr(module, "augment")
    assert hasattr(module, "gaussian_noise")
    assert hasattr(module, "data")
    assert hasattr(module, "hypergrad")
    assert "augment" in module.__all__

