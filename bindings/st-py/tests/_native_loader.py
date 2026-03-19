from __future__ import annotations

import importlib
import sys
import types

import pytest


def ensure_torch_stub() -> None:
    if "torch" in sys.modules:
        return
    torch_stub = types.ModuleType("torch")
    torch_stub.autograd = types.SimpleNamespace(Function=object)
    sys.modules["torch"] = torch_stub


def load_native() -> types.ModuleType | None:
    ensure_torch_stub()
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


def require_native() -> types.ModuleType:
    module = load_native()
    if module is None:
        pytest.skip("native SpiralTorch extension unavailable")
    return module


def build_features(module: types.ModuleType) -> dict[str, bool]:
    build_info = getattr(module, "build_info", None)
    if not callable(build_info):
        return {}
    try:
        payload = build_info()
    except Exception:
        return {}
    features = payload.get("features", {})
    if not isinstance(features, dict):
        return {}
    return {str(key): bool(value) for key, value in features.items()}


def require_wgpu_runtime(module: types.ModuleType) -> None:
    if not build_features(module).get("wgpu", False):
        pytest.skip("native SpiralTorch extension was built without the 'wgpu' feature")
